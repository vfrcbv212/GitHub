"""Controlled temporal-model tests. Fixtures are NOT real volleyball evaluation."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from volleyvision.temporal import (FEATURE_NAMES, TemporalPredictor, assert_disjoint,
    evaluate_policy, features_from_observations, infer_project, load_dataset,
    make_model, observation_windows, publish_artifact, suppress_candidates, train)


def row(visible=True, direction=1):
    values = np.zeros(len(FEATURE_NAMES), np.float32)
    values[0] = .1
    values[1] = int(visible)
    values[2:6] = [.5, .5, .03, .03] if visible else [0., 0., 0., 0.]
    values[6] = direction
    values[8] = int(visible)
    values[9] = 1
    values[16] = .2 if visible else 0.
    return values.tolist()


def dataset(path, match="train", kind="synthetic", count=8):
    windows = []
    for i in range(count):
        label = "attack" if i % 2 else "none"
        times = [round(i+j*.1, 6) for j in range(4)]
        windows.append({"id": f"{match}-{i}", "match_id": match, "source_id": match+"-source",
            "times": times, "anchor": times[2], "features": [row(direction=1 if label == "attack" else -1) for _ in times],
            "label": label, "reviewed": True})
    value = {"schema_version": 1, "data_kind": kind, "feature_names": list(FEATURE_NAMES),
             "window_seconds": .3, "stride_seconds": .1, "windows": windows}
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def artifact(root):
    import torch
    model = make_model(2, hidden_size=4)
    with torch.no_grad():
        for param in model.parameters(): param.zero_()
        model.head.bias[:] = torch.tensor([-10., 10.])
    manifest = publish_artifact(root, model.state_dict(), {"hidden_size": 4, "num_layers": 1}, ["none", "attack"],
        {"mean": [0.]*len(FEATURE_NAMES), "std": [1.]*len(FEATURE_NAMES)},
        {"train_match_ids": ["train"], "train_source_ids": ["train-source"],
         "validation_match_ids": ["validation"], "validation_source_ids": ["validation-source"],
         "window_seconds": .3, "stride_seconds": .1, "data_kind": "synthetic"})
    return manifest


def test_module_import_keeps_torch_optional():
    result = subprocess.run([sys.executable, "-c", "import sys; import volleyvision.temporal; print('torch' in sys.modules)"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_features_do_not_bridge_lost_or_ambiguous_ball():
    ball = lambda x: {"bbox": [x, 10, x+2, 12], "observed": True}
    obs = [{"time": 0., "balls": [ball(10)], "source_pts": 0},
           {"time": .1, "balls": [ball(12)], "source_pts": 10},
           {"time": .2, "balls": [], "source_pts": 20},
           {"time": .3, "balls": [ball(16)], "source_pts": 30},
           {"time": .4, "balls": [ball(18), ball(20)], "source_pts": 40},
           {"time": 1., "balls": [ball(18)], "source_pts": 100}]
    result = list(features_from_observations(obs, 100, 100))
    assert result[1][1][8] == 1
    assert result[2][1][1] == 0 and result[2][1][6] == 0
    assert result[3][1][8] == 0
    assert result[4][1][1] == 0
    assert result[5][1][8] == 0
    assert result[3][2] == 30
    with pytest.raises(ValueError, match="возрастающие"):
        list(features_from_observations([obs[1], obs[0]], 100, 100))


def test_dataset_requires_review_and_disjoint_matches(tmp_path):
    a, b = tmp_path/"a.json", tmp_path/"b.json"
    value = dataset(a)
    dataset(b, match="validation")
    assert_disjoint(load_dataset(a), load_dataset(b))
    with pytest.raises(ValueError, match="Утечка"):
        assert_disjoint(load_dataset(a), load_dataset(a))
    value["windows"][0]["reviewed"] = False
    a.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="проверенная"):
        load_dataset(a)
    value["windows"][0]["reviewed"] = True
    value["windows"][0]["features"][0][0] = float("nan")
    a.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="признаки"):
        load_dataset(a)


def test_raw_high_score_is_only_hypothesis_with_independent_unknowns(tmp_path):
    path = artifact(tmp_path/"versions")
    predictor = TemporalPredictor(path)
    window = {"times": [0., .1, .2, .3], "features": [row() for _ in range(4)], "source_pts": 200}
    e, prediction = predictor.predict_event("source", window)
    assert prediction["raw_score"] > .99
    assert prediction["decision"] == "hypothesis"
    assert e["review"] == "unreviewed" and e["evidence"]["type"] == "hypothesis"
    assert e.get("player_id") is None and e.get("outcome") is None and e.get("team_id") is None
    assert e["source_pts"] == 200 and e["start"] == .2
    window["features"] = [row(visible=False) for _ in range(4)]
    assert predictor.predict_event("source", window)[0] is None
    (path.parent/"model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="контрольная сумма"):
        TemporalPredictor(path)


def test_policy_requires_real_event_reference_and_exact_rules():
    candidates = [{"source_id": "heldout-source", "type": "attack", "start": i*2., "raw_score": .99} for i in range(100)]
    data = {"data_kind": "synthetic", "coverage_verified": True, "fingerprint": "f"*64,
            "windows": [{"source_id": "heldout-source", "match_id": "heldout", "anchor": i*.25} for i in range(793)],
            "annotation_intervals": [{"source_id": "heldout-source", "start": 0., "end": 198., "reviewed": True, "types": ["attack"]}],
            "events": [dict(x, reviewed=True) for x in candidates]}
    synthetic = evaluate_policy(candidates, data, "a"*64, enable_automatic=True)
    assert not synthetic["enabled"] and synthetic["accepted_labels"] == []
    data["data_kind"] = "real"  # Unit fixture exercising the explicit-real gate, not actual footage.
    real_policy = evaluate_policy(candidates, data, "a"*64, enable_automatic=True)
    assert real_policy["enabled"] and real_policy["classes"]["attack"]["event_recall"] == 1
    assert real_policy["classes"]["attack"]["error_upper95"] < .1
    from volleyvision.temporal import _policy_allows
    assert _policy_allows(real_policy, "attack", "a"*64)
    changed = copy.deepcopy(real_policy); changed["rules"]["score_threshold"] = .1
    assert not _policy_allows(changed, "attack", "a"*64)
    assert not _policy_allows(real_policy, "attack", "b"*64)
    data["events"] = []
    assert not evaluate_policy(candidates, data, "a"*64, enable_automatic=True)["enabled"]


def test_temporal_nms_keeps_sources_separate():
    candidates = [{"source_id": "a", "type": "attack", "start": 1., "raw_score": .9},
                  {"source_id": "a", "type": "set", "start": 1.1, "raw_score": .8},
                  {"source_id": "b", "type": "set", "start": 1.1, "raw_score": .8}]
    result = suppress_candidates(candidates, .5)
    assert len(result) == 2 and {r["source_id"] for r in result} == {"a", "b"}


def test_real_training_checkpoint_resume_and_immutable_artifact(tmp_path):
    train_path, val_path = tmp_path/"train.json", tmp_path/"validation.json"
    dataset(train_path, match="train")
    dataset(val_path, match="validation")
    output = tmp_path/"experiment"
    first = train(train_path, val_path, output, epochs=1, hidden_size=4, batch_size=4)
    original_bytes = Path(first["manifest"]).read_bytes()
    second = train(train_path, val_path, output, epochs=2, hidden_size=4, batch_size=4, resume=True)
    assert second["last_epoch"] == 1 and Path(second["manifest"]).exists()
    assert first["automatic_acceptance"] is False
    assert Path(first["manifest"]).read_bytes() == original_bytes
    loaded = TemporalPredictor(second["manifest"])
    assert not loaded.policy["enabled"]
    value = dataset(train_path, match="train"); value["windows"][0]["features"][0][6] += .1
    train_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="изменились"):
        train(train_path, val_path, output, epochs=3, hidden_size=4, batch_size=4, resume=True)


def test_synthetic_calibration_cannot_activate_actions(tmp_path):
    from volleyvision.temporal import calibrate_artifact
    manifest = artifact(tmp_path/"base")
    path = tmp_path/"heldout.json"
    value = dataset(path, match="heldout", count=4)
    value["coverage_verified"] = True
    value["events"] = [{"source_id": "heldout-source", "type": "attack", "start": .2, "reviewed": True}]
    path.write_text(json.dumps(value))
    result = calibrate_artifact(manifest, path, tmp_path/"calibrated", enable_automatic=True)
    assert result["policy"]["enabled"] is False
    loaded = TemporalPredictor(result["manifest"])
    assert not loaded.policy["enabled"]


def test_project_inference_retains_source_pts_and_manual_edit(tmp_path):
    from volleyvision.store import Project, dumps
    manifest = artifact(tmp_path/"base")
    with Project(tmp_path/"project", create=True) as project:
        project.put("source", {"id": "source", "path": "missing.mp4", "duration": 1., "offset": 0., "display_width": 100, "display_height": 100})
        project.put("job", {"id": "run", "source_id": "source", "state": "complete"})
        for i in range(11):
            observation = {"time": i/10, "source_pts": i*100, "balls": [{"bbox": [10+i, 10, 12+i, 12], "observed": True}], "people": [], "geometry_valid": False}
            project.db.execute("INSERT INTO observations VALUES (?,?,?,?)", ("run", "source", i/10, dumps(observation)))
        project.db.commit()
        result = infer_project(project, "run", manifest)
        assert result["proposals"] > 0 and result["accepted"] == 0
        e = project.events()[0]
        assert e["source_pts"] == round(e["start"]*1000)
        e["review"] = "confirmed"; e["type"] = "serve"; e["evidence"]["type"] = "confirmed"
        project.save_event(e)
        count = len(project.events())
        infer_project(project, "run", manifest)
        assert len(project.events()) == count
        assert next(e for e in project.events() if e["review"] == "confirmed")["type"] == "serve"


def test_temporal_cancel_resume_global_nms_and_finished_cache(tmp_path, monkeypatch):
    from volleyvision.store import Project, dumps
    from volleyvision.media import Cancelled
    manifest = artifact(tmp_path/"base")

    def populate(project):
        project.put("source", {"id": "source", "path": "missing.mp4", "duration": 3., "offset": 0., "display_width": 100, "display_height": 100})
        project.put("job", {"id": "run", "source_id": "source", "state": "complete", "samples": 25, "last_time": 2.4})
        for i in range(25):
            observation = {"time": i/10, "source_pts": i*100, "balls": [{"bbox": [10+i, 10, 12+i, 12], "observed": True}], "people": []}
            project.db.execute("INSERT INTO observations VALUES (?,?,?,?)", ("run", "source", i/10, dumps(observation)))
        project.db.commit()

    with Project(tmp_path/"baseline", create=True) as baseline:
        populate(baseline)
        expected = infer_project(baseline, "run", manifest)
        expected_times = [e["start"] for e in baseline.events()]
    with Project(tmp_path/"resumed", create=True) as project:
        populate(project)
        stopped = [False]
        def progress(value, text):
            if value > .3: stopped[0] = True
        with pytest.raises(Cancelled):
            infer_project(project, "run", manifest, cancel=lambda: stopped[0], progress=progress)
        checkpoint = project.all("temporal_job")[0]
        assert checkpoint["state"] == "cancelled" and checkpoint["windows"] > 1
        previous_anchor = checkpoint["last_anchor"]
        original = TemporalPredictor.predict_event
        anchors = []
        def traced(self, source_id, window):
            anchors.append(window["anchor"])
            return original(self, source_id, window)
        monkeypatch.setattr(TemporalPredictor, "predict_event", traced)
        resumed = infer_project(project, "run", manifest)
        assert anchors and min(anchors) > previous_anchor
        assert resumed["windows"] == expected["windows"]
        assert [e["start"] for e in project.events()] == expected_times
        def unexpected(*args, **kwargs): raise AssertionError("completed temporal pass was recomputed")
        monkeypatch.setattr(TemporalPredictor, "predict_event", unexpected)
        assert infer_project(project, "run", manifest)["cache_reused"] is True
        monkeypatch.setattr(TemporalPredictor, "predict_event", original)
        observation = {"time": 2.5, "source_pts": 2500, "balls": [{"bbox": [35, 10, 37, 12], "observed": True}], "people": []}
        project.db.execute("INSERT INTO observations VALUES (?,?,?,?)", ("run", "source", 2.5, dumps(observation)))
        project.db.commit()
        upstream = project.get("job", "run"); upstream.update(samples=26, last_time=2.5); project.put("job", upstream)
        changed = infer_project(project, "run", manifest)
        assert changed["id"] != resumed["id"] and changed["windows"] > resumed["windows"]


def test_incomplete_reference_window_coverage_cannot_enable_policy():
    candidates = [{"source_id": "s", "type": "attack", "start": i*2., "raw_score": .99} for i in range(100)]
    data = {"data_kind": "real", "coverage_verified": True, "fingerprint": "f"*64,
            "windows": [{"source_id": "s", "match_id": "heldout", "anchor": 0.}],
            "annotation_intervals": [{"source_id": "s", "start": 0., "end": 198., "reviewed": True, "types": ["attack"]}],
            "events": [dict(x, reviewed=True) for x in candidates]}
    policy = evaluate_policy(candidates, data, "a"*64, enable_automatic=True)
    assert policy["enabled"] is False and policy["coverage_intervals_valid"] is False


def test_gallery_role_edit_invalidates_only_temporal_cache_and_filters_context(tmp_path, monkeypatch):
    from volleyvision.store import Project, dumps
    from volleyvision.temporal import TrackBindingIndex
    manifest = artifact(tmp_path/"base")
    with Project(tmp_path/"project", create=True) as project:
        project.put("source", {"id": "source", "path": "missing.mp4", "duration": 1., "offset": 0., "display_width": 100, "display_height": 100})
        upstream = {"id": "run", "source_id": "source", "state": "complete", "samples": 11, "last_time": 1.}
        project.put("job", upstream)
        for i in range(11):
            obs = {"time": i/10, "source_pts": i*100,
                   "balls": [{"bbox": [10+i, 10, 12+i, 12], "observed": True}],
                   "people": [{"track_id": "person-1", "bbox": [5, 5, 25, 70], "role": "participant", "observed": True}],
                   "people_sampled": True}
            project.db.execute("INSERT INTO observations VALUES (?,?,?,?)", ("run", "source", i/10, dumps(obs)))
        project.db.commit()
        original = TemporalPredictor.predict_event
        masks = []
        def capture(self, source_id, window):
            masks.extend(row[10] for row in window["features"])
            return original(self, source_id, window)
        monkeypatch.setattr(TemporalPredictor, "predict_event", capture)
        first = infer_project(project, "run", manifest)
        assert masks and all(x == 1 for x in masks)
        assert infer_project(project, "run", manifest)["cache_reused"] is True
        binding = project.bind_track("run", "person-1", 0, 1, role="referee")
        masks.clear()
        second = infer_project(project, "run", manifest)
        assert first["id"] != second["id"]
        assert first["inputs"]["track_bindings_sha256"] != second["inputs"]["track_bindings_sha256"]
        assert masks and all(x == 0 for x in masks)
        assert project.get("job", "run") == upstream
        assert all(p["role"] == "participant" for obs in project.observations("run") for p in obs["people"])
        assert all(e.get("player_id") is None and e.get("outcome") is None for e in project.events())
        project.bind_track("other-run", "person-1", 0, 1, role="outside")
        assert infer_project(project, "run", manifest)["cache_reused"] is True
        # A later correction wins only on its inclusive interval, not the whole track.
        correction = project.bind_track("run", "person-1", .3, .5, role="participant")
        index = TrackBindingIndex(project.all("track_binding"), "run")
        assert index.at("person-1", .2)["id"] == binding["id"]
        assert index.at("person-1", .3)["id"] == correction["id"]
        assert index.at("person-1", .5)["id"] == correction["id"]
        assert index.at("person-1", float(np.nextafter(.5, np.inf)))["id"] == binding["id"]
        assert index.at("person-1", 1.)["id"] == binding["id"]
        assert index.at("person-1", float(np.nextafter(1., np.inf))) is None
        assert index.at("unseen-track", .4) is None


@pytest.mark.parametrize('key',['calibration_id','camera_session_id','tracking_segment'])
def test_observation_windows_and_velocity_stop_at_view_boundaries(key):
    obs=[{'time':i/10,'balls':[{'bbox':[10+i,10,12+i,12],'track_id':'ball'}],key:'a' if i<5 else 'b'} for i in range(11)]
    features=list(features_from_observations(obs,100,100))
    assert features[4][1][8]==1 and features[5][1][8]==0 and features[5][1][0]==0
    windows=list(observation_windows(obs,100,100,window_seconds=.3,stride_seconds=.1))
    assert windows and all(w['end']<.5 or w['start']>=.5 for w in windows)
    obs[7]['balls'][0]['track_id']='another-ball'
    assert list(features_from_observations(obs,100,100))[7][1][8]==0


def test_policy_uses_all_competitors_before_scoring_reviewed_types():
    candidates=[];truth=[]
    for i in range(100):
        candidates.extend([{'source_id':'s','type':label,'start':4*i+offset,'raw_score':score}
            for label,offset,score in [('serve',0,.999),('attack',.5,.99),('attack',1,.98)]])
        truth.append({'source_id':'s','type':'attack','start':4*i+.5,'reviewed':True})
    data={'data_kind':'real','coverage_verified':True,'fingerprint':'f'*64,
        'windows':[{'source_id':'s','match_id':'heldout','anchor':i*.25} for i in range(1593)],
        'annotation_intervals':[{'source_id':'s','start':0,'end':398,'reviewed':True,'types':['attack']}],
        'events':truth}
    policy=evaluate_policy(candidates,data,'a'*64,enable_automatic=True)
    assert not policy['enabled'] and policy['classes']['attack']['tp']==0
    assert policy['classes']['attack']['fp']==100 and policy['classes']['attack']['fn']==100
    # Low-margin proposals also participate in runtime NMS before acceptance.
    candidates[0]['proposal_eligible']=False
    assert not evaluate_policy(candidates,data,'a'*64,enable_automatic=True)['enabled']


def test_cached_model_reactivation_restores_bases_without_inference(tmp_path,monkeypatch):
    from volleyvision.store import Project,dumps
    a=artifact(tmp_path/'a');predictor=TemporalPredictor(a)
    b=publish_artifact(tmp_path/'b',predictor.model.state_dict(),predictor.manifest['model_config'],
        ['none','serve'],predictor.manifest['normalization'],predictor.manifest['provenance'])
    with Project(tmp_path/'project',create=True) as p:
        p.put('source',{'id':'s','path':'missing.mp4','duration':1,'offset':0,'display_width':100,'display_height':100})
        p.put('job',{'id':'run','source_id':'s','state':'complete'})
        for i in range(11):p.db.execute('INSERT INTO observations VALUES(?,?,?,?)',('run','s',i/10,dumps({'time':i/10,'balls':[{'bbox':[10,10,12,12]}]})))
        p.db.commit();first=infer_project(p,'run',a)
        assert {e['type'] for e in p.events() if e['review']!='rejected'}=={'attack'}
        infer_project(p,'run',b)
        assert {e['type'] for e in p.events() if e['review']!='rejected'}=={'serve'}
        def unexpected(*args,**kwargs):raise AssertionError('Cached GRU must not run')
        monkeypatch.setattr(TemporalPredictor,'predict_event',unexpected)
        restored=infer_project(p,'run',a)
        assert restored['cache_reused'] and restored['id']==first['id']
        assert {e['type'] for e in p.events() if e['review']!='rejected'}=={'attack'}
        e=p.events()[0];e['type']='dig';e['evidence']['type']='confirmed';e['review']='confirmed';p.save_event(e)
        infer_project(p,'run',b);infer_project(p,'run',a)
        assert p.events()[0]['type']=='dig' and p.events()[0]['review']=='confirmed'


def test_high_precision_with_low_event_recall_cannot_enable_policy():
    candidates=[{'source_id':'s','type':'attack','start':i*20.,'raw_score':.99} for i in range(100)]
    data={'data_kind':'real','coverage_verified':True,'fingerprint':'f'*64,
        'windows':[{'source_id':'s','match_id':'heldout','anchor':i*.25} for i in range(7993)],
        'annotation_intervals':[{'source_id':'s','start':0,'end':1998,'reviewed':True,'types':['attack']}],
        'events':[{'source_id':'s','type':'attack','start':i*2.,'reviewed':True} for i in range(1000)]}
    policy=evaluate_policy(candidates,data,'a'*64,enable_automatic=True)
    assert not policy['enabled'] and not policy['classes']['attack']['passed']
    assert policy['classes']['attack']['tp']==100 and policy['classes']['attack']['fp']==0
    assert policy['classes']['attack']['event_recall']==pytest.approx(.1)
