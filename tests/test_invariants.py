import json
from pathlib import Path

import numpy as np
import pytest

from volleyvision.analytics import report
from volleyvision.calibration import fit, normalize, project_point
from volleyvision.demo import create, make_video
from volleyvision.domain import event, usable
from volleyvision.exporting import export_all
from volleyvision.identity import consensus
from volleyvision.media import create_proxy, export_clip, frames, import_video, montage, probe
from volleyvision.store import Project


@pytest.fixture
def project(tmp_path):
    create(tmp_path / "demo")
    with Project(tmp_path / "demo") as p:
        yield p


def test_formulas_unknown_and_group_block(project):
    data = report(project)
    p = next(p for p in data["players"] if p["id"] == "A1")
    assert p["attacks_recorded"] == 5
    assert p["attack_efficiency"]["value"] == -25
    assert p["attack_efficiency"]["numerator"] == -1
    assert p["attack_efficiency"]["denominator"] == 4
    assert p["attack_efficiency"]["missing"] == 1
    assert p["directions"]["diagonal"]["numerator"] == 2
    assert p["directions"]["diagonal"]["denominator"] == 3
    assert p["directions"]["diagonal"]["missing"] == 2
    assert len(data["point_ledger"]) == 3
    b = next(t for t in data["teams"] if t["id"] == "B")
    assert b["points_from_known_rallies"] == 2
    assert b["block_point_participations"] == 1
    assert next(p for p in data["players"] if p["id"] == "B1")["credited_points"] == 1
    assert next(p for p in data["players"] if p["id"] == "B2")["credited_points"] == 0
    assert data["completeness"] is None


def test_no_observations_means_no_rate(project):
    player = next(p for p in report(project)["players"] if p["id"] == "B2")
    assert player["attack_efficiency"]["value"] is None
    assert player["attack_efficiency"]["denominator"] == 0


def test_edit_changes_stats_and_undo(project):
    e = next(e for e in project.events() if e.get("outcome") == "kill")
    e["player_id"] = "A2"
    e["outcome"] = "error"
    project.save_event(e, e["revision"])
    a2 = next(p for p in report(project)["players"] if p["id"] == "A2")
    assert a2["attack_efficiency"]["value"] == -100
    assert project.undo()
    assert next(p for p in report(project)["players"] if p["id"] == "A1")["attack_kills"] == 1


def test_rerun_never_overwrites_manual_and_no_duplicates(project):
    src = project.all("source")[0]
    auto = event(src["id"], 6.8, "attack", review="unreviewed", origin="automatic")
    auto["evidence"]["type"] = "hypothesis"
    eid = project.propose(auto, "stable-key")
    e = next(e for e in project.events() if e["id"] == eid)
    e.update(player_id="B1", team_id="B", outcome="kill", review="confirmed")
    e["evidence"].update(type="confirmed", player_id="confirmed", team_id="confirmed", outcome="confirmed")
    project.save_event(e)
    count = len(project.events())
    auto["outcome"] = "error"
    project.propose(auto, "stable-key")
    assert len(project.events()) == count
    after = next(e for e in project.events() if e["id"] == eid)
    assert after["outcome"] == "kill" and after["player_id"] == "B1"


def test_field_uncertainty_and_auto_not_human(project):
    src = project.all("source")[0]
    e = event(src["id"], 7.2, "attack", team_id="A", player_id="A1", outcome="kill", direction="line", review="unreviewed", origin="automatic")
    e["evidence"].update(type="accepted",team_id="accepted",player_id="accepted",outcome="hypothesis",direction="unknown")
    project.propose(e, "auto-accepted")
    automatic = next(x for x in report(project)["events"] if x["id"]==e["id"])
    assert automatic["player_id"] == "A1" and automatic["outcome"] is None and automatic["direction"] is None
    assert e["id"] not in {x["id"] for x in report(project,"confirmed")["events"]}
    assert automatic["review"] == "unreviewed"


def test_floor_geometry_control_and_air():
    image = [[0,0],[1800,0],[1800,900],[0,900],[900,0],[900,900]]
    world = [[0,0],[18,0],[18,9],[0,9],[9,0],[9,9]]
    c = fit(image,world,[[600,450]],[[6,4.5]])
    assert c["verified"]
    assert project_point(c["matrix"],[600,450],on_floor=True)==pytest.approx([6,4.5])
    with pytest.raises(ValueError): project_point(c["matrix"],[600,450])
    with pytest.raises(ValueError): fit([[0,0],[1,0],[2,0],[3,1]],[[0,0],[6,0],[9,0],[18,9]])
    assert normalize([3,2],"right") == [15,7]
    assert normalize([3,2],None) is None


def test_index_proxy_vfr_clips_and_reopen(project, tmp_path):
    src = project.all("source")[0]
    assert src["variable_fps"]
    proxy = create_proxy(project,src,height=180)
    assert proxy.exists()
    src = project.get("source",src["id"])
    assert src["proxy_max_time_error"] <= .002
    e = next(e for e in project.events() if e["type"] == "attack")
    clip = tmp_path / "clip.mp4"
    m = export_clip(project,e,clip,before=.1,after=.2)
    # Export contract is the half-open interval [start,end), including VFR boundaries.
    original_times = [f.time for f in frames(project.resolve(src),src,start=m["requested_start"],end=m["requested_end"])
                      if f.time < m["requested_end"]-1e-9]
    clip_times = [f.time for f in frames(clip)]
    assert len(clip_times) == len(original_times)
    assert np.asarray(clip_times)==pytest.approx(np.asarray(original_times)-original_times[0],abs=.002)
    assert probe(clip)["duration"] == pytest.approx(m["requested_end"]-original_times[0],abs=.002)
    montage([clip,clip],tmp_path/"montage.mp4")
    assert probe(tmp_path/"montage.mp4")["duration"] == pytest.approx(2*probe(clip)["duration"],abs=.002)
    target = tmp_path / "backup.sqlite"
    project.backup(target)
    reopened_dir = tmp_path / "reopened"; reopened_dir.mkdir(); target.rename(reopened_dir/"project.sqlite")
    with Project(reopened_dir) as other: assert other.events()==project.events()


def test_unknown_outcome_not_point_and_export_empty(project,tmp_path):
    assert len(report(project)["point_ledger"]) == 3
    path = export_all(project,tmp_path/"empty",filters={"type":"freeball"})
    data = json.loads((path/"report.json").read_text(encoding="utf-8"))
    assert data["events"]==[]
    assert (path/"report.html").exists() and (path/"statistics.xlsx").exists()
    assert (path/"events.csv").read_bytes().startswith(b'\xef\xbb\xbf')
    source = project.all("source")[0]; source["path"]="missing.MOV"
    with pytest.raises(FileNotFoundError,match="Восстановить путь"): project.resolve(source)


def test_ocr_voting_does_not_create_identity():
    assert consensus([{"number":7}])["number"] is None
    assert consensus([{"number":7}]*4+[{"number":1}])["number"] == 7
    assert consensus([{"number":7}]*3+[{"number":1}]*3)["number"] is None


def test_multiple_files_gap(project,tmp_path):
    f=tmp_path/"second.mp4"; make_video(f,seconds=1,vfr=False)
    src = import_video(project,f)
    assert src["gap_unknown"] and src["gap_before"] is None
    assert src["offset"]==pytest.approx(project.all("source")[0]["duration"])
    first=project.all('source')[0];first['proxy_verified']=False;project.put('source',first)
    assert project.all('source')[0]['id']==first['id']


def test_track_player_interval_correction_and_unknown_return(project):
    project.bind_track("run","fragment-1",0,8,"A1")
    project.bind_track("run","fragment-1",3,4,"A2")
    assert project.track_identity("run","fragment-1",2)["player_id"]=="A1"
    assert project.track_identity("run","fragment-1",3.5)["player_id"]=="A2"
    assert project.track_identity("run","fragment-1",5)["player_id"]=="A1"
    # A lost/returning fragment with the same jersey number has no implicit identity.
    assert project.track_identity("run","fragment-after-loss",5) is None
    a=project.get("player","A1");b=project.get("player","B1")
    assert a["number"]==b["number"] and a["id"]!=b["id"] and a["team_id"]!=b["team_id"]
