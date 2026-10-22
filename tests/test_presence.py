import json

import pytest

from volleyvision.presence import presence_report
from volleyvision.store import Project


@pytest.fixture
def project(tmp_path):
    with Project(tmp_path / "presence", create=True) as p:
        p.put("source", {"id": "source", "width": 100, "height": 100,
                         "display_width": 100, "display_height": 100, "duration": 10, "offset": 0})
        p.put("job", {"id": "run", "source_id": "source",
                      "params": {"params": {"people_max_gap": .4}}})
        p.put("team", {"id": "A", "name": "А"})
        p.put("player", {"id": "A1", "team_id": "A", "number": 1})
        p.put("player", {"id": "A2", "team_id": "A", "number": 2})
        yield p


def detection(track="track", x=20, y=40, observed=True):
    return {"track_id": track, "bbox": [x-5, y-20, x+5, y], "class_id": 1, "observed": observed}


def observe(p, t, people, sampled=True, source="source", run="run"):
    obs = {"time": t, "people": people, "people_sampled": sampled}
    with p.db:
        p.db.execute("INSERT INTO observations VALUES(?,?,?,?)", (run, source, t, json.dumps(obs)))


def total(grid):
    return sum(map(sum, grid))


def test_irregular_pts_counts_are_not_presence_seconds(project):
    project.bind_track("run", "track", 0, 10, "A1")
    observe(project, 0, [detection(x=20)])
    observe(project, .1, [detection(x=80)])
    observe(project, .35, [detection(x=80)])
    data = presence_report(project, grid=(2, 1))
    row = data["sources"][0]
    assert row["detection_grid"] == [[1, 2]]
    assert row["detection_count"] == 3
    assert row["presence_seconds_grid"][0] == pytest.approx([.05, .30])
    assert row["estimated_person_seconds"] == pytest.approx(.35)
    player = row["players"][0]
    assert player["player_id"] == "A1"
    assert player["estimated_presence_seconds"] == pytest.approx(.35)
    assert player["estimate"]["denominator_seconds"] == pytest.approx(.35)
    assert player["estimate"]["sample_size"] == 2
    assert data["completeness"] is None


def test_lost_sample_long_gap_and_new_track_are_not_bridged(project):
    observe(project, 0, [detection()])
    observe(project, .1, [])
    observe(project, .2, [detection()])
    observe(project, 1.0, [detection()])
    observe(project, 1.1, [detection(track="returning-fragment")])
    row = presence_report(project)["sources"][0]
    assert row["detection_count"] == 4
    assert row["estimated_person_seconds"] is None
    assert total(row["presence_seconds_grid"]) == 0
    assert row["diagnostics"]["unsupported_sample_gap_seconds"] == pytest.approx(.8)


def test_unsampled_ball_frame_is_not_a_person_loss(project):
    observe(project, 0, [detection()])
    observe(project, .05, [], sampled=False)
    observe(project, .1, [detection()])
    row = presence_report(project)["sources"][0]
    assert row["people_sampled_frames"] == 2
    assert row["estimated_person_seconds"] == pytest.approx(.1)


def test_referee_and_outside_are_excluded_unknown_is_not_assigned(project):
    project.bind_track("run", "ref", 0, 10, role="referee")
    project.bind_track("run", "bench", 0, 10, role="outside")
    for t in (0, .2):
        observe(project, t, [detection("ref"), detection("bench"), detection("unknown")])
    row = presence_report(project)["sources"][0]
    assert row["detection_count"] == 2
    assert row["estimated_person_seconds"] == pytest.approx(.2)
    assert row["players"] == []
    assert row["diagnostics"]["excluded_role_detections"] == 4
    assert row["diagnostics"]["unknown_identity_detections"] == 2
    assert row["diagnostics"]["unknown_identity_person_seconds"] == pytest.approx(.2)


def test_same_player_on_simultaneous_tracks_has_no_double_credit(project):
    for track in ("one", "two"):
        project.bind_track("run", track, 0, 10, "A1")
    for t in (0, .2):
        observe(project, t, [detection("one", x=20), detection("two", x=80)])
    row = presence_report(project)["sources"][0]
    assert row["detection_count"] == 4  # detector observations, not four accepted player actions
    assert row["estimated_person_seconds"] is None
    player = row["players"][0]
    assert player["detection_count"] == 0
    assert player["estimated_presence_seconds"] is None
    assert player["conflicting_detection_samples"] == 2
    assert player["conflicting_interval_seconds"] == pytest.approx(.2)
    assert row["conflicts_sample"][0]["player_id"] == "A1"


def test_manual_interval_correction_splits_estimate(project):
    project.bind_track("run", "track", 0, 10, "A1")
    project.bind_track("run", "track", .1, .2, "A2")
    observe(project, 0, [detection()])
    observe(project, .3, [detection()])
    players = {p["player_id"]: p for p in presence_report(project)["players"]}
    assert players["A1"]["estimated_presence_seconds"] == pytest.approx(.2)
    assert players["A2"]["estimated_presence_seconds"] == pytest.approx(.1)


def test_evidence_has_actual_interval_endpoints_when_player_has_no_detection(project):
    project.bind_track("run", "track", 0, 10, "A1")
    project.bind_track("run", "track", .1, .2, "A2")
    observe(project, 0, [detection(x=20)])
    observe(project, .3, [detection(x=80)])
    data = presence_report(project, grid=(2, 1))
    players = {p["player_id"]: p for p in data["players"]}
    middle_player = players["A2"]
    assert middle_player["detection_count"] == 0
    assert middle_player["presence_seconds_grid"][0] == pytest.approx([.05, .05])
    for evidence in middle_player["evidence_grid"][0]:
        assert evidence["first_time"] == 0
        assert evidence["last_time"] == .3
        assert evidence["representative_times"] == [0, .3]
        assert evidence["detection_samples"] == 0
        assert evidence["interval_contributions"] == 1
        assert evidence["estimate_interval_start"] == pytest.approx(.1)
        assert evidence["estimate_interval_end"] == pytest.approx(.2)
    source = data["sources"][0]
    assert source["evidence_time_basis"] == "source_seconds_from_first_video_frame"
    assert source["evidence_grid"][0][0]["first_time"] == 0
    assert source["evidence_grid"][0][1]["last_time"] == .3


def test_evidence_is_bounded_and_empty_cells_have_no_false_pointer(project):
    project.bind_track("run", "track", 0, 10, "A1")
    for index in range(1000):
        observe(project, index / 100, [detection()])
    data = presence_report(project, grid=(2, 1))
    source = data["sources"][0]
    for row in (source, source["players"][0]):
        evidence = row["evidence_grid"][0][0]
        assert evidence["first_time"] == 0
        assert evidence["last_time"] == 9.99
        assert len(evidence["representative_times"]) == 3
        assert evidence["representative_times"][0] == 0
        assert evidence["representative_times"][-1] == 9.99
        assert evidence["detection_samples"] == 1000
        assert evidence["interval_contributions"] == 999
        assert row["evidence_grid"][0][1] is None


def test_identity_conflict_at_endpoint_does_not_credit_stable_player_interval(project):
    for track in ("one", "two"):
        project.bind_track("run", track, 0, 10, "A1")
    observe(project, 0, [detection("one")])
    observe(project, .2, [detection("one"), detection("two")])
    row = presence_report(project)["sources"][0]
    assert row["players"][0]["estimated_presence_seconds"] is None
    assert row["players"][0]["conflicting_interval_seconds"] == pytest.approx(.2)


def test_no_prediction_or_other_file_bridge_and_latest_run(project):
    observe(project, 0, [detection()])
    observe(project, .1, [detection(observed=False)])
    observe(project, .2, [detection()])
    observe(project, .25, [detection()], source="other-file")
    observe(project, .3, [detection()])
    row = presence_report(project)["sources"][0]
    assert row["estimated_person_seconds"] is None
    assert row["diagnostics"]["excluded_predictions"] == 1
    assert row["diagnostics"]["other_source_rows"] == 1
    project.put("job", {"id": "new-run", "source_id": "source", "params": {"params": {}}})
    assert presence_report(project)["sources"][0]["run_id"] == "new-run"
    explicit = presence_report(project, run_ids=["run", "new-run"])
    assert len(explicit["sources"]) == 2
    assert all(s["estimated_person_seconds"] is None for s in explicit["sources"])


def test_invalid_and_duplicate_track_observations_are_not_integrated(project):
    observe(project, 0, [detection(), detection()])
    observe(project, .1, [detection()])
    observe(project, .2, [{"track_id": "invalid", "bbox": [0,0,float("nan"),20], "observed": True}])
    row = presence_report(project)["sources"][0]
    assert row["estimated_person_seconds"] is None
    assert row["diagnostics"]["duplicate_track_samples"] == 1
    assert row["diagnostics"]["invalid_detections"] == 1


def test_empty_project_and_bad_grid(project):
    assert presence_report(project)["sources"][0]["estimated_person_seconds"] is None
    assert presence_report(project, source_ids=["not-selected"])["sources"] == []
    with pytest.raises(ValueError): presence_report(project, grid=(0, 1))
    with pytest.raises(ValueError): presence_report(project, run_ids=["missing"])


def test_view_boundary_breaks_time_even_if_input_reuses_track_id(project):
    from volleyvision.store import dumps
    for t,segment in ((0,'one'),(.1,'one'),(.2,'two'),(.3,'two')):
        body={'time':t,'people_sampled':True,'people':[detection()],'tracking_segment':segment,
              'approximate_zones_valid':False}
        project.db.execute('INSERT INTO observations VALUES(?,?,?,?)',('run','source',t,dumps(body)))
    project.db.commit();row=presence_report(project)['sources'][0]
    assert row['estimated_person_seconds']==pytest.approx(.2) and row['view_segment_count']==2
    assert row['people_search_roi'] is None and any('нескольких сегментов' in s for s in row['limitations'])
