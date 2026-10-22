import os

import cv2
import numpy as np
import pytest
from PySide6 import QtCore

from volleyvision.gallery import (
    BindingConflict, GalleryDialog, assign_fragments, assign_interval,
    effective_bindings, existing_conflicts, gallery_summary,
)
from volleyvision.store import Project


@pytest.fixture
def project(tmp_path):
    with Project(tmp_path / "gallery", create=True) as project:
        project.put("source", {"id": "source", "path": "video.mp4", "duration": 20, "offset": 0})
        for team in ("A", "B"):
            project.put("team", {"id": team, "name": f"Команда {team}"})
            project.put("player", {"id": f"{team}7", "team_id": team, "number": 7, "name": None})
        samples = []
        for i, t in enumerate((0.123456, 0.6, 1.8)):
            path = project.root / "cache" / f"sample-{i}.jpg"
            cv2.imwrite(str(path), np.full((60, 40, 3), 40 + 50 * i, dtype=np.uint8))
            samples.append({"path": project.relpath(path), "time": t})
        for track_id, start, end in (("first", 0, 2), ("later", 3, 5), ("simultaneous", 1, 4)):
            project.put("track", {"id": track_id, "run_id": "run", "source_id": "source", "start": start,
                "end": end, "samples": samples if track_id == "first" else [],
                "number_consensus": {"number": 7, "votes": {"7": 4}, "reason": "согласованные наблюдения"},
                "ocr_observations": [{"time": 1.8, "number": 7, "raw_score": .8, "reason": "наблюдение OCR"}]})
        yield project


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_explicit_nonoverlapping_fragments_do_not_propagate_identity(project):
    bindings = assign_fragments(project, ["first", "later"], "A7")
    assert len(bindings) == 2
    assert all(b["identity_status"] == "confirmed" and b["team_id"] == "A" for b in bindings)
    assert project.track_identity("run", "first", 1)["player_id"] == "A7"
    assert project.track_identity("run", "later", 4)["player_id"] == "A7"
    assert project.track_identity("run", "simultaneous", 2) is None
    summary = gallery_summary(project)
    assert summary["players_confirmed"] == 1 and summary["unknown_tracks"] == 1
    assert summary["players_without_samples"] == ["B7"]


def test_overlapping_merge_is_atomic_and_different_teams_stay_distinct(project):
    with pytest.raises(BindingConflict, match="пересекаются"):
        assign_fragments(project, ["first", "simultaneous"], "A7")
    assert project.all("track_binding") == []
    assign_interval(project, "first", 0, 2, "A7", "A")
    with pytest.raises(BindingConflict, match="другой траектории"):
        assign_interval(project, "simultaneous", 1, 4, "A7", "A")
    assign_interval(project, "simultaneous", 1, 4, "B7", "B")
    assert project.track_identity("run", "first", 1.5)["player_id"] == "A7"
    assert project.track_identity("run", "simultaneous", 1.5)["player_id"] == "B7"
    assert existing_conflicts(project) == []


def test_interval_override_and_unknown_revoke_without_removing_history(project):
    assign_interval(project, "first", 0, 2, "A7")
    assign_interval(project, "first", .5, 1.5, "B7")
    # The old A7 claim has been corrected on this interval, so another track may use it.
    assign_interval(project, "simultaneous", 1, 1.4, "A7")
    assert project.track_identity("run", "first", .2)["player_id"] == "A7"
    assert project.track_identity("run", "first", 1.2)["player_id"] == "B7"
    assign_interval(project, "first", 0, 2)
    assert project.track_identity("run", "first", 1.2)["player_id"] is None
    assert len(project.all("track_binding")) == 4
    assert all(b.get("player_id") is None for b in effective_bindings(project) if b["track_id"] == "first")


def test_invalid_intervals_and_legacy_conflicts_are_visible(project):
    with pytest.raises(ValueError, match="внутри"):
        assign_interval(project, "first", -1, 2, "A7")
    with pytest.raises(ValueError, match="конечные"):
        assign_interval(project, "first", float("nan"), 1, "A7")
    project.bind_track("run", "first", 0, 2, "A7")
    project.bind_track("run", "simultaneous", 1, 4, "A7")
    conflicts = existing_conflicts(project)
    assert len(conflicts) == 1 and conflicts[0]["start"] == 1
    assert len(gallery_summary(project)["conflicts"]) == 1


def test_role_assignment_never_gives_referee_a_roster_identity(project):
    binding = assign_interval(project, "first", 0, 2, "A7", "A", role="referee")
    assert binding["player_id"] is None and binding["team_id"] is None
    assert binding["role"] == "referee"


def test_known_team_can_keep_unknown_player(project):
    binding = assign_interval(project, "first", 0, 2, team_id="A")
    assert binding["team_id"] == "A" and binding["player_id"] is None
    assert binding["identity_status"] == "unknown"
    assert gallery_summary(project)["players_confirmed"] == 0
    with pytest.raises(ValueError, match="не принадлежит"):
        assign_interval(project, "later", 3, 5, player_id="B7", team_id="A")


def test_multi_image_gallery_precise_preview_and_no_ocr_autobinding(project, qapp):
    project.put("identity_hypothesis", {"id": "hypothesis", "track_id": "first", "candidates": [
        {"player_id": "A7", "reference_track": "later", "appearance_distance": .1}]})
    dialog = GalleryDialog(project)
    row = next(i for i in range(dialog.track_list.count())
               if dialog.track_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == "first")
    dialog.track_list.setCurrentRow(row)
    assert dialog.sample_list.count() == 3
    assert all(not dialog.sample_list.item(i).icon().isNull() for i in range(3))
    assert "Голоса OCR" in dialog.evidence.toPlainText()
    assert "не подтверждённые привязки" in dialog.evidence.toPlainText()
    assert dialog.team.currentData() is None and dialog.player.currentData() is None
    assert project.all("track_binding") == []
    events = []
    dialog.previewRequested.connect(lambda source_id, t: events.append((source_id, t)))
    dialog.sample_list.setCurrentRow(0)
    dialog.preview_sample()
    assert events == [("source", .123456)]
    dialog.team.setCurrentIndex(dialog.team.findData("B"))
    assert dialog.player.findData("B7") >= 0 and dialog.player.findData("A7") == -1
    dialog.player.setCurrentIndex(dialog.player.findData("B7"))
    dialog.save_interval()
    assert project.track_identity("run", "first", 1)["player_id"] == "B7"
    dialog.close()
