import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.export_temporal import export_dataset
from volleyvision.media import Cancelled
from volleyvision.store import Project, dumps
from volleyvision.temporal import FEATURE_NAMES, load_dataset


@pytest.fixture
def project(tmp_path):
    with Project(tmp_path / "project", create=True, synthetic=True) as project:
        project.put("source", {"id": "source", "path": "missing-original.mp4", "sha256": "a" * 64,
            "duration": 4.1, "display_width": 100, "display_height": 100, "offset": 0,
            "time_base": "1/90000", "first_pts": 4500})
        project.put("job", {"id": "run", "source_id": "source", "state": "complete", "last_time": 4., "samples": 41})
        with project.db:
            for index in range(41):
                t = index / 10
                observation = {"time": t, "source_pts": 4500 + index * 9000, "people_sampled": True,
                    "people": [{"bbox": [30, 20, 50, 90], "observed": True, "track_id": "person"}],
                    "balls": [{"bbox": [40 + index / 10, 10, 44 + index / 10, 14], "observed": True}],
                    "geometry_valid": False}
                project.db.execute("INSERT INTO observations VALUES(?,?,?,?)", ("run", "source", t, dumps(observation)))
        yield project


def test_export_is_unreviewed_with_real_pts_and_training_schema(project, tmp_path):
    target = tmp_path / "windows.json"
    result = export_dataset(project, "run", target, stride_seconds=.5)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert result["windows"] > 2
    assert data["data_kind"] == "synthetic" and tuple(data["feature_names"]) == FEATURE_NAMES
    assert data["coverage_verified"] is False and data["annotation_intervals"] == []
    assert data["source_sha256"] == "a" * 64 and data["source_time_base"] == "1/90000"
    for window in data["windows"]:
        assert window["reviewed"] is False and window["label"] is None
        assert window["source_pts"] == 4500 + round(window["anchor"] * 90000)
        assert window["evidence"]["viewer_time"] == window["anchor"]
        assert window["match_id"] == data["match_id"] and window["data_kind"] == "synthetic"
    with pytest.raises(ValueError, match="проверенная разметка"):
        load_dataset(target)
    # Human annotation is a separate explicit operation, not performed by the exporter.
    for window in data["windows"]:
        window.update(label="none", reviewed=True)
    target.write_text(json.dumps(data), encoding="utf-8")
    reviewed = load_dataset(target)
    assert len(reviewed["windows"]) == result["windows"]
    viewer = target.with_suffix(".viewer.html").read_text(encoding="utf-8")
    assert 'data-time="' in viewer and "Видео сейчас недоступно" in viewer


def test_export_uses_current_gallery_roles_without_changing_observations(project, tmp_path):
    project.put("track_binding", {"id": "referee", "run_id": "run", "track_id": "person", "start": 0,
                                 "end": 4.1, "role": "referee", "player_id": None})
    target = tmp_path / "roles.json"
    export_dataset(project, "run", target)
    data = json.loads(target.read_text(encoding="utf-8"))
    person_visible = FEATURE_NAMES.index("person_visible")
    assert all(row[person_visible] == 0 for window in data["windows"] for row in window["features"])
    assert next(project.observations("run"))["people"][0].get("role") is None
    assert data["track_bindings_sha256"]


def test_match_identity_is_stable_across_exports_and_distinct_projects(project, tmp_path):
    first = export_dataset(project, "run", tmp_path / "first.json", max_windows=2)
    second = export_dataset(project, "run", tmp_path / "second.json", max_windows=2)
    a = json.loads((tmp_path / "first.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "second.json").read_text(encoding="utf-8"))
    assert first["match_id"] == second["match_id"]
    assert [w["id"] for w in a["windows"]] == [w["id"] for w in b["windows"]]
    assert a["export_summary"]["truncated"] is True and len(a["windows"]) == 2
    with pytest.raises(ValueError, match="другой match_id"):
        export_dataset(project, "run", tmp_path / "renamed.json", match_id="wrong-other-match")
    with Project(tmp_path / "other", create=True, synthetic=True) as other:
        other.put("source", project.get("source", "source"))
        other.put("job", project.get("job", "run"))
        result = export_dataset(other, "run", tmp_path / "other.json")
        assert result["match_id"] != first["match_id"]
        assert result["windows"] == 0 and result["warning"] is not None


def test_cancel_preserves_old_annotation_and_cleans_partial_files(project, tmp_path):
    target = tmp_path / "annotated.json"
    original = '{"reviewed":true,"user_note":"сохранить"}'
    target.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="уже существует"):
        export_dataset(project, "run", target)
    stopped = False
    def progress(fraction, message):
        nonlocal stopped
        stopped = True
    with pytest.raises(Cancelled):
        export_dataset(project, "run", target, overwrite=True, progress=progress, cancel=lambda: stopped)
    assert target.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".*.partial"))
    assert not target.with_suffix(".viewer.html").exists()


def test_invalid_source_interval_and_unknown_provenance_do_not_create_dataset(project, tmp_path):
    target = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="Не найдено задание"):
        export_dataset(project, "unknown-run", target)
    with pytest.raises(ValueError, match="внутри исходного"):
        export_dataset(project, "run", target, start=-1)
    with pytest.raises(ValueError, match="max_steps"):
        export_dataset(project, "run", target, max_steps=300)
    match = project.get("match", "match")
    match["synthetic"] = None
    project.put("match", match)
    with pytest.raises(ValueError, match="происхождение"):
        export_dataset(project, "run", target)
    assert not target.exists()


def test_cli_export_is_usable_and_never_sets_background_label(project, tmp_path):
    target = tmp_path / "cli.json"
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/export_temporal.py"),
        str(project.root), "run", str(target), "--start", ".3", "--end", "3.8", "--max-windows", "3"],
        capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
    assert result.returncode == 0, result.stderr
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data["windows"]) == 3
    assert all(w["label"] is None and w["reviewed"] is False for w in data["windows"])
    assert data["requested_interval"] == {"start": .3, "end": 3.8}
    assert json.loads(result.stdout)["windows"] == 3
