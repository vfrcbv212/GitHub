"""Export real observation windows for annotation, without inventing action labels."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from volleyvision.domain import uid
from volleyvision.media import Cancelled, check_cancel
from volleyvision.store import Project
from volleyvision.temporal import (
    ACTION_CLASSES, FEATURE_NAMES, SCHEMA_VERSION, TrackBindingIndex, observation_windows,
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _match_identity(project, requested):
    """Give all parts of one project a stable ID, including after a disk move.

    Older projects use the literal 'match' as their storage key; using that as a
    dataset identity would incorrectly merge every separate exported match.
    """
    saved = project.get("dataset_identity", "temporal")
    if requested is not None and (not isinstance(requested, str) or not requested.strip()):
        raise ValueError("match_id должен быть непустой строкой")
    requested = requested.strip() if requested is not None else None
    if saved:
        if requested is not None and requested != saved["match_id"]:
            raise ValueError("Для проекта уже сохранён другой match_id. Используйте прежний ID, чтобы части одного матча не попали в разные выборки")
        return saved["match_id"]
    match = project.get("match", "match") or {}
    match_id = requested or match.get("dataset_match_id") or "match-" + uid()
    if not isinstance(match_id, str) or not match_id.strip():
        raise ValueError("В проекте задан некорректный dataset_match_id")
    project.put("dataset_identity", {"id": "temporal", "match_id": match_id,
                                    "origin": "explicit" if requested else "stable_project_id"})
    return match_id


def _viewer_header(stream, video_url, source_name, time_note):
    stream.write('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Разметка временных окон</title>')
    stream.write('<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;color:#203344}video{width:100%;max-height:65vh}td,th{padding:8px;border-bottom:1px solid #ccd4da;text-align:left}table{border-collapse:collapse;width:100%}.warning{background:#ffe6b0;padding:12px}button{cursor:pointer}</style>')
    stream.write('<h1>Окна для разметки — действия не размечены</h1>')
    stream.write('<p class="warning">Это исходные временные признаки, а не эталон и не качество модели. Метки label=null и reviewed=false нужно заполнить после просмотра последовательности. Не называйте отсутствие чтения мяча фоновым действием автоматически.</p>')
    stream.write('<p>Источник: ' + html.escape(source_name) + '. ' + html.escape(time_note) + '</p>')
    if video_url:
        stream.write('<video controls preload="metadata" src="' + html.escape(video_url, quote=True) + '"></video>')
    else:
        stream.write('<p>Видео сейчас недоступно. Восстановите путь в приложении; времена и PTS сохранены в JSON.</p>')
    stream.write('<p>Браузер может не поддерживать MOV/HEVC. Для точного покадрового PTS-просмотра используйте приложение. Оригинал не меняется.</p>')
    stream.write('<table><thead><tr><th>Окно</th><th>Опорный кадр, с</th><th>Интервал</th><th>Исходный PTS</th></tr></thead><tbody>')


def _viewer_footer(stream):
    stream.write('</tbody></table><script>const video=document.querySelector("video");document.querySelectorAll("button[data-time]").forEach(button=>button.addEventListener("click",()=>{if(video){video.currentTime=Number(button.dataset.time);video.scrollIntoView({block:"center"});}}));</script></html>')


def export_dataset(project, run_id, destination, *, window_seconds=2., stride_seconds=.25,
                   max_steps=256, start=0., end=None, match_id=None, max_windows=None,
                   overwrite=False, progress=None, cancel=None):
    """Stream a training-schema JSON plus optional local viewer, all unreviewed.

    Only one bounded observation window is retained at a time. Exporting a
    complete source does not claim its events or window labels are complete.
    A failed/cancelled export never replaces an existing annotation JSON.
    """
    destination = Path(destination).resolve()
    if destination.suffix.lower() != ".json":
        raise ValueError("Выберите файл JSON для набора временных окон")
    viewer = destination.with_suffix(".viewer.html")
    if not overwrite and (destination.exists() or viewer.exists()):
        raise ValueError("Файл экспорта или просмотрщик уже существует. Выберите другое имя; --overwrite явно заменяет прежнюю разметку")
    for value in (window_seconds, stride_seconds):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Длительность и шаг окна должны быть положительными конечными числами")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 3 <= max_steps <= 256:
        raise ValueError("max_steps должен быть целым числом от 3 до 256")
    if max_windows is not None and (not isinstance(max_windows, int) or isinstance(max_windows, bool) or max_windows < 1):
        raise ValueError("max_windows должен быть положительным целым числом")
    job = project.get("job", run_id)
    if not job:
        raise ValueError("Не найдено задание детекции. Возьмите run_id из вывода analyze или setup-scan")
    source = project.get("source", job.get("source_id"))
    if not source:
        raise ValueError("Не найден исходник задания детекции")
    source_sha = source.get("sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in source_sha):
        raise ValueError("В исходнике нет корректной SHA256. Сначала завершите импорт видео")
    source_sha = source_sha.lower()
    width, height = source.get("display_width"), source.get("display_height")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (width, height)):
        raise ValueError("Не сохранены размеры повёрнутого исходного кадра")
    end = source["duration"] if end is None else end
    if (any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end))
            or start < 0 or end <= start or end > source["duration"] + 1e-6):
        raise ValueError("Нужен непустой интервал внутри исходного файла, в секундах от первого видеокадра")
    match = project.get("match", "match")
    if not match or not isinstance(match.get("synthetic"), bool):
        raise ValueError("В проекте не указано происхождение данных synthetic=true/false; не объявляйте его реальным по умолчанию")
    check_cancel(cancel)
    stable_match_id = _match_identity(project, match_id)
    bindings = TrackBindingIndex(project.all("track_binding"), run_id)
    statistics = project.db.execute(
        "SELECT COUNT(*),MIN(time),MAX(time) FROM observations WHERE run_id=? AND time>=? AND time<=?",
        (run_id, start, end)).fetchone()
    video_url, video_kind = None, "unavailable"
    try:
        use_proxy = bool(source.get("proxy_verified") and source.get("proxy"))
        video_url = project.resolve(source, proxy=use_proxy).as_uri()
        video_kind = "verified_proxy" if use_proxy else "original"
    except FileNotFoundError:
        pass
    metadata = {
        "schema_version": SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES),
        "data_kind": "synthetic" if match["synthetic"] else "real",
        "window_seconds": float(window_seconds), "stride_seconds": float(stride_seconds),
        "max_steps": max_steps, "match_id": stable_match_id,
        "source_id": source["id"], "source_sha256": source_sha,
        "source_time_base": source.get("time_base"), "source_first_pts": source.get("first_pts"),
        "source_dimensions": {"width": width, "height": height, "coordinate_space": "rotated_original"},
        "run_id": run_id, "detector_job_state": job.get("state", "unknown"),
        "detector_last_time": job.get("last_time"),
        "observation_summary": {"rows": statistics[0], "first_time": statistics[1], "last_time": statistics[2]},
        "requested_interval": {"start": float(start), "end": float(end)},
        "track_bindings_sha256": bindings.signature,
        "annotation_intervals": [], "coverage_verified": False,
        "allowed_labels": list(ACTION_CLASSES), "annotation_state": "unreviewed",
        "evidence_viewer": {"file": viewer.name, "video_kind": video_kind,
                            "time_origin": "first_original_video_frame", "original_sha_rechecked": False},
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "notes": ["label=null и reviewed=false во всех окнах; метки не выведены из модели или отсутствия детекций",
                  "Ближайший к мячу человек не является доказанным участником действия",
                  "Полнота эталона не подтверждается экспортом; разбиение на train/validation/test выполняется по матчам"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uid()
    partial = destination.with_name(f".{destination.name}.{token}.partial")
    partial_viewer = viewer.with_name(f".{viewer.name}.{token}.partial")
    count, truncated, first_anchor, last_anchor = 0, False, None, None
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as stream, partial_viewer.open("w", encoding="utf-8", newline="\n") as display:
            stream.write("{\n")
            for key, value in metadata.items():
                stream.write(_json(key) + ":" + _json(value) + ",\n")
            stream.write('"windows":[\n')
            _viewer_header(display, video_url, Path(source["path"]).name,
                           "Все времена относятся к первому кадру оригинала; гипотезы не подтверждены человеком.")
            observations = bindings.overlay(project.observations(run_id, start=start, end=end))
            for window in observation_windows(observations, width, height, window_seconds, stride_seconds, max_steps):
                check_cancel(cancel)
                if max_windows is not None and count >= max_windows:
                    truncated = True
                    break
                identifier = hashlib.sha256(_json({"source_sha256": source_sha, "run_id": run_id,
                    "feature_names": FEATURE_NAMES, "window": window, "track_bindings_sha256": bindings.signature}).encode("utf-8")).hexdigest()[:32]
                window_id = "window-" + identifier
                exported = window | {
                    "id": window_id, "match_id": stable_match_id, "source_id": source["id"],
                    "source_sha256": source_sha, "source_time_base": source.get("time_base"),
                    "source_first_pts": source.get("first_pts"), "run_id": run_id, "data_kind": metadata["data_kind"],
                    "label": None, "reviewed": False,
                    "unknown_reasons": {"label": "окно ещё не размечено человеком"},
                    "evidence": {"viewer": viewer.name + "#" + window_id, "source_id": source["id"],
                                 "viewer_time": window["anchor"], "source_pts": window["source_pts"],
                                 "window_start": window["start"], "window_end": window["end"]},
                }
                if count:
                    stream.write(",\n")
                stream.write(_json(exported))
                display.write('<tr id="' + window_id + '"><td>' + window_id + '</td><td><button data-time="' +
                    f"{window['anchor']:.9f}" + '">' + f"{window['anchor']:.6f}" + '</button></td><td>' +
                    f"{window['start']:.6f}–{window['end']:.6f}" + '</td><td>' + html.escape(str(window["source_pts"])) + '</td></tr>')
                count += 1
                first_anchor = window["anchor"] if first_anchor is None else first_anchor
                last_anchor = window["anchor"]
                if progress:
                    progress(min(1., (window["end"]-start)/(end-start)), f"Экспортировано временных окон: {count}")
            check_cancel(cancel)
            summary = {"windows": count, "reviewed_windows": 0, "truncated": truncated,
                       "first_anchor": first_anchor, "last_anchor": last_anchor,
                       "warning": "Слишком мало наблюдений для выбранного окна" if not count else None}
            stream.write('\n],"export_summary":' + _json(summary) + "\n}\n")
            _viewer_footer(display)
        partial_viewer.replace(viewer)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
        partial_viewer.unlink(missing_ok=True)
    return {"destination": str(destination), "viewer": str(viewer), "match_id": stable_match_id,
            "data_kind": metadata["data_kind"], **summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Экспорт временных окон из сохранённых детекций для ручной разметки")
    parser.add_argument("project", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--window-seconds", type=float, default=2.)
    parser.add_argument("--stride-seconds", type=float, default=.25)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--end", type=float)
    parser.add_argument("--match-id")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Явно заменить существующий JSON и просмотрщик, включая прежние метки")
    args = parser.parse_args(argv)
    try:
        with Project(args.project) as project:
            result = export_dataset(project, args.run_id, args.destination,
                window_seconds=args.window_seconds, stride_seconds=args.stride_seconds, max_steps=args.max_steps,
                start=args.start, end=args.end, match_id=args.match_id, max_windows=args.max_windows,
                overwrite=args.overwrite,
                progress=lambda fraction, message: print(message, file=sys.stderr, flush=True))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, Cancelled, KeyboardInterrupt) as exc:
        print(f"Экспорт временных окон: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
