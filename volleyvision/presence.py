"""Streaming image-space presence estimates, separate from volleyball events.

No detector observation proves a player was absent elsewhere. Time is estimated
only between adjacent sampled observations of the same temporary track. The
bottom centre of a box remains an image-space anchor, including during jumps;
this module never projects it onto the court plane.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict


EXCLUDED_ROLES = {"referee", "outside", "spectator", "bench"}
LIMITATIONS = [
    "Карта детекций считает наблюдения детектора людей, а не игровые действия.",
    "Оценка времени использует только соседние наблюдения одной траектории; длительные потери не заполняются.",
    "Время распределено поровну между ячейками двух наблюдаемых опорных точек; это оценка, не непрерывное измерение.",
    "Опорная точка — низ рамки в координатах видео; положение стоп, прыжок и координаты на полу не установлены.",
    "Нулевое число детекций не доказывает отсутствие игрока. Полнота относительно всего матча неизвестна.",
    "Временная траектория не равна личности игрока; неизвестные участники не распределяются между составом.",
    "Ссылки карты времени ведут к наблюдаемым концам интервала; они не подтверждают личность игрока в этих кадрах при изменении интервальной привязки.",
]


def _matrix(columns, rows):
    return [[0 for _ in range(columns)] for _ in range(rows)]


def _evidence_matrix(columns, rows):
    return [[None for _ in range(columns)] for _ in range(rows)]


def _add_evidence(grid, cell, times, *, detection=False, interval=None):
    """At most three video pointers per cell, independent of recording length.

    The first and latest time are retained, together with one representative
    closest to their midpoint among the bounded candidates available. This is
    a bounded preview sample, not a reservoir of every contributing observation.
    """
    evidence = grid[cell[0]][cell[1]]
    if evidence is None:
        evidence = {"first_time": None, "last_time": None, "representative_times": [],
                    "detection_samples": 0, "interval_contributions": 0,
                    "estimate_interval_start": None, "estimate_interval_end": None}
        grid[cell[0]][cell[1]] = evidence
    if detection:
        evidence["detection_samples"] += 1
    if interval is not None:
        evidence["interval_contributions"] += 1
        left, right = interval
        evidence["estimate_interval_start"] = (left if evidence["estimate_interval_start"] is None
                                               else min(left, evidence["estimate_interval_start"]))
        evidence["estimate_interval_end"] = (right if evidence["estimate_interval_end"] is None
                                             else max(right, evidence["estimate_interval_end"]))
    candidates = sorted(set(evidence["representative_times"]) | set(times))
    evidence["first_time"] = candidates[0] if evidence["first_time"] is None else min(evidence["first_time"], candidates[0])
    evidence["last_time"] = candidates[-1] if evidence["last_time"] is None else max(evidence["last_time"], candidates[-1])
    if len(candidates) > 3:
        middle = (candidates[0] + candidates[-1]) / 2
        representative = min(candidates[1:-1], key=lambda t: abs(t - middle))
        candidates = [candidates[0], representative, candidates[-1]]
    evidence["representative_times"] = candidates


def _cell(point, width, height, columns, rows):
    x, y = point
    return min(rows - 1, int(y / height * rows)), min(columns - 1, int(x / width * columns))


def _add(grid, cell, value):
    grid[cell[0]][cell[1]] += value


class _Bindings:
    """Indexed manual intervals; newest matching row wins, as in Project."""

    def __init__(self, project, run_id):
        self.rows = defaultdict(list)
        for row in project.all("track_binding"):
            if row.get("run_id") == run_id:
                self.rows[row["track_id"]].append(row)
        self.players = {p["id"]: p for p in project.all("player")}

    def at(self, track_id, t):
        for row in reversed(self.rows.get(track_id, [])):
            if row["start"] <= t <= row["end"]:
                return row
        return None

    def label(self, track_id, t):
        binding = self.at(track_id, t)
        role = binding.get("role", "unknown") if binding else "unknown"
        player_id = binding.get("player_id") if binding else None
        if role != "participant" or player_id not in self.players:
            player_id = None
        return role, player_id

    def boundaries(self, track_ids, start, end):
        points = {start, end}
        for tid in track_ids:
            for row in self.rows.get(tid, []):
                for value in (row["start"], row["end"]):
                    if start < value < end:
                        points.add(value)
        return sorted(points)


def _player_row(player, source_id, run_id, columns, rows):
    return {"player_id": player["id"], "team_id": player["team_id"],
            "source_id": source_id, "run_id": run_id, "number": player.get("number"),
            "detection_count": 0, "detection_grid": _matrix(columns, rows),
            "evidence_grid": _evidence_matrix(columns, rows),
            "presence_seconds_grid": _matrix(columns, rows), "estimated_presence_seconds": None,
            "supported_intervals": 0, "conflicting_detection_samples": 0,
            "conflicting_interval_seconds": 0.0, "completeness": None,
            "coordinate_space": "rotated_original_pixels"}


def _observed_people(obs, width, height, diagnostics):
    people = {}
    duplicates = set()
    for detection in obs.get("people", []):
        if detection.get("observed") is not True:
            diagnostics["excluded_predictions"] += 1
            continue
        if detection.get("class_id", 1) != 1:
            continue
        tid, box = detection.get("track_id"), detection.get("bbox")
        try:
            if not isinstance(tid, str) or not tid or len(box) != 4:
                raise ValueError
            x1, y1, x2, y2 = map(float, box)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                raise ValueError
            point = ((x1 + x2) / 2, y2)
            if not (0 <= point[0] <= width and 0 <= point[1] <= height):
                raise ValueError
        except (ValueError, TypeError):
            diagnostics["invalid_detections"] += 1
            continue
        if tid in people:
            duplicates.add(tid)
        people[tid] = point
    for tid in duplicates:
        del people[tid]
    diagnostics["duplicate_track_samples"] += len(duplicates)
    return people


def _one_source(project, job, source, columns, rows, max_gap):
    source_id, run_id = source["id"], job["id"]
    width = source.get("display_width", source.get("width"))
    height = source.get("display_height", source.get("height"))
    if not width or not height or width <= 0 or height <= 0:
        raise ValueError("Для карты наблюдений нужны размеры исходного изображения")
    settings = job.get("params", {}).get("params", {})
    gap = max_gap if max_gap is not None else settings.get("people_max_gap", 0.4)
    if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0:
        raise ValueError("Максимальный разрыв наблюдений должен быть положительным")
    bindings = _Bindings(project, run_id)
    diagnostic_names = ("excluded_predictions", "invalid_detections", "duplicate_track_samples",
                        "excluded_role_detections", "excluded_role_track_seconds", "unknown_identity_detections",
                        "unknown_identity_person_seconds", "conflicting_track_seconds", "conflicting_detection_samples",
                        "conflicting_intervals", "unsupported_sample_gap_seconds", "invalid_observation_rows",
                        "other_source_rows")
    diagnostics = {name: 0 for name in diagnostic_names}
    output = {"source_id": source_id, "run_id": run_id, "width": width, "height": height,
              "grid_columns": columns, "grid_rows": rows, "coordinate_space": "rotated_original_pixels",
              "anchor": "bbox_bottom_center", "grid_unit": "pixels_of_rotated_original",
              "detection_map_unit": "detector_observations", "presence_map_unit": "estimated_person_seconds",
              "detection_grid": _matrix(columns, rows), "presence_seconds_grid": _matrix(columns, rows),
              "evidence_grid": _evidence_matrix(columns, rows),
              "evidence_time_basis": "source_seconds_from_first_video_frame",
              "evidence_sample_limit_per_cell": 3,
              "detection_count": 0, "estimated_person_seconds": None, "supported_track_intervals": 0,
              "people_sampled_frames": 0, "sampled_timeline_seconds": 0.0,
              "first_sample_time": None, "last_sample_time": None, "max_gap_seconds": gap,
              "completeness": None, "diagnostics": diagnostics, "conflicts_sample": [],
              "limitations": list(LIMITATIONS), "players": []}
    calibration = job.get("params", {}).get("calibration", {}) or {}
    output["people_search_roi"] = calibration.get("people_roi")
    output["obstacles"] = calibration.get("obstacles", [])
    output['configured_people_search_roi']=calibration.get('people_roi')
    output['configured_obstacles']=calibration.get('obstacles',[])
    output['view_segments']=[];output['view_segment_count']=0;output['view_segments_truncated']=False
    previous_scope=None;active_segment=None;roi_unreliable=False
    player_rows = {}

    def player(pid):
        if pid not in player_rows:
            player_rows[pid] = _player_row(bindings.players[pid], source_id, run_id, columns, rows)
        return player_rows[pid]

    previous_time, previous_people, previous_conflicts = None, {}, set()
    for dbrow in project.db.execute("SELECT source_id,time,body FROM observations WHERE run_id=? ORDER BY time", (run_id,)):
        if dbrow[0] != source_id:
            diagnostics["other_source_rows"] += 1
            previous_time, previous_people, previous_conflicts = None, {}, set()
            continue
        try:
            t = float(dbrow[1])
            obs = json.loads(dbrow[2])
            if not math.isfinite(t) or t < 0 or not isinstance(obs, dict):
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError):
            diagnostics["invalid_observation_rows"] += 1
            previous_time, previous_people, previous_conflicts = None, {}, set()
            continue
        scope=tuple(obs.get(k) for k in ('calibration_id','camera_session_id','tracking_segment'))
        if scope!=previous_scope or active_segment is None:
            previous_time,previous_people,previous_conflicts=None,{},set()
            active_segment={'calibration_id':scope[0],'camera_session_id':scope[1],'tracking_segment':scope[2],
                            'start':t,'end':t,'people_sampled_frames':0}
            output['view_segment_count']+=1
            if len(output['view_segments'])<100:output['view_segments'].append(active_segment)
            else:output['view_segments_truncated']=True
        active_segment['end']=t;previous_scope=scope
        roi_unreliable|=not bool(obs.get('approximate_zones_valid',obs.get('geometry_valid',False)))
        if not obs.get("people_sampled", bool(obs.get("people"))):
            continue
        active_segment['people_sampled_frames']+=1
        output["people_sampled_frames"] += 1
        if output["first_sample_time"] is None:
            output["first_sample_time"] = t
        output["last_sample_time"] = t
        current_people = _observed_people(obs, width, height, diagnostics)
        by_player = defaultdict(list)
        for tid, point in current_people.items():
            role, pid = bindings.label(tid, t)
            if role in EXCLUDED_ROLES:
                diagnostics["excluded_role_detections"] += 1
                continue
            cell = _cell(point, width, height, columns, rows)
            _add(output["detection_grid"], cell, 1)
            _add_evidence(output["evidence_grid"], cell, [t], detection=True)
            output["detection_count"] += 1
            if pid:
                by_player[pid].append((tid, cell))
            else:
                diagnostics["unknown_identity_detections"] += 1
        current_conflicts = {pid for pid, detections in by_player.items() if len(detections) > 1}
        for pid, detections in by_player.items():
            target = player(pid)
            if len(detections) > 1:
                target["conflicting_detection_samples"] += 1
                diagnostics["conflicting_detection_samples"] += 1
            else:
                target["detection_count"] += 1
                _add(target["detection_grid"], detections[0][1], 1)
                _add_evidence(target["evidence_grid"], detections[0][1], [t], detection=True)

        if previous_time is not None:
            delta = t - previous_time
            if 0 < delta <= gap + 1e-9:
                output["sampled_timeline_seconds"] += delta
                common_tracks = current_people.keys() & previous_people.keys()
                boundaries = bindings.boundaries(common_tracks, previous_time, t)
                for left, right in zip(boundaries, boundaries[1:]):
                    duration, midpoint = right - left, (left + right) / 2
                    active = {}
                    aliases = defaultdict(list)
                    for tid in common_tracks:
                        role, pid = bindings.label(tid, midpoint)
                        if role in EXCLUDED_ROLES:
                            diagnostics["excluded_role_track_seconds"] += duration
                            continue
                        active[tid] = pid
                        if pid:
                            aliases[pid].append(tid)
                    conflicted_players = current_conflicts | previous_conflicts
                    conflicts = {tid for pid, tids in aliases.items() if len(tids) > 1 or pid in conflicted_players for tid in tids}
                    for pid, tids in aliases.items():
                        if len(tids) > 1 or pid in conflicted_players:
                            player(pid)["conflicting_interval_seconds"] += duration
                            diagnostics["conflicting_intervals"] += 1
                            if len(output["conflicts_sample"]) < 50:
                                output["conflicts_sample"].append({"player_id": pid, "track_ids": sorted(tids),
                                    "start": left, "end": right, "reason": "одна личность одновременно привязана к нескольким траекториям"})
                    diagnostics["conflicting_track_seconds"] += len(conflicts) * duration
                    for tid, pid in active.items():
                        if tid in conflicts:
                            continue
                        first_cell = _cell(previous_people[tid], width, height, columns, rows)
                        last_cell = _cell(current_people[tid], width, height, columns, rows)
                        for cell in (first_cell, last_cell):
                            _add(output["presence_seconds_grid"], cell, duration / 2)
                        for cell in {first_cell, last_cell}:
                            _add_evidence(output["evidence_grid"], cell, [previous_time, t], interval=(left, right))
                        output["estimated_person_seconds"] = (output["estimated_person_seconds"] or 0) + duration
                        output["supported_track_intervals"] += 1
                        if pid:
                            target = player(pid)
                            for cell in (first_cell, last_cell):
                                _add(target["presence_seconds_grid"], cell, duration / 2)
                            for cell in {first_cell, last_cell}:
                                _add_evidence(target["evidence_grid"], cell, [previous_time, t], interval=(left, right))
                            target["estimated_presence_seconds"] = (target["estimated_presence_seconds"] or 0) + duration
                            target["supported_intervals"] += 1
                        else:
                            diagnostics["unknown_identity_person_seconds"] += duration
            elif delta > gap:
                diagnostics["unsupported_sample_gap_seconds"] += delta
        # Empty or lost samples intentionally erase the previous track set.
        previous_time, previous_people, previous_conflicts = t, current_people, current_conflicts

    denominator = output["sampled_timeline_seconds"]
    for row in player_rows.values():
        row["estimate"] = {"numerator_seconds": row["estimated_presence_seconds"],
                           "denominator_seconds": denominator if denominator else None,
                           "sample_size": row["supported_intervals"], "missing": None,
                           "completeness": None,
                           "label": "Оценка времени по соседним детекциям; знаменатель — доступная выборочная временная шкала, не весь матч"}
    output["players"] = sorted(player_rows.values(), key=lambda x: x["player_id"])
    if output['view_segment_count']>1:
        output['limitations'].append('Карта объединяет пиксели нескольких сегментов/ракурсов одного исходника: одна ячейка не означает один участок корта. Интервалы через границы сегментов исключены.')
    if roi_unreliable or output['view_segment_count']>1:
        output['people_search_roi']=None;output['obstacles']=[]
        output['limitations'].append('Единая действующая область поиска не указана: совмещение не подтверждено во всех кадрах или менялся сегмент. Сохранённые настройки областей не доказывают наблюдаемость.')
    return output


def presence_report(project, run_ids=None, grid=(16, 9), max_gap=None, source_ids=None):
    """Return source-separated grids and player estimates from SQLite streaming.

    Default: the most recently stored job for each source. Explicit run IDs keep
    independent runs separate; neither grids nor player times are merged across
    runs or files. Mixed-view pixel grids carry explicit segment warnings and
    never assert that a cell denotes the same floor region throughout a source.
    """
    if len(grid) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or not 1 <= x <= 256 for x in grid):
        raise ValueError("Размер карты должен содержать два целых числа от 1 до 256")
    jobs = project.all("job")
    if run_ids is None:
        latest = {}
        for job in jobs:
            latest[job["source_id"]] = job
        selected = list(latest.values())
    else:
        requested = {run_ids} if isinstance(run_ids, str) else set(run_ids)
        selected = [job for job in jobs if job["id"] in requested]
        if requested - {job["id"] for job in selected}:
            raise ValueError("Задание для карты наблюдений не найдено")
    if source_ids is not None:
        included = {source_ids} if isinstance(source_ids, str) else set(source_ids)
        selected = [job for job in selected if job["source_id"] in included]
    sources = []
    for job in selected:
        source = project.get("source", job["source_id"])
        if source:
            sources.append(_one_source(project, job, source, *grid, max_gap))
    return {"schema_version": 1, "coordinate_space": "rotated_original_pixels",
            "sources": sources, "players": [p for source in sources for p in source["players"]],
            "completeness": None, "real_video_quality": None, "limitations": list(LIMITATIONS)}
