"""Optional temporal action learning. Importing this module does not import torch.

No volleyball weights are shipped. Window classification is not event quality:
automatic acceptance needs a separately evaluated event-level policy on reviewed
real held-out sequences. Actor, outcome and geometry are always independent.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import bisect
import heapq
from collections import Counter, deque
from pathlib import Path

import numpy as np

from .domain import EVENT_TYPES, event

SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "dt_s", "ball_visible", "ball_x", "ball_y", "ball_w", "ball_h",
    "ball_vx", "ball_vy", "ball_velocity_valid", "people_sampled", "person_visible",
    "person_x", "person_y", "person_w", "person_h", "ball_person_distance",
    "ball_candidates_scaled", "geometry_valid",
)
ACTION_CLASSES = ("none", "serve", "receive", "set", "attack", "tip", "block_attempt",
                  "block_touch", "dig", "cover", "freeball", "rally_start", "rally_end")
DEFAULT_RULES = {"score_threshold": 0.9, "min_margin": 0.2, "min_ball_fraction": 0.4,
                 "min_ball_observations": 3, "max_gap_s": 0.5, "nms_radius_s": 0.6,
                 "tolerance_s": 0.25, "min_accepted": 30, "max_error_upper95": 0.1,"min_event_recall":.5}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _box(d, width, height):
    x1, y1, x2, y2 = np.asarray(d["bbox"], dtype=float)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("Вырожденная рамка в признаках временной модели")
    return [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]


class TrackBindingIndex:
    """Snapshot current gallery bindings with O(log N) per-track interval lookup.

    Project bindings are ordered by insertion/replacement row order. Later rows
    win on an overlap, exactly as Project.track_identity; boundaries are inclusive.
    No SQL is issued while individual detections are processed.
    """
    def __init__(self, bindings, run_id):
        self.snapshot = copy.deepcopy([b for b in bindings if b.get("run_id") == run_id])
        self.signature = hashlib.sha256(_canonical(self.snapshot)).hexdigest()
        tracks = {}
        for priority, binding in enumerate(self.snapshot):
            start, end, track_id = binding.get("start"), binding.get("end"), binding.get("track_id")
            if (not isinstance(track_id, str) or not track_id or
                not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or
                not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start):
                raise ValueError("Некорректный интервал привязки траектории в галерее")
            tracks.setdefault(track_id, []).append((float(start), float(np.nextafter(float(end), np.inf)), priority, binding))
        self.index = {}
        for track_id, intervals in tracks.items():
            starts = {}
            boundaries = set()
            for start, end, priority, binding in intervals:
                starts.setdefault(start, []).append((-priority, end, binding))
                boundaries.update((start, end))
            times, values, active = [], [], []
            previous = object()
            for t in sorted(boundaries):
                for entry in starts.get(t, []):
                    heapq.heappush(active, entry)
                while active and active[0][1] <= t:
                    heapq.heappop(active)
                current = active[0][2] if active else None
                if current is not previous:
                    times.append(t)
                    values.append(current)
                    previous = current
            self.index[track_id] = (times, values)

    def at(self, track_id, time):
        pair = self.index.get(track_id)
        if pair is None:
            return None
        times, values = pair
        i = bisect.bisect_right(times, time)-1
        return values[i] if i >= 0 else None

    def overlay(self, observations):
        """Copy observations and overlay roles without altering stored detections."""
        for observation in observations:
            copied = dict(observation)
            people = []
            for detection in observation.get("people", []):
                person = dict(detection)
                binding = self.at(person.get("track_id"), observation["time"])
                if binding is not None:
                    person["role"] = binding.get("role", "unknown")
                    person["player_id"] = binding.get("player_id")
                    person["track_binding_id"] = binding.get("id")
                people.append(person)
            copied["people"] = people
            yield copied


def _features_with_context(observations, width, height, max_motion_gap=0.2):
    """Yield (original time, feature row, original PTS), without interpolating gaps.

    Multiple ball candidates mean no known ball. Nearest visible person supplies
    contextual geometry, never a proven actor identity. Only observed detections
    are used; absent people samples are distinguished from empty observations.
    """
    if width <= 0 or height <= 0 or max_motion_gap <= 0:
        raise ValueError("Нужны положительные размеры кадра и допуск пропуска")
    previous_time, previous_ball, previous_scope, previous_track = None, None, None, None
    for obs in observations:
        t = float(obs["time"])
        if not math.isfinite(t) or t < 0 or (previous_time is not None and t <= previous_time):
            raise ValueError("Нужны возрастающие фактические временные метки")
        scope=tuple(obs.get(k) for k in ('calibration_id','camera_session_id','tracking_segment'))
        boundary=previous_time is not None and scope!=previous_scope
        dt = 0 if previous_time is None or boundary else t-previous_time
        balls = [d for d in obs.get("balls", []) if d.get("observed", True)]
        ball = _box(balls[0], width, height) if len(balls) == 1 else None
        track=balls[0].get('track_id') if len(balls)==1 else None
        changed_track=track is not None and previous_track is not None and track!=previous_track
        velocity = [0., 0.]
        velocity_valid = ball is not None and previous_ball is not None and not changed_track and 0 < dt <= max_motion_gap
        if velocity_valid:
            velocity = [(ball[0]-previous_ball[0])/dt, (ball[1]-previous_ball[1])/dt]
        people = [d for d in obs.get("people", []) if d.get("observed", True) and d.get("role") not in ("referee", "outside", "bench", "spectator")]
        person, distance = None, 0.
        if ball is not None and people:
            choices = [_box(d, width, height) for d in people]
            person = min(choices, key=lambda p: math.hypot(ball[0]-p[0], ball[1]-p[1]))
            distance = math.hypot(ball[0]-person[0], ball[1]-person[1])
        row = [dt, float(ball is not None), *(ball or [0.]*4), *velocity, float(velocity_valid),
               float(bool(obs.get("people_sampled", bool(people)))), float(person is not None),
               *(person or [0.]*4), distance, min(len(balls), 5)/5, float(bool(obs.get("geometry_valid")))]
        yield t, row, obs.get("source_pts"), scope
        previous_time, previous_ball, previous_scope, previous_track = t, ball, scope, track


def features_from_observations(observations,width,height,max_motion_gap=.2):
    for t,row,pts,scope in _features_with_context(observations,width,height,max_motion_gap):
        yield t,row,pts


def observation_windows(observations, width, height, window_seconds=2., stride_seconds=.25, max_steps=256, after_anchor=None):
    """Bounded streaming windows; anchor is an actually observed middle timestamp."""
    if window_seconds <= 0 or stride_seconds <= 0 or max_steps < 3:
        raise ValueError("Некорректный размер или шаг временного окна")
    buffer = deque(maxlen=max_steps)
    last_anchor = -math.inf if after_anchor is None else after_anchor
    previous_scope=None
    for t, row, pts, scope in _features_with_context(observations, width, height):
        if scope!=previous_scope:buffer.clear()
        previous_scope=scope
        buffer.append((t, row, pts))
        while buffer and buffer[0][0] < t-window_seconds:
            buffer.popleft()
        if len(buffer) < 3 or t-buffer[0][0] < .8*window_seconds:
            continue
        middle = buffer[len(buffer)//2]
        if middle[0]-last_anchor < stride_seconds-1e-9:
            continue
        last_anchor = middle[0]
        yield {"times": [x[0] for x in buffer], "features": [x[1] for x in buffer],
               "anchor": middle[0], "source_pts": middle[2], "start": buffer[0][0], "end": t}


def _validate_window(window, require_label=True):
    x = np.asarray(window.get("features"), dtype=np.float32)
    ts = np.asarray(window.get("times"), dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not 3 <= len(x) <= 256:
        raise ValueError("Окно должно содержать 3–256 строк согласованного набора признаков")
    if ts.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0) or ts[0] < 0:
        raise ValueError("Некорректные признаки или временные метки окна")
    if not np.isclose(float(window.get("anchor", ts[len(ts)//2])), ts[len(ts)//2], atol=1e-6):
        raise ValueError("Метка действия должна относиться к фактическому среднему кадру окна")
    for index in (1, 8, 9, 10, 17):
        if not np.isin(x[:, index], [0., 1.]).all():
            raise ValueError("Маски видимости должны быть 0 или 1")
    if np.any(x[:, 0] < 0):
        raise ValueError("Интервалы dt не могут быть отрицательными")
    if require_label:
        if window.get("reviewed") is not True:
            raise ValueError("Для обучения/оценки нужна проверенная разметка каждого окна")
        if window.get("label") not in ACTION_CLASSES:
            raise ValueError("Неподдерживаемая временная метка; исход очка не выводится этим модулем")
        for field in ("id", "match_id", "source_id"):
            if not isinstance(window.get(field), str) or not window[field]:
                raise ValueError(f"У каждого окна требуется {field}")
    return x, ts


def load_dataset(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION or tuple(data.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("Несовместимая схема признаков временной разметки")
    if data.get("data_kind") not in ("real", "synthetic"):
        raise ValueError("Укажите data_kind=real или synthetic; неизвестное происхождение не скрывать")
    for key, default in (("window_seconds", 2.), ("stride_seconds", .25)):
        value = data.setdefault(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Некорректная длительность/шаг окон в разметке")
    windows = data.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Нужен непустой набор временных окон")
    ids = set()
    for window in windows:
        _, times = _validate_window(window)
        if not .8*data["window_seconds"] <= times[-1]-times[0] <= 1.05*data["window_seconds"]:
            raise ValueError("Длительность окна не соответствует window_seconds набора")
        if window["id"] in ids:
            raise ValueError("Повторяющийся ID временного окна")
        ids.add(window["id"])
    data["fingerprint"] = _sha(path)
    return data


def assert_disjoint(*datasets):
    """Reject leakage by match, source ID and source checksum when available."""
    for i, a in enumerate(datasets):
        for b in datasets[i+1:]:
            for field in ("match_id", "source_id", "source_sha256"):
                aa = {w[field] for w in a["windows"] if w.get(field)}
                bb = {w[field] for w in b["windows"] if w.get(field)}
                if aa & bb:
                    raise ValueError(f"Утечка между наборами: пересекается {field}")


def make_model(num_classes, hidden_size=48, num_layers=1):
    import torch
    from torch import nn

    class TemporalGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(len(FEATURE_NAMES), hidden_size, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, num_classes)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return self.head(h[-1])

    return TemporalGRU()


def _batch(windows, mean, std, classes, device):
    import torch
    values = [torch.tensor((np.asarray(w["features"], np.float32)-mean)/std) for w in windows]
    lengths = torch.tensor([len(x) for x in values], dtype=torch.int64)
    x = torch.nn.utils.rnn.pad_sequence(values, batch_first=True).to(device)
    y = torch.tensor([classes.index(w["label"]) for w in windows], dtype=torch.int64, device=device)
    return x, lengths, y


def classification_metrics(truth, predicted, classes):
    result = {}
    for label in classes:
        tp = sum(a == b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        result[label] = {"tp": tp, "fp": fp, "fn": fn,
                         "precision": tp/(tp+fp) if tp+fp else None,
                         "recall": tp/(tp+fn) if tp+fn else None}
    return {"unit": "reviewed windows, not event detection", "classes": result,
            "sample_size": len(truth), "correct": sum(a == b for a, b in zip(truth, predicted))}


def validate_rules(rules=None):
    r = DEFAULT_RULES | (rules or {})
    if set(r) != set(DEFAULT_RULES):
        raise ValueError("Неизвестные параметры политики временной модели")
    for key in ("score_threshold", "min_margin", "min_ball_fraction", "max_error_upper95","min_event_recall"):
        if not isinstance(r[key], (float, int)) or not math.isfinite(r[key]) or not 0 <= r[key] <= 1:
            raise ValueError(f"{key}: требуется число 0–1")
    for key in ("min_ball_observations", "min_accepted"):
        if not isinstance(r[key], int) or isinstance(r[key], bool) or r[key] < 1:
            raise ValueError(f"{key}: требуется положительное целое")
    for key in ("max_gap_s", "nms_radius_s", "tolerance_s"):
        if not isinstance(r[key], (int, float)) or not math.isfinite(r[key]) or r[key] <= 0:
            raise ValueError(f"{key}: требуется положительное число")
    return r


def window_quality(window, rules=None):
    r = validate_rules(rules)
    x, ts = _validate_window(window, require_label=False)
    count = int(x[:, 1].sum())
    reasons = []
    if count < r["min_ball_observations"] or count/len(x) < r["min_ball_fraction"]:
        reasons.append("недостаточно наблюдений мяча")
    if float(np.diff(ts).max()) > r["max_gap_s"]:
        reasons.append("разрыв временной последовательности")
    return {"eligible": not reasons, "reasons": reasons, "ball_observations": count,
            "sample_size": len(x), "ball_fraction": count/len(x)}


def suppress_candidates(candidates, radius):
    """Deterministic temporal NMS across action types, grouped by source.

    This event proposal rule must be evaluated together with the model; its
    radius is part of the acceptance-policy checksum, not a hidden heuristic.
    """
    kept = []
    for item in sorted(candidates, key=lambda p: (-p["raw_score"], p["source_id"], p["start"], p["type"])):
        if not any(p["source_id"] == item["source_id"] and abs(p["start"]-item["start"]) <= radius for p in kept):
            kept.append(item)
    return sorted(kept, key=lambda p: (p["source_id"], p["start"]))


def _error_upper95(errors, n):
    if not n:
        return None
    z = 1.959963984540054
    p = errors/n
    return (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def _reference_coverage(dataset):
    """Require explicit fully reviewed intervals with dense evaluation windows."""
    intervals = dataset.get("annotation_intervals", [])
    if not intervals:
        return False, "не указаны полностью размеченные интервалы оценки", []
    stride = dataset.get("stride_seconds", .25)
    for interval in intervals:
        if (interval.get("reviewed") is not True or not isinstance(interval.get("source_id"), str) or
            not isinstance(interval.get("start"), (int, float)) or not isinstance(interval.get("end"), (int, float)) or
            not math.isfinite(interval["start"]) or not math.isfinite(interval["end"]) or
            interval["start"] < 0 or interval["end"] <= interval["start"] or
            not interval.get("types") or any(c not in ACTION_CLASSES[1:] for c in interval["types"])):
            return False, "некорректный или непроверенный интервал эталона", []
        anchors = sorted(float(w.get("anchor", w.get("times", [math.nan])[len(w.get("times", [0]))//2]))
                         for w in dataset["windows"] if w["source_id"] == interval["source_id"])
        anchors = [t for t in anchors if interval["start"] <= t <= interval["end"]]
        if (not anchors or anchors[0]-interval["start"] > 1.75*stride or
            interval["end"]-anchors[-1] > 1.75*stride or
            (len(anchors) > 1 and max(np.diff(anchors)) > 1.75*stride)):
            return False, "в интервале эталона есть непокрытые временными окнами участки", []
    return True, None, intervals


def _inside_intervals(item, intervals):
    return any(i["source_id"] == item["source_id"] and i["start"] <= item["start"] <= i["end"] and
               item["type"] in i["types"] for i in intervals)


def evaluate_policy(candidates, dataset, model_sha256, rules=None, enable_automatic=False):
    """Event-level held-out policy; synthetic/window-only evaluation cannot enable it."""
    from .evaluation import events as evaluate_events
    r = validate_rules(rules)
    reasons = []
    if dataset.get("data_kind") != "real":
        reasons.append("политика не проверена на реальных данных")
    if dataset.get("coverage_verified") is not True:
        reasons.append("полнота эталона не подтверждена")
    coverage_ok, coverage_reason, intervals = _reference_coverage(dataset)
    if not coverage_ok:
        reasons.append(coverage_reason)
    truth = dataset.get("events", [])
    if not truth:
        reasons.append("нет событий эталона; точность окон не заменяет качество событий")
    source_ids = {w["source_id"] for w in dataset["windows"]}
    checked_truth = []
    for e in truth:
        if e.get("reviewed") is not True or e.get("type") not in ACTION_CLASSES[1:] or e.get("source_id") not in source_ids:
            raise ValueError("Для политики нужен проверенный эталон событий тех же исходников")
        if not isinstance(e.get("start"), (float, int)) or not math.isfinite(e["start"]) or e["start"] < 0:
            raise ValueError("Некорректное время эталонного события")
        if coverage_ok and not _inside_intervals(e, intervals):
            raise ValueError("Эталонное событие вне заявленных размеченных интервалов")
        checked_truth.append(e)
    predictions = suppress_candidates(candidates, r["nms_radius_s"])
    predictions=[c for c in predictions if c.get('proposal_eligible',True) and
                 (not coverage_ok or _inside_intervals(c,intervals))]
    measured = evaluate_events(checked_truth, predictions, r["tolerance_s"])
    classes, eligible = {}, []
    for label, values in measured["events"].items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        n = tp+fp
        upper = _error_upper95(fp, n)
        recall=tp/(tp+fn) if tp+fn else None
        passed = (n >= r["min_accepted"] and upper is not None and upper <= r["max_error_upper95"] and
                  recall is not None and recall>=r['min_event_recall'])
        classes[label] = {"tp": tp, "fp": fp, "fn": fn, "accepted_sample_size": n,
                          "error_rate": fp/n if n else None, "error_upper95": upper,
                          "event_recall":recall, "passed": passed}
        if passed and label in ACTION_CLASSES[1:]:
            eligible.append(label)
    enabled = bool(enable_automatic and not reasons and eligible)
    if not enable_automatic:
        reasons.append("автоматическое принятие явно не включено")
    return {"version": 3, "selection_version":3, "enabled": enabled, "accepted_labels": eligible if enabled else [],
            "evaluation_unit": "events", "model_sha256": model_sha256, "rules": r,
            "rules_sha256": hashlib.sha256(_canonical(r)).hexdigest(), "classes": classes,
            "data_kind": dataset.get("data_kind"), "dataset_sha256": dataset.get("fingerprint"),
            "match_ids": sorted({w["match_id"] for w in dataset["windows"]}),
            "source_ids": sorted(source_ids), "coverage_verified": dataset.get("coverage_verified") is True,
            "coverage_intervals_valid": coverage_ok, "annotation_intervals": intervals,
            "reasons": reasons, "warning": "scores не калиброванные вероятности; пороги не согласованы автоматически с пользователем"}


def _disabled_policy(model_sha256):
    r = validate_rules()
    return {"version": 1, "enabled": False, "accepted_labels": [], "evaluation_unit": None,
            "model_sha256": model_sha256, "rules": r, "rules_sha256": hashlib.sha256(_canonical(r)).hexdigest(),
            "reasons": ["нет отдельной проверки качества событий на реальных отложенных матчах"]}


def _policy_allows(policy, label, model_sha256):
    rules = validate_rules(policy.get("rules"))
    valid = (policy.get("enabled") is True and policy.get("evaluation_unit") == "events" and
             policy.get('selection_version')==3 and
             policy.get("data_kind") == "real" and policy.get("coverage_verified") is True and
             policy.get("coverage_intervals_valid") is True and
             policy.get("dataset_sha256") and policy.get("match_ids") and
             policy.get("model_sha256") == model_sha256 and
             policy.get("rules_sha256") == hashlib.sha256(_canonical(rules)).hexdigest() and
             label in policy.get("accepted_labels", []) and policy.get("classes", {}).get(label, {}).get("passed") is True)
    return bool(valid)


def publish_artifact(output, state_dict, model_config, classes, normalization, provenance, policy=None):
    """Write a content-addressed immutable version; never overwrite an active model."""
    import torch
    if "none" not in classes or len(classes) < 2 or len(classes) != len(set(classes)) or any(c not in ACTION_CLASSES for c in classes):
        raise ValueError("Нужен непустой уникальный словарь, включая none и поддерживаемое действие")
    blob = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, blob)
    payload = blob.getvalue()
    weight_hash = hashlib.sha256(payload).hexdigest()
    body = {"schema_version": SCHEMA_VERSION, "architecture": "gru-1", "feature_names": list(FEATURE_NAMES),
            "model_config": model_config, "classes": list(classes), "normalization": normalization,
            "weights": "model.pt", "weights_sha256": weight_hash, "provenance": provenance,
            "policy": policy or _disabled_policy(weight_hash)}
    if body["policy"].get("model_sha256") != weight_hash:
        raise ValueError("Политика не относится к публикуемым весам")
    manifest_hash = hashlib.sha256(_canonical(body)).hexdigest()
    folder = Path(output)/manifest_hash[:24]
    folder.mkdir(parents=True, exist_ok=True)
    weight_path, manifest_path = folder/"model.pt", folder/"manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != body | {"manifest_sha256": manifest_hash} or _sha(weight_path) != weight_hash:
            raise ValueError("Неизменяемая версия модели уже существует с другим содержимым")
        return manifest_path
    if weight_path.exists() and _sha(weight_path) != weight_hash:
        raise ValueError("В каталоге версии уже есть другие веса")
    partial_weights = folder/"model.partial"
    partial_weights.write_bytes(payload)
    partial_weights.replace(weight_path)
    partial_manifest = folder/"manifest.partial"
    partial_manifest.write_text(json.dumps(body | {"manifest_sha256": manifest_hash}, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_manifest.replace(manifest_path)
    return manifest_path


class TemporalPredictor:
    def __init__(self, artifact, device="cpu", policy=None):
        import torch
        artifact = Path(artifact)
        path = artifact/"manifest.json" if artifact.is_dir() else artifact
        body = json.loads(path.read_text(encoding="utf-8"))
        digest = body.pop("manifest_sha256", None)
        if digest != hashlib.sha256(_canonical(body)).hexdigest():
            raise ValueError("Контрольная сумма manifest временной модели не совпала")
        if body.get("schema_version") != SCHEMA_VERSION or body.get("architecture") != "gru-1" or tuple(body.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("Несовместимая архитектура или признаки временной модели")
        classes = body.get("classes", [])
        if len(classes) < 2 or "none" not in classes or len(classes) != len(set(classes)) or any(c not in ACTION_CLASSES for c in classes):
            raise ValueError("Некорректный словарь временных действий")
        weights = path.parent/body["weights"]
        if weights.parent.resolve() != path.parent.resolve() or _sha(weights) != body["weights_sha256"]:
            raise ValueError("Весов нет или их контрольная сумма не совпала")
        self.manifest, self.manifest_path, self.classes = body, path, classes
        self.version = body["weights_sha256"]
        from .models import choose_device
        self.device, self.warning = choose_device(device)
        self.model = make_model(len(classes), **body["model_config"])
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.mean = np.asarray(body["normalization"]["mean"], np.float32)
        self.std = np.asarray(body["normalization"]["std"], np.float32)
        if self.mean.shape != (len(FEATURE_NAMES),) or self.std.shape != self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Некорректная нормализация временной модели")
        self.policy = copy.deepcopy(policy if policy is not None else body["policy"])
        self.rules = validate_rules(self.policy.get("rules"))
        # A changed rules checksum does not raise away the whole report: it disables acceptance.

    def predict(self, window):
        import torch
        quality = window_quality(window, self.rules)
        x, times = _validate_window(window, require_label=False)
        duration = self.manifest["provenance"].get("window_seconds", 2.)
        if not .8*duration <= times[-1]-times[0] <= 1.05*duration:
            quality["eligible"] = False
            quality["reasons"].append("длительность окна не соответствует оценённой модели")
        with torch.inference_mode():
            tensor = torch.tensor((x-self.mean)/self.std, device=self.device)[None]
            logits = self.model(tensor, torch.tensor([len(x)]))
            scores = torch.softmax(logits[0], dim=0).cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Временная модель вернула неконечные оценки")
        order = np.argsort(scores)[::-1]
        label = self.classes[int(order[0])]
        score = float(scores[order[0]])
        margin = score-float(scores[order[1]])
        eligible = quality["eligible"] and score >= self.rules["score_threshold"] and margin >= self.rules["min_margin"] and label != "none"
        accepted = eligible and _policy_allows(self.policy, label, self.version)
        return {"label": label, "raw_score": score, "margin": margin,
                "raw_scores": {c: float(s) for c, s in zip(self.classes, scores)},
                "decision": "accepted" if accepted else "hypothesis" if label != "none" and quality["eligible"] else "abstain",
                "proposal_eligible": bool(eligible), "quality": quality,
                "reason": "проверенная политика событий" if accepted else "тип не принят автоматически; см. политику/видимость",
                "score_meaning": "raw softmax score, not calibrated probability", "model_sha256": self.version}

    def predict_event(self, source_id, window):
        prediction = self.predict(window)
        if prediction["decision"] == "abstain":
            return None, prediction
        anchor = float(window.get("anchor", window["times"][len(window["times"])//2]))
        e = event(source_id, anchor, prediction["label"], review="unreviewed", origin="automatic",
                  model_version=self.version, source_pts=window.get("source_pts"), visibility="partial",
                  raw_scores=prediction["raw_scores"], context={"temporal_window_start": window["times"][0],
                      "temporal_window_end": window["times"][-1], "temporal_manifest_sha256": json.loads(self.manifest_path.read_text(encoding='utf-8'))["manifest_sha256"],
                      "temporal_policy": {"enabled": self.policy.get("enabled"), "rules_sha256": self.policy.get("rules_sha256")},
                      "actor_not_inferred_from_nearest_person": True})
        e["evidence"]["type"] = prediction["decision"]
        e["unknown_reasons"].update(player_id="временная модель не определяет личность участника",
                    team_id="команда не определена независимым модулем", outcome="модель типа действия не определяет результат",
                    direction="направление на корте не определено", target_zone="пространственная оценка отсутствует")
        return e, prediction


def train(train_path, validation_path, output, epochs=10, resume=False, hidden_size=48,
          num_layers=1, batch_size=16, learning_rate=.001, device="cpu", seed=42,
          cancel=None, progress=None):
    """Train an actual GRU, checkpoint each epoch, publish hypotheses-only weights.

    Validation selects no automatic policy. A separate calibration dataset and
    event reference are required by calibrate_artifact(). Test data are not used.
    """
    import random
    import torch
    if epochs < 1 or batch_size < 1 or not 4 <= hidden_size <= 512 or not 1 <= num_layers <= 4 or learning_rate <= 0:
        raise ValueError("Некорректные параметры обучения временной модели")
    training, validation = load_dataset(train_path), load_dataset(validation_path)
    assert_disjoint(training, validation)
    if any(training[k] != validation[k] for k in ("window_seconds", "stride_seconds")):
        raise ValueError("Train/validation должны использовать одинаковую схему окон")
    classes = [c for c in ACTION_CLASSES if c in {w["label"] for w in training["windows"]}]
    if "none" not in classes or len(classes) < 2:
        raise ValueError("Нужны фоновые окна none и хотя бы один тип действия")
    if any(w["label"] not in classes for w in validation["windows"]):
        raise ValueError("В validation есть класс, отсутствующий в train")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    from .models import choose_device
    selected_device, warning = choose_device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = {"train": training["fingerprint"], "validation": validation["fingerprint"]}
    all_values = np.concatenate([np.asarray(w["features"], np.float32) for w in training["windows"]])
    mean = all_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = all_values.std(axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Признаки дают неконечную нормализацию; проверьте диапазоны данных")
    std[std < .001] = 1.
    config = {"hidden_size": hidden_size, "num_layers": num_layers}
    model = make_model(len(classes), **config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_epoch = 0
    checkpoint_path = output/"checkpoint.pt"
    training_config = {"model": config, "batch_size": batch_size, "learning_rate": learning_rate,
                       "classes": classes, "seed": seed}
    if resume:
        # This checkpoint is created locally by this training run; artifact inference
        # loads weights_only=True and never unpickles arbitrary training state.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["fingerprints"] != fingerprint or checkpoint["training_config"] != training_config:
            raise ValueError("Входы/параметры изменились: создайте новый эксперимент вместо resume")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        if selected_device == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch = checkpoint["epoch"]+1
    provenance = {"training_config": training_config, "fingerprints": fingerprint,
                  "train_match_ids": sorted({w["match_id"] for w in training["windows"]}),
                  "validation_match_ids": sorted({w["match_id"] for w in validation["windows"]}),
                  "train_source_ids": sorted({w["source_id"] for w in training["windows"]}),
                  "validation_source_ids": sorted({w["source_id"] for w in validation["windows"]}),
                  "training_source_sha256": sorted({w["source_sha256"] for w in training["windows"]+validation["windows"] if w.get("source_sha256")}),
                  "data_kind": "real" if training["data_kind"] == validation["data_kind"] == "real" else "synthetic",
                  "device": selected_device, "device_warning": warning, "torch_version": torch.__version__,
                  "anchor_rule": "actual middle observation timestamp", "window_seconds": training["window_seconds"], "stride_seconds": training["stride_seconds"],
                  "statement": "training/validation windows are not proof of event-level volleyball accuracy"}
    for epoch in range(start_epoch, epochs):
        model.train()
        indices = list(range(len(training["windows"])))
        random.shuffle(indices)
        loss_sum = 0.
        for offset in range(0, len(indices), batch_size):
            if cancel and cancel():
                from .media import Cancelled
                raise Cancelled("Обучение остановлено; последняя завершённая эпоха сохранена")
            windows = [training["windows"][i] for i in indices[offset:offset+batch_size]]
            x, lengths, target = _batch(windows, mean, std, classes, selected_device)
            loss = torch.nn.functional.cross_entropy(model(x, lengths), target)
            if not torch.isfinite(loss):
                raise ValueError("Неконечная функция потерь временной модели")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach())*len(windows)
        model.eval()
        predicted = []
        with torch.inference_mode():
            for offset in range(0, len(validation["windows"]), batch_size):
                batch = validation["windows"][offset:offset+batch_size]
                x, lengths, _ = _batch(batch, mean, std, classes, selected_device)
                predicted.extend(classes[i] for i in model(x, lengths).argmax(dim=1).cpu().tolist())
        metrics = classification_metrics([w["label"] for w in validation["windows"]], predicted, classes)
        record = {"epoch": epoch, "train_loss_mean": loss_sum/len(indices), "validation": metrics,
                  "data_kind": provenance["data_kind"], "automatic_acceptance": False}
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "fingerprints": fingerprint, "training_config": training_config,
                      "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                      "numpy_rng": np.random.get_state(), "cuda_rng": torch.cuda.get_rng_state_all() if selected_device == "cuda" else None}
        temporary = checkpoint_path.with_suffix(".partial")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        (output/f"epoch-{epoch:04d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress((epoch+1)/epochs, record)
    provenance["last_epoch"] = max(start_epoch-1, epochs-1)
    manifest = publish_artifact(output/"versions", model.state_dict(), config, classes,
                                {"mean": mean.tolist(), "std": std.tolist()}, provenance)
    return {"manifest": str(manifest), "automatic_acceptance": False,
            "data_kind": provenance["data_kind"], "last_epoch": provenance["last_epoch"], "device": selected_device}


def _dataset_candidates(predictor, dataset):
    candidates = []
    for window in dataset["windows"]:
        prediction = predictor.predict(window)
        if prediction["decision"] != 'abstain':
            candidates.append({"source_id": window["source_id"], "type": prediction["label"],
                               "start": float(window["times"][len(window["times"])//2]),
                               "raw_score": prediction["raw_score"], "window_id": window["id"],
                               "proposal_eligible":prediction['proposal_eligible']})
    return candidates


def calibrate_artifact(artifact, dataset_path, output, rules=None, enable_automatic=False, device="cpu"):
    """Publish a new immutable policy version, bound to the exact weight hash."""
    predictor = TemporalPredictor(artifact, device=device)
    dataset = load_dataset(dataset_path)
    provenance = predictor.manifest["provenance"]
    forbidden_matches = set(provenance.get("train_match_ids", [])) | set(provenance.get("validation_match_ids", []))
    forbidden_sources = set(provenance.get("train_source_ids", [])) | set(provenance.get("validation_source_ids", []))
    if forbidden_matches & {w["match_id"] for w in dataset["windows"]} or forbidden_sources & {w["source_id"] for w in dataset["windows"]}:
        raise ValueError("Политику нельзя проверять на train/validation матчах или исходниках")
    if any(dataset[k] != provenance.get(k) for k in ("window_seconds", "stride_seconds")):
        raise ValueError("Для политики нужна та же схема окон, что при обучении")
    if set(provenance.get("training_source_sha256", [])) & {w["source_sha256"] for w in dataset["windows"] if w.get("source_sha256")}:
        raise ValueError("Политика пересекает train/validation по SHA исходного видео")
    predictor.rules = validate_rules(rules)
    candidates = _dataset_candidates(predictor, dataset)
    policy = evaluate_policy(candidates, dataset, predictor.version, predictor.rules, enable_automatic)
    updated = copy.deepcopy(provenance)
    updated["policy_dataset_sha256"] = dataset["fingerprint"]
    manifest = publish_artifact(output, predictor.model.state_dict(), predictor.manifest["model_config"], predictor.classes,
                                predictor.manifest["normalization"], updated, policy)
    return {"manifest": str(manifest), "policy": policy}


def infer_project(project, run_id, artifact_dir, policy=None, window_seconds=None, stride_seconds=None,
                  progress=None, cancel=None):
    """Offline second pass with durable window checkpoints and global temporal NMS.

    Cache keys bind immutable weights/manifest/policy/window contract and upstream
    progress. Candidate and NMS tables are optional-module caches; no images or
    full observation sequence are retained in memory. A restarted pass resumes
    the GRU scan; cheap NMS can restart over the complete persisted candidate set.
    """
    from .media import Cancelled
    from .store import dumps
    predictor = TemporalPredictor(artifact_dir, device="cpu", policy=policy)
    contract = predictor.manifest["provenance"]
    window_seconds = contract.get("window_seconds", 2.) if window_seconds is None else window_seconds
    stride_seconds = contract.get("stride_seconds", .25) if stride_seconds is None else stride_seconds
    if window_seconds != contract.get("window_seconds", 2.) or stride_seconds != contract.get("stride_seconds", .25):
        predictor.policy["enabled"] = False
        predictor.policy.setdefault("reasons", []).append("изменена оценённая схема окон")
    job = project.get("job", run_id)
    if not job:
        raise ValueError("Не найдено задание детекции для временного прохода")
    source = project.get("source", job["source_id"])
    if not source:
        raise ValueError("Не найден исходник временного прохода")
    manifest_sha = json.loads(predictor.manifest_path.read_text(encoding='utf-8'))["manifest_sha256"]
    bindings = TrackBindingIndex(project.all("track_binding"), run_id)
    observation_progress = project.db.execute(
        "SELECT COUNT(*),MIN(time),MAX(time) FROM observations WHERE run_id=?", (run_id,)).fetchone()
    inputs = {"run_id": run_id, "model_sha256": predictor.version, "manifest_sha256": manifest_sha,
              "policy": predictor.policy, "window_seconds": window_seconds, "stride_seconds": stride_seconds,
              "upstream": {k: job.get(k) for k in ("samples", "last_time", "state")},
              "observations": list(observation_progress), "source_sha256": source.get("sha256"),
              "dimensions": [source["display_width"], source["display_height"]],
              "track_bindings_sha256": bindings.signature, "track_bindings": bindings.snapshot, "temporal_code": 4}
    pass_id = "temporal:"+hashlib.sha256(_canonical(inputs)).hexdigest()[:32]
    checkpoint = project.get("temporal_job", pass_id)
    if checkpoint and checkpoint.get("state") == "complete":
        active=project.get('active_temporal',source['id'])
        if not active or active.get('pass_id')!=pass_id:
            _activate_temporal(project,pass_id,source['id'],cancel)
        return checkpoint | {"cache_reused": True}
    project.db.executescript("""
        CREATE TABLE IF NOT EXISTS temporal_candidates(
            pass_id TEXT,source_id TEXT,anchor REAL,score REAL,kind TEXT,body TEXT,
            PRIMARY KEY(pass_id,anchor));
        CREATE INDEX IF NOT EXISTS temporal_candidates_priority
            ON temporal_candidates(pass_id,score DESC,source_id,anchor,kind);
        CREATE TABLE IF NOT EXISTS temporal_selected(
            pass_id TEXT,source_id TEXT,anchor REAL,event_id TEXT,
            PRIMARY KEY(pass_id,anchor));
        CREATE INDEX IF NOT EXISTS temporal_selected_time
            ON temporal_selected(pass_id,source_id,anchor);
    """)
    checkpoint = checkpoint or {"id": pass_id, "run_id": run_id, "source_id": source["id"],
        "model_sha256": predictor.version, "manifest_sha256": manifest_sha, "inputs": inputs,
        "windows": 0, "last_anchor": None, "last_window_end": None, "abstentions": {},
        "classification_complete": False, "cache_reused": False}
    checkpoint["state"] = "running"
    project.put("temporal_job", checkpoint)
    counts = Counter(checkpoint["abstentions"])
    try:
        if not checkpoint["classification_complete"]:
            # Reconstruct enough original context; the saved stride anchor keeps
            # resumed window boundaries identical to uninterrupted processing.
            context_start = max(0., (checkpoint["last_window_end"] or 0.)-window_seconds-2*stride_seconds)
            observations = bindings.overlay(project.observations(run_id, start=context_start))
            for window in observation_windows(observations, source["display_width"], source["display_height"],
                    window_seconds, stride_seconds, after_anchor=checkpoint["last_anchor"]):
                if cancel and cancel():
                    raise Cancelled("Временной проход остановлен; контрольная точка окна сохранена")
                e, prediction = predictor.predict_event(source["id"], window)
                if e is None:
                    for reason in prediction["quality"]["reasons"] or ["фоновое окно"]:
                        counts[reason] += 1
                checkpoint.update(windows=checkpoint["windows"]+1, last_anchor=window["anchor"],
                                  last_window_end=window["end"], abstentions=dict(counts))
                with project.db:
                    if e is not None:
                        project.db.execute("INSERT OR REPLACE INTO temporal_candidates VALUES(?,?,?,?,?,?)",
                            (pass_id, source["id"], e["start"], prediction["raw_score"], e["type"], dumps(e)))
                    project.db.execute("INSERT OR REPLACE INTO objects VALUES ('temporal_job',?,?)",
                                       (pass_id, dumps(checkpoint)))
                if progress:
                    progress(min(1., window["end"]/max(source["duration"], .001)), "Временные признаки: типы действий/отказы")
            checkpoint["classification_complete"] = True
            project.put("temporal_job", checkpoint)
        # Disk-backed global NMS sees both sides of every resumed/chunk boundary.
        # Restarting this inexpensive phase cannot change its deterministic result.
        with project.db:
            project.db.execute("DELETE FROM temporal_selected WHERE pass_id=?", (pass_id,))
        priority = project.db.execute("SELECT source_id,anchor FROM temporal_candidates WHERE pass_id=? ORDER BY score DESC,source_id,anchor,kind", (pass_id,))
        for sid, anchor in priority:
            if cancel and cancel():
                raise Cancelled("Временной проход остановлен; предложения сохранены перед NMS")
            existing = project.db.execute("SELECT 1 FROM temporal_selected WHERE pass_id=? AND source_id=? AND anchor>=? AND anchor<=? LIMIT 1",
                (pass_id, sid, anchor-predictor.rules["nms_radius_s"], anchor+predictor.rules["nms_radius_s"])).fetchone()
            if not existing:
                with project.db:
                    project.db.execute("INSERT INTO temporal_selected VALUES(?,?,?,NULL)", (pass_id, sid, anchor))
        ids,accepted=_activate_temporal(project,pass_id,source['id'],cancel)
        checkpoint.update(state="complete", proposals=len(ids), accepted=int(accepted), hypotheses=len(ids)-int(accepted),
            event_ids=ids, policy_enabled=predictor.policy.get("enabled", False),
            data_kind=predictor.manifest["provenance"].get("data_kind"),
            limitations="Тип не определяет участника/команду/исход. Качество окон не равно качеству событий. Без реального эталона готовность не заявлена.")
        project.put("temporal_job", checkpoint)
        return checkpoint
    except BaseException as exc:
        checkpoint["state"] = "cancelled" if isinstance(exc, (Cancelled, KeyboardInterrupt)) else "failed"
        checkpoint["error"] = str(exc)
        project.put("temporal_job", checkpoint)
        raise


def _activate_temporal(project,pass_id,source_id,cancel=None):
    from .media import Cancelled
    from .store import dumps
    ids,accepted=[],0
    selected=project.db.execute('SELECT c.body,c.anchor FROM temporal_candidates c JOIN temporal_selected s '
        'ON c.pass_id=s.pass_id AND c.anchor=s.anchor WHERE c.pass_id=? ORDER BY c.anchor',(pass_id,))
    for body,anchor in selected:
        if cancel and cancel():raise Cancelled('Временной проход остановлен; выборка событий сохранена')
        e=json.loads(body);accepted+=e['evidence']['type']=='accepted'
        eid=project.propose(e,f"{source_id}:temporal:{e['start']:.6f}");ids.append(eid)
        with project.db:
            project.db.execute('UPDATE temporal_selected SET event_id=? WHERE pass_id=? AND anchor=?',(eid,pass_id,anchor))
    current_ids=set(ids)
    stale=project.db.execute('SELECT id,base,manual FROM events WHERE auto_key LIKE ?',(source_id+':temporal:%',)).fetchall()
    with project.db:
        for old in stale:
            if old['id'] in current_ids or json.loads(old['manual']):continue
            base=json.loads(old['base'])
            if base.get('review')=='rejected':continue
            project.db.execute('INSERT INTO model_revisions(event_id,body) VALUES(?,?)',(old['id'],old['base']))
            base['review']='rejected';base.setdefault('context',{})['superseded_by_temporal_pass']=pass_id
            project.db.execute('UPDATE events SET base=? WHERE id=?',(dumps(base),old['id']))
    project.put('active_temporal',{'id':source_id,'pass_id':pass_id})
    return ids,int(accepted)
