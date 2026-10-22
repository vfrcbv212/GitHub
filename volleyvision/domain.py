from __future__ import annotations

import copy
import math
import uuid

SCHEMA_VERSION = 2
EVENT_TYPES = {
    "rally_start": "Начало розыгрыша", "rally_end": "Конец розыгрыша",
    "serve": "Подача", "receive": "Приём подачи", "set": "Передача",
    "attack": "Атака", "tip": "Скидка", "block_attempt": "Попытка блока",
    "block_touch": "Касание блока", "block_point": "Результативный блок",
    "dig": "Защита", "cover": "Страховка", "freeball": "Свободный мяч",
    "error": "Ошибка", "point": "Присуждение очка", "substitution": "Замена",
    "side_change": "Смена сторон", "rotation_change": "Изменение расстановки",
    "candidate_contact": "Кандидат касания — действие неизвестно",
    "candidate_rally_start": "Кандидат начала — проверить",
    "candidate_rally_end": "Кандидат конца — исход неизвестен",
}
VALUE_FIELDS = (
    "type", "team_id", "player_id", "participants", "outcome", "source_zone",
    "target_zone", "direction", "quality", "target_player_id", "rotation",
    "lineup_id", "reception_quality", "blockers", "attack_type", "floor_point",
    "block_net_position", "initial_direction", "after_block_direction",
    "defender_first_contact", "landing_zone", "credited_player_id",
)
OUTCOMES = ("kill", "error", "blocked", "continue", "ace", "in_play", "saved", "softened")
STATUSES = ("confirmed", "accepted", "observed", "hypothesis", "unknown")


def uid() -> str:
    return uuid.uuid4().hex


def unknown(v):
    return v is None or v == "" or v == []


def event(source_id: str, start: float, kind: str, **fields) -> dict:
    result = {"id": uid(), "source_id": source_id, "start": float(start),
              "end": float(start), "type": kind, "review": "confirmed",
              "origin": "manual", "visibility": "unknown", "evidence": {},
              "context": {}, "unknown_reasons": {}}
    result.update(fields)
    for key in VALUE_FIELDS:
        value = result.get(key)
        result["evidence"].setdefault(key, "unknown" if unknown(value) else "confirmed")
        if unknown(value):
            result["unknown_reasons"].setdefault(key, "не установлено")
    validate_event(result)
    return result


def validate_event(e: dict):
    if e.get("type") not in EVENT_TYPES:
        raise ValueError("Неизвестный тип события")
    for k in ("start", "end"):
        if not isinstance(e.get(k), (int, float)) or not math.isfinite(e[k]) or e[k] < 0:
            raise ValueError("Время события должно быть конечным и неотрицательным")
    if e["end"] < e["start"]:
        raise ValueError("Конец события раньше начала")
    if e.get("review") not in ("confirmed", "unreviewed", "rejected"):
        raise ValueError("Неизвестный статус проверки")
    if any(v not in STATUSES for v in e.get("evidence", {}).values()):
        raise ValueError("Неизвестный статус достоверности поля")
    for k in ("source_zone", "target_zone", "landing_zone", "rotation"):
        if e.get(k) is not None and e[k] not in range(1, 7):
            raise ValueError(f"{k}: допустимы зоны/расстановки 1–6 или неизвестно")
    if e.get("direction") not in (None, "line", "diagonal", "short", "deep"):
        raise ValueError("Неизвестное направление")
    p = e.get("floor_point")
    if p is not None:
        if not e.get("context", {}).get("floor_contact_confirmed"):
            raise ValueError("Для точки пола требуется подтверждение контакта с полом")
        if len(p) != 2 or not (0 <= p[0] <= 18 and 0 <= p[1] <= 9):
            raise ValueError("Точка пола вне корта 18×9 м")


def usable(e: dict, mode="automatic") -> dict | None:
    if e.get("review") == "rejected":
        return None
    if mode == "confirmed" and e.get("review") != "confirmed":
        return None
    result = copy.deepcopy(e)
    for key in VALUE_FIELDS:
        state = e.get("evidence", {}).get(key, "unknown")
        allowed = {"confirmed"} if mode == "confirmed" else {"confirmed", "accepted"}
        if mode == "preliminary":
            allowed |= {"observed", "hypothesis"}
        if state not in allowed:
            result[key] = None
    if result.get("type") is None:
        return None
    return result
