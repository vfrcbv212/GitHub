from __future__ import annotations

from collections import Counter, defaultdict
import copy

from .domain import usable

MODES = {"automatic": "Автоматический анализ; события не проверены человеком",
         "confirmed": "Только события и поля, подтверждённые человеком",
         "preliminary": "ПРЕДВАРИТЕЛЬНО: включает автоматические предположения"}


def metric(numerator, denominator, ids=(), missing=0, recorded=None, label=""):
    return {"numerator": numerator, "denominator": denominator, "sample_size": denominator,
            "value": 100 * numerator / denominator if denominator else None,
            "missing": missing, "recorded": denominator + missing if recorded is None else recorded,
            "completeness": None, "episode_ids": list(ids), "label": label}


def value_allowed(status, mode):
    return status == "confirmed" or (mode != "confirmed" and status == "accepted") or (
        mode == "preliminary" and status in ("observed", "hypothesis"))


def select(project, mode="automatic", filters=None):
    filters = {k: v for k, v in (filters or {}).items() if v is not None and v != ""}
    result = []
    for raw in project.events():
        e = usable(raw, mode)
        if e is None:
            continue
        if raw.get("origin") == "automatic" and raw.get("context",{}).get("geometry_valid") is not True:
            for field in ("floor_point","source_zone","target_zone","direction","landing_zone","block_net_position"):
                if raw.get("evidence",{}).get(field) == "accepted":
                    e[field] = None
        if any(e.get(k) != v for k, v in filters.items()):
            continue
        result.append(e)
    return result


def action_stats(events, match):
    attacks = [e for e in events if e["type"] in ("attack", "tip")]
    assessed = [e for e in attacks if e.get("outcome") in ("kill", "error", "blocked", "continue")]
    kills = [e for e in assessed if e["outcome"] == "kill"]
    errors = [e for e in assessed if e["outcome"] in ("error", "blocked")]
    ids = [e["id"] for e in assessed]
    serves = [e for e in events if e["type"] == "serve"]
    evaluated_serves = [e for e in serves if e.get("outcome") in ("ace", "error", "in_play")]
    receives = [e for e in events if e["type"] == "receive"]
    scored_receives = [e for e in receives if e.get("quality") in match["quality_scale"]]
    positive = [e for e in scored_receives if e["quality"] in match["quality_positive"]]
    excellent = [e for e in scored_receives if e["quality"] in match["quality_excellent"]]
    by_type = Counter(e["type"] for e in events)
    out = {"observed_events": len(events), "event_ids": [e["id"] for e in events],
        "counts": dict(by_type), "attacks_recorded": len(attacks), "attacks_classified": len(assessed),
        "attack_kills": len(kills), "attack_errors_including_blocked": len(errors),
        "attack_continues": sum(e["outcome"] == "continue" for e in assessed),
        "attack_unknown_outcomes": len(attacks) - len(assessed),
        "kill_rate": metric(len(kills), len(assessed), ids, len(attacks) - len(assessed), label="K/оценённые атаки"),
        "attack_efficiency": metric(len(kills) - len(errors), len(assessed), ids,
                                    len(attacks) - len(assessed), label="(K−E)/оценённые атаки"),
        "serves_recorded": len(serves), "aces": sum(e.get("outcome") == "ace" for e in serves),
        "serve_errors": sum(e.get("outcome") == "error" for e in serves),
        "serve_in_play": sum(e.get("outcome") == "in_play" for e in serves),
        "serve_unknown": len(serves) - len(evaluated_serves),
        "ace_rate": metric(sum(e["outcome"] == "ace" for e in evaluated_serves), len(evaluated_serves),
                           [e["id"] for e in evaluated_serves], len(serves) - len(evaluated_serves)),
        "receptions_recorded": len(receives), "reception_errors": sum(e.get("outcome") == "error" for e in receives),
        "reception_quality": dict(Counter(e["quality"] for e in scored_receives)),
        "positive_reception": metric(len(positive), len(scored_receives), [e["id"] for e in scored_receives], len(receives) - len(scored_receives)),
        "excellent_reception": metric(len(excellent), len(scored_receives), [e["id"] for e in scored_receives], len(receives) - len(scored_receives)),
        "sets": by_type["set"], "block_attempts": by_type["block_attempt"], "block_touches": by_type["block_touch"],
        "block_point_participations": by_type["block_point"],
        "digs": by_type["dig"], "covers": by_type["cover"],
        "defence_saved": sum(e.get("outcome") == "saved" for e in events if e["type"] in ("dig", "cover")),
        "directions": {}, "distributions": {}}
    directional = [e for e in attacks if e.get("direction")]
    for direction in ("line", "diagonal", "short", "deep"):
        matches = [e for e in directional if e["direction"] == direction]
        out["directions"][direction] = metric(len(matches), len(directional), [e["id"] for e in matches],
                                               len(attacks) - len(directional), label="распределение среди известных направлений")
    # Counts and opportunity denominators are kept for every coaching comparison.
    for kind, field, context in (("attack", "direction", "source_zone"), ("attack", "direction", "reception_quality"),
                                ("attack", "outcome", "blockers"), ("set", "target_player_id", "reception_quality"),
                                ("serve", "target_zone", "outcome"), ("dig", "source_zone", "quality"),
                                ("block_touch", "block_net_position", "outcome")):
        eligible = [e for e in events if e["type"] == kind]
        groups = defaultdict(list)
        for e in eligible:
            if e.get(field) is not None and e.get(context) is not None:
                groups[str(e[context])].append(e)
        rows = []
        for group, items in groups.items():
            for v in sorted({str(e[field]) for e in items}):
                chosen = [e for e in items if str(e[field]) == v]
                rows.append({"context": group, "value": v, **metric(len(chosen), len(items), [e["id"] for e in chosen],
                            len(eligible) - sum(len(g) for g in groups.values()))})
        out["distributions"][f"{kind}/{context}/{field}"] = rows
    return out


def observation_maps(project,mode,filters):
    """Observation-time maps have a different denominator from action filters."""
    if mode=="confirmed" or any(value is not None for key,value in filters.items()
        if key not in {"source_id","player_id","team_id"}):
        return {"sources":[],"players":[],"completeness":None,
            "limitations":["Карта времени не рассчитана: фильтр событий/партии или только человеческой проверки нельзя автоматически перенести на наблюдения детектора."]}
    from .presence import presence_report
    fingerprint=(project.db.execute("SELECT MAX(rowid) FROM observations").fetchone()[0],
        tuple(project.db.execute("SELECT COUNT(*),MAX(rowid) FROM objects WHERE kind IN ('job','source','player','track_binding')").fetchone()))
    cached=getattr(project,"_presence_report_cache",None)
    if cached is None or cached[0]!=fingerprint:
        cached=(fingerprint,presence_report(project));project._presence_report_cache=cached
    result=copy.deepcopy(cached[1])
    if filters.get("source_id"):
        result["sources"]=[s for s in result["sources"] if s["source_id"]==filters["source_id"]]
        result["players"]=[p for p in result["players"] if p["source_id"]==filters["source_id"]]
    if filters.get("player_id") or filters.get("team_id"):
        chosen=[p for p in result["players"] if (not filters.get("player_id") or p["player_id"]==filters["player_id"])
                and (not filters.get("team_id") or p["team_id"]==filters["team_id"])]
        result["players"]=chosen
        for source in result["sources"]:
            rows=[p for p in chosen if p["source_id"]==source["source_id"] and p["run_id"]==source["run_id"]]
            h,w=source["grid_rows"],source["grid_columns"]
            source["detection_grid"]=[[sum(p["detection_grid"][y][x] for p in rows) for x in range(w)] for y in range(h)]
            source["presence_seconds_grid"]=[[sum(p["presence_seconds_grid"][y][x] for p in rows) for x in range(w)] for y in range(h)]
            source["detection_count"]=sum(p["detection_count"] for p in rows)
            durations=[p["estimated_presence_seconds"] for p in rows if p["estimated_presence_seconds"] is not None]
            source["estimated_person_seconds"]=sum(durations) if durations else None
            source["players"]=rows;source["scope"]="Только явно привязанные выбранные игроки; неизвестные участники не распределены между составом"
            source["evidence_grid"]=[[next((p["evidence_grid"][y][x] for p in rows if p["evidence_grid"][y][x] is not None),None)
                                       for x in range(w)] for y in range(h)]
    result["filters"]=filters
    return result


def report(project, mode="automatic", filters=None):
    if mode not in MODES:
        raise ValueError("Неизвестный режим отчёта")
    match = project.get("match", "match")
    filters = filters or {}
    events = select(project, mode, filters)
    from .calibration import normalize, side_at
    for e in events:
        side = side_at(project, e.get("team_id"), e["source_id"], e["start"])
        e["normalized_floor_point"] = normalize(e["floor_point"], side) if e.get("floor_point") else None
        e["team_side"] = side
    players, teams = project.all("player"), project.all("team")
    rows = []
    for p in players:
        pe = [e for e in events if e.get("player_id") == p["id"] or p["id"] in (e.get("participants") or [])]
        from .roster import player_label
        rows.append({"id": p["id"], "name":player_label(p), "team_id": p["team_id"],
                     "number": p.get("number"), **action_stats(pe, match)})
    rallies = [r for r in project.all("rally") if not filters.get("set_id") or r.get("set_id") == filters["set_id"]]
    known = []
    for r in rallies:
        result = dict(r)
        for field in ("winner_team_id", "server_team_id"):
            if not value_allowed(r.get("evidence", {}).get(field, "unknown"), mode):
                result[field] = None
        known.append(result)
    ledger, conflicts = [], []
    for r in known:
        winner = r.get("winner_team_id")
        if not winner:
            continue
        candidates = [e for e in events if e.get("rally_id") == r["id"] and
                      ((e["type"] in ("attack", "tip") and e.get("outcome") == "kill") or
                       (e["type"] == "serve" and e.get("outcome") == "ace") or e["type"] == "block_point")]
        candidates = [e for e in candidates if e.get("team_id") == winner]
        causes = {e.get("context", {}).get("scoring_action_id", e["id"]): e for e in candidates}
        scoring = next(iter(causes.values())) if len(causes) == 1 else None
        if len(causes) > 1:
            conflicts.append({"rally_id": r["id"], "reason": "несколько взаимоисключающих авторов очка", "event_ids": [e["id"] for e in candidates]})
        credit = (scoring.get("credited_player_id") if scoring["type"] == "block_point" else scoring.get("player_id")) if scoring else None
        ledger.append({"rally_id": r["id"], "team_id": winner, "player_id": credit,
                       "event_id": scoring["id"] if scoring else None,
                       "cause": scoring["type"] if scoring else "unknown_or_opponent_error"})
    restrictive = any(filters.get(k) for k in ("player_id", "type", "outcome", "direction", "source_zone", "target_zone", "lineup_id"))
    team_rows = []
    for team in teams:
        tid = team["id"]
        te = [e for e in events if e.get("team_id") == tid]
        tr = {"id": tid, "name": team["name"], **action_stats(te, match)}
        tr['is_ours']=tid==match.get('our_team_id')
        eligible = [r for r in known if r.get("server_team_id") and r.get("winner_team_id")]
        receive = [r for r in eligible if r["server_team_id"] != tid] if not restrictive else []
        serving = [r for r in eligible if r["server_team_id"] == tid] if not restrictive else []
        tr.update(points_from_known_rallies=sum(x["team_id"] == tid for x in ledger),
                  unknown_rally_outcomes=sum(not r.get("winner_team_id") for r in known),
                  side_out=metric(sum(r["winner_team_id"] == tid for r in receive), len(receive),
                                  [r["id"] for r in receive], len(known) - len(eligible)),
                  break_point=metric(sum(r["winner_team_id"] == tid for r in serving), len(serving),
                                     [r["id"] for r in serving], len(known) - len(eligible)),
                  unknown_player_events=sum(not e.get("player_id") and not e.get("participants") for e in te),
                  rally_filter_limitation="показатели розыгрышей недоступны при фильтре отдельных действий/игроков" if restrictive else None)
        team_rows.append(tr)
    for p in rows:
        p["credited_points"] = sum(x["player_id"] == p["id"] for x in ledger)
    conclusions = []
    for row in rows + team_rows:
        for d, m in row["directions"].items():
            if m["numerator"]:
                conclusions.append({"entity": row["id"], "text": f"{row['name']}: {d} — {m['numerator']} из {m['denominator']} атак с известным направлением; ещё {m['missing']} без направления",
                    "small_sample": m["denominator"] < match["min_sample"], "metric": m,
                    "kind": "описательное наблюдение, причинность не установлена"})
    uncertain = []
    accepted_ids = {e["id"] for e in events}
    for e in project.events():
        missing = {k: e.get("unknown_reasons", {}).get(k, "неизвестно") for k in
                   ("player_id", "team_id", "outcome", "direction", "target_zone")
                   if e.get(k) is None or not value_allowed(e.get("evidence", {}).get(k), mode)}
        if missing or e["id"] not in accepted_ids:
            uncertain.append({"event_id": e["id"], "type": e["type"], "included": e["id"] in accepted_ids,
                              "review": e["review"], "missing": missing})
    total = match.get("known_event_total")
    from .performance import processing_diagnostics
    diagnostics = processing_diagnostics(project)
    from .scoring import score_report
    from .roster import identity_mode
    return {"schema_version": 2, "dictionary_version": "local-1", "match": match, "mode": mode,
            'manual_score':score_report(project,mode,filters.get('set_id')),
            'identity_mode':identity_mode(project),'identity_hypotheses':[h for h in project.all('identity_hypothesis') if h.get('mode','hybrid')==identity_mode(project)],
            'identity_warnings':project.all('identity_warning'),
            'team_kit_samples':[s for s in project.all('kit_sample') if s.get('active',True)],
            "mode_label": MODES[mode], "filters": filters, "events": events, "players": rows,
            "teams": team_rows, "rallies": known, "point_ledger": ledger, "conflicts": conflicts,
            "uncertainties": uncertain, "conclusions": conclusions,
            "completeness": None if total is None or total == 0 else len(events) / total,
            "known_total": total, "processing_diagnostics": diagnostics,
            "observation_maps":observation_maps(project,mode,filters),
            "temporal_diagnostics":project.all("temporal_job"),
            "geometry_warnings": project.all("geometry_warning"),
            "camera_sessions":project.all("camera_session"),
            "unknown_team_events": sum(not e.get("team_id") for e in events),
            "excluded_events": len(project.events()) - len(events),
            "limitations": ["Качество на реальных записях не измерено", "Полнота не определена без эталона",
                "Детекция людей и мяча не доказывает типы действий", "Геометрия мяча в воздухе не проецируется на пол",
                "Неизвестные составы, личности, зоны и исходы не восстанавливаются догадками"]}
