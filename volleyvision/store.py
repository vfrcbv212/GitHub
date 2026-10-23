from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

from .domain import SCHEMA_VERSION, uid, validate_event


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Project:
    """One movable folder, SQLite transactions, separate automatic and manual layers."""

    def __init__(self, root, create=False, title="Новый матч", synthetic=False):
        self.root = Path(root).resolve()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        if not create and not (self.root / "project.sqlite").exists():
            raise ValueError("В папке нет project.sqlite")
        self.db = sqlite3.connect(self.root / "project.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.migrate()
        if not self.get("match", "match"):
            self.put("match", {"id": "match", "title": title, "synthetic": synthetic,
                               "schema_version": SCHEMA_VERSION, "known_event_total": None,
                               "min_sample": 10, "quality_positive": [2, 3],
                               "quality_excellent": [3], "quality_scale": [0, 1, 2, 3]})
        for folder in ("cache", "exports", "models", "media"):
            (self.root / folder).mkdir(exist_ok=True)

    def migrate(self):
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise ValueError("Проект создан более новой версией программы")
        if version < 1:
            self.db.executescript("""
            CREATE TABLE objects(kind TEXT NOT NULL,id TEXT NOT NULL,body TEXT NOT NULL,
                                 PRIMARY KEY(kind,id));
            CREATE TABLE events(id TEXT PRIMARY KEY,auto_key TEXT UNIQUE,base TEXT NOT NULL,
                                manual TEXT NOT NULL DEFAULT '{}',revision INTEGER DEFAULT 0);
            CREATE TABLE history(seq INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT NOT NULL,
                                 before_json TEXT,after_json TEXT,undone INTEGER DEFAULT 0,
                                 created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE frames(source_id TEXT,idx INTEGER,pts INTEGER,time REAL,keyframe INTEGER,
                                PRIMARY KEY(source_id,idx));
            CREATE INDEX frames_time ON frames(source_id,time);
            PRAGMA user_version=1;
            """)
        if version < 2:
            self.db.executescript("""
            CREATE TABLE observations(run_id TEXT,source_id TEXT,time REAL,body TEXT NOT NULL,
                                      PRIMARY KEY(run_id,time));
            CREATE TABLE model_revisions(seq INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT,
                                         body TEXT NOT NULL,created TEXT DEFAULT CURRENT_TIMESTAMP);
            PRAGMA user_version=2;
            """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def put(self, kind, value):
        value = copy.deepcopy(value)
        value.setdefault("id", uid())
        self.db.execute("INSERT OR REPLACE INTO objects VALUES (?,?,?)",
                        (kind, value["id"], dumps(value)))
        self.db.commit()
        return value

    def get(self, kind, object_id):
        r = self.db.execute("SELECT body FROM objects WHERE kind=? AND id=?", (kind, object_id)).fetchone()
        return json.loads(r[0]) if r else None

    def all(self, kind):
        values = [json.loads(r[0]) for r in self.db.execute("SELECT body FROM objects WHERE kind=? ORDER BY rowid", (kind,))]
        return sorted(values,key=lambda source:source.get("offset",0)) if kind=="source" else values

    def record_observation(self, job, observation, tracks=()):
        """Commit one detection checkpoint, including its tracks, atomically."""
        track_rows=[('track',track['id'],dumps(track)) for track in tracks]
        observation_row=(job['id'],job['source_id'],observation['time'],dumps(observation))
        job_row=('job',job['id'],dumps(job))
        with self.db:
            self.db.executemany('INSERT OR REPLACE INTO objects VALUES (?,?,?)',track_rows)
            self.db.execute('INSERT OR REPLACE INTO observations VALUES (?,?,?,?)',observation_row)
            self.db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)',job_row)

    def resolve(self, source, proxy=False):
        key = "proxy" if proxy and source.get("proxy") else "path"
        p = Path(source[key])
        if not p.is_absolute():
            p = self.root / p
        if not p.is_file():
            raise FileNotFoundError(f"Видео не найдено: {p}. Используйте «Восстановить путь».")
        return p

    def relpath(self, path):
        path = Path(path).resolve()
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def check_links(self, e):
        source = self.get("source", e["source_id"])
        if not source:
            raise ValueError("Сначала импортируйте исходное видео")
        if e["end"] > source["duration"] + 0.05:
            raise ValueError("Событие выходит за длительность исходного файла")
        if e.get("team_id") and not self.get("team", e["team_id"]):
            raise ValueError("Команда не существует")
        for player_id in list(e.get("participants") or []) + [e.get("player_id"), e.get("credited_player_id")]:
            if not player_id:
                continue
            p = self.get("player", player_id)
            if not p or (e.get("team_id") and p["team_id"] != e["team_id"]):
                raise ValueError("Игрок отсутствует в выбранной команде")
        if e.get("rally_id"):
            rally = self.get("rally", e["rally_id"])
            if not rally:
                raise ValueError("Розыгрыш не существует")
            if e.get("set_id") and e["set_id"] != rally.get("set_id"):
                raise ValueError("Партия события и розыгрыша не совпадают")

    def events(self):
        out = []
        for row in self.db.execute("SELECT * FROM events"):
            e = json.loads(row["base"])
            e.update(json.loads(row["manual"]))
            e["id"] = row["id"]
            e["revision"] = row["revision"]
            out.append(e)
        offsets = {s["id"]:s["offset"] for s in self.all("source")}
        return sorted(out, key=lambda e: (offsets.get(e["source_id"],0), e["start"], e["id"]))

    def save_event(self, e, expected_revision=None):
        validate_event(e)
        self.check_links(e)
        clean = {k: v for k, v in e.items() if k != "revision"}
        with self.db:
            row = self.db.execute("SELECT * FROM events WHERE id=?", (e["id"],)).fetchone()
            if row:
                if expected_revision is not None and expected_revision != row["revision"]:
                    raise ValueError("Событие изменилось. Откройте редактор заново")
                before = row["manual"]
                # Full manual snapshot: even explicit unknowns survive every model rerun.
                self.db.execute("UPDATE events SET manual=?, revision=revision+1 WHERE id=?", (dumps(clean), e["id"]))
            else:
                before = None
                self.db.execute("INSERT INTO events(id,base,manual) VALUES (?,?,?)", (e["id"], dumps(clean), dumps(clean)))
            self.db.execute("INSERT INTO history(event_id,before_json,after_json) VALUES(?,?,?)", (e["id"], before, dumps(clean)))

    def propose(self, e, auto_key):
        validate_event(e)
        self.check_links(e)
        with self.db:
            row = self.db.execute("SELECT id FROM events WHERE auto_key=?", (auto_key,)).fetchone()
            if not row:
                for old in self.events():
                    if old["source_id"] == e["source_id"] and abs(old["start"] - e["start"]) < 0.3:
                        if old.get("review") == "confirmed" or old.get("type") == e["type"]:
                            row = {"id": old["id"]}
                            break
            if row:
                eid = row["id"]
                self.db.execute("INSERT INTO model_revisions(event_id,body) VALUES(?,?)", (eid, dumps(e)))
                if self.db.execute("SELECT auto_key FROM events WHERE id=?", (eid,)).fetchone()[0]:
                    self.db.execute("UPDATE events SET base=? WHERE id=?", (dumps(e), eid))
                return eid
            self.db.execute("INSERT INTO events(id,auto_key,base) VALUES (?,?,?)", (e["id"], auto_key, dumps(e)))
            self.db.execute("INSERT INTO model_revisions(event_id,body) VALUES(?,?)", (e["id"], dumps(e)))
            return e["id"]

    def undo(self):
        with self.db:
            row = self.db.execute("SELECT * FROM history WHERE undone=0 ORDER BY seq DESC LIMIT 1").fetchone()
            if not row:
                return False
            if row["before_json"] is None:
                self.db.execute("DELETE FROM events WHERE id=?", (row["event_id"],))
            else:
                self.db.execute("UPDATE events SET manual=?,revision=revision+1 WHERE id=?", (row["before_json"], row["event_id"]))
            self.db.execute("UPDATE history SET undone=1 WHERE seq=?", (row["seq"],))
        return True

    def bind_track(self, run_id, track_id, start, end, player_id=None, role="participant"):
        if end < start or (player_id and not self.get("player", player_id)):
            raise ValueError("Проверьте интервал и игрока")
        return self.put("track_binding", {"run_id": run_id, "track_id": track_id,
                        "start": start, "end": end, "player_id": player_id, "role": role})

    def track_identity(self, run_id, track_id, t):
        rows = [r for r in self.all("track_binding") if r["run_id"] == run_id and r["track_id"] == track_id
                and r["start"] <= t <= r["end"]]
        return rows[-1] if rows else None

    def observations(self, run_id, start=0, end=1e99):
        for row in self.db.execute("SELECT body FROM observations WHERE run_id=? AND time>=? AND time<=? ORDER BY time", (run_id, start, end)):
            yield json.loads(row[0])

    def snapshot(self):
        return {"schema_version": SCHEMA_VERSION, "objects": [dict(r) | {"body": json.loads(r["body"])}
                for r in self.db.execute("SELECT * FROM objects")], "events": self.events(),
                "layers": [dict(r) for r in self.db.execute("SELECT * FROM events")],
                "history": [dict(r) for r in self.db.execute("SELECT * FROM history")]}

    def backup(self, target):
        dst = sqlite3.connect(target)
        self.db.backup(dst)
        dst.close()
