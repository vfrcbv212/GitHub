from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .domain import event
from .exporting import export_all
from .media import import_video
from .store import Project


def make_video(path, seconds=8, vfr=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt, s.time_base = 640, 360, "yuv420p", Fraction(1, 90000)
        t, i = 0.0, 0
        while t < seconds:
            rgb = np.full((360, 640, 3), (35, 47, 58), np.uint8)
            cv2.rectangle(rgb, (50, 100), (590, 330), (225, 110, 45), -1)
            cv2.rectangle(rgb, (50, 100), (590, 330), (245, 245, 235), 2)
            cv2.line(rgb, (320, 100), (320, 330), (240, 240, 240), 2)
            cv2.putText(rgb, "SYNTHETIC DEMO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
            for j, (x,y) in enumerate(((170,210),(260,280),(430,220),(510,290))):
                dx = int(8*np.sin(t+j))
                cv2.rectangle(rgb, (x+dx-16,y-50), (x+dx+16,y), (30,90,210) if j<2 else (230,225,200), -1)
                cv2.circle(rgb, (x+dx,y-61), 10, (210,180,150), -1)
                cv2.putText(rgb, str(j+1), (x+dx-8,y-20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2)
            cv2.circle(rgb, (int(70+(t*100)%500), int(80+40*np.sin(t*3))), 6, (255,250,80), -1)
            f = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            f.pts, f.time_base = round(t*90000), Fraction(1,90000)
            for packet in s.encode(f): c.mux(packet)
            t += (1/20 if i%5 == 0 else 1/30) if vfr else 1/30
            i += 1
        for packet in s.encode(): c.mux(packet)


def create(root):
    root = Path(root)
    if (root / "project.sqlite").exists():
        raise ValueError("Демо уже существует; выберите новую папку")
    with Project(root, create=True, title="ДЕМО: синтетический волейбол", synthetic=True) as p:
        video = p.root / "media" / "synthetic-vfr.mp4"
        make_video(video)
        source = import_video(p, video)
        for tid, name in (("A", "Команда А"), ("B", "Команда Б")):
            p.put("team", {"id": tid, "name": name})
            for n in (1, 2):
                p.put("player", {"id": tid+str(n), "team_id": tid, "number": n, "name": None,
                                 "role": "libero" if n==2 else "unknown"})
        p.put("set", {"id": "set1", "number": 1})
        for i, (server,winner) in enumerate((("A","A"),("A","B"),("B","B"),("B",None)), 1):
            p.put("rally", {"id": f"r{i}", "set_id": "set1", "source_id": source["id"],
                   "start": (i-1)*1.5, "end": i*1.5, "server_team_id": server, "winner_team_id": winner,
                   "partial": winner is None, "evidence": {"server_team_id": "confirmed", "winner_team_id": "confirmed" if winner else "unknown"}})
        examples = [
            (0.4,"serve",dict(team_id="A",player_id="A1",outcome="in_play",rally_id="r1")),
            (0.8,"attack",dict(team_id="A",player_id="A1",outcome="kill",direction="diagonal",source_zone=4,target_zone=5,rally_id="r1")),
            (1.7,"attack",dict(team_id="A",player_id="A1",outcome="blocked",direction="line",source_zone=4,rally_id="r2")),
            (1.8,"block_point",dict(team_id="B",participants=["B1","B2"],credited_player_id="B1",rally_id="r2")),
            (3.2,"attack",dict(team_id="A",player_id="A1",outcome="error",source_zone=2,rally_id="r3")),
            (4.6,"attack",dict(team_id="A",player_id="A1",outcome="continue",direction="diagonal",source_zone=4,rally_id="r4")),
            (5.2,"attack",dict(team_id="A",player_id="A1",rally_id="r4")),
            (5.8,"receive",dict(team_id="A",player_id="A2",quality=3)),
            (6.2,"set",dict(team_id="A",player_id="A2",target_player_id="A1",reception_quality=3,target_zone=4)),
        ]
        for t,kind,values in examples:
            p.save_event(event(source["id"],t,kind,set_id="set1",**values))
        candidate = event(source["id"],7,"candidate_contact",origin="automatic",review="unreviewed")
        candidate["evidence"]["type"]="hypothesis"
        p.propose(candidate,"synthetic-candidate")
        export_all(p)
        return str(p.root)
