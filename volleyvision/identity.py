from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .models import verify


def torso(rgb, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return rgb[max(0, int(y1 + 0.18*h)):min(rgb.shape[0], int(y1 + 0.62*h)),
               max(0, int(x1 + 0.13*w)):min(rgb.shape[1], int(x2 - 0.13*w))]


def appearance(crop):
    if not crop.size:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    hist /= max(hist.sum(), 1)
    return hist.tolist()


class NumberOCR:
    def __init__(self, model_root, device="cpu"):
        import easyocr
        verify(model_root, "ocr")
        self.reader = easyocr.Reader(["en"], gpu=device == "cuda", detector=False,
                model_storage_directory=str(model_root), user_network_directory=str(model_root / "network"),
                download_enabled=False, verbose=False, quantize=False)

    def read(self, crop):
        if not crop.size or crop.shape[0] < 32 or crop.shape[1] < 16:
            return {"number": None, "reason": "номер слишком мал"}
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < 35:
            return {"number": None, "reason": "размыто", "sharpness": sharpness}
        h, w = gray.shape
        items = self.reader.recognize(gray, horizontal_list=[[0, w, 0, h]], free_list=[],
                allowlist="0123456789", detail=1, paragraph=False, batch_size=1)
        if not items:
            return {"number": None, "reason": "номер не читается", "sharpness": sharpness}
        _, text, score = max(items, key=lambda row: row[2])
        return {"number": int(text) if text.isdigit() and 1 <= len(text) <= 2 and score >= 0.6 else None,
                "text": text, "raw_score": float(score), "sharpness": sharpness,
                "reason": "порог OCR — не калиброванная вероятность"}


def consensus(observations, minimum=3, ratio=0.8):
    numbers = [o["number"] for o in observations if o.get("number") is not None]
    if not numbers:
        return {"number": None, "reason": "номер не читается", "votes": {}}
    counts = Counter(numbers)
    number, count = counts.most_common(1)[0]
    accepted = count >= minimum and count / len(numbers) >= ratio
    return {"number": number if accepted else None, "votes": dict(counts), "observations": len(numbers),
            "reason": "согласованные OCR-наблюдения; личность ещё не установлена" if accepted else "противоречивые или недостаточные OCR-наблюдения"}


def ensure_number_evidence(project,run_id,reader=None,cancel=None):
    from .image_io import read_image
    from .media import check_cancel
    tracks=[t for t in project.all("track") if t["run_id"]==run_id]
    pending=[]
    for t in tracks:
        observed={o.get("time") for o in t.get("ocr_observations",[])}
        samples=[s for s in t.get("samples",[]) if s["time"] not in observed]
        if samples:pending.append((t,samples))
    if not pending:return
    reader=reader or NumberOCR(project.root/"models","cpu")
    for track,samples in pending:
        for sample in samples:
            check_cancel(cancel)
            raw=read_image(project.root/sample["path"])
            if raw is None:continue
            result=reader.read(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))|{"time":sample["time"],"source_pts":sample.get("source_pts")}
            track.setdefault("ocr_observations",[]).append(result)
            track["ocr_observations"]=track["ocr_observations"][-200:]
        track["number_consensus"]=consensus(track.get("ocr_observations",[]));project.put("track",track)


def second_pass(project,run_id,mode=None,encoder=None,read_numbers=False,cancel=None,progress=None):
    """Persistent appearance memories; all new links remain independent hypotheses."""
    from .roster import identity_mode
    from .kits import hypotheses
    from .person_memory import enrich,match_tracks
    mode=identity_mode(project,mode)
    if mode!="appearance" and read_numbers:ensure_number_evidence(project,run_id,cancel=cancel)
    enrich(project,run_id,encoder,cancel,progress)
    tracks=[t for t in project.all("track") if t["run_id"]==run_id]
    for track in tracks:
        proposal=hypotheses(project,track)
        project.put("kit_hypothesis",proposal|{"id":run_id+":"+track["id"],"run_id":run_id,"track_id":track["id"]})
    match_tracks(project,run_id,mode)
