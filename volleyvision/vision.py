from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque

import cv2
import numpy as np
import psutil

from . import __version__
from .calibration import CameraGuard, validate_profile, within
from .domain import event, uid
from .identity import NumberOCR, appearance, consensus, second_pass, torso
from .media import Cancelled, check_cancel, frames, prefetch_frames, sha256
from .models import choose_device, verify
from .store import dumps
from .image_io import read_image,write_image
from .scene_setup import TeamMemory,MainBall,validate_scene

DEFAULTS = {"people_fps": 5, "ball_fps": 15, "person_threshold": 0.65, "ball_threshold": 0.25,
            "ball_tile": 768, "ball_overlap": 0.2, "ball_tiled": True, "ocr": True,
            "ocr_interval": 0.5, "people_max_gap": 0.4, "ball_max_gap": 0.15, "device": "auto"}


class Detector:
    def __init__(self, model_root, device="auto"):
        import torch
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
        info, path = verify(model_root, "detector")
        self.version = info["sha256"]
        self.device, self.warning = choose_device(device)
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.person_embedding_version=self.version+':whole-person-fpn-2x1-v1'

    def person_embedding(self,rgb):
        """Generic learned whole-person descriptor, NOT a validated ReID model."""
        import torch
        if rgb.size==0 or rgb.shape[0]<48 or rgb.shape[1]<20:return None
        # Preserve body proportions; no face crop or external identity database.
        scale=min(128/rgb.shape[1],256/rgb.shape[0]);w=max(1,round(rgb.shape[1]*scale));h=max(1,round(rgb.shape[0]*scale))
        canvas=np.full((256,128,3),127,np.uint8);small=cv2.resize(rgb,(w,h));left=(128-w)//2;top=(256-h)//2
        canvas[top:top+h,left:left+w]=small
        tensor=torch.from_numpy(canvas).permute(2,0,1).float().div(255).to(self.device)
        mean=tensor.new_tensor(self.model.transform.image_mean)[:,None,None];std=tensor.new_tensor(self.model.transform.image_std)[:,None,None]
        with torch.inference_mode():
            features=self.model.backbone(((tensor-mean)/std)[None])
            vector=torch.nn.functional.adaptive_avg_pool2d(next(iter(features.values())),(2,1)).flatten()
            vector=torch.nn.functional.normalize(vector,dim=0)
        return vector.cpu().tolist()

    def predict_many(self, images, threshold=0.25, classes=(1, 37), batch_size=1):
        """Batch equal-sized images without reducing ball tile resolution.

        Bound GPU memory and retry an OOM at a smaller batch. No image is dropped.
        Images of different sizes are grouped separately, preserving the exact
        resize policy of the original single-image detector.
        """
        import torch
        result=[None]*len(images)
        buckets={}
        for i,rgb in enumerate(images):buckets.setdefault(rgb.shape[:2],[]).append(i)
        for shape,indices in buckets.items():
            self.model.transform.min_size=(min(shape),) if classes==(37,) else (320,)
            self.model.transform.max_size=max(shape) if classes==(37,) else 640
            limit=max(1,min(int(batch_size),len(indices)));offset=0
            while offset<len(indices):
                selected=indices[offset:offset+limit]
                tensors=[torch.from_numpy(np.ascontiguousarray(images[i])).permute(2,0,1)
                         .to(device=self.device,dtype=torch.float32).div_(255) for i in selected]
                try:
                    with torch.inference_mode():outputs=self.model(tensors)
                except torch.cuda.OutOfMemoryError:
                    del tensors
                    if self.device!='cuda' or len(selected)==1:raise
                    torch.cuda.empty_cache();limit=max(1,len(selected)//2);continue
                for i,out in zip(selected,outputs):
                    boxes=out['boxes'].detach().float().cpu().numpy()
                    labels=out['labels'].detach().cpu().numpy()
                    scores=out['scores'].detach().float().cpu().numpy()
                    result[i]=[{'bbox':b.tolist(),'class_id':int(c),'score':float(s)}
                               for b,c,s in zip(boxes,labels,scores) if c in classes and s>=threshold]
                del tensors,outputs
                offset+=len(selected)
        return result

    def predict(self, rgb, threshold=0.25, classes=(1, 37)):
        return self.predict_many([rgb],threshold,classes)[0]

    def tile_batch_size(self, params):
        requested=params.get('inference_batch')
        if requested is not None:
            if not isinstance(requested,int) or isinstance(requested,bool) or not 1<=requested<=8:
                raise ValueError('Размер пакета должен быть 1–8')
            return requested
        if self.device!='cuda':return 1
        import torch
        free,_=torch.cuda.mem_get_info()
        # Leave space for the model, OCR and display; OOM retries are still bounded.
        return max(1,min(4,int(max(0,free-1024**3)//(800*1024**2))))

    def balls(self, rgb, params, polygon=None):
        if not params["ball_tiled"]:
            items = self.predict(rgb, params["ball_threshold"], (37,))
            return [d for d in items if not polygon or within(polygon,
                [(d["bbox"][0]+d["bbox"][2])/2,(d["bbox"][1]+d["bbox"][3])/2])]
        from torchvision.ops import nms
        import torch
        h, w = rgb.shape[:2]
        size = params["ball_tile"]
        stride = max(64, int(size * (1 - params["ball_overlap"])))
        xs = sorted(set([*range(0, max(1, w - size + 1), stride), max(0, w - size)]))
        ys = sorted(set([*range(0, max(1, h - size + 1), stride), max(0, h - size)]))
        found = [];tiles=[];origins=[]
        bounds=cv2.boundingRect(np.asarray(polygon,np.float32)) if polygon else None
        for y in ys:
            for x in xs:
                if bounds:
                    bx,by,bw,bh=bounds
                    if x+size<bx or x>bx+bw or y+size<by or y>by+bh:continue
                crop = rgb[y:y+size, x:x+size]
                tiles.append(crop);origins.append((x,y))
        predictions=self.predict_many(tiles,params['ball_threshold'],(37,),self.tile_batch_size(params))
        for (x,y),items in zip(origins,predictions):
            for d in items:
                b = d["bbox"]
                d["bbox"] = [b[0] + x, b[1] + y, b[2] + x, b[3] + y]
                center = [(d["bbox"][0] + d["bbox"][2])/2, (d["bbox"][1] + d["bbox"][3])/2]
                if not polygon or within(polygon, center):
                    found.append(d)
        if not found:
            return []
        keep = nms(torch.tensor([d["bbox"] for d in found],dtype=torch.float32), torch.tensor([d["score"] for d in found]), 0.4)
        return [found[i] for i in keep.tolist()]


class Trackers:
    def __init__(self, params, width, height, segment):
        from norfair import Tracker
        self.width, self.height, self.segment = width, height, segment
        self.people = Tracker(distance_function="iou", distance_threshold=0.7,
            hit_counter_max=max(1, math.ceil(params["people_max_gap"] * params["people_fps"])), initialization_delay=0)
        self.ball = Tracker(distance_function="euclidean", distance_threshold=0.12,
            hit_counter_max=max(1, math.ceil(params["ball_max_gap"] * params["ball_fps"])), initialization_delay=0)

    def update(self, detections, t, kind):
        from norfair import Detection
        tracker = self.people if kind == "person" else self.ball
        rows = []
        for d in detections:
            b = np.array(d["bbox"], float).reshape(2, 2) / [self.width, self.height]
            points = b if kind == "person" else b.mean(axis=0)[None]
            rows.append(Detection(points, scores=np.full(len(points), d["score"]), data={**d, "time": t}))
        objects = tracker.update(rows)
        out = []
        for obj in objects:
            # Never persist Kalman predictions as observations during a lost ball/person interval.
            if obj.last_detection.data["time"] == t:
                out.append({**obj.last_detection.data, "track_id": f"{self.segment}-{kind}-{obj.id}", "observed": True})
        return out


class TemporalCandidates:
    def __init__(self):
        self.window = deque(maxlen=3)
        self.last_contact = -1e9

    def update(self, t, balls, people, max_gap):
        if len(balls) != 1:
            self.window.clear()
            return None
        box = balls[0]["bbox"]
        center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
        if self.window and t - self.window[-1][0] > max_gap:
            self.window.clear()
        self.window.append((t, center))
        if len(self.window) < 3 or t - self.last_contact < 0.4:
            return None
        (t0, a), (t1, b), (t2, c) = self.window
        va, vb = (b-a)/max(t1-t0, 1e-6), (c-b)/max(t2-t1, 1e-6)
        angle = float(np.dot(va, vb) / max(np.linalg.norm(va)*np.linalg.norm(vb), 1))
        if angle > 0.3:
            return None
        near = []
        for p in people:
            x1,y1,x2,y2 = p["bbox"]
            if x1-0.15*(x2-x1) <= b[0] <= x2+0.15*(x2-x1) and y1-0.2*(y2-y1) <= b[1] <= y2:
                near.append(p["track_id"])
        if not near:
            return None
        self.last_contact = t1
        return {"time": t1, "candidate_tracks": near, "cosine": angle,
                "explanation": "3 наблюдения мяча, изменение экранной траектории рядом с человеком; это не доказанное касание и не тип действия"}


def analyze(project, source_id, config=None, progress=None, cancel=None, setup_only=False):
    source = project.get("source", source_id)
    if not source:
        raise ValueError("Сначала импортируйте видео")
    path = project.resolve(source)
    decode_metadata = source
    configuration=config or {}
    temporal_artifact=configuration.get("temporal_artifact")
    from .roster import identity_mode,set_identity_mode
    selected_mode=identity_mode(project,configuration.get('identity_mode'))
    if 'identity_mode' in configuration:set_identity_mode(project,selected_mode)
    params = DEFAULTS | {k:v for k,v in configuration.items() if not k.startswith("temporal_") and k!='identity_mode'}
    if not (0 < params["people_fps"] <= params["ball_fps"] <= 120):
        raise ValueError("Нужны 0 < частота людей <= частота мяча <= 120")
    if sha256(path, cancel) != source["sha256"]:
        raise ValueError("Содержимое исходника изменилось. Импортируйте новую версию отдельно")
    if source.get("hdr"):
        # Tone-map at original display resolution for CV: do not force a tiny-ball pass through 720p.
        from .media import create_proxy, probe
        path = create_proxy(project, source, height=source["display_height"], progress=progress, cancel=cancel)
        decode_metadata = probe(path)
    calibrations = []
    for candidate_profile in project.all("calibration"):
        if candidate_profile.get("source_id") != source_id:
            continue
        try:
            calibrations.append(validate_profile(candidate_profile,source_id=source_id,
                source_sha256=source["sha256"],image_size=[source["display_width"],source["display_height"]]))
        except ValueError as exc:
            project.put("geometry_warning",{"id":"invalid:"+candidate_profile["id"],"source_id":source_id,
                "calibration_id":candidate_profile["id"],"reason":str(exc),"state":"invalid","time":candidate_profile.get("start",0)})
    fallback_profile = {"id":None,"source_id":source_id,"geometry":"video","verified":False}
    calibration = calibrations[-1] if calibrations else fallback_profile
    scene=project.get('scene_setup',source_id)
    if scene:
        try:scene=validate_scene(scene,source)
        except ValueError:scene=None
    model_info, _ = verify(project.root / "models", "detector")
    key_data = {"source": source["sha256"], "model": model_info["sha256"], "params": params,
                "calibration": calibration, "calibration_profiles":calibrations, "code": __version__,"observations_contract":3}
    if scene:key_data['scene_setup']=scene
    run_id = hashlib.sha256(dumps(key_data).encode()).hexdigest()[:24]
    job = project.get("job", run_id) or {"id": run_id, "source_id": source_id, "last_time": -1,
              "state": "new", "params": key_data, "samples": 0, "segments": 0}
    if job["state"] == "complete" or (setup_only and job["state"] == "setup_complete"):
        second_pass(project,run_id,selected_mode,read_numbers=params['ocr'],cancel=cancel,progress=progress)
        if not setup_only:
            run_optional_temporal(project,run_id,temporal_artifact,progress,cancel)
            from .exporting import export_all
            export_all(project)  # report can change after an optional manual edit
        return job
    detector = Detector(project.root / "models", params["device"])
    ocr = NumberOCR(project.root / "models", detector.device) if params["ocr"] and selected_mode!='appearance' else None
    job.update(state="running", segments=job["segments"]+1, device=detector.device, warning=detector.warning,
               automation_limit="action classifiers, outcomes and reliable re-identification need annotated video; contact candidates only")
    project.put("job", job)
    start_wall = time.perf_counter()
    start = max(0, job["last_time"] + 1e-5)
    end = min(source["duration"], params.get("setup_end", source["duration"])) if setup_only else source["duration"]
    trackers = Trackers(params, source["display_width"], source["display_height"], f"{run_id[:8]}s{job['segments']}")
    temporal = TemporalCandidates()
    team_memory=TeamMemory(scene)
    main_ball=MainBall(scene)
    people, last_people, last_guard = [], -1e9, -1e9
    guard = None
    geometry_ok = aligned = False
    uninitialised_profile = object()
    last_profile_id = uninitialised_profile
    last_guard_state = "unknown"
    active_camera_session=None
    tracks = {}
    process = psutil.Process()
    max_ram = process.memory_info().rss
    ball_samples, ball_missing = job.get("ball_samples", 0), job.get("ball_missing", 0)
    timings=dict(job.get('timings',{}));iteration_finished=time.perf_counter()
    decoded=prefetch_frames(frames(path,decode_metadata,start=start,end=end,sample_fps=params['ball_fps']),cancel=cancel)
    try:
        for frame in decoded:
            check_cancel(cancel)
            timings['decode_wait_s']=timings.get('decode_wait_s',0)+time.perf_counter()-iteration_finished
            rgb, t = frame.rgb, frame.time
            original_pts = project.db.execute("SELECT pts FROM frames WHERE source_id=? AND time>=? ORDER BY time LIMIT 1", (source_id,t-0.002)).fetchone()
            original_pts = original_pts[0] if original_pts else None
            matching=[c for c in calibrations if c.get("start",0)<=t and (c.get("end") is None or t<=c["end"])]
            calibration=matching[-1] if matching else fallback_profile
            if calibration["id"] != last_profile_id:
                if active_camera_session is not None:
                    active_camera_session["end"]=t;project.put("camera_session",active_camera_session);active_camera_session=None
                guard=None;aligned=geometry_ok=False;last_guard=-1e9;last_guard_state="unknown"
                if last_profile_id is not uninitialised_profile:
                    job["segments"]+=1
                    trackers=Trackers(params,source["display_width"],source["display_height"],f"{run_id[:8]}s{job['segments']}")
                    temporal=TemporalCandidates();people=[];last_people=-1e9
                    main_ball=MainBall(scene)
                reference_path=calibration.get("reference_image")
                if reference_path:
                    reference=read_image(project.root/reference_path)
                    if reference is not None:guard=CameraGuard(cv2.cvtColor(reference,cv2.COLOR_BGR2RGB))
                if calibration["id"] and guard is None:
                    project.put("geometry_warning",{"id":"reference:"+calibration["id"],"source_id":source_id,
                        "calibration_id":calibration["id"],"time":t,"state":"unknown",
                        "reason":"Нет доступного опорного кадра; геометрия и старые области поиска отключены, обработка продолжается"})
                last_profile_id=calibration["id"]
            if guard and t - last_guard >= 2:
                state = guard.check(rgb)
                aligned=state["state"]=="stable"
                geometry_ok = aligned and calibration.get("verified", False)
                if last_guard>-1e8 and (last_guard_state=="stable") != aligned:
                    job["segments"]+=1
                    trackers=Trackers(params,source["display_width"],source["display_height"],f"{run_id[:8]}s{job['segments']}")
                    temporal=TemporalCandidates();people=[];last_people=-1e9
                    main_ball=MainBall(scene if aligned else None)
                if state["state"] != "stable":
                    project.put("geometry_warning", {"id": f"{run_id}:{t:.3f}", "source_id": source_id, "time": t, **state})
                if state["state"]=="shifted" and active_camera_session is None:
                    active_camera_session=project.put("camera_session",{"id":f"{run_id}:shift:{t:.6f}","run_id":run_id,
                        "source_id":source_id,"start":t,"end":None,"previous_calibration_id":calibration["id"],
                        "geometry":"video","reason":"Обнаружен сдвиг камеры; прежняя геометрия отключена без остановки обработки"})
                elif state["state"]=="stable" and active_camera_session is not None:
                    active_camera_session["end"]=t;project.put("camera_session",active_camera_session);active_camera_session=None
                last_guard = t
                last_guard_state=state["state"]
            if t - last_people >= 1/params["people_fps"] - 1e-5:
                inference_start=time.perf_counter()
                found = detector.predict(rgb, params["person_threshold"], (1,))
                timings['people_inference_s']=timings.get('people_inference_s',0)+time.perf_counter()-inference_start
                polygon = calibration.get("people_roi") if aligned else None
                excluded = calibration.get("obstacles", []) if aligned else []
                found = [p for p in found if (not polygon or within(polygon, [(p["bbox"][0]+p["bbox"][2])/2,p["bbox"][3]]))
                         and not any(within(poly, [(p["bbox"][0]+p["bbox"][2])/2,p["bbox"][3]]) for poly in excluded)]
                people = trackers.update(found, t, "person")
                last_people = t
                scene_bindings={}
                for person in people:
                    person['referee_region_hint']=bool(aligned and calibration.get('referee_region') and
                        within(calibration['referee_region'],[(person['bbox'][0]+person['bbox'][2])/2,(person['bbox'][1]+person['bbox'][3])/2]))
                    tid = person["track_id"]
                    track = tracks.get(tid) or project.get("track", tid) or {"id": tid, "run_id": run_id,
                              "source_id": source_id, "start": t, "end": t, "ocr_observations": [], "samples": []}
                    track["end"] = t
                    samples=track.setdefault('person_samples',[])
                    if len(samples)<3 and (not samples or t-samples[-1]['time']>=.5):
                        x1,y1,x2,y2=map(int,person['bbox']);full=rgb[max(0,y1):min(rgb.shape[0],y2),max(0,x1):min(rgb.shape[1],x2)]
                        if full.size and full.shape[0]>=48 and full.shape[1]>=20 and float(cv2.Laplacian(cv2.cvtColor(full,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())>=35:
                            sample_path=project.root/'cache'/f'person-{tid}-{len(samples)}.jpg'
                            if write_image(sample_path,cv2.cvtColor(full,cv2.COLOR_RGB2BGR)):
                                samples.append({'path':project.relpath(sample_path),'time':t,'source_pts':original_pts,'bbox':person['bbox']})
                    binding = project.track_identity(run_id, tid, t)
                    if binding:scene_bindings[tid]=binding
                    if binding:
                        person.update(player_id=binding.get("player_id"), role=binding["role"])
                    crop=torso(rgb,person['bbox'])
                    if crop.size and len(track['samples'])<3 and (not track['samples'] or t-track['samples'][-1]['time']>=.5):
                        sharpness=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
                        if crop.shape[0]>=16 and crop.shape[1]>=12 and sharpness>=35:
                            crop_path=project.root/'cache'/f"{tid}-{len(track['samples'])}.jpg"
                            write_image(crop_path,cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                            track['samples'].append({'path':project.relpath(crop_path),'time':t});track['appearance']=appearance(crop)
                    if ocr and (not track["ocr_observations"] or t-track["ocr_observations"][-1]["time"] >= params["ocr_interval"]):
                        result = ocr.read(crop) | {"time": t, "source_pts": original_pts}
                        track["ocr_observations"].append(result)
                        # Bound per-track memory for arbitrarily long matches.
                        track["ocr_observations"] = track["ocr_observations"][-200:]
                        track["number_consensus"] = consensus(track["ocr_observations"])
                    tracks[tid] = track
                    project.put("track", track)
                if scene and aligned:
                    team_memory.update(people,t,scene_bindings)
                    for person in people:
                        track=tracks[person['track_id']];track['team_proposal']=person['team_proposal'];project.put('track',track)
                elif scene:team_memory=TeamMemory(scene)
            inference_start=time.perf_counter()
            balls = trackers.update(detector.balls(rgb, params, calibration.get("ball_roi") if aligned else None), t, "ball")
            timings['ball_inference_s']=timings.get('ball_inference_s',0)+time.perf_counter()-inference_start
            ball_candidates=balls
            balls,ball_selection=main_ball.select(balls,t)
            for ball in balls:
                ball['net_region_overlap_hint']=bool(aligned and calibration.get('net_region') and
                    within(calibration['net_region'],[(ball['bbox'][0]+ball['bbox'][2])/2,(ball['bbox'][1]+ball['bbox'][3])/2]))
            ball_samples += 1
            ball_missing += int(not balls)
            candidate = temporal.update(t, balls, [p for p in people if p.get('scene_role')!='referee'] if t-last_people < 1/params["people_fps"]+0.02 else [], params["ball_max_gap"])
            if candidate:
                e = event(source_id, candidate["time"], "candidate_contact", review="unreviewed", origin="automatic",
                          model_version=model_info["sha256"], visibility="partial", context=candidate,
                          raw_scores={"trajectory_cosine": candidate["cosine"]})
                e["evidence"]["type"] = "hypothesis"
                e["unknown_reasons"].update(player_id="ближайший игрок не доказывает касание", outcome="не наблюдался",
                                           direction="экранный изгиб не задаёт направление на корте", target_zone="недостаточная геометрия/нет контакта пола")
                project.propose(e, f"{source_id}:contact:{candidate['time']:.3f}")
            observation = {"time": t, "source_pts": original_pts, "people": people if t == last_people else [],
                           "people_sampled": t == last_people, "balls": balls, "ball_missing": not balls,
                           "ball_candidates":ball_candidates,"ball_selection":ball_selection,
                           "geometry_valid": geometry_ok, "approximate_zones_valid":aligned,
                           "tracking_segment":f"{run_id[:8]}s{job['segments']}",
                           "calibration_id": calibration["id"],"camera_session_id":active_camera_session["id"] if active_camera_session else None}
            job.update(last_time=t, samples=job["samples"]+1, ball_samples=ball_samples, ball_missing=ball_missing)
            with project.db:
                project.db.execute("INSERT OR REPLACE INTO observations VALUES (?,?,?,?)", (run_id, source_id, t, dumps(observation)))
                project.db.execute("INSERT OR REPLACE INTO objects VALUES ('job',?,?)", (run_id, dumps(job)))
            max_ram = max(max_ram, process.memory_info().rss)
            # Tracks that are no longer live are stored in SQLite, not retained in RAM.
            tracks = {k: v for k, v in tracks.items() if t-v["end"] <= 2}
            if progress:
                elapsed=max(.001,time.perf_counter()-start_wall);speed=max(0,t-start)/elapsed
                remaining=f'; примерно {(end-t)/speed/60:.1f} мин осталось' if speed>0 and elapsed>3 else ''
                progress(t/max(end, 0.01), f"{detector.device.upper()}: {t:.1f}/{end:.1f} с видео; {speed:.2f}×{remaining}; пропуски мяча {ball_missing}/{ball_samples}")
            iteration_finished=time.perf_counter()
        identity_start=time.perf_counter()
        second_pass(project,run_id,selected_mode,encoder=detector,read_numbers=params['ocr'],cancel=cancel,progress=progress)
        timings['identity_s']=timings.get('identity_s',0)+time.perf_counter()-identity_start
        job["state"] = "setup_complete" if setup_only and end < source["duration"] - 0.002 else "complete"
        job["processed_until"] = end
    except Cancelled:
        job["state"] = "cancelled"
        raise
    except BaseException as exc:
        job.update(state="failed", error=str(exc))
        raise
    finally:
        decoded.close()
        import torch
        job.update(elapsed_s=job.get("elapsed_s", 0)+time.perf_counter()-start_wall, peak_ram_bytes=max_ram,
                   peak_vram_bytes=torch.cuda.max_memory_allocated() if detector.device == "cuda" else None,
                   timings=timings,
                   resume_policy="new track segment after checkpoint; old identities never assigned by proximity")
        project.put("job", job)
    if not setup_only:
        run_optional_temporal(project,run_id,temporal_artifact,progress,cancel)
        from .exporting import export_all
        target = export_all(project)
        job["report"] = str(target)
        project.put("job", job)
    return job


def run_optional_temporal(project,run_id,artifact,progress=None,cancel=None):
    if not artifact:return None
    from pathlib import Path
    path=Path(artifact)
    if not path.is_absolute():path=project.root/path
    try:
        from .temporal import infer_project
        return infer_project(project,run_id,path,progress=progress,cancel=cancel)
    except (ValueError,FileNotFoundError,RuntimeError,ImportError) as exc:
        # A bad optional action model must not erase observations or block a partial report.
        source=project.get("job",run_id)["source_id"]
        warning={"id":"temporal-error:"+run_id,"run_id":run_id,"source_id":source,
            "state":"failed","accepted":0,"reason":str(exc),
            "limitations":"Дополнительная модель действий не использована; доступный отчёт создан без неё"}
        project.put("temporal_job",warning)
        return warning
