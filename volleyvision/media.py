from __future__ import annotations

import hashlib
import json
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from .domain import uid


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel and cancel():
        raise Cancelled("Операция отменена; контрольная точка сохранена")


def sha256(path, cancel=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            check_cancel(cancel)
            h.update(chunk)
    return h.hexdigest()


def rotation_of(stream):
    return int(round(float(stream.metadata.get("rotate", 0)))) % 360


def probe(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Видео не найдено: {path}")
    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            first = next(c.decode(s))
            info = {"codec": s.codec_context.name, "width": s.width, "height": s.height,
                    "time_base": str(s.time_base), "average_rate": str(s.average_rate),
                    "first_pts": first.pts, "first_time": float(first.pts * first.time_base),
                    "duration": float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / av.time_base,
                    "rotation": (-int(getattr(first, "rotation", -rotation_of(s)))) % 360, "audio": bool(c.streams.audio),
                    "color_transfer": str(first.color_trc), "pixel_format": first.format.name}
    except (av.FFmpegError, IndexError, StopIteration) as exc:
        raise ValueError(f"Декодер не смог открыть видео ({type(exc).__name__}). Проверьте кодек и файл") from exc
    if shutil.which("ffprobe"):
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                                 "-show_format", "-of", "json", str(path)], capture_output=True, check=True)
        raw = json.loads(result.stdout)
        st = raw["streams"][0]
        info["ffprobe"] = raw
        info["color_transfer"] = st.get("color_transfer", info["color_transfer"])
        for side in st.get("side_data_list", []):
            if "rotation" in side:
                info["rotation"] = (-int(side["rotation"])) % 360
    info["hdr"] = info["color_transfer"] in ("smpte2084", "arib-std-b67", "16", "18")
    info["display_width"], info["display_height"] = ((info["height"], info["width"])
        if info["rotation"] in (90, 270) else (info["width"], info["height"]))
    return info


@dataclass
class VideoFrame:
    index: int
    pts: int
    time: float
    rgb: np.ndarray


def frames(path, metadata=None, start=0, end=float("inf"), sample_fps=None):
    """Sequential decode, at most one decoded frame retained. Time is actual PTS."""
    metadata = metadata or probe(path)
    origin = metadata["first_time"]
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if start > 0:
            c.seek(int((origin + start) / float(s.time_base)), stream=s, backward=True)
        next_sample = start
        for index, f in enumerate(c.decode(s)):
            if f.pts is None:
                raise ValueError("Кадр без PTS: точная временная привязка недоступна")
            t = float(f.pts * f.time_base) - origin
            if t < start - 1e-6:
                continue
            if t > end + 1e-6:
                break
            if sample_fps and t + 1e-6 < next_sample:
                continue
            if sample_fps:
                next_sample = t + 1 / sample_fps
            rgb = f.to_ndarray(format="rgb24")
            rotation = metadata.get("rotation", 0)
            if rotation:
                rgb = np.ascontiguousarray(np.rot90(rgb, -rotation // 90))
            yield VideoFrame(index, f.pts, t, rgb)


def frame_at(path, metadata, time_s):
    # First frame at/after the requested timestamp, never fps-derived positioning.
    return next(frames(path, metadata, max(0, time_s)))


def import_video(project, path, gap_before=None, progress=None, cancel=None):
    info = probe(path)
    file_hash = sha256(path, cancel)
    existing = [s for s in project.all("source") if s["sha256"] == file_hash]
    if existing:
        return index_video(project, existing[0], progress, cancel)
    sources = project.all("source")
    offset = sum(s["duration"] + (s.get("gap_before") or 0) for s in sources) + (gap_before or 0)
    source = project.put("source", {"id": uid(), "path": project.relpath(path), "sha256": file_hash,
            **info, "offset": offset, "gap_before": gap_before if sources else 0,
            "gap_unknown": bool(sources and gap_before is None), "index_complete": False})
    return index_video(project, source, progress, cancel)


def index_video(project, source, progress=None, cancel=None):
    if source.get("index_complete"):
        return source
    path = project.resolve(source)
    previous = None
    minimum, maximum, last = 1e99, 0, 0
    batch = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        count = 0
        for i, f in enumerate(c.decode(s)):
            check_cancel(cancel)
            if f.pts is None:
                raise ValueError("Не найден PTS кадра. Индексирование остановлено")
            t = float(f.pts * f.time_base) - source["first_time"]
            if previous is not None:
                dt = t - previous
                if dt <= 0:
                    raise ValueError("Немонотонные PTS: необходима проверка исходника")
                minimum, maximum = min(minimum, dt), max(maximum, dt)
                last = dt
            previous = t
            count = i + 1
            batch.append((source["id"], i, f.pts, t, int(f.key_frame)))
            if len(batch) == 500:
                with project.db:
                    project.db.executemany("INSERT OR REPLACE INTO frames VALUES(?,?,?,?,?)", batch)
                batch.clear()
                if progress:
                    progress(min(0.99, t / max(source["duration"], 0.01)), "Индекс PTS")
    with project.db:
        project.db.executemany("INSERT OR REPLACE INTO frames VALUES(?,?,?,?,?)", batch)
    source.update(index_complete=True, frame_count=count, duration=previous + last,
                  min_frame_interval=None if minimum == 1e99 else minimum, max_frame_interval=maximum,
                  variable_fps=maximum - minimum > 0.001)
    if progress:
        progress(1.0, f"Импорт: {count} кадров")
    return project.put("source", source)


def run_ffmpeg(args, log_path, duration=1, progress=None, cancel=None):
    executable = shutil.which("ffmpeg")
    if not executable:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise ValueError("FFmpeg не найден. Установите imageio-ffmpeg или добавьте ffmpeg.exe в PATH") from exc
    messages = queue.Queue(maxsize=32)
    with open(log_path, "wb") as err:
        proc = subprocess.Popen([executable, "-hide_banner", "-y", "-nostats", "-progress", "pipe:1", *map(str, args)],
                                stdout=subprocess.PIPE, stderr=err)
        def reader():
            for line in proc.stdout:
                try:
                    messages.put_nowait(line.decode(errors="replace").strip())
                except queue.Full:
                    pass
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            while proc.poll() is None:
                check_cancel(cancel)
                try:
                    line = messages.get(timeout=0.1)
                    if line.startswith("out_time_us=") and progress:
                        progress(min(0.99, int(line.split("=")[1]) / 1e6 / max(duration, 0.01)), "FFmpeg")
                except (queue.Empty, ValueError):
                    pass
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        finally:
            thread.join(timeout=1)
            proc.stdout.close()
    if proc.returncode:
        detail = Path(log_path).read_text(errors="replace")[-2400:]
        raise ValueError(f"Ошибка FFmpeg. Кодек или HDR-фильтр может не поддерживаться.\n{detail}")


def display_filters(source, hdr=True):
    filters = []
    rotation = source.get("rotation", 0)
    if rotation == 90:
        filters += ["transpose=1"]
    elif rotation == 270:
        filters += ["transpose=2"]
    elif rotation == 180:
        filters += ["hflip", "vflip"]
    if source.get("hdr") and hdr:
        filters += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
                    "tonemap=tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv", "format=yuv420p"]
    return filters


def fix_final_duration(path, seconds):
    """Keep the last VFR frame visible for its actual interval, without re-encoding.

    FFmpeg with an explicit fine encoder time base may write 1 us for the last
    packet. That preserves PTS but would shorten a montage by one frame per clip.
    Our encoder disables B frames, so the last packet is the last displayed frame.
    """
    path = Path(path)
    temp = path.with_name(path.stem + ".remux.mp4")
    try:
        with av.open(str(path)) as src, av.open(str(temp),"w",options={"movflags":"+faststart","use_editlist":"0","video_track_timescale":"1000000"}) as dst:
            stream = src.streams.video[0]
            output = dst.add_stream_from_template(stream)
            previous = None
            for packet in src.demux(stream):
                if packet.dts is None:
                    continue
                if previous is not None:
                    previous.stream = output
                    dst.mux(previous)
                previous = packet
            if previous is not None:
                previous.duration = max(1,round(seconds/float(previous.time_base)))
                previous.stream = output
                dst.mux(previous)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def create_proxy(project, source, height=720, progress=None, cancel=None):
    target = project.root / "cache" / f"{source['sha256'][:20]}-{height}-proxy.mp4"
    if target.exists() and source.get("proxy_verified") and source.get("proxy_height") == height:
        return target
    temp = target.with_name(target.stem + ".partial.mp4")
    filters = ["setpts=PTS-STARTPTS", *display_filters(source), f"scale=-2:{int(height)}"]
    try:
        run_ffmpeg(["-noautorotate", "-i", project.resolve(source), "-an", "-vf", ",".join(filters),
                    "-c:v", "libx264", "-bf", "0", "-use_editlist", "0", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-fps_mode", "passthrough", "-enc_time_base", "1:1000000", "-video_track_timescale", "1000000", "-movflags", "+faststart", temp],
                   target.with_suffix(".log"), source["duration"], progress, cancel)
        last_time = project.db.execute("SELECT MAX(time) FROM frames WHERE source_id=?", (source["id"],)).fetchone()[0]
        fix_final_duration(temp, source["duration"]-last_time)
        error = verify_pts(project, source, temp, cancel)
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    source.update(proxy=project.relpath(target), proxy_height=height, proxy_verified=True,
                  proxy_max_time_error=error, proxy_metadata=probe(target),
                  transforms={"original_rotation_degrees": source["rotation"],
                              "display_to_proxy_scale": height / source["display_height"],
                              "lens": "none; distortion not measured", "time": "source PTS minus first video PTS"})
    project.put("source", source)
    return target


def verify_pts(project, source, proxy_path, cancel=None):
    errors = 0.0
    expected = iter(project.db.execute("SELECT time FROM frames WHERE source_id=? ORDER BY idx", (source["id"],)))
    with av.open(str(proxy_path)) as c:
        stream = c.streams.video[0]
        for f in c.decode(stream):
            check_cancel(cancel)
            row = next(expected, None)
            if row is None:
                raise ValueError("Proxy содержит лишние кадры")
            errors = max(errors, abs(float(f.pts * f.time_base) - row[0]))
    if next(expected, None) is not None or errors > 0.002:
        raise ValueError(f"Proxy не прошла проверку соответствия PTS: {errors:.6f} с")
    return errors


def export_clip(project, e, target, before=2, after=3, cancel=None, output_size=None):
    source = project.get("source", e["source_id"])
    start, end = max(0, e["start"] - before), min(source["duration"], e["end"] + after)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Decode from start for exact video PTS even when audio/video have different start times.
    filters = ["setpts=PTS-STARTPTS", f"trim=start={start:.9f}:end={end:.9f}",
               "setpts=PTS-STARTPTS", *display_filters(source)]
    if output_size:
        w,h = output_size
        filters += [f"scale={w}:{h}:force_original_aspect_ratio=decrease",f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2", "setsar=1"]
    run_ffmpeg(["-noautorotate", "-i", project.resolve(source), "-an", "-vf", ",".join(filters),
                "-c:v", "libx264", "-bf", "0", "-use_editlist", "0", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-fps_mode", "passthrough", "-enc_time_base", "1:1000000", "-video_track_timescale", "1000000", "-movflags", "+faststart", target],
               target.with_suffix(".log"), end - start, cancel=cancel)
    last_time = project.db.execute("SELECT MAX(time) FROM frames WHERE source_id=? AND time>=? AND time<?", (source["id"],start,end)).fetchone()[0]
    if last_time is not None:
        fix_final_duration(target,end-last_time)
    times = project.db.execute("SELECT time FROM frames WHERE source_id=? AND time>=? AND time<? ORDER BY idx",
                               (source["id"], start, end))
    first = times.fetchone()
    manifest = {"event_id": e["id"], "source_id": source["id"], "requested_start": start, "requested_end": end,
                "first_source_time": first[0] if first else None, "source_first_pts": source["first_pts"],
                "source_time_base": source["time_base"], "source_sha256": source["sha256"],
                "audio": False, "output_size": output_size, "event_review": e.get("review"),
                "event_field_evidence": e.get("evidence"), "origin": e.get("origin"), "model_version": e.get("model_version"),
                "boundary": "clamped to source file; recording gaps never filled"}
    target.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def montage(clips, target, cancel=None):
    target = Path(target)
    # All inputs are normalised to the same dimensions by export_all.
    listing = target.with_suffix(".concat.txt")
    listing.write_text("\n".join("file '" + Path(p).resolve().as_posix().replace("'", "'\\''") + "'" for p in clips), encoding="utf-8")
    try:
        run_ffmpeg(["-f","concat","-safe","0","-i",listing,"-an","-c:v","copy","-movflags","+faststart",target],
                   target.with_suffix(".log"),cancel=cancel)
    finally:
        listing.unlink(missing_ok=True)
