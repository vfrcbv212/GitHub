import subprocess
import shutil
import imageio_ffmpeg

import cv2
import numpy as np
import pytest

from volleyvision.calibration import CameraGuard
from volleyvision.demo import make_video
from volleyvision.media import frame_at,probe
from volleyvision.models import choose_device


def test_rotation_with_and_without_ffprobe(tmp_path,monkeypatch):
    source=tmp_path/"original.mp4";rotated=tmp_path/"rotated.mp4"
    make_video(source,seconds=.3,vfr=False)
    subprocess.run([shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-y","-display_rotation","90","-i",str(source),"-c","copy",str(rotated)],check=True)
    metadata=probe(rotated)
    import volleyvision.media as media
    monkeypatch.setattr(media.shutil,"which",lambda _:None)
    fallback=probe(rotated)
    assert fallback["rotation"]==metadata["rotation"]==270
    original=frame_at(source,probe(source),0).rgb
    actual=frame_at(rotated,fallback,0).rgb
    assert np.array_equal(actual,np.rot90(original,1))


def test_cuda_missing_is_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    device,warning=choose_device("cuda")
    assert device=="cpu" and "CUDA" in warning
    assert choose_device("cpu")==("cpu",None)


def test_bad_codec_file_has_clear_error(tmp_path):
    bad=tmp_path/"bad.mov";bad.write_bytes(b"not a video")
    with pytest.raises(ValueError,match="Декодер"):probe(bad)


def test_camera_guard_movement_and_missing_landmarks():
    rng=np.random.default_rng(51);reference=rng.integers(0,256,(360,640,3),dtype=np.uint8)
    guard=CameraGuard(reference)
    assert guard.check(reference)["state"]=="stable"
    shifted=cv2.warpAffine(reference,np.float32([[1,0,20],[0,1,0]]),(640,360))
    assert guard.check(shifted)["state"]=="shifted"
    assert CameraGuard(np.zeros_like(reference)).check(reference)["state"]=="unknown"
