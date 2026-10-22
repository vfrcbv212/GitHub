import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

import av
import psutil


def diagnose():
    result = {"platform": platform.platform(), "python": sys.version, "ram_bytes": psutil.virtual_memory().total,
              "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"),
              "pyav_library_versions": {k: list(v) for k,v in av.library_versions.items()}, "packages": {}}
    for n in ("torch", "torchvision", "PySide6", "av", "numpy", "opencv-python-headless", "norfair", "easyocr", "openpyxl"):
        try: result["packages"][n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: result["packages"][n] = None
    smi = shutil.which("nvidia-smi")
    if smi:
        p = subprocess.run([smi,"--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True)
        result["nvidia_smi"] = p.stdout.strip() if p.returncode==0 else p.stderr.strip()
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_runtime"] = torch.version.cuda
        if result["cuda_available"]:
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
            result["cuda_compute_test"] = (torch.ones(2, device="cuda")*2).cpu().tolist()
        else:
            result["mode"] = "CPU; Windows/CUDA не проверены этой командой"
    except (ImportError, RuntimeError) as exc:
        result["torch_error"] = str(exc)
    return result
