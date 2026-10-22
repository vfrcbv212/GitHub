"""Optional private CUDA runtime; a failed preparation preserves the CPU bundle."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

from .media import check_cancel

CUDA_INDEX='https://download.pytorch.org/whl/cu126'
PACKAGES=('torch==2.13.0','torchvision==0.28.0')
PROBE="""import sys,json
sys.path.insert(0,sys.argv[1])
import torch,torchvision
assert torch.version.cuda and torch.cuda.is_available(), 'CUDA недоступна; CPU-сборка сохранена'
value=(torch.ones(2,device='cuda')*2).cpu().tolist()
assert value==[2.,2.]
print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,
 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
 'vram':torch.cuda.get_device_properties(0).total_memory,'compute':value}))
"""


def probe(bundle,packages):
    result=subprocess.run([str(Path(bundle)/'runtime'/'python.exe'),'-I','-c',PROBE,str(packages)],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=os.environ|{'PYTHONIOENCODING':'utf-8'})
    if result.returncode:raise RuntimeError((result.stderr or result.stdout)[-1800:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def selected_packages(bundle):
    folder=Path(bundle)/'gpu'
    if not (folder/'READY.json').is_file():return None
    try:
        probe(bundle,folder)
        return folder
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired):
        # No imports changed in this process: the base CPU runtime remains usable.
        return None


def _install(command,log,cancel=None,progress=None):
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=os.environ|{'PYTHONIOENCODING':'utf-8'})
        try:
            while process.poll() is None:
                check_cancel(cancel)
                if progress:progress(.4,'Автоматическая загрузка NVIDIA/CUDA. CPU-сборка остаётся доступна; журнал: '+str(log))
                threading.Event().wait(.2)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if process.returncode:raise RuntimeError('Не удалось подготовить NVIDIA. CPU-сборка сохранена. Журнал: '+str(log))


def prepare(bundle,progress=None,cancel=None):
    bundle=Path(bundle);ready=selected_packages(bundle)
    if ready:return 'NVIDIA уже подготовлена. Перезапустите программу для использования GPU'
    check_cancel(cancel)
    if shutil.disk_usage(bundle).free<6*1024**3:raise ValueError('Для автоматической подготовки NVIDIA нужно не менее 6 ГБ свободного места')
    pending=bundle/'gpu.pending';final=bundle/'gpu'
    if pending.exists():shutil.rmtree(pending)
    pending.mkdir();log=bundle/'nvidia-setup.log'
    command=[str(bundle/'runtime'/'python.exe'),'-I','-m','pip','--isolated','install',
        '--disable-pip-version-check','--no-warn-script-location','--only-binary=:all:',
        '--no-deps','--target',str(pending),'--cache-dir',str(bundle/'download-cache'),
        '--index-url',CUDA_INDEX,*PACKAGES]
    _install(command,log,cancel,progress)
    check_cancel(cancel)
    try:info=probe(bundle,pending)
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:
        raise RuntimeError('Проверка NVIDIA не пройдена; CPU-сборка сохранена. '+str(exc)) from exc
    check_cancel(cancel)
    if final.exists():shutil.rmtree(final)
    pending.replace(final)
    (final/'READY.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return 'NVIDIA проверена: '+info['gpu']+'. Закройте программу и запустите VolleyVision.exe снова; GPU включится автоматически'
