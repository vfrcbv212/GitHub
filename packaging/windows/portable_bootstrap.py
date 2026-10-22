"""Runs with the private interpreter in isolated mode; no installed Python used."""
import ctypes
import os
from pathlib import Path
import sys
import traceback

bundle = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(bundle / 'app'))
os.environ['VOLLEYVISION_BUNDLE_ROOT'] = str(bundle)
cpu_workers=str(max(1,min(8,(os.cpu_count() or 2)//2)))
os.environ.setdefault('OMP_NUM_THREADS', cpu_workers)
os.environ.setdefault('MKL_NUM_THREADS', cpu_workers)
try:
    from volleyvision.portable_gpu import selected_packages
    gpu = selected_packages(bundle)
    if gpu is not None:
        sys.path.insert(0, str(gpu))
    from volleyvision.portable import main
    raise SystemExit(main(bundle))
except SystemExit:
    raise
except BaseException:
    detail = traceback.format_exc()
    folder = Path(os.environ.get('LOCALAPPDATA', str(bundle))) / 'VolleyVision' / 'Logs'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'startup-error.txt'
    path.write_text(detail, encoding='utf-8')
    if '--self-test' not in sys.argv:
        ctypes.windll.user32.MessageBoxW(None, 'Ошибка запуска. Диагностика сохранена:\n' + str(path), 'VolleyVision', 0x10)
    raise SystemExit(2)
