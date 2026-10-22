import hashlib
import json

import pytest

from volleyvision.models import seed_bundled,fetch,verify
from volleyvision.portable import default_project


def bundle(tmp_path,monkeypatch):
    root=tmp_path/'bundle';models=root/'models';models.mkdir(parents=True)
    data={}
    for name in ('detector','ocr'):
        path=models/(name+'.pth');path.write_bytes(name.encode())
        data[name]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (models/'manifest.json').write_text(json.dumps(data))
    monkeypatch.setenv('VOLLEYVISION_BUNDLE_ROOT',str(root))
    return root,models,data


def test_portable_models_prepare_offline_and_survive_reopen(tmp_path,monkeypatch):
    root,source,data=bundle(tmp_path,monkeypatch);target=tmp_path/'project'/'models'
    # fetch returns before torch/EasyOCR/download code; no network setup needed.
    prepared=fetch(target,include_ocr=False)
    assert set(prepared)=={'detector'} and not (target/'ocr.pth').exists()
    verify(target,'ocr');assert (target/'ocr.pth').read_bytes()==b'ocr'
    assert json.loads((target/'manifest.json').read_text(encoding="utf-8"))==data
    assert seed_bundled(target)==data
    assert default_project(root)==root/'Projects'/'Match1'


def test_portable_corrupt_shipped_model_fails_without_download(tmp_path,monkeypatch):
    root,source,data=bundle(tmp_path,monkeypatch)
    (source/'detector.pth').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='повреждена'):fetch(tmp_path/'project'/'models')


def test_portable_does_not_replace_project_model(tmp_path,monkeypatch):
    root,source,data=bundle(tmp_path,monkeypatch);target=tmp_path/'models';target.mkdir()
    custom=target/'custom.pth';custom.write_bytes(b'custom-trained-model')
    info={'file':custom.name,'sha256':hashlib.sha256(custom.read_bytes()).hexdigest()}
    (target/'manifest.json').write_text(json.dumps({'detector':info}))
    seed_bundled(target)
    assert verify(target,'detector')[0]==info and custom.read_bytes()==b'custom-trained-model'
    assert not (target/'detector.pth').exists()


def test_portable_refuses_existing_corrupt_file_and_path_escape(tmp_path,monkeypatch):
    root,source,data=bundle(tmp_path,monkeypatch);target=tmp_path/'models';target.mkdir()
    (target/'detector.pth').write_bytes(b'unrelated')
    with pytest.raises(ValueError,match='Локальная модель повреждена'):seed_bundled(target)
    data['detector']['file']='../escape.pth';(source/'manifest.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='путь модели'):seed_bundled(tmp_path/'other')


def test_unicode_images_roundtrip_and_missing_file(tmp_path):
    import numpy as np
    from volleyvision.image_io import read_image,write_image
    path=tmp_path/'Иван — площадка.png';image=np.full((40,30,3),128,np.uint8)
    assert write_image(path,image) and np.array_equal(read_image(path),image)
    assert read_image(tmp_path/'нет.png') is None
    assert not write_image(tmp_path/'missing'/'sample.png',image)


def test_gpu_install_is_private_and_activates_only_after_compute_test(tmp_path,monkeypatch):
    from volleyvision import portable_gpu as gpu
    from types import SimpleNamespace
    monkeypatch.setattr(gpu.shutil,'disk_usage',lambda *a:SimpleNamespace(free=10*1024**3))
    commands=[]
    monkeypatch.setattr(gpu,'_install',lambda command,*a:commands.append(command))
    info={'gpu':'RTX test','compute':[2.,2.]}
    monkeypatch.setattr(gpu,'probe',lambda *a:info)
    assert gpu.selected_packages(tmp_path) is None
    gpu.prepare(tmp_path)
    command=commands[0]
    assert '--target' in command and str(tmp_path/'gpu.pending') in command
    assert '--no-deps' in command and command[command.index('--index-url')+1]==gpu.CUDA_INDEX
    assert json.loads((tmp_path/'gpu/READY.json').read_text(encoding="utf-8"))==info
    assert gpu.selected_packages(tmp_path)==tmp_path/'gpu'
    gpu.prepare(tmp_path);assert len(commands)==1


def test_failed_gpu_probe_preserves_cpu_and_never_activates(tmp_path,monkeypatch):
    from volleyvision import portable_gpu as gpu
    from types import SimpleNamespace
    monkeypatch.setattr(gpu.shutil,'disk_usage',lambda *a:SimpleNamespace(free=10*1024**3))
    base=tmp_path/'runtime';base.mkdir();(base/'cpu.txt').write_text('preserve')
    monkeypatch.setattr(gpu,'_install',lambda *a:None)
    def failed(*a):raise RuntimeError('no CUDA')
    monkeypatch.setattr(gpu,'probe',failed)
    with pytest.raises(RuntimeError,match='CPU-сборка сохранена'):gpu.prepare(tmp_path)
    assert not (tmp_path/'gpu/READY.json').exists() and gpu.selected_packages(tmp_path) is None
    assert (base/'cpu.txt').read_text(encoding="utf-8")=='preserve'
    final=tmp_path/'gpu';final.mkdir();(final/'READY.json').write_text('{}')
    assert gpu.selected_packages(tmp_path) is None


def test_gpu_low_disk_abstains_before_download(tmp_path,monkeypatch):
    from volleyvision import portable_gpu as gpu
    from types import SimpleNamespace
    monkeypatch.setattr(gpu.shutil,'disk_usage',lambda *a:SimpleNamespace(free=1024))
    monkeypatch.setattr(gpu,'_install',lambda *a:pytest.fail('Low disk must not start a download'))
    with pytest.raises(ValueError,match='свободного места'):gpu.prepare(tmp_path)
    assert not (tmp_path/'gpu.pending').exists()


def test_public_package_omits_private_planning_and_pc_reports(tmp_path):
    from scripts.build_windows_portable import copy_public_docs
    repo=tmp_path/'repo';docs=repo/'docs';docs.mkdir(parents=True)
    packaging=repo/'packaging/windows';packaging.mkdir(parents=True)
    (packaging/'public-docs.txt').write_text('FEATURES.md\n',encoding='utf-8')
    (docs/'FEATURES.md').write_text('Public guide',encoding='utf-8')
    for name in ('PROJECT_SPEC.md','PROJECT_STATE.md','WORK_LOG.md','PC_report.txt'):
        (docs/name).write_text('Private metadata',encoding='utf-8')
    (docs/'validation').mkdir()
    (docs/'validation/target-pc-summary.json').write_text('{}',encoding='utf-8')
    destination=tmp_path/'out';destination.mkdir()
    (destination/'old-private.txt').write_text('stale',encoding='utf-8')
    copy_public_docs(repo,destination)
    assert {str(p.relative_to(destination)) for p in destination.rglob('*')}=={'FEATURES.md'}
    (docs/'validation/native-windows-ci.json').write_text('{"native_windows_ci":true}',encoding='utf-8')
    copy_public_docs(repo,destination)
    assert (destination/'validation/native-windows-ci.json').is_file()
    assert not (destination/'validation/target-pc-summary.json').exists()
