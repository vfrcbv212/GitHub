from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .media import sha256

MODEL_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn/COCO_V1"


def seed_bundled(root,include_ocr=True):
    """Copy verified shipped weights locally; preserve existing project models."""
    bundle=os.environ.get('VOLLEYVISION_BUNDLE_ROOT')
    if not bundle:return None
    source=Path(bundle)/'models';root=Path(root);root.mkdir(parents=True,exist_ok=True)
    supplied=manifest(source);current=manifest(root)
    for name in (('detector','ocr') if include_ocr else ('detector',)):
        if name in current:continue
        info=supplied.get(name)
        if not info:raise ValueError('В переносимой сборке отсутствует модель '+name)
        filename=info['file']
        if Path(filename).name!=filename:raise ValueError('Неверный путь модели в manifest')
        original=source/filename
        if not original.is_file() or sha256(original)!=info['sha256']:
            raise ValueError('Встроенная модель повреждена: '+name)
        target=root/filename
        if not target.exists():
            temporary=target.with_suffix('.bundled.partial');shutil.copyfile(original,temporary);temporary.replace(target)
        if sha256(target)!=info['sha256']:raise ValueError('Локальная модель повреждена: '+name)
        current[name]=info
    (root/'manifest.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    return current


def manifest(root):
    p = Path(root) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def fetch(root,include_ocr=True):
    """The ONLY operation allowed to download weights; ordinary analysis is offline."""
    bundled=seed_bundled(root,include_ocr)
    if bundled is not None:return bundled
    import torch
    import torchvision
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    url = W.COCO_V1.url
    target = root / Path(url).name
    prefix = target.stem.rsplit("-", 1)[-1]
    if not target.exists():
        temp = target.with_suffix(".partial")
        torch.hub.download_url_to_file(url, str(temp), hash_prefix=prefix)
        temp.replace(target)
    digest = sha256(target)
    if not digest.startswith(prefix):
        raise ValueError("Контрольная сумма весов TorchVision не совпадает")
    data = manifest(root)|{"detector": {"name": MODEL_NAME, "file": target.name, "url": url, "sha256": digest,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "code_license": "BSD-3-Clause", "weights_terms": "TorchVision pretrained-model notice; training-data terms apply",
            "training_data": "COCO; annotations CC-BY-4.0, images have individual rights; see docs/MODELS.md"}}
    if not include_ocr:
        (root/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        return data
    import easyocr
    easyocr.Reader(["en"], gpu=False, detector=False, model_storage_directory=str(root),
                   user_network_directory=str(root / "network"), download_enabled=True, verbose=False, quantize=False)
    from easyocr.config import recognition_models
    config = recognition_models["gen2"]["english_g2"]
    ocr_file = root / config["filename"]
    data["ocr"] = {"file": ocr_file.name, "url": config["url"], "sha256": sha256(ocr_file),
                   "upstream_md5": config["md5sum"], "version": "EasyOCR 1.7.2 english_g2",
                   "code_license": "Apache-2.0", "weights_license": "no separate license statement verified; not assumed unrestricted",
                   "training_data": "upstream recognition training mix; per-image rights not independently audited"}
    (root / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def verify(root, name):
    if name not in manifest(root):seed_bundled(root,include_ocr=name=='ocr')
    info = manifest(root).get(name)
    if not info:
        raise ValueError("Веса не подготовлены. Один раз выполните команду fetch-models с подключением к сети")
    path = Path(root) / info["file"]
    if not path.is_file() or sha256(path) != info["sha256"]:
        raise ValueError(f"Веса {name} отсутствуют или изменились; проверьте manifest.json")
    return info, path


def choose_device(requested="auto"):
    import torch
    if requested == "cpu":
        return "cpu", None
    if torch.cuda.is_available():
        try:
            (torch.ones(1, device="cuda") + 1).cpu()
            return "cuda", None
        except RuntimeError as exc:
            return "cpu", f"CUDA недоступна для вычислений: {exc}. Выбран CPU"
    return "cpu", "CUDA недоступна. Выбран CPU; обработка может быть медленнее"
