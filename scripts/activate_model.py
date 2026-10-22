import argparse,json,shutil
from pathlib import Path
from volleyvision.media import sha256

p=argparse.ArgumentParser();p.add_argument("weights",type=Path);p.add_argument("model_dir",type=Path);a=p.parse_args()
manifest=a.model_dir/"manifest.json";data=json.loads(manifest.read_text(encoding="utf-8"));digest=sha256(a.weights)
backup=a.model_dir/f"manifest-before-{digest[:12]}.json";shutil.copyfile(manifest,backup)
target=a.model_dir/f"custom-{digest[:12]}.pth";shutil.copyfile(a.weights,target)
data["detector"].update(file=target.name,sha256=digest,parent_sha256=data["detector"]["sha256"],url=None,name="local fine-tuned Faster R-CNN")
manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"Для отката восстановите {backup.name} как manifest.json. Исходные веса сохранены.")
