"""Fine-tune the connected detector from reviewed COCO boxes; resumable epochs."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms.functional import pil_to_tensor

from volleyvision.evaluation import detections
from volleyvision.media import sha256
from volleyvision.models import choose_device,verify


class Coco(Dataset):
    def __init__(self,path):
        self.root=path.parent;self.data=json.loads(path.read_text(encoding="utf-8"));self.images=self.data["images"]
        self.annotations=defaultdict(list)
        for a in self.data["annotations"]:
            if a.get("reviewed") is False:raise ValueError("В разметке остались неподтверждённые предложения; сначала проверьте все кадры")
            if a["category_id"] not in (1,37):raise ValueError("Сохраните COCO category_id person=1, sports ball=37")
            self.annotations[a["image_id"]].append(a)
        if any(not i.get("match_id") for i in self.images):raise ValueError("У каждого изображения нужен match_id для защиты от утечки")

    def __len__(self):return len(self.images)

    def __getitem__(self,i):
        image=self.images[i];rgb=Image.open(self.root/image["file_name"]).convert("RGB")
        annotations=self.annotations[image["id"]];boxes=[];labels=[]
        for a in annotations:
            x,y,w,h=a["bbox"]
            if w<=0 or h<=0:raise ValueError("Вырожденная рамка")
            boxes.append([x,y,x+w,y+h]);labels.append(a["category_id"])
        return pil_to_tensor(rgb).float()/255,{"boxes":torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                                             "labels":torch.tensor(labels,dtype=torch.int64),"image_id":torch.tensor(image["id"])}


def evaluate(model,dataset,device):
    model.eval();truth=[];pred=[]
    with torch.inference_mode():
        for i in range(len(dataset)):
            image,target=dataset[i];out=model([image.to(device)])[0];image_id=int(target["image_id"])
            for b,c in zip(target["boxes"],target["labels"]):truth.append({"image_id":image_id,"class_id":int(c),"bbox":b.tolist()})
            for b,c,s in zip(out["boxes"].cpu(),out["labels"].cpu(),out["scores"].cpu()):
                if c in (1,37) and s>=.5:pred.append({"image_id":image_id,"class_id":int(c),"bbox":b.tolist()})
    return detections(truth,pred)


def main():
    p=argparse.ArgumentParser();p.add_argument("train",type=Path);p.add_argument("validation",type=Path);p.add_argument("model_dir",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--epochs",type=int,default=5);p.add_argument("--resume",action="store_true");p.add_argument("--device",default="auto");p.add_argument("--lr",type=float,default=.0001)
    a=p.parse_args();random.seed(42);np.random.seed(42);torch.manual_seed(42)
    train,val=Coco(a.train),Coco(a.validation)
    if {i["match_id"] for i in train.images}&{i["match_id"] for i in val.images}:raise ValueError("Матчи train и validation пересекаются")
    if not len(train) or not len(val):raise ValueError("Нужны непустые train и validation из разных матчей")
    device,warning=choose_device(a.device)
    if warning:print(warning)
    a.out.mkdir(parents=True,exist_ok=True);model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=None,weights_backbone=None)
    info,path=verify(a.model_dir,"detector");model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr);start=0
    fingerprints={"train":sha256(a.train),"validation":sha256(a.validation)}
    if a.resume:
        checkpoint=torch.load(a.out/"checkpoint.pth",map_location="cpu",weights_only=False)
        if checkpoint["fingerprints"]!=fingerprints:raise ValueError("Разметка изменилась с момента checkpoint; начните новый эксперимент")
        model.load_state_dict(checkpoint["model"]);optimizer.load_state_dict(checkpoint["optimizer"]);start=checkpoint["epoch"]+1
        torch.set_rng_state(checkpoint["torch_rng"]);random.setstate(checkpoint["python_rng"]);np.random.set_state(checkpoint["numpy_rng"])
        if device=="cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=0,collate_fn=lambda batch:tuple(zip(*batch)))
    for epoch in range(start,a.epochs):
        model.train();loss_sum=0
        for images,targets in loader:
            loss=sum(model([x.to(device) for x in images],[{k:v.to(device) for k,v in t.items()} for t in targets]).values())
            if not torch.isfinite(loss):raise ValueError("Неконечная функция потерь; проверьте разметку")
            optimizer.zero_grad();loss.backward();optimizer.step();loss_sum+=float(loss.detach())
        # Checkpoint before potentially lengthy evaluation. Crash resumes at next epoch.
        checkpoint={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"fingerprints":fingerprints,
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else None,
            "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"parent_sha256":info["sha256"]}
        temp=a.out/"checkpoint.partial";torch.save(checkpoint,temp);temp.replace(a.out/"checkpoint.pth")
        torch.save(model.state_dict(),a.out/f"epoch-{epoch:03}.pth")
        result={"epoch":epoch,"train_loss_mean":loss_sum/len(loader),"validation":evaluate(model,val,device)}
        (a.out/f"epoch-{epoch:03}.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result))
    print("Новая версия не активирована автоматически. Сравните на отдельном тестовом матче; сохраните прежний manifest.json для отката.")


if __name__=="__main__":main()
