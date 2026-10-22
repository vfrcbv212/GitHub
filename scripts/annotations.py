"""Export candidate boxes to editable COCO; never label proposals as ground truth."""
import argparse
import json
from pathlib import Path

import cv2

from volleyvision.media import frame_at
from volleyvision.store import Project


def main():
    p=argparse.ArgumentParser();p.add_argument("project",type=Path);p.add_argument("run_id");p.add_argument("destination",type=Path);p.add_argument("--every",type=float,default=2)
    a=p.parse_args();a.destination.mkdir(parents=True,exist_ok=True);images=[];annotations=[];last=-1e9
    with Project(a.project) as project:
        job=project.get("job",a.run_id);source=project.get("source",job["source_id"])
        for obs in project.observations(a.run_id):
            if obs["time"]-last<a.every:continue
            last=obs["time"];f=frame_at(project.resolve(source),source,obs["time"])
            name=f"frame-{len(images):06}.jpg";cv2.imwrite(str(a.destination/name),cv2.cvtColor(f.rgb,cv2.COLOR_RGB2BGR))
            image_id=len(images)+1;images.append({"id":image_id,"file_name":name,"width":f.rgb.shape[1],"height":f.rgb.shape[0],
                "match_id":str(project.root.name),"view_id":source["id"],"source_time":f.time,"source_pts":f.pts})
            for d in obs["people"]+obs["balls"]:
                x1,y1,x2,y2=d["bbox"];annotations.append({"id":len(annotations)+1,"image_id":image_id,"category_id":d["class_id"],
                    "bbox":[x1,y1,x2-x1,y2-y1],"area":(x2-x1)*(y2-y1),"iscrowd":0,"reviewed":False,"origin":"model proposal"})
    data={"images":images,"annotations":annotations,"categories":[{"id":1,"name":"person"},{"id":37,"name":"sports ball"}],
          "info":{"warning":"PROPOSALS ONLY; review every frame, add missed boxes, delete false positives before training"}}
    (a.destination/"annotations.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
