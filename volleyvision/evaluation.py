"""Explicit matching rules; no synthetic result is a real-world quality estimate."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def scores(tp,fp,fn):
    return {"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else None,
            "recall":tp/(tp+fn) if tp+fn else None,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def detections(truth,predicted,threshold=.5):
    result={}
    for category in (1,37):
        gt=[x for x in truth if x["class_id"]==category];pr=[x for x in predicted if x["class_id"]==category]
        cost=np.array([[1-iou(a["bbox"],b["bbox"]) if a["image_id"]==b["image_id"] else 1e6 for b in pr] for a in gt])
        matched=[]
        if gt and pr:
            aa,bb=linear_sum_assignment(cost)
            matched=[(a,b) for a,b in zip(aa,bb) if cost[a,b]<=1-threshold]
        result[str(category)]=scores(len(matched),len(pr)-len(matched),len(gt)-len(matched))
    return {"matching":"one-to-one Hungarian, same frame and class, IoU >= threshold", "iou_threshold":threshold,"classes":result,
            "metric":"precision/recall at one threshold, NOT COCO mAP"}


def events(truth,predicted,tolerance=.25):
    result={};field_matches=defaultdict(lambda:{"known":0,"correct":0,"unknown":0,"total":0})
    for kind in sorted({e["type"] for e in truth+predicted}):
        gt=[e for e in truth if e["type"]==kind];pr=[e for e in predicted if e["type"]==kind]
        cost=np.array([[abs(a["start"]-b["start"]) if a["source_id"]==b["source_id"] else 1e6 for b in pr] for a in gt])
        pairs=[]
        if gt and pr:
            aa,bb=linear_sum_assignment(cost)
            pairs=[(a,b) for a,b in zip(aa,bb) if cost[a,b]<=tolerance]
        result[kind]=scores(len(pairs),len(pr)-len(pairs),len(gt)-len(pairs))
        result[kind]["timing_abs_error_s"]=[float(cost[a,b]) for a,b in pairs]
        for a,b in pairs:
            for field in ("player_id","team_id","source_zone","target_zone","direction","outcome"):
                if gt[a].get(field) is None:continue
                m=field_matches[field];m["total"]+=1
                if pr[b].get(field) is None:m["unknown"]+=1
                else:m["known"]+=1;m["correct"]+=int(gt[a][field]==pr[b][field])
    for m in field_matches.values():
        m["accuracy_on_known"]=m["correct"]/m["known"] if m["known"] else None
        m["coverage_among_matched_events"]=m["known"]/m["total"] if m["total"] else None
    return {"matching":"one-to-one Hungarian, same source and event type, absolute start-time tolerance",
            "tolerance_s":tolerance,"events":result,"fields":dict(field_matches),
            "warning":"field accuracy is conditional on matched events; inspect event recall separately"}
