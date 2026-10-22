import threading
from types import SimpleNamespace

import numpy as np
import pytest

from volleyvision.media import Cancelled,prefetch_frames
from volleyvision.vision import Detector,DEFAULTS


def test_batched_inference_preserves_predictions_sizes_and_order():
    import torch
    class Model:
        transform=SimpleNamespace()
        def __init__(self):self.calls=[]
        def __call__(self,images):
            self.calls.append((len(images),self.transform.min_size,self.transform.max_size))
            return [{'boxes':torch.tensor([[float(im[0,0,0]*255),2,8,9],[1,2,3,4]]),
                     'labels':torch.tensor([37,1]),'scores':torch.tensor([.8,.99])} for im in images]
    detector=object.__new__(Detector);detector.device='cpu';detector.model=Model()
    images=[np.full((h,w,3),n,np.uint8) for n,h,w in ((1,80,90),(2,70,80),(3,80,90),(4,80,90))]
    sequential=[detector.predict(im,.5,(37,)) for im in images];detector.model.calls=[]
    batched=detector.predict_many(images,.5,(37,),batch_size=3)
    assert batched==sequential
    assert detector.model.calls==[(3,(80,),90),(1,(70,),80)]
    assert [r[0]['bbox'][0] for r in batched]==pytest.approx([1,2,3,4])


def test_tile_batch_offsets_nms_and_roi_skip_outside_tiles():
    detector=object.__new__(Detector);detector.device='cpu';seen=[]
    def predict(images,threshold,classes,batch_size):
        seen.append((len(images),batch_size))
        return [[{'bbox':[10,20,30,40],'class_id':37,'score':.9}] for image in images]
    detector.predict_many=predict
    params=DEFAULTS|{'ball_tile':200,'ball_overlap':0.,'inference_batch':2}
    image=np.zeros((400,600,3),np.uint8)
    assert len(detector.balls(image,params))==6 and seen[-1]==(6,2)
    result=detector.balls(image,params,[[405,5],[450,5],[450,60],[405,60]])
    assert seen[-1]==(1,2) and result[0]['bbox']==[410,20,430,40]


def test_prefetch_keeps_order_propagates_failure_and_closes_producer():
    closed=threading.Event()
    def source():
        try:
            yield 1;yield 2;raise ValueError('decoder failed')
        finally:closed.set()
    rows=prefetch_frames(source());assert next(rows)==1 and next(rows)==2
    with pytest.raises(ValueError,match='decoder failed'):next(rows)
    assert closed.wait(1)


def test_prefetch_cancellation_and_early_close_do_not_leave_workers():
    def run(cancel):
        closed=threading.Event()
        def source():
            try:
                for i in range(10000):yield i
            finally:closed.set()
        rows=prefetch_frames(source(),depth=1,cancel=cancel)
        return rows,closed
    rows,closed=run(None);assert next(rows)==0;rows.close();assert closed.wait(1)
    cancelled=threading.Event();rows,closed=run(cancelled.is_set)
    assert next(rows)==0;cancelled.set()
    with pytest.raises(Cancelled):next(rows)
    assert closed.wait(1)
