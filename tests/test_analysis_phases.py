import pytest

from volleyvision.demo import make_video
from volleyvision.media import import_video
from volleyvision.store import Project
from volleyvision import vision


def test_limited_setup_is_not_completed_match_and_full_run_resumes(tmp_path,monkeypatch):
    with Project(tmp_path/'project',create=True,synthetic=True) as project:
        video=project.root/'media/test.mp4';make_video(video,seconds=2,vfr=False)
        source=import_video(project,video)
        project.put('calibration',{'id':'old-profile','source_id':source['id'],'source_sha256':'different-original',
                                  'geometry':'video','people_roi':[[0,0],[20,0],[20,20],[0,20]]})
        monkeypatch.setattr(vision,'verify',lambda *args:({'sha256':'test-detector'},None))
        class Detector:
            device='cpu';warning=None
            def __init__(self,*args):pass
            def predict(self,*args):return []
            def balls(self,*args):return []
        monkeypatch.setattr(vision,'Detector',Detector)
        params={'people_fps':1,'ball_fps':2,'ocr':False,'ball_tiled':False,'setup_end':.5,'device':'cpu'}
        first=vision.analyze(project,source['id'],params,setup_only=True)
        assert first['state']=='setup_complete'
        assert project.get('geometry_warning','invalid:old-profile') is not None
        count=project.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        again=vision.analyze(project,source['id'],params,setup_only=True)
        assert again['samples']==first['samples']
        final=vision.analyze(project,source['id'],params,setup_only=False)
        assert final['state']=='complete' and final['id']==first['id']
        assert final['last_time']>.5 and final['processed_until']==pytest.approx(source['duration'])
        assert project.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]>count
        assert (project.root/'exports/automatic/report.html').exists()


def test_camera_recovery_starts_new_track_segment(tmp_path,monkeypatch):
    import cv2
    import numpy as np
    with Project(tmp_path/'project',create=True,synthetic=True) as project:
        video=project.root/'media/test.mp4';make_video(video,seconds=5,vfr=False)
        source=import_video(project,video)
        ref=project.root/'cache/reference.jpg';cv2.imwrite(str(ref),np.zeros((360,640,3),np.uint8))
        project.put('calibration',{'id':'view','source_id':source['id'],'source_sha256':source['sha256'],
            'geometry':'video','image_size':[source['display_width'],source['display_height']],
            'reference_image':project.relpath(ref),'start':0,'end':5})
        monkeypatch.setattr(vision,'verify',lambda *args:({'sha256':'test-detector'},None))
        class Detector:
            device='cpu';warning=None
            def __init__(self,*args):pass
            def predict(self,*args):return [{'bbox':[50,100,100,200],'score':.9,'class_id':1}]
            def balls(self,*args):return []
        class Guard:
            def __init__(self,*args):self.calls=0
            def check(self,*args):
                state=('stable','shifted','stable')[min(self.calls,2)];self.calls+=1
                return {'state':state,'reason':state}
        class Tracker:
            def __init__(self,params,w,h,prefix):self.prefix=prefix
            def update(self,detections,t,kind):return [d|{'track_id':self.prefix+'p1','observed':True} for d in detections]
        monkeypatch.setattr(vision,'Detector',Detector);monkeypatch.setattr(vision,'CameraGuard',Guard);monkeypatch.setattr(vision,'Trackers',Tracker)
        job=vision.analyze(project,source['id'],{'people_fps':1,'ball_fps':1,'ocr':False,'ball_tiled':False,'device':'cpu'})
        observations=list(project.observations(job['id']))
        tids={round(o['time']):o['people'][0]['track_id'] for o in observations}
        assert len({tids[0],tids[2],tids[4]})==3 and job['segments']==3
        sessions=project.all('camera_session');assert len(sessions)==1 and sessions[0]['end']==pytest.approx(4)
