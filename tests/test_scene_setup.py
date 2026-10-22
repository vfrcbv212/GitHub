import copy

import numpy as np
import pytest

from volleyvision.scene_setup import TeamMemory,MainBall,save_scene,validate_scene
from volleyvision.store import Project


def scene():
    return {'source_id':'src','source_sha256':'sha','image_size':[640,360],'time':0,
            'entries':{'center':[[0,180],[640,180]],'team_a':[[320,280]],'team_b':[[320,80]]}}


def person(tid='p',bbox=None):
    return {'track_id':tid,'bbox':bbox or [200,200,250,290]}


def test_team_does_not_flip_during_jump_or_reassign_identity_on_new_track():
    memory=TeamMemory(scene())
    for t in (1,1.2,1.4):
        row=person();memory.update([row],t)
    assert row['team_proposal']['team_id']=='A' and row['team_proposal']['evidence']=='hypothesis'
    airborne=person(bbox=[200,20,250,150]);memory.update([airborne],1.6)
    assert airborne['team_proposal']['team_id']=='A'
    replacement=person('new',[200,20,250,150]);memory.update([replacement],1.8)
    assert replacement['team_proposal']['team_id'] is None and 'player_id' not in replacement


def test_manual_person_seed_survives_airborne_feet_and_overlapping_people_abstain():
    value=scene();value['entries']['person_a']=[[225,90]]
    memory=TeamMemory(value);row=person(bbox=[200,20,250,150]);memory.update([row],0)
    assert row['team_proposal']['team_id']=='A' and row['team_proposal']['evidence']=='confirmed'
    row=person(bbox=[200,20,250,150]);memory.update([row],1)
    assert row['team_proposal']['team_id']=='A'
    memory=TeamMemory(value);rows=[person('one',[200,20,250,150]),person('two',[200,20,250,150])]
    memory.update(rows,0);assert all(r['team_proposal']['team_id'] is None for r in rows)


def test_manual_correction_wins_and_referee_is_not_assigned_a_team():
    memory=TeamMemory(scene());row=person()
    for t in (1,1.2,1.4):memory.update([row],t)
    memory.update([row],1.6,{'p':{'role':'participant','team_id':'B'}})
    assert row['team_proposal']['team_id']=='B' and row['team_proposal']['evidence']=='confirmed'
    memory.update([row],1.8,{'p':{'role':'referee','team_id':None}})
    assert row['team_proposal']['team_id'] is None


def test_manual_referee_box_identifies_only_matching_track():
    value=scene();value['entries']['referee']=[[10,10],[50,130]]
    memory=TeamMemory(value);referee=person('ref',[10,10,50,130]);player=person('player',[200,200,250,290])
    memory.update([referee,player],0)
    assert referee['scene_role']=='referee' and referee['team_proposal']['team_id'] is None
    assert 'scene_role' not in player
    later=person('ref',[15,15,55,135]);memory.update([later],.2)
    assert later['scene_role']=='referee'


def test_setup_partial_contour_curved_net_antennas_and_source_validation(tmp_path):
    source={'id':'src','sha256':'sha','display_width':640,'display_height':360,'duration':4}
    entries=scene()['entries']|{'court':[[0,190],[640,190],[640,360],[0,360]],
        'net':[[10,90],[300,100],[630,90],[630,170],[300,180],[10,170]],
        'antenna_a':[[10,90],[10,50]],'ball':[[100,100],[110,100]]}
    with Project(tmp_path/'project',create=True) as project:
        project.put('source',source)
        saved=save_scene(project,source,np.zeros((360,640,3),np.uint8),0,entries)
        assert len(saved['entries']['net'])==6 and saved['metric_verified'] is False
        profile=project.get('calibration','scene-src')
        assert profile['matrix'] is None and profile['geometry']=='video' and profile['net_region']==entries['net']
        assert (project.root/profile['reference_image']).exists()
        bad=copy.deepcopy(saved);bad['source_sha256']='other'
        with pytest.raises(ValueError,match='другому видео'):validate_scene(bad,source)
        bad=copy.deepcopy(saved);bad['entries']['antenna_a'].append([30,40])
        with pytest.raises(ValueError,match='ровно две'):validate_scene(bad,source)


def test_manual_ball_seed_selects_one_ball_and_never_fills_missing_observations():
    value=scene();value['entries']['ball']=[[100,100],[110,100]];selector=MainBall(value)
    a={'track_id':'ball1','bbox':[95,95,105,105]};b={'track_id':'ball2','bbox':[200,200,210,210]}
    selected,info=selector.select([a,b],0)
    assert selected==[a] and info['state']=='manual_seed_matched'
    assert selector.select([],1)[0]==[]
    assert selector.select([b,dict(b,track_id='ball3')],1.2)[1]['state']=='ambiguous'
    assert selector.select([b],1.3)[0]==[] and selector.select([b],1.4)[0]==[]
    selected,info=selector.select([b],1.5)
    assert selected==[b] and info['state']=='return_hypothesis' and info['same_physical_ball_verified'] is False


def test_scene_team_memory_is_used_by_analysis_and_saved_with_observations(tmp_path,monkeypatch):
    from volleyvision import vision
    from volleyvision.demo import make_video
    from volleyvision.media import import_video,frame_at
    class Detector:
        device='cpu';warning=None
        def __init__(self,*a):self.calls=0
        def predict(self,*a):
            self.calls+=1
            return [{'bbox':[200,200,250,290] if self.calls<=2 else [200,20,250,150],'class_id':1,'score':.9}]
        def balls(self,*a):return []
    class Tracker:
        def __init__(self,params,w,h,prefix):self.prefix=prefix
        def update(self,detections,time_s,kind):return [r|{'track_id':self.prefix+kind,'observed':True} for r in detections]
    class Guard:
        def __init__(self,*a):pass
        def check(self,*a):return {'state':'stable','reason':'test'}
    monkeypatch.setattr(vision,'Detector',Detector);monkeypatch.setattr(vision,'Trackers',Tracker);monkeypatch.setattr(vision,'CameraGuard',Guard)
    monkeypatch.setattr(vision,'verify',lambda *a:({'sha256':'test'},None))
    with Project(tmp_path/'project',create=True) as project:
        path=project.root/'media/test.mp4';make_video(path,seconds=2,vfr=False);source=import_video(project,path)
        entries=scene()['entries']|{'person_a':[[225,250]]}
        save_scene(project,source,frame_at(path,source,0).rgb,0,entries)
        job=vision.analyze(project,source['id'],{'identity_mode':'appearance','people_fps':2,'ball_fps':2,'ocr':False,'ball_tiled':False})
        observed=[p for row in project.observations(job['id']) for p in row['people']]
        assert len(observed)>=3 and all(p['team_proposal']['team_id']=='A' for p in observed)
        assert observed[-1]['bbox'][3]<180 and observed[-1]['team_proposal']['evidence']=='confirmed'
        assert job['params']['scene_setup']['version']==1 and 'ball_inference_s' in job['timings']
