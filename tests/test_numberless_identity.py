import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import numpy as np
import pytest
from PySide6 import QtWidgets as W

from volleyvision.analytics import report
from volleyvision.gallery import GalleryDialog,assign_interval
from volleyvision.identity import second_pass
from volleyvision.person_memory import enrich
from volleyvision.roster import save_player,player_label,set_identity_mode,identity_mode
from volleyvision.setup_ui import RosterDialog,KitDialog
from volleyvision.store import Project


@pytest.fixture
def project(tmp_path):
    with Project(tmp_path/'p',create=True,synthetic=True) as p:
        for tid in ('A','B'):p.put('team',{'id':tid,'name':tid})
        p.put('source',{'id':'video','path':'missing.mp4','sha256':'hash','duration':100,'offset':0,
            'width':100,'height':100,'display_width':100,'display_height':100})
        p.put('job',{'id':'run','source_id':'video','state':'complete'})
        yield p


def track(project,tid,start,end,vector=(1,0,0),version='test-v1',run_id='run',number=None):
    return project.put('track',{'id':tid,'run_id':run_id,'source_id':'video','start':start,'end':end,
        'samples':[],'ocr_observations':[],'number_consensus':{'number':number},
        'person_samples':[{'path':f'cache/{tid}-{i}.jpg','time':t,'embedding':list(vector),'embedding_version':version}
                          for i,t in enumerate(np.linspace(start,end,3).tolist())]})


def hypothesis(p,tid='query'):
    return p.get('identity_hypothesis','run:'+tid)


def test_multiple_numberless_cards_stay_distinct_and_survive_reopen(project):
    first=save_player(project,'A');second=save_player(project,'A')
    assert first['id']!=second['id'] and first['number'] is None and second['number'] is None
    assert first['label']!=second['label'] and 'None' not in player_label(first)
    changed=save_player(project,'A',name='Миша',role='setter',player_id=first['id'])
    assert changed['id']==first['id'] and changed['number'] is None
    save_player(project,'A',7);save_player(project,'B',7)
    with pytest.raises(ValueError,match='уже есть'):save_player(project,'A',7)
    data=report(project);assert len(data['players'])==4 and all('None' not in p['name'] for p in data['players'])
    set_identity_mode(project,'appearance')
    with Project(project.root) as reopened:
        assert identity_mode(reopened)=='appearance' and reopened.get('player',first['id'])['name']=='Миша'


def test_person_memory_suggests_reappearance_without_any_numbers(project):
    player=save_player(project,'A',name='Первый');track(project,'reference',0,1);track(project,'query',5,6,vector=(.99,.01,0))
    assign_interval(project,'reference',0,1,player['id']);set_identity_mode(project,'appearance')
    before=project.all('track_binding');second_pass(project,'run')
    result=hypothesis(project)
    assert result['mode']=='appearance' and result['proposed_player_id']==player['id']
    assert result['candidates'][0]['number_agreement'] is None and not result['automatic_binding']
    assert project.all('track_binding')==before and report(project)['events']==[]
    with Project(project.root) as reopened:
        second_pass(reopened,'run')
        assert hypothesis(reopened)['proposed_player_id']==player['id']
    # Long gaps have no invented trajectory, while the image memory still exists.
    track(project,'later',90,91);second_pass(project,'run')
    assert hypothesis(project,'later')['proposed_player_id']==player['id']


def test_similar_people_and_temporal_conflicts_abstain_and_corrections_refresh(project):
    a=save_player(project,'A');b=save_player(project,'A')
    track(project,'ref-a',0,1);track(project,'ref-b',0,1);track(project,'query',3,4)
    assign_interval(project,'ref-a',0,1,a['id']);assign_interval(project,'ref-b',0,1,b['id'])
    second_pass(project,'run',mode='appearance')
    assert hypothesis(project)['proposed_player_id'] is None and len(hypothesis(project)['candidates'])==2
    # Remove b's effective memory through a later interval correction.
    assign_interval(project,'ref-b',0,1,role='unknown');second_pass(project,'run',mode='appearance')
    assert hypothesis(project)['proposed_player_id']==a['id']
    track(project,'occupied',3,4);assign_interval(project,'occupied',3,4,a['id'])
    second_pass(project,'run',mode='appearance');assert hypothesis(project)['proposed_player_id'] is None
    assign_interval(project,'occupied',3,4,role='unknown');second_pass(project,'run',mode='appearance')
    assert hypothesis(project)['proposed_player_id']==a['id']
    assign_interval(project,'ref-a',0,1,role='unknown');second_pass(project,'run',mode='appearance')
    assert hypothesis(project)['candidates']==[]


def test_appearance_ignores_ocr_and_hybrid_reports_conflict(project):
    a=save_player(project,'A',7);track(project,'ref',0,1);track(project,'query',3,4,number=99)
    assign_interval(project,'ref',0,1,a['id'])
    second_pass(project,'run',mode='appearance');assert hypothesis(project)['proposed_player_id']==a['id']
    second_pass(project,'run',mode='hybrid');assert hypothesis(project)['proposed_player_id'] is None
    second_pass(project,'run',mode='numbers');assert hypothesis(project)['candidates']==[]
    # Different descriptor contracts never silently compare incompatible vectors.
    track(project,'query',3,4,version='other-model');second_pass(project,'run',mode='appearance')
    assert hypothesis(project)['candidates']==[]


def test_bindings_only_create_memory_inside_confirmed_sample_times(project):
    player=save_player(project,'A');track(project,'ref',0,10);track(project,'query',20,21)
    # Samples exist only at 0,5,10; this interval confirms none of them.
    assign_interval(project,'ref',1,2,player['id']);second_pass(project,'run',mode='appearance')
    assert hypothesis(project)['candidates']==[]
    assign_interval(project,'ref',5,5,player['id']);second_pass(project,'run',mode='appearance')
    assert hypothesis(project)['proposed_player_id']==player['id']


def test_reference_memory_across_runs_and_incremental_extraction_cancel(project,tmp_path):
    import cv2
    from volleyvision.media import Cancelled
    player=save_player(project,'A');track(project,'old-ref',0,1,run_id='old-run')
    assign_interval(project,'old-ref',0,1,player['id']);track(project,'query',0,1)
    second_pass(project,'run',mode='appearance')
    assert hypothesis(project)['proposed_player_id']==player['id']
    source=track(project,'fresh',10,11)
    for s in source['person_samples']:
        s.pop('embedding');path=project.root/s['path'];path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path),np.full((100,40,3),128,np.uint8))
    project.put('track',source)
    calls=[]
    class Encoder:
        person_embedding_version='test-v1'
        def person_embedding(self,rgb):calls.append(1);return [1,0,0]
    stopped=[False]
    with pytest.raises(Cancelled):
        enrich(project,'run',Encoder(),cancel=lambda:stopped[0],progress=lambda *a:stopped.__setitem__(0,True))
    assert len(calls)==1
    enrich(project,'run',Encoder());assert len(calls)==3
    enrich(project,'run',Encoder());assert len(calls)==3


def test_roster_dialog_numberless_edit_gallery_and_mode_selection(project):
    from volleyvision.gui import MainWindow
    app=W.QApplication.instance() or W.QApplication([])
    dialog=RosterDialog(project);dialog.name.setText('Первый');dialog.save();dialog.name.setText('Второй');dialog.save()
    players=project.all('player');assert len(players)==2 and all(p['number'] is None for p in players)
    dialog.table.selectRow(0);dialog.name.setText('Исправлен');dialog.save()
    assert len(project.all('player'))==2 and project.get('player',players[0]['id'])['name']=='Исправлен'
    set_identity_mode(project,'appearance');track(project,'ref',0,1)
    gallery=GalleryDialog(project);assert 'None' not in gallery.player_label(players[0]['id'])
    assert 'номера не используются' in gallery.evidence.toPlainText().lower()
    kit=KitDialog(project,project.get('source','video'),np.zeros((100,100,3),np.uint8))
    assert all('None' not in kit.player.itemText(i) for i in range(kit.player.count()))
    # MainWindow would try to decode the deliberately missing fixture source.
    project.db.execute("DELETE FROM objects WHERE kind='source'");project.db.commit()
    window=MainWindow(project.root);assert window.identification.currentData()=='appearance'
    window.identification.setCurrentIndex(window.identification.findData('hybrid'))
    assert identity_mode(project)=='hybrid'
    window.close();kit.close();gallery.close();dialog.close();app.processEvents()


def test_analysis_mode_change_reuses_detections_and_appearance_never_initializes_ocr(tmp_path,monkeypatch):
    from volleyvision import vision
    from volleyvision.demo import make_video
    from volleyvision.media import import_video
    import volleyvision.identity as identity_module
    with Project(tmp_path/'p',create=True,synthetic=True) as p:
        video=p.root/'media/v.mp4';make_video(video,seconds=2,vfr=False);source=import_video(p,video)
        monkeypatch.setattr(vision,'verify',lambda *a:({'sha256':'test-detector'},None))
        calls={'detection':0,'ocr':0}
        class Detector:
            device='cpu';warning=None;person_embedding_version='test-v1'
            def __init__(self,*a):pass
            def predict(self,*a):calls['detection']+=1;return [{'bbox':[140,100,210,270],'class_id':1,'score':.99}]
            def balls(self,*a):return []
            def person_embedding(self,image):return [1.,0.,0.]
        def forbidden(*a,**kw):raise AssertionError('appearance mode initialized OCR')
        monkeypatch.setattr(vision,'Detector',Detector);monkeypatch.setattr(vision,'NumberOCR',forbidden);monkeypatch.setattr(identity_module,'NumberOCR',forbidden)
        config={'identity_mode':'appearance','ocr':True,'people_fps':2,'ball_fps':2,'device':'cpu','ball_tiled':False}
        first=vision.analyze(p,source['id'],config);before=calls['detection']
        assert p.all('track')[0]['person_samples'][0]['embedding']==[1.,0.,0.]
        assert p.all('track')[0]['ocr_observations']==[]
        class OCR:
            def __init__(self,*a):calls['ocr']+=1
            def read(self,image):return {'number':7,'sharpness':100,'raw_score':.9}
        monkeypatch.setattr(identity_module,'NumberOCR',OCR)
        changed=vision.analyze(p,source['id'],config|{'identity_mode':'numbers'})
        assert changed['id']==first['id'] and calls['detection']==before and calls['ocr']==1
        assert any(t.get('number_consensus',{}).get('number')==7 for t in p.all('track'))
        monkeypatch.setattr(identity_module,'NumberOCR',forbidden)
        again=vision.analyze(p,source['id'],config)
        assert again['id']==first['id'] and calls['detection']==before
