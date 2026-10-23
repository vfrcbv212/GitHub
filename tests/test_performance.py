import json
import sqlite3

import pytest

from volleyvision.performance import processing_diagnostics,describe_performance,write_diagnostics
from volleyvision.store import Project


def test_diagnostics_distinguish_cancelled_interval_and_latest_run_without_raw_data(tmp_path):
    with Project(tmp_path/'project',create=True) as project:
        with pytest.raises(ValueError,match='обработку'):write_diagnostics(project)
        project.put('source',{'id':'v','duration':540,'path':'private-video.mp4'})
        project.put('job',{'id':'old','source_id':'v','state':'complete','elapsed_s':100,'processed_until':540})
        project.put('job',{'id':'new','source_id':'v','state':'cancelled','last_time':30,'elapsed_s':60,
            'device':'cpu','timings':{'ball_inference_s':40,'storage_s':.2}})
        rows=processing_diagnostics(project)
        assert len(rows)==1 and rows[0]['id']=='new' and rows[0]['speed_x']==.5
        assert rows[0]['processed_video_s']==30
        assert processing_diagnostics(project,'missing')==[]
        text=describe_performance(rows[0]);assert 'остановлено' in text and 'Поиск мяча: 40.0 с' in text
        path=write_diagnostics(project);saved=json.loads(path.read_text(encoding='utf-8'))
        assert saved['runs']==rows and 'private-video' not in path.read_text(encoding='utf-8')
        project.put('job',{'id':'legacy','source_id':'v','state':'complete'})
        assert processing_diagnostics(project)[0]['speed_x'] is None


def test_detection_checkpoint_commits_all_tracks_and_rolls_back_as_one_unit(tmp_path):
    with Project(tmp_path/'project',create=True) as project:
        project.put('job',{'id':'run','source_id':'v','last_time':0})
        job={'id':'run','source_id':'v','last_time':.2}
        obs={'time':.2,'people':[{'track_id':'p1'},{'track_id':'p2'}],'balls':[]}
        tracks=[{'id':'p1','end':.2},{'id':'p2','end':.2}]
        project.db.execute("CREATE TRIGGER fail_observation BEFORE INSERT ON observations BEGIN SELECT RAISE(ABORT,'test failure'); END")
        project.db.commit()
        with pytest.raises(sqlite3.IntegrityError,match='test failure'):project.record_observation(job,obs,tracks)
        assert project.get('track','p1') is None and project.get('job','run')['last_time']==0
        assert list(project.observations('run'))==[]
        project.db.execute('DROP TRIGGER fail_observation');project.db.commit()
        project.record_observation(job,obs,tracks)
        with Project(project.root) as reader:
            assert reader.get('track','p1')['end']==reader.get('job','run')['last_time']==.2
            assert reader.get('track','p2')['end']==.2 and list(reader.observations('run'))==[obs]
