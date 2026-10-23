import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import pytest
from PySide6 import QtCore as C,QtWidgets as W,QtTest

from volleyvision.demo import make_video
from volleyvision.media import import_video,frame_at
from volleyvision.store import Project
from volleyvision.simple_ui import SimpleWindow
from volleyvision.quick_setup_ui import SceneDialog


@pytest.fixture(scope='module')
def app():
    app=W.QApplication.instance() or W.QApplication([])
    yield app


def prepared(tmp_path):
    root=tmp_path/'project'
    with Project(root,create=True) as project:
        path=project.root/'media/test.mp4';make_video(path,seconds=.4,vfr=False);source=import_video(project,path)
    return root,source


def test_simple_start_hides_advanced_editors_and_can_restore_them(tmp_path,app):
    root,source=prepared(tmp_path);window=SimpleWindow(root);window.show();app.processEvents()
    visible=lambda:{window.tabs.tabText(i) for i in range(window.tabs.count()) if window.tabs.isTabVisible(i)}
    assert visible()=={'Начало','Отчёт'} and window.identification.isVisible() and not window.filters.isVisible()
    window.advanced_button.click();assert 'Настройка' in visible() and window.filters.isVisible() is False
    # Filter belongs to the report page; it becomes visible when that page is selected.
    window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count()) if window.tabs.tabText(i)=='Отчёт'))
    assert window.filters.isVisible()
    window.advanced_button.click();assert visible()=={'Начало','Отчёт'}
    window.close()


def test_scene_points_can_be_dragged_and_saved_without_full_court(tmp_path,app):
    root,source=prepared(tmp_path)
    with Project(root) as project:
        rgb=frame_at(project.resolve(source),source,0).rgb;dialog=SceneDialog(project,source,rgb)
        dialog.show();app.processEvents();canvas=dialog.canvas
        for point in ([10,200],[600,200],[600,350],[10,350]):canvas.add_point(list(point))
        start=canvas.mapFromScene(C.QPointF(10*canvas.ratio,200*canvas.ratio))
        end=canvas.mapFromScene(C.QPointF(30*canvas.ratio,210*canvas.ratio))
        QtTest.QTest.mousePress(canvas.viewport(),C.Qt.MouseButton.LeftButton,pos=start)
        QtTest.QTest.mouseMove(canvas.viewport(),end)
        QtTest.QTest.mouseRelease(canvas.viewport(),C.Qt.MouseButton.LeftButton,pos=end)
        assert canvas.entries['court'][0][0]==pytest.approx(30,abs=3)
        dialog.save();saved=project.get('scene_setup',source['id'])
        assert saved and saved['metric_verified'] is False and len(saved['entries']['court'])==4
        reopened=SceneDialog(project,source,rgb);assert reopened.entries==saved['entries'];reopened.reject()


def test_each_new_import_offers_optional_scene_setup(tmp_path,app,monkeypatch):
    root,source=prepared(tmp_path);window=SimpleWindow(root);called=[]
    monkeypatch.setattr(window,'scene_setup',lambda:called.append(window.source['id']))
    window._import_before={source['id']}
    path=root/'media/other.mp4';make_video(path,seconds=.6,vfr=True);other=import_video(window.project,path)
    window.done('Импорт завершён');app.processEvents()
    assert called==[other['id']] and window.source['id']==other['id']
    window.close()


def test_multiple_imports_offer_each_file_and_failure_does_not_leak_import_state(tmp_path,app,monkeypatch):
    root,source=prepared(tmp_path);window=SimpleWindow(root);called=[];errors=[]
    monkeypatch.setattr(window,'scene_setup',lambda:called.append(window.source['id']))
    monkeypatch.setattr(window,'error',lambda text:errors.append(text))
    window._import_before={source['id']}
    for name,seconds in (('second',.6),('third',.8)):
        path=root/f'media/{name}.mp4';make_video(path,seconds=seconds,vfr=False);import_video(window.project,path)
    added=[s['id'] for s in window.project.all('source') if s['id']!=source['id']]
    # A later file failed, but successfully imported parts still need optional setup.
    window.failed('Cannot open next part')
    for _ in range(5):app.processEvents()
    assert errors==['Cannot open next part'] and called==added
    assert not window._scene_queue and window._import_before is None
    window.done('Другая операция');app.processEvents();assert called==added
    window.close()


def test_performance_panel_follows_selected_video(tmp_path,app):
    root,source=prepared(tmp_path);window=SimpleWindow(root)
    window.project.put('job',{'id':'j','source_id':source['id'],'state':'complete','device':'cpu',
        'processed_until':source['duration'],'elapsed_s':2,'timings':{'storage_s':.1}})
    window.refresh();assert 'CPU' in window.performance.text() and 'Сохранение: 0.1' in window.performance.text()
    window.save_performance();assert (root/'exports/diagnostics/processing.json').exists()
    path=root/'media/new.mp4';make_video(path,seconds=.8,vfr=False);new=import_video(window.project,path)
    window.refresh();window.sources.setCurrentIndex(window.sources.findData(new['id']))
    assert 'появятся после обработки' in window.performance.text()
    window.close()


def test_closing_window_cancels_deferred_scene_prompts(tmp_path,app,monkeypatch):
    root,source=prepared(tmp_path);window=SimpleWindow(root);called=[]
    monkeypatch.setattr(window,'scene_setup',lambda:called.append(window.source['id']))
    window._import_before={source['id']}
    path=root/'media/second.mp4';make_video(path,seconds=.6,vfr=False);import_video(window.project,path)
    window.done('Импорт завершён');window.close();app.processEvents()
    assert called==[] and not window._scene_queue
