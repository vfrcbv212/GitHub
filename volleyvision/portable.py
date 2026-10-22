"""Offline desktop entry point for the portable Windows distribution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def default_project(bundle):
    # Preserve portable projects beside the program; fall back for read-only media.
    folder = Path(bundle) / 'Projects'
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / '.write-check'
        with probe.open('x') as stream:stream.write('')
        probe.unlink()
    except OSError:
        folder = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'VolleyVision' / 'Projects'
        folder.mkdir(parents=True, exist_ok=True)
    return folder / 'Match1'


def self_test(bundle, output):
    """Execute shipped Windows native extensions, weights and UI, strictly offline."""
    import socket
    import platform
    import sys
    import cv2
    import numpy as np
    import torch
    import torchvision
    import av
    from PySide6 import QtWidgets as W
    from .demo import make_video
    from .media import import_video
    from .models import verify
    from .store import Project
    from .roster import save_player, set_identity_mode
    from .vision import Detector, analyze
    from .identity import NumberOCR
    from .exporting import export_all
    from .gui import CalibrationDialog
    from .simple_ui import SimpleWindow
    from .quick_setup_ui import SceneDialog
    from .setup_ui import RosterDialog,KitDialog,ScoreDialog
    from .gallery import GalleryDialog
    from .image_io import read_image,write_image
    output = Path(output).resolve();output.mkdir(parents=True, exist_ok=True)
    sample=np.full((80,40,3),128,np.uint8)
    unicode_image=output/'Русское имя.png'
    assert write_image(unicode_image,sample) and np.array_equal(read_image(unicode_image),sample)
    # No package or model download is permitted in this execution path.
    def deny_network(*args, **kwargs):raise RuntimeError('Portable offline test attempted network access')
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    project_root = output / 'test-project'
    with Project(project_root, create=True, synthetic=True) as project:
        for tid in ('A','B'):
            project.put('team', {'id':tid,'name':tid})
            if not any(p['team_id']==tid for p in project.all('player')):
                for _ in range(2):save_player(project,tid)
        set_identity_mode(project,'appearance')
        for name in ('detector','ocr'):verify(project.root/'models',name)
        path = project.root/'media'/'synthetic.mp4'
        if not path.exists():make_video(path,seconds=1,vfr=False)
        sources=project.all('source');source=sources[0] if sources else import_video(project,path)
        detector=Detector(project.root/'models','cpu')
        vector=detector.person_embedding(np.full((256,128,3),128,np.uint8))
        assert len(vector)==512 and np.isfinite(vector).all()
        text=np.full((150,180,3),255,np.uint8)
        cv2.putText(text,'17',(15,110),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),6)
        ocr=NumberOCR(project.root/'models','cpu').read(text)
        job=analyze(project,source['id'],{'identity_mode':'appearance','device':'cpu','people_fps':1,'ball_fps':2,'ball_tiled':False})
        assert job['state']=='complete'
        export_all(project)
        assert (project.root/'exports'/'automatic'/'report.html').is_file()
    app=W.QApplication.instance() or W.QApplication([]);app.setStyle('Fusion')
    window=SimpleWindow(project_root);window.show();app.processEvents()
    assert window.identification.currentData()=='appearance'
    roster=RosterDialog(window.project,window);roster.show();app.processEvents()
    roster.grab().save(str(output/'roster.png'));roster.close()
    kit=KitDialog(window.project,window.source,window.current.rgb,0,window)
    kit.show();app.processEvents();kit.ours.setCurrentIndex(kit.ours.findData('A'))
    kit.team.setCurrentIndex(kit.team.findData('A'));kit.click(155,190);kit.click(180,230);kit.save()
    assert window.project.all('kit_sample')
    kit.grab().save(str(output/'kits.png'));kit.close()
    window.project.put('set',{'id':'selftest-set','number':1})
    score=ScoreDialog(window.project,window.source['id'],0,window)
    score.show();app.processEvents();score.boxes['A'].setValue(10);score.boxes['B'].setValue(8);score.save()
    assert window.project.all('score_snapshot')
    score.grab().save(str(output/'score.png'));score.close()
    calibration=CalibrationDialog(window.project,window.source,window.current.rgb)
    calibration.show();app.processEvents();calibration.grab().save(str(output/'calibration.png'));calibration.close()
    gallery=GalleryDialog(window.project,window);gallery.show();app.processEvents();gallery.close()
    scene=SceneDialog(window.project,window.source,window.current.rgb,window.current.time,window)
    scene.show();app.processEvents()
    scene.entries.update(court=[[100,100],[600,100],[600,340],[100,340]],
        net=[[100,140],[300,150],[600,140],[600,190],[300,200],[100,190]],
        center=[[100,200],[600,200]],team_a=[[320,280]],team_b=[[320,120]],
        antenna_a=[[100,140],[100,100]],antenna_b=[[600,140],[600,100]])
    scene.canvas.redraw();scene.grab().save(str(output/'simple-scene.png'));scene.save()
    assert window.project.get('scene_setup',window.source['id'])
    export_all(window.project)
    window.grab().save(str(output/'window.png'));window.close();app.processEvents()
    result={'platform':platform.platform(),'python':sys.version,'torch':torch.__version__,
        'torchvision':torchvision.__version__,'av':av.__version__,'cpu_only':torch.version.cuda is None,
        'offline':True,'bundled_weights_verified':True,'unicode_image_roundtrip':True,'person_vector_dimension':len(vector),
        'ocr_result':ocr,'synthetic_pipeline':job['state'],'gui_opened':True,
        'numberless_roster_opened':True,'kit_sample_saved':True,'manual_score_saved':True,
        'calibration_and_gallery_opened':True,
        'simple_ui_opened':True,'optional_scene_saved':True,
        'real_reid_quality':None,'native_windows_verified':False,
        'limits':'Synthetic input; no real volleyball accuracy or NVIDIA test. Execution host is verified separately.'}
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0


def main(bundle, argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    os.environ['VOLLEYVISION_BUNDLE_ROOT']=str(Path(bundle).resolve())
    if args.self_test:return self_test(bundle,args.out or Path(bundle)/'Check')
    root=args.project or default_project(bundle)
    if args.demo:
        from .demo import create
        root=root.parent/'Demo'
        if not (root/'project.sqlite').exists():create(root)
    from .store import Project
    from .roster import set_identity_mode
    from .models import seed_bundled
    new=not (root/'project.sqlite').exists()
    with Project(root,create=True) as project:
        if new:set_identity_mode(project,'appearance')
        seed_bundled(project.root/'models',include_ocr=True)
    from .gui import run
    return run(root)
