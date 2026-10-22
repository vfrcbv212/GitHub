import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import copy
import numpy as np
import pytest
from PySide6 import QtWidgets as W

from volleyvision.line_geometry import fit_from_lines
from volleyvision.calibration import validate_profile,map_floor,lens_distort
from volleyvision.gui import CalibrationDialog
from volleyvision.store import Project


def segments(lens=None):
    rows=[('x=0',[[100,180],[100,380]]),('x=18',[[820,180],[820,380]]),
          ('y=0',[[180,100],[740,100]]),('y=9',[[180,460],[740,460]])]
    return [{'line':name,'image':lens_distort(points,lens).tolist()} for name,points in rows]


@pytest.mark.parametrize('lens',[None,{'type':'pinhole','K':[[700,0,500],[0,700,300],[0,0,1]],'D':[-.05,0,0,0,0],'image_size':[1000,600]}])
def test_visible_line_segments_fit_without_invented_corner_correspondences(lens):
    control=lens_distort([[340,100]],lens)
    p=fit_from_lines(segments(lens),[1000,600],control,[[6,0]],lens)|{'id':'cal','source_id':'v','source_sha256':'hash','image_size':[1000,600]}
    assert p['verified'] and p['image_points']==[] and p['world_points']==[]
    checked=validate_profile(p,source_id='v',source_sha256='hash',image_size=[1000,600])
    point=lens_distort([[500,280]],lens)[0]
    assert map_floor(checked,point,'v',0,on_floor=True)['floor_point']==pytest.approx([10,4.5],abs=1e-4)
    edited=copy.deepcopy(p);edited['matrix'][0][2]+=2
    assert not validate_profile(edited)['verified']


def test_parallel_insufficient_and_dependent_controls_refuse_metric_geometry():
    rows=segments()
    with pytest.raises(ValueError):fit_from_lines(rows[:3],[1000,600])
    parallel=[{'line':f'x={x}','image':[[100+40*x,180],[100+40*x,380]]} for x in (0,6,12,18)]
    with pytest.raises(ValueError):fit_from_lines(parallel,[1000,600])
    with pytest.raises(ValueError,match='уже задано'):fit_from_lines(rows,[1000,600],[[100,100]],[[0,0]])
    assert not fit_from_lines(rows,[1000,600])['verified']


def test_line_calibration_and_net_referee_polygons_save_and_reopen(tmp_path):
    app=W.QApplication.instance() or W.QApplication([])
    with Project(tmp_path/'p',create=True) as p:
        source=p.put('source',{'id':'v','path':'missing.mp4','duration':10,'sha256':'hash','display_width':1000,'display_height':600})
        rgb=np.zeros((600,1000,3),np.uint8);dialog=CalibrationDialog(p,source,rgb)
        for s in segments():
            dialog.mode.setCurrentIndex(dialog.mode.findData('floor_line'));dialog.floor_line.setCurrentIndex(dialog.floor_line.findData(s['line']))
            for x,y in s['image']:dialog.click(x,y)
            dialog.finish_polygon()
        dialog.controls=[([340,100],[6,0])]
        for mode in ('net_region','referee_region'):
            dialog.mode.setCurrentIndex(dialog.mode.findData(mode))
            for x,y in [[420,200],[450,200],[450,300],[420,300]]:dialog.click(x,y)
            dialog.finish_polygon()
        dialog.save();assert W.QDialog.result(dialog)==W.QDialog.DialogCode.Accepted
        saved=p.all('calibration')[-1];assert saved['verified'] and len(saved['net_region'])==4 and len(saved['referee_region'])==4
        again=CalibrationDialog(p,source,rgb);again.save();reopened=p.all('calibration')[-1]
        assert reopened['line_constraints']==saved['line_constraints'] and reopened['verified']
        assert np.array(reopened['matrix'])==pytest.approx(np.array(saved['matrix']))
        again.close();dialog.close();app.processEvents()
