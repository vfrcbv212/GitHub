"""Metric floor initialization from labelled visible line segments, no fake corners."""
from __future__ import annotations

import numpy as np
from .calibration import lens_correct,validate_lens_model,_transform_points,_points,_matrix

FLOOR_LINES={**{f'x={x}':(0,x) for x in (0,6,9,12,18)},**{f'y={y}':(1,y) for y in (0,9)}}
LINE_LABELS={'x=0':'Лицевая 1','x=6':'Трёхметровая 1','x=9':'Средняя линия на полу',
    'x=12':'Трёхметровая 2','x=18':'Лицевая 2','y=0':'Боковая 1','y=9':'Боковая 2'}


def validate_segments(segments,image_size):
    if image_size is None or len(image_size)!=2 or not np.isfinite(image_size).all() or np.any(np.asarray(image_size)<=0):
        raise ValueError('Для отрезков нужны размеры исходного изображения')
    rows=[]
    for s in segments:
        if s.get('line') not in FLOOR_LINES:raise ValueError('Выберите назначение линии пола')
        p=_points(s.get('image'),'Отрезок линии')
        if len(p)!=2 or np.linalg.norm(p[1]-p[0])<12:raise ValueError('Нужны две точки достаточно длинного видимого отрезка')
        if np.any(p<0) or np.any(p>image_size):raise ValueError('Отрезок должен находиться внутри кадра')
        rows.append({'line':s['line'],'image':p.tolist()})
    return rows


def _solve(segments,image_size,lens):
    constraints=[]
    for s in segments:
        axis,value=FLOOR_LINES[s['line']];line=np.zeros(3);line[axis]=[18,9][axis];line[2]=-value
        line/=np.linalg.norm(line)
        for p in lens_correct(s['image'],lens)/image_size:
            constraints.append(np.outer(line,[*p,1]).ravel())
    a=np.asarray(constraints)
    if len(a)<8:raise ValueError('Для метров нужны независимые ограничения: минимум четыре линии двух направлений')
    _,singular,vh=np.linalg.svd(a,full_matrices=True)
    if np.linalg.matrix_rank(a,tol=singular[0]*1e-7)<8 or singular[min(7,len(singular)-1)]/singular[0]<1e-5:
        raise ValueError('Эти линии не определяют устойчивую геометрию: добавьте независимые линии другого направления')
    h=np.diag([18.,9.,1.])@vh[-1].reshape(3,3)@np.diag([1/image_size[0],1/image_size[1],1])
    _matrix(h)
    if abs(h[2,2])>1e-10:h/=h[2,2]
    return h


def line_error(matrix,segments,lens):
    errors=[]
    for s in segments:
        axis,value=FLOOR_LINES[s['line']]
        p=_transform_points(matrix,lens_correct(s['image'],lens));errors.extend(p[:,axis]-value)
    return float(np.sqrt(np.mean(np.square(errors))))


def fit_from_lines(segments,image_size,control_image=(),control_world=(),lens_model=None):
    segments=validate_segments(segments,image_size);lens=validate_lens_model(lens_model,image_size)
    h=_solve(segments,np.asarray(image_size),lens);error=line_error(h,segments,lens)
    if error>.5:raise ValueError('Обозначенные линии противоречат друг другу; проверьте назначение и точки')
    ci,cw=[],[];control_error=None
    if len(control_image):
        ci=_points(control_image);cw=_points(control_world)
        if len(ci)!=len(cw) or np.any(cw<0) or np.any(cw>[18,9]):raise ValueError('Некорректные контрольные точки')
        used={s['line'] for s in segments}
        for p in cw:
            if sum(abs(p[FLOOR_LINES[name][0]]-FLOOR_LINES[name][1])<1e-7 for name in used)>=2:
                raise ValueError('Контрольное пересечение уже задано опорными линиями; выберите независимое')
        control_error=float(np.sqrt(np.mean(np.sum((_transform_points(h,lens_correct(ci,lens))-cw)**2,axis=1))))
    elif len(control_world):raise ValueError('Для контроля нужны координаты изображения и пола')
    raw=np.concatenate([np.asarray(s['image']) for s in segments]);anchors=lens_correct(raw,lens);base=_transform_points(h,anchors)
    rng=np.random.default_rng(42);deviations=[]
    for _ in range(40):
        jitter=[s|{'image':(np.asarray(s['image'])+rng.normal(0,2,(2,2))).tolist()} for s in segments]
        try:deviations.append(np.linalg.norm(_transform_points(_solve(jitter,np.asarray(image_size),lens),anchors)-base,axis=1).max())
        except ValueError:continue
    if not deviations:raise ValueError('Отрезки неустойчивы к смещению кликов')
    sensitivity=float(np.quantile(deviations,.95));verified=control_error is not None and control_error<=.3 and sensitivity<=.4
    return {'matrix':h.tolist(),'fit_kind':'labelled_lines','line_constraints':segments,'image_points':[],'world_points':[],
        'control_image_points':np.asarray(ci).tolist(),'control_world_points':np.asarray(cw).tolist(),
        'fit_rmse_m':error,'control_rmse_m':control_error,'click_sensitivity_2px_p95_m':sensitivity,'verified':verified,
        'geometry':'court' if verified else 'video','lens_model':lens,'coordinate_space':'rotated_original',
        'homography_space':'undistorted_original' if lens else 'upright_original',
        'warning':'Линии определяют плоскость пола; точность требует независимого контроля. Невидимые углы не являются наблюдениями.'}
