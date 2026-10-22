"""Explicit local kit samples; appearance produces hypotheses, never identity."""
from __future__ import annotations

from .image_io import read_image,write_image

import math
import cv2
import numpy as np

from .domain import uid
from .identity import appearance


def set_our_team(project,team_id):
    if team_id is not None and not project.get('team',team_id):raise ValueError('Команда не найдена')
    match=project.get('match','match');match['our_team_id']=team_id;project.put('match',match)


def save_sample(project,source_id,time,rgb,bbox,team_id,kit_kind='regular',player_id=None):
    source=project.get('source',source_id);team=project.get('team',team_id)
    if not source or not team:raise ValueError('Выберите исходник и команду')
    if not isinstance(time,(int,float)) or not math.isfinite(time) or not 0<=time<=source['duration']:raise ValueError('Неверное время образца')
    if kit_kind not in ('regular','libero'):raise ValueError('Неизвестный вариант формы')
    player=project.get('player',player_id) if player_id else None
    if player_id and (not player or player['team_id']!=team_id):raise ValueError('Игрок не принадлежит выбранной команде')
    if kit_kind=='libero' and not player_id:raise ValueError('Для формы либеро выберите его карточку в составе')
    h,w=rgb.shape[:2]
    if [w,h]!=[source.get('display_width',w),source.get('display_height',h)]:raise ValueError('Нужен кадр исходного размера и ориентации')
    if len(bbox)!=4 or not np.isfinite(bbox).all():raise ValueError('Выделите прямоугольник формы')
    x1,y1,x2,y2=map(int,bbox)
    if not 0<=x1<x2<=w or not 0<=y1<y2<=h or x2-x1<12 or y2-y1<16:raise ValueError('Выделите более крупный образец формы внутри кадра')
    crop=rgb[y1:y2,x1:x2];sample_id=uid();path=project.root/'cache'/f'kit-{sample_id}.png'
    if not write_image(path,cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)):raise RuntimeError('Не удалось сохранить образец формы')
    return project.put('kit_sample',{'id':sample_id,'team_id':team_id,'player_id':player_id,'kit_kind':kit_kind,
        'source_id':source_id,'source_sha256':source.get('sha256'),'time':float(time),'bbox':[x1,y1,x2,y2],
        'image_size':[w,h],'path':project.relpath(path),'appearance':appearance(crop),'origin':'manual','active':True})


def hypotheses(project,track):
    descriptors=[]
    for sample in track.get('samples',[]):
        raw=read_image(project.root/sample['path'])
        if raw is not None:descriptors.append(appearance(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)))
    if not descriptors and track.get('appearance'):descriptors=[track['appearance']]
    if not descriptors:return {'status':'unknown','candidates':[],'reason':'Нет пригодных образцов траектории'}
    by_team={}
    for sample in project.all('kit_sample'):
        if not sample.get('active',True):continue
        # A project is one match: clothing examples can cover its recording parts.
        # Their original must still be registered with the same checksum. Court
        # geometry remains source-scoped and is never transferred this way.
        source=project.get('source',sample['source_id'])
        if not source or source.get('sha256')!=sample.get('source_sha256'):continue
        distances=[float(cv2.compareHist(np.asarray(d,np.float32),np.asarray(sample['appearance'],np.float32),cv2.HISTCMP_BHATTACHARYYA)) for d in descriptors]
        distance=float(np.median(distances));tid=sample['team_id']
        if tid not in by_team or distance<by_team[tid]['appearance_distance']:
            by_team[tid]={'team_id':tid,'appearance_distance':distance,'reference_sample_id':sample['id'],
                'kit_kind':sample['kit_kind'],'sample_count':len(descriptors)}
    candidates=sorted(by_team.values(),key=lambda x:x['appearance_distance'])
    similar=[c for c in candidates if c['appearance_distance']<=.35]
    ambiguous=len(similar)>1 and similar[1]['appearance_distance']-similar[0]['appearance_distance']<.12
    return {'status':'hypothesis' if similar and not ambiguous else 'unknown','candidates':candidates,
        'proposed_team_id':similar[0]['team_id'] if similar and not ambiguous else None,
        'reason':'Похожая форма нескольких команд; команда неизвестна' if ambiguous else
                 'Внешнее сходство образцов; цвет не доказывает команду и личность',
        'thresholds':'Unvalidated engineering thresholds; distance is not probability'}
