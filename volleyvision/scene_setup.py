"""Optional image-space scene setup. Floor, net and airborne objects stay separate."""
from __future__ import annotations

import copy

import cv2
import numpy as np

from .image_io import write_image


TOOLS={
    'court':('Контур площадки','#d637af','polygon'),
    'center':('Центральная линия на полу','#2496ff','line'),
    'attack_a':('Трёхметровая линия А','#ff4747','line'),
    'attack_b':('Трёхметровая линия Б','#ff4747','line'),
    'end_a':('Лицевая линия А','#d637af','line'),
    'end_b':('Лицевая линия Б','#d637af','line'),
    'sideline_a':('Боковая линия 1','#d637af','line'),
    'sideline_b':('Боковая линия 2','#d637af','line'),
    'net':('Контур сетки','#ffe441','polygon'),
    'antenna_a':('Антенна 1','#60e339','segment'),
    'antenna_b':('Антенна 2','#60e339','segment'),
    'ball':('Мяч','#60e339','circle'),
    'referee':('Судья','#ff4747','rectangle'),
    'team_a':('Наша команда А: точка на половине','#ff4747','point'),
    'team_b':('Соперник Б: точка на половине','#ff4747','point'),
    'person_a':('Игрок нашей команды А','#3bd3e0','points'),
    'person_b':('Игрок соперника Б','#ffb952','points'),
}


def validate_scene(value,source):
    value=copy.deepcopy(value)
    if value.get('source_id')!=source['id'] or value.get('source_sha256')!=source['sha256']:
        raise ValueError('Разметка относится к другому видео')
    w,h=source['display_width'],source['display_height']
    if value.get('image_size')!=[w,h]:raise ValueError('Размер кадра не совпадает')
    if not 0<=value.get('time',-1)<=source['duration']:raise ValueError('Выберите кадр внутри записи')
    entries=value.setdefault('entries',{})
    if set(entries)-set(TOOLS):raise ValueError('Неизвестный элемент разметки')
    for key,points in entries.items():
        if not points:continue
        p=np.asarray(points,dtype=float)
        if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or np.any(p<0) or np.any(p>[w,h]):
            raise ValueError('Точки должны находиться внутри кадра')
        kind=TOOLS[key][2];n=len(p)
        if kind=='polygon' and (n<3 or abs(cv2.contourArea(p.astype(np.float32)))<4):
            raise ValueError(TOOLS[key][0]+': нужны минимум три точки контура')
        if kind in ('line','segment','circle','rectangle') and (n<2 or np.linalg.norm(p[-1]-p[0])<2):
            raise ValueError(TOOLS[key][0]+': нужны две разные точки')
        if kind in ('segment','circle','rectangle') and n!=2:raise ValueError('Нужны ровно две точки')
        if kind=='point' and n!=1:raise ValueError('Нужна одна точка')
        entries[key]=p.tolist()
    # A visible contour can close at the frame edge: it is a search annotation,
    # never a metric court or an observed out-of-bounds decision.
    value['coordinate_space']='upright_original';value['metric_verified']=False
    return value


def save_scene(project,source,rgb,time_s,entries):
    teams=project.all('team')
    if not teams:
        for tid,label in (('A','Наша команда'),('B','Соперник')):project.put('team',{'id':tid,'name':label})
        teams=project.all('team')
    if len(teams)==1:
        tid='B' if teams[0]['id']!='B' else 'A';project.put('team',{'id':tid,'name':'Соперник'});teams=project.all('team')
    match=project.get('match','match');our=match.get('our_team_id')
    if our not in {t['id'] for t in teams}:our=teams[0]['id']
    opponent=next(t['id'] for t in teams if t['id']!=our)
    old=project.get('scene_setup',source['id']) or {}
    value=validate_scene({'id':source['id'],'source_id':source['id'],'source_sha256':source['sha256'],
        'image_size':[source['display_width'],source['display_height']],
        'time':float(time_s),'entries':entries,'version':old.get('version',0)+1,'origin':'manual',
        'team_ids':{'A':our,'B':opponent}},source)
    if list(rgb.shape[:2])!=[source['display_height'],source['display_width']]:raise ValueError('Размер опорного кадра не совпадает с видео')
    path=project.root/'cache'/('scene-'+source['id']+'-v'+str(value['version'])+'-reference.png')
    value['reference_image']=project.relpath(path)
    profile={'id':'scene-'+source['id'],'source_id':source['id'],'source_sha256':source['sha256'],
        'image_size':value['image_size'],'start':0.,'end':source['duration'],'reference_time':time_s,
        'reference_image':value['reference_image'],'geometry':'video','verified':False,
        'coordinate_space':'rotated_original','matrix':None,'scene_setup_version':value['version']}
    if entries.get('court'):profile['visible_image']=entries['court']
    if entries.get('net'):profile['net_region']=entries['net']
    if entries.get('referee'):
        (a,b),(c,d)=entries['referee'];profile['referee_region']=[[min(a,c),min(b,d)],[max(a,c),min(b,d)],
            [max(a,c),max(b,d)],[min(a,c),max(b,d)]]
    from .calibration import validate_profile
    validate_profile(profile,source_id=source['id'],source_sha256=source['sha256'],image_size=value['image_size'])
    if not write_image(path,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise ValueError('Не удалось сохранить опорный кадр')
    project.put('scene_history',value|{'id':source['id']+':'+str(value['version']),'profile':profile})
    project.put('scene_setup',value);project.put('calibration',profile)
    match['our_team_id']=our;project.put('match',match)
    return value


def side_of_line(line,point):
    """Signed pixel distance to the nearest segment of a hand-drawn floor line."""
    p=np.asarray(point,dtype=float);best=None
    for a,b in zip(line,line[1:]):
        a=np.asarray(a,float);v=np.asarray(b,float)-a;length=float(np.linalg.norm(v))
        if length<2:continue
        u=float(np.clip(np.dot(p-a,v)/(length*length),0,1));distance=float(np.linalg.norm(p-a-u*v))
        signed=float(np.cross(v,p-a)/length)
        if best is None or distance<best[0]:best=(distance,signed)
    return best[1] if best else None


class TeamMemory:
    """Keep a team hypothesis on one uninterrupted track, independent of jump height.

    Bounding-box feet are image coordinates, NOT measured ground contact. Spatial
    votes never confirm an identity or a substitution; new fragments start unknown.
    """
    def __init__(self,scene):
        self.scene=scene or {};self.entries=self.scene.get('entries',{});self.state={}

    def update(self,people,time_s,bindings=None):
        e=self.entries;line=e.get('center',[])
        mapping=self.scene.get('team_ids',{'A':'A','B':'B'})
        signs={mapping[tid]:side_of_line(line,e[key][0]) for tid,key in (('A','team_a'),('B','team_b')) if e.get(key)}
        valid=len(signs)==2 and all(s is not None and abs(s)>5 for s in signs.values()) and np.prod(list(signs.values()))<0
        referee_id=None
        if e.get('referee') and abs(time_s-self.scene.get('time',-100))<=.25:
            (a,b),(c,d)=e['referee'];rx1,rx2=sorted((a,c));ry1,ry2=sorted((b,d));ranked=[]
            for row in people:
                x1,y1,x2,y2=row['bbox'];intersection=max(0,min(x2,rx2)-max(x1,rx1))*max(0,min(y2,ry2)-max(y1,ry1))
                union=(x2-x1)*(y2-y1)+(rx2-rx1)*(ry2-ry1)-intersection
                if union>0:ranked.append((intersection/union,row['track_id']))
            ranked.sort(reverse=True)
            if ranked and ranked[0][0]>=.4 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.15):referee_id=ranked[0][1]
        for person in people:
            tid=person['track_id'];state=self.state.setdefault(tid,{'votes':{},'team_id':None,'evidence':'unknown','last':time_s})
            state['last']=time_s;x1,y1,x2,y2=person['bbox'];binding=(bindings or {}).get(tid)
            if referee_id==tid and not binding:state['role']='referee'
            if state.get('role')=='referee' and not binding:
                person['scene_role']='referee';person['team_proposal']={'team_id':None,'evidence':'unknown','reason':'Судья, указанный пользователем'}
                continue
            if binding and binding.get('role')!='participant':
                person['team_proposal']={'team_id':None,'evidence':'unknown','reason':'Человек не является игроком на площадке'}
                continue
            if binding and binding.get('team_id') and binding.get('role')=='participant':
                state.update(team_id=binding['team_id'],evidence='confirmed',reason='Ручная привязка траектории')
            if abs(time_s-self.scene.get('time',-100))<=.25:
                chosen=set()
                for team,key in (('A','person_a'),('B','person_b')):
                    for x,y in e.get(key,[]):
                        containing=[p for p in people if p['bbox'][0]<=x<=p['bbox'][2] and p['bbox'][1]<=y<=p['bbox'][3]]
                        if len(containing)==1 and containing[0]['track_id']==tid:chosen.add(mapping[team])
                if len(chosen)==1 and not (binding and binding.get('team_id')):
                    state.update(team_id=chosen.pop(),evidence='confirmed',reason='Пользователь указал человека на опорном кадре')
            if not state['team_id'] and valid:
                side=side_of_line(line,[(x1+x2)/2,y2]);margin=max(8,(x2-x1)*.15)
                if side is not None and abs(side)>margin:
                    team=next((t for t,s in signs.items() if s*side>0),None)
                    if team:state['votes'][team]=state['votes'].get(team,0)+1
                    ranked=sorted(state['votes'].items(),key=lambda x:x[1],reverse=True)
                    if ranked and ranked[0][1]>=3 and ranked[0][1]>=sum(state['votes'].values())*.8:
                        state.update(team_id=ranked[0][0],evidence='hypothesis',reason='Повторные положения нижней точки рамки; контакт с полом не доказан')
            person['team_proposal']={k:state.get(k) for k in ('team_id','evidence','reason')}
        self.state={k:v for k,v in self.state.items() if time_s-v['last']<=2}


class MainBall:
    """Use an optional manual ball circle; never emit predicted/lost positions.

    A single consistent detection after loss is a return hypothesis. Multiple
    candidates remain ambiguous, even if one is near the last screen position.
    """
    def __init__(self,scene):
        self.scene=scene or {};self.seed=self.scene.get('entries',{}).get('ball')
        self.track_id=None;self.last=None;self.pending=None;self.count=0

    def select(self,balls,time_s):
        if not self.seed:return balls,{'state':'unseeded','candidates':len(balls)}
        if abs(time_s-self.scene.get('time',0))<=.1:
            center,rim=self.seed;radius=max(8,math_dist(center,rim)*2)
            nearby=[b for b in balls if math_dist(center,[(b['bbox'][0]+b['bbox'][2])/2,(b['bbox'][1]+b['bbox'][3])/2])<=radius]
            if len(nearby)==1:
                self.track_id=nearby[0]['track_id'];self.last=time_s
                return nearby,{'state':'manual_seed_matched','candidates':len(balls)}
        current=[b for b in balls if b['track_id']==self.track_id]
        if len(current)==1:
            self.last=time_s;self.pending=None;self.count=0
            return current,{'state':'observed_track','candidates':len(balls)}
        if not balls:
            self.pending=None;self.count=0
            return [],{'state':'outside_or_not_detected','candidates':0}
        if len(balls)!=1:
            self.pending=None;self.count=0
            return [],{'state':'ambiguous','candidates':len(balls)}
        candidate=balls[0]['track_id']
        self.count=self.count+1 if candidate==self.pending else 1;self.pending=candidate
        if self.count>=3:
            self.track_id=candidate;self.last=time_s
            return balls,{'state':'return_hypothesis','candidates':1,'same_physical_ball_verified':False}
        return [],{'state':'checking_single_candidate','candidates':1}


def math_dist(a,b):
    return float(np.linalg.norm(np.asarray(a,dtype=float)-b))
