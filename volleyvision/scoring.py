"""Manual score evidence, independent of inferred actions and player credit."""
from __future__ import annotations

import copy
import math

from .domain import uid


def save_score(project,set_id,scores,source_id=None,time=None,final=False,initial=False,object_id=None):
    if not project.get('set',set_id):raise ValueError('Сначала выберите или создайте партию')
    teams={t['id'] for t in project.all('team')}
    if len(teams)!=2 or set(scores)!=teams:raise ValueError('Укажите очки обеих команд')
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in scores.values()):
        raise ValueError('Очки должны быть целыми неотрицательными числами')
    if (source_id is None)!=(time is None):raise ValueError('Источник и время задаются вместе')
    if source_id is not None:
        source=project.get('source',source_id)
        if not source or not isinstance(time,(int,float)) or not math.isfinite(time) or not 0<=time<=source['duration']:
            raise ValueError('Время счёта должно находиться в исходной записи')
    if not final and source_id is None:raise ValueError('Для промежуточного счёта нужно время записи')
    if initial and (final or any(scores.values())):raise ValueError('Начало партии задаётся счётом 0:0 без отметки итога')
    if initial and any(s.get('initial') and s['set_id']==set_id and s['id']!=object_id and not s.get('removed') for s in project.all('score_snapshot')):
        raise ValueError('Начало партии уже задано; выберите его для исправления')
    if final and any(s.get('final') and s['set_id']==set_id and s['id']!=object_id and not s.get('removed') for s in project.all('score_snapshot')):
        raise ValueError('Итог партии уже задан; выберите его для исправления')
    old=project.get('score_snapshot',object_id) if object_id else None
    if object_id and not old:raise ValueError('Запись счёта больше не существует')
    result={'id':object_id or uid(),'set_id':set_id,'scores':dict(scores),'source_id':source_id,
        'time':time,'final':bool(final),'initial':bool(initial),'origin':'manual','evidence':'confirmed',
        'revision':(old or {}).get('revision',0)+1,'removed':False}
    # Audit and edited value belong to one transaction.
    from .store import dumps
    with project.db:
        history_id=uid()
        project.db.execute('INSERT INTO objects VALUES(?,?,?)',('score_history',history_id,dumps({'id':history_id,'before':old,'after':result})))
        project.db.execute('INSERT OR REPLACE INTO objects VALUES(?,?,?)',('score_snapshot',result['id'],dumps(result)))
    return result


def remove_score(project,object_id):
    old=project.get('score_snapshot',object_id)
    if not old:raise ValueError('Запись счёта не найдена')
    new=copy.deepcopy(old);new['removed']=True;new['revision']+=1
    from .store import dumps
    with project.db:
        history_id=uid()
        project.db.execute('INSERT INTO objects VALUES(?,?,?)',('score_history',history_id,dumps({'id':history_id,'before':old,'after':new})))
        project.db.execute('UPDATE objects SET body=? WHERE kind=? AND id=?',(dumps(new),'score_snapshot',object_id))


def score_report(project,mode='automatic',set_id=None):
    sources={s['id']:s for s in project.all('source')};teams=project.all('team');tids=[t['id'] for t in teams]
    def moment(source_id,time):
        source=sources.get(source_id)
        return source.get('offset',0)+time if source and time is not None else None
    snapshots=[s for s in project.all('score_snapshot') if not s.get('removed')]
    rallies=project.all('rally');rows=[];conflicts=[]
    for party in project.all('set'):
        if set_id and party['id']!=set_id:continue
        known=[r for r in rallies if r.get('set_id')==party['id'] and r.get('winner_team_id') in tids and
               r.get('evidence',{}).get('winner_team_id') in ('confirmed','accepted' if mode!='confirmed' else 'confirmed')]
        overlapping=set();by_source={}
        for r in known:by_source.setdefault(r.get('source_id'),[]).append(r)
        for group in by_source.values():
            active=[]
            for r in sorted(group,key=lambda x:x.get('start',0)):
                active=[p for p in active if p.get('end',0)>=r.get('start',0)]
                for p in active:
                    if (max(p.get('start',0),r.get('start',0))<min(p.get('end',0),r.get('end',0)) or
                        p.get('start')==p.get('end')==r.get('start')==r.get('end')):
                        overlapping.update([p['id'],r['id']])
                active.append(r)
        if overlapping:conflicts.append({'set_id':party['id'],'rally_ids':sorted(overlapping),
            'reason':'Интервалы известных розыгрышей перекрываются; их очки нельзя считать независимыми без исправления'})
        ss=[copy.deepcopy(s) for s in snapshots if s.get('set_id')==party['id']]
        for s in ss:s['timeline_time']=moment(s.get('source_id'),s.get('time'))
        ss.sort(key=lambda s:(math.inf if s['timeline_time'] is None else s['timeline_time'],s['id']))
        intervals=[];previous=None
        for s in ss:
            current=s['timeline_time'];lower={tid:0 for tid in tids}
            for r in known:
                t=moment(r.get('source_id'),r.get('end'))
                if t is None:continue
                if (current is None or t<=current) and (previous is None or previous['timeline_time'] is None or t>previous['timeline_time']):
                    lower[r['winner_team_id']]+=1
            reasons=['Сверка неполна: есть перекрывающиеся известные розыгрыши'] if overlapping else []
            delta=None;unattributed=None
            if previous is not None:
                delta={tid:s['scores'][tid]-previous['scores'][tid] for tid in tids}
                if any(v<0 for v in delta.values()):reasons.append('Счёт уменьшился внутри партии')
                if previous.get('final'):reasons.append('После итога партии есть ещё запись счёта')
                if previous['timeline_time']==current:reasons.append('Несколько записей счёта на один момент; интервал неоднозначен')
                if any(lower[tid]>delta[tid] for tid in tids):reasons.append('Известных победителей розыгрышей больше, чем прирост счёта')
                if not reasons:unattributed={tid:delta[tid]-lower[tid] for tid in tids}
            elif any(lower[tid]>s['scores'][tid] for tid in tids):
                reasons.append('Известных победителей розыгрышей больше, чем введённых очков')
            if s.get('final') and len(set(s['scores'].values()))==1:reasons.append('Итог партии содержит равный счёт; победитель не определён')
            entry={'from_snapshot_id':previous['id'] if previous else None,'to_snapshot_id':s['id'],
                'score_delta':delta,'known_rally_points':lower,'unattributed_team_points':unattributed,'conflicts':reasons}
            intervals.append(entry)
            conflicts.extend({'set_id':party['id'],'snapshot_id':s['id'],'reason':reason} for reason in reasons)
            previous=s
        final=next((s for s in reversed(ss) if s.get('final')),None)
        winner=None
        if final and len(final['scores'])==2 and len(set(final['scores'].values()))==2:
            winner=max(final['scores'],key=final['scores'].get)
        rows.append({'set_id':party['id'],'number':party.get('number'),'snapshots':ss,'final_scores':final['scores'] if final else None,
            'winner_team_id':winner,'known_rally_points':{tid:sum(r['winner_team_id']==tid for r in known) for tid in tids},
            'known_rally_points_independent':not bool(overlapping),'reconciliation':intervals})
    return {'sets':rows,'conflicts':conflicts,'origin':'manual','teams':teams,
        'limitations':['Ручной счёт не определяет автора, тип, причину и время каждого очка между отметками.',
            'Счёт относится к партиям; фильтры отдельных действий, игроков и источников к нему не применяются.',
            'Отсутствие конфликтов с известными розыгрышами не доказывает полноту их обнаружения.']}
