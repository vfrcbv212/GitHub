"""Whole-person appearance exemplars. Scores are unvalidated matching hypotheses.

Uses the already installed detector's neural backbone, not face identification.
Explicit bindings create a memory only at their actual sample times. Corrections
are applied afresh, so stale identities never remain in a permanent prototype.
"""
from __future__ import annotations

import cv2
import numpy as np

from .media import check_cancel
from .image_io import read_image


def enrich(project,run_id,encoder=None,cancel=None,progress=None):
    from .models import manifest
    info=manifest(project.root/'models').get('detector',{})
    expected=getattr(encoder,'person_embedding_version',None) or (info['sha256']+':whole-person-fpn-2x1-v1' if info.get('sha256') else None)
    bound_tracks={b['track_id'] for b in project.all('track_binding') if b.get('player_id') and b.get('role')=='participant'}
    tracks=[t for t in project.all('track') if t['run_id']==run_id or t['id'] in bound_tracks]
    pending=[(t,s) for t in tracks for s in t.get('person_samples',[]) if not s.get('embedding') or
             (expected is not None and s.get('embedding_version')!=expected)]
    if not pending:return
    if encoder is None:
        from .vision import Detector
        encoder=Detector(project.root/'models','cpu')
    if not callable(getattr(encoder,'person_embedding',None)):
        project.put('identity_warning',{'id':run_id+':embedding','run_id':run_id,
            'reason':'Выбранный детектор не предоставляет признаки человека; повторная личность неизвестна'})
        return
    for i,(track,sample) in enumerate(pending):
        check_cancel(cancel)
        image=read_image(project.root/sample['path'])
        if image is None:sample['reason']='Образец человека недоступен';continue
        vector=encoder.person_embedding(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if vector is not None:
            sample.update(embedding=vector,embedding_version=encoder.person_embedding_version)
            project.put('track',track)
        if progress:progress((i+1)/len(pending),'Запоминание внешности по кадрам человека')


def _vectors(track,start=None,end=None):
    result=[]
    for s in track.get('person_samples',[]):
        if not s.get('embedding') or not s.get('embedding_version'):continue
        if start is not None and not start<=s['time']<=end:continue
        v=np.asarray(s['embedding'],np.float32);norm=np.linalg.norm(v)
        if v.ndim==1 and len(v)>1 and np.isfinite(v).all() and norm>1e-8:
            result.append((s,v/norm))
    return result


def match_tracks(project,run_id,mode):
    from .temporal import TrackBindingIndex
    all_tracks=project.all('track');players={p['id']:p for p in project.all('player')}
    bindings=project.all('track_binding');indexes={}
    for b in bindings:
        if b['run_id'] not in indexes:indexes[b['run_id']]=TrackBindingIndex(bindings,b['run_id'])
    references=[];occupied={}
    for ref in all_tracks:
        index=indexes.get(ref['run_id'])
        if index is None:continue
        for sample,vector in _vectors(ref):
            b=index.at(ref['id'],sample['time'])
            if b and b.get('role')=='participant' and b.get('player_id') in players:
                references.append((ref,sample,vector,b['player_id']))
        # Include partial bindings as temporal conflicts even if they have no crop.
        times,values=index.index.get(ref['id'],([],[]))
        for i,b in enumerate(values):
            if b and b.get('player_id') in players and b.get('role')=='participant':
                occupied.setdefault(b['player_id'],[]).append((ref,{'start':times[i],
                    'end':times[i+1] if i+1<len(times) else float('inf')}))
    for track in all_tracks:
        if track['run_id']!=run_id:continue
        vectors=_vectors(track);number=track.get('number_consensus',{}).get('number');by_player={}
        target_index=indexes.get(run_id)
        bound=target_index.at(track['id'],(track['start']+track['end'])/2) if target_index else None
        for ref,sample,vector,pid in references:
            if ref['id']==track['id']:continue
            # A memory belongs to one project/match; different recording parts can
            # refer to the same person. No motion is inferred across their gaps.
            if any(r['run_id']==run_id and r['source_id']==track['source_id'] and
                   max(b['start'],track['start'])<min(b['end'],np.nextafter(track['end'],np.inf)) and r['id']!=track['id']
                   for r,b in occupied.get(pid,[])):continue
            player=players[pid];agreement=None if number is None or player.get('number') is None else number==player['number']
            if mode=='numbers' and agreement is not True:continue
            distances=[float(max(0.,1.-np.dot(v,vector))) for s,v in vectors
                       if s['embedding_version']==sample['embedding_version'] and v.shape==vector.shape]
            if not distances:continue
            distance=float(np.median(distances))
            candidate={'player_id':pid,'reference_track':ref['id'],'reference_time':sample['time'],
                'appearance_distance':distance,'number_agreement':agreement if mode!='appearance' else None,
                'query_samples':len(distances),'embedding_version':sample['embedding_version']}
            if pid not in by_player or distance<by_player[pid]['appearance_distance']:by_player[pid]=candidate
        matches=sorted(by_player.values(),key=lambda x:x['appearance_distance'])
        good=bool(matches and matches[0]['appearance_distance']<=.25)
        ambiguous=len(matches)>1 and matches[1]['appearance_distance']-matches[0]['appearance_distance']<.08
        conflicting_number=bool(mode=='hybrid' and good and matches[0]['number_agreement'] is False)
        proposed=matches[0]['player_id'] if good and not ambiguous and not conflicting_number else None
        if bound and bound.get('player_id'):proposed=None
        reason=('Номера не используются; сопоставление изображений человека' if mode=='appearance' else 'Внешность с независимыми подсказками номера')
        if not vectors:reason='Нет пригодных кадров человека; личность неизвестна'
        elif ambiguous:reason='Несколько людей выглядят похоже; личность неизвестна'
        elif conflicting_number:reason='Номер и внешность противоречат друг другу; личность неизвестна'
        project.put('identity_hypothesis',{'id':run_id+':'+track['id'],'run_id':run_id,'track_id':track['id'],
            'mode':mode,'candidates':matches[:5],'proposed_player_id':proposed,'status':'hypothesis' if proposed else 'unknown',
            'rule':reason,'automatic_binding':False,
            'limitations':'Нейропризнаки общего детектора, не проверенная ReID-модель. Расстояние не вероятность. Автоподтверждение отключено.'})
