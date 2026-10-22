"""Stable player cards; a shirt number is optional and never the primary key."""
from .domain import uid

IDENTITY_MODES={'hybrid':'Внешность и номера','appearance':'По внешности, без номеров',
                'numbers':'По номерам с проверкой команды'}
ROLES={'unknown':'Не указано','setter':'Связующий','opposite':'Диагональный','outside':'Доигровщик',
       'middle':'Центральный','libero':'Либеро'}


def identity_mode(project,value=None):
    if value is None:value=(project.get('identity_config','default') or {}).get('mode','hybrid')
    if value not in IDENTITY_MODES:raise ValueError('Неизвестный режим идентификации')
    return value


def set_identity_mode(project,value):
    mode=identity_mode(project,value)
    project.put('identity_config',{'id':'default','mode':mode})
    return mode


def player_label(player):
    number=player.get('number');name=player.get('name') or player.get('label')
    if not name:return f'№{number}' if number is not None else 'Игрок '+player['id'][:6]
    return name+(f' · №{number}' if number is not None else '')


def save_player(project,team_id,number=None,name=None,role='unknown',player_id=None):
    if not project.get('team',team_id):raise ValueError('Выберите команду')
    if number is not None and (not isinstance(number,int) or isinstance(number,bool) or not 0<=number<=99):
        raise ValueError('Номер должен быть 0–99 или отсутствовать')
    if role not in ROLES:raise ValueError('Неизвестное амплуа')
    old=project.get('player',player_id) if player_id else None
    if player_id and not old:raise ValueError('Карточка игрока не найдена')
    if old and old['team_id']!=team_id:
        linked=(any(b.get('player_id')==player_id for b in project.all('track_binding')) or
                any(player_id in ([e.get('player_id'),e.get('credited_player_id')]+list(e.get('participants') or [])) for e in project.events()) or
                any(s.get('player_id')==player_id for s in project.all('kit_sample')) or
                any(player_id in s.get('players',[]) for s in project.all('lineup')))
        if linked:raise ValueError('Карточка уже связана с наблюдениями этой команды. Для другой команды создайте отдельную карточку')
    if number is not None and any(p['id']!=player_id and p['team_id']==team_id and p.get('number')==number for p in project.all('player')):
        raise ValueError('Этот номер уже есть у команды. Выберите существующую карточку для исправления')
    result=dict(old or {});result.update(id=player_id or uid(),team_id=team_id,number=number,name=(name or '').strip() or None,role=role)
    if not result.get('label'):
        used={p.get('label') for p in project.all('player') if p['team_id']==team_id};n=1
        while f'Игрок {n}' in used:n+=1
        result['label']=f'Игрок {n}'
    return project.put('player',result)
