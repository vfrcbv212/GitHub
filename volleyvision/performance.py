"""Local processing measurements; no match-accuracy or GPU-utilisation claims."""
import json


STAGES={'decode_wait_s':'Ожидание кадров','people_inference_s':'Поиск людей',
        'ball_inference_s':'Поиск мяча','identity_s':'Идентификация','storage_s':'Сохранение'}
SCOPE=('Поиск объектов и идентификация. Время не включает импорт, загрузку моделей, '
       'подготовку HDR, отдельную модель действий и создание отчёта. '
       'Ожидание кадров измеряется после параллельного декодирования. '
       'Скорость не является оценкой точности или загрузки GPU.')


def processing_diagnostics(project,source_id=None):
    sources={s['id']:s for s in project.all('source')}
    latest={}
    for job in project.all('job'):
        if source_id is None or job['source_id']==source_id:latest[job['source_id']]=job
    rows=[]
    for sid,job in latest.items():
        row={k:job.get(k) for k in ('id','source_id','state','device','samples','ball_samples','ball_missing',
             'elapsed_s','peak_ram_bytes','peak_vram_bytes','warning','automation_limit')}
        duration=sources.get(sid,{}).get('duration',0)
        processed=job.get('processed_until') if job.get('state') in ('complete','setup_complete') else job.get('last_time',0)
        processed=max(0,min(duration,processed or 0))
        elapsed=job.get('elapsed_s') or 0
        row.update(processed_video_s=processed,speed_x=processed/elapsed if processed>0 and elapsed>0 else None,
                   timings=dict(job.get('timings',{})),timing_scope=SCOPE)
        rows.append(row)
    return rows


def describe_performance(row):
    elapsed=row.get('elapsed_s')
    if not elapsed:return 'Замер ещё не завершён.'
    device={'cuda':'GPU (CUDA)','cpu':'CPU'}.get(row.get('device'),'Устройство не записано')
    speed=f"; {row['speed_x']:.2f}× длительности видео" if row.get('speed_x') is not None else ''
    state={'complete':'завершено','setup_complete':'предварительный проход',
           'cancelled':'остановлено','failed':'ошибка','running':'в процессе'}.get(row.get('state'),'')
    lines=[f'{device} · {state} · {elapsed:.1f} с{speed}']
    timings=row.get('timings',{})
    if timings:
        lines.append(' · '.join(f'{label}: {timings[key]:.1f} с' for key,label in STAGES.items() if key in timings))
    lines.append('Время поиска и идентификации; импорт, загрузка моделей и создание отчёта считаются отдельно.')
    if row.get('warning'):lines.append(str(row['warning']))
    return '\n'.join(lines)


def write_diagnostics(project):
    rows=processing_diagnostics(project)
    if not rows:raise ValueError('Сначала запустите обработку видео')
    folder=project.root/'exports'/'diagnostics';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'processing.json';temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps({'schema_version':1,'timing_scope':SCOPE,'runs':rows},
                                    ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)
    return path
