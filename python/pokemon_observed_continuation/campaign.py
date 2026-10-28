"""Matched full-battle PPO, fixed endpoints, and exhaustive original-update replay."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import time
import numpy as np
from pokemon_fit.data import fingerprint
from pokemon_full_train.storage import ROOT, freeze, admission, digest, atomic_json
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.campaign import value_summary
from pokemon_full.learner import parameter_hash
from pokemon_shared.learning import mastery
from .common import tactical
from pokemon_long_transfer.value import summarize_value
from pokemon_transfer.common import specs, select
from pokemon_credit.campaign import record, same
from .common import (PLAN, read, validate_plan, admit, initial, load_policy, team_keys,
    save, load, same_state, evaluation_specs, diagnostic_specs, check_collection,
    selfplay_value, analysis, budget)
from pokemon_worker_watch.collector import Pool
from pokemon_nonlinear_policy.learning import optimize_batch
TIMING = ['elapsed_seconds','collection_seconds','update_seconds_including_compile',
          'peak_parent_rss_mib','profiles']


def light(games):
    return [{k:g[k] for k in ['outcome','decisions','learner_side','terminal_value_mse',
        'value_mean','unavailable_choices']} for g in games]


def training_regimes(arm):
    quota=budget(arm)['selected_rows']//2
    return [(None,quota),('tactical',quota)]


def training_specs(plan,seed,first,opponent):
    if opponent not in [None,'tactical']:raise ValueError('Unknown train opponent')
    stream={**plan,'training_seed_namespace':plan['training_seed_namespace']+'/'+(opponent or 'self_play')}
    return [{**job,'opponent':opponent,'learner_side':'p2' if job['episode']%2 else 'p1'}
            for job in specs(stream,seed,first)]


def collect_update(pool,p,plan,seed,index,cursor,arm):
    games=[];profiles=[];first=cursor;strata=[];batches=[]
    for opponent,quota in training_regimes(arm):
        collection=[];start=cursor
        while sum(len(rows) for rows,_ in collection)<quota:
            if cursor-start>=budget(arm)['maximum_games_per_stream']:
                raise ValueError('Fixed training stream cap exhausted')
            jobs=training_specs(plan,seed,cursor,opponent)
            chunk,profile=pool.collect(p,jobs)
            check_collection(chunk,cursor,8,'train',opponent)
            collection.extend(chunk);profiles.append(profile);cursor+=8
        selection_seed=derived_seed('f29-stratum-selection',seed,opponent)
        batch,selection=select(collection,selection_seed,index,quota)
        strata.append({'opponent':opponent,'first_episode':start,'next_episode':cursor,'selection':selection,
            'run_seed':jobs[0]['seed']})
        batches.append(batch);games.extend(g for _,g in collection)
    batch={k:np.concatenate([b[k] for b in batches]) for k in batches[0]}
    if len(batch['actions'])!=budget(arm)['selected_rows']:raise ValueError('Selected cohort row budget')
    return batch,{'first_episode':first,'next_episode':cursor,'games':games,'strata':strata,
        'selection':{'selected_rows':budget(arm)['selected_rows'],'collected_rows':sum(s['selection']['collected_rows'] for s in strata),
            'selected_batch_sha256':fingerprint(batch)}},profiles


def campaign(run,original=None):
    plan=read(PLAN);validate_plan(plan);previous,oracle,old_train,past=admit(plan)
    catalog=read(ROOT/plan['catalog'])
    if original:
        old_manifest=read(original/'inputs/manifest.json');admission(original,old_manifest)
        same(old_manifest['plan'],plan,'plan')
    run.mkdir(parents=True,exist_ok=False);manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json')
    if original:same(manifest,old_manifest,'manifest')
    def evidence(rel,value):record(run/rel,value,original/rel if original else None)
    begun=time.perf_counter();profiles=[];counts=Counter();training=[];evaluation=[];diagnostics=[]
    training_keys=set(old_train);evaluation_keys=set();diagnostic_keys=set();seeds={};update_seconds=0.
    evidence('roles.json',{'historical_training_teams':len(old_train),'historical_development_teams':len(past)})

    def roles(games,role):
        keys=team_keys(games)
        if role=='training':
            if keys&(past|evaluation_keys|diagnostic_keys):raise ValueError('Development team entered training')
            training_keys.update(keys)
        else:
            if keys&(training_keys|past):raise ValueError('Evaluation exact-team history overlap')
            other=diagnostic_keys if role=='evaluation' else evaluation_keys
            if keys&other:raise ValueError('Self-play diagnostic/main panel overlap')
            (evaluation_keys if role=='evaluation' else diagnostic_keys).update(keys)

    def panel(pool,p,seed,base,include_diagnostics):
        panels={};diag=[]
        for opponent in plan['opponents']:
            games=[]
            for first in range(0,128,4):
                jobs=evaluation_specs(plan,seed,opponent,first,4)
                chunk,profile=pool.collect(p,jobs);gs=[g for _,g in chunk]
                evidence(str(base/'games'/f'{opponent}-{first:04d}.json'),{'specs':jobs,'games':gs})
                check_collection(chunk,first,4,'development',opponent);roles(gs,'evaluation')
                games.extend(gs);profiles.append(profile)
                atomic_json(run/'progress.json',{'stage':'evaluation','seed':seed,'endpoint':str(base),
                    'opponent':opponent,'pairs':first+4,'elapsed_seconds':time.perf_counter()-begun})
            panels[opponent]=games;evaluation.extend(light(games));counts['evaluation_games']+=len(games)
        if include_diagnostics:
            for first in range(0,16,4):
                jobs=diagnostic_specs(plan,seed,first,4);chunk,profile=pool.collect(p,jobs);gs=[g for _,g in chunk]
                evidence(str(base/'selfplay'/f'{first:04d}.json'),{'specs':jobs,'games':gs})
                check_collection(chunk,first,4,'development');roles(gs,'diagnostic')
                diag.extend(gs);profiles.append(profile)
            diagnostics.extend(light(diag));counts['selfplay_diagnostic_games']+=len(diag)
        t=tactical(p,oracle,pool.architecture,catalog)
        summary={'parameter_hash':parameter_hash(p),'panels':{o:value_summary(gs) for o,gs in panels.items()},
            'seat_panels':{o:{s:value_summary([g for g in gs if g['learner_side']==s]) for s in ['p1','p2']} for o,gs in panels.items()},
            'terminal_value':{o:summarize_value(gs) for o,gs in panels.items()},
            'actions':{o:dict(Counter(str(a['action']) for g in gs for a in g['actor_trace'])) for o,gs in panels.items()},
            'selfplay_value':selfplay_value(diag) if include_diagnostics else None,
            'mastery':mastery(t,plan['mastery_thresholds'])}
        evidence(str(base/'tactical.json'),t);evidence(str(base/'summary.json'),summary)
        return panels,summary

    for seed in plan['seeds']:
        base=Path(str(seed))
        panels={};arms={}
        for arm in plan['arms']:
            cfg=budget(arm);p,opt,identity=initial(previous,seed,arm);cursor=identity['next_episode'];start_cursor=cursor;arm_train=[];endpoints={};panels[arm]={};exposure=Counter()
            a=base/arm;evidence(str(a/'initial-identity.json'),identity)
            def checkpoint(cohort):
                folder=run/a/'checkpoints'/str(cohort)
                save(folder,p,opt,h,seed,arm,cohort,cursor)
                q,qo=load(folder,h,seed,arm,cohort,cursor);same_state(p,opt,q,qo)
                if original:
                    q,qo=load(original/a/'checkpoints'/str(cohort),h,seed,arm,cohort,cursor)
                    same_state(p,opt,q,qo)
                counts['checkpoint_comparisons']+=1
            checkpoint(cfg['start_cohort'])
            with Pool(run/a/'workers',8,arm,catalog) as pool:
                panels[arm]['262144'],endpoints['262144']=panel(pool,p,seed,a/str(cfg['start_cohort']),True)
                for index in range(cfg['start_cohort'],cfg['cohorts']):
                    if index in cfg['recovery_indices']:
                        q,qo=load((original or run)/a/'checkpoints'/str(index),h,seed,arm,index,cursor)
                        same_state(p,opt,q,qo);p,opt=q,qo;counts['original_next_update_recoveries']+=1
                    before=parameter_hash(p);batch,ledger,ps=collect_update(pool,p,plan,seed,index,cursor,arm)
                    for stratum in ledger['strata']:
                        exposure[stratum['opponent'] or 'self_play']+=stratum['selection']['selected_rows']
                    cursor=ledger['next_episode'];roles(ledger['games'],'training');profiles.extend(ps)
                    t0=time.perf_counter()
                    p,opt,metrics=optimize_batch(p,opt,batch,derived_seed(plan['optimizer_seed_namespace'],seed,index),arm)
                    update_seconds+=time.perf_counter()-t0
                    if int(opt[2])!=cfg['optimizer_steps_per_cohort']*(index+1) or len(metrics)!=cfg['optimizer_steps_per_cohort']:raise ValueError('Optimizer linkage')
                    evidence(str(a/f'update-{index:03d}.json'),{'ledger':ledger,'metrics':metrics,
                        'parameter_hash_before':before,'parameter_hash_after':parameter_hash(p),'optimizer_step':int(opt[2])})
                    arm_train.extend(light(ledger['games']));counts.update(cohorts=1,optimizer_steps=cfg['optimizer_steps_per_cohort'],
                        selected_ppo_rows=cfg['selected_rows'],ppo_row_presentations=2*cfg['selected_rows'])
                    atomic_json(run/'progress.json',{'stage':'training','seed':seed,'arm':arm,'cohorts':index+1,
                        'games':cursor-start_cursor,'next_episode':cursor,'additional_cohorts':index+1-cfg['start_cohort'],'elapsed_seconds':time.perf_counter()-begun})
                    if index+1 in cfg['checkpoint_cohorts']:checkpoint(index+1)
                    if index+1 in cfg['endpoints']:
                        gs,summary=panel(pool,p,seed,a/str(index+1),True);endpoints[str((index+1)*cfg['selected_rows'])]=summary
                        if index+1==cfg['cohorts']:panels[arm]['1048576']=gs
            expected_exposure={'self_play':393216,'tactical':393216}
            if dict(exposure)!=expected_exposure:raise ValueError('Selected-row opponent mixture mismatch')
            training.extend(arm_train)
            arms[arm]={'selected_rows_by_opponent':dict(exposure),'training':value_summary(arm_train),'endpoints':endpoints,'source_identity':identity,
                'optimizer_steps':int(opt[2]),'additional_optimizer_steps':int(opt[2])-identity['optimizer_step'],
                'initial_next_episode':start_cursor,'final_next_episode':cursor}
        result={'arms':arms,'analysis':analysis(panels,plan)}
        evidence(str(base/'result.json'),result);seeds[str(seed)]=result
    if dict(counts)!=plan['expected_counts']:raise ValueError('Fixed counts mismatch: '+str(dict(counts)))
    scientific={'status':'passed','input_hash':h,'plan':plan,'counts':dict(counts),'seeds':seeds,
        'training':value_summary(training),'evaluation':value_summary(evaluation),
        'selfplay_diagnostics':value_summary(diagnostics),
        'unavailable_choices':sum(g['unavailable_choices'] for g in training+evaluation+diagnostics),
        'roles':{'historical_train':len(old_train),'historical_development':len(past),
            'training_teams_with_history':len(training_keys),'evaluation_teams':len(evaluation_keys),
            'selfplay_diagnostic_teams':len(diagnostic_keys),'disallowed_overlap':0},
        'all_seed_learning':{a:all(s['analysis']['learning'][a]['passed'] for s in seeds.values()) for a in plan['arms']},
        'all_seed_feature_benefit':all(s['analysis']['feature_benefit']['passed'] for s in seeds.values()),
        'all_seed_base_power_benefit':all(s['analysis']['base_power_benefit']['passed'] for s in seeds.values()),
        'promotion_authorized':False,'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}
    evidence('scientific.json',scientific);admission(run,manifest);admit(plan)
    if original:admission(original,old_manifest)
    report={**scientific,'elapsed_seconds':time.perf_counter()-begun,'profiles':profiles,
        'collection_seconds':sum(p['collection_seconds'] for p in profiles),
        'update_seconds_including_compile':update_seconds,
        'peak_parent_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    atomic_json(run/'report.json',report)
    if original:
        atomic_json(run/'verification.json',{'status':'passed','input_hash':h,
            'report_sha256':digest(original/'report.json'),'scientific_sha256':digest(original/'scientific.json'),
            'counts':dict(counts),'training':report['training'],'evaluation':report['evaluation'],
            'selfplay_diagnostics':report['selfplay_diagnostics']})
    print({'status':'passed','seconds':report['elapsed_seconds'],'learning':report['all_seed_learning'],
        'feature_benefit':report['all_seed_feature_benefit']},flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original',type=Path)
    a=p.parse_args();campaign(a.run.resolve(),a.original.resolve() if a.original else None)
