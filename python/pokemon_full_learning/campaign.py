"""F4 fixed-budget learning; persist cohorts and evaluate only fixed endpoints."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import optimize, derived_seed, PPO
from pokemon_full_train.storage import (ROOT, atomic_json, canonical, digest, freeze,
                                       admission, save_checkpoint, load_checkpoint)
from pokemon_full_train.campaign import value_summary
from pokemon_throughput.collector import merge_rows
from .collector import Pool
from .statistics import compare, learning_gate


def validate_plan(plan):
    if plan['version']!='full-learning-f4-v1' or plan['ppo']!=PPO:
        raise ValueError('Wrong F4 policy/update version')
    if plan['opponents']!=['random','base_power','tactical'] or plan['endpoints']!=['initial','final']:
        raise ValueError('Wrong frozen evaluation panel')
    if plan['workers']!=8 or plan['cohort_games']!=8:
        raise ValueError('F4 requires eight workers and eight-game cohorts')
    for key in ['curriculum','public_server','final_reserved_data','prime']:
        if plan[key] is not False: raise ValueError('Unsupported F4 scope')
    for key,lo,hi in [('train_games_per_seed',8,4096),('evaluation_pairs_per_opponent',2,512),
                      ('evaluation_chunk_pairs',1,16),('max_turns',2,120)]:
        if type(plan[key]) is not int or not lo<=plan[key]<=hi: raise ValueError('Invalid budget: '+key)
    if plan['train_games_per_seed']%8 or plan['evaluation_pairs_per_opponent']%plan['evaluation_chunk_pairs']:
        raise ValueError('Incomplete cohort/evaluation chunk')
    if not plan['seeds'] or len(set(plan['seeds']))!=len(plan['seeds']): raise ValueError('Invalid seeds')
    if any(type(s) is not int or not 0<=s<2**32 for s in [*plan['seeds'],plan['evaluation_seed']]):
        raise ValueError('Invalid random seed')
    settings=plan['statistics']
    if settings['family_cells']!=3*len(plan['seeds']) or settings['method']!='paired-cluster-percentile-bootstrap':
        raise ValueError('Wrong inference family')
    if settings['alpha']!=.05 or settings['practical_improvement']!=.05 or settings['max_final_cutoff_rate']!=.05:
        raise ValueError('Changed inferential gate')
    if type(settings['resamples']) is not int or not 100<=settings['resamples']<=20000:
        raise ValueError('Invalid bootstrap budget')
    if type(settings['seed']) is not int or not 0<=settings['seed']<2**32:
        raise ValueError('Invalid bootstrap seed')
    if plan['profile']=='production':
        expected=json.loads((ROOT/'contracts/full-learning-v1/plan.json').read_text())
        if plan!=expected: raise ValueError('Production plan must equal registered plan')
    elif plan['profile'] not in ['test','preflight']:
        raise ValueError('Unknown profile')
    if jax.default_backend()!='cpu' or jax.config.x64_enabled or not jax.config.jax_threefry_partitionable:
        raise ValueError('F4 requires original CPU FP32 partitionable-Threefry runtime')


def evaluation_specs(plan,seed,opponent,first,count):
    return [dict(seed=plan['evaluation_seed'],episode=episode,max_turns=plan['max_turns'],
        split='development',swap=swap,opponent=opponent,learner_side='p2' if swap else 'p1',
        actor_seed=derived_seed('f4-evaluation',seed,opponent))
        for episode in range(first,first+count) for swap in [False,True]]


def check_games(games,*,split,first,count,opponent=None):
    expected=[(i,False) for i in range(first,first+count)] if split=='train' else [
        (i,s) for i in range(first,first+count) for s in [False,True]]
    if len(games)!=len(expected): raise ValueError('Wrong game count')
    for game,(episode,swap) in zip(games,expected):
        if (game['episode'],game['swap'])!=(episode,swap) or game['split']!=split or game['opponent']!=opponent:
            raise ValueError('Wrong game coordinates')
        if split=='development' and game['learner_side']!=('p2' if swap else 'p1'):
            raise ValueError('Wrong paired learner seat')
        if not all(t['partition']==split for t in game['provenance']['teams']):
            raise ValueError('Wrong team partition')
        if len(game['trace'])!=game['decisions'] or sum(game['phases'].values())!=game['decisions']:
            raise ValueError('Wrong decision ledger')
        if game['outcome']['kind'] not in ['terminal','administrative_cutoff']:
            raise ValueError('Missing outcome')


def evaluate(run,pool,parameters,seed,endpoint,plan,input_hash):
    results={}
    for opponent in plan['opponents']:
        records=[]
        for first in range(0,plan['evaluation_pairs_per_opponent'],plan['evaluation_chunk_pairs']):
            admission(run,json.loads((run/'inputs/manifest.json').read_text()))
            count=plan['evaluation_chunk_pairs']
            path=run/f'seed-{seed}/evaluation/{endpoint}-{opponent}-{first:04d}.json'
            coords={'seed':seed,'endpoint':endpoint,'opponent':opponent,'first_pair':first,'pairs':count}
            if path.exists():
                body=json.loads(path.read_text());sha=body.pop('content_sha256')
                if hashlib.sha256(canonical(body)).hexdigest()!=sha: raise ValueError('Evaluation digest mismatch')
                if body['input_hash']!=input_hash or body['parameter_hash']!=parameter_hash(parameters) or body['coordinates']!=coords:
                    raise ValueError('Evaluation cache binding mismatch')
            else:
                started=time.perf_counter()
                collection,profile=pool.collect(parameters,evaluation_specs(plan,seed,opponent,first,count))
                if any(rows for rows,_ in collection): raise ValueError('Evaluation generated learner rows')
                games=[g for _,g in collection]
                check_games(games,split='development',first=first,count=count,opponent=opponent)
                admission(run,json.loads((run/'inputs/manifest.json').read_text()))
                body={'input_hash':input_hash,'parameter_hash':parameter_hash(parameters),'coordinates':coords,
                    'games':games,'collection_profile':profile,'collection_and_admission_seconds':time.perf_counter()-started}
                atomic_json(path,{**body,'content_sha256':hashlib.sha256(canonical(body)).hexdigest()})
            check_games(body['games'],split='development',first=first,count=count,opponent=opponent)
            records.extend(body['games'])
            atomic_json(run/'progress.json',{'stage':'evaluation','seed':seed,'endpoint':endpoint,
                'opponent':opponent,'pairs_completed':first+count,'pairs':plan['evaluation_pairs_per_opponent']})
        results[opponent]=records
    return results


def validate_checkpoint(run,seed,plan,input_hash):
    parameters,opt,meta=load_checkpoint(run/f'seed-{seed}/checkpoints',input_hash)
    if meta['seed']!=seed or meta['next_episode']!=meta['cohorts']*8 or meta['cohorts']!=len(meta['cohort_records']):
        raise ValueError('Checkpoint counter/seed mismatch')
    if not 0<=meta['next_episode']<=plan['train_games_per_seed']: raise ValueError('Checkpoint exceeds budget')
    steps=0;previous=parameter_hash(initialize(seed))
    for i,record in enumerate(meta['cohort_records']):
        path=run/record['file']
        if digest(path)!=record['sha256']: raise ValueError('Committed evidence drift')
        cohort=json.loads(path.read_text())
        pointer=cohort['input_checkpoint']
        if Path(pointer['file']).name!=pointer['file'] or digest(run/f'seed-{seed}/checkpoints'/pointer['file'])!=pointer['sha256']:
            raise ValueError('Cohort input-checkpoint binding mismatch')
        if cohort['seed']!=seed or cohort['first_episode']!=i*8 or cohort['input_hash']!=input_hash or cohort['parameter_hash_before']!=previous:
            raise ValueError('Cohort linkage mismatch')
        check_games(cohort['episodes'],split='train',first=i*8,count=8)
        if cohort['decisions']!=sum(g['decisions'] for g in cohort['episodes']): raise ValueError('Cohort decisions mismatch')
        if len(cohort['metrics'])!=PPO['epochs']*((cohort['decisions']+PPO['minibatch_rows']-1)//PPO['minibatch_rows']):
            raise ValueError('Cohort update-count mismatch')
        steps+=len(cohort['metrics']);previous=cohort['parameter_hash_after']
    if steps!=int(opt[2]) or previous!=parameter_hash(parameters): raise ValueError('Optimizer/model linkage mismatch')
    return parameters,opt,meta


def run_campaign(run,plan,resume=False):
    validate_plan(plan)
    run=Path(run).resolve()
    if run.exists() and not resume: raise ValueError('Run exists; explicit unchanged-source resume required')
    run.mkdir(parents=True,exist_ok=True)
    with open(run/'run.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _locked(run,plan,resume)


def _locked(run,plan,resume):
    started=time.perf_counter()
    if resume:
        manifest=json.loads((run/'inputs/manifest.json').read_text())
        if manifest['plan']!=plan: raise ValueError('Resume plan mismatch')
    else:
        manifest=freeze(run,plan)
        manifest['schema_version']='f4-inputs-v1'
        manifest['threefry_partitionable']=True
        atomic_json(run/'inputs/manifest.json',manifest)
    admission(run,manifest)
    input_hash=digest(run/'inputs/manifest.json')
    if (run/'report.json').exists(): raise ValueError('Completed run cannot be extended')
    attempt=str(time.time_ns());results=[]
    try:
        with Pool(run/f'transport/{attempt}',8) as pool:
            commit=json.loads((ROOT/'showdown.pin.json').read_text())['commit']
            if any(w.hello['reference_commit']!=commit for w in pool.workers): raise ValueError('Wrong reference build')
            for seed in plan['seeds']:
                initial=initialize(seed);checkpoints=run/f'seed-{seed}/checkpoints'
                if (checkpoints/'latest.json').exists():
                    parameters,opt,meta=validate_checkpoint(run,seed,plan,input_hash)
                else:
                    parameters=initial;zeros={k:jnp.zeros_like(v) for k,v in parameters.items()}
                    opt=(zeros,dict(zeros),jnp.int32(0))
                    meta={'seed':seed,'input_hash':input_hash,'cohorts':0,'next_episode':0,'cohort_records':[]}
                    save_checkpoint(checkpoints,parameters,opt,meta)
                before=evaluate(run,pool,initial,seed,'initial',plan,input_hash)
                for first in range(meta['next_episode'],plan['train_games_per_seed'],8):
                    cycle=time.perf_counter();admission(run,manifest)
                    old_hash=parameter_hash(parameters)
                    input_checkpoint=json.loads((checkpoints/'latest.json').read_text())
                    collection,profile=pool.collect(parameters,[dict(seed=seed,episode=i,max_turns=plan['max_turns']) for i in range(first,first+8)])
                    games=[g for _,g in collection];check_games(games,split='train',first=first,count=8)
                    rows=merge_rows(collection)
                    if len(rows)!=sum(g['decisions'] for g in games): raise ValueError('Training row count mismatch')
                    t0=time.perf_counter()
                    parameters,opt,metrics=optimize(parameters,opt,rows,derived_seed('f4-update',seed,first))
                    update_seconds=time.perf_counter()-t0
                    if parameter_hash(parameters)==old_hash: raise ValueError('Nonempty update left parameters unchanged')
                    admission(run,manifest)
                    cohort={'input_hash':input_hash,'input_checkpoint':input_checkpoint,'seed':seed,'first_episode':first,'episodes':games,'decisions':len(rows),
                        'collection_profile':profile,'update_seconds':update_seconds,'metrics':metrics,
                        'parameter_hash_before':old_hash,'parameter_hash_after':parameter_hash(parameters),
                        'compute_and_admission_seconds':time.perf_counter()-cycle,
                        'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    path=run/f'seed-{seed}/cohorts/{first:05d}-{time.time_ns()}.json'
                    atomic_json(path,cohort)
                    meta={**meta,'next_episode':first+8,'cohorts':meta['cohorts']+1,
                        'cohort_records':[*meta['cohort_records'],{'file':str(path.relative_to(run)),'sha256':digest(path)}]}
                    save_checkpoint(checkpoints,parameters,opt,meta)
                    elapsed=time.perf_counter()-cycle
                    atomic_json(run/f'seed-{seed}/cycle-telemetry/{first:05d}.json',{
                        'first_episode':first,'full_cycle_seconds':elapsed,'decisions':len(rows),
                        'decisions_per_full_cycle_second':len(rows)/elapsed,'optimizer_steps':int(opt[2])})
                    atomic_json(run/'progress.json',{'stage':'training','seed':seed,'games_completed':first+8,
                        'games':plan['train_games_per_seed'],'optimizer_steps':int(opt[2]),
                        'decisions_per_full_cycle_second':len(rows)/elapsed})
                restored,ro,rm=validate_checkpoint(run,seed,plan,input_hash)
                if parameter_hash(restored)!=parameter_hash(parameters) or int(ro[2])!=int(opt[2]): raise ValueError('Checkpoint roundtrip mismatch')
                after=evaluate(run,pool,restored,seed,'final',plan,input_hash)
                cohorts=[json.loads((run/ref['file']).read_text()) for ref in rm['cohort_records']]
                train=[g for c in cohorts for g in c['episodes']]
                telemetry=[json.loads(p.read_text()) for p in sorted((run/f'seed-{seed}/cycle-telemetry').glob('*.json'))]
                result={'seed':seed,'initial_parameter_hash':parameter_hash(initial),'final_parameter_hash':parameter_hash(restored),
                    'optimizer_steps':int(ro[2]),'training':value_summary(train),
                    'training_collection_seconds':sum(c['collection_profile']['collection_seconds'] for c in cohorts),
                    'training_update_seconds':sum(c['update_seconds'] for c in cohorts),
                    'recorded_complete_cycle_seconds':sum(t['full_cycle_seconds'] for t in telemetry),
                    'complete_cycle_records':len(telemetry),'expected_cycle_records':len(cohorts),
                    'checkpoint_roundtrip_verified':True,
                    'evaluation':{o:compare(before[o],after[o],plan['statistics']) for o in plan['opponents']}}
                results.append(result)
                atomic_json(run/f'seed-{seed}/summary.json',result)
        admission(run,manifest)
        report={'schema_version':'full-learning-f4-report-v1','status':'completed','profile':plan['profile'],
            'input_hash':input_hash,'plan':plan,'seeds':results,'learning_gate':learning_gate(results,plan['statistics']),
            'current_process_seconds':time.perf_counter()-started,'resumed':resume,
            'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'limitations':['Approximate bootstrap coverage; report rigorous Hoeffding bounds alongside it.',
                'Repeated development teams across seeds/opponents; no species-family holdout or public ladder claim.',
                'Terminal-only value error conditions on policy-dependent terminal outcomes.',
                'Current F1 compact features and model; no curriculum, league or search.',
                'Complete-cycle telemetry is non-authoritative and may be absent immediately after an interrupted commit; missing timings are not imputed.',
                'Parent RSS is not combined parent-plus-eight-workers memory or a sustained stability qualification.']}
        atomic_json(run/'report.json',report)
        atomic_json(run/'progress.json',{'stage':'completed','learning_gate_passed':report['learning_gate']['passed']})
        return report
    except BaseException as exc:
        atomic_json(run/f'failures/{attempt}.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'input_hash':input_hash,'elapsed_seconds':time.perf_counter()-started,'resume':resume})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--plan',default=str(ROOT/'contracts/full-learning-v1/plan.json'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    result=run_campaign(args.run,json.loads(Path(args.plan).read_text()),args.resume)
    print(json.dumps({'status':result['status'],'learning_gate':result['learning_gate']}),flush=True)
