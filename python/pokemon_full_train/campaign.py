"""Run the preregistered local F2 pilot; explicit resume never expands its budget."""
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
from .learning import collect, optimize, derived_seed, PPO
from .storage import ROOT, atomic_json, canonical, digest, freeze, admission, save_checkpoint, load_checkpoint
from .transport import Worker


def validate_plan(plan):
    if plan['version'] != 'full-training-f2-v1' or plan['ppo'] != PPO:
        raise ValueError('Unsupported learner plan')
    if plan['opponents'] != ['random','base_power'] or plan['cohort_games'] != 8:
        raise ValueError('Unsupported opponent/cohort profile')
    for key, low, high in [('train_games_per_seed',8,8192),('evaluation_pairs_per_opponent',2,256),('max_turns',2,240)]:
        if type(plan[key]) is not int or not low <= plan[key] <= high:
            raise ValueError(f'Invalid plan {key}')
    if plan['train_games_per_seed'] % 8 or not plan['seeds'] or len(set(plan['seeds'])) != len(plan['seeds']):
        raise ValueError('Invalid game budget/seeds')
    if not all(type(s) is int and 0 <= s < 2**32 for s in [*plan['seeds'],plan['evaluation_seed']]):
        raise ValueError('Invalid seed')
    if plan.get('curriculum') is not False or plan.get('public_server') is not False:
        raise ValueError('F2 requires official local-only games')


def value_summary(records):
    terminal = [r for r in records if r['outcome']['kind'] == 'terminal']
    return {'games':len(records), 'terminals':len(terminal), 'cutoffs':len(records)-len(terminal),
        'decisions':sum(r['decisions'] for r in records),
        'learner_wins':sum(r['outcome']['rewards'][r['learner_side']]==1 for r in terminal),
        'learner_losses':sum(r['outcome']['rewards'][r['learner_side']]==0 for r in terminal),
        'true_ties':sum(r['outcome']['rewards'][r['learner_side']]==.5 for r in terminal),
        'terminal_game_mean_value_mse': float(np.mean([r['terminal_value_mse'] for r in terminal])) if terminal else None,
        'game_mean_predicted_value':float(np.mean([r['value_mean'] for r in records])) if records else None}


def score_interval(record):
    outcome = record['outcome']
    if outcome['kind'] == 'terminal':
        value = outcome['rewards'][record['learner_side']]
        return value,value
    if outcome['kind'] != 'administrative_cutoff':
        raise ValueError('Unknown evaluation outcome')
    return 0.,1.


def compare(initial, final, cells):
    if len(initial) != len(final) or len(initial) % 2:
        raise ValueError('Unpaired evaluation')
    lows, highs = [], []
    for i in range(0,len(initial),2):
        low = high = 0.
        for a,b in zip(initial[i:i+2], final[i:i+2]):
            for key in ['episode','swap','learner_side','opponent','provenance']:
                if a[key] != b[key]:
                    raise ValueError('Evaluation pair identity mismatch')
            al,ah = score_interval(a); bl,bh = score_interval(b)
            low += (bl-ah)/2; high += (bh-al)/2
        lows.append(low); highs.append(high)
    # Independent paired team/mechanics clusters; each difference is in [-1,1].
    radius = float(np.sqrt(2*np.log(2*cells/.05)/len(lows)))
    lower = max(-1.,float(np.mean(lows))-radius)
    upper = min(1.,float(np.mean(highs))+radius)
    return {'pairs':len(lows), 'paired_improvement_interval':[float(np.mean(lows)),float(np.mean(highs))],
        'simultaneous_95pct_hoeffding_interval':[lower,upper],
        'gate':'improvement_established' if lower > 0 else 'regression_established' if upper < 0 else 'inconclusive',
        'initial':value_summary(initial), 'final':value_summary(final)}


def evaluate(run, worker, parameters, seed, endpoint, plan, input_hash):
    results = {}
    for opponent in plan['opponents']:
        records = []
        for episode in range(plan['evaluation_pairs_per_opponent']):
            path = run/f'seed-{seed}/evaluation/{endpoint}-{opponent}-{episode:04d}.json'
            coordinates = {'seed':seed,'endpoint':endpoint,'opponent':opponent,'episode':episode}
            if path.exists():
                body = json.loads(path.read_text())
                expected = body.pop('content_sha256')
                if hashlib.sha256(canonical(body)).hexdigest() != expected or body['input_hash'] != input_hash or body['parameter_hash'] != parameter_hash(parameters):
                    raise ValueError('Evaluation record binding mismatch')
            else:
                games = []
                for swap in [False,True]:
                    _, report = collect(worker, parameters, seed=plan['evaluation_seed'], episode=episode,
                        max_turns=plan['max_turns'], split='development', swap=swap, opponent=opponent,
                        learner_side='p2' if swap else 'p1', actor_seed=derived_seed('eval',seed,opponent))
                    games.append(report)
                body = {'input_hash':input_hash, 'parameter_hash':parameter_hash(parameters), 'coordinates':coordinates, 'games':games}
                atomic_json(path, {**body,'content_sha256':hashlib.sha256(canonical(body)).hexdigest()})
            if body.get('coordinates') != coordinates or len(body['games']) != 2:
                raise ValueError('Evaluation coordinates mismatch')
            for swap,game in zip([False,True],body['games']):
                if game['episode'] != episode or game['opponent'] != opponent or game['swap'] is not swap or game['learner_side'] != ('p2' if swap else 'p1') or game['split'] != 'development':
                    raise ValueError('Evaluation game identity mismatch')
            records.extend(body['games'])
            atomic_json(run/'progress.json', {'stage':'evaluation','seed':seed,'endpoint':endpoint,
                'opponent':opponent,'completed_pairs':episode+1,'pairs':plan['evaluation_pairs_per_opponent']})
        results[opponent] = records
    return results


def run_campaign(run, plan, resume=False):
    run = Path(run).resolve()
    if run.exists() and not resume:
        raise ValueError('Run already exists; choose a new path or explicit resume')
    run.mkdir(parents=True,exist_ok=True)
    with open(run/'run.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_locked(run,plan,resume)


def _run_locked(run,plan,resume):
    started = time.perf_counter()
    if resume:
        manifest = json.loads((run/'inputs/manifest.json').read_text())
        if plan != manifest['plan']:
            raise ValueError('Resume plan mismatch')
    else:
        manifest = freeze(run,plan)
    admission(run,manifest)
    input_hash = digest(run/'inputs/manifest.json')
    if (run/'report.json').exists():
        raise ValueError('Completed run cannot be resumed or extended')
    attempt = str(time.time_ns())
    results = []
    try:
        with Worker(run/'transport') as worker:
            if worker.hello['reference_commit'] != json.loads((ROOT/'showdown.pin.json').read_text())['commit']:
                raise ValueError('Reference identity mismatch')
            for seed in plan['seeds']:
                initial = initialize(seed)
                init_hash = parameter_hash(initial)
                checkpoints = run/f'seed-{seed}/checkpoints'
                if (checkpoints/'latest.json').exists():
                    parameters, optimizer, meta = load_checkpoint(checkpoints,input_hash)
                    if meta['seed'] != seed:
                        raise ValueError('Checkpoint seed mismatch')
                    if meta['next_episode'] != meta['cohorts']*8 or meta['cohorts'] != len(meta['cohort_records']) or not 0 <= meta['next_episode'] <= plan['train_games_per_seed']:
                        raise ValueError('Checkpoint game/counter linkage mismatch')
                    steps = 0
                    for index,record in enumerate(meta['cohort_records']):
                        if digest(run/record['file']) != record['sha256']:
                            raise ValueError('Committed cohort evidence drift')
                        content = json.loads((run/record['file']).read_text())
                        if content['first_episode'] != index*8 or [e['episode'] for e in content['episodes']] != list(range(index*8,index*8+8)):
                            raise ValueError('Committed episode budget/linkage mismatch')
                        steps += len(content['metrics'])
                    if steps != int(optimizer[2]):
                        raise ValueError('Optimizer counter linkage mismatch')
                else:
                    parameters = initial
                    zeros = {k:jnp.zeros_like(v) for k,v in parameters.items()}
                    optimizer = (zeros,dict(zeros),jnp.int32(0))
                    meta = {'seed':seed,'input_hash':input_hash,'cohorts':0,'next_episode':0,'cohort_records':[]}
                    save_checkpoint(checkpoints,parameters,optimizer,meta)
                before = evaluate(run,worker,initial,seed,'initial',plan,input_hash)
                for first in range(meta['next_episode'],plan['train_games_per_seed'],8):
                    admission(run,manifest)
                    t0 = time.perf_counter()
                    rows, episodes = [], []
                    for episode in range(first,first+8):
                        new, record = collect(worker,parameters,seed=seed,episode=episode,max_turns=plan['max_turns'])
                        offset = len(rows)
                        for row in new:
                            if row['next_index'] >= 0:
                                row['next_index'] += offset
                            row['trajectory'] += episode*2
                        rows.extend(new); episodes.append(record)
                    collected = time.perf_counter()-t0
                    old_hash = parameter_hash(parameters)
                    parameters, optimizer, metrics = optimize(parameters,optimizer,rows,derived_seed('update',seed,first))
                    elapsed = time.perf_counter()-t0
                    if parameter_hash(parameters) == old_hash:
                        raise ValueError('Nonempty update left parameters unchanged')
                    admission(run,manifest)
                    cohort = {'first_episode':first,'games':8,'decisions':len(rows),'collection_seconds':collected,
                        'update_seconds':elapsed-collected,'decisions_per_cycle_second':len(rows)/elapsed,
                        'parameter_hash_before':old_hash,'parameter_hash_after':parameter_hash(parameters),
                        'metrics':metrics,'episodes':episodes}
                    record_path = run/f'seed-{seed}/cohorts/{first:05d}-{time.time_ns()}.json'
                    atomic_json(record_path,cohort)
                    meta = {**meta,'cohorts':meta['cohorts']+1,'next_episode':first+8,
                        'cohort_records':[*meta['cohort_records'],{'file':str(record_path.relative_to(run)),'sha256':digest(record_path)}]}
                    save_checkpoint(checkpoints,parameters,optimizer,meta)
                    atomic_json(run/'progress.json',{'stage':'training','seed':seed,'games':first+8,
                        'budget':plan['train_games_per_seed'],'optimizer_steps':int(optimizer[2]),
                        'decisions_per_cycle_second':len(rows)/elapsed})
                # Reading persisted state is mandatory before evaluating final weights.
                restored, restored_opt, restored_meta = load_checkpoint(checkpoints,input_hash)
                if parameter_hash(restored) != parameter_hash(parameters) or int(restored_opt[2]) != int(optimizer[2]):
                    raise ValueError('Checkpoint roundtrip mismatch')
                after = evaluate(run,worker,restored,seed,'final',plan,input_hash)
                cohort_data = [json.loads((run/r['file']).read_text()) for r in meta['cohort_records']]
                train_episodes = [e for c in cohort_data for e in c['episodes']]
                cells = len(plan['seeds'])*len(plan['opponents'])
                results.append({'seed':seed,'initial_parameter_hash':init_hash,'final_parameter_hash':parameter_hash(restored),
                    'optimizer_steps':int(optimizer[2]),'training':value_summary(train_episodes),
                    'training_collection_seconds':sum(c['collection_seconds'] for c in cohort_data),
                    'training_update_seconds':sum(c['update_seconds'] for c in cohort_data),
                    'checkpoint_roundtrip_verified':True,
                    'evaluation':{o:compare(before[o],after[o],cells) for o in plan['opponents']}})
        admission(run,manifest)
        report = {'schema_version':'full-training-f2-report-v1','status':'completed', 'input_hash':input_hash,
            'plan':plan,'seeds':results,'learning_gate_passed':all(c['gate']=='improvement_established' for r in results for c in r['evaluation'].values()),
            'current_process_seconds':time.perf_counter()-started,
            'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'limitations':['Small exploratory development screen; wide simultaneous confidence intervals.',
                'The same development team/mechanics pairs are reused across all training seeds and opponents; cells are not independent team replications.',
                'Ordered whole-team partition does not imply species or scenario-family disjointness.',
                'Tiny lossy F1 representation; no league, curriculum, search or public ladder measurement.',
                'Cohort-boundary recovery only; incomplete games replay from identical seeds.',
                'Terminal-only value metrics condition on the policy reaching a terminal.']}
        atomic_json(run/'report.json',report)
        atomic_json(run/'progress.json',{'stage':'completed','learning_gate_passed':report['learning_gate_passed']})
        return report
    except BaseException as exc:
        atomic_json(run/f'failures/{attempt}.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),
            'input_hash':input_hash,'resume':resume,'elapsed_seconds':time.perf_counter()-started})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--plan',default=str(ROOT/'contracts/full-training-v1/plan.json'))
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text()); validate_plan(plan)
    result = run_campaign(args.run,plan,args.resume)
    print(json.dumps({'status':result['status'],'learning_gate_passed':result['learning_gate_passed']}),flush=True)


if __name__ == '__main__':
    main()
