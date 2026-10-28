"""F5 fixed optimization budget, atomic update commits, paired development panel."""
import argparse
import fcntl
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
from pokemon_full_train.learning import PPO, derived_seed
from pokemon_full_train.storage import (ROOT, atomic_json, digest, freeze, admission,
                                       save_checkpoint, load_checkpoint)
from pokemon_full_train.campaign import value_summary
from pokemon_full_learning.campaign import evaluate, evaluation_specs, check_games
from .collector import Pool
from .learning import (collect_update, optimize_batch, quotas_for, training_specs,
                       check_training, array_hash)
from .statistics import contrasts, mixture_gate


def read(path):
    return json.loads(Path(path).read_text())


def validate_plan(plan):
    if plan['version'] != 'opponent-mixture-f5-v1' or plan['ppo'] != PPO:
        raise ValueError('Wrong F5 version or PPO settings')
    if plan['arms'] != ['self_play', 'mixture'] or plan['arm_weights'] != {
            'self_play': {'self_play': 1.}, 'mixture': {'self_play': .5, 'base_power': .25, 'tactical': .25}}:
        raise ValueError('Wrong comparison arms')
    if plan['opponents'] != ['random', 'base_power', 'tactical'] or plan['endpoints'] != ['initial', 'self_play', 'mixture']:
        raise ValueError('Wrong evaluation panel')
    if plan['workers'] != 8 or plan['cohort_games'] != 8:
        raise ValueError('Eight-worker batches required')
    for key in ['curriculum', 'public_server', 'final_reserved_data', 'prime']:
        if plan[key] is not False:
            raise ValueError('Unsupported scope')
    for key, lo, hi in [('updates_per_arm', 1, 128), ('selected_rows_per_update', 256, 1024),
                        ('max_games_per_stratum_update', 8, 64), ('max_turns', 2, 120),
                        ('evaluation_pairs_per_opponent', 2, 256), ('evaluation_chunk_pairs', 1, 16)]:
        if type(plan[key]) is not int or not lo <= plan[key] <= hi:
            raise ValueError('Invalid budget: '+key)
    if (plan['selected_rows_per_update'] % 256 or plan['max_games_per_stratum_update'] % 8 or
            plan['evaluation_pairs_per_opponent'] % plan['evaluation_chunk_pairs']):
        raise ValueError('Partial batch budget')
    if not plan['seeds'] or len(set(plan['seeds'])) != len(plan['seeds']):
        raise ValueError('Invalid seeds')
    settings = plan['statistics']
    if any(type(s) is not int or not 0 <= s < 2**32 for s in [*plan['seeds'], plan['evaluation_seed'], settings['seed']]):
        raise ValueError('Invalid random seed')
    if settings['family_cells'] != 9*len(plan['seeds']) or settings['method'] != 'paired-cluster-percentile-bootstrap':
        raise ValueError('Wrong contrast family')
    if settings['alpha'] != .05 or settings['practical_improvement'] != .03 or settings['max_final_cutoff_rate'] != .05:
        raise ValueError('Changed inferential gate')
    if type(settings['resamples']) is not int or not 100 <= settings['resamples'] <= 20000:
        raise ValueError('Invalid bootstrap budget')
    if plan['profile'] == 'production':
        if plan != read(ROOT/'contracts/opponent-mixture-v1/plan.json'):
            raise ValueError('Production must equal registered plan')
    elif plan['profile'] not in ['test', 'preflight']:
        raise ValueError('Unknown profile')
    if jax.default_backend() != 'cpu' or jax.config.x64_enabled or not jax.config.jax_threefry_partitionable:
        raise ValueError('Original CPU FP32 partitionable-Threefry runtime required')


def checkpoint_dir(run, seed, arm):
    return run/f'seed-{seed}/arms/{arm}/checkpoints'


def optimizer_hashes(opt):
    return {'m': parameter_hash(opt[0]), 'v': parameter_hash(opt[1]), 'step': int(opt[2])}


def validate_checkpoint(run, seed, arm, plan, input_hash):
    directory = checkpoint_dir(run, seed, arm)
    parameters, opt, meta = load_checkpoint(directory, input_hash)
    if meta['seed'] != seed or meta['arm'] != arm or meta['cohorts'] != len(meta['cohort_records']):
        raise ValueError('Checkpoint identity mismatch')
    if not 0 <= meta['cohorts'] <= plan['updates_per_arm']:
        raise ValueError('Checkpoint exceeds update budget')
    cursors = {s: 0 for s in quotas_for(plan, arm)}
    steps = 0; previous = parameter_hash(initialize(seed))
    zeros = {k: np.zeros_like(v) for k, v in initialize(seed).items()}
    previous_opt = optimizer_hashes((zeros, zeros, 0))
    for i, ref in enumerate(meta['cohort_records']):
        path = run/ref['file']
        if digest(path) != ref['sha256']:
            raise ValueError('Committed evidence drift')
        c = read(path); pointer = c['input_checkpoint']
        if Path(pointer['file']).name != pointer['file'] or digest(directory/pointer['file']) != pointer['sha256']:
            raise ValueError('Input checkpoint drift')
        if (c['seed'], c['arm'], c['update_index'], c['input_hash'], c['parameter_hash_before']) != (seed, arm, i, input_hash, previous):
            raise ValueError('Update linkage mismatch')
        if c['optimizer_before'] != previous_opt or set(c['strata']) != set(cursors) or set(c['selection']) != set(cursors):
            raise ValueError('Optimizer/strata mismatch')
        for s, quota in quotas_for(plan, arm).items():
            stratum = c['strata'][s]; games = stratum['episodes']; n = len(games)
            if stratum['first_episode'] != cursors[s] or n % 8 or not 8 <= n <= plan['max_games_per_stratum_update']:
                raise ValueError('Stratum coordinate/budget mismatch')
            specs = training_specs(plan, seed, arm, s, cursors[s], n)
            if specs != stratum['specs']:
                raise ValueError('Training seed/role mismatch')
            check_training(games, specs)
            if len(stratum['profiles']) != n//8:
                raise ValueError('Collection batch mismatch')
            selected = c['selection'][s]; available = sum(g['learner_rows'] for g in games)
            if (selected['collected_rows'], selected['selected_rows'], selected['excluded_rows']) != (available, quota, available-quota):
                raise ValueError('Selection budget mismatch')
            ids = selected['selected_indices']
            expected_ids = np.random.default_rng(derived_seed('f5-select', seed, arm, i, s)).choice(available, size=quota, replace=False).tolist()
            if ids != expected_ids or len(set(ids)) != quota:
                raise ValueError('Selection RNG mismatch')
            if sum(g['learner_rows'] for g in games[:-8]) >= quota:
                raise ValueError('Collection continued beyond fixed quota rule')
            cursors[s] += n
        count = PPO['epochs']*plan['selected_rows_per_update']//PPO['minibatch_rows']
        if len(c['metrics']) != count or not all(np.isfinite(v) for m in c['metrics'] for v in m.values()):
            raise ValueError('Wrong or nonfinite optimizer updates')
        steps += count; previous = c['parameter_hash_after']; previous_opt = c['optimizer_after']
        if previous_opt['step'] != steps:
            raise ValueError('Optimizer step mismatch')
    if meta['next_episodes'] != cursors or steps != int(opt[2]) or previous != parameter_hash(parameters) or previous_opt != optimizer_hashes(opt):
        raise ValueError('Final checkpoint linkage mismatch')
    return parameters, opt, meta


def summarize_arm(run, seed, arm, parameters, opt, meta):
    updates = [read(run/ref['file']) for ref in meta['cohort_records']]
    games = [g for c in updates for s in c['strata'].values() for g in s['episodes']]
    telemetry = [read(p) for p in sorted((run/f'seed-{seed}/arms/{arm}/telemetry').glob('*.json'))]
    return {'final_parameter_hash': parameter_hash(parameters), 'optimizer': optimizer_hashes(opt),
        'updates': len(updates), 'training': value_summary(games),
        'collected_learner_rows': sum(g['learner_rows'] for g in games),
        'selected_learner_rows': sum(s['selected_rows'] for c in updates for s in c['selection'].values()),
        'excluded_learner_rows': sum(s['excluded_rows'] for c in updates for s in c['selection'].values()),
        'strata': {s: value_summary([g for c in updates for g in c['strata'][s]['episodes']]) for s in meta['next_episodes']},
        'training_collection_seconds': sum(p['collection_seconds'] for c in updates for s in c['strata'].values() for p in s['profiles']),
        'training_update_seconds': sum(c['update_seconds'] for c in updates),
        'recorded_complete_cycle_seconds': sum(t['full_cycle_seconds'] for t in telemetry),
        'complete_cycle_records': len(telemetry), 'expected_cycle_records': len(updates),
        'checkpoint_roundtrip_verified': True}


def run_campaign(run, plan, resume=False):
    validate_plan(plan)
    run = Path(run).resolve()
    if run.exists() and not resume:
        raise ValueError('Run exists; explicit unchanged-source resume required')
    run.mkdir(parents=True, exist_ok=True)
    with open(run/'run.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _locked(run, plan, resume)


def _locked(run, plan, resume):
    started = time.perf_counter()
    if resume:
        manifest = read(run/'inputs/manifest.json')
        if manifest['plan'] != plan:
            raise ValueError('Resume plan mismatch')
    else:
        manifest = freeze(run, plan)
        manifest.update(schema_version='f5-inputs-v1', threefry_partitionable=True)
        atomic_json(run/'inputs/manifest.json', manifest)
    admission(run, manifest); input_hash = digest(run/'inputs/manifest.json')
    if (run/'report.json').exists():
        raise ValueError('Completed run cannot be extended')
    attempt = str(time.time_ns()); results = []
    try:
        with Pool(run/f'transport/{attempt}', 8) as pool:
            commit = read(ROOT/'showdown.pin.json')['commit']
            if any(w.hello['reference_commit'] != commit for w in pool.workers):
                raise ValueError('Wrong reference build')
            for seed in plan['seeds']:
                initial = initialize(seed)
                before = evaluate(run, pool, initial, seed, 'initial', plan, input_hash)
                summaries, finals = {}, {}
                for arm in plan['arms']:
                    directory = checkpoint_dir(run, seed, arm)
                    if (directory/'latest.json').exists():
                        parameters, opt, meta = validate_checkpoint(run, seed, arm, plan, input_hash)
                    else:
                        parameters = initial; zeros = {k: jnp.zeros_like(v) for k, v in parameters.items()}
                        opt = (zeros, dict(zeros), jnp.int32(0))
                        meta = {'seed': seed, 'arm': arm, 'input_hash': input_hash, 'cohorts': 0,
                                'next_episodes': {s: 0 for s in quotas_for(plan, arm)}, 'cohort_records': []}
                        save_checkpoint(directory, parameters, opt, meta)
                    for i in range(meta['cohorts'], plan['updates_per_arm']):
                        cycle = time.perf_counter(); admission(run, manifest)
                        old_hash = parameter_hash(parameters); old_opt = optimizer_hashes(opt)
                        pointer = read(directory/'latest.json')
                        batch, strata, selection, cursors = collect_update(pool, parameters, plan, seed, arm, i, meta['next_episodes'])
                        t0 = time.perf_counter()
                        parameters, opt, metrics = optimize_batch(parameters, opt, batch, derived_seed('f5-update', seed, i))
                        seconds = time.perf_counter()-t0
                        if parameter_hash(parameters) == old_hash:
                            raise ValueError('Nonempty update left parameters unchanged')
                        admission(run, manifest)
                        cohort = {'seed': seed, 'arm': arm, 'update_index': i, 'input_hash': input_hash,
                            'input_checkpoint': pointer, 'strata': strata, 'selection': selection,
                            'selected_batch_sha256': array_hash(batch), 'metrics': metrics, 'update_seconds': seconds,
                            'parameter_hash_before': old_hash, 'parameter_hash_after': parameter_hash(parameters),
                            'optimizer_before': old_opt, 'optimizer_after': optimizer_hashes(opt),
                            'parent_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                        path = run/f'seed-{seed}/arms/{arm}/updates/{i:04d}-{time.time_ns()}.json'
                        atomic_json(path, cohort)
                        meta = {**meta, 'cohorts': i+1, 'next_episodes': cursors,
                            'cohort_records': [*meta['cohort_records'], {'file': str(path.relative_to(run)), 'sha256': digest(path)}]}
                        save_checkpoint(directory, parameters, opt, meta)
                        elapsed = time.perf_counter()-cycle
                        atomic_json(run/f'seed-{seed}/arms/{arm}/telemetry/{i:04d}.json', {
                            'update_index': i, 'full_cycle_seconds': elapsed,
                            'selected_rows': len(batch['actions']), 'optimizer_steps': int(opt[2])})
                        atomic_json(run/'progress.json', {'stage': 'training', 'seed': seed, 'arm': arm,
                            'updates_completed': i+1, 'updates': plan['updates_per_arm'], 'games_collected': sum(cursors.values()),
                            'selected_rows': (i+1)*plan['selected_rows_per_update'], 'optimizer_steps': int(opt[2])})
                    restored, ro, rm = validate_checkpoint(run, seed, arm, plan, input_hash)
                    if parameter_hash(restored) != parameter_hash(parameters) or optimizer_hashes(ro) != optimizer_hashes(opt):
                        raise ValueError('Checkpoint roundtrip mismatch')
                    finals[arm] = evaluate(run, pool, restored, seed, arm, plan, input_hash)
                    summaries[arm] = summarize_arm(run, seed, arm, restored, ro, rm)
                result = {'seed': seed, 'initial_parameter_hash': parameter_hash(initial), 'arms': summaries,
                          'contrasts': contrasts(before, finals, plan['statistics'])}
                results.append(result); atomic_json(run/f'seed-{seed}/summary.json', result)
        admission(run, manifest)
        report = {'schema_version': 'opponent-mixture-f5-report-v1', 'status': 'completed',
            'profile': plan['profile'], 'input_hash': input_hash, 'plan': plan, 'seeds': results,
            'learning_gate': mixture_gate(results, plan['statistics']), 'current_process_seconds': time.perf_counter()-started,
            'resumed': resume, 'parent_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'limitations': ['Equal selected learner rows and Adam steps, not equal collected interactions, games or wall time.',
                'Complete trajectories contribute GAE context even for rows excluded from direct loss.',
                'Approximate bootstrap coverage; bounded Hoeffding intervals reported separately.',
                'Same development team panel across seeds/opponents; no species/family or public ladder generalization.',
                'Terminal-only value errors condition on policy-dependent terminal outcomes; not calibration.',
                'Training includes the evaluated heuristic identities, on disjoint whole teams; no unseen-opponent generalization.',
                'Parent RSS excludes eight Node workers; no sustained stability qualification.',
                'Complete-cycle timing may be absent after an interrupted commit; missing timing is not imputed.']}
        atomic_json(run/'report.json', report)
        atomic_json(run/'progress.json', {'stage': 'completed', 'learning_gate_passed': report['learning_gate']['passed']})
        return report
    except BaseException as exc:
        atomic_json(run/f'failures/{attempt}.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
            'input_hash': input_hash, 'elapsed_seconds': time.perf_counter()-started, 'resume': resume})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--plan', default=str(ROOT/'contracts/opponent-mixture-v1/plan.json'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run_campaign(args.run, read(args.plan), args.resume)
    print(json.dumps({'status': result['status'], 'learning_gate': result['learning_gate']}), flush=True)
