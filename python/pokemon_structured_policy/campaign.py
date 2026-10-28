"""One preregistered F34 candidate: fixed fresh-data PPO and two endpoints."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import time

import numpy as np
from pokemon_fit.data import fingerprint
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.campaign import value_summary
from pokemon_full_train.storage import ROOT, atomic_json, digest
from pokemon_transfer.common import select
from .checkpoint import load_f32, save, load
from .collector import Pool
from .common import (SEEDS, read, before_deadline, training_specs, evaluation_specs,
                     admit_games, state_hash, learning_analysis, verify_inputs)
from .network import extend, PARAMETER_COUNT
from .learning import optimize_batch


def compact(game):
    return {k: v for k, v in game.items() if k not in ('trace', 'actor_trace')}


def collect_cohort(pool, parameters, plan, seed, cohort, cursors, quarantine):
    batches, strata, games, profiles, sample = [], [], [], [], []
    for label, opponent, quota in [('self_play', None, 256), ('frozen_f32', 'frozen_f32', 256),
                                    ('tactical', 'tactical', 512)]:
        stream, first = [], cursors[label]
        while sum(len(rows) for rows, _ in stream) < quota:
            before_deadline()
            if cursors[label] - first >= plan['maximum_games_per_stream']:
                raise ValueError('Fixed complete-game stream cap exhausted')
            specs = training_specs(plan, seed, cursors[label], opponent)
            chunk, profile = pool.collect(parameters, specs)
            admit_games([g for _, g in chunk], 'train', quarantine)
            if len(chunk) != len(specs) or any(g['opponent'] != opponent for _, g in chunk):
                raise ValueError('Training stream inventory')
            if any(g['episode'] != s['episode'] or g['learner_side'] != s['learner_side']
                   for s, (_, g) in zip(specs, chunk)):
                raise ValueError('Training schedule identity')
            if not stream and cohort in plan['sampled_update_cohorts']:
                sample.append({'spec': specs[0], 'game': chunk[0][1], 'stream': label})
            stream.extend(chunk)
            profiles.append(profile)
            cursors[label] += len(specs)
        selected, selection = select(stream, derived_seed('f34-row-selection-v1', seed, label), cohort, quota)
        batches.append(selected)
        strata.append({'stream': label, 'opponent': opponent, 'first_episode': first,
            'next_episode': cursors[label], 'games': len(stream), 'selection': selection})
        games.extend(compact(g) for _, g in stream)
    batch = {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}
    if len(batch['actions']) != 1024:
        raise ValueError('Fixed selected row count')
    return batch, {'strata': strata, 'games': games, 'samples': sample,
        'batch_sha256': fingerprint(batch)}, profiles


def evaluate(pool, parameters, plan, seed, folder, quarantine):
    panels, profiles = {}, []
    folder.mkdir(parents=True, exist_ok=False)
    for opponent, pairs in plan['evaluation_pairs'].items():
        games = []
        for first in range(0, pairs, 4):
            before_deadline()
            specs = evaluation_specs(plan, seed, opponent, first, 4)
            chunk, profile = pool.collect(parameters, specs)
            if any(rows for rows, _ in chunk):
                raise ValueError('Evaluation generated learner rows')
            current = [g for _, g in chunk]
            admit_games(current, 'development', quarantine)
            for spec, game in zip(specs, current):
                if any(game[k] != spec[k] for k in ('episode', 'swap', 'learner_side', 'opponent')):
                    raise ValueError('Evaluation schedule identity')
            if len(current) != 8:
                raise ValueError('Evaluation pair count')
            # All evaluation decisions are small enough to retain; replay is sampled.
            atomic_json(folder / f'{opponent}-{first:03d}.json', {'specs': specs, 'games': current})
            games.extend(current)
            profiles.append(profile)
        panels[opponent] = games
    atomic_json(folder / 'summary.json', {o: value_summary(gs) for o, gs in panels.items()})
    return panels, profiles


def run(run_path, admission_path):
    before_deadline()
    run_path, admission_path = Path(run_path), Path(admission_path)
    admission = verify_inputs(admission_path)
    plan = admission['plan']
    if plan['seeds'] != list(SEEDS) or plan['selected_rows_per_cohort'] != 1024:
        raise ValueError('F34 fixed seeds/row budget')
    if plan['stream_selected_rows'] != {'self_play': 256, 'frozen_f32': 256, 'tactical': 512}:
        raise ValueError('F34 selected row mixture')
    run_path.mkdir(parents=True, exist_ok=False)
    # Preserve the exact admitted bytes independently of the mutable review folder.
    import shutil
    shutil.copytree(admission_path, run_path / 'inputs')
    input_hash = digest(run_path / 'inputs/manifest.json')
    quarantine = set(plan['quarantined_team_keys'])
    begun, profiles, counts, seeds = time.perf_counter(), [], Counter(), {}
    update_seconds = replay_update_seconds = 0.
    try:
        for seed in plan['seeds']:
            before_deadline()
            host, catalog, identity = load_f32(seed)
            parameters, optimizer = extend(host, derived_seed('f34-structured-initialization-v1', seed))
            cursors = {label: 0 for label in plan['stream_selected_rows']}
            base = run_path / str(seed)
            base.mkdir()
            def checkpoint(cohort):
                metadata = {'input_hash': input_hash, 'seed': seed, 'cohorts': cohort,
                    'next_episode': sum(cursors.values()), 'stream_cursors': dict(cursors),
                    'selected_rows': cohort * 1024}
                folder = base / 'checkpoints' / str(cohort)
                save(folder, parameters, optimizer, metadata)
                q, qo, _ = load(folder, metadata)
                if state_hash(parameters, optimizer) != state_hash(q, qo):
                    raise ValueError('Checkpoint round-trip mismatch')
                counts['checkpoint_comparisons'] += 1
            checkpoint(0)
            training, initial_hash = [], parameter_hash(parameters)
            with Pool(base / 'workers', plan['workers'], catalog, host) as pool:
                initial, ps = evaluate(pool, parameters, plan, seed, base / 'initial', quarantine)
                profiles.extend(ps)
                for cohort in range(plan['cohorts_per_seed']):
                    before_deadline()
                    if cohort and cohort in plan['sampled_update_cohorts']:
                        checkpoint(cohort)
                    batch, ledger, ps = collect_cohort(pool, parameters, plan, seed, cohort, cursors, quarantine)
                    profiles.extend(ps)
                    before = parameter_hash(parameters)
                    optimizer_seed = derived_seed('f34-optimizer-v1', seed, cohort)
                    sampled = cohort in plan['sampled_update_cohorts']
                    if sampled:
                        np.savez_compressed(base / f'batch-{cohort:04d}.npz', **batch)
                        q, qo, _ = load(base / 'checkpoints' / str(cohort),
                            {'input_hash': input_hash, 'seed': seed, 'cohorts': cohort})
                        if state_hash(parameters, optimizer) != state_hash(q, qo):
                            raise ValueError('Original recovery state mismatch')
                    started = time.perf_counter()
                    parameters, optimizer, metrics = optimize_batch(parameters, optimizer, batch, optimizer_seed)
                    update_seconds += time.perf_counter() - started
                    if int(optimizer[2]) != (cohort + 1) * 8 or len(metrics) != 8:
                        raise ValueError('Optimizer counter linkage')
                    if sampled:
                        replay_started = time.perf_counter()
                        q, qo, replay_metrics = optimize_batch(q, qo, batch, optimizer_seed)
                        replay_update_seconds += time.perf_counter() - replay_started
                        if state_hash(q, qo) != state_hash(parameters, optimizer) or metrics != replay_metrics:
                            raise ValueError('Original-next-update mismatch')
                        counts['original_next_update_recoveries'] += 1
                    record = {'cohort': cohort, 'optimizer_seed': optimizer_seed,
                        'parameter_hash_before': before, 'parameter_hash_after': parameter_hash(parameters),
                        'state_hash_after': state_hash(parameters, optimizer), 'optimizer_step': int(optimizer[2]),
                        'ledger': ledger, 'metrics': metrics, 'sampled_update_verified': sampled}
                    atomic_json(base / f'update-{cohort:04d}.json', record)
                    training.extend(ledger['games'])
                    counts.update(cohorts=1, optimizer_steps=8, selected_rows=1024, row_presentations=2048,
                                  training_games=len(ledger['games']))
                    atomic_json(run_path / 'progress.json', {'stage': 'training', 'seed': seed,
                        'cohorts_for_seed': cohort + 1, 'cohorts_total': counts['cohorts'],
                        'elapsed_seconds': time.perf_counter() - begun})
                checkpoint(plan['cohorts_per_seed'])
                final, ps = evaluate(pool, parameters, plan, seed, base / 'final', quarantine)
                profiles.extend(ps)
            counts['evaluation_games'] += sum(len(gs) for gs in initial.values()) + sum(len(gs) for gs in final.values())
            result = {'seed': seed, 'identity': identity, 'initial_parameter_hash': initial_hash,
                'final_parameter_hash': parameter_hash(parameters), 'parameter_count': PARAMETER_COUNT,
                'optimizer_steps': int(optimizer[2]), 'stream_cursors': cursors,
                'training': value_summary(training),
                'initial': {o: value_summary(gs) for o, gs in initial.items()},
                'final': {o: value_summary(gs) for o, gs in final.items()},
                'analysis': learning_analysis(initial, final, plan['statistics'])}
            atomic_json(base / 'result.json', result)
            seeds[str(seed)] = result
        expected = plan['expected_counts']
        if any(counts[k] != v for k, v in expected.items()):
            raise ValueError('Fixed count mismatch: ' + str(dict(counts)))
        verify_inputs(run_path / 'inputs')
        if digest(run_path / 'inputs/manifest.json') != input_hash:
            raise ValueError('Input manifest identity changed during training')
        report = {'status': 'passed', 'input_hash': input_hash, 'plan': plan, 'counts': dict(counts),
            'seeds': seeds, 'all_seed_learning_passed': all(x['analysis']['passed'] for x in seeds.values()),
            'elapsed_seconds': time.perf_counter() - begun,
            'collection_seconds': sum(p['collection_seconds'] for p in profiles),
            'update_seconds_including_compile': update_seconds,
            'sampled_replay_update_seconds': replay_update_seconds,
            'peak_parent_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            'profiles': profiles, 'public_server_used': False, 'final_reserved_data_used': False,
            'scope': 'Fresh-game development adaptation from F32; new representation, optimizer reset and opponent mixture are not isolated causal effects.'}
        atomic_json(run_path / 'report.json', report)
        print({'status': 'passed', 'seconds': report['elapsed_seconds'],
               'all_seed_learning_passed': report['all_seed_learning_passed']}, flush=True)
        return report
    except BaseException as exc:
        atomic_json(run_path / 'failure.json', {'status': 'failed', 'error': repr(exc), 'counts': dict(counts),
            'completed_seeds': sorted(seeds), 'elapsed_seconds': time.perf_counter() - begun})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.admission)
