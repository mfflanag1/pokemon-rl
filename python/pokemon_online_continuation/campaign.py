"""Fixed teacher-free F39 continuation with unchanged F34 PPO and F35 local collection."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
from pokemon_fit.data import fingerprint
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed, prepare_batch
from pokemon_full_train.campaign import value_summary
from pokemon_full_train.storage import atomic_json, digest
from pokemon_throughput.collector import merge_rows
from pokemon_transfer.common import select
from pokemon_structured_policy.checkpoint import save, load
from pokemon_structured_policy.learning import optimize_batch
from pokemon_structured_policy.network import PARAMETER_COUNT
from pokemon_teacher_assisted.collector import Pool
from .common import (SEEDS, STREAMS, deadline, training_specs, evaluation_specs,
                     admit_games, state_hash, verify_inputs, load_initial)


def compact(game):
    return {k: v for k, v in game.items() if k not in ('trace', 'actor_trace')}


def annotate_selection(collection, selection):
    """Retain rejected decisions in trajectories and count their selected rows."""
    selected, offset = set(selection['selected_indices']), 0
    counts = Counter(rejected_attempts=0, rejected_controlled_attempts=0, rejected_selected_rows=0)
    for rows, game in collection:
        if (game['eligible_learner_rows'] != len(rows)
                or game['accepted_decisions'] + len(game['rejected_attempts']) != game['decisions']
                or len(game['rejected_attempts']) != game['unavailable_choices']):
            raise ValueError('Complete game row/rejection accounting mismatch')
        for attempt in game['rejected_attempts']:
            row = attempt['learner_row_index']
            if row is not None and (type(row) is not int or not 0 <= row < len(rows)):
                raise ValueError('Rejected learner row lies outside its complete game')
            attempt['selected_for_update'] = offset + row in selected if row is not None else False
            counts['rejected_attempts'] += 1
            counts['rejected_controlled_attempts'] += int(attempt['controlled'])
            counts['rejected_selected_rows'] += int(attempt['selected_for_update'])
        offset += len(rows)
    if offset != selection['collected_rows']:
        raise ValueError('Selection/complete trajectory row accounting mismatch')
    return dict(counts)


def collect_cohort(pool, parameters, plan, seed, cohort, cursors, quarantine):
    batches, strata, games, profiles, samples = [], [], [], [], []
    for label, opponent in STREAMS.items():
        quota = plan['stream_selected_rows'][label]
        stream, first, row_count = [], cursors[label], 0
        while row_count < quota:
            deadline(plan)
            specs = training_specs(plan, seed, cursors[label], opponent)
            if cursors[label] - first + len(specs) > plan['maximum_games_per_stream']:
                raise ValueError('Fixed complete-game stream cap exhausted')
            chunk, profile = pool.collect(parameters, specs)
            admit_games([game for _, game in chunk], 'train', quarantine)
            if len(chunk) != len(specs) or any(game['opponent'] != opponent for _, game in chunk):
                raise ValueError('Training stream inventory')
            if any(game[key] != spec[key] for spec, (_, game) in zip(specs, chunk)
                   for key in ('episode', 'swap', 'learner_side', 'opponent')):
                raise ValueError('Training schedule identity')
            if not stream and cohort in plan['sampled_update_cohorts']:
                samples.append({'spec': specs[0], 'game': chunk[0][1], 'stream': label,
                    'rows_sha256': fingerprint(prepare_batch(merge_rows([chunk[0]])))})
            stream.extend(chunk)
            row_count += sum(len(rows) for rows, _ in chunk)
            profiles.append(profile)
            cursors[label] += len(specs)
        selection_seed = derived_seed(plan['namespaces']['row_selection'], seed, label)
        selected, selection = select(stream, selection_seed, cohort, quota)
        rejection_selection = annotate_selection(stream, selection)
        batches.append(selected)
        strata.append({'stream': label, 'opponent': opponent, 'first_episode': first,
            'next_episode': cursors[label], 'games': len(stream), 'selection': selection,
            'selection_seed': selection_seed, 'rejection_selection': rejection_selection})
        games.extend(compact(game) for _, game in stream)
    batch = {key: np.concatenate([part[key] for part in batches]) for key in batches[0]}
    if len(batch['actions']) != 1024:
        raise ValueError('Fixed selected row count')
    return batch, {'strata': strata, 'games': games, 'samples': samples,
        'batch_sha256': fingerprint(batch)}, profiles


def evaluate(pool, parameters, plan, seed, folder, quarantine):
    panels, profiles = {}, []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    for opponent, pairs in plan['evaluation_pairs'].items():
        games = []
        for first in range(0, pairs, 4):
            deadline(plan)
            count = min(4, pairs - first)
            specs = evaluation_specs(plan, seed, opponent, first, count)
            chunk, profile = pool.collect(parameters, specs)
            if any(rows for rows, _ in chunk):
                raise ValueError('Evaluation generated learner rows')
            current = [game for _, game in chunk]
            admit_games(current, 'development', quarantine)
            if len(current) != 2 * count or len(current) != len(specs):
                raise ValueError('Evaluation pair count')
            for spec, game in zip(specs, current):
                if any(game[key] != spec[key] for key in ('episode', 'swap', 'learner_side', 'opponent')):
                    raise ValueError('Evaluation schedule identity')
            atomic_json(folder / f'{opponent}-{first:03d}.json', {'specs': specs, 'games': current})
            games.extend(current)
            profiles.append(profile)
        panels[opponent] = games
    atomic_json(folder / 'summary.json', {opponent: value_summary(games) for opponent, games in panels.items()})
    return panels, profiles


def peak_parent_rss_mib():
    divisor = 1024**2 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def run(run_path, admission_path):
    """A single fixed execution; failures are retained and never resumed or replaced."""
    from .reporting import analyze
    run_path, admission_path = Path(run_path), Path(admission_path)
    admission = verify_inputs(admission_path)
    plan = admission['plan']
    deadline(plan)
    run_path.mkdir(parents=True, exist_ok=False)
    shutil.copytree(admission_path, run_path / 'inputs')
    input_hash = digest(run_path / 'inputs/manifest.json')
    quarantine = set(plan['quarantined_team_keys'])
    begun, profiles, counts, seeds = time.perf_counter(), [], Counter(), {}
    update_seconds = replay_update_seconds = 0.
    try:
        for seed in SEEDS:
            deadline(plan)
            parameters, optimizer, catalog, frozen, identity = load_initial(seed)
            initial_parameter_hash = parameter_hash(parameters)
            initial_state_hash = state_hash(parameters, optimizer)
            initial_cursors = dict(identity['f34_metadata']['stream_cursors'])
            cursors = dict(initial_cursors)
            base = run_path / str(seed)
            base.mkdir()

            def checkpoint(cohort):
                metadata = {'input_hash': input_hash, 'seed': seed, 'cohorts': cohort,
                    'next_episode': sum(cursors.values()), 'stream_cursors': dict(cursors),
                    'selected_rows': cohort * 1024, 'row_presentations': cohort * 2048,
                    'continuation_cohorts': cohort - 512, 'starting_parameter_hash': initial_parameter_hash,
                    'starting_state_hash': initial_state_hash, 'adam_reset': False}
                folder = base / 'checkpoints' / str(cohort)
                save(folder, parameters, optimizer, metadata)
                recovered, recovered_optimizer, _ = load(folder, metadata)
                if state_hash(parameters, optimizer) != state_hash(recovered, recovered_optimizer):
                    raise ValueError('Checkpoint round-trip mismatch')
                counts['checkpoint_comparisons'] += 1

            checkpoint(512)
            training, stream_counts = [], {}
            with Pool(base / 'workers', plan['workers'], catalog, frozen) as pool:
                initial, ps = evaluate(pool, parameters, plan, seed, base / 'initial', quarantine)
                profiles.extend(ps)
                counts['evaluation_games'] += sum(map(len, initial.values()))
                for cohort in range(plan['initial_cohorts_per_seed'], plan['final_cohorts_per_seed']):
                    deadline(plan)
                    if cohort in plan['sampled_update_cohorts'] and cohort != 512:
                        checkpoint(cohort)
                    batch, ledger, ps = collect_cohort(pool, parameters, plan, seed, cohort, cursors, quarantine)
                    profiles.extend(ps)
                    before = parameter_hash(parameters)
                    before_state = state_hash(parameters, optimizer)
                    optimizer_seed = derived_seed(plan['namespaces']['optimizer'], seed, cohort)
                    sampled = cohort in plan['sampled_update_cohorts']
                    if sampled:
                        np.savez_compressed(base / f'batch-{cohort:04d}.npz', **batch)
                        recovered, recovered_optimizer, _ = load(base / 'checkpoints' / str(cohort),
                            {'input_hash': input_hash, 'seed': seed, 'cohorts': cohort})
                        if before_state != state_hash(recovered, recovered_optimizer):
                            raise ValueError('Original recovery state mismatch')
                    deadline(plan)
                    started = time.perf_counter()
                    parameters, optimizer, metrics = optimize_batch(parameters, optimizer, batch, optimizer_seed)
                    cohort_update_seconds = time.perf_counter() - started
                    update_seconds += cohort_update_seconds
                    cohort_replay_seconds = 0.
                    if int(optimizer[2]) != (cohort + 1) * 8 or len(metrics) != 8:
                        raise ValueError('Optimizer cumulative counter linkage')
                    if sampled:
                        deadline(plan)
                        replay_started = time.perf_counter()
                        recovered, recovered_optimizer, replay_metrics = optimize_batch(
                            recovered, recovered_optimizer, batch, optimizer_seed)
                        cohort_replay_seconds = time.perf_counter() - replay_started
                        replay_update_seconds += cohort_replay_seconds
                        if state_hash(recovered, recovered_optimizer) != state_hash(parameters, optimizer) or metrics != replay_metrics:
                            raise ValueError('Original-next-update mismatch')
                        counts['original_next_update_recoveries'] += 1
                    record = {'cohort': cohort, 'optimizer_seed': optimizer_seed,
                        'parameter_hash_before': before, 'state_hash_before': before_state,
                        'parameter_hash_after': parameter_hash(parameters),
                        'state_hash_after': state_hash(parameters, optimizer), 'optimizer_step': int(optimizer[2]),
                        'ledger': ledger, 'metrics': metrics, 'sampled_update_verified': sampled,
                        'update_seconds_including_compile': cohort_update_seconds,
                        'sampled_replay_update_seconds': cohort_replay_seconds}
                    atomic_json(base / f'update-{cohort:04d}.json', record)
                    training.extend(ledger['games'])
                    for stratum in ledger['strata']:
                        sc = stream_counts.setdefault(stratum['stream'], Counter())
                        sc.update({key: stratum['selection'][key] for key in
                            ('collected_rows', 'selected_rows', 'excluded_rows')})
                        sc.update(stratum['rejection_selection'])
                        sc['games'] += stratum['games']
                    counts.update(cohorts=1, optimizer_steps=8, selected_rows=1024,
                                  row_presentations=2048, training_games=len(ledger['games']))
                    atomic_json(run_path / 'progress.json', {'stage': 'training', 'seed': seed,
                        'cohorts_for_seed': cohort + 1 - 512, 'cumulative_cohorts_for_seed': cohort + 1,
                        'cohorts_total': counts['cohorts'], 'elapsed_seconds': time.perf_counter() - begun})
                checkpoint(768)
                final, ps = evaluate(pool, parameters, plan, seed, base / 'final', quarantine)
                profiles.extend(ps)
                counts['evaluation_games'] += sum(map(len, final.values()))
            result = {'seed': seed, 'identity': identity,
                'initial_parameter_hash': initial_parameter_hash, 'initial_state_hash': initial_state_hash,
                'final_parameter_hash': parameter_hash(parameters), 'final_state_hash': state_hash(parameters, optimizer),
                'parameter_count': PARAMETER_COUNT, 'initial_optimizer_step': 4096,
                'final_optimizer_step': int(optimizer[2]), 'optimizer_steps': int(optimizer[2]),
                'continuation_optimizer_steps': 2048, 'cumulative_selected_rows': 786432,
                'cumulative_row_presentations': 1572864, 'initial_stream_cursors': initial_cursors,
                'stream_cursors': dict(cursors), 'stream_counts': {key: dict(v) for key, v in stream_counts.items()},
                'training': value_summary(training),
                'training_unavailable_choices': sum(game['unavailable_choices'] for game in training),
                'initial': {opponent: value_summary(games) for opponent, games in initial.items()},
                'final': {opponent: value_summary(games) for opponent, games in final.items()},
                'analysis': analyze(initial, final, plan['statistics'])}
            atomic_json(base / 'result.json', result)
            seeds[str(seed)] = result
        if any(counts[key] != value for key, value in plan['expected_counts'].items()):
            raise ValueError('Fixed count mismatch: ' + str(dict(counts)))
        deadline(plan)
        verify_inputs(run_path / 'inputs')
        if digest(run_path / 'inputs/manifest.json') != input_hash:
            raise ValueError('Input manifest identity changed during training')
        report = {'schema_version': 'f39-online-continuation-run-v1', 'status': 'passed',
            'input_hash': input_hash, 'plan': plan, 'counts': dict(counts), 'seeds': seeds,
            'all_seed_learning_passed': all(row['analysis']['primary_gate']['passed'] for row in seeds.values()),
            'elapsed_seconds': time.perf_counter() - begun,
            'collection_seconds': sum(profile['collection_seconds'] for profile in profiles),
            'update_seconds_including_compile': update_seconds,
            'sampled_replay_update_seconds': replay_update_seconds, 'peak_parent_rss_mib': peak_parent_rss_mib(),
            'memory_scope': 'Parent process high-water RSS only; excludes worker processes and total machine memory.',
            'profiles': profiles, 'public_server_used': False, 'final_reserved_data_used': False,
            'teacher_used': False, 'demonstrations_used': False,
            'scope': 'Fixed continuation of all F34 final parameters, Adam moments and cursors; fresh local development schedules.'}
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
