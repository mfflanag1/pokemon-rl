"""Fixed F35 matched PPO budgets; demonstration loss is isolated to treatment."""
from collections import Counter
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from pokemon_fit.data import fingerprint
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.campaign import value_summary
from pokemon_full_train.storage import atomic_json, digest
from pokemon_transfer.common import select
from pokemon_structured_policy.checkpoint import save, load
from pokemon_structured_policy.learning import optimize_batch as original_optimize
from pokemon_structured_policy.network import PARAMETER_COUNT
from .collector import Pool
from .common import (SEEDS, ARMS, before_deadline, training_specs, evaluation_specs,
                     admit_games, state_hash, verify_inputs, load_initial)
from .learning import optimize_batch as assisted_optimize, probe_metrics


def compact(game):
    return {k: v for k, v in game.items() if k not in ('trace', 'actor_trace')}


def annotate_selection(collection, selection):
    """Map inherited rejected learner rows through complete-game row selection."""
    selected, offset = set(selection['selected_indices']), 0
    counts = Counter()
    for rows, game in collection:
        for attempt in game['rejected_attempts']:
            row = attempt['learner_row_index']
            if row is not None and not 0 <= row < len(rows):
                raise ValueError('Rejected learner row lies outside its complete game')
            attempt['selected_for_update'] = (offset + row in selected) if row is not None else False
            counts['rejected_attempts'] += 1
            counts['rejected_controlled_attempts'] += int(attempt['controlled'])
            counts['rejected_selected_rows'] += int(attempt['selected_for_update'])
        offset += len(rows)
    if offset != selection['collected_rows']:
        raise ValueError('Selection/complete trajectory row accounting mismatch')
    return dict(counts)


def collect_cohort(pool, parameters, plan, seed, cohort, cursors, quarantine):
    batches, strata, games, profiles, samples = [], [], [], [], []
    for label, opponent, quota in [('self_play', None, 256), ('frozen_f32', 'frozen_f32', 256),
                                    ('tactical', 'tactical', 512)]:
        stream, first, row_count = [], cursors[label], 0
        while row_count < quota:
            before_deadline()
            specs = training_specs(plan, seed, cursors[label], opponent)
            if cursors[label] - first + len(specs) > plan['maximum_games_per_stream']:
                raise ValueError('Fixed complete-game stream cap exhausted')
            chunk, profile = pool.collect(parameters, specs)
            admit_games([g for _, g in chunk], 'train', quarantine)
            if len(chunk) != len(specs) or any(g['opponent'] != opponent for _, g in chunk):
                raise ValueError('Training stream inventory')
            if any(g[k] != spec[k] for spec, (_, g) in zip(specs, chunk)
                   for k in ('episode', 'swap', 'learner_side', 'opponent')):
                raise ValueError('Training schedule identity')
            if not stream and cohort in plan['sampled_update_cohorts']:
                samples.append({'spec': specs[0], 'game': chunk[0][1], 'stream': label})
            stream.extend(chunk)
            row_count += sum(len(rows) for rows, _ in chunk)
            profiles.append(profile)
            cursors[label] += len(specs)
        selected, selection = select(stream, derived_seed('f35-row-selection-v1', seed, label), cohort, quota)
        rejection_selection = annotate_selection(stream, selection)
        batches.append(selected)
        strata.append({'stream': label, 'opponent': opponent, 'first_episode': first,
            'next_episode': cursors[label], 'games': len(stream), 'selection': selection,
            'rejection_selection': rejection_selection})
        games.extend(compact(g) for _, g in stream)
    batch = {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}
    if len(batch['actions']) != 1024:
        raise ValueError('Fixed selected row count')
    return batch, {'strata': strata, 'games': games, 'samples': samples,
        'batch_sha256': fingerprint(batch)}, profiles


def evaluate(pool, parameters, plan, seed, folder, quarantine):
    panels, profiles = {}, []
    folder.mkdir(parents=True, exist_ok=False)
    for opponent, pairs in plan['evaluation_pairs'].items():
        games = []
        for first in range(0, pairs, 4):
            before_deadline()
            count = min(4, pairs - first)
            specs = evaluation_specs(plan, seed, opponent, first, count)
            chunk, profile = pool.collect(parameters, specs)
            if any(rows for rows, _ in chunk):
                raise ValueError('Evaluation generated learner rows')
            current = [g for _, g in chunk]
            admit_games(current, 'development', quarantine)
            if len(current) != 2 * count or len(current) != len(specs):
                raise ValueError('Evaluation pair count')
            for spec, game in zip(specs, current):
                if any(game[k] != spec[k] for k in ('episode', 'swap', 'learner_side', 'opponent')):
                    raise ValueError('Evaluation schedule identity')
            atomic_json(folder / f'{opponent}-{first:03d}.json', {'specs': specs, 'games': current})
            games.extend(current)
            profiles.append(profile)
        panels[opponent] = games
    atomic_json(folder / 'summary.json', {o: value_summary(gs) for o, gs in panels.items()})
    return panels, profiles


def optimize(arm, parameters, optimizer, batch, optimizer_seed, dataset, demo_seed, cohort):
    if arm == 'control':
        return original_optimize(parameters, optimizer, batch, optimizer_seed)
    if arm != 'assisted':
        raise ValueError('Unknown F35 arm')
    return assisted_optimize(parameters, optimizer, batch, optimizer_seed, dataset=dataset,
        demo_seed=demo_seed, cohort=cohort, coefficient=.1)


def run(run_path, admission_path, dataset, probe_dataset):
    """Called once by the driver with the validated train-only immutable corpus."""
    before_deadline()
    if dataset.split != 'train' or probe_dataset.split != 'development':
        raise ValueError('Training/probe corpus separation')
    if not dataset.manifest_sha256 or dataset.manifest_sha256 != probe_dataset.manifest_sha256:
        raise ValueError('Training/probe corpus manifest identity')
    run_path, admission_path = Path(run_path), Path(admission_path)
    admission = verify_inputs(admission_path)
    plan = admission['plan']
    if (plan['seeds'] != list(SEEDS) or plan['arms'] != list(ARMS)
            or plan['selected_rows_per_cohort'] != 1024 or plan['cohorts_per_seed_arm'] != 256
            or plan['stream_selected_rows'] != {'self_play': 256, 'frozen_f32': 256, 'tactical': 512}
            or plan['sampled_update_cohorts'] != [0, 255]):
        raise ValueError('F35 fixed seeds, arms, rows, cohorts or sample budget')
    run_path.mkdir(parents=True, exist_ok=False)
    shutil.copytree(admission_path, run_path / 'inputs')
    input_hash = digest(run_path / 'inputs/manifest.json')
    quarantine = set(plan['quarantined_team_keys']) | set(probe_dataset.team_keys)
    begun, profiles, counts, seeds = time.perf_counter(), [], Counter(), {}
    update_seconds = replay_update_seconds = probe_seconds = 0.
    try:
        for seed in plan['seeds']:
            before_deadline()
            initial_parameters, reset_optimizer, catalog, frozen, identity = load_initial(seed)
            initial_state_hash = state_hash(initial_parameters, reset_optimizer)
            initial_parameter_hash = parameter_hash(initial_parameters)
            base = run_path / str(seed)
            base.mkdir()
            probe_started = time.perf_counter()
            initial_probes = {label: probe_metrics(initial_parameters, corpus, reference_parameters=initial_parameters)
                for label, corpus in [('train', dataset), ('probe', probe_dataset)]}
            probe_seconds += time.perf_counter() - probe_started
            atomic_json(base / 'initial-probes.json', initial_probes)
            with Pool(base / 'initial-workers', plan['workers'], catalog, frozen) as pool:
                initial, ps = evaluate(pool, initial_parameters, plan, seed, base / 'initial', quarantine)
                profiles.extend(ps)
            counts['evaluation_games'] += sum(map(len, initial.values()))
            results = {}
            for arm in ARMS:
                parameters, optimizer = initial_parameters, reset_optimizer
                if state_hash(parameters, optimizer) != initial_state_hash or int(optimizer[2]) != 0:
                    raise ValueError('Matched F34 initial state / reset Adam mismatch')
                cursors = {label: 0 for label in plan['stream_selected_rows']}
                arm_path = base / arm
                arm_path.mkdir()
                def checkpoint(cohort):
                    metadata = {'input_hash': input_hash, 'seed': seed, 'arm': arm, 'cohorts': cohort,
                        'next_episode': sum(cursors.values()), 'stream_cursors': dict(cursors),
                        'selected_rows': cohort * 1024, 'starting_parameter_hash': initial_parameter_hash}
                    folder = arm_path / 'checkpoints' / str(cohort)
                    save(folder, parameters, optimizer, metadata)
                    recovered, recovered_optimizer, _ = load(folder, metadata)
                    if state_hash(parameters, optimizer) != state_hash(recovered, recovered_optimizer):
                        raise ValueError('Checkpoint round-trip mismatch')
                    counts['checkpoint_comparisons'] += 1
                checkpoint(0)
                training, stream_counts, arm_profiles = [], {}, []
                arm_update_seconds = arm_recovery_seconds = 0.
                demo_seed = derived_seed('f35-demonstration-minibatches-v1', seed)
                with Pool(arm_path / 'workers', plan['workers'], catalog, frozen) as pool:
                    for cohort in range(plan['cohorts_per_seed_arm']):
                        before_deadline()
                        if cohort in plan['sampled_update_cohorts'] and cohort:
                            checkpoint(cohort)
                        batch, ledger, ps = collect_cohort(pool, parameters, plan, seed, cohort, cursors, quarantine)
                        arm_profiles.extend(ps)
                        before = parameter_hash(parameters)
                        optimizer_seed = derived_seed('f35-optimizer-v1', seed, cohort)
                        sampled = cohort in plan['sampled_update_cohorts']
                        if sampled:
                            np.savez_compressed(arm_path / f'batch-{cohort:04d}.npz', **batch)
                            q, qo, _ = load(arm_path / 'checkpoints' / str(cohort),
                                {'input_hash': input_hash, 'seed': seed, 'arm': arm, 'cohorts': cohort})
                            if state_hash(parameters, optimizer) != state_hash(q, qo):
                                raise ValueError('Original recovery state mismatch')
                        started = time.perf_counter()
                        parameters, optimizer, metrics = optimize(arm, parameters, optimizer, batch,
                            optimizer_seed, dataset, demo_seed, cohort)
                        arm_update_seconds += time.perf_counter() - started
                        if int(optimizer[2]) != (cohort + 1) * 8 or len(metrics) != 8:
                            raise ValueError('Optimizer counter linkage')
                        if sampled:
                            replay_started = time.perf_counter()
                            q, qo, replay_metrics = optimize(arm, q, qo, batch, optimizer_seed,
                                dataset, demo_seed, cohort)
                            arm_recovery_seconds += time.perf_counter() - replay_started
                            if state_hash(q, qo) != state_hash(parameters, optimizer) or metrics != replay_metrics:
                                raise ValueError('Original-next-update mismatch')
                            counts['original_next_update_recoveries'] += 1
                        record = {'cohort': cohort, 'arm': arm, 'optimizer_seed': optimizer_seed,
                            'demo_seed': demo_seed if arm == 'assisted' else None,
                            'parameter_hash_before': before, 'parameter_hash_after': parameter_hash(parameters),
                            'state_hash_after': state_hash(parameters, optimizer), 'optimizer_step': int(optimizer[2]),
                            'ledger': ledger, 'metrics': metrics, 'sampled_update_verified': sampled}
                        atomic_json(arm_path / f'update-{cohort:04d}.json', record)
                        training.extend(ledger['games'])
                        for stratum in ledger['strata']:
                            sc = stream_counts.setdefault(stratum['stream'], Counter())
                            sc.update({k: stratum['selection'][k] for k in
                                ('collected_rows', 'selected_rows', 'excluded_rows')})
                            sc.update(stratum['rejection_selection'])
                            sc['games'] += stratum['games']
                        counts.update(cohorts=1, optimizer_steps=8, selected_rows=1024,
                                      row_presentations=2048, training_games=len(ledger['games']))
                        if arm == 'assisted':
                            counts['demonstration_presentations'] += 8 * 256
                        atomic_json(run_path / 'progress.json', {'stage': 'training', 'seed': seed,
                            'arm': arm, 'cohorts_for_arm': cohort + 1, 'cohorts_total': counts['cohorts'],
                            'elapsed_seconds': time.perf_counter() - begun})
                    checkpoint(plan['cohorts_per_seed_arm'])
                    final, ps = evaluate(pool, parameters, plan, seed, arm_path / 'final', quarantine)
                    arm_profiles.extend(ps)
                counts['evaluation_games'] += sum(map(len, final.values()))
                probe_started = time.perf_counter()
                final_probes = {label: probe_metrics(parameters, corpus, reference_parameters=initial_parameters)
                    for label, corpus in [('train', dataset), ('probe', probe_dataset)]}
                probe_seconds += time.perf_counter() - probe_started
                atomic_json(arm_path / 'final-probes.json', final_probes)
                result = {'seed': seed, 'arm': arm, 'initial_parameter_hash': initial_parameter_hash,
                    'initial_state_hash': initial_state_hash, 'final_parameter_hash': parameter_hash(parameters),
                    'final_state_hash': state_hash(parameters, optimizer), 'parameter_count': PARAMETER_COUNT,
                    'optimizer_steps': int(optimizer[2]), 'stream_cursors': cursors,
                    'stream_counts': {k: dict(v) for k, v in stream_counts.items()},
                    'demonstration_probes': final_probes,
                    'training': value_summary(training), 'training_unavailable_choices': sum(g['unavailable_choices'] for g in training),
                    'final': {o: value_summary(gs) for o, gs in final.items()},
                    'final_unavailable_choices': {o: sum(g['unavailable_choices'] for g in gs) for o, gs in final.items()},
                    'collection_seconds': sum(p['collection_seconds'] for p in arm_profiles),
                    'update_seconds_including_compile': arm_update_seconds,
                    'sampled_replay_update_seconds': arm_recovery_seconds,
                    'final_checkpoint': str(arm_path / 'checkpoints' / str(plan['cohorts_per_seed_arm'])),
                    'initial_panels': str(base / 'initial'), 'final_panels': str(arm_path / 'final')}
                atomic_json(arm_path / 'result.json', result)
                profiles.extend(arm_profiles)
                update_seconds += arm_update_seconds
                replay_update_seconds += arm_recovery_seconds
                results[arm] = result
            seed_result = {'seed': seed, 'identity': identity, 'initial_parameter_hash': initial_parameter_hash,
                'initial_demonstration_probes': initial_probes,
                'initial': {o: value_summary(gs) for o, gs in initial.items()},
                'initial_unavailable_choices': {o: sum(g['unavailable_choices'] for g in gs) for o, gs in initial.items()},
                'arms': results}
            atomic_json(base / 'result.json', seed_result)
            seeds[str(seed)] = seed_result
        local_keys = ('cohorts', 'optimizer_steps', 'selected_rows', 'row_presentations',
            'demonstration_presentations', 'evaluation_games', 'checkpoint_comparisons', 'original_next_update_recoveries')
        if any(counts[k] != plan['expected_counts'][k] for k in local_keys):
            raise ValueError('Fixed count mismatch: ' + str(dict(counts)))
        verify_inputs(run_path / 'inputs')
        if digest(run_path / 'inputs/manifest.json') != input_hash:
            raise ValueError('Input manifest changed during training')
        report = {'status': 'passed', 'input_hash': input_hash, 'plan': plan, 'counts': dict(counts),
            'seeds': seeds, 'elapsed_seconds': time.perf_counter() - begun,
            'collection_seconds': sum(p['collection_seconds'] for p in profiles),
            'update_seconds_including_compile': update_seconds,
            'sampled_replay_update_seconds': replay_update_seconds,
            'demonstration_probe_seconds': probe_seconds,
            'demonstration_manifest_sha256': dataset.manifest_sha256,
            'peak_parent_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            'profiles': profiles, 'public_server_used': False, 'final_reserved_data_used': False,
            'scope': 'Matched PPO budget development comparison; baseline-v1 historically known and excluded from F35 training; external teacher evaluation reported separately.'}
        atomic_json(run_path / 'report.json', report)
        return report
    except BaseException as exc:
        atomic_json(run_path / 'failure.json', {'status': 'failed', 'error': repr(exc), 'counts': dict(counts),
            'completed_seeds': sorted(seeds), 'elapsed_seconds': time.perf_counter() - begun})
        raise


def main():
    import argparse
    from .data import Dataset
    from .demonstrations import schedule
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--demonstrations', type=Path, required=True)
    args = parser.parse_args()
    admitted = verify_inputs(args.admission)
    plan = admitted['plan']
    _, _, catalog, _, _ = load_initial(plan['seeds'][0])
    input_hash = digest(args.admission / 'manifest.json')
    dataset_hash = digest(args.demonstrations / 'report.json')
    datasets = [Dataset.load(args.demonstrations, expected_input_hash=input_hash,
        expected_schedule=schedule(plan), catalog=catalog, split=split,
        quarantine=plan['quarantined_team_keys'], expected_manifest_sha256=dataset_hash)
        for split in ('train', 'development')]
    report = run(args.run, args.admission, *datasets)
    print({'status': report['status'], 'counts': report['counts'],
        'seconds': report['elapsed_seconds']}, flush=True)


if __name__ == '__main__':
    main()
