"""Prespecified F39 local replay and original-next-update verification.

No stochastic teacher/search or F35 demonstration module is admitted. The 66 local games are
fresh executions using recorded original parameters and original schedules.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

from pokemon_fit.data import fingerprint
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed, prepare_batch
from pokemon_full_train.storage import atomic_json, canonical, digest
from pokemon_structured_policy.checkpoint import load
from pokemon_structured_policy.learning import optimize_batch
from pokemon_teacher_assisted.collector import Pool
from pokemon_throughput.collector import merge_rows
from .common import (SEEDS, deadline, evaluation_specs, training_specs, state_hash,
                     verify_inputs, read, load_initial)

STREAMS = {'self_play': None, 'frozen_f32': 'frozen_f32', 'tactical': 'tactical'}
OPPONENTS = ('baseline-v1', 'tactical', 'base_power', 'random')
ENDPOINTS = ('initial', 'final')
FORBIDDEN_MODULE_PREFIXES = (
    'pokemon_external_benchmark', 'pokemon_structured_policy.external',
    'pokemon_teacher_assisted.external', 'pokemon_teacher_assisted.external_evaluation',
    'pokemon_teacher_assisted.teacher', 'pokemon_teacher_assisted.demonstrations',
    'pokemon_teacher_assisted.data', 'pokemon_teacher_assisted.learning',
    'pokemon_teacher_assisted.verification', 'pokemon_teacher_runtime',
    'pokemon_teacher_legality', 'fp', 'poke_engine', 'foul_play')


def teacher_search_modules():
    """Snapshot known stochastic teacher/search and F35 demonstration namespaces."""
    return sorted(name for name in sys.modules if any(name == prefix or name.startswith(prefix + '.')
        for prefix in FORBIDDEN_MODULE_PREFIXES))

COHORTS = (512, 767)
CHECKPOINTS = (512, 767, 768)
EXPECTED = {'training_games': 18, 'evaluation_games': 48, 'game_executions': 66,
            'sampled_updates': 6, 'optimizer_steps_recomputed': 48,
            'checkpoint_comparisons': 9}


def same(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError('F39 sampled replay mismatch: ' + label)


def check_game(rows, actual, expected, split, selection=None, rows_sha256=None):
    """Compare actual original actions/values and complete-game GAE row bytes."""
    for key in ('episode', 'split', 'swap', 'learner_side', 'opponent', 'outcome',
                'provenance', 'decisions', 'phases', 'unavailable_choices', 'trace',
                'actor_trace', 'value_mean', 'terminal_value_mse', 'accepted_decisions',
                'accepted_controlled_decisions', 'eligible_learner_rows'):
        same(actual[key], expected[key], 'game/' + key)
    observed, recorded = deepcopy(actual['rejected_attempts']), deepcopy(expected['rejected_attempts'])
    if split == 'train' and (selection is None or rows_sha256 is None):
        raise ValueError('Training replay requires original row selection and fingerprint')
    for attempt in recorded:
        row = attempt['learner_row_index']
        wanted = (row in selection['selected_indices']) if row is not None and selection is not None else False
        if split == 'train' and attempt['selected_for_update'] is not wanted:
            raise ValueError('Sampled rejected-row selection attribution')
    for attempt in [*observed, *recorded]:
        attempt['selected_for_update'] = None
    same(observed, recorded, 'game/rejected_attempts')
    expected_rows = len(expected['actor_trace']) if split == 'train' else 0
    if len(rows) != expected_rows:
        raise ValueError('Sampled learner-row count')
    if split == 'train' and fingerprint(prepare_batch(merge_rows([(rows, actual)]))) != rows_sha256:
        raise ValueError('Sampled original complete-game rows differ')
    return expected_rows


def check_initial(parameters, optimizer, metadata, original, original_optimizer, identity):
    """A reset Adam or changed stream cursor must fail even with equal weights."""
    if state_hash(parameters, optimizer) != state_hash(original, original_optimizer):
        raise ValueError('F34 initial parameters and Adam moments were not preserved')
    if int(optimizer[2]) != 4096 or metadata['cohorts'] != 512:
        raise ValueError('F34 initial optimizer/cohort counters were not preserved')
    source = identity['f34_metadata']
    for key in ('stream_cursors', 'next_episode', 'selected_rows'):
        same(metadata[key], source[key], 'inherited F34 ' + key)


def run(training_run, output):
    training_run, output = map(lambda p: Path(p).resolve(), (training_run, output))
    if output == training_run or training_run in output.parents or output in training_run.parents:
        raise ValueError('Verification output must be outside original evidence directories')
    output.mkdir(parents=True, exist_ok=False)
    begun, counts, records, inventory = time.perf_counter(), Counter(), [], {}
    module_inventory = {'before': None, 'after': None,
                        'forbidden_prefixes': list(FORBIDDEN_MODULE_PREFIXES)}

    def bind(path):
        path = Path(path).resolve()
        observed = digest(path)
        if str(path) in inventory and inventory[str(path)] != observed:
            raise ValueError('Verification input changed: ' + str(path))
        inventory[str(path)] = observed
        return path

    def bound_json(path):
        path = bind(path)
        value = read(path)
        bind(path)
        return value

    def record(item):
        records.append(item)
        atomic_json(output / 'progress.json', {'status': 'running', 'counts': dict(counts),
            'last': item, 'elapsed_seconds': time.perf_counter() - begun})

    def checkpoint(seed, cohort, input_hash):
        folder = training_run / str(seed) / 'checkpoints' / str(cohort)
        pointer = bound_json(folder / 'latest.json')
        if Path(pointer['file']).name != pointer['file']:
            raise ValueError('Checkpoint basename')
        bind(folder / pointer['file'])
        p, opt, meta = load(folder, {'input_hash': input_hash, 'seed': seed, 'cohorts': cohort})
        if int(opt[2]) != 8 * cohort or meta['selected_rows'] != 1024 * cohort:
            raise ValueError('Verification checkpoint optimizer/row counters')
        counts['checkpoint_comparisons'] += 1
        record({'kind': 'checkpoint', 'seed': seed, 'cohort': cohort,
                'state_hash': state_hash(p, opt)})
        return p, opt, meta

    try:
        manifest = bound_json(training_run / 'inputs/manifest.json')
        plan = manifest['plan']
        deadline(plan)
        module_inventory['before'] = teacher_search_modules()
        if module_inventory['before']:
            raise ValueError('Stochastic teacher/search or demonstration module present before verification')
        admitted = verify_inputs(training_run / 'inputs')
        same(admitted['plan'], plan, 'admitted plan')
        input_hash = digest(training_run / 'inputs/manifest.json')
        report = bound_json(training_run / 'report.json')
        if report['status'] != 'passed' or report['input_hash'] != input_hash:
            raise ValueError('Completed training input binding')
        same(report['plan'], plan, 'training plan')
        if (set(report['seeds']) != {str(s) for s in SEEDS}
                or plan['sampled_update_cohorts'] != list(COHORTS)
                or set(plan['evaluation_pairs']) != set(OPPONENTS)):
            raise ValueError('Prespecified verification coverage')
        for seed in SEEDS:
            deadline(plan)
            original, original_opt, catalog, frozen, identity = load_initial(seed)
            row = bound_json(training_run / str(seed) / 'result.json')
            same(row, report['seeds'][str(seed)], 'seed report')
            same(row['identity'], identity, 'F34 predecessor identity')
            states = {c: checkpoint(seed, c, input_hash) for c in CHECKPOINTS}
            check_initial(*states[512], original, original_opt, identity)
            for endpoint, cohort in [('initial', 512), ('final', 768)]:
                p, opt, _ = states[cohort]
                same(parameter_hash(p), row[endpoint + '_parameter_hash'], endpoint + ' parameter hash')
                same(state_hash(p, opt), row[endpoint + '_state_hash'], endpoint + ' state hash')
            with Pool(output / str(seed) / 'local-workers', plan['workers'], catalog, frozen) as pool:
                for cohort in COHORTS:
                    deadline(plan)
                    p, opt, meta = states[cohort]
                    update = bound_json(training_run / str(seed) / f'update-{cohort:04d}.json')
                    optimizer_seed = derived_seed('f39-optimizer-v1', seed, cohort)
                    if (update['cohort'] != cohort or update['optimizer_step'] != (cohort + 1) * 8
                            or update['parameter_hash_before'] != parameter_hash(p)
                            or update['state_hash_before'] != state_hash(p, opt)
                            or update['optimizer_seed'] != optimizer_seed):
                        raise ValueError('Sampled update source linkage')
                    samples, strata = update['ledger']['samples'], update['ledger']['strata']
                    if (len(samples) != 3 or {s['stream'] for s in samples} != set(STREAMS)
                            or len(strata) != 3 or {s['stream'] for s in strata} != set(STREAMS)):
                        raise ValueError('Training sample/stratum inventory')
                    strata = {s['stream']: s for s in strata}
                    for sample in samples:
                        label = sample['stream']
                        first = strata[label]['first_episode']
                        same(meta['stream_cursors'][label], first, 'sampled stream cursor')
                        spec = training_specs(plan, seed, first, STREAMS[label])[0]
                        same(sample['spec'], spec, 'training first-game schedule')
                        same({key: sample['game'][key] for key in
                              ('episode', 'split', 'swap', 'learner_side', 'opponent')},
                             {key: spec[key] for key in
                              ('episode', 'split', 'swap', 'learner_side', 'opponent')},
                             'training sample is first complete stream game (zero row offset)')
                        deadline(plan)
                        replay, _ = pool.collect(p, [spec])
                        if len(replay) != 1:
                            raise ValueError('Training replay game count')
                        rows, actual = replay[0]
                        n = check_game(rows, actual, sample['game'], 'train',
                                       strata[label]['selection'], sample['rows_sha256'])
                        counts.update(training_games=1, game_executions=1)
                        record({'kind': 'fresh_training', 'seed': seed, 'cohort': cohort,
                            'stream': label, 'learner_rows': n, 'rows_sha256': sample['rows_sha256'],
                            'rejected_attempts': actual['unavailable_choices']})
                    path = bind(training_run / str(seed) / f'batch-{cohort:04d}.npz')
                    with np.load(path, allow_pickle=False) as saved:
                        batch = {k: saved[k] for k in saved.files}
                    bind(path)
                    same(fingerprint(batch), update['ledger']['batch_sha256'], 'original selected batch')
                    if len(batch['actions']) != 1024:
                        raise ValueError('Original selected batch row count')
                    offset = 0
                    for label in STREAMS:
                        selection = strata[label]['selection']
                        count = plan['stream_selected_rows'][label]
                        same(selection['selected_rows'], count, 'original stream selected rows')
                        selected = {k: a[offset:offset + count] for k, a in batch.items()}
                        same(fingerprint(selected), selection['selected_batch_sha256'], 'original stream batch ' + label)
                        offset += count
                    deadline(plan)
                    q, qo, metrics = optimize_batch(p, opt, batch, optimizer_seed)
                    if (state_hash(q, qo) != update['state_hash_after']
                            or parameter_hash(q) != update['parameter_hash_after']
                            or int(qo[2]) != update['optimizer_step']):
                        raise ValueError('Independent original-next-update state')
                    same(metrics, update['metrics'], 'original-next-update metrics')
                    counts.update(sampled_updates=1, optimizer_steps_recomputed=8)
                    record({'kind': 'fresh_update', 'seed': seed, 'cohort': cohort,
                            'selected_rows': len(batch['actions']), 'batch_sha256': fingerprint(batch)})
                for endpoint, cohort in [('initial', 512), ('final', 768)]:
                    for opponent in OPPONENTS:
                        saved = bound_json(training_run / str(seed) / endpoint / f'{opponent}-000.json')
                        specs = evaluation_specs(plan, seed, opponent, 0, 1)
                        if len(saved['specs']) != 8 or len(saved['games']) != 8:
                            raise ValueError('Evaluation source chunk inventory')
                        same(saved['specs'][:2], specs, 'evaluation first-pair schedule')
                        deadline(plan)
                        replay, _ = pool.collect(states[cohort][0], specs)
                        if len(replay) != 2:
                            raise ValueError('Evaluation replay pair count')
                        for (rows, actual), expected_game in zip(replay, saved['games'][:2]):
                            check_game(rows, actual, expected_game, 'development')
                        counts.update(evaluation_games=2, game_executions=2)
                        record({'kind': 'fresh_evaluation_pair', 'seed': seed,
                                'endpoint': endpoint, 'opponent': opponent})
        same(dict(counts), EXPECTED, 'fixed verification counts')
        deadline(plan)
        verify_inputs(training_run / 'inputs')
        for path, expected in inventory.items():
            same(digest(path), expected, 'input unchanged ' + path)
        module_inventory['after'] = teacher_search_modules()
        if module_inventory['after']:
            raise ValueError('Stochastic teacher/search or demonstration module present after verification')
        result = {'status': 'passed', 'input_hash': input_hash, 'counts': dict(counts),
            'expected_counts': EXPECTED, 'records': records, 'input_files': inventory,
            'elapsed_seconds': time.perf_counter() - begun,
            'coverage': {'exhaustive': False, 'teacher_search_module_inventory': module_inventory,
                'teacher_search_execution_scope': 'Before/after Python module snapshots cover the listed stochastic teacher/search and F35 demonstration namespaces. Source inspection binds this verifier to the local collector; snapshots do not audit arbitrary subprocess execution or transient imports.',
                'training': 'First complete game per stream at absolute cohorts 512 and 767 for every seed.',
                'evaluation': 'First paired seats per opponent at initial and final endpoints for every seed.',
                'updates': 'Six original saved selected batches and exact original next states/metrics.',
                'checkpoints': 'Nine sampled states; initial F34 parameters, Adam and cursors preserved.',
                'rows': 'Complete sampled game GAE row fingerprints and original actions/log probabilities/values.'}}
        atomic_json(output / 'report.json', result)
        return result
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error': repr(exc),
            'counts': dict(counts), 'records': records, 'input_files': inventory,
            'teacher_search_module_inventory': module_inventory, 'elapsed_seconds': time.perf_counter() - begun})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('training-run', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.training_run, args.output)
    print({'status': result['status'], 'counts': result['counts']}, flush=True)
