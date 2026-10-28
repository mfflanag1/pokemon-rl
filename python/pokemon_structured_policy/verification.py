"""Prespecified sampled F34 replay, separate from production and external search.

Fresh-policy replay covers 18 training and 36 local evaluation games. Twelve
external games replay their recorded actions through Showdown; unseeded Rust search
is not rerun. Six saved batches independently reproduce their original next update.
This is bounded sampled verification, not exhaustive campaign replay.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pokemon_external_benchmark.transport import JsonWorker
from pokemon_fit.data import fingerprint
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, canonical, digest
from .checkpoint import load, load_f32
from .collector import Pool
from .common import (SEEDS, before_deadline, evaluation_specs, training_specs,
                     state_hash, verify_inputs, read)
from .learning import optimize_batch
from .network import extend

STREAMS = {'self_play': None, 'frozen_f32': 'frozen_f32', 'tactical': 'tactical'}
OPPONENTS = ('random', 'base_power', 'tactical')
COHORTS = (0, 511)
EXPECTED = {'training_games': 18, 'evaluation_games': 36, 'external_action_replay_games': 12,
            'game_executions': 66, 'sampled_updates': 6, 'optimizer_steps_recomputed': 48}


def same(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError('Sampled replay mismatch: ' + label)


def check_game(rows, actual, expected, split):
    for key in ('episode', 'split', 'swap', 'learner_side', 'opponent', 'outcome',
                'provenance', 'decisions', 'phases', 'unavailable_choices', 'trace',
                'actor_trace', 'value_mean', 'terminal_value_mse'):
        same(actual[key], expected[key], 'game/' + key)
    expected_rows = len(expected['actor_trace']) if split == 'train' else 0
    if len(rows) != expected_rows:
        raise ValueError('Sampled replay learner-row count')
    return expected_rows


def external_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replay_external(record, transcript, worker):
    """Replay admitted recorded actions; compare every available projected boundary."""
    if record.get('status') != 'completed' or record.get('learner_uses_history') is not True:
        raise ValueError('External source game completion/history mode')
    steps = record['steps']
    if len(transcript) != len(steps) + 1:
        raise ValueError('External transcript boundary count')
    worker.request('hello')
    bundle = worker.request('start', run_seed=record['run_seed'], episode_index=record['episode_index'],
        swap=record['swap'], external_side=record['external_side'], tactical_side=None)
    same(bundle['frame']['provenance'], record['provenance'], 'external/provenance')
    rejections = {'p1': 0, 'p2': 0}
    learner_side = 'p2' if record['external_side'] == 'p1' else 'p1'
    for index, step in enumerate(steps):
        same(bundle, transcript[index], 'external/boundary/' + str(index))
        frame = bundle['frame']
        if frame['outcome'] is not None or step['index'] != index:
            raise ValueError('External step after outcome or wrong index')
        if step['turn'] != frame['players'][learner_side]['turn']:
            raise ValueError('External recorded turn')
        if step['external_input_sha256'] != external_digest(bundle['external']):
            raise ValueError('External input digest')
        pending = {s for s in ('p1', 'p2') if any(frame['players'][s]['mask'])}
        if set(step['actions']) != pending or set(step['request_ids']) != pending:
            raise ValueError('External recorded pending sides')
        for side in pending:
            player, action = frame['players'][side], step['actions'][side]
            if (type(action) is not int or not 0 <= action < 14 or not player['mask'][action]
                    or step['request_ids'][side] != player['request_id']):
                raise ValueError('External recorded action/request')
        bundle = worker.request('step', actions=step['actions'], request_ids=step['request_ids'])
        same(bundle['frame'].get('submitted', []), step['submitted'], 'external/submitted')
        for submitted in step['submitted']:
            rejections[submitted['side']] += not submitted['accepted']
    same(bundle, transcript[-1], 'external/final recorded boundary')
    kind = record['outcome']['kind']
    if kind == 'administrative_cutoff':
        frame = bundle['frame']
        if frame['outcome'] is not None or not all(frame['players'][s]['phase'] == 'move'
                and frame['players'][s]['turn'] >= record['max_turns']
                and any(frame['players'][s]['mask']) for s in ('p1', 'p2')):
            raise ValueError('External cutoff is not the recorded live joint boundary')
        bundle = worker.request('cutoff')
    elif kind != 'terminal':
        raise ValueError('External unsupported outcome')
    same(bundle['frame']['outcome'], record['outcome'], 'external/outcome')
    same(rejections, record['rejected_actions'], 'external/rejection accounting')
    same(record['illegal_actions'], {'p1': 0, 'p2': 0}, 'external/illegal actions')
    return {'steps': len(steps), 'projected_boundaries': len(transcript),
            'decisions': sum(len(s['actions']) for s in steps), 'outcome': record['outcome']}


def run(training_run, external_initial, external_final, output):
    training_run, external_initial, external_final, output = map(
        lambda p: Path(p).resolve(), (training_run, external_initial, external_final, output))
    if any(output == p or p in output.parents for p in (training_run, external_initial, external_final)):
        raise ValueError('Verification output must be outside original evidence directories')
    output.mkdir(parents=True, exist_ok=False)
    begun, counts, records, inventory = time.perf_counter(), Counter(), [], {}

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

    def checkpoint(seed, cohort, input_hash):
        folder = training_run / str(seed) / 'checkpoints' / str(cohort)
        pointer = bound_json(folder / 'latest.json')
        if Path(pointer['file']).name != pointer['file']:
            raise ValueError('Checkpoint basename')
        bind(folder / pointer['file'])
        p, opt, metadata = load(folder, {'input_hash': input_hash, 'seed': seed, 'cohorts': cohort})
        if int(opt[2]) != 8 * cohort:
            raise ValueError('Verification checkpoint optimizer step')
        return p, opt, metadata

    def record(item):
        records.append(item)
        atomic_json(output / 'progress.json', {'status': 'running', 'counts': dict(counts),
            'last': item, 'elapsed_seconds': time.perf_counter() - begun})

    try:
        before_deadline()
        report = bound_json(training_run / 'report.json')
        manifest_path = bind(training_run / 'inputs/manifest.json')
        admission = verify_inputs(training_run / 'inputs')
        plan, input_hash = admission['plan'], digest(manifest_path)
        if report['status'] != 'passed' or report['input_hash'] != input_hash:
            raise ValueError('Completed training input binding')
        same(report['plan'], plan, 'training plan')
        if (plan['seeds'] != list(SEEDS) or plan['cohorts_per_seed'] != 512
                or plan['sampled_update_cohorts'] != list(COHORTS)
                or set(plan['evaluation_pairs']) != set(OPPONENTS)
                or set(report['seeds']) != {str(s) for s in SEEDS}):
            raise ValueError('Prespecified verification coverage')
        external_reports = {endpoint: bound_json(folder / 'report.json') for endpoint, folder in
                            [('initial', external_initial), ('final', external_final)]}
        initial_external, final_external = external_reports['initial'], external_reports['final']
        same(initial_external['plan'], final_external['plan'], 'external endpoint plans')
        same(initial_external['plan_sha256'], final_external['plan_sha256'], 'external plan hashes')
        external_plan = initial_external['plan']
        matching = [rel for rel, sha in admission['source_files'].items()
                    if sha == initial_external['plan_sha256']]
        if not matching:
            raise ValueError('External plan is absent from frozen training inputs')
        for rel in matching:
            same(bound_json(training_run / 'inputs/source' / rel), external_plan, 'frozen external plan')
        if external_plan['seeds'] != list(SEEDS) or external_plan['pairs_per_seed'] < 1:
            raise ValueError('External sample coverage')
        for endpoint, external_report in external_reports.items():
            if (external_report['status'] != 'passed' or external_report['endpoint'] != endpoint
                    or external_report.get('input_hash') != input_hash
                    or set(external_report['seeds']) != {str(s) for s in SEEDS}):
                raise ValueError('Completed external endpoint inventory')

        for seed in SEEDS:
            before_deadline()
            host, catalog, identity = load_f32(seed)
            seed_report = bound_json(training_run / str(seed) / 'result.json')
            same(seed_report, report['seeds'][str(seed)], 'seed report')
            same(seed_report['identity'], identity, 'F32 source identity')
            initial, initial_opt, _ = checkpoint(seed, 0, input_hash)
            warm, warm_opt = extend(host, derived_seed('f34-structured-initialization-v1', seed))
            if state_hash(initial, initial_opt) != state_hash(warm, warm_opt):
                raise ValueError('Function-preserving initial model/reset optimizer binding')
            if parameter_hash(initial) != seed_report['initial_parameter_hash']:
                raise ValueError('Initial report parameter hash')
            final, final_opt, _ = checkpoint(seed, 512, input_hash)
            if parameter_hash(final) != seed_report['final_parameter_hash']:
                raise ValueError('Final report parameter hash')
            before_deadline()
            with Pool(output / str(seed) / 'local-workers', plan['workers'], catalog, host) as pool:
                for cohort in COHORTS:
                    before_deadline()
                    p, opt, metadata = checkpoint(seed, cohort, input_hash)
                    update = bound_json(training_run / str(seed) / f'update-{cohort:04d}.json')
                    if (update['cohort'] != cohort or update['optimizer_step'] != (cohort + 1) * 8
                            or parameter_hash(p) != update['parameter_hash_before']
                            or update['optimizer_seed'] != derived_seed('f34-optimizer-v1', seed, cohort)):
                        raise ValueError('Sampled update source linkage')
                    samples, strata = update['ledger']['samples'], update['ledger']['strata']
                    if len(samples) != 3 or {s['stream'] for s in samples} != set(STREAMS):
                        raise ValueError('Training sample stream inventory')
                    if len(strata) != 3 or {s['stream'] for s in strata} != set(STREAMS):
                        raise ValueError('Training stratum inventory')
                    strata = {s['stream']: s for s in strata}
                    for sample in samples:
                        label, expected_game = sample['stream'], sample['game']
                        first = strata[label]['first_episode']
                        if metadata['stream_cursors'][label] != first:
                            raise ValueError('Sampled checkpoint stream cursor')
                        expected_spec = training_specs(plan, seed, first, STREAMS[label])[0]
                        same(sample['spec'], expected_spec, 'training sample schedule')
                        before_deadline()
                        replay, _ = pool.collect(p, [expected_spec])
                        if len(replay) != 1:
                            raise ValueError('Training replay game count')
                        rows, game = replay[0]
                        row_count = check_game(rows, game, expected_game, 'train')
                        counts.update(training_games=1, game_executions=1)
                        record({'kind': 'fresh_training', 'seed': seed, 'cohort': cohort,
                                'stream': label, 'learner_rows': row_count})
                    batch_path = bind(training_run / str(seed) / f'batch-{cohort:04d}.npz')
                    with np.load(batch_path, allow_pickle=False) as saved:
                        batch = {key: saved[key] for key in saved.files}
                    bind(batch_path)
                    if fingerprint(batch) != update['ledger']['batch_sha256']:
                        raise ValueError('Sampled selected-batch fingerprint')
                    before_deadline()
                    q, qo, metrics = optimize_batch(p, opt, batch, update['optimizer_seed'])
                    if (state_hash(q, qo) != update['state_hash_after']
                            or parameter_hash(q) != update['parameter_hash_after']
                            or int(qo[2]) != update['optimizer_step']):
                        raise ValueError('Independent sampled next-update state')
                    same(metrics, update['metrics'], 'independent next-update metrics')
                    counts.update(sampled_updates=1, optimizer_steps_recomputed=8)
                    record({'kind': 'fresh_update', 'seed': seed, 'cohort': cohort,
                            'selected_rows': len(batch['actions']), 'batch_sha256': fingerprint(batch)})
                for endpoint, p in [('initial', initial), ('final', final)]:
                    for opponent in OPPONENTS:
                        saved = bound_json(training_run / str(seed) / endpoint / f'{opponent}-000.json')
                        specs = evaluation_specs(plan, seed, opponent, 0, 1)
                        if len(saved['specs']) != 8 or len(saved['games']) != 8:
                            raise ValueError('Evaluation source chunk inventory')
                        same(saved['specs'][:2], specs, 'evaluation first-pair schedule')
                        before_deadline()
                        replay, _ = pool.collect(p, specs)
                        if len(replay) != 2:
                            raise ValueError('Evaluation replay pair count')
                        for (rows, game), expected_game in zip(replay, saved['games'][:2]):
                            check_game(rows, game, expected_game, 'development')
                        counts.update(evaluation_games=2, game_executions=2)
                        record({'kind': 'fresh_evaluation_pair', 'seed': seed,
                                'endpoint': endpoint, 'opponent': opponent, 'games': 2})

            for endpoint, folder in [('initial', external_initial), ('final', external_final)]:
                external_seed = bound_json(folder / str(seed) / 'result.json')
                same(external_seed, external_reports[endpoint]['seeds'][str(seed)], 'external seed report')
                wanted_hash = parameter_hash(host if endpoint == 'initial' else final)
                if external_seed['parameter_hash'] != wanted_hash:
                    raise ValueError('External endpoint policy identity')
                games = external_seed['games']
                if len(games) != 2 * external_plan['pairs_per_seed']:
                    raise ValueError('External full panel game count')
                for swap, game in zip((False, True), games[:2]):
                    if game['episode'] != 0 or game['swap'] is not swap:
                        raise ValueError('External first-pair schedule')
                    source = folder / str(seed) / f'000-{int(swap)}'
                    if (ROOT / game['game_record']).resolve() != (source / 'result.json').resolve():
                        raise ValueError('External original evidence path')
                    if digest(source / 'result.json') != game['game_record_sha256']:
                        raise ValueError('External original evidence digest')
                    original = bound_json(source / 'result.json')
                    coordinates = {'run_seed': derived_seed('f34-evaluation-teams-v1', seed, 'foul_play'),
                        'episode_index': 0, 'swap': swap, 'external_side': 'p1' if swap else 'p2',
                        'search_time_ms': external_plan['search_time_ms'], 'max_turns': external_plan['max_turns'],
                        'fp_seed': derived_seed('f34-external-sampling-v1', seed, 0, swap)}
                    same({k: original[k] for k in coordinates}, coordinates, 'external coordinates')
                    for key in ('outcome', 'provenance'):
                        same(original[key], game[key], 'external summary/' + key)
                    learner = bound_json(source / 'learner.json')
                    if learner['parameter_hash'] != wanted_hash:
                        raise ValueError('External recorded learner parameter hash')
                    side = 'p2' if swap else 'p1'
                    recorded_actions = [(s['request_ids'][side], s['actions'][side])
                                        for s in original['steps'] if side in s['actions']]
                    same([[a['request_id'], a['action']] for a in learner['decisions']],
                         recorded_actions, 'external learner decision ledger')
                    transcript_path = bind(source / 'transcript.jsonl')
                    transcript = [json.loads(line) for line in transcript_path.read_text().splitlines()]
                    bind(transcript_path)
                    before_deadline()
                    destination = output / str(seed) / 'external' / endpoint / str(int(swap))
                    with JsonWorker(['node', 'packages/external-benchmark/worker.mjs'],
                                    destination / 'showdown.stderr') as worker:
                        replay = replay_external(original, transcript, worker)
                    if replay['decisions'] != game['decisions']:
                        raise ValueError('External replay decision accounting')
                    counts.update(external_action_replay_games=1, game_executions=1)
                    record({'kind': 'recorded_external_actions', 'seed': seed, 'endpoint': endpoint,
                            'swap': swap, **replay, 'foul_play_search_reexecuted': False,
                            'learner_policy_recomputed': False})

        if dict(counts) != EXPECTED:
            raise ValueError('Prespecified verification count mismatch: ' + str(dict(counts)))
        verify_inputs(training_run / 'inputs')
        for path, expected in inventory.items():
            if digest(path) != expected:
                raise ValueError('Verification input changed: ' + path)
        result = {'status': 'passed', 'input_hash': input_hash, 'counts': dict(counts),
            'expected_counts': EXPECTED, 'records': records, 'input_files': inventory,
            'elapsed_seconds': time.perf_counter() - begun,
            'coverage': {'training': 'First game of each of three streams at cohorts 0 and 511, all three seeds.',
                'evaluation': 'First seat-swapped pair per local opponent at initial and final, all three seeds.',
                'external': 'First pair at both endpoints, all three seeds; Showdown recorded-action replay only.',
                'updates': 'Saved selected batches at cohorts 0 and 511 independently recomputed from before-checkpoints.',
                'exhaustive': False, 'external_search_reexecuted': False}}
        atomic_json(output / 'report.json', result)
        return result
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error': repr(exc),
            'counts': dict(counts), 'records': records, 'input_files': inventory,
            'elapsed_seconds': time.perf_counter() - begun})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--external-initial', type=Path, required=True)
    parser.add_argument('--external-final', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.training_run, args.external_initial, args.external_final, args.output)
    print({'status': result['status'], 'counts': result['counts']}, flush=True)
