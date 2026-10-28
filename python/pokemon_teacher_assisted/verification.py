"""Bounded, prespecified F35 replay; teacher search is never rerun."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from pokemon_external_benchmark.transport import JsonWorker
from pokemon_fit.data import fingerprint
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, canonical, digest
from pokemon_structured_policy.actor import batched_actor
from pokemon_structured_policy.checkpoint import load
from .campaign import optimize
from .collector import Pool
from .common import (ARMS, SEEDS, DEADLINE, before_deadline, evaluation_specs, training_specs,
                     state_hash, verify_inputs, read, load_initial)
from .data import Dataset
from .demonstrations import schedule
from .external import json_hash, label_candidate, accept_label

STREAMS = {'self_play': None, 'frozen_f32': 'frozen_f32', 'tactical': 'tactical'}
OPPONENTS = ('baseline-v1', 'tactical', 'base_power', 'random')
COHORTS = (0, 255)
EXPECTED = {'training_games': 36, 'evaluation_games': 48, 'external_action_replay_games': 18,
    'teacher_action_replay_games': 12, 'game_executions': 114, 'sampled_updates': 12,
    'optimizer_steps_recomputed': 96, 'checkpoint_comparisons': 18}
QUALIFICATIONS = ('artifacts/reviews/f35-teacher-assisted/qualification-teacher-v1.json',
                  'artifacts/reviews/f35-teacher-assisted/qualification-worker-campaign-v3.json')


def same(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError('F35 sampled replay mismatch: ' + label)


def check_fixture_qualifications(teacher, learner):
    if teacher.get('checks', {}).get('real_unavailable_choice_fixture') is not True:
        raise ValueError('Required explicit unavailable-choice fixture is missing')
    fixture = learner.get('explicit_unavailable_choice_fixture')
    if not isinstance(fixture, dict) or fixture.get('passed') is not True:
        raise ValueError('Required learner rejection/row fixture is missing')
    if fixture.get('evidence_log') not in learner.get('evidence_files', {}):
        raise ValueError('Learner rejection/row fixture log is not bound')


def check_game(rows, actual, expected, split, selection=None):
    for key in ('episode', 'split', 'swap', 'learner_side', 'opponent', 'outcome',
                'provenance', 'decisions', 'phases', 'unavailable_choices', 'trace',
                'actor_trace', 'value_mean', 'terminal_value_mse', 'accepted_decisions',
                'accepted_controlled_decisions', 'eligible_learner_rows'):
        same(actual[key], expected[key], 'game/' + key)
    observed, recorded = deepcopy(actual['rejected_attempts']), deepcopy(expected['rejected_attempts'])
    if split == 'train' and selection is None:
        raise ValueError('Training replay requires original row selection')
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
    return expected_rows


def replay_external(record, transcript, worker, *, learner_record=None, parameters=None, catalog=None,
                    label_arrays=None, label_metadata=None):
    """Fresh Showdown replay of all recorded attempts, including unavailable ones."""
    if (record.get('status') != 'completed' or record.get('learner_uses_history') is not True
            or record.get('split') not in ('train', 'development')):
        raise ValueError('External source game completion/history/split')
    steps, outcome = record['steps'], record['outcome']
    extra = 2 if outcome['kind'] == 'administrative_cutoff' else 1
    if len(transcript) != len(steps) + extra:
        raise ValueError('External transcript boundary count')
    if learner_record is not None and (parameters is None or catalog is None):
        raise ValueError('Learner recomputation needs parameters and catalog')
    if (label_arrays is None) != (label_metadata is None):
        raise ValueError('Complete sampled label evidence required')
    def request(op, **payload):
        before_deadline()
        worker.timeout = min(60., DEADLINE - time.time())
        return worker.request(op, **payload)
    request('hello')
    bundle = request('start', run_seed=record['run_seed'], episode_index=record['episode_index'],
        swap=record['swap'], external_side=record['external_side'], split=record['split'], tactical_side=None)
    same(bundle['frame']['provenance'], record['provenance'], 'external/provenance')
    rejections, learner_decisions, accepted_labels, waits = {'p1': 0, 'p2': 0}, [], [], 0
    teacher_side = record['external_side']
    learner_side = 'p2' if teacher_side == 'p1' else 'p1'
    for index, step in enumerate(steps):
        same(bundle, transcript[index], 'external/boundary/' + str(index))
        frame = bundle['frame']
        if frame['outcome'] is not None or step['index'] != index:
            raise ValueError('External step after outcome or wrong index')
        if step['turn'] != frame['players'][learner_side]['turn']:
            raise ValueError('External recorded turn')
        if step['external_input_sha256'] != json_hash(bundle['external']):
            raise ValueError('External input digest')
        pending = {s for s in ('p1', 'p2') if any(frame['players'][s]['mask'])}
        if set(step['actions']) != pending or set(step['request_ids']) != pending:
            raise ValueError('External recorded pending sides')
        for side in pending:
            player, action = frame['players'][side], step['actions'][side]
            if (type(action) is not int or not 0 <= action < 14 or not player['mask'][action]
                    or step['request_ids'][side] != player['request_id']):
                raise ValueError('External recorded action/request')
        if teacher_side in pending:
            if (step['external'].get('action') != step['actions'][teacher_side]
                    or step['external'].get('request_id') != step['request_ids'][teacher_side]):
                raise ValueError('Teacher decision/request mismatch')
        else:
            same(step['external'], {'waiting': True}, 'external/wait')
            if 'teacher_feedback' in step:
                raise ValueError('Waiting teacher has feedback')
            waits += 1
        if learner_record is not None and learner_side in pending:
            player = frame['players'][learner_side]
            out = batched_actor(parameters, [player], [frame['observable_history'][learner_side]],
                [learner_record['action_seed']], [len(learner_decisions)], 2, catalog)[0]
            prediction = {'request_id': player['request_id'], 'action': out[2],
                          'log_probability': out[3], 'value': out[4]}
            if out[2] != step['actions'][learner_side]:
                raise ValueError('Recorded learner action differs from recomputed policy')
            learner_decisions.append(prediction)
        candidate = None
        if label_arrays is not None:
            candidate = label_candidate(frame['players'][teacher_side], frame['observable_history'][teacher_side],
                step['external'], step_index=index, external_input_sha256=json_hash(bundle['external']), catalog=catalog)
        bundle = request('step', actions=step['actions'], request_ids=step['request_ids'])
        same(bundle['frame'].get('submitted', []), step['submitted'], 'external/submitted')
        for submitted in step['submitted']:
            if type(submitted.get('accepted')) is not bool:
                raise ValueError('Missing accepted-submission flag')
            rejections[submitted['side']] += not submitted['accepted']
            if submitted['side'] == teacher_side:
                same(step.get('teacher_feedback'), {'acknowledged': True,
                    'request_id': submitted['request_id'], 'accepted': submitted['accepted']}, 'external/teacher feedback')
        accepted = accept_label(candidate, step['submitted'])
        if accepted is not None:
            accepted_labels.append(accepted)
    same(bundle, transcript[len(steps)], 'external/final live or terminal boundary')
    if outcome['kind'] == 'administrative_cutoff':
        frame = bundle['frame']
        if frame['outcome'] is not None or not all(frame['players'][s]['phase'] == 'move'
                and frame['players'][s]['turn'] >= record['max_turns']
                and any(frame['players'][s]['mask']) for s in ('p1', 'p2')):
            raise ValueError('External cutoff requires its recorded live joint boundary')
        bundle = request('cutoff')
        same(bundle, transcript[-1], 'external/post-cutoff boundary')
    elif outcome['kind'] != 'terminal':
        raise ValueError('External unsupported outcome')
    same(bundle['frame']['outcome'], outcome, 'external/outcome')
    same(rejections, record['rejected_actions'], 'external/rejection accounting')
    same(record['illegal_actions'], {'p1': 0, 'p2': 0}, 'external/illegal actions')
    if learner_record is not None:
        same(learner_decisions, learner_record['decisions'], 'external/learner action log probability value')
    if label_arrays is not None:
        if len(accepted_labels) != record['accepted_labels'] or len(label_metadata['labels']) != len(accepted_labels):
            raise ValueError('Sampled teacher label count')
        for row, accepted in enumerate(accepted_labels):
            same({'row': row, **accepted['metadata']}, label_metadata['labels'][row], 'teacher/label metadata')
            for key, value in [('features', accepted['features']), ('mask', accepted['mask']),
                               ('actions', accepted['action']), ('step_indices', accepted['metadata']['step_index']),
                               ('request_ids', accepted['metadata']['request_id'])]:
                if not np.array_equal(label_arrays[key][row], value):
                    raise ValueError('Sampled teacher label array ' + key)
    return {'steps': len(steps), 'projected_boundaries': len(transcript),
        'decisions': sum(len(s['actions']) for s in steps), 'outcome': outcome,
        'rejected_attempts': sum(rejections.values()), 'teacher_wait_frames': waits,
        'learner_decisions_recomputed': len(learner_decisions), 'accepted_labels_recomputed': len(accepted_labels),
        'value_mean': float(np.mean([d['value'] for d in learner_decisions])) if learner_decisions else None,
        'terminal_value_mse': float(np.mean([(d['value'] - outcome['rewards'][learner_side]) ** 2
            for d in learner_decisions])) if learner_decisions and outcome['kind'] == 'terminal' else None}


def run(training_run, external_run, demonstrations, output):
    training_run, external_run, demonstrations, output = map(lambda p: Path(p).resolve(),
        (training_run, external_run, demonstrations, output))
    if any(output == p or p in output.parents for p in (training_run, external_run, demonstrations)):
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

    def record(item):
        records.append(item)
        atomic_json(output / 'progress.json', {'status': 'running', 'counts': dict(counts),
            'last': item, 'elapsed_seconds': time.perf_counter() - begun})

    def checkpoint(seed, arm, cohort, input_hash):
        folder = training_run / str(seed) / arm / 'checkpoints' / str(cohort)
        pointer = bound_json(folder / 'latest.json')
        if Path(pointer['file']).name != pointer['file']:
            raise ValueError('Checkpoint basename')
        bind(folder / pointer['file'])
        p, opt, meta = load(folder, {'input_hash': input_hash, 'seed': seed, 'arm': arm, 'cohorts': cohort})
        if int(opt[2]) != 8 * cohort or meta['selected_rows'] != 1024 * cohort:
            raise ValueError('Verification checkpoint optimizer/row counter')
        counts['checkpoint_comparisons'] += 1
        return p, opt, meta

    def replay_source(source, destination, p, catalog, *, coordinates, action_seed, labels=False):
        original = bound_json(source / 'result.json')
        same({k: original[k] for k in coordinates}, coordinates, 'recorded-action source coordinates')
        learner = bound_json(source / 'learner.json')
        if learner['parameter_hash'] != parameter_hash(p) or learner['action_seed'] != action_seed:
            raise ValueError('External learner parameter identity')
        transcript_path = bind(source / 'transcript.jsonl')
        transcript = [json.loads(line) for line in transcript_path.read_text().splitlines()]
        bind(transcript_path)
        arrays, metadata = None, None
        if labels:
            metadata = bound_json(source / 'labels.json')
            path = bind(source / 'demonstrations.npz')
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
        before_deadline()
        with JsonWorker(['node', 'packages/teacher-assisted/external-worker.mjs'], destination / 'showdown.stderr') as worker:
            replay = replay_external(original, transcript, worker, learner_record=learner,
                parameters=p, catalog=catalog, label_arrays=arrays, label_metadata=metadata)
        return original, replay

    try:
        before_deadline()
        report = bound_json(training_run / 'report.json')
        manifest = bind(training_run / 'inputs/manifest.json')
        admitted = verify_inputs(training_run / 'inputs')
        plan, input_hash = admitted['plan'], digest(manifest)
        if report['status'] != 'passed' or report['input_hash'] != input_hash:
            raise ValueError('Completed training input binding')
        same(report['plan'], plan, 'training plan')
        if set(report['seeds']) != {str(s) for s in SEEDS} or plan['sampled_update_cohorts'] != list(COHORTS):
            raise ValueError('Prespecified verification coverage')
        external_report = bound_json(external_run / 'report.json')
        if (external_report['status'] != 'passed' or external_report['input_hash'] != input_hash
                or set(external_report['seeds']) != {str(s) for s in SEEDS}):
            raise ValueError('External completion/input inventory')
        same(external_report['plan'], plan, 'external plan')
        teacher_report = bound_json(demonstrations / 'report.json')
        if digest(demonstrations / 'report.json') != report['demonstration_manifest_sha256']:
            raise ValueError('Training demonstration manifest identity')
        for game in teacher_report['games']:
            for relative in game['files']:
                bind(demonstrations / game['directory'] / relative)
        _, _, corpus_catalog, _, _ = load_initial(SEEDS[0])
        dataset = Dataset.load(demonstrations, expected_input_hash=input_hash, expected_schedule=schedule(plan),
            catalog=corpus_catalog, quarantine=plan['quarantined_team_keys'],
            expected_manifest_sha256=report['demonstration_manifest_sha256'])
        qualifications = {}
        for rel in QUALIFICATIONS:
            q = bound_json(ROOT / rel)
            if admitted['source_files'].get(rel) != digest(ROOT / rel):
                raise ValueError('Qualification record absent from source admission')
            if q['status'] != 'passed':
                raise ValueError('Unavailable-choice qualification did not pass')
            for source, expected in q['source_files'].items():
                if admitted['source_files'].get(source) != expected or digest(ROOT / source) != expected:
                    raise ValueError('Qualification source admission changed: ' + source)
            for evidence, expected in q.get('evidence_files', {}).items():
                if admitted['source_files'].get(evidence) != expected or digest(bind(ROOT / evidence)) != expected:
                    raise ValueError('Qualification evidence admission changed: ' + evidence)
            qualifications[rel] = digest(ROOT / rel)
        check_fixture_qualifications(*(bound_json(ROOT / rel) for rel in QUALIFICATIONS))

        for seed in SEEDS:
            before_deadline()
            initial, initial_opt, catalog, frozen, identity = load_initial(seed)
            seed_report = bound_json(training_run / str(seed) / 'result.json')
            same(seed_report, report['seeds'][str(seed)], 'seed report')
            same(seed_report['identity'], identity, 'F34 predecessor identity')
            if set(seed_report['arms']) != set(ARMS) or seed_report['initial_parameter_hash'] != parameter_hash(initial):
                raise ValueError('Matched initial state/arm inventory')
            final_parameters = {}
            with Pool(output / str(seed) / 'local-workers', plan['workers'], catalog, frozen) as pool:
                for arm in ARMS:
                    arm_report = bound_json(training_run / str(seed) / arm / 'result.json')
                    same(arm_report, seed_report['arms'][arm], 'arm report')
                    states = {c: checkpoint(seed, arm, c, input_hash) for c in (0, 255, 256)}
                    if state_hash(*states[0][:2]) != state_hash(initial, initial_opt):
                        raise ValueError('Same F34 parameters/reset Adam not preserved')
                    final, final_opt, _ = states[256]
                    if (parameter_hash(final) != arm_report['final_parameter_hash']
                            or state_hash(final, final_opt) != arm_report['final_state_hash']):
                        raise ValueError('Final report/checkpoint identity')
                    final_parameters[arm] = final
                    for cohort in COHORTS:
                        p, opt, metadata = states[cohort]
                        update = bound_json(training_run / str(seed) / arm / f'update-{cohort:04d}.json')
                        optimizer_seed = derived_seed('f35-optimizer-v1', seed, cohort)
                        demo_seed = derived_seed('f35-demonstration-minibatches-v1', seed)
                        if (update['cohort'] != cohort or update['arm'] != arm
                                or update['optimizer_step'] != (cohort + 1) * 8
                                or parameter_hash(p) != update['parameter_hash_before']
                                or update['optimizer_seed'] != optimizer_seed
                                or update['demo_seed'] != (demo_seed if arm == 'assisted' else None)):
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
                            spec = training_specs(plan, seed, first, STREAMS[label])[0]
                            same(sample['spec'], spec, 'training sample schedule')
                            before_deadline()
                            replay, _ = pool.collect(p, [spec])
                            if len(replay) != 1:
                                raise ValueError('Training replay game count')
                            rows, actual = replay[0]
                            row_count = check_game(rows, actual, expected_game, 'train', strata[label]['selection'])
                            counts.update(training_games=1, game_executions=1)
                            record({'kind': 'fresh_training', 'seed': seed, 'arm': arm, 'cohort': cohort,
                                'stream': label, 'learner_rows': row_count, 'rejected_attempts': actual['unavailable_choices']})
                        batch_path = bind(training_run / str(seed) / arm / f'batch-{cohort:04d}.npz')
                        with np.load(batch_path, allow_pickle=False) as saved:
                            batch = {key: saved[key] for key in saved.files}
                        if fingerprint(batch) != update['ledger']['batch_sha256']:
                            raise ValueError('Sampled selected-batch fingerprint')
                        before_deadline()
                        q, qo, metrics = optimize(arm, p, opt, batch, optimizer_seed, dataset, demo_seed, cohort)
                        if (state_hash(q, qo) != update['state_hash_after']
                                or parameter_hash(q) != update['parameter_hash_after'] or int(qo[2]) != update['optimizer_step']):
                            raise ValueError('Independent sampled next-update state')
                        same(metrics, update['metrics'], 'independent next-update metrics')
                        counts.update(sampled_updates=1, optimizer_steps_recomputed=8)
                        record({'kind': 'fresh_update', 'seed': seed, 'arm': arm, 'cohort': cohort,
                            'selected_rows': len(batch['actions']), 'batch_sha256': fingerprint(batch)})
                    for opponent in OPPONENTS:
                        saved = bound_json(training_run / str(seed) / arm / 'final' / f'{opponent}-000.json')
                        specs = evaluation_specs(plan, seed, opponent, 0, 1)
                        if len(saved['specs']) != 8 or len(saved['games']) != 8:
                            raise ValueError('Evaluation source chunk inventory')
                        same(saved['specs'][:2], specs, 'evaluation first-pair schedule')
                        before_deadline()
                        replay, _ = pool.collect(final, specs)
                        if len(replay) != 2:
                            raise ValueError('Evaluation replay pair count')
                        for (rows, actual), expected_game in zip(replay, saved['games'][:2]):
                            check_game(rows, actual, expected_game, 'development')
                        counts.update(evaluation_games=2, game_executions=2)
                        record({'kind': 'fresh_local_pair', 'seed': seed, 'arm': arm, 'opponent': opponent})
            endpoints = external_report['seeds'][str(seed)]
            if set(endpoints) != {'initial', *ARMS}:
                raise ValueError('External endpoint inventory')
            for endpoint in ('initial', *ARMS):
                p = initial if endpoint == 'initial' else final_parameters[endpoint]
                panel = bound_json(external_run / str(seed) / endpoint / 'result.json')
                same(panel, endpoints[endpoint], 'external endpoint report')
                if panel['parameter_hash'] != parameter_hash(p) or len(panel['games']) != 2 * plan['external_pairs_per_seed_state']:
                    raise ValueError('External endpoint policy/count')
                specs = evaluation_specs(plan, seed, 'foul_play', 0, 1)
                for spec, game in zip(specs, panel['games'][:2]):
                    source = external_run / str(seed) / endpoint / f"000-{int(spec['swap'])}"
                    if ((ROOT / game['game_record']).resolve() != source / 'result.json'
                            or digest(source / 'result.json') != game['game_record_sha256']):
                        raise ValueError('External source record binding')
                    coordinates = {'run_seed': spec['seed'], 'episode_index': 0, 'swap': spec['swap'],
                        'external_side': 'p1' if spec['swap'] else 'p2', 'split': 'development',
                        'fp_seed': derived_seed('f35-external-sampling-v1', seed, 0, spec['swap']),
                        'max_turns': plan['max_turns'], 'search_time_ms': plan['teacher_data']['search_time_ms']}
                    action_seed = derived_seed('actor', spec['actor_seed'], 0, spec['swap'], spec['learner_side'])
                    original, replay = replay_source(source, output / str(seed) / 'external' / endpoint / str(int(spec['swap'])),
                        p, catalog, coordinates=coordinates, action_seed=action_seed)
                    for key in ('value_mean', 'terminal_value_mse'):
                        same(replay[key], game[key], 'recomputed external summary/' + key)
                    for key in ('outcome', 'provenance', 'decisions'):
                        same(original[key], game[key], 'external summary/' + key)
                    counts.update(external_action_replay_games=1, game_executions=1)
                    record({'kind': 'recorded_external_actions', 'seed': seed, 'endpoint': endpoint,
                        'swap': spec['swap'], **replay, 'foul_play_search_reexecuted': False})
            for role in ('train', 'probe'):
                selected = [g for g in teacher_report['games'] if g['host_seed'] == seed and g['role'] == role and g['episode_index'] == 0]
                if len(selected) != 2 or [g['swap'] for g in selected] != [False, True]:
                    raise ValueError('Teacher first-pair sample inventory')
                for game in selected:
                    source = demonstrations / game['directory']
                    coordinates = {k: game[k] for k in ('run_seed', 'episode_index', 'swap', 'split', 'external_side', 'fp_seed')}
                    actor_seed = derived_seed('f35-teacher-host-actor-v1', role, seed)
                    side = 'p1' if game['external_side'] == 'p2' else 'p2'
                    action_seed = derived_seed('actor', actor_seed, game['episode_index'], game['swap'], side)
                    original, replay = replay_source(source, output / str(seed) / 'teacher' / role / str(int(game['swap'])),
                        initial, catalog, coordinates=coordinates, action_seed=action_seed, labels=True)
                    same(original['game_id'], game['game_id'], 'teacher game identity')
                    counts.update(teacher_action_replay_games=1, game_executions=1)
                    record({'kind': 'recorded_teacher_actions', 'seed': seed, 'role': role,
                        'swap': game['swap'], **replay, 'foul_play_search_reexecuted': False})
        if dict(counts) != EXPECTED:
            raise ValueError('Prespecified verification count mismatch: ' + str(dict(counts)))
        verify_inputs(training_run / 'inputs')
        for path, expected in inventory.items():
            if digest(path) != expected:
                raise ValueError('Verification input changed: ' + path)
        result = {'status': 'passed', 'input_hash': input_hash, 'counts': dict(counts),
            'expected_counts': EXPECTED, 'records': records, 'input_files': inventory,
            'qualification_files': qualifications, 'elapsed_seconds': time.perf_counter() - begun,
            'coverage': {'exhaustive': False, 'external_search_reexecuted': False,
                'training': 'First complete game per stream at cohorts 0 and 255, all three seeds and both arms.',
                'local': 'First paired schedule per opponent at both final arms, all three seeds.',
                'external': 'First paired schedule at all three states, every learner policy/value decision recomputed.',
                'teacher': 'First train and probe paired schedules per host, accepted labels re-encoded and all attempts replayed.',
                'updates': 'Twelve original selected batches recomputed with identical train-only demo sampling.',
                'unavailable_choices': 'Explicit deterministic qualified fixtures plus the actual sampled-replay rejection counts.'}}
        atomic_json(output / 'report.json', result)
        return result
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error': repr(exc), 'counts': dict(counts),
            'records': records, 'input_files': inventory, 'elapsed_seconds': time.perf_counter() - begun})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--external-run', type=Path, required=True)
    parser.add_argument('--demonstrations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.training_run, args.external_run, args.demonstrations, args.output)
    print({'status': result['status'], 'counts': result['counts']}, flush=True)
