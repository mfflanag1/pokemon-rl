"""F37 game boundary: same F35 game algorithm, own legal mask and raw reply journal.

The narrowly copied run loop preserves frozen collection, label, split, feedback,
and cutoff behavior. Only the teacher command and its pre-validation journal are
new. Unchanged helpers continue to come from the sealed F35 implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from pokemon_teacher_runtime.transport import JsonWorker, ROOT
from pokemon_teacher_assisted.external import (json_hash, _write_json, label_candidate,
                                               accept_label, save_labels)
from . import TEACHER_POLICY, RESPONSE_JOURNAL


def append_response(path, row):
    """Close and flush each received reply before the parent validates its action."""
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()


def rejected_wire_response(exception):
    """Recover a decoded worker error reply retained by unchanged F36 transport."""
    prefix = 'External worker rejected command: '
    message = str(exception)
    if not message.startswith(prefix):
        return None  # Timeouts/EOF need not have produced any reply.
    try:
        return json.loads(message[len(prefix):])
    except (ValueError, TypeError):
        return None


def run_game(output, *, run_seed, episode_index, swap, external_side, split,
             search_time_ms=100, fp_seed, max_turns=240, learner,
             learner_uses_history=True, purpose, capture_demonstrations=False, deadline_epoch=None):
    """Run one prescribed game, preserving rejected attempts and never replacing it."""
    if split not in ('train', 'development'):
        raise ValueError('Only explicit train/development splits are admitted')
    if (type(run_seed) is not int or not 0 <= run_seed < 2**32 or type(episode_index) is not int
            or episode_index < 0 or type(swap) is not bool or external_side not in ('p1', 'p2')
            or type(fp_seed) is not int or not 0 <= fp_seed < 2**32
            or type(max_turns) is not int or max_turns < 1 or search_time_ms != 100
            or type(learner_uses_history) is not bool or type(capture_demonstrations) is not bool
            or not isinstance(purpose, str) or not purpose.strip()):
        raise ValueError('Explicit valid F35 game coordinates and fixed teacher budget required')
    if learner is not None and not callable(learner):
        raise ValueError('Learner must be an explicit callback or engineering tactical None')
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError('F35 acquisition/goal deadline reached before launch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    learner_side = 'p1' if external_side == 'p2' else 'p2'
    record = {'schema_version': 'f35-external-game-v1', 'purpose': purpose,
        'run_seed': run_seed, 'episode_index': episode_index, 'swap': swap, 'external_side': external_side,
        'split': split, 'fp_seed': fp_seed, 'search_time_ms': search_time_ms, 'parallelism': 1, 'search_threads': 1,
        'max_turns': max_turns, 'steps': [], 'status': 'running', 'outcome': None,
        'illegal_actions': {'p1': 0, 'p2': 0}, 'rejected_actions': {'p1': 0, 'p2': 0},
        'learner_uses_history': learner_uses_history, 'capture_demonstrations': capture_demonstrations,
        'teacher_policy': TEACHER_POLICY, 'teacher_response_journal': RESPONSE_JOURNAL,
        'teacher_policy_counts': {'forced_request': 0, 'native_legal_root': 0, 'waiting': 0, 'errors': 0},
        'teacher_bookkeeping': 'Commit chosen move only on accepted submission; restore prior move after rejection.'}
    started, labels = time.perf_counter(), []
    journal = output / RESPONSE_JOURNAL
    journal.touch(exist_ok=False)

    def request(worker, op, **payload):
        if deadline_epoch is not None:
            remaining = deadline_epoch - time.time()
            if remaining <= 0:
                raise TimeoutError('F35 acquisition/goal deadline reached')
            worker.timeout = min(60., remaining)
        return worker.request(op, **payload)

    try:
        with JsonWorker(['node', 'packages/teacher-assisted/external-worker.mjs'], output / 'showdown.stderr') as node, \
             JsonWorker([str(ROOT / '.venv-f34-external/bin/python'), '-m', 'pokemon_teacher_legality.teacher'],
                        output / 'foul-play.stderr') as teacher:
            record['worker'] = request(node, 'hello')
            request(teacher, 'start', side=external_side, search_time_ms=search_time_ms, seed=fp_seed)
            bundle = request(node, 'start', run_seed=run_seed, episode_index=episode_index, swap=swap,
                             split=split, external_side=external_side, tactical_side=learner_side if learner is None else None)
            record['provenance'] = bundle['frame']['provenance']
            record['game_id'] = bundle['frame']['episode_id']
            quarantine = set(json.loads((ROOT / 'contracts/policy-reset-v1/plan.json').read_text())['quarantined_team_keys'])
            if (record['provenance']['split'] != split or any(t['partition'] != split or t['team_key'] in quarantine
                    for t in record['provenance']['teams'])):
                raise ValueError('Teacher game initial partition/quarantine violation')
            with (output / 'transcript.jsonl').open('x') as transcript:
                def write_frame(value):
                    transcript.write(json.dumps({'frame': value['frame'], 'external': value['external']}, allow_nan=False) + '\n')
                    transcript.flush()
                for index in range(2000):
                    frame = bundle['frame']
                    write_frame(bundle)
                    if frame['outcome']:
                        record['outcome'] = frame['outcome']
                        break
                    if all(frame['players'][s]['phase'] == 'move' and frame['players'][s]['turn'] >= max_turns for s in ('p1', 'p2')):
                        bundle = request(node, 'cutoff')
                        write_frame(bundle)
                        record['outcome'] = bundle['frame']['outcome']
                        break
                    decision_started = time.perf_counter()
                    input_hash = json_hash(bundle['external'])
                    teacher_player = frame['players'][external_side]
                    teacher_history = frame['observable_history'][external_side]
                    coordinates = {'schema_version': 'f37-teacher-response-v1',
                        'step_index': index, 'request_id': teacher_player['request_id'],
                        'legal_mask': list(teacher_player['mask']), 'external_input_sha256': input_hash}
                    try:
                        external = request(teacher, 'choose', input=bundle['external'],
                            legal_mask=coordinates['legal_mask'], request_id=coordinates['request_id'])
                    except BaseException as exc:
                        record['teacher_policy_counts']['errors'] += 1
                        error_row = {**coordinates, 'status': 'error', 'raw_response': None,
                            'error': str(exc), 'error_type': type(exc).__name__,
                            'raw_error_response': rejected_wire_response(exc)}
                        try:
                            append_response(journal, error_row)
                        except BaseException as journal_error:
                            exc.add_note('Teacher error journal also failed: ' + repr(journal_error))
                        raise
                    # Persist the received object before any action/label validation.
                    append_response(journal, {**coordinates, 'status': 'response', 'raw_response': external})
                    if external == {'waiting': True}:
                        record['teacher_policy_counts']['waiting'] += 1
                    else:
                        kind = external.get('selection', {}).get('kind') if isinstance(external, dict) else None
                        key = {'forced-request': 'forced_request', 'native-legal-root': 'native_legal_root'}.get(kind)
                        if key is None or external.get('teacher_policy') != TEACHER_POLICY:
                            raise ValueError('Missing F37 teacher selection policy metadata')
                        record['teacher_policy_counts'][key] += 1
                    decision_wall = time.perf_counter() - decision_started
                    candidate = label_candidate(teacher_player, teacher_history, external,
                        step_index=index, external_input_sha256=input_hash) if capture_demonstrations else None
                    actions, ids = {}, {}
                    for side in ('p1', 'p2'):
                        player = frame['players'][side]
                        if not any(player['mask']):
                            continue
                        if side == external_side:
                            action = external.get('action')
                            if external.get('request_id') != player['request_id']:
                                raise ValueError('Teacher request ID mismatch')
                        elif learner is None:
                            action = frame['tactical_actions'][side]
                        elif learner_uses_history:
                            action = learner(player, index, frame['observable_history'][side])
                        else:
                            action = learner(player, index)
                        if type(action) is not int or not 0 <= action < 14 or not player['mask'][action]:
                            record['illegal_actions'][side] += 1
                            raise ValueError(f'Illegal {side} action {action}')
                        actions[side], ids[side] = action, player['request_id']
                    if not any(teacher_player['mask']) and external != {'waiting': True}:
                        raise ValueError('Waiting teacher returned an action')
                    step = {'index': index, 'turn': frame['players'][learner_side]['turn'],
                        'actions': actions, 'request_ids': ids, 'external': external, 'external_phase': teacher_player['phase'],
                        'external_wall_seconds': decision_wall, 'external_input_sha256': input_hash}
                    record['steps'].append(step)
                    bundle = request(node, 'step', actions=actions, request_ids=ids)
                    step['submitted'] = bundle['frame'].get('submitted', [])
                    if len(step['submitted']) != len(actions):
                        raise ValueError('Submitted action inventory mismatch')
                    for submitted in step['submitted']:
                        if not submitted['accepted']:
                            record['rejected_actions'][submitted['side']] += 1
                        if submitted['side'] == external_side:
                            step['teacher_feedback'] = request(teacher, 'feedback',
                                request_id=ids[external_side], accepted=submitted['accepted'])
                    accepted = accept_label(candidate, step['submitted'])
                    if accepted is not None:
                        labels.append(accepted)
                else:
                    raise RuntimeError('F35 game exceeded fixed 2,000-decision-group cap')
        if record['outcome']['kind'] not in ('terminal', 'administrative_cutoff'):
            raise ValueError('Game did not reach an admitted outcome')
        record['status'] = 'completed'
    except BaseException as exc:
        record['status'], record['error'] = 'failed', repr(exc)
        raise
    finally:
        record['seconds'] = time.perf_counter() - started
        decisions = [s for s in record['steps'] if 'action' in s['external']]
        record['decisions'] = sum(len(s['actions']) for s in record['steps'])
        record['foul_play_decisions'] = len(decisions)
        record['total_foul_play_seconds'] = sum(s['external_wall_seconds'] for s in decisions)
        record['mean_foul_play_seconds'] = record['total_foul_play_seconds'] / max(1, len(decisions))
        if capture_demonstrations:
            save_labels(output, record, labels)
        _write_json(output / 'result.json', record)
    return record
