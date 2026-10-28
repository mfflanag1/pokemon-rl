"""Fresh teacher-executed games with accepted-action, player-local demonstrations."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pokemon_external_benchmark.transport import JsonWorker, ROOT
from pokemon_structured_policy.features import encode_packed, FEATURES


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@lru_cache(maxsize=1)
def public_catalog():
    plan = json.loads((ROOT / 'contracts/observed-continuation-v1/plan.json').read_text())
    path = ROOT / plan['catalog']
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan['catalog_sha256']:
        raise ValueError('Frozen public catalog drift')
    return json.loads(path.read_text())


def label_candidate(player, history, external, *, step_index, external_input_sha256, catalog=None):
    """Only the acting teacher's projection/history are accepted at this boundary."""
    if not any(player['mask']):
        if external != {'waiting': True}:
            raise ValueError('Waiting teacher produced a decision')
        return None
    if history['side'] != player['side']:
        raise ValueError('Foreign teacher history')
    action = external.get('action')
    if (type(action) is not int or not 0 <= action < 14 or not player['mask'][action]
            or external.get('request_id') != player['request_id']):
        raise ValueError('Teacher action/request does not match advertised legal request')
    features, mask = encode_packed(player, public_catalog() if catalog is None else catalog, history)
    if features.dtype != np.float32 or features.shape != (FEATURES,) or mask.dtype != np.bool_ or mask.shape != (14,):
        raise ValueError('Teacher feature contract')
    if not np.array_equal(mask, np.asarray(player['mask'], dtype=bool)) or not mask[action]:
        raise ValueError('Teacher encoder mask differs from current request')
    return {'features': features, 'mask': mask, 'action': action,
            'metadata': {'step_index': step_index, 'request_id': player['request_id'],
                         'side': player['side'], 'action': action,
                         'player_sha256': json_hash(player), 'history_sha256': json_hash(history),
                         'features_sha256': array_hash(features), 'mask_sha256': array_hash(mask),
                         'external_input_sha256': external_input_sha256}}


def accept_label(candidate, submitted):
    if candidate is None:
        return None
    side = candidate['metadata']['side']
    matches = [(i, row) for i, row in enumerate(submitted) if row['side'] == side]
    if len(matches) != 1:
        raise ValueError('Exactly one teacher submission required')
    index, row = matches[0]
    if (row['action'] != candidate['action'] or row['request_id'] != candidate['metadata']['request_id']
            or type(row['accepted']) is not bool):
        raise ValueError('Teacher submitted action/request mismatch')
    if not row['accepted']:
        return None
    candidate['metadata']['submission_index'] = index
    return candidate


def save_labels(output, record, labels):
    n = len(labels)
    arrays = {'features': np.stack([r['features'] for r in labels]) if n else np.empty((0, FEATURES), np.float32),
              'mask': np.stack([r['mask'] for r in labels]) if n else np.empty((0, 14), bool),
              'actions': np.asarray([r['action'] for r in labels], np.int32),
              'step_indices': np.asarray([r['metadata']['step_index'] for r in labels], np.int32),
              'request_ids': np.asarray([r['metadata']['request_id'] for r in labels], np.int32)}
    with (output / 'demonstrations.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    _write_json(output / 'labels.json', {'schema_version': 'f35-teacher-labels-v1',
        'game_id': record.get('game_id'), 'split': record['split'], 'external_side': record['external_side'],
        'labels': [{'row': i, **r['metadata']} for i, r in enumerate(labels)]})
    record['accepted_labels'] = n
    record['demonstrations'] = {'arrays': 'demonstrations.npz', 'metadata': 'labels.json',
        'features': FEATURES, 'action_count': 14,
        'scope': 'Accepted teacher actions; no PPO log probabilities, advantages, returns or critic targets.'}


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
        'teacher_bookkeeping': 'Commit chosen move only on accepted submission; restore prior move after rejection.'}
    started, labels = time.perf_counter(), []

    def request(worker, op, **payload):
        if deadline_epoch is not None:
            remaining = deadline_epoch - time.time()
            if remaining <= 0:
                raise TimeoutError('F35 acquisition/goal deadline reached')
            worker.timeout = min(60., remaining)
        return worker.request(op, **payload)

    try:
        with JsonWorker(['node', 'packages/teacher-assisted/external-worker.mjs'], output / 'showdown.stderr') as node, \
             JsonWorker([str(ROOT / '.venv-f34-external/bin/python'), '-m', 'pokemon_teacher_assisted.teacher'],
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
                    external = request(teacher, 'choose', input=bundle['external'])
                    decision_wall = time.perf_counter() - decision_started
                    input_hash = json_hash(bundle['external'])
                    teacher_player = frame['players'][external_side]
                    teacher_history = frame['observable_history'][external_side]
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
