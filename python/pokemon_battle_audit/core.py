"""Exact public-view replay, single-action continuation and train prefix resets."""
from collections import Counter
import hashlib
import json
from pokemon_full.features import encode
from pokemon_full_train.storage import canonical
from pokemon_full_train.learning import derived_seed
from pokemon_throughput.collector import batched_actor

SIDES = ('p1', 'p2')


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def feature_hash(player):
    return hashlib.sha256(encode(player)[0].tobytes()).hexdigest()


def public_context(player):
    keys = ('species','condition','moves','boosts','tera','types')
    def active(roster):
        mon = next((m for m in roster if m and m.get('active')), {})
        return {k:mon.get(k) for k in keys}
    return {'own':active(player['own']),'opponent':active(player['opponent']),
            'side_conditions':player['side_conditions'],'weather':player['weather'],'terrain':player['terrain']}


def start_spec(game, seed):
    return dict(run_seed=seed, episode_index=game['episode'], split=game['split'], swap=game['swap'])


def pending(frame):
    return [s for s in SIDES if any(frame['players'][s]['mask'])]


def cutoff_boundary(frame, max_turns):
    return (not frame['outcome'] and all(frame['players'][s]['phase'] == 'move' and
            any(frame['players'][s]['mask']) for s in SIDES) and frame['players']['p1']['turn'] >= max_turns)


def traced_command(frame, trace, cursor, learner):
    actions, ids = {}, {}
    for side in pending(frame):
        if cursor >= len(trace):
            raise ValueError('Truncated factual trace')
        row = trace[cursor]
        p = frame['players'][side]
        if row[0] != side or row[1] != p['request_id'] or row[4] != feature_hash(p):
            raise ValueError('Factual player feature/request mismatch')
        if side != learner and row[2] != frame['tactical_actions'][side]:
            raise ValueError('Frozen tactical opponent mismatch')
        actions[side], ids[side] = row[2], row[1]
        cursor += 1
    return dict(actions=actions, request_ids=ids), cursor


class Selector:
    """Deterministic first qualifying point per game; no outcome argument."""
    def __init__(self, families, quota):
        self.families, self.quota = tuple(families), quota
        self.counts, self.games = Counter(), set()

    def select(self, game_id, diagnostic):
        if game_id in self.games or not diagnostic['disagreement']:
            return []
        tags = [t for t in self.families if t in diagnostic['tags'] and self.counts[t] < self.quota]
        if tags:
            self.games.add(game_id)
            self.counts.update(tags)
        return tags


def actor_seeds(actor_seed, episode, swap):
    return [derived_seed('actor', actor_seed, episode, swap, s) for s in SIDES]


def policy_command(parameters, frame, seeds, counters, learner, capacity=16):
    outputs = batched_actor(parameters, [frame['players'][s] for s in SIDES], seeds, counters, capacity)
    actions, ids, values = {}, {}, {}
    for i, side in enumerate(SIDES):
        if any(frame['players'][side]['mask']):
            actions[side] = outputs[i][2] if side == learner else frame['tactical_actions'][side]
            ids[side] = frame['players'][side]['request_id']
            values[side] = outputs[i][4] if side == learner else None
    return dict(actions=actions, request_ids=ids), values


def record_step(frame, command, values, result):
    return {'before': fingerprint(frame['players']), 'command': command, 'values': values,
            'submitted': result['submitted'], 'after': fingerprint(result['players']),
            'diagnostics': result['diagnostics']}


def continue_game(worker, parameters, frame, seeds, counts, learner, plan):
    steps = []
    while not frame['outcome']:
        if cutoff_boundary(frame, plan['max_turns']):
            frame = worker.request('cutoff')
            break
        if sum(counts) >= 10000:
            raise RuntimeError('Decision hard limit')
        command, values = policy_command(parameters, frame, seeds, counts, learner, plan['actor_capacity'])
        after = worker.request('step', **command)
        steps.append(record_step(frame, command, values, after))
        for i,s in enumerate(SIDES):
            counts[i] += int(s in command['actions'])
        frame = after
    return steps, frame


def validate_scenario(scenario):
    if scenario.get('schema_version') != 'f11-train-scenario-v1' or scenario.get('split') != 'train':
        raise ValueError('Only F11 train scenarios admitted')
    if scenario.get('start', {}).get('split') != 'train' or scenario['start'].get('swap') is not False:
        raise ValueError('Development/final or swapped scenario rejected')
    if scenario.get('learner_side') not in SIDES:
        raise ValueError('Invalid scenario seat')
    payload = {k:v for k,v in scenario.items() if k != 'content_hash'}
    if scenario.get('content_hash') != fingerprint(payload):
        raise ValueError('Scenario content changed')
    if any(t.get('partition') != 'train' for t in scenario['provenance']['teams']):
        raise ValueError('Nontrain provenance')


def reset_scenario(worker, scenario):
    """No hidden engine restore; replay public action prefix, then hand off BEFORE target.

    Returned prefix contains no training rows. Caller collects fresh actions and
    rewards from this live frame through a natural terminal or explicit cutoff.
    The content hash and every prefix frame are checked, as is regenerated R7
    team admission in the unchanged official simulator session.
    """
    validate_scenario(scenario)
    frame = worker.request('start', **scenario['start'])
    if frame['provenance'] != scenario['provenance']:
        raise ValueError('Scenario team provenance changed')
    counts = [0, 0]
    for step in scenario['prefix']:
        if frame['outcome'] or fingerprint(frame['players']) != step['before']:
            raise ValueError('Scenario prefix mismatch')
        frame = worker.request('step', **step['command'])
        if fingerprint(frame['players']) != step['after'] or frame['submitted'] != step['submitted']:
            raise ValueError('Scenario prefix transition mismatch')
        for i,s in enumerate(SIDES):
            counts[i] += int(s in step['command']['actions'])
    target = frame['players'][scenario['learner_side']]
    if (frame['outcome'] or counts != scenario['counters'] or
        fingerprint(frame['players']) != scenario['target_frame_hash'] or
        feature_hash(target) != scenario['target_feature_hash'] or not any(target['mask'])):
        raise ValueError('Scenario handoff is not the unsolved target')
    return frame, counts
