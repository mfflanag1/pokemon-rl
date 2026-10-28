"""Whole-trajectory collection with same-player links across replacement decisions.

Only a fresh joint move boundary may be administratively cut. Frozen-policy
values bootstrap that boundary; terminal win/loss/tie rewards are paid once.
"""
from collections import Counter
from hashlib import sha256
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax_learner.core import linked_gae
from .features import encode
from .policy import act, update, finite_parameters

SIDES = ['p1', 'p2']


def parameter_hash(parameters):
    digest = sha256()
    for name, value in sorted(parameters.items()):
        array = np.asarray(value)
        digest.update(name.encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def predict(parameters, frame, run_seed, counters):
    encoded = [encode(frame['players'][side]) for side in SIDES]
    features = np.stack([row[0] for row in encoded])
    mask = np.stack([row[1] for row in encoded])
    root = jax.random.key(run_seed)
    keys = jnp.stack([jax.random.fold_in(root, counters[side]) for side in SIDES])
    actions, logs, values = map(np.asarray, act(parameters, jnp.asarray(features), jnp.asarray(mask), keys))
    if not np.all(np.isfinite(logs)) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('Nonfinite/unbounded actor output')
    for i, side in enumerate(SIDES):
        if mask[i].any() and (actions[i] < 0 or not mask[i, actions[i]]):
            raise ValueError('Actor selected masked action')
    return features, mask, actions, logs, values


def finish_rows(rows, last, outcome, bootstrap):
    terminal = outcome['kind'] == 'terminal'
    if outcome['kind'] not in ['terminal', 'administrative_cutoff']:
        raise ValueError('Unknown episode boundary')
    for i, side in enumerate(SIDES):
        index = last[side]
        if index is None:
            continue
        rows[index]['terminal'] = terminal
        rows[index]['reward'] = outcome['rewards'][side] if terminal else 0.
        rows[index]['bootstrap'] = 0. if terminal else float(bootstrap[i])


def collect_episode(worker, parameters, run_seed, episode_index, rows, max_turns=60, curriculum=False):
    frame = worker.request('start', run_seed=run_seed, episode_index=episode_index, curriculum=curriculum)
    provenance = frame['provenance']
    last = {side: None for side in SIDES}
    # Disjoint side/episode counters; no dependence on battle RNG or hidden state.
    counters = {side: 1_000_000*episode_index + i*400_000 for i, side in enumerate(SIDES)}
    phases, decisions, rejections, initial = Counter(), 0, 0, len(rows)
    while True:
        features, mask, actions, logs, values = predict(parameters, frame, run_seed, counters)
        if frame['outcome'] is not None:
            finish_rows(rows, last, frame['outcome'], values)
            break
        joint = all(frame['players'][s]['phase'] == 'move' and mask[i].any() for i, s in enumerate(SIDES))
        if joint and frame['players']['p1']['turn'] >= max_turns:
            frame = worker.request('cutoff')
            finish_rows(rows, last, frame['outcome'], values)
            break
        if decisions >= 10_000:
            raise RuntimeError('Safety limit reached away from admitted cutoff: discard incomplete episode')
        submission, request_ids = {}, {}
        for i, side in enumerate(SIDES):
            if not mask[i].any():
                continue
            index = len(rows)
            if last[side] is not None:
                rows[last[side]]['next_index'] = index
            rows.append({'features': features[i].copy(), 'mask': mask[i].copy(), 'action': int(actions[i]),
                         'old_logp': float(logs[i]), 'value': float(values[i]), 'reward': 0.,
                         'next_index': -1, 'bootstrap': 0., 'terminal': False,
                         'trajectory': episode_index*2+i})
            last[side] = index
            submission[side] = int(actions[i]); request_ids[side] = frame['players'][side]['request_id']
            phases[frame['players'][side]['phase']] += 1
            decisions += 1; counters[side] += 1
        frame = worker.request('step', actions=submission, request_ids=request_ids)
        rejections += sum(not row['accepted'] for row in frame['submitted'])
    if len(rows)-initial != decisions:
        raise ValueError('Decision ledger mismatch')
    return {'episode_id': frame['episode_id'], 'provenance': provenance, 'decisions': decisions,
            'phases': dict(phases), 'unavailable_choices': rejections, 'outcome': frame['outcome']}


def build_batch(rows):
    if not rows:
        return None
    advantages, returns = linked_gae(
        np.asarray([r['reward'] for r in rows], np.float32),
        np.asarray([r['value'] for r in rows], np.float32),
        np.asarray([r['next_index'] for r in rows], np.int32),
        np.asarray([r['bootstrap'] for r in rows], np.float32),
        np.asarray([r['terminal'] for r in rows], bool),
        np.asarray([r['trajectory'] for r in rows], np.int32), gamma=1., lam=.95)
    if np.any(np.asarray(returns) < -1e-5) or np.any(np.asarray(returns) > 1+1e-5):
        raise ValueError('Bounded-value target outside admitted range')
    return {'features': jnp.asarray(np.stack([r['features'] for r in rows])),
            'mask': jnp.asarray(np.stack([r['mask'] for r in rows])),
            'actions': jnp.asarray([r['action'] for r in rows], jnp.int32),
            'old_logp': jnp.asarray([r['old_logp'] for r in rows], jnp.float32),
            'advantages': advantages, 'returns': returns}


def train_cohort(worker, parameters, optimizer, *, run_seed, first_episode, games=8, max_turns=60, curriculum=False):
    if games != 8:
        raise ValueError('F1 cohort is fixed at eight games')
    rows, episodes = [], []
    before = parameter_hash(parameters)
    started = time.perf_counter()
    for episode in range(first_episode, first_episode+games):
        episodes.append(collect_episode(worker, parameters, run_seed, episode, rows, max_turns, curriculum))
    collection_seconds = time.perf_counter()-started
    batch = build_batch(rows)
    metrics = []
    m, v, step = optimizer
    started = time.perf_counter()
    if batch is not None:
        for _ in range(2):
            parameters, m, v, step, result = update(parameters, m, v, step, batch)
            finite_parameters(parameters)
            record = {k: float(x) for k, x in result.items()}
            if not all(np.isfinite(x) for x in record.values()):
                raise ValueError('Nonfinite optimizer metrics')
            metrics.append(record)
    update_seconds = time.perf_counter()-started
    after = parameter_hash(parameters)
    if batch is not None and before == after:
        raise ValueError('Nonempty cohort did not change model parameters')
    return parameters, (m, v, step), {'episodes': episodes, 'decisions': len(rows),
        'collection_seconds': collection_seconds, 'update_seconds_including_compile': update_seconds,
        'decisions_per_collection_second': len(rows)/collection_seconds,
        'full_cycle_seconds': collection_seconds+update_seconds,
        'decisions_per_full_cycle_second': len(rows)/(collection_seconds+update_seconds),
        'parameter_hash_before': before, 'parameter_hash_after': after, 'parameters_changed': before != after,
        'optimizer_steps': len(metrics), 'metrics': metrics,
        'returns_min': float(batch['returns'].min()) if batch else None,
        'returns_max': float(batch['returns'].max()) if batch else None}
