"""Player-only actors, whole-episode GAE and fixed-shape, masked PPO updates."""
from collections import Counter
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_fixed.math import adam_step
from pokemon_full.features import encode
from pokemon_full.policy import act, forward, distribution, finite_parameters
from pokemon_full.learner import finish_rows

SIDES = ['p1', 'p2']
PPO = {'epochs':2,'minibatch_rows':256,'learning_rate':.0003,'clip':.2,'entropy':.01,
       'value_coefficient':.5,'gamma':1.,'lambda':.95}
BATCH = PPO['minibatch_rows']


def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], 'big')


def actor(parameters, player, key):
    """Accepts one projected player only; coordinator provenance never reaches it."""
    x, mask = encode(player)
    action, logp, value = map(np.asarray, act(parameters, jnp.asarray(x[None]),
        jnp.asarray(mask[None]), jnp.asarray(key[None])))
    if not np.isfinite(logp[0]) or not np.isfinite(value[0]) or not 0 <= value[0] <= 1:
        raise ValueError('Invalid actor output')
    if mask.any() and (action[0] < 0 or not mask[action[0]]):
        raise ValueError('Illegal actor action')
    return x, mask, int(action[0]), float(logp[0]), float(value[0])


def baseline(player, kind, rng):
    _, mask = encode(player)
    legal = np.flatnonzero(mask)
    if not len(legal):
        return -1
    if kind == 'random':
        return int(rng.choice(legal))
    if kind != 'base_power':
        raise ValueError('Unknown frozen opponent')
    # Deliberately simple: raw request base power, not a damage calculator.
    moves = [int(a) for a in legal if a < 8]
    if any(a % 4 >= len(player['active_moves']) for a in moves):
        raise ValueError('Legal move mask disagrees with projected request')
    if moves:
        return max(moves, key=lambda a: (player['active_moves'][a % 4]['base_power'], -a))
    return int(legal[0])


def collect(worker, parameters, *, seed, episode, max_turns, split='train',
            swap=False, opponent=None, learner_side='p1', actor_seed=None):
    if split not in ['train', 'development'] or (split == 'development' and opponent is None):
        raise ValueError('Invalid collection role')
    if split == 'train' and (opponent is not None or swap):
        raise ValueError('Training profile is official self-play')
    frame = worker.request('start', run_seed=seed, episode_index=episode, split=split, swap=swap)
    provenance = frame['provenance']
    if any(t['partition'] != split for t in provenance['teams']):
        raise ValueError('Worker partition mismatch')
    rows, trace, values, phases = [], [], [], Counter()
    last = {s: None for s in SIDES}
    counts = {s: 0 for s in SIDES}
    actor_seed = seed if actor_seed is None else actor_seed
    roots = {s: jax.random.key(derived_seed('actor', actor_seed, episode, swap, s)) for s in SIDES}
    rngs = {s: np.random.default_rng(derived_seed('opponent', actor_seed, episode, swap, s)) for s in SIDES}
    decisions = rejections = 0
    while True:
        outputs = [actor(parameters, frame['players'][s], jax.random.fold_in(roots[s], counts[s])) for s in SIDES]
        if frame['outcome'] is not None:
            finish_rows(rows, last, frame['outcome'], [o[4] for o in outputs])
            break
        joint = all(frame['players'][s]['phase'] == 'move' and outputs[i][1].any() for i, s in enumerate(SIDES))
        if joint and frame['players']['p1']['turn'] >= max_turns:
            frame = worker.request('cutoff')
            finish_rows(rows, last, frame['outcome'], [o[4] for o in outputs])
            break
        if decisions >= 10000:
            raise RuntimeError('Episode decision cap: no incomplete trajectory enters an update')
        submission, ids = {}, {}
        for i, side in enumerate(SIDES):
            x, mask, action, logp, value = outputs[i]
            if not mask.any():
                continue
            controlled = opponent is None or side == learner_side
            if not controlled:
                action = baseline(frame['players'][side], opponent, rngs[side])
            if split == 'train':
                index = len(rows)
                if last[side] is not None:
                    rows[last[side]]['next_index'] = index
                rows.append({'features': x.copy(), 'mask': mask.copy(), 'action': action,
                    'old_logp': logp, 'value': value, 'reward': 0., 'next_index': -1,
                    'bootstrap': 0., 'terminal': False, 'trajectory': i})
                last[side] = index
            if controlled:
                values.append((side,value))
            trace.append([side, frame['players'][side]['request_id'], action,
                          value if controlled else None, hashlib.sha256(x.tobytes()).hexdigest()])
            submission[side], ids[side] = action, frame['players'][side]['request_id']
            counts[side] += 1
            decisions += 1
            phases[frame['players'][side]['phase']] += 1
        frame = worker.request('step', actions=submission, request_ids=ids)
        rejections += sum(not r['accepted'] for r in frame['submitted'])
    outcome = frame['outcome']
    reward = outcome['rewards'][learner_side] if outcome['kind'] == 'terminal' else None
    return rows, {'episode': episode, 'split': split, 'swap': swap, 'learner_side': learner_side,
        'opponent': opponent, 'provenance': provenance, 'outcome': outcome, 'decisions': decisions,
        'phases': dict(phases), 'unavailable_choices': rejections, 'trace': trace,
        'value_mean': float(np.mean([v for _,v in values])) if values else None,
        'terminal_value_mse': float(np.mean([(v-outcome['rewards'][s])**2 for s,v in values])) if reward is not None and values else None}


def objective(parameters, batch):
    valid = batch['valid']
    features = jnp.where(valid[:, None], batch['features'], 0.)
    mask = jnp.where(valid[:, None], batch['mask'], True)
    actions = jnp.where(valid, batch['actions'], 0)
    logits, values = forward(parameters, features)
    p, logs = distribution(logits, mask)
    old, advantages, targets = [jax.lax.stop_gradient(jnp.where(valid, batch[k], 0.))
        for k in ['old_logp', 'advantages', 'returns']]
    log_ratio = logs[jnp.arange(len(values)), actions]-old
    ratio = jnp.exp(log_ratio)
    active = jnp.where(advantages >= 0, ratio <= 1+PPO['clip'], ratio >= 1-PPO['clip'])
    surrogate = jnp.where(active, ratio, jax.lax.stop_gradient(jnp.clip(ratio, 1-PPO['clip'], 1+PPO['clip'])))*advantages
    def mean(a):
        return jnp.where(valid, a, 0.).sum()/jnp.maximum(valid.sum(), 1)
    actor_loss, value_loss, entropy = -mean(surrogate), .5*mean((values-targets)**2), -mean(jnp.sum(p*logs, 1))
    loss = actor_loss + PPO['value_coefficient']*value_loss - PPO['entropy']*entropy
    return loss, {'loss': loss, 'actor_loss': actor_loss, 'value_loss': value_loss, 'entropy': entropy,
                  'approx_kl': mean(ratio-1-log_ratio)}


@jax.jit
def update(parameters, m, v, step, batch):
    (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(parameters, batch)
    parameters, m, v, step, norm = adam_step(parameters, grads, m, v, step, lr=PPO['learning_rate'], max_norm=1.)
    return parameters, m, v, step, {**metrics, 'gradient_norm': norm}


def prepare_batch(rows):
    """Same-player GAE in NumPy; no compilation per variable episode length."""
    advantage = np.zeros(len(rows), np.float32)
    for i in range(len(rows)-1, -1, -1):
        r = rows[i]
        nxt = r['next_index']
        if nxt < -1 or (r['terminal'] and r['bootstrap'] != 0) or (nxt >= 0 and r['bootstrap'] != 0):
            raise ValueError('Invalid trajectory bootstrap')
        if nxt >= 0 and (nxt <= i or nxt >= len(rows) or rows[nxt]['trajectory'] != r['trajectory'] or r['terminal']):
            raise ValueError('Invalid same-player trajectory link')
        next_value = 0. if r['terminal'] else rows[nxt]['value'] if nxt >= 0 else r['bootstrap']
        delta = np.float32(r['reward']) + np.float32(next_value) - np.float32(r['value'])
        advantage[i] = delta + (np.float32(PPO['lambda'])*advantage[nxt] if nxt >= 0 else np.float32(0))
    returns = advantage + np.asarray([r['value'] for r in rows], np.float32)
    if not np.isfinite(returns).all() or np.any(returns < -1e-5) or np.any(returns > 1+1e-5):
        raise ValueError('Invalid bounded targets')
    return {'features':np.stack([r['features'] for r in rows]), 'mask':np.stack([r['mask'] for r in rows]),
        'actions':np.asarray([r['action'] for r in rows],np.int32),
        'old_logp':np.asarray([r['old_logp'] for r in rows],np.float32),
        'advantages':advantage, 'returns':returns}


def optimize(parameters, optimizer, rows, seed):
    if not rows:
        raise ValueError('Empty training cohort')
    batch = prepare_batch(rows)
    rng = np.random.default_rng(seed)
    m, v, step = optimizer
    metrics = []
    for epoch in range(PPO['epochs']):
        order = rng.permutation(len(rows))
        for first in range(0, len(rows), BATCH):
            ids = order[first:first+BATCH]
            part = {k: np.zeros((BATCH, *a.shape[1:]), dtype=a.dtype) for k, a in batch.items()}
            for k, a in batch.items():
                part[k][:len(ids)] = a[ids]
            part['valid'] = np.arange(BATCH) < len(ids)
            parameters, m, v, step, result = update(parameters, m, v, step, {k: jnp.asarray(a) for k,a in part.items()})
            record = {k: float(x) for k,x in result.items()}
            if not all(np.isfinite(x) for x in record.values()):
                raise ValueError('Nonfinite update')
            metrics.append(record)
    for state in [parameters, m, v]:
        finite_parameters(state)
    return parameters, (m, v, step), metrics
