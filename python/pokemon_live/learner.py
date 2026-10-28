"""Fresh-request windows continue the same cohort; PPO traces stop at each window."""
import json
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_continuation import engine, carry, chance
from pokemon_fixed.math import adam_step
from pokemon_learner.core import linked_gae
from pokemon_switch_engine import learner as predecessor
from pokemon_switch_engine.scalar import require
from . import policy

ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / 'contracts/live-learner-v1/plan.json').read_text())
BATCH, GROUPS, LANES = PLAN['batch_games'], PLAN['groups_per_window'], PLAN['batch_games'] * 2
Buffer = predecessor.Buffer


class Session(NamedTuple):
    parameters: dict
    m: dict
    v: dict
    step: jax.Array
    windows: jax.Array
    updates: jax.Array
    next_game: jax.Array
    seed: jax.Array
    ids: jax.Array
    environment: engine.State


def teams(ids, seed):
    return jnp.stack([(ids + seed % 24) % 24, (ids // 24 + (seed // 24) % 24) % 24], 1).astype(jnp.int32)


def start(seed):
    require(type(seed) is int and seed in PLAN['seeds'], 'Canary seed')
    p = policy.initialize(seed); ids = jnp.arange(BATCH, dtype=jnp.uint32)
    return Session(p, jax.tree.map(jnp.zeros_like, p), jax.tree.map(jnp.zeros_like, p),
                   jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.uint32(BATCH), jnp.uint32(seed),
                   ids, engine.initial_kernel(teams(ids, seed)))


def observe(state):
    views = [engine.view_kernel(state, side) for side in (0, 1)]
    return policy.encode(engine.PlayerView(*[jnp.stack([views[0][k], views[1][k]], 1).reshape(
        (-1, *views[0][k].shape[1:])) for k in range(6)]))


@jax.jit
def collect(parameters, state, ids, seed, boundary):
    master = jax.random.key(seed); actor_master = jax.random.fold_in(master, 0x41435436)

    def one(state, _):
        obs = observe(state); live = jnp.repeat(~carry.stopped(state, boundary), 2)
        mask = obs.mask & live[:, None]
        obs = policy.ActorInput(jnp.where(mask.any(1)[:, None], obs.features, 0), mask)
        def keys(game, group):
            base = jax.random.fold_in(jax.random.fold_in(actor_master, game), group)
            return jnp.stack([jax.random.fold_in(base, side) for side in (0, 1)])
        keys_ = jax.vmap(keys)(ids, state.group.astype(jnp.uint32)).reshape(-1)
        actions, logp, values = policy.act(parameters, obs, keys_)
        after, _ = carry.step_kernel(state, actions.reshape(BATCH, 2),
            chance.sample(master, ids, state.group.astype(jnp.uint32)), boundary)
        return after, (obs.features, obs.mask, actions, logp, values, obs.mask.any(1))

    final, rows = jax.lax.scan(one, state, None, length=GROUPS)
    _, bootstrap = policy.forward(parameters, observe(final))
    bootstrap = jnp.where(jnp.repeat(final.winner == -1, 2), bootstrap, 0)
    # A cohort is never refilled within a scan. Fresh joint boundaries ensure both
    # players act in any window that later terminates. Already-ended lanes are inert.
    target = predecessor.targets(rows[4], rows[5], final.winner, bootstrap)
    flatten = lambda x: x.reshape((-1, *x.shape[2:]))
    return Buffer(*[flatten(x) for x in (*rows, *target)]), final


@jax.jit
def optimize(session, buffer):
    def epoch(carry_, _):
        p, m, v, step = carry_
        (_, metrics), grad = jax.value_and_grad(policy.objective, has_aux=True)(p, buffer)
        p, m, v, step, norm = adam_step(p, grad, m, v, step, lr=PLAN['learning_rate'])
        return (p, m, v, step), {**metrics, 'gradient_norm': norm}
    return jax.lax.scan(epoch, (session.parameters, session.m, session.v, session.step), None, length=PLAN['epochs'])


def validate_buffer(buffer, before, final, boundary, parameters):
    require(type(buffer) is Buffer, 'Buffer schema'); engine.validate_state(final)
    n = GROUPS * LANES
    for key, value in buffer._asdict().items():
        shape = (n, policy.FEATURES) if key == 'features' else (n, 4) if key == 'mask' else (n,)
        dtype = np.bool_ if key in ('mask', 'valid', 'terminal') else np.int32 if key in ('actions', 'next_index') else np.float32
        require(value.shape == shape and np.asarray(value).dtype == dtype and np.isfinite(value).all(), 'Buffer schema/finite: ' + key)
    valid, mask, actions = map(np.asarray, (buffer.valid, buffer.mask, buffer.actions))
    require(np.array_equal(valid, mask.any(1)), 'Mask/valid linkage')
    require(np.all(actions[~valid] == -1) and np.all((actions[valid] >= 0) & (actions[valid] < 4)), 'Action bounds')
    require(np.all(mask[np.flatnonzero(valid), actions[valid]]), 'Known-forbidden action')
    require(np.all(carry.stopped(final, boundary)) and np.all(np.asarray(final.turn) <= boundary), 'Unfinished window')
    previously_done = np.asarray(before.winner) != -1
    for old, new in zip(before, final):
        require(np.array_equal(np.asarray(old)[previously_done], np.asarray(new)[previously_done]), 'Terminal state drift')
    selected = np.flatnonzero(valid); inverse = np.full(n, -1); inverse[selected] = np.arange(len(selected))
    nxt = np.asarray(buffer.next_index)[valid]
    require(np.all((nxt == -1) | ((nxt > selected) & (nxt < n) & (nxt % LANES == selected % LANES))), 'Same-player forward links')
    require(np.all(inverse[nxt[nxt >= 0]] >= 0), 'Link to padding')
    mapped = np.where(nxt >= 0, inverse[np.maximum(nxt, 0)], -1)
    if len(selected):
        advantages, returns = linked_gae(np.asarray(buffer.rewards)[valid], np.asarray(buffer.values)[valid], mapped,
            np.asarray(buffer.bootstrap)[valid], np.asarray(buffer.terminal)[valid], selected % LANES,
            gamma=PLAN['gamma'], lam=PLAN['lambda'])
        require(np.allclose(advantages, np.asarray(buffer.advantages)[valid], atol=2e-6, rtol=2e-6) and
                np.allclose(returns, np.asarray(buffer.returns)[valid], atol=2e-6, rtol=2e-6), 'Independent GAE parity')
    reward, terminal = map(np.asarray, (buffer.rewards, buffer.terminal))
    new_end = (np.asarray(final.winner) != -1) & ~previously_done
    require(np.sum(reward) == new_end.sum() and terminal.sum() == 2 * new_end.sum(), 'New terminal accounting')
    for lane in range(LANES):
        ix = selected[selected % LANES == lane]; game, seat = divmod(lane, 2)
        require(len(ix) > 0 if new_end[game] else True, 'Missing terminal decision')
        if len(ix):
            require(np.array_equal(np.asarray(buffer.next_index)[ix], np.r_[ix[1:], -1]), 'Immediate next same-player decision')
        expected = np.zeros(GROUPS, bool)
        if new_end[game]: expected[ix[-1] // LANES] = True
        require(np.array_equal(terminal[lane::LANES], expected), 'Terminal placement')
        require(np.array_equal(reward[lane::LANES], expected.astype(np.float32) * (int(final.winner[game]) == seat)), 'Reward seat')
    _, boot = policy.forward(parameters, observe(final))
    boot = np.asarray(boot)
    expected_boot = np.zeros(n, np.float32)
    for lane in range(LANES):
        ix = selected[selected % LANES == lane]
        if len(ix) and int(final.winner[lane // 2]) == -1: expected_boot[ix[-1]] = boot[lane]
    require(np.array_equal(np.asarray(buffer.bootstrap), expected_boot), 'Frozen collecting-policy bootstrap')
    for name in ('features', 'old_logp', 'values', 'rewards', 'bootstrap', 'advantages', 'returns'):
        require(np.all(np.asarray(getattr(buffer, name))[~valid] == 0), 'Nonzero padding: ' + name)
    require(np.all(np.asarray(buffer.next_index)[~valid] == -1) and not terminal[~valid].any(), 'Padding metadata')
    for name in ('values', 'bootstrap', 'returns'):
        a = np.asarray(getattr(buffer, name)); tol = 2e-6 if name == 'returns' else 0
        require(np.all((a >= -tol) & (a <= 1 + tol)), 'Bounded ' + name)


def transaction(session, seed, *, fault=None, audit_only=False):
    from .checkpoint import validate
    validate(session)
    require(type(seed) is int and seed == int(session.seed), 'Master seed binding')
    require(type(audit_only) is bool, 'Audit mode')
    require(int(session.windows) < PLAN['windows_per_seed'] or audit_only and int(session.windows) == PLAN['windows_per_seed'], 'Production budget; endpoint requires audit_only')
    window = int(session.windows) % 5; boundary = PLAN['boundaries'][window]
    ids, before, next_game = session.ids, session.environment, session.next_game
    if window == 0 and int(session.windows) > 0:
        ids = next_game + jnp.arange(BATCH, dtype=jnp.uint32)
        before = engine.initial_kernel(teams(ids, session.seed)); next_game = next_game + BATCH
    buffer, final = collect(session.parameters, before, ids, session.seed, jnp.int32(boundary))
    jax.block_until_ready((buffer, final)); validate_buffer(buffer, before, final, boundary, session.parameters)
    if fault == 'collect': raise RuntimeError('Injected collection rollback')
    active = bool(np.asarray(buffer.valid).any())
    if active:
        (p, m, v, step), metrics = optimize(session, buffer)
    else:
        p, m, v, step = session[:4]
        metrics = {k: jnp.zeros(PLAN['epochs'], jnp.float32) for k in (
            'loss', 'actor_loss', 'value_loss', 'entropy', 'clip_fraction', 'approx_kl', 'gradient_norm')}
    candidate = Session(p, m, v, step, session.windows + 1, session.updates + int(active), next_game,
                        session.seed, ids, final)
    jax.block_until_ready((candidate, metrics)); validate(candidate, audit_only=audit_only)
    require(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(metrics)), 'Finite metrics')
    if fault == 'update': raise RuntimeError('Injected update rollback')
    return candidate, buffer, final, metrics
