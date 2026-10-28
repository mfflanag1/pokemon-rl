"""Fixed-budget bounded-value canary on the existing capped M3 simulator.

This additive collector changes only the value-head calls. It intentionally still
resets unfinished games; episode continuation requires its own mechanics contract.
"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_fixed.math import adam_step
from pokemon_switch_engine import engine, chance, learner as original
from pokemon_switch_engine.scalar import require
from . import policy

ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / 'contracts/bounded-value-v1/plan.json').read_text())
CONFIG = original.CONFIG
BATCH, GROUPS = original.BATCH, original.GROUPS
Learner, Buffer = original.Learner, original.Buffer


def start(seed):
    return original.start(seed)


@jax.jit
def collect(parameters, next_game, seed):
    ids = next_game + jnp.arange(BATCH, dtype=jnp.uint32)
    teams = jnp.stack([(ids + seed % 24) % 24, (ids // 24 + (seed // 24) % 24) % 24], axis=1).astype(jnp.int32)
    state = engine.initial_kernel(teams)
    master = jax.random.key(seed)
    actor_master = jax.random.fold_in(master, 0x41435434)

    def one(state, _):
        obs = original.observe(state)
        live = jnp.repeat((state.winner == -1) & (state.turn <= 12), 2)
        obs = policy.ActorInput(jnp.where(live[:, None], obs.features, 0), obs.mask & live[:, None])
        valid = obs.mask.any(1)

        def keys(game, group):
            base = jax.random.fold_in(jax.random.fold_in(actor_master, game), group)
            return jnp.stack([jax.random.fold_in(base, side) for side in (0, 1)])

        actor_keys = jax.vmap(keys)(ids, state.group.astype(jnp.uint32)).reshape(-1)
        actions, logp, values = policy.act(parameters, obs, actor_keys)
        after, _ = engine.step_kernel(state, actions.reshape(BATCH, 2),
                                      chance.sample(master, ids, state.group.astype(jnp.uint32)))
        return after, (obs.features, obs.mask, actions, logp, values, valid)

    final, rows = jax.lax.scan(one, state, xs=None, length=GROUPS)
    _, bootstrap = policy.forward(parameters, original.observe(final))
    bootstrap = jnp.where(jnp.repeat(final.winner == -1, 2), bootstrap, 0)
    target = original.targets(rows[4], rows[5], final.winner, bootstrap)
    flatten = lambda x: x.reshape((-1, *x.shape[2:]))
    return Buffer(*[flatten(x) for x in (*rows, *target)]), final


@jax.jit
def update(state, buffer):
    def epoch(carry, _):
        p, m, v, step = carry
        (_, metrics), gradient = jax.value_and_grad(policy.objective, has_aux=True)(p, buffer)
        p, m, v, step, norm = adam_step(p, gradient, m, v, step, lr=CONFIG['learning_rate'])
        return (p, m, v, step), {**metrics, 'gradient_norm': norm}

    (p, m, v, step), metrics = jax.lax.scan(epoch, (state.parameters, state.m, state.v, state.step),
                                          xs=None, length=CONFIG['epochs'])
    return Learner(p, m, v, step, state.version + 1, state.next_game + BATCH), metrics


def validate_buffer(buffer, final):
    original.validate_buffer(buffer, final)
    for name in ('values', 'bootstrap'):
        values = np.asarray(getattr(buffer, name))
        require(np.all((values >= 0) & (values <= 1)), 'Bounded ' + name)
    returns = np.asarray(buffer.returns)
    tolerance = PLAN['acceptance']['target_range_tolerance']
    require(np.all((returns >= -tolerance) & (returns <= 1 + tolerance)), 'Bounded targets within FP32 tolerance')


def transaction(state, seed):
    from .checkpoint import validate
    validate(state)
    require(type(seed) is int and seed in PLAN['seeds'], 'Canary seed')
    require(int(state.version) <= PLAN['updates_per_seed'], 'Canary budget including one endpoint replay')
    buffer, final = collect(state.parameters, state.next_game, jnp.uint32(seed))
    jax.block_until_ready((buffer, final)); validate_buffer(buffer, final)
    candidate, metrics = update(state, buffer)
    jax.block_until_ready((candidate, metrics)); validate(candidate)
    require(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(metrics)), 'Finite learner metrics')
    return candidate, buffer, final, metrics
