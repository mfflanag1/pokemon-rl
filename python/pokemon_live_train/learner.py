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
from pokemon_live import policy

ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / 'contracts/live-training-v1/plan.json').read_text())
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
    require(type(seed) is int and seed in PLAN['seeds'], 'Training seed')
    p = policy.initialize(seed); ids = jnp.arange(BATCH, dtype=jnp.uint32)
    return Session(p, jax.tree.map(jnp.zeros_like, p), jax.tree.map(jnp.zeros_like, p),
                   jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.uint32(BATCH), jnp.uint32(seed),
                   ids, engine.initial_kernel(teams(ids, seed)))


# These exact functions, their JIT caches and PPO configuration are inherited.
# New work here changes supported counters and checkpoint identity only.
from pokemon_live.learner import observe, collect, optimize, validate_buffer


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
