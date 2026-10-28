"""Unchanged frozen-target PPO/GAE/Adam rules on the F29 host with a static arm.

The objective, distribution, inclusive clip gradient, Adam step, cohort schedule and
batch validation are the sealed matchup ones; only `forward` gains the static `arm`.
"""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import distribution
from pokemon_fixed.math import adam_step
from pokemon_full_train.learning import PPO
from pokemon_matchup.learning import validate
from .network import forward, check_state, check_arm, NEW_SHAPES

ROWS, EPOCHS, MINIBATCH = 1024, 2, 256
FIELDS = {'features', 'mask', 'actions', 'old_logp', 'advantages', 'returns'}


def objective(p, b, arm):
    valid = b['valid']; x = jnp.where(valid[:, None], b['features'], 0.)
    mask = jnp.where(valid[:, None], b['mask'], True); actions = jnp.where(valid, b['actions'], 0)
    logits, values = forward(p, x, arm); probs, logs = distribution(logits, mask)
    old, adv, targets = [jax.lax.stop_gradient(jnp.where(valid, b[k], 0.)) for k in ['old_logp', 'advantages', 'returns']]
    lr = logs[jnp.arange(len(values)), actions] - old; ratio = jnp.exp(lr)
    active = jnp.where(adv >= 0, ratio <= 1 + PPO['clip'], ratio >= 1 - PPO['clip'])
    surrogate = jnp.where(active, ratio, jax.lax.stop_gradient(jnp.clip(ratio, 1 - PPO['clip'], 1 + PPO['clip']))) * adv
    def mean(a): return jnp.where(valid, a, 0.).sum() / jnp.maximum(valid.sum(), 1)
    actor = -mean(surrogate); value = .5 * mean((values - targets)**2); entropy = -mean(jnp.sum(probs * logs, 1))
    loss = actor + PPO['value_coefficient'] * value - PPO['entropy'] * entropy
    return loss, dict(loss=loss, actor_loss=actor, value_loss=value, entropy=entropy,
        approx_kl=mean(ratio - 1 - lr), clip_fraction=mean((jnp.abs(ratio - 1) > PPO['clip']).astype(jnp.float32)))


@partial(jax.jit, static_argnames=('arm',))
def kernel(p, m, v, step, b, arm):
    (_, metrics), g = jax.value_and_grad(objective, has_aux=True)(p, b, arm)
    p, m, v, step, norm = adam_step(p, g, m, v, step, lr=PPO['learning_rate'], max_norm=1.)
    return p, (m, v, step), {**metrics, 'gradient_norm': norm}


def update(p, opt, b, arm):
    check_arm(arm); check_state(p, opt)
    if int(opt[2]) >= 2**31 - 1: raise ValueError('Counter overflow')
    before = {k: np.asarray(p[k]).copy() for k in NEW_SHAPES} if arm == 'control' else {}
    q, qo, metrics = kernel(p, *opt, validate(b), arm); check_state(q, qo)
    if int(qo[2]) != int(opt[2]) + 1 or not all(np.isfinite(float(v)) for v in metrics.values()):
        raise ValueError('Update linkage/metrics')
    if any(not np.array_equal(before[k], np.asarray(q[k])) for k in before):
        raise ValueError('Control branch changed')
    return q, qo, {k: float(v) for k, v in metrics.items()}


def optimize_batch(p, opt, batch, seed, arm):
    check_arm(arm)
    if not isinstance(batch, dict) or set(batch) != FIELDS or len(batch['actions']) != ROWS:
        raise ValueError('Fixed cohort inventory')
    b = validate({**batch, 'valid': np.ones(ROWS, bool)})
    check_state(p, opt)
    if int(opt[2]) > 2**31 - 1 - EPOCHS * (ROWS // MINIBATCH): raise ValueError('Counter overflow')
    rng = np.random.default_rng(seed); records = []
    for _ in range(EPOCHS):
        order = rng.permutation(ROWS)
        for first in range(0, ROWS, MINIBATCH):
            ids = order[first:first + MINIBATCH]; part = {k: v[ids] for k, v in b.items()}
            p, opt, metrics = update(p, opt, part, arm); records.append(metrics)
    return p, opt, records
