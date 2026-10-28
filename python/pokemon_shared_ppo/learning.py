"""Architecture-dispatched PPO; host validation surrounds pure loss kernels."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pokemon_fixed.math import adam_step
from pokemon_full.policy import distribution
from pokemon_full_train import learning as flat
from pokemon_shared import network
from pokemon_shared.learning import check_state, shapes
from pokemon_transfer.learning import _validate_ppo


def objective(p, batch, architecture):
    """Pure, differentiable masked PPO loss with frozen rollout targets."""
    shapes(architecture)
    if architecture == 'flat':
        return flat.objective(p, batch)
    valid = batch['valid']
    features = jnp.where(valid[:, None], batch['features'], 0.)
    mask = jnp.where(valid[:, None], batch['mask'], True)
    actions = jnp.where(valid, batch['actions'], 0)
    logits, values = network.forward(p, features)
    probabilities, logs = distribution(logits, mask)
    old, advantages, targets = [jax.lax.stop_gradient(jnp.where(valid, batch[k], 0.))
                                for k in ['old_logp', 'advantages', 'returns']]
    log_ratio = logs[jnp.arange(len(values)), actions] - old
    ratio = jnp.exp(log_ratio)
    active = jnp.where(advantages >= 0, ratio <= 1.2, ratio >= .8)
    surrogate = jnp.where(active, ratio,
        jax.lax.stop_gradient(jnp.clip(ratio, .8, 1.2))) * advantages

    def mean(a):
        return jnp.where(valid, a, 0.).sum() / jnp.maximum(valid.sum(), 1)

    actor = -mean(surrogate)
    value = .5 * mean((values - targets)**2)
    entropy = -mean(jnp.sum(probabilities * logs, 1))
    loss = actor + .5 * value - .01 * entropy
    return loss, dict(loss=loss, actor_loss=actor, value_loss=value,
                      entropy=entropy, approx_kl=mean(ratio - 1 - log_ratio))


def _state(p, opt, architecture, steps):
    shapes(architecture)
    if not isinstance(opt, (tuple, list)) or len(opt) != 3:
        raise ValueError('Invalid optimizer state')
    if any(not isinstance(group, dict) for group in (p, opt[0], opt[1])):
        raise ValueError('Invalid parameter/moment mapping')
    check_state(p, opt, architecture)
    if int(np.asarray(opt[2])) > np.iinfo(np.int32).max - steps:
        raise ValueError('Optimizer counter would overflow')


def _minibatch(batch):
    """Validate live rows; padding may contain arbitrary transport payloads."""
    fields = {'features', 'mask', 'actions', 'old_logp', 'advantages', 'returns', 'valid'}
    if not isinstance(batch, dict) or set(batch) != fields:
        raise ValueError('Unexpected PPO input fields')
    b = {k: np.asarray(a) for k, a in batch.items()}
    valid = b['valid']
    if valid.ndim != 1 or valid.dtype != np.bool_ or not len(valid):
        raise ValueError('Invalid PPO valid mask')
    n = len(valid)
    for k, a in b.items():
        shape = (n, 928) if k == 'features' else (n, 14) if k == 'mask' else (n,)
        dtype = np.bool_ if k in ('valid', 'mask') else np.int32 if k == 'actions' else np.float32
        if a.shape != shape or a.dtype != dtype or not np.isfinite(a[valid]).all():
            raise ValueError('Invalid PPO ' + k)
    actions, mask = b['actions'][valid], b['mask'][valid]
    if (not mask.any(1).all() or not ((actions >= 0) & (actions < 14)).all()
            or not mask[np.arange(len(actions)), actions].all()):
        raise ValueError('Invalid PPO mask/actions')
    if (b['old_logp'][valid] > 0).any() or ((b['returns'][valid] < -1e-5)
            | (b['returns'][valid] > 1 + 1e-5)).any():
        raise ValueError('Invalid PPO targets/log probability')
    return {k: jnp.asarray(a) for k, a in b.items()}


@partial(jax.jit, static_argnames=('architecture',))
def _update(p, m, v, step, batch, architecture):
    if architecture == 'flat':
        return flat.update(p, m, v, step, batch)
    (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(p, batch, architecture)
    p, m, v, step, norm = adam_step(p, grads, m, v, step, lr=.0003, max_norm=1.)
    return p, m, v, step, {**metrics, 'gradient_norm': norm}


def update(p, m, v, step, batch, architecture):
    """One checked Adam step (including the sealed empty-row loss semantics)."""
    _state(p, (m, v, step), architecture, 1)
    batch = _minibatch(batch)
    result = _update(p, m, v, step, batch, architecture)
    q, a, b, counter, metrics = result
    _state(q, (a, b, counter), architecture, 0)
    if int(counter) != int(step) + 1 or not all(np.isfinite(float(x)) for x in metrics.values()):
        raise ValueError('Invalid PPO update')
    return result


def optimize_batch(p, opt, batch, seed, architecture):
    """Two independently shuffled epochs over a validated 1024-row cohort."""
    _state(p, opt, architecture, 8)
    if not isinstance(batch, dict):
        raise ValueError('Invalid PPO batch mapping')
    arrays = _validate_ppo(batch)
    rng = np.random.default_rng(seed)
    m, v, step = opt
    records = []
    for _ in range(2):
        order = rng.permutation(1024)
        for first in range(0, 1024, 256):
            ids = order[first:first + 256]
            part = {k: jnp.asarray(a[ids]) for k, a in arrays.items()}
            part['valid'] = jnp.ones(256, jnp.bool_)
            p, m, v, step, metrics = update(p, m, v, step, part, architecture)
            records.append({k: float(x) for k, x in metrics.items()})
    return p, (m, v, step), records
