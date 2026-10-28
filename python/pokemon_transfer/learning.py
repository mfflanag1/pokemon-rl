"""F9: one clipped Adam step for sealed PPO plus supervised rehearsal.

The host boundary accepts already selected, complete-GAE rows. It neither
recomputes nor normalizes advantages or returns. Pure kernels trust that boundary.
"""
import jax
import jax.numpy as jnp
import numpy as np

from pokemon_fixed.math import adam_step
from pokemon_full.policy import finite_parameters
from pokemon_full_train import learning as ppo
from pokemon_fit import learning as fit


def objective(parameters, ppo_batch, supervised_batch, coefficient):
    loss, metrics = ppo.objective(parameters, ppo_batch)
    nll = fit.objective(parameters, supervised_batch)
    total = loss + coefficient * nll
    return total, {**metrics, 'total_loss': total, 'rehearsal_nll': nll}


@jax.jit
def update(parameters, m, v, step, ppo_batch, supervised_batch, coefficient):
    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
        parameters, ppo_batch, supervised_batch, coefficient)
    parameters, m, v, step, norm = adam_step(
        parameters, gradients, m, v, step,
        lr=ppo.PPO['learning_rate'], max_norm=1.)
    return parameters, m, v, step, {**metrics, 'gradient_norm': norm}


def _validate_ppo(batch):
    shapes = {'features': (1024, 928), 'mask': (1024, 14),
              'actions': (1024,), 'old_logp': (1024,),
              'advantages': (1024,), 'returns': (1024,)}
    if set(batch) != set(shapes):
        raise ValueError('Unexpected PPO input fields')
    arrays = {k: np.asarray(a) for k, a in batch.items()}
    for key, shape in shapes.items():
        a = arrays[key]
        dtype = np.bool_ if key == 'mask' else np.int32 if key == 'actions' else np.float32
        if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
            raise ValueError(f'Invalid PPO {key}')
    mask, actions = arrays['mask'], arrays['actions']
    if not mask.any(1).all() or not ((actions >= 0) & (actions < 14)).all():
        raise ValueError('Invalid PPO mask/actions')
    if not mask[np.arange(1024), actions].all():
        raise ValueError('Illegal PPO action')
    if (arrays['old_logp'] > 0).any():
        raise ValueError('Invalid PPO log probability')
    targets = arrays['returns']
    if ((targets < -1e-5) | (targets > 1 + 1e-5)).any():
        raise ValueError('Invalid bounded PPO returns')
    return arrays


def _validate_state(parameters, optimizer):
    shapes = {'w': (928, 64), 'b': (64,), 'pw': (64, 14),
              'pb': (14,), 'vw': (64,), 'vb': ()}
    if not isinstance(optimizer, (tuple, list)) or len(optimizer) != 3:
        raise ValueError('Invalid optimizer state')
    for state in (parameters, optimizer[0], optimizer[1]):
        if set(state) != set(shapes):
            raise ValueError('Unexpected parameter/state fields')
        for key, shape in shapes.items():
            a = np.asarray(state[key])
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f'Invalid parameter/state {key}')
    if any((np.asarray(a) < 0).any() for a in optimizer[1].values()):
        raise ValueError('Negative Adam second moment')
    step = np.asarray(optimizer[2])
    if step.shape != () or step.dtype != np.int32 or not 0 <= step < np.iinfo(np.int32).max - 8:
        raise ValueError('Invalid optimizer step')


def optimize_batch(parameters, optimizer, batch, supervised_batch, seed, coefficient):
    """Validate once, then shuffle 1024 rows twice into eight full minibatches."""
    weight = np.asarray(coefficient)
    if (weight.shape != () or weight.dtype.kind not in 'fiu' or
            not np.isfinite(weight) or weight < 0 or weight > np.finfo(np.float32).max):
        raise ValueError('Coefficient must be finite and nonnegative FP32')
    arrays = _validate_ppo(batch)
    fit.validate_batch(supervised_batch)
    _validate_state(parameters, optimizer)
    supervised = {k: jnp.asarray(a) for k, a in supervised_batch.items()}
    rng = np.random.default_rng(seed)
    m, v, step = optimizer
    metrics = []
    for _ in range(2):
        order = rng.permutation(1024)
        for first in range(0, 1024, 256):
            ids = order[first:first + 256]
            part = {k: jnp.asarray(a[ids]) for k, a in arrays.items()}
            part['valid'] = jnp.ones(256, dtype=jnp.bool_)
            parameters, m, v, step, result = update(
                parameters, m, v, step, part, supervised, np.float32(weight))
            record = {k: float(a) for k, a in result.items()}
            if not all(np.isfinite(a) for a in record.values()):
                raise ValueError('Nonfinite joint update')
            for state in (parameters, m, v):
                finite_parameters(state)
            metrics.append(record)
    return parameters, (m, v, step), metrics
