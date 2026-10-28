"""Qualified whole-trajectory PPO arithmetic with packed structured-input checks."""
import jax
import jax.numpy as jnp
import numpy as np

from pokemon_full.policy import distribution
from pokemon_full_train.learning import PPO
from pokemon_fixed.math import adam_step
from pokemon_shared_ppo.learning import _minibatch
from .features import CATEGORICAL_SIZES, FIELD_SHAPES, FIELD_SLICES as SLICES, FEATURES
from .network import forward, check_state, EVENT_FIELDS

ROWS, EPOCHS, MINIBATCH = 1024, 2, 256
FIELDS = {'features', 'mask', 'actions', 'old_logp', 'advantages', 'returns'}
CATEGORY_FIELDS = {'species': 'species', 'ability': 'ability', 'item': 'item',
                   'types': 'type', 'moves': 'move', 'status': 'status',
                   'action_move': 'move', 'action_type': 'type'}


def validate_features(features, valid):
    x, live = np.asarray(features), np.asarray(valid)
    if live.dtype != np.bool_ or live.ndim != 1 or not len(live):
        raise ValueError('Structured feature valid mask')
    if x.shape != (len(live), FEATURES) or x.dtype != np.float32:
        raise ValueError('Structured feature shape/dtype')
    if not np.isfinite(x[live]).all():
        raise ValueError('Nonfinite structured feature')
    def field(name):
        return x[live, SLICES[name]].reshape((-1, *FIELD_SHAPES[name]))
    def ids(values, domain):
        if np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values >= CATEGORICAL_SIZES[domain]):
            raise ValueError('Structured categorical ID: ' + domain)
    for name, domain in CATEGORY_FIELDS.items():
        ids(field(name), domain)
    event_ids = field('event_ids')
    for i, domain in enumerate(EVENT_FIELDS):
        ids(event_ids[:, :, i], domain)
    if np.any((field('event_mask') != 0) & (field('event_mask') != 1)):
        raise ValueError('Structured event mask')
    if np.any(field('public_move_counts') < 0):
        raise ValueError('Negative public move counts')
    return x


def validate(batch):
    if not isinstance(batch, dict) or set(batch) != FIELDS | {'valid'}:
        raise ValueError('Structured PPO input inventory')
    x = validate_features(batch['features'], batch['valid'])
    # Preserve the predecessor's action, target, dtype and padding admission.
    _minibatch({**batch, 'features': x[:, :928]})
    return {k: jnp.asarray(v) for k, v in batch.items()}


def objective(parameters, batch):
    valid = batch['valid']
    x = jnp.where(valid[:, None], batch['features'], 0.)
    mask = jnp.where(valid[:, None], batch['mask'], True)
    actions = jnp.where(valid, batch['actions'], 0)
    logits, values = forward(parameters, x)
    probs, logs = distribution(logits, mask)
    old, advantages, targets = [jax.lax.stop_gradient(jnp.where(valid, batch[k], 0.))
                                for k in ['old_logp', 'advantages', 'returns']]
    log_ratio = logs[jnp.arange(len(values)), actions] - old
    ratio = jnp.exp(log_ratio)
    active = jnp.where(advantages >= 0, ratio <= 1 + PPO['clip'], ratio >= 1 - PPO['clip'])
    surrogate = jnp.where(active, ratio,
        jax.lax.stop_gradient(jnp.clip(ratio, 1 - PPO['clip'], 1 + PPO['clip']))) * advantages
    def mean(a):
        return jnp.where(valid, a, 0.).sum() / jnp.maximum(valid.sum(), 1)
    actor = -mean(surrogate)
    value = .5 * mean((values - targets)**2)
    entropy = -mean(jnp.sum(probs * logs, 1))
    loss = actor + PPO['value_coefficient'] * value - PPO['entropy'] * entropy
    return loss, dict(loss=loss, actor_loss=actor, value_loss=value, entropy=entropy,
                     approx_kl=mean(ratio - 1 - log_ratio),
                     clip_fraction=mean((jnp.abs(ratio - 1) > PPO['clip']).astype(jnp.float32)))


@jax.jit
def kernel(parameters, m, v, step, batch):
    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(parameters, batch)
    parameters, m, v, step, norm = adam_step(parameters, gradients, m, v, step,
                                           lr=PPO['learning_rate'], max_norm=1.)
    return parameters, (m, v, step), {**metrics, 'gradient_norm': norm}


def update(parameters, optimizer, batch):
    check_state(parameters, optimizer)
    b = validate(batch)
    if not np.asarray(batch['valid']).any():
        return parameters, optimizer, {k: 0. for k in
            ['loss', 'actor_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction', 'gradient_norm']}
    if int(optimizer[2]) >= 2**31 - 1:
        raise ValueError('Optimizer counter overflow')
    q, qo, metrics = kernel(parameters, *optimizer, b)
    check_state(q, qo)
    if int(qo[2]) != int(optimizer[2]) + 1 or not all(np.isfinite(float(v)) for v in metrics.values()):
        raise ValueError('Structured update linkage/metrics')
    return q, qo, {k: float(v) for k, v in metrics.items()}


def optimize_batch(parameters, optimizer, batch, seed):
    if not isinstance(batch, dict) or set(batch) != FIELDS or len(batch['actions']) != ROWS:
        raise ValueError('Fixed structured PPO cohort inventory')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise ValueError('Unsigned optimizer shuffle seed required')
    b = validate({**batch, 'valid': np.ones(ROWS, bool)})
    check_state(parameters, optimizer)
    if int(optimizer[2]) > 2**31 - 1 - EPOCHS * (ROWS // MINIBATCH):
        raise ValueError('Optimizer counter overflow')
    rng = np.random.default_rng(int(seed))
    records = []
    for _ in range(EPOCHS):
        order = rng.permutation(ROWS)
        for first in range(0, ROWS, MINIBATCH):
            ids = order[first:first + MINIBATCH]
            parameters, optimizer, metrics = update(parameters, optimizer, {k: v[ids] for k, v in b.items()})
            records.append(metrics)
    return parameters, optimizer, records
