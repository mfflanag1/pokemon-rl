"""Fixed supervised likelihood objective on the unchanged F1 network/law."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import forward, distribution, finite_parameters
from pokemon_fixed.math import adam_step
from .data import require


def validate_batch(batch):
    require(set(batch) == {'features','mask','targets'}, 'Unexpected model/loss input fields')
    x, mask, targets = [np.asarray(batch[k]) for k in ['features','mask','targets']]
    require(x.shape == (128,928) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid feature batch')
    require(mask.shape == (128,14) and mask.dtype == np.bool_ and mask.any(1).all(), 'Invalid mask batch')
    require(targets.shape == (128,) and targets.dtype == np.int32 and ((0 <= targets) & (targets < 14)).all(), 'Invalid target batch')
    require(mask[np.arange(128), targets].all(), 'Illegal target action')


def model_batch(split):
    batch = {k: split[k] for k in ['features','mask','targets']}
    validate_batch(batch)
    return {k:jnp.asarray(a) for k,a in batch.items()}


def objective(parameters, batch):
    logits, _ = forward(parameters, batch['features'])
    _, logs = distribution(logits, batch['mask'])
    return -jnp.mean(logs[jnp.arange(len(logits)), batch['targets']])


@jax.jit
def update(parameters, m, v, step, batch):
    loss, grads = jax.value_and_grad(objective)(parameters, batch)
    parameters, m, v, step, norm = adam_step(parameters, grads, m, v, step, lr=.0003, max_norm=1.)
    return parameters, (m,v,step), {'pre_update_nll':loss, 'gradient_norm':norm}


@jax.jit
def predictions(parameters, features, mask):
    logits, _ = forward(parameters, features)
    probabilities, logs = distribution(logits, mask)
    return probabilities, logs


def checked_step(parameters, optimizer, batch):
    parameters, optimizer, metrics = update(parameters, *optimizer, batch)
    record = {k:float(value) for k,value in metrics.items()}
    require(all(np.isfinite(x) for x in record.values()), 'Nonfinite supervised update')
    for state in [parameters,optimizer[0],optimizer[1]]:
        finite_parameters(state)
    return parameters,optimizer,record
