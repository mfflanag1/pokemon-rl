"""Two-action policy with exact dyadic word selection, conditional on uniform words.

The rare FP32 mass and its real complement define the law. Logs use log/log1p
of that same mass; rounded displayed probabilities never define the sampler.
Only this binary profile is admitted. The integer sampler is not differentiated.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax_learner.model import (ActorInput, forward, validate_input,
                                       validate_parameters)
from pokemon_learner.core import require

FLOOR = np.float32(2**-24)
DOMAIN = 0x44594231


def parts(logits, mask):
    both = jnp.all(mask, axis=1)
    # Forced/padded rows must have no logit gradient, including their difference.
    safe = jnp.where(both[:, None], logits, jnp.float32(0))
    delta = safe[:, 0] - safe[:, 1]
    rare_action = jnp.where(delta >= 0, 1, 0).astype(jnp.int32)
    tail = jnp.exp(-jnp.abs(delta))
    raw = tail / (jnp.float32(1) + tail)
    rare = jnp.where(raw > FLOOR, raw, FLOOR)  # zero derivative at the floor
    is_rare = jnp.arange(2)[None, :] == rare_action[:, None]
    p = jnp.where(is_rare, rare[:, None], jnp.float32(1) - rare[:, None])
    logs = jnp.where(is_rare, jnp.log(rare)[:, None], jnp.log1p(-rare)[:, None])
    # Forced legal action: exact one/zero; terminal: exact all-zero transport.
    return (jnp.where(both[:, None], p, mask.astype(jnp.float32)),
            jnp.where(both[:, None], logs, jnp.float32(0)), rare_action, rare)


def distribution_kernel(logits, mask):
    return parts(logits, mask)[:2]


def threshold_words(rare):
    """Exact high/low words of rare*2^64 for normal FP32 rare in [2^-24,1/2].

    rare = m * 2^(e-150), m a 24-bit significand and biased exponent e.
    The shift e-86 lies in [17,40], so two uint32 words suffice without x64.
    """
    bits = jax.lax.bitcast_convert_type(rare, jnp.uint32)
    mantissa = (bits & jnp.uint32(0x7fffff)) | jnp.uint32(0x800000)
    shift = ((bits >> jnp.uint32(23)) & jnp.uint32(255)).astype(jnp.int32) - 86
    high = jnp.where(shift < 32,
        mantissa >> jnp.maximum(32 - shift, 0).astype(jnp.uint32),
        mantissa << jnp.maximum(shift - 32, 0).astype(jnp.uint32))
    low = jnp.where(shift < 32, mantissa << jnp.minimum(shift, 31).astype(jnp.uint32), jnp.uint32(0))
    return jnp.stack((high, low), axis=-1)


def select_words(rare, rare_action, mask, words):
    """Pure injected-word selector used identically by tests and keyed sampling."""
    threshold = threshold_words(rare)
    choose_rare = ((words[:, 0] < threshold[:, 0]) |
                   ((words[:, 0] == threshold[:, 0]) & (words[:, 1] < threshold[:, 1])))
    selected = jnp.where(choose_rare, rare_action, 1 - rare_action)
    live = jnp.any(mask, axis=1)
    return jnp.where(live, jnp.where(jnp.all(mask, axis=1), selected,
                                   jnp.argmax(mask, axis=1)), -1).astype(jnp.int32)


@jax.jit
def predict_kernel(parameters, obs):
    logits, values = forward(parameters, obs)
    probabilities, logs = distribution_kernel(logits, obs.mask)
    return probabilities, logs, jnp.where(jnp.any(obs.mask, axis=1), values, 0)


@jax.jit
def action_kernel(parameters, obs, keys):
    logits, values = forward(parameters, obs)
    _, logs, rare_action, rare = parts(logits, obs.mask)
    words = jax.vmap(lambda key: jax.random.bits(jax.random.fold_in(key, DOMAIN),
                                               (2,), dtype=jnp.uint32))(keys)
    actions = select_words(rare, rare_action, obs.mask, words)
    live = jnp.any(obs.mask, axis=1)
    selected = logs[jnp.arange(len(actions)), jnp.maximum(actions, 0)]
    return actions, jnp.where(live, selected, 0), jnp.where(live, values, 0)


def act(parameters, obs, keys):
    require(jax.default_backend() == 'cpu' and not jax.config.x64_enabled, 'CPU FP32 only')
    n = validate_input(obs)
    validate_parameters(parameters)
    require(keys.shape == (n,) and str(keys.dtype) == 'key<fry>', 'One Threefry key per row')
    result = action_kernel(parameters, obs, keys)
    actions, mask = np.asarray(result[0]), np.asarray(obs.mask)
    live = mask.any(axis=1)
    require(np.all(actions[~live] == -1) and np.all(mask[np.arange(n)[live], actions[live]]), 'Illegal action')
    require(all(np.all(np.isfinite(x)) for x in result), 'Nonfinite actor output')
    return result
