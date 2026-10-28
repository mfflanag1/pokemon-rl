"""14 actions embedded in a 16-leaf dyadic conditional tree; bounded value head.

Every binary conditional reuses the sealed FP32 rare-mass law. Selection and
stored log probabilities use the same path; the two padded leaves stay masked.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_binary.policy import parts as binary_parts, select_words
from pokemon_fixed.math import adam_step
from .features import FEATURES


def _group(logits, mask):
    live = mask.any(1)
    maximum = jax.lax.stop_gradient(jnp.max(jnp.where(mask, logits, -jnp.inf), 1))
    maximum = jnp.where(live, maximum, 0)
    mass = jnp.sum(jnp.where(mask, jnp.exp(jnp.where(mask, logits-maximum[:, None], 0)), 0), 1)
    return maximum + jnp.log(jnp.where(live, mass, 1))


def nodes(logits, mask):
    logits = jnp.pad(logits, ((0, 0), (0, 2)))
    mask = jnp.pad(mask, ((0, 0), (0, 2)))
    inputs, masks = [], []
    for level in range(4):
        width = 16 >> level
        for group in range(1 << level):
            left, middle, right = group*width, group*width + width//2, (group+1)*width
            inputs.append(jnp.stack([_group(logits[:, left:middle], mask[:, left:middle]),
                                     _group(logits[:, middle:right], mask[:, middle:right])], 1))
            masks.append(jnp.stack([mask[:, left:middle].any(1), mask[:, middle:right].any(1)], 1))
    inputs, masks = jnp.stack(inputs, 1), jnp.stack(masks, 1)
    p, logs, rare_action, rare = binary_parts(inputs.reshape(-1, 2), masks.reshape(-1, 2))
    return p.reshape(-1, 15, 2), logs.reshape(-1, 15, 2), rare_action, rare, masks


def distribution(logits, mask):
    p, logs, *_ = nodes(logits, mask)
    probabilities, logps = [], []
    for action in range(14):
        node, probability, logp = 0, jnp.ones(logits.shape[0]), jnp.zeros(logits.shape[0])
        for level in range(4):
            bit = (action >> (3-level)) & 1
            probability = probability * p[:, node, bit]
            logp = logp + logs[:, node, bit]
            node = 2*node + 1 + bit
        probabilities.append(probability); logps.append(logp)
    return jnp.where(mask, jnp.stack(probabilities, 1), 0), jnp.where(mask, jnp.stack(logps, 1), 0)


def select(logits, mask, words):
    _, _, rare_action, rare, masks = nodes(logits, mask)
    decisions = select_words(rare, rare_action, masks.reshape(-1, 2), words.reshape(-1, 2)).reshape(-1, 15)
    node, action = jnp.zeros(len(logits), jnp.int32), jnp.zeros(len(logits), jnp.int32)
    for _ in range(4):
        bit = jnp.maximum(decisions[jnp.arange(len(logits)), node], 0)
        action = action*2 + bit; node = 2*node + 1 + bit
    return jnp.where(mask.any(1), action, -1)


def initialize(seed):
    a, b = jax.random.split(jax.random.key(seed))
    return {'w': jax.random.normal(a, (FEATURES, 64))*jnp.float32(.025), 'b': jnp.zeros(64, jnp.float32),
            'pw': jax.random.normal(b, (64, 14))*jnp.float32(.025), 'pb': jnp.zeros(14, jnp.float32),
            'vw': jnp.zeros(64, jnp.float32), 'vb': jnp.zeros((), jnp.float32)}


def forward(parameters, features):
    h = jnp.tanh(features @ parameters['w'] + parameters['b'])
    return h @ parameters['pw'] + parameters['pb'], jax.nn.sigmoid(h @ parameters['vw'] + parameters['vb'])


@jax.jit
def act(parameters, features, mask, keys):
    logits, values = forward(parameters, features)
    words = jax.vmap(lambda key: jax.random.bits(key, (15, 2), dtype=jnp.uint32))(keys)
    actions = select(logits, mask, words)
    _, logs = distribution(logits, mask)
    return actions, jnp.where(mask.any(1), logs[jnp.arange(len(mask)), jnp.maximum(actions, 0)], 0), values


def objective(parameters, batch):
    logits, values = forward(parameters, batch['features'])
    p, logs = distribution(logits, batch['mask'])
    old, advantages, targets = [jax.lax.stop_gradient(batch[k]) for k in ['old_logp', 'advantages', 'returns']]
    ratio = jnp.exp(logs[jnp.arange(len(values)), batch['actions']] - old)
    active = jnp.where(advantages >= 0, ratio <= 1.2, ratio >= .8)
    surrogate = jnp.where(active, ratio, jax.lax.stop_gradient(jnp.clip(ratio, .8, 1.2))) * advantages
    actor, value, entropy = -surrogate.mean(), .5*jnp.mean((values-targets)**2), -jnp.mean(jnp.sum(p*logs, 1))
    loss = actor + .5*value - .01*entropy
    return loss, {'loss': loss, 'actor_loss': actor, 'value_loss': value, 'entropy': entropy,
                  'approx_kl': jnp.mean((ratio-1)-jnp.log(ratio))}


@jax.jit
def update(parameters, m, v, step, batch):
    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(parameters, batch)
    parameters, m, v, step, norm = adam_step(parameters, gradients, m, v, step, lr=.001, max_norm=1.)
    return parameters, m, v, step, {**metrics, 'gradient_norm': norm}


def finite_parameters(parameters):
    if not all(np.all(np.isfinite(np.asarray(x))) for x in parameters.values()):
        raise ValueError('Nonfinite full-Showdown parameters')
