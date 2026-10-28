"""Player plus that player's observed history; original masked finite-word sampler."""
import jax
import jax.numpy as jnp
import numpy as np

from pokemon_full.policy import select, distribution
from .features import encode_packed, FEATURES
from .network import forward


@jax.jit
def infer(parameters, x, mask, seeds, counters):
    roots = jax.vmap(jax.random.key)(seeds)
    keys = jax.vmap(jax.random.fold_in)(roots, counters)
    logits, values = forward(parameters, x)
    words = jax.vmap(lambda k: jax.random.bits(k, (15, 2), dtype=jnp.uint32))(keys)
    actions = select(logits, mask, words)
    _, logs = distribution(logits, mask)
    return actions, jnp.where(mask.any(1), logs[jnp.arange(len(mask)), jnp.maximum(actions, 0)], 0), values


def batched_actor(parameters, players, histories, seeds, counters, capacity, catalog):
    n = len(players)
    if type(capacity) is not int or not 0 < n <= capacity or any(len(v) != n for v in [histories, seeds, counters]):
        raise ValueError('Structured actor batch shape')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           or not 0 <= int(v) < 2**32 for v in [*seeds, *counters]):
        raise ValueError('Structured actor seed/counter')
    x = np.zeros((capacity, FEATURES), np.float32)
    mask = np.zeros((capacity, 14), bool)
    s, c = np.zeros(capacity, np.uint32), np.zeros(capacity, np.uint32)
    for i, (player, history) in enumerate(zip(players, histories)):
        x[i], mask[i] = encode_packed(player, catalog, history)
        s[i], c[i] = seeds[i], counters[i]
    actions, logp, values = map(np.asarray, infer(parameters, x, mask, s, c))
    if not np.isfinite(logp).all() or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Structured actor finite range')
    for i in range(n):
        if mask[i].any():
            if not 0 <= actions[i] < 14 or not mask[i, actions[i]]:
                raise ValueError('Illegal structured actor action')
        elif actions[i] != -1 or logp[i] != 0:
            raise ValueError('Structured waiting actor')
    return [(x[i].copy(), mask[i].copy(), int(actions[i]), float(logp[i]), float(values[i])) for i in range(n)]
