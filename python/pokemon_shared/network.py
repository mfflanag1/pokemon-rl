"""Shared, permutation-equivariant policy on validated public features."""

import jax
import jax.numpy as jnp


SHAPES = {
    "mon_w": (64, 32), "mon_b": (32,),
    "move_w": (24, 32), "move_b": (32,),
    "context_w": (64, 32), "context_b": (32,),
    "score_w": (195, 281), "score_b": (281,),
    "logit_w": (281,), "logit_b": (),
    "value_w": (160,), "value_b": (),
}


def initialize(seed):
    """Initialize the five policy weights in the specified PRNG split order."""
    parameters = {name: jnp.zeros(shape, jnp.float32)
                  for name, shape in SHAPES.items()}
    names = ("mon_w", "move_w", "context_w", "score_w", "logit_w")
    for name, key in zip(names, jax.random.split(jax.random.key(seed), 5)):
        parameters[name] = jax.random.normal(
            key, SHAPES[name], dtype=jnp.float32) * jnp.float32(.025)
    return parameters


def forward(parameters, features):
    """Return unmasked action logits and sigmoid values for [N, 928] inputs."""
    p = parameters
    mons = features[:, :768].reshape((-1, 12, 64))
    embedded = jnp.tanh(mons @ p["mon_w"] + p["mon_b"])
    present = mons[:, :, 0:1]
    masked = embedded * present
    own_mean = masked[:, :6].sum(axis=1) / jnp.maximum(
        present[:, :6].sum(axis=1), 1)
    opponent_mean = masked[:, 6:].sum(axis=1) / jnp.maximum(
        present[:, 6:].sum(axis=1), 1)
    active = masked * mons[:, :, 1:2]
    context = jnp.tanh(features[:, 864:928] @ p["context_w"] + p["context_b"])
    global_features = jnp.concatenate((own_mean, opponent_mean,
                                      active[:, :6].sum(axis=1),
                                      active[:, 6:].sum(axis=1), context), axis=-1)
    moves = jnp.tanh(features[:, 768:864].reshape((-1, 4, 24))
                     @ p["move_w"] + p["move_b"])
    candidates = jnp.concatenate((moves, moves, embedded[:, :6]), axis=1)
    kinds = jax.nn.one_hot(jnp.array([0]*4 + [1]*4 + [2]*6), 3)
    scorer_input = jnp.concatenate((
        jnp.broadcast_to(global_features[:, None, :],
                         (features.shape[0], 14, 160)),
        candidates, jnp.broadcast_to(kinds, (features.shape[0], 14, 3))), axis=-1)
    hidden = jnp.tanh(scorer_input @ p["score_w"] + p["score_b"])
    logits = hidden @ p["logit_w"] + p["logit_b"]
    values = jax.nn.sigmoid(global_features @ p["value_w"] + p["value_b"])
    return logits, values
