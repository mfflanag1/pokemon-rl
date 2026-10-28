"""Full 128-event player history, sealed categorical actor, bounded value."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from pokemon_continuation.engine import PlayerView
from pokemon_switch_engine import policy as categorical
from pokemon_switch_engine.scalar import require

FEATURES = 1187
SHAPES = {'w': (FEATURES, 32), 'b': (32,), 'pw': (32, 4), 'pb': (4,), 'vw': (32,), 'vb': ()}


class ActorInput(NamedTuple):
    features: jax.Array
    mask: jax.Array


def encode(view):
    require(type(view) is PlayerView, 'Only continuation PlayerView accepted')
    n = view.context.shape[0]
    shapes = [(n, 2, 9), (n, 2, 2), (n, 2, 4), (n, 5), (n, 128, 9), (n, 4)]
    require(all(v.shape == s for v, s in zip(view, shapes)), 'PlayerView shapes')
    require(all(v.dtype == jnp.int32 for v in view[:-1]) and view.mask.dtype == jnp.bool_, 'PlayerView dtypes')
    return ActorInput(jnp.concatenate([v.reshape(n, -1) for v in view[:-1]], 1).astype(jnp.float32) / 128, view.mask)


def initialize(seed):
    require(type(seed) is int and 0 <= seed < 2**32, 'Model seed')
    a, b = jax.random.split(jax.random.key(seed))
    return {'w': jax.random.normal(a, SHAPES['w']) * jnp.float32(.05), 'b': jnp.zeros(32, jnp.float32),
            'pw': jax.random.normal(b, (32, 4)) * jnp.float32(.05), 'pb': jnp.zeros(4, jnp.float32),
            'vw': jnp.zeros(32, jnp.float32), 'vb': jnp.zeros((), jnp.float32)}


def forward(parameters, obs):
    require(type(obs) is ActorInput, 'Only live ActorInput accepted')
    h = jnp.tanh(obs.features @ parameters['w'] + parameters['b'])
    return h @ parameters['pw'] + parameters['pb'], jax.nn.sigmoid(h @ parameters['vw'] + parameters['vb'])


@jax.jit
def act(parameters, obs, keys):
    logits, values = forward(parameters, obs)
    words = jax.vmap(lambda k: jax.random.bits(k, (3, 2), dtype=jnp.uint32))(keys)
    actions = categorical.select(logits, obs.mask, words)
    _, logs = categorical.distribution(logits, obs.mask)
    live = obs.mask.any(1)
    return actions, jnp.where(live, logs[jnp.arange(len(actions)), jnp.maximum(actions, 0)], 0), jnp.where(live, values, 0)


def objective(parameters, buffer):
    obs = ActorInput(jnp.where(buffer.valid[:, None], buffer.features, 0), buffer.mask)
    logits, values = forward(parameters, obs)
    return categorical.loss(logits, values, buffer.mask, buffer.actions, buffer.old_logp,
                            buffer.advantages, buffer.returns, buffer.valid, .2, .5, .01)
