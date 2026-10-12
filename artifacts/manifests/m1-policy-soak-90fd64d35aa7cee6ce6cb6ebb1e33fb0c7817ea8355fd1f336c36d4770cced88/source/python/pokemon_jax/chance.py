"""Event-addressed Threefry streams; exact range mapping conditional on uniform bits."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_slice.engine import require
from .engine import Chance

LIMIT24 = 4294967280  # floor(2**32/24)*24: discard the 16 excess words.


def accepted24(word):
    return word < jnp.uint32(LIMIT24)


def finish24(key, word):
    """Finish rejection from an injected first word (also exercises rare branches)."""
    def condition(pair):
        return ~accepted24(pair[1])
    def redraw(pair):
        key, _ = pair
        next_key, draw_key = jax.random.split(key)
        return next_key, jax.random.bits(draw_key, dtype=jnp.uint32)
    _, word = jax.lax.while_loop(condition,redraw,(key,word))
    return word % jnp.uint32(24)


def _crit(key):
    next_key, draw_key = jax.random.split(key)
    word = jax.random.bits(draw_key,dtype=jnp.uint32)
    return finish24(next_key,word) == 0


def event_key(master, game_id, turn, role):
    key=jax.random.fold_in(master,game_id)
    key=jax.random.fold_in(key,turn)
    return jax.random.fold_in(key,role)


def _sample(master, game_id, turn):
    keys=[event_key(master,game_id,turn,role) for role in range(5)]
    crit=jnp.stack([_crit(keys[0]),_crit(keys[1])])
    roll=jnp.stack([85+(jax.random.bits(k,dtype=jnp.uint32)&15).astype(jnp.int32) for k in keys[2:4]])
    tie=(jax.random.bits(keys[4],dtype=jnp.uint32)&1).astype(jnp.int32)
    return Chance(crit,roll,tie)


sample_kernel=jax.jit(jax.vmap(_sample,in_axes=(None,0,0)))


def sample(seed, game_ids, turns):
    require(type(seed) is int and 0<=seed<2**32, 'Seed must be uint32')
    ids=np.asarray(game_ids);ts=np.asarray(turns)
    require(ids.ndim==1 and len(ids)>0 and ids.dtype.kind in 'iu' and np.all((ids>=0)&(ids<2**32)), 'Game IDs must be uint32')
    require(len(np.unique(ids))==len(ids),'Repeated game IDs')
    require(ts.shape==ids.shape and ts.dtype.kind in 'iu' and np.all((ts>=1)&(ts<=4)), 'Turn bounds')
    master=jax.random.key(seed,impl='threefry2x32')
    return sample_kernel(master,jnp.asarray(ids,jnp.uint32),jnp.asarray(ts,jnp.uint32))
