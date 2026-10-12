"""Event-addressed draws with pinned Showdown finite-word maps, not sodium replay."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax.engine import Chance
from pokemon_jax.chance import event_key
from pokemon_learner.core import require
from .mapping import kernel

DOMAIN=0x53445731  # SDW1; intentionally new environment stream


def _words(master,game_id,turn):
    domain=jax.random.fold_in(master,DOMAIN)
    return jnp.stack([jax.random.bits(event_key(domain,game_id,turn,role),dtype=jnp.uint32) for role in range(5)])


words_kernel=jax.jit(jax.vmap(_words,in_axes=(None,0,0)))


@jax.jit
def from_words(words):
    """Rows [crit0,crit1,roll0,roll1,tie], canonical pre-shuffle seat order [0,1]."""
    flat=kernel(words.reshape(-1));n=words.shape[0]
    return Chance(flat['crit'].reshape(n,5)[:,:2],flat['roll'].reshape(n,5)[:,2:4],
                  flat['tie0'].reshape(n,5)[:,4])


@jax.jit
def sample_kernel(master,game_ids,turns):
    return from_words(words_kernel(master,game_ids,turns))


def sample(seed,game_ids,turns):
    require(jax.default_backend()=='cpu' and not jax.config.x64_enabled,'CPU FP32 only')
    require(type(seed) is int and 0<=seed<2**32,'Seed range')
    ids=np.asarray(game_ids);ts=np.asarray(turns)
    require(ids.ndim==1 and len(ids)>0 and ids.dtype.kind in 'iu' and np.all((ids>=0)&(ids<2**32)), 'Game ID range')
    require(len(np.unique(ids))==len(ids),'Duplicate game ID')
    require(ts.shape==ids.shape and ts.dtype.kind in 'iu' and np.all((ts>=1)&(ts<=4)), 'Turn range')
    return sample_kernel(jax.random.key(seed,impl='threefry2x32'),jnp.asarray(ids,jnp.uint32),jnp.asarray(ts,jnp.uint32))
