"""Event-addressed chance; exact pinned word maps, independently keyed events."""
import jax
import jax.numpy as jnp
from pokemon_showdown.mapping import kernel as word_map
from .engine import Chance

DOMAIN = 0x53574531


@jax.jit
def sample(master,game_ids,groups):
    def words(game,group):
        key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(master,DOMAIN),game),group)
        return jnp.stack([jax.random.bits(jax.random.fold_in(key,i),dtype=jnp.uint32) for i in range(5)])
    raw=jax.vmap(words)(game_ids,groups);mapped=word_map(raw.reshape(-1))
    return Chance(mapped['crit'].reshape(-1,5)[:,:2],100-mapped['roll'].reshape(-1,5)[:,2:4],mapped['tie0'].reshape(-1,5)[:,4])
