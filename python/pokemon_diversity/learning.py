"""Prepared sampling, unchanged F16 supervised update and actor inputs."""
import numpy as np
import jax.numpy as jnp
from pokemon_full_train.learning import derived_seed
from pokemon_transfer.learning import _validate_state
from pokemon_fit.learning import checked_step, validate_batch
from pokemon_fit.data import fingerprint

def bc_step(p, opt, sampler, old, seed, index):
    _validate_state(p,opt); validate_batch(old)
    new, ids = sampler.sample(derived_seed('f16-balanced-sample',seed,index))
    oi = np.random.default_rng(derived_seed('f15-old-sample',seed,index)).integers(0,128,64)
    batch = {k:np.concatenate([new[k],old[k][oi]]) for k in new}
    validate_batch(batch)
    before = {k:np.asarray(p[k]).copy() for k in ['vw','vb']}
    p,opt,metrics = checked_step(p,opt,{k:jnp.asarray(v) for k,v in batch.items()})
    _validate_state(p,opt)
    if any(not np.array_equal(before[k],np.asarray(p[k])) for k in before):
        raise ValueError('Value-head changed during BC')
    return p,opt,{'teacher_indices':ids.tolist(),'old_indices':oi.tolist(),'batch_fingerprint':fingerprint(batch),'metrics':metrics}
