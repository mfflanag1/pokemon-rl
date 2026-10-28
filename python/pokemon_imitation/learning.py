"""Broader supervised batches; unchanged fit and PPO kernels."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full_train.learning import derived_seed
from pokemon_transfer.learning import _validate_state,optimize_batch
from pokemon_fit.learning import checked_step,validate_batch,predictions
from pokemon_fit.data import fingerprint
from .data import select,validate,summarize

def reset_optimizer(p):
    z=jax.tree.map(jnp.zeros_like,p)
    return z,z,jnp.int32(0)

def bc_step(p,opt,data,old,seed,index):
    _validate_state(p,opt);validate_batch(old)
    new,ids=select(data,derived_seed('f15-teacher-sample',seed,index),64)
    old_ids=np.random.default_rng(derived_seed('f15-old-sample',seed,index)).integers(0,128,64)
    batch={k:np.concatenate([new[k],old[k][old_ids]]) for k in new};validate_batch(batch)
    before={k:np.asarray(p[k]).copy() for k in ['vw','vb']}
    p,opt,metrics=checked_step(p,opt,{k:jnp.asarray(v) for k,v in batch.items()});_validate_state(p,opt)
    if any(not np.array_equal(before[k],np.asarray(p[k])) for k in before):raise ValueError('BC changed value-head weights')
    return p,opt,{'teacher_indices':ids.tolist(),'old_indices':old_ids.tolist(),'batch_fingerprint':fingerprint(batch),'metrics':metrics}

def teacher_probe(p,data):
    validate(data);probs=[]
    for first in range(0,len(data['targets']),128):
        n=min(128,len(data['targets'])-first);x=data['features'][first:first+n];m=data['mask'][first:first+n]
        if n<128:x=np.concatenate([x,np.repeat(x[:1],128-n,axis=0)]);m=np.concatenate([m,np.repeat(m[:1],128-n,axis=0)])
        values,_=predictions(p,x,m);probs.append(np.asarray(values)[:n])
    probabilities=np.concatenate(probs)
    return probabilities,summarize(probabilities,data['mask'],data['targets'],data['game_index'])
