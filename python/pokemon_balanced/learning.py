"""Only demonstration sampling changes; sealed likelihood/Adam stays identical."""
import numpy as np
import jax.numpy as jnp
from pokemon_full_train.learning import derived_seed
from pokemon_transfer.learning import _validate_state
from pokemon_fit.learning import checked_step,validate_batch
from pokemon_fit.data import fingerprint
from pokemon_imitation.learning import teacher_probe
from pokemon_imitation.data import summarize
from .sampling import select,classify
NAMES=['normal_move','tera_move','voluntary_switch','forced_switch']

def bc_step(p,opt,data,old,seed,index):
    _validate_state(p,opt);validate_batch(old)
    new,ids=select(data,derived_seed('f16-balanced-sample',seed,index))
    old_ids=np.random.default_rng(derived_seed('f15-old-sample',seed,index)).integers(0,128,64)
    batch={k:np.concatenate([new[k],old[k][old_ids]]) for k in new};validate_batch(batch)
    before={k:np.asarray(p[k]).copy() for k in ['vw','vb']}
    p,opt,metrics=checked_step(p,opt,{k:jnp.asarray(v) for k,v in batch.items()});_validate_state(p,opt)
    if any(not np.array_equal(before[k],np.asarray(p[k])) for k in before):raise ValueError('Value-head changed during BC')
    return p,opt,{'teacher_indices':ids.tolist(),'old_indices':old_ids.tolist(),'batch_fingerprint':fingerprint(batch),'metrics':metrics}

def probe(p,data):
    probabilities,total=teacher_probe(p,data);labels=classify(data);families={}
    for i,name in enumerate(NAMES):
        ids=labels==i
        if not ids.any():raise ValueError('Missing probe family '+name)
        families[name]=summarize(probabilities[ids],data['mask'][ids],data['targets'][ids],data['game_index'][ids])
    return probabilities,{'total':total,'families':families}

def support(data,plan):
    labels=classify(data);out={}
    for i,name in enumerate(NAMES):
        ids=labels==i;out[name]={'rows':int(ids.sum()),'games':len(np.unique(data['game_index'][ids]))}
        if not ids.any():raise ValueError('Missing probe family '+name)
    t=plan['coverage_thresholds']
    for name,prefix in [('tera_move','tera'),('voluntary_switch','voluntary_switch')]:
        for key in ['rows','games']:
            if out[name][key]<t['minimum_'+prefix+'_'+key]:raise ValueError('Insufficient fixed probe support '+name)
    return out
