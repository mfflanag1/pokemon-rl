"""Same PPO kernel and two epoch schedule, explicit arm-dependent collection size."""
import numpy as np
from pokemon_matchup.learning import validate,update
from .common import budget,check_state

def optimize_batch(p,opt,batch,seed,arm):
    cfg=budget(arm);n=cfg['selected_rows'];steps=cfg['optimizer_steps_per_cohort']
    if set(batch)!={'features','mask','actions','old_logp','advantages','returns'} or len(batch['actions'])!=n:
        raise ValueError('Fixed cohort inventory')
    b=validate({**batch,'valid':np.ones(n,bool)});check_state(p,opt,arm)
    if int(opt[2])>2**31-1-steps:raise ValueError('Counter overflow')
    rng=np.random.default_rng(seed);records=[]
    for _ in range(2):
        order=rng.permutation(n)
        for first in range(0,n,256):
            ids=order[first:first+256]
            p,opt,metrics=update(p,opt,{k:v[ids] for k,v in b.items()});records.append(metrics)
    if len(records)!=steps:raise ValueError('Minibatch schedule')
    return p,opt,records
