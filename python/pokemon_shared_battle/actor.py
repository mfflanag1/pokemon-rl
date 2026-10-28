"""Observation-only stochastic actor using the original finite-word action law."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode, FEATURES
from pokemon_full.policy import select, distribution
from pokemon_shared.network import forward
from pokemon_throughput.collector import batched_actor as flat_actor

@jax.jit
def infer_shared(parameters, features, masks, seeds, counters):
    roots=jax.vmap(jax.random.key)(seeds)
    keys=jax.vmap(jax.random.fold_in)(roots,counters)
    logits,values=forward(parameters,features)
    words=jax.vmap(lambda k:jax.random.bits(k,(15,2),dtype=jnp.uint32))(keys)
    actions=select(logits,masks,words)
    _,logs=distribution(logits,masks)
    return actions,jnp.where(masks.any(1),logs[jnp.arange(len(masks)),jnp.maximum(actions,0)],0),values

def batched_actor(parameters, players, seeds, counters, capacity, architecture):
    if architecture not in ['flat','shared']:
        raise ValueError('Unknown frozen architecture')
    n=len(players)
    if type(capacity) is not int or not 0<n<=capacity or len(seeds)!=n or len(counters)!=n:
        raise ValueError('Invalid actor batch')
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not 0<=int(v)<2**32
           for v in [*seeds,*counters]):
        raise ValueError('Invalid actor seed/counter')
    if architecture=='flat':
        return flat_actor(parameters,players,seeds,counters,capacity)
    x=np.zeros((capacity,FEATURES),np.float32);m=np.zeros((capacity,14),bool)
    s=np.zeros(capacity,np.uint32);c=np.zeros(capacity,np.uint32)
    for i,player in enumerate(players):
        x[i],m[i]=encode(player);s[i],c[i]=seeds[i],counters[i]
    a,logs,values=map(np.asarray,infer_shared(parameters,x,m,s,c))
    if not np.isfinite(logs).all() or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Invalid actor output')
    for i in range(n):
        if m[i].any():
            if not 0<=a[i]<14 or not m[i,a[i]]:raise ValueError('Illegal frozen action')
        elif a[i]!=-1 or logs[i]!=0:raise ValueError('Invalid waiting action')
    return [(x[i].copy(),m[i].copy(),int(a[i]),float(logs[i]),float(values[i])) for i in range(n)]
