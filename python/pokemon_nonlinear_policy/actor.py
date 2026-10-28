"""Player-projection-only inference; same seeded finite-word action selection."""
import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import select,distribution
from pokemon_terminal_matchup.features import encode,FEATURES
from .network import forward

@partial(jax.jit,static_argnames=('arm',))
def infer(p,x,m,seeds,counters,arm):
    roots=jax.vmap(jax.random.key)(seeds);keys=jax.vmap(jax.random.fold_in)(roots,counters)
    logits,values=forward(p,x,arm)
    words=jax.vmap(lambda k:jax.random.bits(k,(15,2),dtype=jnp.uint32))(keys)
    actions=select(logits,m,words);_,logs=distribution(logits,m)
    return actions,jnp.where(m.any(1),logs[jnp.arange(len(m)),jnp.maximum(actions,0)],0),values

def batched_actor(p,players,seeds,counters,capacity,arm,catalog):
    if arm not in ['control','nonlinear']:raise ValueError('Unknown arm')
    n=len(players)
    if type(capacity)is not int or not 0<n<=capacity or len(seeds)!=n or len(counters)!=n:
        raise ValueError('Actor batch shape')
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not 0<=int(v)<2**32 for v in [*seeds,*counters]):
        raise ValueError('Actor seed/counter')
    x=np.zeros((capacity,FEATURES),np.float32);m=np.zeros((capacity,14),bool)
    s=np.zeros(capacity,np.uint32);c=np.zeros(capacity,np.uint32)
    for i,player in enumerate(players):x[i],m[i]=encode(player,catalog,'matchup');s[i],c[i]=seeds[i],counters[i]
    a,lp,v=map(np.asarray,infer(p,x,m,s,c,arm))
    if not np.isfinite(lp).all() or not np.isfinite(v).all() or np.any((v<0)|(v>1)):raise ValueError('Actor finite range')
    for i in range(n):
        if m[i].any():
            if not 0<=a[i]<14 or not m[i,a[i]]:raise ValueError('Illegal actor action')
        elif a[i]!=-1 or lp[i]!=0:raise ValueError('Waiting actor')
    return [(x[i].copy(),m[i].copy(),int(a[i]),float(lp[i]),float(v[i])) for i in range(n)]
