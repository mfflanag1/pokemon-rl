"""Four-action policy via two levels of exact dyadic binary conditionals.

Each node uses the sealed binary policy's FP32 rare mass and real complement.
Independent 64-bit words sample root and child. Leaf logs are sums of exactly
those conditional logs, avoiding a rounded categorical CDF and unreachable tails.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_binary.policy import parts as binary_parts, select_words
from .engine import PlayerView
from .scalar import require

FEATURES=323


class ActorInput(NamedTuple):
    features: jax.Array
    mask: jax.Array


def encode(view):
    require(type(view) is PlayerView,'Actor encoder accepts only PlayerView')
    n=view.context.shape[0]
    return ActorInput(jnp.concatenate([x.reshape(n,-1) for x in view[:-1]],axis=1).astype(jnp.float32)/128,view.mask)


def _group(logits,mask):
    any_=jnp.any(mask,axis=1);maximum=jax.lax.stop_gradient(jnp.max(jnp.where(mask,logits,-jnp.inf),axis=1))
    maximum=jnp.where(any_,maximum,0);weights=jnp.where(mask,jnp.exp(jnp.where(mask,logits-maximum[:,None],0)),0)
    return maximum+jnp.log(jnp.where(any_,jnp.sum(weights,axis=1),1))


def nodes(logits,mask):
    left,right=logits[:,:2],logits[:,2:];lm,rm=mask[:,:2],mask[:,2:]
    root=jnp.stack([_group(left,lm),_group(right,rm)],axis=1)
    masks=jnp.stack([jnp.stack([lm.any(1),rm.any(1)],axis=1),lm,rm],axis=1)
    inputs=jnp.stack([root,left,right],axis=1)
    p,logs,rare_action,rare=binary_parts(inputs.reshape(-1,2),masks.reshape(-1,2))
    return p.reshape(-1,3,2),logs.reshape(-1,3,2),rare_action,rare,masks


def distribution(logits,mask):
    p,logs,*_=nodes(logits,mask)
    probs=jnp.concatenate([p[:,0,0,None]*p[:,1],p[:,0,1,None]*p[:,2]],axis=1)
    logp=jnp.concatenate([logs[:,0,0,None]+logs[:,1],logs[:,0,1,None]+logs[:,2]],axis=1)
    return jnp.where(mask,probs,0),jnp.where(mask,logp,0)


def select(logits,mask,words):
    _,_,rare_action,rare,masks=nodes(logits,mask)
    decisions=select_words(rare,rare_action,masks.reshape(-1,2),words.reshape(-1,2)).reshape(-1,3)
    root=jnp.maximum(decisions[:,0],0);child=decisions[jnp.arange(len(root)),1+root]
    return jnp.where(mask.any(1),root*2+child,-1).astype(jnp.int32)


def initialize(seed):
    require(type(seed) is int and 0<=seed<2**32,'Model seed')
    a,b=jax.random.split(jax.random.key(seed))
    return {'w':jax.random.normal(a,(FEATURES,32))*jnp.float32(.05),'b':jnp.zeros(32,jnp.float32),
            'pw':jax.random.normal(b,(32,4))*jnp.float32(.05),'pb':jnp.zeros(4,jnp.float32),
            'vw':jnp.zeros(32,jnp.float32),'vb':jnp.zeros((),jnp.float32)}


def forward(parameters,obs):
    require(type(obs) is ActorInput,'Policy/value accept only ActorInput')
    hidden=jnp.tanh(obs.features@parameters['w']+parameters['b'])
    return hidden@parameters['pw']+parameters['pb'],hidden@parameters['vw']+parameters['vb']


@jax.jit
def act(parameters,obs,keys):
    logits,values=forward(parameters,obs);words=jax.vmap(lambda k:jax.random.bits(k,(3,2),dtype=jnp.uint32))(keys)
    actions=select(logits,obs.mask,words);_,logs=distribution(logits,obs.mask)
    live=obs.mask.any(1);selected=logs[jnp.arange(len(actions)),jnp.maximum(actions,0)]
    return actions,jnp.where(live,selected,0),jnp.where(live,values,0)


def loss(logits,values,mask,actions,old_logp,advantages,returns,valid,clip=.2,value_coef=.5,entropy_coef=.01):
    # Empty/wait rows are finite inert padding; neither optimizer sees them.
    safe=jnp.where(valid[:,None],mask,jnp.array([True,False,False,False]));n=jnp.maximum(1,valid.sum())
    logits=jnp.where(valid[:,None],logits,0);values=jnp.where(valid,values,0)
    probs,logs=distribution(logits,safe);rows=jnp.arange(len(values));indices=jnp.where(valid,actions,0)
    old_logp,advantages,returns=[jax.lax.stop_gradient(jnp.where(valid,x,0)) for x in (old_logp,advantages,returns)]
    ratio=jnp.exp(logs[rows,indices]-old_logp);clipped=jnp.clip(ratio,1-clip,1+clip)
    active=jnp.where(advantages>=0,ratio<=1+clip,ratio>=1-clip)
    surrogate=jnp.where(active,ratio,jax.lax.stop_gradient(clipped))*advantages
    mean=lambda x:jnp.sum(jnp.where(valid,x,0))/n
    entropy=mean(-jnp.sum(probs*logs,axis=1));actor=-mean(surrogate);value=.5*mean((values-returns)**2)
    total=actor+value_coef*value-entropy_coef*entropy
    return total,{'loss':total,'actor_loss':actor,'value_loss':value,'entropy':entropy,
                  'clip_fraction':mean(jnp.abs(ratio-1)>clip),'approx_kl':mean((ratio-1)-jnp.log(ratio))}
