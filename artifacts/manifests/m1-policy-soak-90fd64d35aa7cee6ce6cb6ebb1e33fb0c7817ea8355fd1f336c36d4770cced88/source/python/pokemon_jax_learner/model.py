"""Row-independent linear actor/value model; its only data argument is ActorInput."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax.engine import PlayerView
from pokemon_learner.core import require
from .core import distribution_kernel, loss_kernel


class ActorInput(NamedTuple):
    features: jax.Array
    mask: jax.Array


def validate_input(obs):
    require(type(obs) is ActorInput,'Actor accepts only ActorInput')
    x,m=map(np.asarray,obs)
    require(x.ndim==2 and x.shape[0]>0 and x.shape[1]==59 and x.dtype==np.float32,'Actor feature shape/dtype')
    require(m.shape==(len(x),2) and m.dtype==np.bool_,'Actor mask shape/dtype')
    require(np.all(np.isfinite(x)),'Nonfinite actor features')
    return len(x)


def encode(view):
    require(type(view) is PlayerView,'Encoder accepts only PlayerView')
    h,t=map(np.asarray,view);n=len(h)
    require(h.shape==(n,11) and t.shape==(n,8,6) and h.dtype==t.dtype==np.int32,'View shape/dtype')
    require(np.all((h[:,9:]>=0)&(h[:,9:]<=1)),'View mask bounds')
    require(np.all((h[:,1]==0)| (h[:,1]==1)) and np.all(h[:,9:].any(axis=1)==(h[:,1]==0)),'View phase/mask')
    obs=ActorInput(jnp.concatenate([view.header,view.history.reshape(n,48)],axis=1).astype(jnp.float32)/jnp.float32(105),
                   view.header[:,9:].astype(jnp.bool_))
    validate_input(obs);return obs


def initialize(seed):
    require(type(seed) is int and 0<=seed<2**32,'Model seed')
    key=jax.random.key(seed,impl='threefry2x32')
    return {'policy_w':jax.random.normal(key,(59,2),dtype=jnp.float32)*jnp.float32(.01),
            'policy_b':jnp.zeros(2,jnp.float32),'value_w':jnp.zeros(59,jnp.float32),'value_b':jnp.zeros((),jnp.float32)}


def validate_parameters(p):
    expected={'policy_w':(59,2),'policy_b':(2,),'value_w':(59,),'value_b':()}
    require(type(p) is dict and set(p)==set(expected),'Model parameter keys')
    for k,shape in expected.items():
        a=np.asarray(p[k]);require(a.shape==shape and a.dtype==np.float32 and np.all(np.isfinite(a)),'Model FP32 parameters')


def forward(parameters,obs):
    logits=obs.features@parameters['policy_w']+parameters['policy_b']
    values=obs.features@parameters['value_w']+parameters['value_b']
    return logits,values


@jax.jit
def predict_kernel(parameters,obs):
    logits,values=forward(parameters,obs)
    live=jnp.any(obs.mask,axis=1)
    # Empty masks are terminal transport padding, never trainable decisions.
    safe=jnp.where(live[:,None],obs.mask,jnp.array([True,False]))
    probabilities,logp=distribution_kernel(logits,safe)
    return (jnp.where(live[:,None],probabilities,0),jnp.where(live[:,None],logp,0),jnp.where(live,values,0))


def predict(parameters,obs):
    validate_parameters(parameters);validate_input(obs)
    out=predict_kernel(parameters,obs)
    require(all(np.all(np.isfinite(x)) for x in out),'Nonfinite actor output')
    return out


@jax.jit
def action_kernel(parameters,obs,keys):
    probabilities,logp,values=predict_kernel(parameters,obs)
    live=jnp.any(obs.mask,axis=1)
    cdf=jnp.cumsum(probabilities,axis=1).at[:,-1].set(1.)
    u=jax.vmap(lambda key:jax.random.uniform(key,(),dtype=jnp.float32))(keys)
    actions=jnp.sum(u[:,None]>=cdf,axis=1).astype(jnp.int32)
    selected=logp[jnp.arange(len(actions)),actions]
    return jnp.where(live,actions,-1),jnp.where(live,selected,0),values


def act(parameters,obs,keys):
    n=validate_input(obs);validate_parameters(parameters)
    require(keys.shape==(n,) and str(keys.dtype)=='key<fry>','Action keys must be one Threefry key per actor row')
    result=action_kernel(parameters,obs,keys);actions=np.asarray(result[0]);mask=np.asarray(obs.mask);live=mask.any(axis=1)
    require(np.all(actions[~live]==-1) and np.all(mask[np.arange(n)[live],actions[live]]),'Illegal sampled action')
    require(all(np.all(np.isfinite(x)) for x in result),'Nonfinite actor output')
    return result


def model_loss(parameters,features,mask,actions,old_logp,advantages,returns,clip,value_coef,entropy_coef):
    logits,values=forward(parameters,ActorInput(features,mask))
    return loss_kernel(logits,values,mask,actions,old_logp,advantages,returns,clip,value_coef,entropy_coef)


model_loss_grad=jax.jit(jax.value_and_grad(model_loss,has_aux=True),static_argnums=(7,8,9))
