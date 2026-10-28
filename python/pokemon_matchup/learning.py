"""Same frozen-target PPO/GAE/Adam rules, with an additive matchup residual."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import distribution
from pokemon_fixed.math import adam_step
from pokemon_full_train.learning import PPO
from pokemon_shared_ppo.learning import _minibatch
from .network import forward, check_state
from .features import FEATURES

def objective(p,b):
    valid=b['valid'];x=jnp.where(valid[:,None],b['features'],0.)
    mask=jnp.where(valid[:,None],b['mask'],True);actions=jnp.where(valid,b['actions'],0)
    logits,values=forward(p,x);probs,logs=distribution(logits,mask)
    old,adv,targets=[jax.lax.stop_gradient(jnp.where(valid,b[k],0.)) for k in ['old_logp','advantages','returns']]
    lr=logs[jnp.arange(len(values)),actions]-old;ratio=jnp.exp(lr)
    active=jnp.where(adv>=0,ratio<=1+PPO['clip'],ratio>=1-PPO['clip'])
    surrogate=jnp.where(active,ratio,jax.lax.stop_gradient(jnp.clip(ratio,1-PPO['clip'],1+PPO['clip'])))*adv
    def mean(a):return jnp.where(valid,a,0.).sum()/jnp.maximum(valid.sum(),1)
    actor=-mean(surrogate);value=.5*mean((values-targets)**2);entropy=-mean(jnp.sum(probs*logs,1))
    loss=actor+PPO['value_coefficient']*value-PPO['entropy']*entropy
    return loss,dict(loss=loss,actor_loss=actor,value_loss=value,entropy=entropy,
        approx_kl=mean(ratio-1-lr),clip_fraction=mean((jnp.abs(ratio-1)>PPO['clip']).astype(jnp.float32)))

def validate(b):
    if not isinstance(b,dict) or 'features' not in b:raise ValueError('Missing features')
    x=np.asarray(b['features']);valid=np.asarray(b.get('valid'))
    if valid.dtype!=np.bool_ or valid.ndim!=1 or x.shape!=(len(valid),FEATURES) or x.dtype!=np.float32:
        raise ValueError('Augmented batch schema')
    if not np.isfinite(x[valid]).all():raise ValueError('Nonfinite feature')
    _minibatch({**b,'features':x[:,:928]})
    return {k:jnp.asarray(v) for k,v in b.items()}

@jax.jit
def kernel(p,m,v,step,b):
    (_,metrics),g=jax.value_and_grad(objective,has_aux=True)(p,b)
    p,m,v,step,norm=adam_step(p,g,m,v,step,lr=PPO['learning_rate'],max_norm=1.)
    return p,(m,v,step),{**metrics,'gradient_norm':norm}

def update(p,opt,b):
    check_state(p,opt)
    if int(opt[2])>=2**31-1:raise ValueError('Counter overflow')
    q,qo,metrics=kernel(p,*opt,validate(b));check_state(q,qo)
    if int(qo[2])!=int(opt[2])+1 or not all(np.isfinite(float(v)) for v in metrics.values()):
        raise ValueError('Update linkage/metrics')
    return q,qo,{k:float(v) for k,v in metrics.items()}

def optimize_batch(p,opt,batch,seed,arm):
    if arm not in ['zero','matchup']:raise ValueError('Unknown arm')
    if set(batch)!={'features','mask','actions','old_logp','advantages','returns'} or len(batch['actions'])!=1024:
        raise ValueError('Fixed cohort inventory')
    b=validate({**batch,'valid':np.ones(1024,bool)})
    if arm=='zero' and np.any(np.asarray(b['features'])[:,928:]!=0):raise ValueError('Control input leak')
    check_state(p,opt)
    if int(opt[2])>2**31-9:raise ValueError('Counter overflow')
    rng=np.random.default_rng(seed);records=[]
    for _ in range(2):
        order=rng.permutation(1024)
        for first in range(0,1024,256):
            ids=order[first:first+256];part={k:v[ids] for k,v in b.items()}
            p,opt,metrics=update(p,opt,part);records.append(metrics)
    return p,opt,records
