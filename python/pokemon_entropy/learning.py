"""Change only entropy pressure; the .01 control calls the sealed kernel exactly."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full_train.learning import objective as ppo_objective
from pokemon_fit.learning import objective as fit_objective,validate_batch
from pokemon_transfer.learning import update as standard_update,_validate_state
from pokemon_fixed.math import adam_step

def no_entropy_objective(p,part,sup):
    _,metrics=ppo_objective(p,part)
    loss=metrics['actor_loss']+jnp.float32(.5)*metrics['value_loss']
    nll=fit_objective(p,sup);total=loss+jnp.float32(.1)*nll
    return total,{**metrics,'loss':loss,'total_loss':total,'rehearsal_nll':nll}

@jax.jit
def no_entropy_update(p,m,v,t,part,sup):
    (_,metrics),grad=jax.value_and_grad(no_entropy_objective,has_aux=True)(p,part,sup)
    p,m,v,t,norm=adam_step(p,grad,m,v,t,lr=.0003,max_norm=1.)
    return p,m,v,t,{**metrics,'gradient_norm':norm}

def optimize(p,opt,batch,supervised,coefficient):
    if isinstance(coefficient,(bool,np.bool_)) or coefficient not in (0.,.01):raise ValueError('Unregistered entropy coefficient')
    _validate_state(p,opt);validate_batch(supervised)
    shapes={'features':(256,928),'mask':(256,14),'actions':(256,),'old_logp':(256,),'advantages':(256,),'returns':(256,)}
    if set(batch)!=set(shapes):raise ValueError('Unexpected learner fields')
    for k,a in batch.items():
        dtype=np.bool_ if k=='mask' else np.int32 if k=='actions' else np.float32
        if a.shape!=shapes[k] or a.dtype!=dtype or not np.isfinite(a).all():raise ValueError('Invalid learner array: '+k)
    if not batch['mask'].any(1).all() or not ((batch['actions']>=0)&(batch['actions']<14)).all():raise ValueError('Invalid action mask')
    if not batch['mask'][np.arange(256),batch['actions']].all() or (batch['old_logp']>0).any():raise ValueError('Illegal sampled action')
    if ((batch['returns'] < -1e-5)|(batch['returns']>1+1e-5)).any():raise ValueError('Unbounded return')
    part={k:jnp.asarray(v) for k,v in batch.items()};part['valid']=jnp.ones(256,bool)
    sup={k:jnp.asarray(v) for k,v in supervised.items()};metrics=[]
    for _ in range(2):
        if coefficient==.01:p,m,v,t,result=standard_update(p,*opt,part,sup,jnp.float32(.1))
        else:p,m,v,t,result=no_entropy_update(p,*opt,part,sup)
        opt=(m,v,t);values={k:float(x) for k,x in result.items()}
        if not all(np.isfinite(x) for x in values.values()):raise ValueError('Nonfinite learner metrics')
        metrics.append(values);_validate_state(p,opt)
    return p,opt,metrics
