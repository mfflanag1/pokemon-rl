"""256 selected complete-return rows; reuse the sealed joint PPO/rehearsal kernel."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_transfer.learning import update,_validate_state
from pokemon_fit.learning import validate_batch
from pokemon_full.policy import distribution,forward
from pokemon_full_train.learning import objective
from pokemon_fit.learning import objective as supervised_objective
from pokemon_fixed.math import adam_step

def optimize(p,opt,batch,supervised):
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
        p,m,v,t,result=update(p,*opt,part,sup,jnp.float32(.1));opt=(m,v,t)
        values={k:float(x) for k,x in result.items()}
        if not all(np.isfinite(x) for x in values.values()):raise ValueError('Nonfinite learner metrics')
        metrics.append(values);_validate_state(p,opt)
    return p,opt,metrics

@jax.jit
def switch_probability(p,x,mask):
    probabilities,_=distribution(forward(p,x)[0],mask)
    return probabilities[:,8:].sum(1)

def gradient_probe(p,batch,supervised,probe):
    part={k:jnp.asarray(v) for k,v in batch.items()};part['valid']=jnp.ones(len(part['actions']),bool)
    sup={k:jnp.asarray(v) for k,v in supervised.items()};x=jnp.asarray(probe['features']);mask=jnp.asarray(probe['mask'])
    before=float(switch_probability(p,x,mask).mean());result={}
    def component(q,name):
        loss,m=objective(q,part);bc=.1*supervised_objective(q,sup)
        return {'actor':m['actor_loss'],'value':.5*m['value_loss'],'entropy':-.01*m['entropy'],'rehearsal':bc,'joint':loss+bc}[name]
    for name in ['actor','value','entropy','rehearsal','joint']:
        grad=jax.grad(lambda q:component(q,name))(p);zero=jax.tree.map(jnp.zeros_like,p)
        new,_,_,_,norm=adam_step(p,grad,zero,zero,jnp.int32(0),lr=.0003,max_norm=1.)
        after=float(switch_probability(new,x,mask).mean())
        result[name]={'loss':float(component(p,name)),'gradient_norm':float(norm),'switch_probability_before':before,
                      'switch_probability_after':after,'delta':after-before}
    return result
