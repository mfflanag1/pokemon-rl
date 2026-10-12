"""Synchronous collect/update boundary; frozen behavior, no persistent live games."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner.core import require
from .core import Adam
from .model import initialize, ActorInput, predict, model_loss_grad
from .collector import collect, validate_collection, policy_hash


def validate_plan(plan):
    require(plan['version']=='m1-jax-learner-plan-1' and plan['checkpoint_mode']=='completed-update-whole-episodes','Plan mode')
    require(plan['backend']=='cpu' and plan['dtype']=='float32' and plan['x64_enabled'] is False,'CPU FP32 plan')
    require(jax.default_backend()=='cpu' and not jax.config.x64_enabled,'Unsupported runtime')
    for key in ('games_per_collection','epochs','minibatch_size'):
        require(type(plan[key]) is int and plan[key]>0,'Plan positive integer: '+key)
    require(0<=plan['gamma']<=1 and 0<=plan['lambda']<=1 and 0<plan['clip']<1,'Discount/clip range')
    require(all(np.isfinite(plan[k]) and plan[k]>=0 for k in ('learning_rate','max_gradient_norm','value_coefficient','entropy_coefficient')),'Plan coefficients')
    require(plan['learning_rate']>0 and plan['max_gradient_norm']>0,'Optimizer coefficients')
    require(plan['collector_max_turns']==4 and plan['final_dataset_access'] is False,'Collector scope')


class Trainer:
    def __init__(self,seed,plan):
        validate_plan(plan);require(type(seed) is int and 0<=seed<2**32,'Trainer seed')
        self.seed=seed;self.plan=copy.deepcopy(plan);self.parameters=initialize(seed)
        self.optimizer=Adam(self.parameters,lr=plan['learning_rate'],max_norm=plan['max_gradient_norm'])
        self.version=self.updates=self.games=self.decisions=self.next_game=0
        self.phase='idle';self.ledger=[];self._buffer=None

    def collect(self,*,crash_after_collection=False,actor_fn=None):
        require(self.phase=='idle','Collect only at completed-update boundary');self.phase='collecting'
        size=self.plan['games_per_collection'];require(self.next_game+size<=2**32,'Game ID exhaustion')
        kw={} if actor_fn is None else {'actor_fn':actor_fn}
        buffer=collect(self.parameters,self.version,np.arange(self.next_game,self.next_game+size,dtype=np.uint32),
            environment_seed=self.seed,action_seed=self.seed,gamma=self.plan['gamma'],lam=self.plan['lambda'],**kw)
        self.games+=size;self.next_game+=size;self.decisions+=len(buffer.actor.actions)
        if crash_after_collection:raise RuntimeError('Injected collection crash; completed work must be discarded')
        self._buffer=buffer;self.phase='collected';return buffer

    def update(self,buffer,*,crash_after_minibatches=None):
        require(self.phase=='collected' and buffer is self._buffer and buffer.version==self.version,'Stale/reused/foreign behavior buffer')
        n=validate_collection(buffer);require(buffer.policy_hash==policy_hash(self.parameters),'Behavior parameter drift')
        a,t=buffer.actor,buffer.targets
        _,logs,values=predict(self.parameters,ActorInput(jnp.asarray(a.features),jnp.asarray(a.mask)))
        ratios=np.exp(np.asarray(logs)[np.arange(n),a.actions]-a.old_logp)
        max_ratio=float(np.max(np.abs(ratios-1)))
        require(max_ratio<=1e-6 and np.allclose(values,a.old_values,rtol=0,atol=1e-6),'Behavior recomputation drift')
        self.phase='updating';diagnostics=[]
        master=jax.random.fold_in(jax.random.key(self.seed,impl='threefry2x32'),0x534846)
        for epoch in range(self.plan['epochs']):
            key=jax.random.fold_in(jax.random.fold_in(master,self.version),epoch)
            order=np.asarray(jax.random.permutation(key,n))
            for start in range(0,n,self.plan['minibatch_size']):
                ix=order[start:start+self.plan['minibatch_size']]
                (_,metrics),gradients=model_loss_grad(self.parameters,jnp.asarray(a.features[ix]),jnp.asarray(a.mask[ix]),
                    jnp.asarray(a.actions[ix]),jnp.asarray(a.old_logp[ix]),jnp.asarray(t.advantages[ix]),jnp.asarray(t.returns[ix]),
                    self.plan['clip'],self.plan['value_coefficient'],self.plan['entropy_coefficient'])
                metrics={k:float(v) for k,v in metrics.items()}
                require(all(np.isfinite(v) for v in metrics.values()),'Nonfinite learner diagnostics')
                metrics['gradient_norm_before_clip']=self.optimizer.apply(self.parameters,gradients)
                diagnostics.append(metrics)
                if crash_after_minibatches==len(diagnostics):raise RuntimeError('Injected update crash')
        self.ledger.append({'version':self.version,'games':buffer.games,'decisions':n,'minibatches':len(diagnostics),'buffer_hash':buffer.content_hash})
        self.version+=1;self.updates+=1;self.phase='idle';self._buffer=None
        return {'behavior_max_ratio_error':max_ratio,'minibatches':diagnostics}
