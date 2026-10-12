"""G1 canary tasks with JAX table autodiff updates; retain the CPU toy collector."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner.trainer import Trainer as ReferenceTrainer
from pokemon_learner.core import require
from .core import Adam, loss_kernel, distribution


def table_loss(parameters,states,mask,actions,old_logp,advantages,returns,clip,value_coef,entropy_coef):
    return loss_kernel(parameters['logits'][states],parameters['values'][states],mask,actions,old_logp,advantages,returns,clip,value_coef,entropy_coef)


table_loss_grad=jax.jit(jax.value_and_grad(table_loss,has_aux=True),static_argnums=(7,8,9))


class Canary(ReferenceTrainer):
    def __init__(self,task,seed,plan):
        super().__init__(task,seed,plan)
        self.optimizer=Adam({k:jnp.asarray(v) for k,v in self.parameters.items()},lr=plan['learning_rate'],max_norm=plan['max_gradient_norm'])

    def update(self,buffer):
        require(self.phase=='collected' and buffer.version==self.version,'Stale canary buffer');self.phase='updating'
        parameters={k:jnp.asarray(v) for k,v in self.parameters.items()};diagnostics=[]
        for _ in range(self.plan['epochs']):
            order=self.shuffle_rng.permutation(len(buffer.states))
            for start in range(0,len(order),self.plan['minibatch_size']):
                ix=order[start:start+self.plan['minibatch_size']]
                (_,metrics),gradients=table_loss_grad(parameters,jnp.asarray(buffer.states[ix]),jnp.asarray(buffer.mask[ix]),jnp.asarray(buffer.actions[ix]),
                    jnp.asarray(buffer.old_logp[ix]),jnp.asarray(buffer.advantages[ix]),jnp.asarray(buffer.returns[ix]),
                    self.plan['clip'],self.plan['value_coefficient'],self.plan['entropy_coefficient'])
                metrics={k:float(v) for k,v in metrics.items()};metrics['gradient_norm_before_clip']=self.optimizer.apply(parameters,gradients)
                diagnostics.append(metrics)
        self.parameters={k:np.asarray(v).copy() for k,v in parameters.items()};self.updates+=1;self.version+=1;self.phase='idle'
        return diagnostics
