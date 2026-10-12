"""Fixed-shape capped-game collection and categorical PPO, observation-only actor."""
from typing import NamedTuple
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_fixed.math import adam_step
from . import engine, policy, chance
from .scalar import require

PLAN=json.loads((Path(__file__).resolve().parents[2]/'contracts/switch-engine-v1/plan.json').read_text())
CONFIG=PLAN['learner']; BATCH=CONFIG['batch_games']; LANES=BATCH*2; GROUPS=PLAN['max_groups']


class Learner(NamedTuple):
    parameters: dict
    m: dict
    v: dict
    step: jax.Array
    version: jax.Array
    next_game: jax.Array


class Buffer(NamedTuple):
    features: jax.Array
    mask: jax.Array
    actions: jax.Array
    old_logp: jax.Array
    values: jax.Array
    valid: jax.Array
    rewards: jax.Array
    terminal: jax.Array
    next_index: jax.Array
    bootstrap: jax.Array
    advantages: jax.Array
    returns: jax.Array


def start(seed):
    p=policy.initialize(seed)
    return Learner(p,jax.tree.map(jnp.zeros_like,p),jax.tree.map(jnp.zeros_like,p),jnp.int32(0),jnp.int32(0),jnp.uint32(0))


def observe(state):
    views=[engine.view_kernel(state,side) for side in (0,1)]
    return policy.encode(engine.PlayerView(*[jnp.stack([views[0][k],views[1][k]],axis=1).reshape((-1,*views[0][k].shape[1:])) for k in range(6)]))


def targets(values,valid,winners,bootstrap):
    """Skip wait/padding rows; carry next same-player value/advantage across gaps."""
    lanes=values.shape[1];ended=jnp.repeat(winners!=-1,2);won=jnp.repeat(winners,2)==jnp.tile(jnp.arange(2),len(winners))
    def one(carry,row):
        next_v,next_adv,next_i=carry
        v,live,index=row;last=next_i<0;terminal=live&last&ended
        reward=(terminal&won).astype(jnp.float32)
        boundary=jnp.where(live&last&~ended,bootstrap,0)
        vnext=jnp.where(terminal,0,next_v);anext=jnp.where(terminal,0,next_adv)
        advantage=reward+CONFIG['gamma']*vnext-v+CONFIG['gamma']*CONFIG['lambda']*anext
        advantage=jnp.where(live,advantage,0)
        outputs=(reward,terminal,jnp.where(live,next_i,-1),boundary,advantage,jnp.where(live,advantage+v,0))
        return (jnp.where(live,v,next_v),jnp.where(live,advantage,next_adv),jnp.where(live,index,next_i)),outputs
    indices=jnp.arange(values.size,dtype=jnp.int32).reshape(values.shape)
    _,result=jax.lax.scan(one,(bootstrap,jnp.zeros(lanes,jnp.float32),jnp.full(lanes,-1,jnp.int32)),(values,valid,indices),reverse=True)
    return result


@jax.jit
def collect(parameters,next_game,seed):
    ids=next_game+jnp.arange(BATCH,dtype=jnp.uint32)
    teams=jnp.stack([(ids+seed%24)%24,(ids//24+(seed//24)%24)%24],axis=1).astype(jnp.int32)
    state=engine.initial_kernel(teams);master=jax.random.key(seed)
    actor_master=jax.random.fold_in(master,0x41435434)
    def one(state,_):
        obs=observe(state);live=jnp.repeat((state.winner==-1)&(state.turn<=12),2)
        obs=policy.ActorInput(jnp.where(live[:,None],obs.features,0),obs.mask&live[:,None]);valid=obs.mask.any(1)
        def keys(game,group):
            base=jax.random.fold_in(jax.random.fold_in(actor_master,game),group)
            return jnp.stack([jax.random.fold_in(base,side) for side in (0,1)])
        actor_keys=jax.vmap(keys)(ids,state.group.astype(jnp.uint32)).reshape(-1)
        actions,logp,values=policy.act(parameters,obs,actor_keys)
        after,_=engine.step_kernel(state,actions.reshape(BATCH,2),chance.sample(master,ids,state.group.astype(jnp.uint32)))
        return after,(obs.features,obs.mask,actions,logp,values,valid)
    final,rows=jax.lax.scan(one,state,xs=None,length=GROUPS)
    _,boot=policy.forward(parameters,observe(final));boot=jnp.where(jnp.repeat(final.winner==-1,2),boot,0)
    target=targets(rows[4],rows[5],final.winner,boot)
    flatten=lambda x:x.reshape((-1,*x.shape[2:]))
    return Buffer(*[flatten(x) for x in (*rows,*target)]),final


def objective(parameters,buffer):
    obs=policy.ActorInput(jnp.where(buffer.valid[:,None],buffer.features,0),buffer.mask)
    logits,values=policy.forward(parameters,obs)
    return policy.loss(logits,values,buffer.mask,buffer.actions,buffer.old_logp,buffer.advantages,buffer.returns,buffer.valid,
                       CONFIG['clip'],CONFIG['value_coefficient'],CONFIG['entropy'])


@jax.jit
def update(learner,buffer):
    def epoch(carry,_):
        p,m,v,step=carry
        (_,metrics),grad=jax.value_and_grad(objective,has_aux=True)(p,buffer)
        p,m,v,step,norm=adam_step(p,grad,m,v,step,lr=CONFIG['learning_rate'])
        return (p,m,v,step),{**metrics,'gradient_norm':norm}
    (p,m,v,step),metrics=jax.lax.scan(epoch,(learner.parameters,learner.m,learner.v,learner.step),xs=None,length=CONFIG['epochs'])
    return Learner(p,m,v,step,learner.version+1,learner.next_game+BATCH),metrics


def run(learner,seed):
    from .checkpoint import transaction
    return transaction(learner,seed)


def validate_buffer(buffer,final):
    from pokemon_learner.core import linked_gae
    require(type(buffer) is Buffer,'Buffer schema')
    n=GROUPS*LANES
    for key,value in buffer._asdict().items():
        shape=(n,323) if key=='features' else (n,4) if key=='mask' else (n,)
        dtype=np.bool_ if key in ('mask','valid','terminal') else np.int32 if key in ('actions','next_index') else np.float32
        require(value.shape==shape and np.asarray(value).dtype==dtype,'Buffer shape/dtype: '+key)
    require(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(buffer)),'Nonfinite collection')
    valid=np.asarray(buffer.valid);mask=np.asarray(buffer.mask);actions=np.asarray(buffer.actions)
    require(valid.any() and np.array_equal(valid,mask.any(1)),'Decision mask/valid linkage')
    require(np.all(actions[~valid]==-1) and np.all((actions[valid]>=0)&(actions[valid]<4)),'Action bounds')
    require(np.all(mask[np.flatnonzero(valid),actions[valid]]),'Known-forbidden action')
    require(np.all((np.asarray(final.winner)!=-1)|((np.asarray(final.turn)==13)&(np.asarray(final.phase)==0))),'Unfinished capped games')
    # Independently compare the padded scan with the sealed NumPy forward-link GAE.
    selected=np.flatnonzero(valid);inverse=np.full(len(valid),-1);inverse[selected]=np.arange(len(selected))
    nxt=np.asarray(buffer.next_index)[valid];mapped=np.where(nxt>=0,inverse[np.maximum(nxt,0)],-1)
    trajectory=selected%LANES
    advantages,returns=linked_gae(np.asarray(buffer.rewards)[valid],np.asarray(buffer.values)[valid],mapped,
        np.asarray(buffer.bootstrap)[valid],np.asarray(buffer.terminal)[valid],trajectory,gamma=CONFIG['gamma'],lam=CONFIG['lambda'])
    require(np.allclose(advantages,np.asarray(buffer.advantages)[valid],rtol=2e-6,atol=2e-6) and
            np.allclose(returns,np.asarray(buffer.returns)[valid],rtol=2e-6,atol=2e-6),'Independent GAE parity')
    terminal=np.asarray(buffer.terminal);reward=np.asarray(buffer.rewards)
    require(np.all(reward[~terminal]==0) and np.sum(reward)==np.sum(np.asarray(final.winner)!=-1),'Terminal reward accounting')
