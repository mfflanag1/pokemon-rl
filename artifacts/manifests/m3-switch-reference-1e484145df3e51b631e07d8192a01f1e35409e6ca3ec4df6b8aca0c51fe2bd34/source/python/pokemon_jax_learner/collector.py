"""Whole episodes under a frozen actor; targets never enter actor calls."""
from dataclasses import dataclass, fields
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax.engine import initial, step_kernel, view_kernel, PlayerView, validate_state
from pokemon_jax.chance import sample_kernel, event_key
from pokemon_learner.core import require
from pokemon_learner.checkpoint import digest
from .core import linked_gae
from .model import ActorInput, encode, act, validate_parameters


@dataclass(frozen=True)
class ActorTape:
    features: np.ndarray
    mask: np.ndarray
    actions: np.ndarray
    old_logp: np.ndarray
    old_values: np.ndarray
    game_id: np.ndarray
    side: np.ndarray
    turn: np.ndarray


@dataclass(frozen=True)
class Targets:
    rewards: np.ndarray
    terminal: np.ndarray
    next_index: np.ndarray
    trajectory: np.ndarray
    bootstrap: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


@dataclass(frozen=True)
class Collection:
    version: int
    policy_hash: str
    actor: ActorTape
    targets: Targets
    games: int
    content_hash: str


def array_hash(a):
    a=np.asarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes(order='C'))
    return h.hexdigest()


def policy_hash(parameters):
    validate_parameters(parameters)
    return digest({k:array_hash(v) for k,v in sorted(parameters.items())})


def collection_hash(version,policy,actor,targets,games):
    return digest({'version':version,'policy':policy,'games':games,
        'actor':{f.name:array_hash(getattr(actor,f.name)) for f in fields(actor)},
        'targets':{f.name:array_hash(getattr(targets,f.name)) for f in fields(targets)}})


def freeze(values,kind):
    arrays=[]
    for value in values:
        a=np.array(value,copy=True);a.flags.writeable=False;arrays.append(a)
    return kind(*arrays)


def collect(parameters,version,game_ids,*,environment_seed,action_seed,gamma=1.,lam=.95,actor_fn=act):
    require(type(version) is int and version>=0,'Behavior version')
    ids=np.asarray(game_ids)
    require(ids.ndim==1 and len(ids)>0 and ids.dtype.kind in 'iu' and np.all((ids>=0)&(ids<2**32)) and len(np.unique(ids))==len(ids),'Game ID admission')
    for seed in (environment_seed,action_seed):require(type(seed) is int and 0<=seed<2**32,'Collector seed')
    # Separate named key domains even if the two caller seeds happen to coincide.
    env_key=jax.random.fold_in(jax.random.key(environment_seed,impl='threefry2x32'),0x454e56)
    action_key=jax.random.fold_in(jax.random.key(action_seed,impl='threefry2x32'),0x414354)
    frozen={k:jnp.array(v,copy=True) for k,v in parameters.items()};ph=policy_hash(frozen)
    state=initial(len(ids));ids_device=jnp.asarray(ids,jnp.uint32);records=[]
    for _ in range(4):
        live=np.asarray(state.winner)==-1
        # Both perspectives are projected BEFORE either action is sampled/submitted.
        views=[view_kernel(state,seat) for seat in (0,1)]
        view=PlayerView(*[jnp.stack([views[0][k],views[1][k]],axis=1).reshape((-1,*views[0][k].shape[1:])) for k in (0,1)])
        obs=encode(view)
        keys=jax.vmap(lambda game,turn: jnp.stack([event_key(action_key,game,turn,seat) for seat in (0,1)]))(ids_device,state.turn.astype(jnp.uint32)).reshape(-1)
        actions,old_logp,values=actor_fn(frozen,obs,keys)
        actions=np.asarray(actions);old_logp=np.asarray(old_logp);values=np.asarray(values)
        require(actions.shape==old_logp.shape==values.shape==(2*len(ids),),'Actor output shapes')
        require(actions.dtype==np.int32 and old_logp.dtype==values.dtype==np.float32,'Actor output dtype')
        valid=np.repeat(live,2);mask=np.asarray(obs.mask)
        require(np.all((actions[valid]>=0)&(actions[valid]<2)) and np.all(mask[np.arange(len(valid))[valid],actions[valid]]) and np.all(actions[~valid]==-1),'Actor output legality')
        require(np.all(np.isfinite(old_logp)) and np.all(np.isfinite(values)),'Actor output nonfinite')
        chance=sample_kernel(env_key,ids_device,state.turn.astype(jnp.uint32))
        next_state,_=step_kernel(state,jnp.asarray(np.where(valid,actions,0).reshape(-1,2),jnp.int32),chance)
        ended=live & (np.asarray(next_state.winner)!=-1)
        rewards=(ended[:,None] & (np.asarray(next_state.winner)[:,None]==np.arange(2))).astype(np.float32).reshape(-1)
        records.append((np.asarray(obs.features)[valid],mask[valid],actions[valid],old_logp[valid],values[valid],
                        np.repeat(ids.astype(np.uint32),2)[valid],np.tile(np.arange(2,dtype=np.int32),len(ids))[valid],
                        np.repeat(np.asarray(state.turn),2)[valid],rewards[valid],np.repeat(ended,2)[valid]))
        state=next_state
    validate_state(state);require(np.all(np.asarray(state.winner)!=-1),'Whole-episode bound exceeded')
    require(policy_hash(frozen)==ph and policy_hash(parameters)==ph,'Behavior policy changed during collection')
    columns=[np.concatenate([r[k] for r in records]) for k in range(10)]
    actor=freeze(columns[:8],ActorTape);n=len(actor.actions)
    nxt=np.full(n,-1,np.int32);trajectory=np.empty(n,np.int32);last={};groups={}
    for i,(game,seat) in enumerate(zip(actor.game_id,actor.side)):
        key=(int(game),int(seat))
        if key not in groups:groups[key]=len(groups)
        trajectory[i]=groups[key]
        if key in last:nxt[last[key]]=i
        last[key]=i
    terminal=columns[9];require(np.array_equal(terminal,nxt==-1),'Terminal linkage mismatch')
    require(len(last)==2*len(ids) and np.sum(columns[8])==len(ids),'Own terminal scores')
    bootstrap=np.zeros(n,np.float32)
    adv,returns=linked_gae(columns[8],actor.old_values,nxt,bootstrap,terminal,trajectory,gamma=gamma,lam=lam)
    targets=freeze([columns[8],terminal,nxt,trajectory,bootstrap,adv,returns],Targets)
    content=collection_hash(version,ph,actor,targets,len(ids))
    return Collection(version,ph,actor,targets,len(ids),content)


def validate_collection(buffer):
    require(type(buffer) is Collection and type(buffer.actor) is ActorTape and type(buffer.targets) is Targets,'Collection type')
    require(buffer.content_hash==collection_hash(buffer.version,buffer.policy_hash,buffer.actor,buffer.targets,buffer.games),'Collection content changed')
    a,t=buffer.actor,buffer.targets;n=len(a.actions)
    require(n>0 and a.features.shape==(n,59) and a.mask.shape==(n,2),'Collection shapes')
    require(np.all(a.mask.any(axis=1)) and np.all(a.mask[np.arange(n),a.actions]),'Collection legal actions')
    require(np.array_equal(t.terminal,t.next_index==-1),'Whole-episode collection only')
    for i,j in enumerate(t.next_index):
        if j!=-1:require(j>i and a.game_id[i]==a.game_id[j] and a.side[i]==a.side[j] and a.turn[j]==a.turn[i]+1,'Same-player forward linkage')
    return n
