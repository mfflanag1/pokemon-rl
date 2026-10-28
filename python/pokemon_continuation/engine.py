"""Additive 60-turn CPU JAX port; full 128-row history, no learner."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .scalar import ROSTER, require, CAPACITY, MAX_TURNS, MAX_GROUPS, MAX_EVENTS

STATS = jnp.array([[120,69,69,85,85,65], [119,68,85,70,84,63], [115,65,60,82,80,80]], jnp.int32)
PP = jnp.array([[40,56],[40,56],[24,56]], jnp.int32)
POWER = jnp.array([45,40,60], jnp.int32)
TYPE = jnp.array([[-2,1,-1],[-1,-1,1],[1,-1,-1]], jnp.int32)


class State(NamedTuple):
    species: jax.Array
    swift: jax.Array
    hp: jax.Array
    pp: jax.Array
    active: jax.Array
    phase: jax.Array
    turn: jax.Array
    group: jax.Array
    winner: jax.Array
    seen: jax.Array
    history: jax.Array
    count: jax.Array


class Chance(NamedTuple):
    crit: jax.Array
    roll: jax.Array
    tie: jax.Array


class PlayerView(NamedTuple):
    own: jax.Array
    pp: jax.Array
    opponent: jax.Array
    context: jax.Array
    history: jax.Array
    mask: jax.Array


def initial(team_ids):
    require(jax.default_backend() == 'cpu' and not jax.config.x64_enabled, 'CPU int32/FP32 only')
    ids = np.asarray(team_ids)
    require(ids.ndim == 2 and ids.shape[1] == 2 and len(ids) > 0 and ids.dtype.kind in 'iu' and np.all((ids>=0)&(ids<24)), 'Batch team IDs')
    return initial_kernel(jnp.asarray(ids,jnp.int32))


@jax.jit
def initial_kernel(ids):
    species = jnp.array([r[0] for r in ROSTER],jnp.int32)[ids]
    swift = jnp.array([r[1] for r in ROSTER],jnp.int32)[ids]; n = len(ids)
    return State(species, swift, STATS[species,0], PP[species], jnp.zeros((n,2),jnp.int32),
                 jnp.zeros(n,jnp.int32), jnp.ones(n,jnp.int32), jnp.zeros(n,jnp.int32), jnp.full(n,-1,jnp.int32),
                 jnp.tile(jnp.array([[1,0],[1,0]],jnp.int32),(n,1,1)), jnp.full((n,CAPACITY,9),-1,jnp.int32), jnp.zeros(n,jnp.int32))


def damage(source, target, hp, move, crit, roll):
    physical = (move == 1) | (source == 0)
    attack = STATS[source,jnp.where(physical,1,3)]; defense = STATS[target,jnp.where(physical,2,4)]
    attack = jnp.where((move==0)&(3*hp<=STATS[source,0]), (attack*6144+2047)//4096, attack)
    power = jnp.where(move==0,POWER[source],40)
    value = (22*power*attack)//defense//50+2
    value = jnp.where(crit,value*3//2,value)*(100-roll)//100
    value = jnp.where(move==0,(value*6144+2047)//4096,value)
    effect = jnp.where(move==0,TYPE[source,target],0)
    value = jnp.where(effect>0,value*2,jnp.where(effect==-2,value//4,jnp.where(effect==-1,value//2,value)))
    return jnp.maximum(1,value)


def _mask(s):
    rows = jnp.arange(2); hp = s.hp[rows,s.active]
    need = (s.phase!=2)&((s.phase==0)|(hp==0))
    moves = need[:,None]&(s.phase==0)&(s.pp[rows,s.active]>0)
    switches = need[:,None]&(jnp.arange(2)[None,:]!=s.active[:,None])&(s.hp>0)
    return jnp.concatenate([moves,switches],axis=1)


mask_kernel = jax.jit(jax.vmap(_mask))


def _step(s, actions, chance):
    live = (s.winner==-1)&(s.turn<=MAX_TURNS)
    rows = jnp.arange(2); speed = STATS[s.species[rows,s.active],5]+32*s.swift[rows,s.active]
    switches = actions>=2
    first = jnp.where(switches[0]!=switches[1], switches[1].astype(jnp.int32),
                      jnp.where(speed[0]!=speed[1], (speed[1]>speed[0]).astype(jnp.int32),chance.tie))
    first = jnp.where(s.phase==1,jnp.argmax(s.hp[rows,s.active]==0),first)
    hp,pp,active,seen,history,count,winner = s.hp,s.pp,s.active,s.seen,s.history,s.count,s.winner
    events=[]
    for side in (first,1-first):
        action=actions[side]; member=active[side]; other=1-side; target=active[other]
        is_switch=action>=2; new_member=jnp.clip(action-2,0,1); move=jnp.clip(action,0,1)
        execute=live&(winner==-1)&(action>=0)&(is_switch|(hp[side,member]>0))
        do_switch=execute&is_switch; do_move=execute&~is_switch
        source_species=s.species[side,member]
        amount=damage(source_species,s.species[other,target],hp[side,member],move,chance.crit[side],chance.roll[side])
        hp=hp.at[other,target].set(jnp.where(do_move,jnp.maximum(0,hp[other,target]-amount),hp[other,target]))
        pp=pp.at[side,member,move].add(-do_move.astype(jnp.int32))
        active=active.at[side].set(jnp.where(do_switch,new_member,member))
        seen=seen.at[side,new_member].set(jnp.where(do_switch,1,seen[side,new_member]))
        member=jnp.where(is_switch,new_member,member)
        row=jnp.array([s.turn,side,jnp.where(is_switch,0,1),member,s.species[side,member],
                       jnp.where(is_switch,-1,move),jnp.where(is_switch,member,target),
                       jnp.where(is_switch,hp[side,member],hp[other,target]),
                       jnp.where(is_switch,0,chance.crit[side])],jnp.int32)
        index=jnp.minimum(count,CAPACITY-1)
        history=history.at[index].set(jnp.where(execute,row,history[index])); count+=execute.astype(jnp.int32)
        winner=jnp.where(do_move&jnp.all(hp[other]==0),side,winner)
        events.append(jnp.where(execute,row,jnp.full(9,-1,jnp.int32)))
    phase=jnp.where(winner!=-1,2,jnp.any(hp[rows,active]==0).astype(jnp.int32))
    phase=jnp.where(live,phase,s.phase)
    return State(s.species,s.swift,hp,pp,active,phase,s.turn+(live&(phase==0)).astype(jnp.int32),
                 s.group+live.astype(jnp.int32),winner,seen,history,count),jnp.stack(events)


step_kernel = jax.jit(jax.vmap(_step))


def step(state, actions, chance):
    require(type(state) is State and type(chance) is Chance, 'Versioned state/chance')
    validate_state(state)
    n=len(state.turn); actions=np.asarray(actions)
    require(actions.shape==(n,2) and actions.dtype.kind in 'iu', 'Action shape/dtype')
    crit,roll,tie=map(np.asarray,chance)
    require(crit.shape==(n,2) and crit.dtype==np.bool_, 'Crit shape/dtype')
    require(roll.shape==(n,2) and roll.dtype.kind in 'iu' and np.all((roll>=0)&(roll<16)), 'Roll bounds')
    require(tie.shape==(n,) and tie.dtype.kind in 'iu' and np.all((tie>=0)&(tie<=1)), 'Tie bounds')
    legal=np.asarray(mask_kernel(state)); live=(np.asarray(state.winner)==-1)&(np.asarray(state.turn)<=MAX_TURNS)
    for i in range(n):
        for side in range(2):
            action=int(actions[i,side]); need=live[i] and legal[i,side].any()
            require(0<=action<4 and legal[i,side,action] if need else action==-1, 'Illegal action or inactive wait')
    require(np.all(np.asarray(state.count)<=MAX_EVENTS), 'Reachable history bound')
    result,events=step_kernel(state,jnp.asarray(actions,jnp.int32),chance)
    require(np.all(np.asarray(result.count)<=MAX_EVENTS), 'History overflow')
    return result,events


def validate_state(s):
    """Reject malformed host inputs; this does not certify arbitrary states reachable."""
    require(type(s) is State, 'State schema')
    n=np.asarray(s.turn).size
    shapes=[(n,2,2),(n,2,2),(n,2,2),(n,2,2,2),(n,2),(n,),(n,),(n,),(n,),(n,2,2),(n,CAPACITY,9),(n,)]
    require(n>0 and all(np.asarray(a).dtype==np.int32 and a.shape==shape for a,shape in zip(s,shapes)), 'State shape/dtype')
    sp,sw,hp,pp,ac,ph,turn,group,winner,seen,hist,count=map(np.asarray,s)
    require(np.all((sp>=0)&(sp<3)) and np.all(sp[:,:,0]!=sp[:,:,1]), 'Species profile')
    require(np.all((sw==0)|(sw==1)) and np.all((ac==0)|(ac==1)) and np.all((seen==0)|(seen==1)), 'State flags')
    require(np.all((hp>=0)&(hp<=np.asarray(STATS)[sp,0])) and np.all((pp>=0)&(pp<=np.asarray(PP)[sp])), 'HP/PP bounds')
    require(np.all((ph>=0)&(ph<=2)) and np.all((turn>=1)&(turn<=MAX_TURNS+1)) and np.all((group>=0)&(group<=MAX_GROUPS))
            and np.all((winner>=-1)&(winner<=1)) and np.all((count>=0)&(count<=MAX_EVENTS)), 'State counters')
    require(np.all((winner!=-1)==(ph==2)) and np.all(count<=2*group), 'State counter linkage')
    require(np.all((group>=turn-1)&(group<=turn+1+(ph==2))) and np.all((turn<=MAX_TURNS)|(ph==0)), 'Turn/group horizon linkage')
    used=np.arange(CAPACITY)[None,:]<count[:,None]
    require(np.all(np.where(used,(hist[:,:,0]>=1)&(hist[:,:,0]<=turn[:,None])&(hist[:,:,1]>=0)&(hist[:,:,1]<=1)
            &(hist[:,:,2]>=0)&(hist[:,:,2]<=1)&(hist[:,:,3]>=0)&(hist[:,:,3]<=1)
            &(hist[:,:,4]>=0)&(hist[:,:,4]<3)&(hist[:,:,6]>=0)&(hist[:,:,6]<=1)
            &(hist[:,:,7]>=0)&(hist[:,:,7]<=120)&(hist[:,:,8]>=0)&(hist[:,:,8]<=1),True)), 'History entry bounds')
    move=hist[:,:,2]==1
    valid_event=np.where(move,(hist[:,:,5]>=0)&(hist[:,:,5]<=1),
                         (hist[:,:,5]==-1)&(hist[:,:,6]==hist[:,:,3])&(hist[:,:,8]==0))
    require(np.all(np.where(used,valid_event,True)), 'History event/move schema')
    require(np.all(np.where(used[:,1:],np.diff(hist[:,:,0],axis=1)>=0,True)), 'History turn order')
    rows=np.arange(n)[:,None]; sides=np.arange(2)[None,:];active_hp=hp[rows,sides,ac]
    require(np.all(seen[:,:,0]==1) and np.all(seen[rows,sides,ac]==1), 'Visible active')
    require(np.all((ph!=0)|np.all(active_hp>0,axis=1)) and np.all((ph!=1)|np.any(active_hp==0,axis=1)), 'Phase linkage')
    require(np.all(np.where(np.arange(CAPACITY)[None,:]>=count[:,None],np.all(hist==-1,axis=2),True)), 'History padding')


def _percent(hp,maximum):
    n=(100*hp+maximum-1)//maximum
    return jnp.where((n==100)&(hp<maximum),99,n)


def _view(s,side):
    other=1-side; members=jnp.arange(2); specs=s.species[side]; st=STATS[specs].at[:,5].add(32*s.swift[side])
    own=jnp.concatenate([(specs+1)[:,None],s.hp[side,:,None],st,(members==s.active[side]).astype(jnp.int32)[:,None]],axis=1)
    other_specs=s.species[other]; seen=s.seen[other].astype(bool)
    opponent=jnp.stack([jnp.where(seen,other_specs+1,0),jnp.where(seen,_percent(s.hp[other],STATS[other_specs,0]),-1),
                        (members==s.active[other]).astype(jnp.int32),jnp.where(seen,s.hp[other]==0,0)],axis=1)
    h=s.history; used=jnp.arange(CAPACITY)<s.count; actors=jnp.clip(h[:,1],0,1); affected=jnp.where(h[:,2]==0,actors,1-actors)
    species=s.species[affected,jnp.clip(h[:,6],0,1)]
    h=h.at[:,7].set(jnp.where(used&(affected!=side),_percent(h[:,7],STATS[species,0]),h[:,7]))
    h=h.at[:,1].set(jnp.where(used,(actors!=side).astype(jnp.int32),-1))
    mask=_mask(s)[side]; phase=jnp.where(s.phase==2,3,jnp.where(s.phase==0,0,jnp.where(jnp.any(mask),1,2)))
    return PlayerView(own,s.pp[side],opponent,jnp.array([s.turn,phase,s.active[side],s.active[other],s.count],jnp.int32),h,mask)


view_kernel = jax.jit(jax.vmap(_view,in_axes=(0,None)))


def unpack(state,index):
    result={k:np.asarray(v[index]).tolist() for k,v in state._asdict().items()}
    result['history']=result['history'][:result.pop('count')]
    return result
