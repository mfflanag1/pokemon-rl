"""Fixed-shape, integer CPU JAX duel. No full-format simulator admission."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_slice.engine import PROFILE, require


class State(NamedTuple):
    turn: jax.Array
    hp: jax.Array
    pp: jax.Array
    winner: jax.Array  # -1 live, otherwise absolute seat
    history: jax.Array  # [turn, actor, move, crit, hp_lost, target_hp_after]
    count: jax.Array


class Chance(NamedTuple):
    crit: jax.Array  # by actor, not execution order
    roll: jax.Array
    tie: jax.Array


class PlayerView(NamedTuple):
    header: jax.Array
    history: jax.Array


def initial(batch_size, profile=PROFILE):
    require(profile == PROFILE, 'Unsupported fixed profile')
    require(type(batch_size) is int and batch_size > 0, 'Invalid batch size')
    require(jax.default_backend() == 'cpu', 'This milestone is CPU only')
    require(not jax.config.x64_enabled, 'This milestone requires x64 disabled')
    n = batch_size
    return State(jnp.ones(n, jnp.int32), jnp.full((n,2),105,jnp.int32),
                 jnp.tile(jnp.array([[56,48],[56,48]],jnp.int32),(n,1,1)),
                 jnp.full(n,-1,jnp.int32), jnp.full((n,8,6),-1,jnp.int32), jnp.zeros(n,jnp.int32))


def damage(crit, roll):
    base = jnp.where(crit,39,26) * roll // 100
    return (base * 6144 + 2047) // 4096


def _one(state, actions, chance):
    was_live = state.winner == -1
    first = jnp.where(actions[0] == actions[1], chance.tie, jnp.where(actions[0] > actions[1],0,1))
    hp, pp, history, count, winner = state.hp, state.pp, state.history, state.count, state.winner
    events = []
    for actor in (first, 1-first):
        live = (winner == -1) & was_live
        move = actions[actor]
        amount = damage(chance.crit[actor], chance.roll[actor])
        old_hp = hp[1-actor]
        new_hp = jnp.maximum(0, old_hp-amount)
        hp = hp.at[1-actor].set(jnp.where(live,new_hp,old_hp))
        pp = pp.at[actor,move].add(-live.astype(jnp.int32))
        row = jnp.array([state.turn,actor,move,chance.crit[actor],old_hp-new_hp,new_hp],jnp.int32)
        index = jnp.minimum(count,7)
        history = history.at[index].set(jnp.where(live,row,history[index]))
        count = count + live.astype(jnp.int32)
        winner = jnp.where(live & (new_hp==0), actor, winner)
        event = jnp.array([actor,move,chance.crit[actor],chance.roll[actor],amount],jnp.int32)
        events.append(jnp.where(live,event,jnp.full(5,-1,jnp.int32)))
    turn = state.turn + (was_live & (winner == -1)).astype(jnp.int32)
    return State(turn,hp,pp,winner,history,count), jnp.stack(events)


# Internal kernels require the checked host boundary (or states produced by these kernels).
step_kernel = jax.jit(jax.vmap(_one))


def validate_state(state):
    require(type(state) is State, 'Expected State')
    n = len(state.turn)
    shapes = [(n,),(n,2),(n,2,2),(n,),(n,8,6),(n,)]
    arrays = [np.asarray(x) for x in state]
    for x, shape in zip(arrays, shapes):
        require(x.shape == shape and x.dtype == np.int32, 'State shape/dtype')
    turn,hp,pp,winner,history,count = arrays
    require(np.all((turn>=1)&(turn<=4)) and np.all((hp>=0)&(hp<=105)), 'State bounds')
    require(np.all((winner>=-1)&(winner<=1)), 'Winner bounds')
    require(np.all((pp>0)&(pp<=np.array([56,48]))), 'PP bounds')
    require(np.all((count>=0)&(count<=7)), 'History count bounds')
    # Replay public history to reject unsupported imported states, including stale PP.
    replay_hp = np.full((n,2),105); replay_pp = np.tile([[56,48],[56,48]],(n,1,1))
    last_turn = np.zeros(n,dtype=int)
    rows = np.arange(n)
    for k in range(8):
        used = k<count; h=history[:,k]
        require(np.all(h[~used]==-1), 'History padding')
        if not np.any(used): continue
        r=rows[used]; t,a,m,c,lost,after = h[used].T
        require(np.all(t==k//2+1) and np.all(t<=turn[used]), 'History turn')
        require(np.all((a>=0)&(a<=1)&(m>=0)&(m<=1)&(c>=0)&(c<=1)), 'History identity')
        require(np.all(replay_hp[r]>0), 'History after terminal')
        if k:
            previous=history[used,k-1]
            same=previous[:,0]==t
            require(np.all(~same | ((a!=previous[:,1]) & (m<=previous[:,2]))), 'History action order')
        before=replay_hp[r,1-a]
        allowed=np.stack([np.minimum(before,np.asarray(damage(c.astype(bool),roll))) for roll in range(85,101)],axis=1)
        require(np.all(np.any(allowed==lost[:,None],axis=1)) and np.all(after==before-lost), 'History damage')
        replay_hp[r,1-a]=after;replay_pp[r,a,m]-=1;last_turn[r]=t
    require(np.array_equal(hp,replay_hp) and np.array_equal(pp,replay_pp), 'History state mismatch')
    expected=np.where(hp[:,0]==0,1,np.where(hp[:,1]==0,0,-1))
    require(np.array_equal(winner,expected), 'Terminal state mismatch')
    require(np.all(turn==np.where(winner==-1,last_turn+1,last_turn)), 'Turn state mismatch')
    require(np.all(np.where(winner==-1, count==2*(turn-1), (count==2*turn-1)|(count==2*turn))), 'Incomplete resolved turn')
    return n


def step(state, actions, chance):
    n = validate_state(state)
    actions = np.asarray(actions)
    require(actions.shape == (n,2) and actions.dtype.kind in 'iu' and np.all((actions>=0)&(actions<=1)), 'Unsupported action')
    require(type(chance) is Chance, 'Expected semantic chance')
    crit,roll,tie = map(np.asarray,chance)
    require(crit.shape==(n,2) and crit.dtype==np.bool_, 'Crit shape/dtype')
    require(roll.shape==(n,2) and roll.dtype.kind in 'iu' and np.all((roll>=85)&(roll<=100)), 'Roll bounds')
    require(tie.shape==(n,) and tie.dtype.kind in 'iu' and np.all((tie>=0)&(tie<=1)), 'Tie bounds')
    result, events = step_kernel(state,jnp.asarray(actions,jnp.int32),chance)
    validate_state(result)
    return result, events


def _view_one(state, side):
    other=1-side;hist=state.history
    used=jnp.arange(8)<state.count
    seen=jnp.array([jnp.any(used & (hist[:,1]==other) & (hist[:,2]==m)) for m in (0,1)],jnp.int32)
    terminal=state.winner!=-1
    win=jnp.where(terminal,jnp.where(state.winner==side,1,2),0)
    header=jnp.array([state.turn,terminal,win,state.hp[side],state.hp[other],state.pp[side,0],state.pp[side,1],seen[0],seen[1],~terminal,~terminal],jnp.int32)
    hist=hist.at[:,1].set(jnp.where(used,(hist[:,1]!=side).astype(jnp.int32),-1))
    return PlayerView(header,hist)


view_kernel = jax.jit(jax.vmap(_view_one, in_axes=(0,None)))


def actor_probe(view):
    """Fixed deterministic logits/value probe; not a trained Pokemon policy."""
    require(type(view) is PlayerView, 'Actor accepts only PlayerView')
    x=jnp.concatenate([view.header,view.history.reshape((-1,48))],axis=1).astype(jnp.float32)/105
    weights=jnp.sin(jnp.arange(59*3,dtype=jnp.float32).reshape((59,3)))
    out=x@weights
    return out[:,:2],out[:,2]
