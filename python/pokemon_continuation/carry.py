"""Live simulator carry at fresh request boundaries, independent of any learner."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner.checkpoint import digest, wire
from pokemon_switch_engine.checkpoint import sources, runtime
from . import engine
from .scalar import require, MAX_TURNS

ROOT=Path(__file__).resolve().parents[2]
PLAN=json.loads((ROOT/'contracts/continuation-v1/plan.json').read_text())
BOUNDARIES=(13,25,37,49,61)


def admission():return {'format':'continuation-admission-1','plan':PLAN,'sources':sources(),'runtime':runtime()}


def stopped(state,boundary):
    return (state.winner!=-1)|((state.phase==0)&(state.turn>=boundary))


def mask(state,boundary):
    require(type(boundary) is int and boundary in BOUNDARIES,'Window boundary')
    return engine.mask_kernel(state)&~stopped(state,boundary)[:,None,None]


@jax.jit
def step_kernel(state,actions,chance,boundary):
    live=~stopped(state,boundary)
    candidate,events=engine.step_kernel(state,actions,chance)
    after=jax.tree.map(lambda a,b:jnp.where(live.reshape((-1,)+(1,)*(a.ndim-1)),a,b),candidate,state)
    return after,jnp.where(live[:,None,None],events,-1)


def step(state,actions,chance,boundary):
    engine.validate_state(state)
    require(type(boundary) is int and boundary in BOUNDARIES,'Window boundary')
    require(np.all(np.asarray(state.turn)<=boundary),'Skipped window boundary')
    n=len(state.turn);a=np.asarray(actions);crit,roll,tie=map(np.asarray,chance)
    require(type(chance) is engine.Chance,'Continuation chance identity')
    require(a.shape==(n,2) and a.dtype.kind in 'iu','Action shape/dtype')
    require(crit.shape==(n,2) and crit.dtype==np.bool_,'Crit shape/dtype')
    require(roll.shape==(n,2) and roll.dtype.kind in 'iu' and np.all((roll>=0)&(roll<16)),'Roll bounds')
    require(tie.shape==(n,) and tie.dtype.kind in 'iu' and np.all((tie>=0)&(tie<=1)),'Tie bounds')
    legal=np.asarray(mask(state,boundary))
    for i in range(n):
        for side in range(2):
            action=int(a[i,side]);need=legal[i,side].any()
            require(0<=action<4 and legal[i,side,action] if need else action==-1,'Illegal action or stopped wait')
    result,events=step_kernel(state,jnp.asarray(a,jnp.int32),chance,boundary)
    engine.validate_state(result)
    return result,events


def encode(state,game_ids,master_seed,boundary,admitted):
    engine.validate_state(state);ids=np.asarray(game_ids)
    require(type(boundary) is int and boundary in BOUNDARIES,'Carry boundary')
    require(np.all(np.asarray(stopped(state,boundary))) and np.all(np.asarray(state.turn)<=boundary),'Fresh completed window')
    require(ids.dtype==np.uint32 and ids.shape==state.turn.shape and len(ids)<=144 and len(set(map(int,ids)))==len(ids),'Unique uint32 game IDs')
    require(type(master_seed) is int and 0<=master_seed<2**32,'Master seed')
    body={'format':'continuation-carry-1','admission_hash':digest(admitted),'boundary':boundary,
          'game_ids':ids.tolist(),'master_seed':master_seed,'state':{k:{'shape':list(v.shape),'dtype':'<i4',
          'hex':np.asarray(v).astype('<i4').tobytes().hex()} for k,v in state._asdict().items()}}
    return wire({'body':body,'sha256':digest(body)})+b'\n'


def decode(data,game_ids,master_seed,boundary,admitted):
    def pairs(items):
        result={}
        for key,value in items:require(key not in result,'Duplicate JSON key');result[key]=value
        return result
    doc=json.loads(data,object_pairs_hook=pairs)
    require(set(doc)=={'body','sha256'} and digest(doc['body'])==doc['sha256'],'Carry checksum')
    b=doc['body'];ids=np.asarray(game_ids)
    require(ids.dtype==np.uint32 and ids.ndim==1 and 0<len(ids)<=144,'Expected game IDs')
    require(set(b)=={'format','admission_hash','boundary','game_ids','master_seed','state'},'Carry fields')
    require(b['format']=='continuation-carry-1' and b['admission_hash']==digest(admitted),'Carry version/admission')
    require(b['game_ids']==ids.tolist() and b['master_seed']==master_seed and type(b['master_seed']) is int
            and b['boundary']==boundary and type(b['boundary']) is int,'Carry address binding')
    require(all(type(i) is int for i in b['game_ids']),'Carry ID types')
    template=engine.initial(np.zeros((len(ids),2),np.int32));arrays=[]
    require(set(b['state'])==set(template._fields),'State fields')
    for key,target in template._asdict().items():
        item=b['state'][key]
        require(set(item)=={'shape','dtype','hex'} and item['dtype']=='<i4' and item['shape']==list(target.shape)
                and type(item['hex']) is str and len(item['hex'])==target.size*8,'State array schema')
        arrays.append(jnp.asarray(np.frombuffer(bytes.fromhex(item['hex']),dtype='<i4').copy().reshape(target.shape)))
    state=engine.State(*arrays)
    # Reapply counter, phase, history, address and fresh-boundary admission.
    require(encode(state,ids,master_seed,boundary,admitted)==wire(doc)+b'\n','Canonical carry')
    return state


def save(path,*args):
    data=encode(*args)
    with Path(path).open('xb') as stream:stream.write(data)


def load(path,*args):return decode(Path(path).read_bytes(),*args)
