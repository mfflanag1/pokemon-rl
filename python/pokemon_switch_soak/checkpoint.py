"""Extended, bounded completed-update state; predecessor checkpoint stays sealed."""
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner.checkpoint import digest,wire,array_wire,read_array
from pokemon_switch_engine import learner
from pokemon_switch_engine.checkpoint import fingerprint
from pokemon_switch_engine.scalar import require
from .provenance import PLAN,read

SHAPES={'w':(323,32),'b':(32,),'pw':(32,4),'pb':(4,),'vw':(32,),'vb':()}


def counters(state):
    return dict(optimizer_step=int(state.step),updates=int(state.version),next_game=int(state.next_game))


def validate(state):
    require(type(state) is learner.Learner,'Learner schema')
    for table in state[:3]:
        require(type(table) is dict and set(table)==set(SHAPES),'Parameter keys')
        for k,v in table.items():
            a=np.asarray(v);require(a.shape==SHAPES[k] and a.dtype==np.float32 and np.isfinite(a).all(),'Parameter shape/dtype/finite')
    require(all(np.all(np.asarray(v)>=0) for v in state.v.values()),'Nonnegative second moments')
    for a,dtype in zip(state[3:],(np.int32,np.int32,np.uint32)):
        require(np.asarray(a).shape==() and np.asarray(a).dtype==dtype,'Counter shape/dtype')
    c=counters(state)
    require(0<=c['updates']<=PLAN['maximum_updates'] and c['optimizer_step']==2*c['updates']
        and c['next_game']==32*c['updates'],'Nonwrapping counter linkage')


def body(state,binding):
    validate(state)
    return {'format':'switch-soak-checkpoint-1','phase':'completed-update-capped-games','seed':PLAN['seed'],
        'binding':binding,'tables':[{k:array_wire(np.asarray(v)) for k,v in t.items()} for t in state[:3]],
        'counters':counters(state),'environment':{'live_games':0,'partial_buffer':None,'cutoff_games_reset':True}}


def save(path,state,binding,*,interrupt=False):
    path=Path(path);data=body(state,binding);temp=path.with_name(path.name+'.partial')
    require(not path.exists() and not temp.exists(),'Checkpoint/partial already exists; preserve evidence')
    with temp.open('xb') as stream:
        stream.write(wire({'body':data,'sha256':digest(data)})+b'\n');stream.flush();os.fsync(stream.fileno())
    if interrupt:raise RuntimeError('Injected interrupted checkpoint')
    os.replace(temp,path);fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def load(path,binding):
    doc=read(path);require(set(doc)=={'body','sha256'} and digest(doc['body'])==doc['sha256'],'Checkpoint checksum')
    b=doc['body'];template=body(learner.start(PLAN['seed']),binding)
    require(set(b)==set(template),'Checkpoint fields')
    for k in set(b)-{'tables','counters'}:require(b[k]==template[k],'Checkpoint binding: '+k)
    c=b['counters'];require(type(c) is dict and set(c)=={'updates','optimizer_step','next_game'},'Counter fields')
    require(all(type(v) is int and 0<=v<=2**31-1 for v in c.values()),'Counter serialization')
    require(type(b['tables']) is list and len(b['tables'])==3,'Tables')
    tables=[]
    for t in b['tables']:
        require(type(t) is dict and set(t)==set(SHAPES),'Table keys')
        tables.append({k:jnp.asarray(read_array(t[k],shape)) for k,shape in SHAPES.items()})
    state=learner.Learner(*tables,jnp.int32(c['optimizer_step']),jnp.int32(c['updates']),jnp.uint32(c['next_game']))
    validate(state);return state


def transaction(state,*,fault=None):
    """Return a new state only after every collection/update check passes."""
    validate(state);require(int(state.version)<PLAN['maximum_updates'],'Qualification counter budget exhausted')
    buffer,final=learner.collect(state.parameters,state.next_game,jnp.uint32(PLAN['seed']))
    jax.block_until_ready((buffer,final));learner.validate_buffer(buffer,final)
    selected=np.asarray(buffer.old_logp)[np.asarray(buffer.valid)]
    require(np.min(selected)>=-48*np.log(2)-1e-5 and np.max(selected)<=1e-6,'Categorical behavior support')
    if fault=='collect':raise RuntimeError('Injected collection discard')
    candidate,metrics=learner.update(state,buffer);jax.block_until_ready((candidate,metrics));validate(candidate)
    require(int(candidate.version)==int(state.version)+1,'Exactly one published update')
    require(all(np.asarray(v).shape==(2,) and np.isfinite(np.asarray(v)).all() for v in metrics.values()),'Finite epoch metrics')
    if fault=='update':raise RuntimeError('Injected update discard')
    return candidate,buffer,final,metrics


def result_fingerprints(result):
    return {k:fingerprint(v) for k,v in zip(('learner','buffer','final_state','metrics'),result)}
