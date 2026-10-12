"""Atomic completed-update checkpoints; capped games are explicitly reset."""
import json
import os
import platform
import runpy
import sys
from pathlib import Path
import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
from pokemon_learner.checkpoint import wire,digest,array_wire,read_array
from . import learner
from .scalar import require

ROOT=Path(__file__).resolve().parents[2]


def sources():return runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']()


def runtime():
    require(jax.default_backend()=='cpu' and not jax.config.x64_enabled,'CPU FP32 runtime')
    require(str(jax.config.jax_default_prng_impl)=='threefry2x32','Threefry event keys required')
    return {'python':sys.version,'jax':jax.__version__,'jaxlib':jaxlib.__version__,'numpy':np.__version__,
            'platform':platform.platform(),'machine':platform.machine(),'byteorder':sys.byteorder,
            'backend':jax.default_backend(),'x64':bool(jax.config.x64_enabled),'prng':str(jax.config.jax_default_prng_impl)}


def validate(state):
    require(type(state) is learner.Learner,'Learner schema')
    shapes={'w':(323,32),'b':(32,),'pw':(32,4),'pb':(4,),'vw':(32,),'vb':()}
    for table in state[:3]:
        require(type(table) is dict and set(table)==set(shapes),'Parameter keys')
        for key,value in table.items():
            a=np.asarray(value);require(a.shape==shapes[key] and a.dtype==np.float32 and np.isfinite(a).all(),'Parameter schema/finite')
    require(all(np.all(np.asarray(a)>=0) for a in state.v.values()),'Nonnegative second moments')
    for a,dtype in zip(state[3:],(np.int32,np.int32,np.uint32)):
        require(np.asarray(a).shape==() and np.asarray(a).dtype==dtype,'Counter schema')
    step,version,next_game=map(int,state[3:])
    require(0<=version<=learner.CONFIG['updates'] and step==version*learner.CONFIG['epochs']
            and next_game==version*learner.BATCH,'Counter linkage')


def fingerprint(tree):
    leaves,structure=jax.tree.flatten(tree)
    return digest({'structure':str(structure),'arrays':[{'shape':list(np.asarray(x).shape),'dtype':str(np.asarray(x).dtype),
        'hex':np.asarray(x).tobytes().hex()} for x in leaves]})


def body(state,seed):
    validate(state);require(type(seed) is int and 0<=seed<2**32,'Seed')
    return {'format':'switch-checkpoint-1','phase':'completed-update-capped-games','seed':seed,
        'plan':learner.PLAN,'sources':sources(),'runtime':runtime(),
        'tables':[{k:array_wire(np.asarray(v)) for k,v in t.items()} for t in state[:3]],
        'counters':list(map(int,state[3:])),
        'environment':{'live_games':0,'partial_buffer':None,'cutoff_games_reset':True}}


def save(path,state,seed,*,interrupt=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=body(state,seed)
    document={'body':data,'sha256':digest(data)};temp=path.with_name(path.name+'.partial')
    require(not temp.exists(),'Interrupted .partial exists; preserve it under another name before saving')
    with temp.open('xb') as stream:
        stream.write(wire(document)+b'\n');stream.flush();os.fsync(stream.fileno())
    if interrupt:raise RuntimeError('Injected interrupted checkpoint')
    os.replace(temp,path);fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return document['sha256']


def load(path,seed):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate checkpoint key');out[k]=v
        return out
    doc=json.loads(Path(path).read_bytes(),object_pairs_hook=pairs)
    require(set(doc)=={'body','sha256'} and digest(doc['body'])==doc['sha256'],'Checkpoint checksum')
    b=doc['body'];template=body(learner.start(seed),seed)
    require(set(b)==set(template),'Checkpoint fields')
    for key in set(template)-{'tables','counters'}:require(b[key]==template[key],'Checkpoint binding: '+key)
    require(type(b['tables']) is list and len(b['tables'])==3,'Checkpoint tables')
    require(type(b['counters']) is list and len(b['counters'])==3 and all(type(v) is int and 0<=v<=2**31-1 for v in b['counters']),'Counters')
    shapes={k:np.asarray(v).shape for k,v in learner.start(seed).parameters.items()};tables=[]
    for table in b['tables']:
        require(set(table)==set(shapes),'Checkpoint keys')
        tables.append({k:jnp.asarray(read_array(table[k],shape)) for k,shape in shapes.items()})
    state=learner.Learner(*tables,jnp.int32(b['counters'][0]),jnp.int32(b['counters'][1]),jnp.uint32(b['counters'][2]))
    validate(state);return state


def transaction(state,seed,*,fault=None):
    """Caller publishes the returned state only on success; no in-place mutation."""
    validate(state);runtime()
    require(int(state.version)<learner.CONFIG['updates'],'Fixed smoke budget exhausted')
    require(type(seed) is int and 0<=seed<2**32,'Seed')
    buffer,final=learner.collect(state.parameters,state.next_game,jnp.uint32(seed));jax.block_until_ready((buffer,final))
    learner.validate_buffer(buffer,final)
    if fault=='collect':raise RuntimeError('Injected collection rollback')
    candidate,metrics=learner.update(state,buffer);jax.block_until_ready((candidate,metrics))
    validate(candidate);require(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(metrics)),'Finite metrics')
    if fault=='update':raise RuntimeError('Injected update rollback')
    return candidate,buffer,final,metrics
