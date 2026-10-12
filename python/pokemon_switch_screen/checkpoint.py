"""Seed-bound screen checkpoints; unchanged M3 collect/update kernels."""
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner.checkpoint import digest,wire,array_wire,read_array
from pokemon_switch_engine import learner
from pokemon_switch_engine.checkpoint import fingerprint
from pokemon_switch_engine.scalar import require
from pokemon_switch_soak.checkpoint import SHAPES,validate as validate_state,result_fingerprints
from .provenance import PLAN,read


def validate(state):
    validate_state(state);require(int(state.version)<=PLAN['checkpoint_updates'][-1]+1,'Screen update budget')


def transaction(state,seed):
    validate(state);require(type(seed) is int and seed in PLAN['training_seeds'],'Training seed')
    require(int(state.version)<=PLAN['checkpoint_updates'][-1],'One endpoint audit beyond training only')
    buffer,final=learner.collect(state.parameters,state.next_game,jnp.uint32(seed));jax.block_until_ready((buffer,final))
    learner.validate_buffer(buffer,final)
    candidate,metrics=learner.update(state,buffer);jax.block_until_ready((candidate,metrics));validate(candidate)
    require(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(metrics)),'Finite learner metrics')
    return candidate,buffer,final,metrics


def save(path,state,seed,admitted):
    validate(state);path=Path(path);temp=path.with_name(path.name+'.partial')
    require(not path.exists() and not temp.exists(),'Existing checkpoint/partial')
    body={'format':'switch-screen-checkpoint-1','seed':seed,'admission_hash':digest(admitted),
          'tables':[{k:array_wire(np.asarray(v)) for k,v in t.items()} for t in state[:3]],
          'counters':list(map(int,state[3:])),'live_games':0,'cutoff_games_reset':True}
    with temp.open('xb') as f:
        f.write(wire({'body':body,'sha256':digest(body)})+b'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def load(path,seed,admitted):
    d=read(path);require(set(d)=={'body','sha256'} and digest(d['body'])==d['sha256'],'Checkpoint checksum');b=d['body']
    require(set(b)=={'format','seed','admission_hash','tables','counters','live_games','cutoff_games_reset'} and
            b['format']=='switch-screen-checkpoint-1' and type(b['seed']) is int and b['seed']==seed
            and b['admission_hash']==digest(admitted) and type(b['live_games']) is int and b['live_games']==0
            and b['cutoff_games_reset'] is True,'Checkpoint binding')
    c=b['counters'];require(type(c) is list and len(c)==3 and all(type(v) is int and 0<=v<2**31 for v in c),'Counters')
    require(type(b['tables']) is list and len(b['tables'])==3,'Tables');tables=[]
    for t in b['tables']:
        require(type(t) is dict and set(t)==set(SHAPES),'Table fields')
        tables.append({k:jnp.asarray(read_array(t[k],shape)) for k,shape in SHAPES.items()})
    s=learner.Learner(*tables,jnp.int32(c[0]),jnp.int32(c[1]),jnp.uint32(c[2]));validate(s);return s
