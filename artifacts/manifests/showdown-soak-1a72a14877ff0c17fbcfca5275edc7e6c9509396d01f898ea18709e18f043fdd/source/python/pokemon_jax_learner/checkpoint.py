"""Atomic exact continuation at completed whole-episode collect/update boundaries."""
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner.checkpoint import wire,digest,file_hash,array_wire,read_array
from pokemon_learner.core import require
from .trainer import Trainer

ROOT=Path(__file__).resolve().parents[2]


def provenance():
    files=[]
    for package in ('pokemon_jax_learner','pokemon_jax','pokemon_slice','pokemon_learner','pokemon_reference'):
        files.extend(sorted((ROOT/'python'/package).glob('*.py')))
    files.extend(sorted((ROOT/'contracts/jax-learner-v1').glob('*')))
    files.extend(ROOT/p for p in ('requirements-learner.txt','requirements-jax.txt','requirements-contract.txt','contracts/slice-v1/profile.json','contracts/learner-v1/canaries.json'))
    return {'sources':{str(p.relative_to(ROOT)):file_hash(p) for p in files},
        'runtime':{'python':sys.version,'platform':platform.platform(),'backend':jax.default_backend(),'x64_enabled':jax.config.x64_enabled,
                   'byteorder':sys.byteorder,'prng':'threefry2x32',
                   'compiler':{'xla_flags':os.environ.get('XLA_FLAGS',''),
                       'default_matmul_precision':jax.config.jax_default_matmul_precision,
                       'numpy_dtype_promotion':jax.config.jax_numpy_dtype_promotion,
                       'numpy_rank_promotion':jax.config.jax_numpy_rank_promotion},
                   'packages':{p:importlib.metadata.version(p) for p in ('jax','jaxlib','numpy','scipy','ml_dtypes','opt_einsum','jsonschema','attrs','jsonschema-specifications','referencing','rpds-py')}}}


def state(trainer):
    require(trainer.phase=='idle' and trainer._buffer is None,'Only completed-update checkpoints are supported')
    return {'format':'jax-learner-checkpoint-1','mode':'completed-update-whole-episodes','phase':'idle','provenance':provenance(),
        'seed':trainer.seed,'plan':trainer.plan,'parameters':{k:array_wire(np.asarray(v)) for k,v in trainer.parameters.items()},
        'optimizer':{'step':trainer.optimizer.step,'lr':trainer.optimizer.lr,'max_norm':trainer.optimizer.max_norm,
                     'm':{k:array_wire(np.asarray(v)) for k,v in trainer.optimizer.m.items()},'v':{k:array_wire(np.asarray(v)) for k,v in trainer.optimizer.v.items()}},
        'counters':{k:getattr(trainer,k) for k in ('version','updates','games','decisions','next_game')},'ledger':copy.deepcopy(trainer.ledger),
        'random_streams':'seed + domain(environment/action/shuffle) + game_id/turn/seat or version/epoch; no mutable RNG cursor',
        'environment':{'active_games':0,'pending_choices':[],'partial_buffer':None}}


def save(path,trainer,*,interrupt_before_replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=state(trainer);doc={'body':body,'sha256':digest(body)};temporary=path.with_name(path.name+'.partial')
    require(not temporary.exists(),'Interrupted checkpoint .partial exists; preserve before saving')
    with temporary.open('xb') as stream:
        stream.write(wire(doc)+b'\n');stream.flush();os.fsync(stream.fileno())
    if interrupt_before_replace:raise RuntimeError('Injected checkpoint write crash')
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return doc['sha256']


def load(path,*,expected_plan):
    def pairs(items):
        result={}
        for k,v in items:require(k not in result,'Duplicate checkpoint key');result[k]=v
        return result
    doc=json.loads(Path(path).read_bytes(),object_pairs_hook=pairs)
    require(set(doc)=={'body','sha256'} and doc['sha256']==digest(doc['body']),'Checkpoint checksum/envelope')
    b=doc['body'];require(b['format']=='jax-learner-checkpoint-1' and b['mode']=='completed-update-whole-episodes' and b['phase']=='idle','Checkpoint mode')
    require(b['plan']==expected_plan and b['provenance']==provenance(),'Checkpoint source/runtime/plan drift')
    require(b['environment']=={'active_games':0,'pending_choices':[],'partial_buffer':None},'Live checkpoint unsupported')
    trainer=Trainer(b['seed'],expected_plan)
    for table in ('parameters','m','v'):
        target=trainer.parameters if table=='parameters' else getattr(trainer.optimizer,table)
        serialized=b['parameters'] if table=='parameters' else b['optimizer'][table]
        require(set(serialized)==set(target),'Checkpoint parameter keys')
        for k in target:target[k]=jnp.asarray(read_array(serialized[k],target[k].shape))
    require(b['optimizer']['lr']==trainer.optimizer.lr and b['optimizer']['max_norm']==trainer.optimizer.max_norm,'Optimizer configuration drift')
    step=b['optimizer']['step'];require(type(step) is int and step>=0,'Optimizer step');trainer.optimizer.step=step
    require(set(b['counters'])=={'version','updates','games','decisions','next_game'},'Checkpoint counters')
    for k,v in b['counters'].items():require(type(v) is int and v>=0,'Counter type');setattr(trainer,k,v)
    require(trainer.version==trainer.updates and trainer.games==trainer.next_game==trainer.updates*expected_plan['games_per_collection'],'Boundary counters')
    ledger=b['ledger'];require(type(ledger) is list and len(ledger)==trainer.updates,'Collection ledger')
    for i,row in enumerate(ledger):
        require(set(row)=={'version','games','decisions','minibatches','buffer_hash'},'Ledger schema')
        require(row['version']==i and row['games']==expected_plan['games_per_collection'],'Ledger order/games')
        require(type(row['decisions']) is int and 4*row['games']<=row['decisions']<=8*row['games'],'Ledger decisions')
        require(row['minibatches']==expected_plan['epochs']*((row['decisions']+expected_plan['minibatch_size']-1)//expected_plan['minibatch_size']),'Ledger optimizer steps')
        require(isinstance(row['buffer_hash'],str) and len(row['buffer_hash'])==64,'Ledger buffer seal')
    require(sum(r['decisions'] for r in ledger)==trainer.decisions and sum(r['minibatches'] for r in ledger)==step,'Ledger counters mismatch')
    require(all(np.all(np.asarray(v)>=0) for v in trainer.optimizer.v.values()),'Invalid second moment')
    trainer.ledger=ledger
    require(state(trainer)==b,'Checkpoint canonical state mismatch')
    return trainer
