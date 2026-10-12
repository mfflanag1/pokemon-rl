"""Atomic completed-update checkpoint, exact on the declared CPU/runtime only.

Canaries collect whole episodes, so the boundary has no live games, requests,
private histories, partial buffers or minibatch cursors. Other modes reject.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
from .core import require, finite, F32
from .trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


def wire(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value): return hashlib.sha256(wire(value)).hexdigest()
def file_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    paths = sorted((ROOT/'python/pokemon_learner').glob('*.py')) + [ROOT/'contracts/learner-v1/canaries.json', ROOT/'requirements-learner.txt']
    return {'sources': {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
            'runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                        'machine': platform.machine(), 'byteorder': sys.byteorder}}


def array_wire(a):
    require(a.dtype == F32, 'Checkpoint array dtype')
    finite(a)
    return {'dtype': '<f4', 'shape': list(a.shape), 'hex': a.astype('<f4', copy=False).tobytes().hex()}


def read_array(doc, shape):
    require(set(doc) == {'dtype','shape','hex'} and doc['dtype'] == '<f4' and doc['shape'] == list(shape), 'Checkpoint array schema')
    a = np.frombuffer(bytes.fromhex(doc['hex']), dtype='<f4').copy().reshape(shape)
    finite(a)
    return a.astype(F32, copy=False)


def state(trainer):
    require(trainer.phase == 'idle', 'Only completed-update checkpoints are supported')
    return {'version': 'g1-checkpoint-1', 'mode': 'completed-update-whole-episodes', 'provenance': provenance(),
        'task': trainer.task, 'seed': trainer.seed, 'plan': trainer.plan,
        'parameters': {k: array_wire(v) for k,v in trainer.parameters.items()},
        'optimizer': {'step': trainer.optimizer.step, 'lr': trainer.optimizer.lr, 'max_norm': trainer.optimizer.max_norm,
            'm': {k: array_wire(v) for k,v in trainer.optimizer.m.items()},
            'v': {k: array_wire(v) for k,v in trainer.optimizer.v.items()}},
        'rng': {'sampling': trainer.sample_rng.bit_generator.state, 'shuffle': trainer.shuffle_rng.bit_generator.state},
        'counters': {k: getattr(trainer,k) for k in ['version','updates','decisions','episodes']},
        'environment': {'active_episodes': 0, 'pending_requests': [], 'partial_buffer': None}, 'phase': 'idle'}


def save(path, trainer, *, interrupt_before_replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    body = state(trainer); document = {'body': body, 'sha256': digest(body)}
    # A stale partial file is evidence of an interrupted write and is not a checkpoint.
    temp = path.with_name(path.name + '.partial')
    require(not temp.exists(), 'Interrupted checkpoint .partial exists; preserve it under another name before saving')
    with temp.open('xb') as stream:
        stream.write(wire(document) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    if interrupt_before_replace: raise RuntimeError('Injected checkpoint write crash')
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return document['sha256']


def load(path, *, expected_plan):
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out, 'Duplicate checkpoint key'); out[k]=v
        return out
    document = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)
    require(set(document) == {'body','sha256'}, 'Checkpoint envelope')
    body = document['body']; require(document['sha256'] == digest(body), 'Checkpoint checksum')
    require(body['version'] == 'g1-checkpoint-1' and body['mode'] == 'completed-update-whole-episodes' and body['phase'] == 'idle', 'Checkpoint mode')
    require(body['provenance'] == provenance() and body['plan'] == expected_plan, 'Checkpoint source/runtime/plan drift')
    require(body['environment'] == {'active_episodes': 0, 'pending_requests': [], 'partial_buffer': None}, 'Unsupported live checkpoint')
    trainer = Trainer(body['task'], body['seed'], expected_plan)
    for table in ['parameters', 'm', 'v']:
        target = trainer.parameters if table == 'parameters' else getattr(trainer.optimizer, table)
        serialized = body['parameters'] if table == 'parameters' else body['optimizer'][table]
        require(set(serialized) == set(target), 'Checkpoint parameter keys')
        for k in target: target[k] = read_array(serialized[k], target[k].shape)
    for k in ['step','lr','max_norm']: setattr(trainer.optimizer, k, body['optimizer'][k])
    require(trainer.optimizer.lr == expected_plan['learning_rate'] and trainer.optimizer.max_norm == expected_plan['max_gradient_norm'], 'Optimizer configuration drift')
    require(set(body['counters']) == {'version','updates','decisions','episodes'}, 'Checkpoint counters')
    for k,v in body['counters'].items():
        require(type(v) is int and v >= 0, 'Invalid checkpoint counter'); setattr(trainer,k,v)
    per_update = expected_plan[trainer.task]['episodes_per_update']
    decisions_per_episode = 1 if trainer.task == 'bandit' else 6
    require(trainer.version == trainer.updates and trainer.episodes == trainer.updates * per_update and
            trainer.decisions == trainer.episodes * decisions_per_episode, 'Inconsistent boundary counters')
    steps = trainer.updates * expected_plan['epochs'] * ((per_update*decisions_per_episode + expected_plan['minibatch_size']-1)//expected_plan['minibatch_size'])
    require(type(trainer.optimizer.step) is int and trainer.optimizer.step == steps, 'Optimizer step mismatch')
    trainer.sample_rng.bit_generator.state = body['rng']['sampling']
    trainer.shuffle_rng.bit_generator.state = body['rng']['shuffle']
    return trainer
