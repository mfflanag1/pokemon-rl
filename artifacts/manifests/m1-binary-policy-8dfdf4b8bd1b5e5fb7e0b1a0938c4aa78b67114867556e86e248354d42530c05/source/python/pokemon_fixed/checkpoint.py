"""Bounded checkpoint state: no per-update ledger growing during the CPU soak.

Only completed whole-episode updates can be published. A failed collection/update
must load the last durable boundary; no live-game or partial optimizer continuation.
"""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax_learner.checkpoint import provenance as prior_provenance
from pokemon_jax_learner.model import validate_parameters
from pokemon_learner.checkpoint import wire, digest, file_hash, array_wire, read_array
from pokemon_learner.core import require
from .loop import ROOT, Learner, start, plan, check_report


def provenance():
    result = prior_provenance()
    for path in sorted((ROOT / 'python/pokemon_fixed').glob('*.py')) + sorted((ROOT / 'contracts/fixed-loop-v1').glob('*')):
        result['sources'][str(path.relative_to(ROOT))] = file_hash(path)
    return result


class Runner:
    def __init__(self, seed, executable):
        require(type(seed) is int and 0 <= seed < 2**32, 'Seed range')
        self.seed, self.executable, self.plan = seed, executable, executable.plan
        self.learner = start(seed)
        self.decisions = self.executed_moves = 0
        self.phase = 'idle'

    def run(self, *, fault=None):
        require(self.phase == 'idle', 'Uncommitted work: restore a boundary checkpoint')
        require(fault in (None, 'after_collect', 'after_update'), 'Unknown fault')
        require(int(self.learner.version) < self.plan['maximum_updates'], 'Counter budget exhausted')
        self.phase = 'collect-update'
        if fault == 'after_collect':
            buffer, state = self.executable.compiled['collect'](self.learner.parameters, self.learner.next_game, jnp.uint32(self.seed))
            jax.block_until_ready((buffer, state))
            raise RuntimeError('Injected failure after collection before update/commit')
        candidate, report = self.executable(self.learner, self.seed)
        r = check_report(report)  # includes device synchronization
        if fault == 'after_update': raise RuntimeError('Injected failure after update before commit')
        self.learner = candidate
        self.decisions += int(r['decisions'])
        self.executed_moves += int(r['executed_moves'])
        self.phase = 'idle'
        return r


def state(runner):
    require(runner.phase == 'idle', 'Only completed-update checkpoints supported')
    l = runner.learner
    return {'format': 'fixed-loop-checkpoint-1', 'mode': 'completed-update-whole-episodes',
            'provenance': provenance(), 'plan': runner.plan, 'seed': runner.seed,
            'parameters': {k: array_wire(np.asarray(v)) for k, v in l.parameters.items()},
            'm': {k: array_wire(np.asarray(v)) for k, v in l.m.items()},
            'v': {k: array_wire(np.asarray(v)) for k, v in l.v.items()},
            'step': int(l.step), 'version': int(l.version), 'next_game': int(l.next_game),
            'decisions': runner.decisions, 'executed_moves': runner.executed_moves,
            'environment': {'active_games': 0, 'partial_buffer': None},
            'random_streams': 'Threefry(seed); ENV/game/turn/role; ACT/game/turn/seat; SHF/version/epoch; no mutable cursor'}


def save(path, runner, *, interrupt_before_replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = state(runner)
    doc = {'body': body, 'sha256': digest(body)}
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as stream:
        stream.write(wire(doc) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    if interrupt_before_replace: raise RuntimeError('Injected interrupted checkpoint publication')
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return doc['sha256']


def load(path, executable):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'Duplicate checkpoint key')
            result[k] = v
        return result
    doc = json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
    require(set(doc) == {'body', 'sha256'} and doc['sha256'] == digest(doc['body']), 'Checkpoint checksum')
    b = doc['body']
    require(b['format'] == 'fixed-loop-checkpoint-1' and b['mode'] == 'completed-update-whole-episodes', 'Checkpoint mode')
    require(b['plan'] == executable.plan == plan() and b['provenance'] == provenance(), 'Source/runtime/plan drift')
    runner = Runner(b['seed'], executable)
    arrays = {}
    for name in ('parameters', 'm', 'v'):
        require(set(b[name]) == set(runner.learner.parameters), 'Array keys')
        arrays[name] = {k: jnp.asarray(read_array(b[name][k], a.shape)) for k, a in runner.learner.parameters.items()}
        validate_parameters(arrays[name])
    require(all(np.all(np.asarray(a) >= 0) for a in arrays['v'].values()), 'Negative second moment')
    for key in ('step', 'version', 'next_game', 'decisions', 'executed_moves'):
        require(type(b[key]) is int and b[key] >= 0, 'Counter type')
    updates = b['version']
    require(updates <= executable.plan['maximum_updates'] and b['next_game'] == 64 * updates, 'Game/version counter')
    # Each game lasts 2..4 turns: 256..512 decisions, 2..4 nonempty minibatches/epoch.
    require(4 * updates <= b['step'] <= 8 * updates and b['step'] % 2 == 0, 'Optimizer counter')
    require(256 * updates <= b['decisions'] <= 512 * updates and b['decisions'] % 2 == 0, 'Decision counter')
    require(b['decisions'] - 64 * updates <= b['executed_moves'] <= b['decisions'], 'Executed move counter')
    runner.learner = Learner(arrays['parameters'], arrays['m'], arrays['v'], jnp.int32(b['step']),
                             jnp.int32(updates), jnp.uint32(b['next_game']))
    runner.decisions, runner.executed_moves = b['decisions'], b['executed_moves']
    require(state(runner) == b, 'Noncanonical checkpoint or unsupported live environment')
    return runner
