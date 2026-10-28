"""Distinct algorithm identity and immutable CPU canary checkpoint files."""
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from pokemon_learner.checkpoint import digest, wire, array_wire, read_array
from pokemon_switch_engine.checkpoint import fingerprint, runtime, sources
from pokemon_switch_engine.scalar import require
from . import learner

SHAPES = {'w': (323,32), 'b': (32,), 'pw': (32,4), 'pb': (4,), 'vw': (32,), 'vb': ()}


def validate(state):
    require(type(state) is learner.Learner, 'Learner schema')
    for table in state[:3]:
        require(type(table) is dict and set(table) == set(SHAPES), 'Parameter keys')
        for name, value in table.items():
            a = np.asarray(value)
            require(a.shape == SHAPES[name] and a.dtype == np.float32 and np.isfinite(a).all(), 'Parameter shape/dtype/finite')
    require(all(np.all(np.asarray(v) >= 0) for v in state.v.values()), 'Nonnegative variance')
    for value, dtype in zip(state[3:], (np.int32, np.int32, np.uint32)):
        require(np.asarray(value).shape == () and np.asarray(value).dtype == dtype, 'Counter schema')
    step, version, next_game = map(int, state[3:])
    require(0 <= version <= learner.PLAN['updates_per_seed'] + 1 and step == 2 * version
            and next_game == learner.BATCH * version, 'Counter linkage')


def admission():
    return {'format': 'bounded-value-admission-1', 'plan': learner.PLAN,
            'source_files': sources(), 'runtime': runtime()}


def save(path, state, seed, admitted):
    validate(state); require(type(seed) is int and seed in learner.PLAN['seeds'], 'Seed')
    body = {'format': 'bounded-value-checkpoint-1', 'admission_hash': digest(admitted), 'seed': seed,
            'tables': [{k: array_wire(np.asarray(v)) for k, v in t.items()} for t in state[:3]],
            'counters': list(map(int, state[3:])), 'live_games': 0, 'cutoff_games_reset': True}
    path = Path(path)
    # Exclusive immutable write. Any interrupted artifact stays invalid and preserved.
    with path.open('xb') as stream:
        stream.write(wire({'body': body, 'sha256': digest(body)}) + b'\n')


def load(path, seed, admitted):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate key'); result[k] = v
        return result
    doc = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)
    require(set(doc) == {'body', 'sha256'} and digest(doc['body']) == doc['sha256'], 'Checkpoint checksum')
    b = doc['body']
    require(set(b) == {'format', 'admission_hash', 'seed', 'tables', 'counters', 'live_games', 'cutoff_games_reset'}, 'Fields')
    require(b['format'] == 'bounded-value-checkpoint-1' and b['admission_hash'] == digest(admitted)
            and type(b['seed']) is int and b['seed'] == seed and seed in learner.PLAN['seeds']
            and type(b['live_games']) is int and b['live_games'] == 0 and b['cutoff_games_reset'] is True, 'Checkpoint binding')
    require(type(b['tables']) is list and len(b['tables']) == 3, 'Tables')
    tables = []
    for table in b['tables']:
        require(type(table) is dict and set(table) == set(SHAPES), 'Table keys')
        tables.append({k: jnp.asarray(read_array(v, SHAPES[k])) for k, v in table.items()})
    c = b['counters']
    require(type(c) is list and len(c) == 3 and all(type(v) is int and 0 <= v < 2**31 for v in c), 'Counters')
    state = learner.Learner(*tables, jnp.int32(c[0]), jnp.int32(c[1]), jnp.uint32(c[2]))
    validate(state)
    return state
