"""Immutable completed-window checkpoints; live games and addressing are retained."""
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from pokemon_continuation import engine, carry
from pokemon_learner.checkpoint import digest, wire, array_wire, read_array
from pokemon_switch_engine.checkpoint import fingerprint, runtime, sources
from pokemon_switch_engine.scalar import require
from . import learner
from pokemon_live import policy


def admission():
    return {'format': 'live-training-admission-1', 'plan': learner.PLAN, 'sources': sources(), 'runtime': runtime()}


def validate(s, *, audit_only=False):
    require(type(s) is learner.Session, 'Live Session schema')
    for table in s[:3]:
        require(type(table) is dict and set(table) == set(policy.SHAPES), 'Parameter keys')
        for name, value in table.items():
            a = np.asarray(value)
            require(a.shape == policy.SHAPES[name] and a.dtype == np.float32 and np.isfinite(a).all(), 'Parameter schema/finite')
    require(all(np.all(np.asarray(x) >= 0) for x in s.v.values()), 'Nonnegative Adam variance')
    for a, dtype in zip(s[3:8], (np.int32, np.int32, np.int32, np.uint32, np.uint32)):
        require(np.asarray(a).shape == () and np.asarray(a).dtype == dtype, 'Counter dtype/shape')
    step, windows, updates, next_game, seed = map(int, s[3:8])
    require(seed in learner.PLAN['seeds'], 'Seed roster')
    require(0 <= windows <= learner.PLAN['windows_per_seed'] + int(audit_only) and 0 <= updates <= windows
            and step == updates * learner.PLAN['epochs'], 'Window/update/Adam linkage')
    allocated = learner.BATCH * (1 + max(windows - 1, 0) // 5)
    require(next_game == allocated, 'Game allocation linkage')
    ids = np.asarray(s.ids)
    require(ids.shape == (learner.BATCH,) and ids.dtype == np.uint32 and
            np.array_equal(ids, np.arange(next_game - learner.BATCH, next_game, dtype=np.uint32)), 'Cohort game IDs')
    engine.validate_state(s.environment)
    initial = engine.initial_kernel(learner.teams(s.ids, s.seed))
    require(len(s.environment.turn) == learner.BATCH and all(np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(s.environment[:2], initial[:2])), 'Cohort team/hidden-set identity')
    if windows == 0:
        require(fingerprint(s.environment) == fingerprint(initial), 'Initial environment')
    else:
        boundary = learner.PLAN['boundaries'][(windows - 1) % 5]
        require(np.all(carry.stopped(s.environment, boundary)) and np.all(np.asarray(s.environment.turn) <= boundary), 'Completed window boundary')


def body(s, admitted):
    validate(s)
    return {'format': 'live-training-checkpoint-1', 'admission_hash': digest(admitted),
            'phase': 'completed-window-update', 'partial_buffer': None,
            'tables': [{k: array_wire(np.asarray(v)) for k, v in t.items()} for t in s[:3]],
            'counters': list(map(int, s[3:8])), 'ids': np.asarray(s.ids).tolist(),
            'environment': {k: {'shape': list(v.shape), 'dtype': '<i4',
                'hex': np.asarray(v).astype('<i4').tobytes().hex()} for k, v in s.environment._asdict().items()}}


def save(path, state, admitted):
    b = body(state, admitted)
    # Exclusive write, no overwrite. Interrupted bytes are invalid evidence, not a checkpoint.
    with Path(path).open('xb') as f: f.write(wire({'body': b, 'sha256': digest(b)}) + b'\n')


def load(path, seed, admitted):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate checkpoint key'); result[k] = v
        return result
    doc = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)
    require(set(doc) == {'body', 'sha256'} and digest(doc['body']) == doc['sha256'], 'Checkpoint checksum')
    b = doc['body']
    require(set(b) == {'format', 'admission_hash', 'phase', 'partial_buffer', 'tables', 'counters', 'ids', 'environment'}, 'Checkpoint fields')
    require(b['format'] == 'live-training-checkpoint-1' and b['admission_hash'] == digest(admitted)
            and b['phase'] == 'completed-window-update' and b['partial_buffer'] is None, 'Checkpoint binding')
    c = b['counters']
    require(type(c) is list and len(c) == 5 and all(type(x) is int and 0 <= x < 2**32 for x in c), 'Counter types')
    require(all(x < 2**31 for x in c[:3]) and type(seed) is int and c[4] == seed, 'Counter bounds/seed binding')
    require(type(b['ids']) is list and len(b['ids']) == learner.BATCH and all(type(x) is int and 0 <= x < 2**32 for x in b['ids']), 'Game IDs')
    require(type(b['tables']) is list and len(b['tables']) == 3, 'Tables')
    tables = []
    for table in b['tables']:
        require(type(table) is dict and set(table) == set(policy.SHAPES), 'Table keys')
        tables.append({k: jnp.asarray(read_array(v, policy.SHAPES[k])) for k, v in table.items()})
    template = engine.initial_kernel(jnp.zeros((learner.BATCH, 2), jnp.int32)); arrays = []
    require(type(b['environment']) is dict and set(b['environment']) == set(template._fields), 'Environment fields')
    for key, target in template._asdict().items():
        item = b['environment'][key]
        require(set(item) == {'shape', 'dtype', 'hex'} and item['shape'] == list(target.shape) and item['dtype'] == '<i4'
                and type(item['hex']) is str and len(item['hex']) == target.size * 8, 'Environment array schema')
        arrays.append(jnp.asarray(np.frombuffer(bytes.fromhex(item['hex']), dtype='<i4').copy().reshape(target.shape)))
    s = learner.Session(*tables, *[jnp.int32(x) for x in c[:3]], *[jnp.uint32(x) for x in c[3:]],
                        jnp.asarray(b['ids'], jnp.uint32), engine.State(*arrays))
    validate(s); require(body(s, admitted) == b, 'Canonical checkpoint')
    return s
