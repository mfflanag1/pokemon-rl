"""Atomic cohort commits and separate executable snapshots; progress prose is not input."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with open(temporary, 'xb') as out:
        out.write(canonical(value)+b'\n'); out.flush(); os.fsync(out.fileno())
    os.replace(temporary, path)


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'jax': jax.__version__,
        'jaxlib': __import__('jaxlib').__version__, 'backend': jax.default_backend(),
        'x64': jax.config.x64_enabled, 'node': subprocess.check_output(['node','--version'], text=True).strip()}


def source_paths():
    paths = []
    for base in ['python', 'packages', 'contracts', 'tests']:
        paths += [p for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ['.py', '.mjs', '.json']]
    paths += [ROOT/name for name in ['package.json', 'package-lock.json', 'showdown.pin.json']]
    paths += [ROOT/'vendor/pokemon-showdown/.reference-build.json']
    return sorted(paths)


def freeze(run, plan):
    run = Path(run)
    runtime_record = runtime()
    if runtime_record['backend'] != 'cpu' or runtime_record['x64']:
        raise ValueError('F2 admission requires CPU FP32')
    files = {}
    for path in source_paths():
        rel = str(path.relative_to(ROOT))
        target = run/'inputs/source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        with open(target, 'xb') as out:
            out.write(data)
        files[rel] = hashlib.sha256(data).hexdigest()
    manifest = {'schema_version':'f2-inputs-v1', 'plan':plan, 'runtime':runtime_record, 'sources':files}
    atomic_json(run/'inputs/manifest.json', manifest)
    return manifest


def admission(run, manifest):
    if runtime() != manifest['runtime']:
        raise ValueError('Runtime differs from frozen inputs')
    current = {str(p.relative_to(ROOT)) for p in source_paths()}
    if current != set(manifest['sources']):
        raise ValueError('Executable/configuration inventory changed')
    for rel, sha in manifest['sources'].items():
        if digest(ROOT/rel) != sha or digest(Path(run)/'inputs/source'/rel) != sha:
            raise ValueError(f'Executable/configuration drift: {rel}')


def save_checkpoint(directory, parameters, optimizer, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for group, state in [('p',parameters), ('m',optimizer[0]), ('v',optimizer[1])]:
        for key,value in state.items():
            arrays[group+'__'+key] = np.asarray(value)
    meta = {**metadata, 'optimizer_step':int(optimizer[2])}
    arrays['metadata'] = np.frombuffer(canonical(meta), dtype=np.uint8)
    path = directory/f'cohort-{meta["cohorts"]:05d}-{time.time_ns()}.npz'
    with open(path, 'xb') as out:
        np.savez_compressed(out, **arrays); out.flush(); os.fsync(out.fileno())
    pointer = {'file':path.name, 'sha256':digest(path)}
    atomic_json(directory/'latest.json', pointer)
    return pointer


def load_checkpoint(directory, expected_input_hash):
    directory = Path(directory)
    pointer = json.loads((directory/'latest.json').read_text())
    if Path(pointer['file']).name != pointer['file']:
        raise ValueError('Invalid checkpoint filename')
    path = directory/pointer['file']
    if digest(path) != pointer['sha256']:
        raise ValueError('Checkpoint digest mismatch')
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data['metadata'].tobytes())
        if meta['input_hash'] != expected_input_hash:
            raise ValueError('Checkpoint input binding mismatch')
        expected = initialize(meta['seed'])
        states = []
        for prefix in ['p','m','v']:
            state = {}
            for key, shape in expected.items():
                a = data[prefix+'__'+key]
                if a.shape != shape.shape or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError('Checkpoint shape/dtype/finite mismatch')
                state[key] = jnp.asarray(a)
            states.append(state)
    if not isinstance(meta['optimizer_step'], int) or meta['optimizer_step'] < 0:
        raise ValueError('Invalid optimizer counter')
    return states[0], (states[1], states[2], jnp.int32(meta['optimizer_step'])), meta
