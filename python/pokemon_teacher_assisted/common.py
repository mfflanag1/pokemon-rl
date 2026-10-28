"""Fixed F35 schedules, predecessor admission and independent input snapshots."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import jax.numpy as jnp

from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, canonical, digest, runtime
from pokemon_structured_policy.checkpoint import load, load_f32
from pokemon_structured_policy.common import admit_games, state_hash

SEEDS = (7103, 8209, 9311)
ARMS = ('control', 'assisted')
DEADLINE = datetime(2026, 10, 28, 7, 36, 8, tzinfo=timezone.utc).timestamp()
F34_RECEIPT_SHA = 'e88ffc1eeacf74cfa1afd5c3b112335fe60bfbd115d70648104eb1cd255d076a'


def read(path):
    return json.loads(Path(path).read_text())


def before_deadline():
    if time.time() >= DEADLINE:
        raise TimeoutError('F35 eight-hour deadline reached; no new work admitted')


def load_initial(seed):
    if seed not in SEEDS or type(seed) is not int:
        raise ValueError('F35 seed outside fixed inventory')
    receipt_path = ROOT / 'artifacts/reports/f34-policy-reset.json'
    if digest(receipt_path) != F34_RECEIPT_SHA:
        raise ValueError('F34 receipt binding')
    receipt = read(receipt_path)
    run = ROOT / 'artifacts/runs/f34-policy-reset-v1'
    report_path = run / 'report.json'
    if receipt['evidence_files'].get(str(report_path.relative_to(ROOT))) != digest(report_path):
        raise ValueError('F34 report binding')
    report = read(report_path)
    if report['status'] != 'passed':
        raise ValueError('F34 did not complete')
    folder = run / str(seed) / 'checkpoints/512'
    pointer_path = folder / 'latest.json'
    pointer = read(pointer_path)
    for path in (pointer_path, folder / pointer['file']):
        if receipt['evidence_files'].get(str(path.relative_to(ROOT))) != digest(path):
            raise ValueError('F34 final checkpoint not bound by receipt')
    parameters, old_optimizer, metadata = load(folder,
        {'seed': seed, 'cohorts': 512, 'input_hash': report['input_hash']})
    if (int(old_optimizer[2]) != 4096 or
            parameter_hash(parameters) != report['seeds'][str(seed)]['final_parameter_hash']):
        raise ValueError('F34 final parameter/counter binding')
    frozen, catalog, frozen_identity = load_f32(seed)
    optimizer = ({k: jnp.zeros_like(v) for k, v in parameters.items()},
                 {k: jnp.zeros_like(v) for k, v in parameters.items()}, jnp.int32(0))
    identity = {'f34_receipt_sha256': F34_RECEIPT_SHA,
        'f34_pointer': str(pointer_path.relative_to(ROOT)), 'f34_pointer_sha256': digest(pointer_path),
        'f34_parameter_hash': parameter_hash(parameters), 'f34_metadata': metadata,
        'adam_reset': True, 'frozen_f32': frozen_identity}
    return parameters, optimizer, catalog, frozen, identity


def training_specs(plan, seed, first, opponent):
    if opponent not in (None, 'frozen_f32', 'tactical'):
        raise ValueError('F35 training opponent not admitted')
    return [dict(seed=derived_seed('f35-training-teams-v1', seed, opponent), episode=i,
        max_turns=plan['max_turns'], split='train', swap=False, opponent=opponent,
        learner_side='p2' if i % 2 else 'p1',
        actor_seed=derived_seed('f35-training-actor-v1', seed, opponent))
        for i in range(first, first + plan['training_chunk_games'])]


def evaluation_specs(plan, seed, opponent, first, count):
    if opponent not in plan['evaluation_pairs'] and opponent != 'foul_play':
        raise ValueError('F35 evaluation opponent not admitted')
    return [dict(seed=derived_seed('f35-evaluation-teams-v1', seed, opponent),
        episode=i, max_turns=plan['max_turns'], split='development', swap=swap,
        opponent=opponent, learner_side='p2' if swap else 'p1',
        actor_seed=derived_seed('f35-evaluation-actor-v1', seed, opponent))
        for i in range(first, first + count) for swap in (False, True)]


def teacher_specs(plan, seed, role):
    settings = plan['teacher_data']
    if role not in ('train', 'probe'):
        raise ValueError('Teacher role not admitted')
    count = settings[role + '_games_per_host']
    if type(count) is not int or count <= 0 or count % 2:
        raise ValueError('Teacher games must be positive complete seat pairs')
    return [dict(run_seed=derived_seed(settings[role + '_namespace'], seed), episode_index=i,
        swap=swap, external_side='p1' if swap else 'p2',
        split='train' if role == 'train' else 'development',
        fp_seed=derived_seed(settings['sampling_namespace'], role, seed, i, swap))
        for i in range(count // 2) for swap in (False, True)]


def validate_plan(plan):
    if plan.get('schema_version') != 'f35-teacher-assisted-plan-v1':
        raise ValueError('F35 plan schema')
    expected = {'seeds': list(SEEDS), 'arms': list(ARMS), 'cohorts_per_seed_arm': 256,
        'selected_rows_per_cohort': 1024, 'epochs': 2, 'minibatch_rows': 256,
        'optimizer_steps_per_cohort': 8, 'deadline_utc': '2026-10-28T07:36:08Z',
        'stream_selected_rows': {'self_play': 256, 'frozen_f32': 256, 'tactical': 512},
        'evaluation_pairs': {'baseline-v1': 256, 'tactical': 64, 'base_power': 32, 'random': 32},
        'external_pairs_per_seed_state': 64, 'sampled_update_cohorts': [0, 255]}
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('F35 fixed plan coordinate/budget drift')
    mechanism = plan['mechanism']
    if mechanism['kind'] != 'ppo-plus-offline-demo-ce' or mechanism['coefficient'] != .1 or mechanism['minibatch_rows'] != 256:
        raise ValueError('F35 assistance mechanism drift')
    if any(plan.get(k) is not False for k in ['public_server_used', 'final_reserved_data_used', 'github_publication_authorized']):
        raise ValueError('F35 scope drift')
    return plan


def bind_inputs(destination, plan, source_paths, qualification):
    before_deadline()
    validate_plan(plan)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for rel in sorted(set(source_paths)):
        path = ROOT / rel
        if path.resolve().is_relative_to(ROOT.resolve()) is not True:
            raise ValueError('Input path outside project')
        data = path.read_bytes()
        target = destination / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as out:
            out.write(data)
        files[rel] = hashlib.sha256(data).hexdigest()
    execution = runtime()
    if execution['backend'] != 'cpu' or execution['x64']:
        raise ValueError('F35 requires qualified CPU FP32 runtime')
    body = {'schema_version': 'f35-inputs-v1', 'plan': plan, 'source_files': files,
        'runtime': execution, 'qualification': qualification,
        'predecessor': {'path': 'artifacts/reports/f34-policy-reset.json', 'sha256': F34_RECEIPT_SHA}}
    atomic_json(destination / 'manifest.json', body)
    return digest(destination / 'manifest.json')


def verify_inputs(destination):
    destination = Path(destination)
    body = read(destination / 'manifest.json')
    if body.get('schema_version') != 'f35-inputs-v1' or runtime() != body['runtime']:
        raise ValueError('F35 input schema/runtime drift')
    validate_plan(body['plan'])
    if digest(ROOT / body['predecessor']['path']) != F34_RECEIPT_SHA:
        raise ValueError('F35 predecessor drift')
    for rel, expected in body['source_files'].items():
        if digest(ROOT / rel) != expected or digest(destination / 'source' / rel) != expected:
            raise ValueError('F35 input drift: ' + rel)
    return body
