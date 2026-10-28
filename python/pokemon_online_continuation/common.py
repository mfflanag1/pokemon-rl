"""F39 fixed coordinates, inherited Adam admission, and fresh schedule namespaces."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, digest, runtime
from pokemon_structured_policy.checkpoint import load, load_f32
from pokemon_structured_policy.common import admit_games, state_hash

SEEDS = (7103, 8209, 9311)
STREAMS = {'self_play': None, 'frozen_f32': 'frozen_f32', 'tactical': 'tactical'}
DEADLINE_UTC = '2026-10-29T00:13:21Z'
F34_RECEIPT_SHA = 'e88ffc1eeacf74cfa1afd5c3b112335fe60bfbd115d70648104eb1cd255d076a'
NAMESPACES = {key: 'f39-' + value + '-v1' for key, value in {
    'training_teams': 'training-teams', 'training_actor': 'training-actor',
    'evaluation_teams': 'evaluation-teams', 'evaluation_actor': 'evaluation-actor',
    'row_selection': 'row-selection', 'optimizer': 'optimizer'}.items()}
EXPECTED_COUNTS = {'cohorts': 768, 'optimizer_steps': 6144, 'selected_rows': 786432,
    'row_presentations': 1572864, 'evaluation_games': 4608, 'checkpoint_comparisons': 9,
    'original_next_update_recoveries': 6}


def read(path):
    return json.loads(Path(path).read_text())


def deadline(plan):
    """Guard admissions against the fixed authorization clock, never predecessor clocks."""
    if plan.get('deadline_utc') != DEADLINE_UTC:
        raise ValueError('F39 deadline drift')
    stop = datetime.fromisoformat(plan['deadline_utc'].replace('Z', '+00:00')).timestamp()
    if time.time() >= stop:
        raise TimeoutError('F39 four-hour deadline reached; no new work admitted')
    return stop


def validate_plan(plan):
    expected = {'schema_version': 'f39-online-continuation-plan-v1', 'seeds': list(SEEDS),
        'initial_cohorts_per_seed': 512, 'final_cohorts_per_seed': 768, 'cohorts_per_seed': 256,
        'selected_rows_per_cohort': 1024, 'epochs': 2, 'minibatch_rows': 256,
        'optimizer_steps_per_cohort': 8, 'deadline_utc': DEADLINE_UTC,
        'stream_selected_rows': {'self_play': 256, 'frozen_f32': 256, 'tactical': 512},
        'evaluation_pairs': {'baseline-v1': 256, 'tactical': 64, 'base_power': 32, 'random': 32},
        'sampled_update_cohorts': [512, 767], 'expected_counts': EXPECTED_COUNTS,
        'namespaces': NAMESPACES, 'max_turns': 240, 'workers': 8,
        'training_chunk_games': 8, 'maximum_games_per_stream': 64,
        'initial_optimizer_step': 4096, 'final_optimizer_step': 6144,
        'statistics': {'seed': 1027283903, 'resamples': 20000, 'alpha': .05,
            'family_cells': 12, 'practical_improvement': .05, 'max_final_cutoff_rate': .05}}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('F39 fixed plan coordinate/budget/namespace drift')
    for key in ('public_server_used', 'final_reserved_data_used', 'github_publication_authorized',
                'teacher_workers_used', 'demonstration_data_used', 'cloud_training_used', 'promotion_authorized'):
        if plan.get(key) is not False:
            raise ValueError('F39 scope drift: ' + key)
    return plan


def load_initial(seed):
    """Load all F34 parameters and nonzero Adam moments without resetting any state."""
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('F39 seed outside fixed inventory')
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
    if not isinstance(pointer.get('file'), str) or Path(pointer['file']).name != pointer['file']:
        raise ValueError('F34 checkpoint pointer')
    for path in (pointer_path, folder / pointer['file']):
        if receipt['evidence_files'].get(str(path.relative_to(ROOT))) != digest(path):
            raise ValueError('F34 final checkpoint not bound by receipt')
    parameters, optimizer, metadata = load(folder,
        {'seed': seed, 'cohorts': 512, 'input_hash': report['input_hash'], 'selected_rows': 524288})
    seed_result = report['seeds'][str(seed)]
    cursors = metadata.get('stream_cursors')
    if (int(optimizer[2]) != 4096 or seed_result['optimizer_steps'] != 4096 or
            parameter_hash(parameters) != seed_result['final_parameter_hash']):
        raise ValueError('F34 final parameter/Adam counter binding')
    if (not isinstance(cursors, dict) or set(cursors) != set(STREAMS)
            or any(type(v) is not int or v < 0 for v in cursors.values())
            or cursors != seed_result['stream_cursors']
            or metadata['next_episode'] != sum(cursors.values())):
        raise ValueError('F34 final stream cursor binding')
    frozen, catalog, frozen_identity = load_f32(seed)
    identity = {'f34_receipt_sha256': F34_RECEIPT_SHA,
        'f34_pointer': str(pointer_path.relative_to(ROOT)), 'f34_pointer_sha256': digest(pointer_path),
        'f34_checkpoint_sha256': digest(folder / pointer['file']),
        'f34_parameter_hash': parameter_hash(parameters), 'parameter_hash': parameter_hash(parameters),
        'state_hash': state_hash(parameters, optimizer), 'f34_metadata': metadata,
        'adam_reset': False, 'frozen_f32': frozen_identity}
    return parameters, optimizer, catalog, frozen, identity


def training_specs(plan, seed, first, opponent):
    if opponent not in STREAMS.values():
        raise ValueError('F39 training opponent not admitted')
    if type(first) is not int or first < 0:
        raise ValueError('F39 training cursor')
    ns = plan['namespaces']
    return [dict(seed=derived_seed(ns['training_teams'], seed, opponent), episode=i,
        max_turns=plan['max_turns'], split='train', swap=False, opponent=opponent,
        learner_side='p2' if i % 2 else 'p1',
        actor_seed=derived_seed(ns['training_actor'], seed, opponent))
        for i in range(first, first + plan['training_chunk_games'])]


def evaluation_specs(plan, seed, opponent, first, count):
    if opponent not in plan['evaluation_pairs'] or type(first) is not int or type(count) is not int or first < 0 or count <= 0:
        raise ValueError('F39 evaluation coordinate')
    ns = plan['namespaces']
    return [dict(seed=derived_seed(ns['evaluation_teams'], seed, opponent),
        episode=i, max_turns=plan['max_turns'], split='development', swap=swap,
        opponent=opponent, learner_side='p2' if swap else 'p1',
        actor_seed=derived_seed(ns['evaluation_actor'], seed, opponent))
        for i in range(first, first + count) for swap in (False, True)]


def bind_inputs(destination, plan, source_paths, qualification):
    deadline(plan)
    validate_plan(plan)
    if qualification.get('status') != 'passed':
        raise ValueError('F39 qualification did not pass')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for rel in sorted(set(source_paths)):
        path = ROOT / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Input path outside project')
        data = path.read_bytes()
        target = destination / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as out:
            out.write(data)
        files[rel] = hashlib.sha256(data).hexdigest()
    execution = runtime()
    if execution['backend'] != 'cpu' or execution['x64']:
        raise ValueError('F39 requires qualified CPU FP32 runtime')
    body = {'schema_version': 'f39-inputs-v1', 'plan': plan, 'source_files': files,
        'runtime': execution, 'qualification': qualification,
        'predecessor': {'path': 'artifacts/reports/f34-policy-reset.json', 'sha256': F34_RECEIPT_SHA}}
    atomic_json(destination / 'manifest.json', body)
    verify_inputs(destination)
    return digest(destination / 'manifest.json')


def verify_inputs(destination):
    destination = Path(destination)
    body = read(destination / 'manifest.json')
    if body.get('schema_version') != 'f39-inputs-v1' or runtime() != body['runtime']:
        raise ValueError('F39 input schema/runtime drift')
    validate_plan(body['plan'])
    if body.get('predecessor') != {'path': 'artifacts/reports/f34-policy-reset.json', 'sha256': F34_RECEIPT_SHA}:
        raise ValueError('F39 predecessor identity')
    if digest(ROOT / body['predecessor']['path']) != F34_RECEIPT_SHA:
        raise ValueError('F39 predecessor drift')
    if body['qualification'].get('status') != 'passed' or not body['source_files']:
        raise ValueError('F39 incomplete admission')
    for rel, expected in body['source_files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('F39 unsafe input path')
        if digest(ROOT / rel) != expected or digest(destination / 'source' / rel) != expected:
            raise ValueError('F39 input drift: ' + rel)
    return body
