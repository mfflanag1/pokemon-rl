#!/usr/bin/env python3
"""F40 verification-only completion check; read-only apart from one exclusive postcheck."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = 'artifacts/reviews/f40-online-verification'
ADMISSION = REVIEW + '/admission/manifest.json'
RECEIPT = 'artifacts/reports/f40-online-verification.json'
POSTCHECK = 'artifacts/reviews/f40-online-verification-postcheck.json'
FAILURE_RECEIPT = 'artifacts/reports/f40-online-verification-failure.json'
FAILURE_OUTPUTS = (FAILURE_RECEIPT, REVIEW + '/workflow-failure.json',
                   REVIEW + '/f40-online-verification-failure-receipt.json')
TRAIN = 'artifacts/runs/f39-online-continuation-v1'
VERIFY = REVIEW + '/sampled-verification-v1'
RESULT = REVIEW + '/measured-results-v1'
FAILURE = 'artifacts/reports/f39-online-continuation-failure.json'
FAILURE_SHA = '0ebefe24c86c9f2b2c534fdcd1516e805cbf8e893c4fd5adf2e1ffc99cbf6d61'
ORIGINAL_ADMISSION = 'artifacts/reviews/f39-online-continuation/admission/manifest.json'
ORIGINAL_ADMISSION_SHA = '2921e7aaed8dfba6f4826ba4e329027172760d5818c944f9fb518bb67e3d937e'
DEADLINE = '2026-10-29T00:13:21Z'
STAGES = ('sampled-verification', 'measured-results')
FORBIDDEN_PREFIXES = ['pokemon_external_benchmark', 'pokemon_structured_policy.external',
    'pokemon_teacher_assisted.external', 'pokemon_teacher_assisted.external_evaluation',
    'pokemon_teacher_assisted.teacher', 'pokemon_teacher_assisted.demonstrations',
    'pokemon_teacher_assisted.data', 'pokemon_teacher_assisted.learning',
    'pokemon_teacher_assisted.verification', 'pokemon_teacher_runtime',
    'pokemon_teacher_legality', 'fp', 'poke_engine', 'foul_play']
SOURCE_COUNT = 2096
EVIDENCE_COUNT = 5789
SEAL_EXCLUSIONS = {REVIEW + '/f40-online-verification-receipt.json', REVIEW + '/completion.json'}
REQUIRED_SOURCES = {'scripts/run-online-verification.py', 'scripts/check-online-verification.py',
    'python/pokemon_online_verification/__init__.py', 'python/pokemon_online_verification/guard.py',
    'python/pokemon_online_verification/verification.py', 'tests/test_f40_completion.py', 'tests/test_f40_guard.py'}
SPEC = importlib.util.spec_from_file_location('f40_frozen_f39_helpers', ROOT / 'scripts/check-online-continuation.py')
old = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old)


def require(condition, label):
    if not condition:
        raise ValueError('F40 verification: ' + label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and '..' not in Path(relative).parts, 'unsafe relative path')
    path = ROOT / relative
    require(path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink(), 'escaped/symlink path')
    return path


def files(base):
    base = Path(base)
    paths = list(base.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'symlink in evidence tree')
    return {str(p.relative_to(base)): p for p in paths if p.is_file()}


def check_inventory(base, expected, label):
    actual = files(base)
    require(set(actual) == set(expected), label + ' complete inventory')
    for relative, expected_sha in expected.items():
        require(sha(actual[relative]) == expected_sha, label + ' hash: ' + relative)


def verify_original():
    """No imports from the scientific implementation; verify every original byte/inventory."""
    receipt_path = bound(FAILURE)
    require(sha(receipt_path) == FAILURE_SHA, 'original failure receipt SHA')
    failure = read(receipt_path)
    require(failure['status'] == 'failed' and failure['completed_comparison'] is False
            and failure['scientific_goal_completed'] is False, 'original failure status')
    body = {k: v for k, v in failure.items() if k != 'content_hash'}
    require(hashlib.sha256(canonical(body)).hexdigest() == failure['content_hash'], 'original content hash')
    require(receipt_path.read_bytes() == bound(failure['archived_receipt_path']).read_bytes(), 'original archived receipt bytes')
    require(failure['admission_manifest'] == ORIGINAL_ADMISSION
            and failure['admission_sha256'] == ORIGINAL_ADMISSION_SHA
            and sha(bound(ORIGINAL_ADMISSION)) == ORIGINAL_ADMISSION_SHA, 'original admission SHA')
    manifest = read(bound(ORIGINAL_ADMISSION))
    require(bound(TRAIN + '/inputs/manifest.json').read_bytes() == bound(ORIGINAL_ADMISSION).read_bytes(), 'original production admission bytes')
    sources, evidence = failure['source_files'], failure['evidence_files']
    require(sources == manifest['source_files'] and len(sources) == SOURCE_COUNT == failure['source_files_count'], 'original source count')
    require(len(evidence) == EVIDENCE_COUNT == failure['evidence_files_count'], 'original evidence count')
    for relative, expected in sources.items():
        require(sha(bound(relative)) == expected, 'original live source hash: ' + relative)
    for key in ('source_archive', 'production_source_archive'):
        check_inventory(bound(failure[key]), sources, key)
    for relative, expected in evidence.items():
        require(sha(bound(relative)) == expected, 'original live evidence hash: ' + relative)
    check_inventory(bound(failure['evidence_archive']), evidence, 'original archived evidence')
    excluded = set(failure['inventory_exclusions'])
    seal_excluded = [bound(p) for p in failure['inventory_seal_outputs_excluded']]
    actual = set()
    for relative in failure['inventory_roots']:
        for name, path in files(bound(relative)).items():
            key = str(path.relative_to(ROOT))
            if key not in excluded and not any(path == p or path.is_relative_to(p) for p in seal_excluded):
                actual.add(key)
    listed_in_roots = {p for p in evidence if any(bound(p).is_relative_to(bound(r)) for r in failure['inventory_roots'])}
    require(actual == listed_in_roots, 'original complete live evidence inventory')
    require(all(not bound(p).exists() for p in failure['required_outputs_absent']), 'original success outputs must remain absent')
    require(failure['training_exit']['exit_code'] == 0 and failure['verification_exit']['exit_code'] == 1
            and failure['workflow_exit']['exit_code'] == 1 and failure['driver_exit_summary']['exit_code'] == 1,
            'original failed workflow linkage')
    return failure, manifest


def verify_admission(failure=None):
    if failure is None:
        failure, _ = verify_original()
    manifest = read(bound(ADMISSION))
    require(manifest['schema_version'] == 'f40-verification-admission-v1' and manifest['deadline_utc'] == DEADLINE,
            'new admission schema/deadline')
    require(manifest['original_failure_receipt'] == {'path': FAILURE, 'sha256': FAILURE_SHA}
            and manifest['original_admission'] == {'path': ORIGINAL_ADMISSION, 'sha256': ORIGINAL_ADMISSION_SHA},
            'new admission original bindings')
    sources = manifest['source_files']
    require(REQUIRED_SOURCES.issubset(sources), 'required additive source inventory')
    require(not set(sources).intersection(failure['source_files']), 'admission cannot replace original source')
    for relative, expected in sources.items():
        require(sha(bound(relative)) == expected, 'additive live source hash: ' + relative)
    check_inventory(bound(REVIEW + '/admission/source'), sources, 'additive frozen source')
    for field in ('qualification', 'authorization', 'guard_policy'):
        binding = manifest[field]
        require(set(binding) == {'path', 'sha256'} and sources.get(binding['path']) == binding['sha256']
                and sha(bound(binding['path'])) == binding['sha256'], 'bound ' + field)
    qualification = read(bound(manifest['qualification']['path']))
    require(qualification['status'] == 'passed', 'qualification must pass')
    authorization = read(bound(manifest['authorization']['path']))
    require(authorization['verification_only'] is True and authorization['no_new_training'] is True
            and authorization['deadline_utc'] == DEADLINE, 'verification-only authorization')
    return manifest


def expected_commands():
    prefix = [str(ROOT / '.venv/bin/python'), '-m', 'pokemon_teacher_runtime.lifecycle', '--']
    return {'sampled-verification': prefix + ['pokemon_online_verification.verification', '--training-run',
                str(bound(TRAIN)), '--output', str(bound(VERIFY))],
            'measured-results': prefix + ['pokemon_online_continuation.reporting', '--training-run', str(bound(TRAIN)),
                '--verification', str(bound(VERIFY)), '--output', str(bound(RESULT))]}


def validate_stages(stages, exits, cleanups, launches):
    require([r['stage'] for r in stages] == list(STAGES), 'exactly two ordered new stages; no training')
    require(all(set(r) == set(STAGES) for r in (exits, cleanups, launches)), 'new stage evidence inventory')
    for row in stages:
        name = row['stage']
        require(row == exits[name] and launches[name] == {k: row[k] for k in ('schema_version', 'stage', 'command', 'pid', 'started_utc')},
                'new stage launch/exit linkage')
        require(row['schema_version'] == 'f40-stage-v1' and row['command'] == expected_commands()[name], 'new stage command')
        require(type(row['exit_code']) is int and row['exit_code'] == 0 and row['deadline_timeout'] is False
                and row['leaked_after_success'] is False and row['wait_error'] is None
                and row['cleanup_status'] == 'passed', 'new stage did not complete cleanly')
        require(old.number(row['elapsed_seconds'], 'elapsed') >= 0, 'negative stage duration')
        require(datetime.fromisoformat(row['ended_utc']).timestamp() <= datetime.fromisoformat(DEADLINE.replace('Z', '+00:00')).timestamp(),
                'stage exceeded original deadline')
        old.validate_cleanup(cleanups[name], row['pid'], name)


def validate_policy(verification, policy, original_manifest):
    record = verification['f40_runtime_policy']
    require(record['schema_version'] == 'f40-runtime-policy-v1' and record['production_repeated'] is False
            and record['original_failure_preserved'] is True and len(record['snapshots']) == 2, 'adapter policy declaration')
    expected_exceptions = {'pokemon_teacher_runtime': 'python/pokemon_teacher_runtime/__init__.py',
                           'pokemon_teacher_runtime.lifecycle': 'python/pokemon_teacher_runtime/lifecycle.py'}
    require(policy['exact_runtime_exceptions'] == expected_exceptions
            and policy['forbidden_prefixes'] == FORBIDDEN_PREFIXES, 'only exact lifecycle exceptions')
    require(policy.get('exception_sha256') == {name: original_manifest['source_files'][relative]
            for name, relative in expected_exceptions.items()}, 'exception policy SHA binding')
    require(verification['coverage']['teacher_search_module_inventory']['forbidden_prefixes'] == FORBIDDEN_PREFIXES,
            'original forbidden prefixes retained in coverage')
    for snapshot in record['snapshots']:
        require(snapshot['forbidden'] == [] and snapshot['exact_runtime_exceptions'] == expected_exceptions
                and snapshot['forbidden_prefixes'] == policy['forbidden_prefixes']
                and snapshot['original_admission_sha256'] == ORIGINAL_ADMISSION_SHA, 'guard snapshot policy linkage')
        allowed = snapshot['allowed_runtime_modules']
        require(len(allowed) == len({row['name'] for row in allowed})
                and 'pokemon_teacher_runtime' in {row['name'] for row in allowed}
                and {row['name'] for row in allowed}.issubset(expected_exceptions), 'actual wrapped runtime module inventory')
        for row in allowed:
            relative = expected_exceptions[row['name']]
            require(row['path'] == relative
                    and row['sha256'] == original_manifest['source_files'][relative], 'runtime exception path/hash')


def validate_reports(training, verification, summary, original_manifest):
    require(training['schema_version'] == 'f39-online-continuation-run-v1' and summary['schema_version'] == 'f39-measured-summary-v1',
            'inherited/report schemas')
    for record in (training, verification, summary):
        require(record['status'] == 'passed' and record['input_hash'] == ORIGINAL_ADMISSION_SHA, 'original report input identity')
    require(training['plan'] == original_manifest['plan'], 'original plan linkage')
    old.validate_summary(summary, training, verification)


def validate_inputs(record, evidence, original_evidence):
    required = {str(bound(TRAIN + '/' + name)) for name in ('report.json', 'inputs/manifest.json')}
    require(required.issubset(record['input_files']), 'missing inherited report input bindings')
    for absolute, expected in record['input_files'].items():
        path = Path(absolute)
        require(path.is_absolute() and not path.is_symlink() and any(path.resolve().is_relative_to(bound(base).resolve())
                for base in (TRAIN, VERIFY)), 'analysis/replay input outside evidence')
        relative = str(path.relative_to(ROOT))
        require(evidence.get(relative, original_evidence.get(relative)) == expected and sha(path) == expected,
                'analysis/replay input SHA binding')


def require_no_failure_evidence():
    require(all(not bound(path).exists() for path in FAILURE_OUTPUTS),
            'failure evidence invalidates candidate success')


def validate_outer_execution(stages, admission_sha256, original_manifest):
    """Check owner, actual cleanup and worker provenance before a success seal exists."""
    require_no_failure_evidence()
    launch = read(bound(REVIEW + '/driver-launch.json'))
    claim = read(bound(REVIEW + '/launcher-claim.json'))
    workflow = read(bound(REVIEW + '/workflow-claim.json'))
    driver = read(bound(REVIEW + '/driver-exit.json'))
    result = read(bound(REVIEW + '/workflow-result.json'))
    require(launch['launcher_pid'] == claim['pid'] and launch['driver_pid'] == workflow['pid'], 'outer owner linkage')
    require(launch['command'] == [str(ROOT / '.venv/bin/python'), str(bound('scripts/run-online-verification.py')), '--worker'], 'outer worker command')
    require(driver['schema_version'] == 'f40-driver-exit-v1' and type(driver['exit_code']) is int
            and driver['exit_code'] == 0 and driver['error'] is None, 'outer driver exit')
    old.validate_cleanup(driver['cleanup'], launch['driver_pid'], 'outer driver')
    require(driver['ended_unix'] <= datetime.fromisoformat(DEADLINE.replace('Z', '+00:00')).timestamp() + 60,
            'outer bounded cleanup deadline')
    require(result['status'] == 'passed' and result['stages'] == stages
            and result['admission_sha256'] == admission_sha256, 'worker result linkage')
    require(result['runtime_before'] == result['runtime_after'] == original_manifest['runtime'], 'runtime before/after identity')
    require(result['preservation_before'] == result['preservation_after'] == {
        'failure_receipt_sha256': FAILURE_SHA, 'source_files_verified': SOURCE_COUNT,
        'evidence_files_verified': EVIDENCE_COUNT}, 'full original before/after preservation')
    return result


def verify_completion():
    require_no_failure_evidence()
    failure, original_manifest = verify_original()
    manifest = verify_admission(failure)
    receipt_path = bound(RECEIPT)
    receipt = read(receipt_path)
    content = {k: v for k, v in receipt.items() if k != 'content_hash'}
    require(hashlib.sha256(canonical(content)).hexdigest() == receipt['content_hash'], 'new receipt content hash')
    require(receipt_path.read_bytes() == bound(REVIEW + '/f40-online-verification-receipt.json').read_bytes(), 'new archived receipt bytes')
    require(receipt['schema_version'] == 'f40-online-verification-receipt-v1' and receipt['status'] == 'completed', 'new receipt status/schema')
    require(receipt['admission_manifest'] == ADMISSION and receipt['admission_sha256'] == sha(bound(ADMISSION))
            and receipt['source_files'] == manifest['source_files'], 'new receipt admission binding')
    require(receipt['original_failure_receipt'] == manifest['original_failure_receipt']
            and receipt['original_workflow_status'] == 'failed' and receipt['deadline_utc'] == DEADLINE, 'failure preserved/deadline')
    require(receipt['output_paths'] == {'inherited_training': TRAIN, 'verification': VERIFY, 'results': RESULT}, 'new output paths')
    require(receipt['new_work'] == {'training_cohorts': 0, 'selected_training_rows': 0, 'training_optimizer_steps': 0,
            'fresh_replay_games': 66, 'recomputed_optimizer_steps': 48}, 'verification-only work counts')
    for key in ('teacher_used', 'demonstrations_used', 'public_server_used', 'final_reserved_data_used', 'promotion_authorized'):
        require(receipt[key] is False, 'receipt scope/' + key)
    require(receipt['learning_success_separate_from_execution'] is True, 'learning/execution distinction')
    evidence = receipt['evidence_files']
    actual = {REVIEW + '/' + p: path for p, path in files(bound(REVIEW)).items() if REVIEW + '/' + p not in SEAL_EXCLUSIONS}
    require(set(actual) == set(evidence), 'complete new evidence inventory')
    for relative, expected in evidence.items():
        require(sha(bound(relative)) == expected, 'new evidence hash: ' + relative)
    required_names = ['launcher-claim.json', 'driver-launch.json', 'workflow-claim.json', 'driver.log',
                      'driver-exit.json', 'workflow-result.json']
    for name in required_names:
        require(REVIEW + '/' + name in evidence, 'missing outer evidence/' + name)
    stage_records = [{name: read(bound(REVIEW + '/' + name + suffix)) for name in STAGES}
                     for suffix in ('-exit.json', '-cleanup.json', '-launch.json')]
    for name in STAGES:
        for suffix in ('.log', '-exit.json', '-cleanup.json', '-launch.json'):
            require(REVIEW + '/' + name + suffix in evidence, 'missing stage evidence')
    validate_stages(receipt['stages'], *stage_records)
    validate_outer_execution(receipt['stages'], receipt['admission_sha256'], original_manifest)
    completion = read(bound(REVIEW + '/completion.json'))
    require(completion['schema_version'] == 'f40-completion-v1' and completion['status'] == 'passed'
            and completion['receipt'] == RECEIPT and completion['receipt_sha256'] == sha(receipt_path)
            and completion['admission_sha256'] == receipt['admission_sha256']
            and completion['source_files_verified'] == len(manifest['source_files'])
            and completion['evidence_files_verified'] == len(evidence), 'completion receipt SHA/count linkage')
    training, verification, summary = [read(bound(p)) for p in (TRAIN + '/report.json', VERIFY + '/report.json', RESULT + '/summary.json')]
    require(receipt['inherited_training'] == {'report': TRAIN + '/report.json', 'sha256': sha(bound(TRAIN + '/report.json')),
            'original_stage': failure['training_exit'], 'counts': training['counts'], 'seeds': list(old.SEEDS)}, 'inherited training record')
    require(receipt['summary'] == RESULT + '/summary.json' and receipt['summary_sha256'] == sha(bound(receipt['summary'])), 'summary SHA')
    validate_reports(training, verification, summary, original_manifest)
    validate_policy(verification, read(bound(manifest['guard_policy']['path'])), original_manifest)
    for record in (verification, summary):
        validate_inputs(record, evidence, failure['evidence_files'])
    require(str(bound(VERIFY + '/report.json')) in summary['input_files'], 'summary must bind new verification report')
    return receipt, summary


def main(output):
    output = Path(output).resolve()
    require(output == bound(POSTCHECK).resolve(), 'postcheck must use exclusive outer F40 path')
    require(not output.exists(), 'postcheck already exists')
    receipt, summary = verify_completion()
    result = {'schema_version': 'f40-postcheck-v1', 'status': 'passed', 'receipt_sha256': sha(bound(RECEIPT)),
        'original_workflow_status': 'failed', 'original_failure_receipt_sha256': FAILURE_SHA,
        'original_source_files_verified': SOURCE_COUNT, 'original_evidence_files_verified': EVIDENCE_COUNT,
        'new_evidence_files_verified': len(receipt['evidence_files']), 'counts': summary['counts'],
        'total_contrasts': summary['total_contrasts'], 'all_seed_primary_passed': summary['all_seed_primary_passed'],
        'any_detected_regression': summary['any_detected_regression'], 'new_work': receipt['new_work'],
        'scope': 'Static hash, complete inventory, original failure preservation, report semantics and process cleanup check; no new scientific execution.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / POSTCHECK))
    main(parser.parse_args().output)
