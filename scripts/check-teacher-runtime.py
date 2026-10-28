#!/usr/bin/env python3
"""F36 byte/path/runtime verification with frozen F35 scientific completion checks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('f35_frozen_completion_checker', ROOT / 'scripts/check-teacher-assisted.py')
_frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frozen)
validate_completion, STAGES, sha = _frozen.validate_completion, _frozen.STAGES, _frozen.sha
REVIEW = 'artifacts/reviews/f36-teacher-runtime'
PLAN_RELATIVE = 'contracts/teacher-runtime-v1/plan.json'
OUTPUT_PATHS = {
    'schedules': REVIEW + '/fixed-schedules-v1',
    'demonstrations': 'artifacts/runs/f36-teacher-data-v1',
    'training': 'artifacts/runs/f36-teacher-runtime-v1',
    'external_evaluation': 'artifacts/runs/f36-teacher-evaluation-v1',
    'verification': REVIEW + '/sampled-verification-v1',
    'results': REVIEW + '/measured-results-v1',
}
RUNTIME_SOURCES = {
    'python/pokemon_teacher_runtime/' + name + '.py'
    for name in ('__init__', 'stage', 'lifecycle', 'transport', 'qualification')
} | {'scripts/' + name + '-teacher-runtime.py' for name in ('run', 'launch', 'check')}
BINDINGS = {
    'failed_predecessor': 'artifacts/reports/f35-teacher-assisted-failure.json',
    'runtime_qualification': REVIEW + '/runtime-qualification-v1.json',
}


def bound_path(relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Invalid receipt path')
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Receipt path outside project')
    return path


def expected_commands():
    review = ROOT / REVIEW
    admission = review / 'admission'
    paths = {k: ROOT / v for k, v in OUTPUT_PATHS.items()}
    tails = {
        'fixed-schedules': ['schedules', '--admission', admission, '--output', paths['schedules']],
        'teacher-acquisition': ['demonstrations', '--admission', admission, '--plan',
            ROOT / PLAN_RELATIVE, '--output', paths['demonstrations']],
        'training': ['campaign', '--admission', admission, '--demonstrations', paths['demonstrations'],
            '--run', paths['training']],
        'external-evaluation': ['external_evaluation', '--admission', admission, '--demonstrations',
            paths['demonstrations'], '--training-run', paths['training'], '--output', paths['external_evaluation']],
        'sampled-verification': ['verification', '--training-run', paths['training'], '--external-run',
            paths['external_evaluation'], '--demonstrations', paths['demonstrations'], '--output', paths['verification']],
        'measured-results': ['reporting', '--training-run', paths['training'], '--external-run',
            paths['external_evaluation'], '--demonstrations', paths['demonstrations'],
            '--verification', paths['verification'], '--output', paths['results']],
    }
    prefix = [str(ROOT / '.venv/bin/python'), '-m', 'pokemon_teacher_runtime.lifecycle',
              '--', 'pokemon_teacher_runtime.stage']
    return {k: prefix + ['pokemon_teacher_assisted.' + v[0], *map(str, v[1:])]
            for k, v in tails.items()}


def validate_runtime(receipt, manifest, completion, workflow_exit, driver_exit):
    expected = [(receipt, 'f36-teacher-runtime-receipt-v1'),
        (completion, 'f36-completion-v1'), (workflow_exit, 'f36-workflow-exit-v1'),
        (driver_exit, 'f36-driver-exit-v1')]
    for record, schema in expected:
        if record.get('schema_version') != schema:
            raise ValueError('F36 record schema: ' + schema)
    if type(driver_exit.get('exit_code')) is not int or driver_exit['exit_code'] != 0:
        raise ValueError('F36 outer driver did not exit cleanly')
    if (receipt['output_paths'] != OUTPUT_PATHS or receipt['summary'] != OUTPUT_PATHS['results'] + '/summary.json'
            or completion['receipt'] != 'artifacts/reports/f36-teacher-runtime.json'
            or workflow_exit['receipt'] != completion['receipt']):
        raise ValueError('F36 output path inventory')
    execution = manifest['plan']['execution']
    if execution.get('schema_version') != 'f36-teacher-runtime-execution-v1' or receipt['execution'] != execution:
        raise ValueError('F36 execution metadata linkage')
    for name, relative in BINDINGS.items():
        entry = execution[name]
        if entry['path'] != relative or receipt['evidence_files'].get(relative) != entry['sha256']:
            raise ValueError('F36 runtime/predecessor evidence linkage: ' + name)
    commands = expected_commands()
    for entry in receipt['stages']:
        if entry.get('schema_version') != 'f36-stage-v1':
            raise ValueError('F36 stage record schema')
        if entry.get('stage') not in commands or entry.get('command') != commands[entry['stage']]:
            raise ValueError('F36 stage wrapper command/paths')


def validate_runtime_cleanup(cleanup):
    for key in ('unresolved_unknown_pids', 'reused_pids', 'handle_signals'):
        if cleanup.get(key) != []:
            raise ValueError('F36 cleanup unresolved or signalled: ' + key)
    if type(cleanup.get('unknown_overflow')) is not int or cleanup['unknown_overflow'] != 0:
        raise ValueError('F36 cleanup unknown identity overflow')
    if 'table_error' not in cleanup or cleanup['table_error'] is not None:
        raise ValueError('F36 cleanup process-table state is unresolved')


def verify_plan_binding(manifest):
    plan_path = bound_path(PLAN_RELATIVE)
    if manifest['source_files'].get(PLAN_RELATIVE) != sha(plan_path):
        raise ValueError('F36 CLI plan is not bound by admission')
    if json.loads(plan_path.read_text()) != manifest['plan']:
        raise ValueError('F36 CLI plan differs from admission plan')
    original_plan = json.loads((ROOT / 'contracts/teacher-assisted-v1/plan.json').read_text())
    if {k: v for k, v in manifest['plan'].items() if k != 'execution'} != original_plan:
        raise ValueError('F36 changed scientific plan')


def verify_runtime_evidence(receipt, manifest):
    if not RUNTIME_SOURCES.issubset(manifest['source_files']):
        raise ValueError('F36 runtime source inventory is incomplete')
    execution = manifest['plan']['execution']
    for name, relative in BINDINGS.items():
        path = bound_path(relative)
        if sha(path) != execution[name]['sha256']:
            raise ValueError('F36 runtime/predecessor evidence bytes: ' + name)
        record = json.loads(path.read_text())
        if record['status'] != ('failed' if name == 'failed_predecessor' else 'passed'):
            raise ValueError('F36 runtime/predecessor evidence status: ' + name)
        if name == 'runtime_qualification':
            qualified = record.get('source_files')
            if (not isinstance(qualified, dict) or not RUNTIME_SOURCES.issubset(qualified)
                    or any(manifest['source_files'].get(p) != h for p, h in qualified.items())):
                raise ValueError('Runtime qualification does not bind the admitted runtime sources')
        if name == 'failed_predecessor':
            if record.get('schema_version') != 'f35-failure-receipt-v1' or record.get('completed_comparison') is not False:
                raise ValueError('Original failed attempt was not preserved')
            original_sources = record['source_files']
            if any(manifest['source_files'].get(p) != h for p, h in original_sources.items()):
                raise ValueError('F35 admitted source bytes were not retained')
            content = record.pop('content_hash')
            encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if hashlib.sha256(encoded).hexdigest() != content:
                raise ValueError('Original failure receipt content hash')


def main(output):
    receipt_path = ROOT / 'artifacts/reports/f36-teacher-runtime.json'
    review = ROOT / 'artifacts/reviews/f36-teacher-runtime'
    body = json.loads(receipt_path.read_text())
    if body.get('schema_version') != 'f36-teacher-runtime-receipt-v1':
        raise ValueError('F36 receipt schema')
    recorded = body.pop('content_hash')
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != recorded or body['status'] != 'completed':
        raise ValueError('Receipt content identity/status')
    if receipt_path.read_bytes() != (review / 'f36-teacher-runtime-receipt.json').read_bytes():
        raise ValueError('Archived receipt bytes differ')
    if body['input_manifest'] != 'artifacts/reviews/f36-teacher-runtime/admission/manifest.json':
        raise ValueError('F36 admission path')
    manifest_path = ROOT / body['input_manifest']
    if sha(manifest_path) != body['input_hash']:
        raise ValueError('Admission manifest identity')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 'f35-inputs-v1':
        raise ValueError('Preserved admission schema')
    verify_plan_binding(manifest)
    if manifest['source_files'] != body['source_files']:
        raise ValueError('Source inventory differs from admitted manifest')
    for rel, expected in body['source_files'].items():
        if sha(bound_path(rel)) != expected or sha(manifest_path.parent / 'source' / rel) != expected:
            raise ValueError('Live/admitted source mismatch: ' + rel)
    for rel, expected in body['evidence_files'].items():
        if sha(bound_path(rel)) != expected:
            raise ValueError('Evidence mismatch: ' + rel)
    completion_records = {}
    for name in ('completion.json', 'workflow-exit.json'):
        record = json.loads((review / name).read_text())
        completion_records[name] = record
        if record['status'] != 'passed' or record['receipt_sha256'] != sha(receipt_path) or record['input_hash'] != body['input_hash']:
            raise ValueError('Completion linkage')
        if name == 'workflow-exit.json' and record['exit_code'] != 0:
            raise ValueError('Workflow exit failure')
    if sha(bound_path(body['summary'])) != body['summary_sha256']:
        raise ValueError('Summary identity')
    summary = json.loads(bound_path(body['summary']).read_text())
    if summary['status'] != 'passed' or summary['input_hash'] != body['input_hash'] or set(summary['seeds']) != {'7103', '8209', '9311'}:
        raise ValueError('Summary completion/seed inventory')
    stage_exits, stage_cleanups = {}, {}
    for name in STAGES:
        for suffix in ('.log', '-launch.json', '-exit.json', '-cleanup.json'):
            path = review / (name + suffix)
            if body['evidence_files'].get(str(path.relative_to(ROOT))) != sha(path):
                raise ValueError('Required stage evidence missing: ' + name + suffix)
        stage_exits[name] = json.loads((review / (name + '-exit.json')).read_text())
        stage_cleanups[name] = json.loads((review / (name + '-cleanup.json')).read_text())
        validate_runtime_cleanup(stage_cleanups[name])
        launch = json.loads((review / (name + '-launch.json')).read_text())
        if any(stage_exits[name].get(key) != value for key, value in launch.items()):
            raise ValueError('Stage launch/exit mismatch: ' + name)
    driver_exit = json.loads((review / 'driver-exit.json').read_text())
    validate_runtime(body, manifest, completion_records['completion.json'],
                     completion_records['workflow-exit.json'], driver_exit)
    verify_runtime_evidence(body, manifest)
    for key, relative in OUTPUT_PATHS.items():
        folder = bound_path(relative)
        required_report = folder / ('summary.json' if key == 'results' else 'report.json')
        if not required_report.is_file():
            raise ValueError('Missing F36 stage report: ' + key)
        report = json.loads(required_report.read_text())
        if report.get('status') != 'passed' or report.get('input_hash') != body['input_hash']:
            raise ValueError('F36 stage report status/input: ' + key)
        for path in folder.rglob('*'):
            if path.is_file() and body['evidence_files'].get(str(path.relative_to(ROOT))) != sha(path):
                raise ValueError('Unbound F36 output: ' + str(path))
    validate_completion(body, manifest, summary, completion_records['completion.json'],
                        completion_records['workflow-exit.json'], stage_exits, stage_cleanups)
    result = {'schema_version': 'f36-postcheck-v1', 'status': 'passed', 'receipt_sha256': sha(receipt_path), 'content_hash': recorded,
        'live_and_archived_sources': len(body['source_files']), 'evidence_files': len(body['evidence_files']),
        'fixed_counts_cells_stages_verified': True,
        'scope': 'All receipt-listed source/evidence bytes, archived receipt equality, successful stages, fixed counts, complete cells/contrasts and completion/exit linkage.'}
    output = Path(output).resolve()
    if output.is_relative_to(review.resolve()):
        raise ValueError('Postcheck must remain outside original review directory')
    with output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
