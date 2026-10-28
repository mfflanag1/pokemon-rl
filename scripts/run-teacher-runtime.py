#!/usr/bin/env python3
"""One bounded F36 workflow; exclusive outputs, no scientific retry or resumption."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_full_train.storage import atomic_json, canonical, digest
from pokemon_teacher_assisted.common import DEADLINE, before_deadline, verify_inputs
from pokemon_teacher_runtime.lifecycle import capture_tree, terminate_tree

REVIEW = ROOT / 'artifacts/reviews/f36-teacher-runtime'
ADMISSION = REVIEW / 'admission'
SCHEDULES = REVIEW / 'fixed-schedules-v1'
DEMOS = ROOT / 'artifacts/runs/f36-teacher-data-v1'
TRAIN = ROOT / 'artifacts/runs/f36-teacher-runtime-v1'
EXTERNAL = ROOT / 'artifacts/runs/f36-teacher-evaluation-v1'
VERIFY = REVIEW / 'sampled-verification-v1'
RESULT = REVIEW / 'measured-results-v1'
RECEIPT = ROOT / 'artifacts/reports/f36-teacher-runtime.json'
PLAN_RELATIVE = 'contracts/teacher-runtime-v1/plan.json'

OUTPUT_PATHS = {name: str(path.relative_to(ROOT)) for name, path in {
    'schedules': SCHEDULES, 'demonstrations': DEMOS, 'training': TRAIN,
    'external_evaluation': EXTERNAL, 'verification': VERIFY, 'results': RESULT}.items()}


def verify_plan_binding(manifest):
    path = ROOT / PLAN_RELATIVE
    if manifest['source_files'].get(PLAN_RELATIVE) != digest(path):
        raise ValueError('F36 CLI plan is not bound by admission')
    if json.loads(path.read_text()) != manifest['plan']:
        raise ValueError('F36 CLI plan differs from admission plan')
    original = json.loads((ROOT / 'contracts/teacher-assisted-v1/plan.json').read_text())
    if {k: v for k, v in manifest['plan'].items() if k != 'execution'} != original:
        raise ValueError('F36 changed the fixed scientific plan')


def verify_execution(manifest):
    verify_plan_binding(manifest)
    execution = manifest['plan']['execution']
    if execution['schema_version'] != 'f36-teacher-runtime-execution-v1':
        raise ValueError('F36 execution metadata schema')
    paths = {'failed_predecessor': 'artifacts/reports/f35-teacher-assisted-failure.json',
             'runtime_qualification': 'artifacts/reviews/f36-teacher-runtime/runtime-qualification-v1.json'}
    for key, relative in paths.items():
        entry = execution[key]
        if entry['path'] != relative or digest(ROOT / relative) != entry['sha256']:
            raise ValueError('F36 execution evidence binding: ' + key)
        record = json.loads((ROOT / relative).read_text())
        if record['status'] != ('failed' if key == 'failed_predecessor' else 'passed'):
            raise ValueError('F36 execution evidence status: ' + key)
    return execution


def now():
    return datetime.now(timezone.utc).isoformat()


def cleanup_direct_child(child):
    """Last-resort bounded cleanup of this Popen child; no descendant claim."""
    evidence = {'pid': child.pid, 'errors': [], 'term_sent': False, 'kill_sent': False}
    if child.poll() is None:
        try:
            child.terminate()
            evidence['term_sent'] = True
        except ProcessLookupError:
            pass
        except BaseException as exc:
            evidence['errors'].append('terminate: ' + repr(exc))
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        except BaseException as exc:
            evidence['errors'].append('wait after terminate: ' + repr(exc))
        if child.poll() is None:
            try:
                child.kill()
                evidence['kill_sent'] = True
            except ProcessLookupError:
                pass
            except BaseException as exc:
                evidence['errors'].append('kill: ' + repr(exc))
            try:
                child.wait(timeout=3)
            except BaseException as exc:
                evidence['errors'].append('wait after kill: ' + repr(exc))
    evidence['returncode'] = child.poll()
    evidence['direct_child_remaining'] = evidence['returncode'] is None
    evidence['descendant_cleanup_verified'] = False
    return evidence


def stage(name, arguments, report):
    before_deadline()
    if arguments[0] != '-m':
        raise ValueError('F36 stages must be Python modules')
    command = [str(ROOT / '.venv/bin/python'), '-m', 'pokemon_teacher_runtime.lifecycle',
        '--', 'pokemon_teacher_runtime.stage', *map(str, arguments[1:])]
    begun = time.perf_counter()
    launch = {'schema_version': 'f36-stage-v1', 'stage': name, 'command': command, 'started_utc': now()}
    timed_out, wait_error, snapshot = False, None, None
    with (REVIEW / f'{name}.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ,
            'PYTHONPATH': str(ROOT / 'python'), 'JAX_PLATFORMS': 'cpu',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        launch['pid'] = child.pid
        try:
            atomic_json(REVIEW / f'{name}-launch.json', launch)
            atomic_json(REVIEW / 'workflow-progress.json', {**launch, 'status': 'running'})
            snapshot = capture_tree(child)
            while child.poll() is None:
                snapshot = capture_tree(child, snapshot)
                remaining = DEADLINE - time.time()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    child.wait(timeout=min(1., remaining))
                except subprocess.TimeoutExpired:
                    continue
        except BaseException as exc:
            wait_error = repr(exc)
        finally:
            try:
                cleanup = terminate_tree(child, snapshot=snapshot, grace_seconds=10, kill_wait_seconds=5)
            except BaseException as exc:
                direct = cleanup_direct_child(child)
                cleanup = {'status': 'failed', 'root_pid': child.pid,
                    'returncode': child.poll(), 'errors': [repr(exc)],
                    'remaining_pids': [child.pid] if child.poll() is None else [],
                    'live_descendants_at_cleanup': [], 'cleanup_exception': True,
                    'direct_child_cleanup': direct, 'descendant_cleanup_unresolved': True}
            code = child.poll()
            atomic_json(REVIEW / f'{name}-cleanup.json', cleanup)
            leaked_after_success = (not timed_out and code == 0 and
                bool(cleanup['live_descendants_at_cleanup']))

    record = {**launch, 'exit_code': code, 'ended_utc': now(), 'deadline_timeout': timed_out,
        'elapsed_seconds': time.perf_counter() - begun, 'cleanup_status': cleanup['status'],
        'leaked_after_success': leaked_after_success, 'wait_error': wait_error}
    atomic_json(REVIEW / f'{name}-exit.json', record)
    if code != 0 or timed_out or wait_error or cleanup['status'] != 'passed' or leaked_after_success or not report.is_file():
        raise RuntimeError(f'{name}: exit={code}, deadline={timed_out}, report_exists={report.is_file()}')
    data = json.loads(report.read_text())
    if data.get('status') != 'passed':
        raise RuntimeError(name + ' report did not pass')
    return record


def main():
    before_deadline()
    with (REVIEW / 'workflow-claim.json').open('x') as out:
        json.dump({'pid': os.getpid(), 'started_utc': now()}, out)
    begun, stages, input_hash = time.perf_counter(), [], digest(ADMISSION / 'manifest.json')
    try:
        admitted = verify_inputs(ADMISSION)
        execution = verify_execution(admitted)
        for path in (SCHEDULES, DEMOS, TRAIN, EXTERNAL, VERIFY, RESULT, RECEIPT):
            if path.exists():
                raise ValueError('Existing scientific/output evidence; no restart: ' + str(path))
        stages.append(stage('fixed-schedules', ['-m', 'pokemon_teacher_assisted.schedules',
            '--admission', ADMISSION, '--output', SCHEDULES], SCHEDULES / 'report.json'))
        stages.append(stage('teacher-acquisition', ['-m', 'pokemon_teacher_assisted.demonstrations',
            '--admission', ADMISSION, '--plan', ROOT / PLAN_RELATIVE,
            '--output', DEMOS], DEMOS / 'report.json'))
        stages.append(stage('training', ['-m', 'pokemon_teacher_assisted.campaign',
            '--admission', ADMISSION, '--demonstrations', DEMOS, '--run', TRAIN], TRAIN / 'report.json'))
        stages.append(stage('external-evaluation', ['-m', 'pokemon_teacher_assisted.external_evaluation',
            '--admission', ADMISSION, '--demonstrations', DEMOS, '--training-run', TRAIN,
            '--output', EXTERNAL], EXTERNAL / 'report.json'))
        stages.append(stage('sampled-verification', ['-m', 'pokemon_teacher_assisted.verification',
            '--training-run', TRAIN, '--external-run', EXTERNAL, '--demonstrations', DEMOS,
            '--output', VERIFY], VERIFY / 'report.json'))
        stages.append(stage('measured-results', ['-m', 'pokemon_teacher_assisted.reporting',
            '--training-run', TRAIN, '--external-run', EXTERNAL, '--demonstrations', DEMOS,
            '--verification', VERIFY, '--output', RESULT], RESULT / 'summary.json'))
        admitted = verify_inputs(ADMISSION)
        if digest(ADMISSION / 'manifest.json') != input_hash:
            raise ValueError('Input manifest drift during F36')
        verify_execution(admitted)
        evidence = {entry['path']: entry['sha256'] for entry in
                    (execution['failed_predecessor'], execution['runtime_qualification'])}
        for base in (SCHEDULES, DEMOS, TRAIN, EXTERNAL, VERIFY, RESULT):
            for path in sorted(base.rglob('*')):
                if path.is_file():
                    evidence[str(path.relative_to(ROOT))] = digest(path)
        for entry in stages:
            for suffix in ('.log', '-launch.json', '-exit.json', '-cleanup.json'):
                path = REVIEW / (entry['stage'] + suffix)
                evidence[str(path.relative_to(ROOT))] = digest(path)
        receipt = {'schema_version': 'f36-teacher-runtime-receipt-v1', 'status': 'completed',
            'input_hash': input_hash, 'input_manifest': str((ADMISSION / 'manifest.json').relative_to(ROOT)),
            'source_files': admitted['source_files'], 'evidence_files': evidence,
            'execution': execution, 'output_paths': OUTPUT_PATHS,
            'summary': str((RESULT / 'summary.json').relative_to(ROOT)),
            'summary_sha256': digest(RESULT / 'summary.json'), 'stages': stages,
            'verification_scope': '114 sampled games, 12 original-next-update recoveries, exact accepted-label admission and explicit unavailable-choice engineering fixtures; not exhaustive replay or repeated teacher search.',
            'elapsed_seconds': time.perf_counter() - begun, 'completed_utc': now(),
            'learning_success_separate_from_execution': True, 'public_server_used': False,
            'final_reserved_data_used': False, 'promotion_authorized': False}
        receipt['content_hash'] = hashlib.sha256(canonical(receipt)).hexdigest()
        with RECEIPT.open('xb') as out:
            out.write(canonical(receipt) + b'\n')
        saved = REVIEW / 'f36-teacher-runtime-receipt.json'
        with saved.open('xb') as out:
            out.write(RECEIPT.read_bytes())
        if saved.read_bytes() != RECEIPT.read_bytes():
            raise ValueError('Receipt byte equality')
        for relative, expected in evidence.items():
            if digest(ROOT / relative) != expected:
                raise ValueError('Evidence drift during receipt: ' + relative)
        completion = {'schema_version': 'f36-completion-v1', 'status': 'passed', 'receipt': str(RECEIPT.relative_to(ROOT)),
            'receipt_sha256': digest(RECEIPT), 'input_hash': input_hash,
            'source_files_verified': len(admitted['source_files']), 'evidence_files_verified': len(evidence),
            'completed_utc': now()}
        atomic_json(REVIEW / 'completion.json', completion)
        atomic_json(REVIEW / 'workflow-exit.json', {**completion, 'schema_version': 'f36-workflow-exit-v1', 'exit_code': 0})
        print(json.dumps(completion), flush=True)
    except BaseException as exc:
        failure = {'schema_version': 'f36-workflow-failure-v1', 'status': 'failed', 'error': repr(exc), 'stages_completed': stages,
            'input_hash': input_hash, 'elapsed_seconds': time.perf_counter() - begun,
            'ended_utc': now(), 'completed_comparison': False}
        atomic_json(REVIEW / 'workflow-failure.json', failure)
        atomic_json(REVIEW / 'workflow-exit.json', {**failure, 'schema_version': 'f36-workflow-exit-v1', 'exit_code': 1})
        raise


if __name__ == '__main__':
    main()
