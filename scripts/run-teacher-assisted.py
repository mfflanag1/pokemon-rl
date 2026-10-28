#!/usr/bin/env python3
"""One bounded F35 workflow; exclusive outputs, no scientific retry or resumption."""
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
from pokemon_teacher_assisted.lifecycle import capture_tree, terminate_tree

REVIEW = ROOT / 'artifacts/reviews/f35-teacher-assisted'
ADMISSION = REVIEW / 'admission'
SCHEDULES = REVIEW / 'fixed-schedules-v1'
DEMOS = ROOT / 'artifacts/runs/f35-teacher-data-v1'
TRAIN = ROOT / 'artifacts/runs/f35-teacher-assisted-v1'
EXTERNAL = ROOT / 'artifacts/runs/f35-teacher-evaluation-v1'
VERIFY = REVIEW / 'sampled-verification-v1'
RESULT = REVIEW / 'measured-results-v1'
RECEIPT = ROOT / 'artifacts/reports/f35-teacher-assisted.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def stage(name, arguments, report):
    before_deadline()
    if arguments[0] != '-m':
        raise ValueError('F35 stages must be Python modules')
    command = [str(ROOT / '.venv/bin/python'), '-m', 'pokemon_teacher_assisted.lifecycle',
        '--', *map(str, arguments[1:])]
    begun = time.perf_counter()
    launch = {'stage': name, 'command': command, 'started_utc': now()}
    timed_out = False
    with (REVIEW / f'{name}.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ,
            'PYTHONPATH': str(ROOT / 'python'), 'JAX_PLATFORMS': 'cpu',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        launch['pid'] = child.pid
        atomic_json(REVIEW / f'{name}-launch.json', launch)
        atomic_json(REVIEW / 'workflow-progress.json', {**launch, 'status': 'running'})
        snapshot = capture_tree(child)
        wait_error = None
        try:
            while child.poll() is None:
                capture_tree(child, snapshot)
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
        cleanup = terminate_tree(child, snapshot=snapshot, grace_seconds=10, kill_wait_seconds=5)
        code = child.returncode
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
        verify_inputs(ADMISSION)
        for path in (SCHEDULES, DEMOS, TRAIN, EXTERNAL, VERIFY, RESULT, RECEIPT):
            if path.exists():
                raise ValueError('Existing scientific/output evidence; no restart: ' + str(path))
        stages.append(stage('fixed-schedules', ['-m', 'pokemon_teacher_assisted.schedules',
            '--admission', ADMISSION, '--output', SCHEDULES], SCHEDULES / 'report.json'))
        stages.append(stage('teacher-acquisition', ['-m', 'pokemon_teacher_assisted.demonstrations',
            '--admission', ADMISSION, '--plan', ROOT / 'contracts/teacher-assisted-v1/plan.json',
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
            raise ValueError('Input manifest drift during F35')
        evidence = {}
        for base in (SCHEDULES, DEMOS, TRAIN, EXTERNAL, VERIFY, RESULT):
            for path in sorted(base.rglob('*')):
                if path.is_file():
                    evidence[str(path.relative_to(ROOT))] = digest(path)
        for entry in stages:
            for suffix in ('.log', '-launch.json', '-exit.json', '-cleanup.json'):
                path = REVIEW / (entry['stage'] + suffix)
                evidence[str(path.relative_to(ROOT))] = digest(path)
        receipt = {'schema_version': 'f35-teacher-assisted-receipt-v1', 'status': 'completed',
            'input_hash': input_hash, 'input_manifest': str((ADMISSION / 'manifest.json').relative_to(ROOT)),
            'source_files': admitted['source_files'], 'evidence_files': evidence,
            'summary': str((RESULT / 'summary.json').relative_to(ROOT)),
            'summary_sha256': digest(RESULT / 'summary.json'), 'stages': stages,
            'verification_scope': '114 sampled games, 12 original-next-update recoveries, exact accepted-label admission and explicit unavailable-choice engineering fixtures; not exhaustive replay or repeated teacher search.',
            'elapsed_seconds': time.perf_counter() - begun, 'completed_utc': now(),
            'learning_success_separate_from_execution': True, 'public_server_used': False,
            'final_reserved_data_used': False, 'promotion_authorized': False}
        receipt['content_hash'] = hashlib.sha256(canonical(receipt)).hexdigest()
        with RECEIPT.open('xb') as out:
            out.write(canonical(receipt) + b'\n')
        saved = REVIEW / 'f35-teacher-assisted-receipt.json'
        with saved.open('xb') as out:
            out.write(RECEIPT.read_bytes())
        if saved.read_bytes() != RECEIPT.read_bytes():
            raise ValueError('Receipt byte equality')
        for relative, expected in evidence.items():
            if digest(ROOT / relative) != expected:
                raise ValueError('Evidence drift during receipt: ' + relative)
        completion = {'status': 'passed', 'receipt': str(RECEIPT.relative_to(ROOT)),
            'receipt_sha256': digest(RECEIPT), 'input_hash': input_hash,
            'source_files_verified': len(admitted['source_files']), 'evidence_files_verified': len(evidence),
            'completed_utc': now()}
        atomic_json(REVIEW / 'completion.json', completion)
        atomic_json(REVIEW / 'workflow-exit.json', {'exit_code': 0, **completion})
        print(json.dumps(completion), flush=True)
    except BaseException as exc:
        failure = {'status': 'failed', 'error': repr(exc), 'stages_completed': stages,
            'input_hash': input_hash, 'elapsed_seconds': time.perf_counter() - begun,
            'ended_utc': now(), 'completed_comparison': False}
        atomic_json(REVIEW / 'workflow-failure.json', failure)
        atomic_json(REVIEW / 'workflow-exit.json', {'exit_code': 1, **failure})
        raise


if __name__ == '__main__':
    main()
