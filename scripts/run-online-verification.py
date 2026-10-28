#!/usr/bin/env python3
"""One additive F40 verification-only attempt, with a bounded outer owner and no retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'python'))
SPEC = importlib.util.spec_from_file_location('f40_checker', ROOT / 'scripts/check-online-verification.py')
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
REVIEW = ROOT / check.REVIEW
STOP = datetime.fromisoformat(check.DEADLINE.replace('Z', '+00:00')).timestamp()
ENV = {**os.environ, 'PYTHONPATH': str(ROOT / 'python'), 'JAX_PLATFORMS': 'cpu',
       'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    """All F40 evidence is exclusive, including failures; an attempt cannot restart."""
    with Path(path).open('xb') as stream:
        stream.write(check.canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())


def deadline():
    if time.time() >= STOP:
        raise TimeoutError('F40 original hard cutoff reached; no extension authorized')


def preservation():
    failure, original = check.verify_original()
    return failure, original, {'failure_receipt_sha256': check.FAILURE_SHA,
        'source_files_verified': check.SOURCE_COUNT, 'evidence_files_verified': check.EVIDENCE_COUNT}


def cleanup(child, snapshot):
    from pokemon_teacher_runtime.lifecycle import terminate_tree
    try:
        return terminate_tree(child, snapshot=snapshot, grace_seconds=10, kill_wait_seconds=5)
    except BaseException as exc:
        errors = [repr(exc)]
        if child.poll() is None:
            try:
                child.terminate(); child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    child.kill(); child.wait(timeout=3)
                except BaseException as direct:
                    errors.append(repr(direct))
            except BaseException as direct:
                errors.append(repr(direct))
        return {'status': 'failed', 'root_pid': child.pid, 'returncode': child.poll(), 'errors': errors,
            'remaining_pids': [child.pid] if child.poll() is None else [],
            'live_descendants_at_cleanup': [], 'descendant_cleanup_unresolved': True}


def stage(name, command, report):
    from pokemon_teacher_runtime.lifecycle import capture_tree
    deadline()
    check.require(command == check.expected_commands()[name], 'only the admitted verification/report commands may launch')
    start = time.perf_counter()
    row = {'schema_version': 'f40-stage-v1', 'stage': name, 'command': command, 'started_utc': now()}
    snapshot, timed_out, error, child = None, False, None, None
    # Reserve all launch/log paths before creating a child.
    with (REVIEW / (name + '-launch.json')).open('x') as launch, (REVIEW / (name + '.log')).open('xb') as log:
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row['pid'] = child.pid
            json.dump(row, launch); launch.flush(); os.fsync(launch.fileno())
            while child.poll() is None:
                snapshot = capture_tree(child, snapshot)
                remaining = STOP - time.time()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    child.wait(timeout=min(1., remaining))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as exc:
            error = repr(exc)
        finally:
            cleaned = cleanup(child, snapshot) if child is not None else {
                'status': 'failed', 'root_pid': None, 'returncode': None,
                'errors': ['stage spawn failed'], 'live_descendants_at_cleanup': [], 'remaining_pids': []}
            write(REVIEW / (name + '-cleanup.json'), cleaned)
    code = child.poll() if child is not None else None
    record = {**row, 'exit_code': code, 'ended_utc': now(), 'deadline_timeout': timed_out,
        'elapsed_seconds': time.perf_counter() - start, 'cleanup_status': cleaned['status'],
        'leaked_after_success': code == 0 and bool(cleaned['live_descendants_at_cleanup']), 'wait_error': error}
    write(REVIEW / (name + '-exit.json'), record)
    check.require(code == 0 and not timed_out and error is None and cleaned['status'] == 'passed'
                  and not record['leaked_after_success'] and report.is_file(), 'failed new stage: ' + name)
    check.require(check.read(report).get('status') == 'passed', 'new stage report failed: ' + name)
    deadline()
    return record


def worker():
    """Child may run only the two admitted stages; outer process owns final sealing."""
    deadline()
    write(REVIEW / 'workflow-claim.json', {'schema_version': 'f40-workflow-claim-v1', 'pid': os.getpid(), 'started_utc': now()})
    begun, stages = time.perf_counter(), []
    try:
        owner = check.read(REVIEW / 'driver-launch.json')
        check.require(owner['driver_pid'] == os.getpid() and owner['launcher_pid'] == os.getppid()
                      and check.read(REVIEW / 'launcher-claim.json')['pid'] == os.getppid(),
                      'worker must belong to the exclusive outer launcher')
        failure, original, before = preservation()
        admitted = check.verify_admission(failure)
        admission_sha = check.sha(ROOT / check.ADMISSION)
        from pokemon_full_train.storage import runtime
        runtime_before = runtime()
        check.require(runtime_before == original['runtime'], 'qualified original runtime before')
        for path in (check.VERIFY, check.RESULT, check.RECEIPT):
            check.require(not (ROOT / path).exists(), 'existing output; no restart: ' + path)
        for name, command in check.expected_commands().items():
            filename = 'report.json' if name == 'sampled-verification' else 'summary.json'
            directory = check.VERIFY if name == 'sampled-verification' else check.RESULT
            stages.append(stage(name, command, ROOT / directory / filename))
        _, _, after = preservation()
        check.require(check.verify_admission(failure) == admitted and check.sha(ROOT / check.ADMISSION) == admission_sha,
                      'new admission changed during execution')
        runtime_after = runtime()
        check.require(runtime_before == runtime_after, 'runtime changed during execution')
        deadline()
        write(REVIEW / 'workflow-result.json', {'schema_version': 'f40-workflow-result-v1', 'status': 'passed',
            'stages': stages, 'runtime_before': runtime_before, 'runtime_after': runtime_after,
            'preservation_before': before, 'preservation_after': after,
            'admission_sha256': admission_sha, 'elapsed_seconds': time.perf_counter() - begun, 'ended_utc': now()})
        return 0
    except BaseException as exc:
        write(REVIEW / 'workflow-failure.json', {'schema_version': 'f40-workflow-failure-v1', 'status': 'failed',
            'error': repr(exc), 'stages_completed': stages, 'elapsed_seconds': time.perf_counter() - begun,
            'original_workflow_status': 'failed', 'completed_comparison': False, 'ended_utc': now()})
        raise


def inventory(exclusions=()):
    return {check.REVIEW + '/' + relative: check.sha(path)
            for relative, path in sorted(check.files(REVIEW).items())
            if check.REVIEW + '/' + relative not in exclusions}


def seal_success():
    """Seal after the owner has written its actual exit and cleanup evidence."""
    deadline()
    failure, original, _ = preservation()
    admission = check.verify_admission(failure)
    result = check.read(REVIEW / 'workflow-result.json')
    check.validate_outer_execution(result['stages'], check.sha(ROOT / check.ADMISSION), original)
    training = check.read(ROOT / check.TRAIN / 'report.json')
    verification = check.read(ROOT / check.VERIFY / 'report.json')
    summary = check.read(ROOT / check.RESULT / 'summary.json')
    check.validate_reports(training, verification, summary, original)
    check.validate_policy(verification, check.read(ROOT / admission['guard_policy']['path']), original)
    stage_records = [{name: check.read(REVIEW / (name + suffix)) for name in check.STAGES}
                     for suffix in ('-exit.json', '-cleanup.json', '-launch.json')]
    check.validate_stages(result['stages'], *stage_records)
    evidence = inventory(check.SEAL_EXCLUSIONS)
    for record in (verification, summary):
        check.validate_inputs(record, evidence, failure['evidence_files'])
    receipt = {'schema_version': 'f40-online-verification-receipt-v1', 'status': 'completed',
        'admission_manifest': check.ADMISSION, 'admission_sha256': check.sha(ROOT / check.ADMISSION),
        'source_files': admission['source_files'], 'evidence_files': evidence,
        'original_failure_receipt': admission['original_failure_receipt'], 'original_workflow_status': 'failed',
        'deadline_utc': check.DEADLINE, 'output_paths': {'inherited_training': check.TRAIN, 'verification': check.VERIFY, 'results': check.RESULT},
        'inherited_training': {'report': check.TRAIN + '/report.json', 'sha256': check.sha(ROOT / check.TRAIN / 'report.json'),
            'original_stage': failure['training_exit'], 'counts': training['counts'], 'seeds': list(check.old.SEEDS)},
        'stages': result['stages'], 'summary': check.RESULT + '/summary.json',
        'summary_sha256': check.sha(ROOT / check.RESULT / 'summary.json'),
        'new_work': {'training_cohorts': 0, 'selected_training_rows': 0, 'training_optimizer_steps': 0,
                     'fresh_replay_games': 66, 'recomputed_optimizer_steps': 48},
        'verification_scope': '66 sampled fresh local games, 6 original-next-update recoveries (48 recomputed optimizer steps), 9 checkpoint comparisons; all original ledgers and 12 contrasts checked. No new training.',
        'completed_utc': now(), 'learning_success_separate_from_execution': True,
        'teacher_used': False, 'demonstrations_used': False, 'public_server_used': False,
        'final_reserved_data_used': False, 'promotion_authorized': False}
    preservation(); check.verify_admission(failure); deadline()
    check.require(inventory(check.SEAL_EXCLUSIONS) == evidence, 'new evidence changed before seal')
    receipt['content_hash'] = hashlib.sha256(check.canonical(receipt)).hexdigest()
    write(ROOT / check.RECEIPT, receipt)
    with (REVIEW / 'f40-online-verification-receipt.json').open('xb') as stream:
        stream.write((ROOT / check.RECEIPT).read_bytes()); stream.flush(); os.fsync(stream.fileno())
    write(REVIEW / 'completion.json', {'schema_version': 'f40-completion-v1', 'status': 'passed',
        'receipt': check.RECEIPT, 'receipt_sha256': check.sha(ROOT / check.RECEIPT),
        'admission_sha256': receipt['admission_sha256'], 'source_files_verified': len(admission['source_files']),
        'evidence_files_verified': len(evidence), 'completed_utc': now()})
    check.verify_completion()
    print(json.dumps({'status': 'completed', 'receipt': check.RECEIPT, 'original_workflow_status': 'failed'}), flush=True)


def seal_failure(error):
    """Preserve the actual new partial attempt; never synthesize F39 success evidence."""
    target = ROOT / check.FAILURE_RECEIPT
    if target.exists():
        raise FileExistsError('F40 failure receipt already exists; no restart')
    try:
        _, _, original = preservation()
    except BaseException as exc:
        original = {'status': 'failed', 'error': repr(exc)}
    evidence = inventory({check.REVIEW + '/f40-online-verification-failure-receipt.json'})
    partial_success = {relative: check.sha(ROOT / relative) for relative in (check.RECEIPT,
        check.REVIEW + '/f40-online-verification-receipt.json', check.REVIEW + '/completion.json')
        if (ROOT / relative).is_file()}
    receipt = {'schema_version': 'f40-online-verification-failure-v1', 'status': 'failed', 'error': repr(error),
        'original_workflow_status': 'failed', 'completed_comparison': False, 'scientific_completion': False,
        'no_retry': True, 'partial_success_artifacts': partial_success, 'candidate_success_authoritative': False,
        'seal_scope': 'Any candidate success files are incomplete or invalidated; the failed attempt is authoritative.',
        'deadline_utc': check.DEADLINE, 'original_preservation': original, 'evidence_files': evidence,
        'new_stages': [check.read(REVIEW / (name + '-exit.json')) for name in check.STAGES
                       if (REVIEW / (name + '-exit.json')).exists()], 'ended_utc': now()}
    receipt['content_hash'] = hashlib.sha256(check.canonical(receipt)).hexdigest()
    write(target, receipt)
    with (REVIEW / 'f40-online-verification-failure-receipt.json').open('xb') as stream:
        stream.write(target.read_bytes()); stream.flush(); os.fsync(stream.fileno())


def main():
    from pokemon_teacher_runtime.lifecycle import capture_tree
    deadline()
    failure, _, _ = preservation()
    check.verify_admission(failure)
    for relative in (check.VERIFY, check.RESULT, check.RECEIPT, 'artifacts/reports/f40-online-verification-failure.json'):
        check.require(not (ROOT / relative).exists(), 'existing attempt output; no restart: ' + relative)
    started = time.time()
    write(REVIEW / 'launcher-claim.json', {'schema_version': 'f40-launcher-claim-v1', 'pid': os.getpid(), 'started_unix': started})
    child, snapshot, error, cleaned, code = None, None, None, None, 1
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/run-online-verification.py'), '--worker']
    try:
        with (REVIEW / 'driver-launch.json').open('x') as launch, (REVIEW / 'driver.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            json.dump({'schema_version': 'f40-driver-launch-v1', 'launcher_pid': os.getpid(),
                'driver_pid': child.pid, 'command': command, 'started_unix': started}, launch)
            launch.flush(); os.fsync(launch.fileno())
            while child.poll() is None:
                snapshot = capture_tree(child, snapshot)
                remaining = STOP + 45 - time.time()
                if remaining <= 0:
                    raise TimeoutError('F40 driver exceeded original cutoff plus bounded cleanup grace')
                try:
                    child.wait(timeout=min(1., remaining))
                except subprocess.TimeoutExpired:
                    pass
            code = child.returncode
    except BaseException as exc:
        error = repr(exc)
    finally:
        if child is not None:
            cleaned = cleanup(child, snapshot)
            if cleaned['status'] != 'passed' or cleaned.get('live_descendants_at_cleanup'):
                code = 1
        if error:
            code = 1
        write(REVIEW / 'driver-exit.json', {'schema_version': 'f40-driver-exit-v1', 'exit_code': code,
            'elapsed_seconds': time.time() - started, 'ended_unix': time.time(), 'error': error, 'cleanup': cleaned})
    try:
        if code != 0:
            raise RuntimeError('F40 worker/outer cleanup failed: ' + str(error))
        seal_success()
    except BaseException as exc:
        seal_failure(exc)
        raise
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(worker() if args.worker else main())
