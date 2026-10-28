#!/usr/bin/env python3
"""One persistent F34 driver. No retries, resumption, or outcome-dependent changes."""
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
from pokemon_structured_policy.common import verify_inputs, before_deadline

REVIEW = ROOT / 'artifacts/reviews/f34-policy-reset'
ADMISSION = REVIEW / 'admission'
INITIAL = ROOT / 'artifacts/runs/f34-external-initial-v1'
TRAIN = ROOT / 'artifacts/runs/f34-policy-reset-v1'
FINAL = ROOT / 'artifacts/runs/f34-external-final-v1'
VERIFY = REVIEW / 'sampled-verification-v1'
RESULT = REVIEW / 'measured-results-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def stage(name, arguments, report):
    before_deadline()
    command = [str(ROOT / '.venv/bin/python'), *arguments]
    started = time.perf_counter()
    launch = {'stage': name, 'command': command, 'started_utc': now()}
    with (REVIEW / f'{name}.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ,
            'PYTHONPATH': str(ROOT / 'python'), 'JAX_PLATFORMS': 'cpu',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, stdout=log, stderr=subprocess.STDOUT)
        launch['pid'] = child.pid
        atomic_json(REVIEW / f'{name}-launch.json', launch)
        atomic_json(REVIEW / 'workflow-progress.json', {**launch, 'status': 'running'})
        code = child.wait()
    exit_record = {**launch, 'exit_code': code, 'ended_utc': now(),
        'elapsed_seconds': time.perf_counter() - started}
    atomic_json(REVIEW / f'{name}-exit.json', exit_record)
    if code != 0 or not report.is_file():
        raise RuntimeError(f'{name} exited {code}; required report exists={report.is_file()}')
    data = json.loads(report.read_text())
    if data.get('status') != 'passed':
        raise RuntimeError(f'{name} did not produce a passed completion report')
    return exit_record


def main():
    before_deadline()
    # Atomic ownership and unique output paths make accidental second launches fail closed.
    with (REVIEW / 'workflow-claim.json').open('x') as claim:
        json.dump({'pid': os.getpid(), 'started_utc': now()}, claim)
    begun, stages = time.perf_counter(), []
    input_hash = digest(ADMISSION / 'manifest.json')
    try:
        verify_inputs(ADMISSION)
        for path in (INITIAL, TRAIN, FINAL, VERIFY, RESULT):
            if path.exists():
                raise ValueError('Existing scientific/verification output; no resume: ' + str(path))
        external_args = ['-m', 'pokemon_structured_policy.external', '--plan',
            str(ROOT / 'contracts/policy-reset-v1/external.json'), '--admission', str(ADMISSION)]
        stages.append(stage('external-initial', [*external_args, '--run', str(INITIAL),
            '--endpoint', 'initial'], INITIAL / 'report.json'))
        stages.append(stage('training', ['-m', 'pokemon_structured_policy.campaign', '--run', str(TRAIN),
            '--admission', str(ADMISSION)], TRAIN / 'report.json'))
        stages.append(stage('external-final', [*external_args, '--run', str(FINAL), '--endpoint', 'final',
            '--training-run', str(TRAIN)], FINAL / 'report.json'))
        stages.append(stage('sampled-verification', ['-m', 'pokemon_structured_policy.verification',
            '--training-run', str(TRAIN), '--external-initial', str(INITIAL),
            '--external-final', str(FINAL), '--output', str(VERIFY)], VERIFY / 'report.json'))
        stages.append(stage('measured-results', ['-m', 'pokemon_structured_policy.reporting',
            '--training-run', str(TRAIN), '--external-initial', str(INITIAL),
            '--external-final', str(FINAL), '--verification', str(VERIFY), '--output', str(RESULT)],
            RESULT / 'summary.json'))
        admitted = verify_inputs(ADMISSION)
        if digest(ADMISSION / 'manifest.json') != input_hash:
            raise ValueError('Frozen input manifest drift')
        # Independent content hashing after all scientific writers have exited.
        evidence = {}
        for base in (INITIAL, TRAIN, FINAL, VERIFY, RESULT):
            for path in sorted(base.rglob('*')):
                if path.is_file():
                    evidence[str(path.relative_to(ROOT))] = digest(path)
        for entry in stages:
            for suffix in ('.log', '-launch.json', '-exit.json'):
                path = REVIEW / (entry['stage'] + suffix)
                evidence[str(path.relative_to(ROOT))] = digest(path)
        receipt = {'schema_version': 'f34-policy-reset-receipt-v1', 'status': 'completed',
            'input_hash': input_hash, 'input_manifest': str((ADMISSION / 'manifest.json').relative_to(ROOT)),
            'source_files': admitted['source_files'], 'evidence_files': evidence,
            'summary': str((RESULT / 'summary.json').relative_to(ROOT)),
            'summary_sha256': digest(RESULT / 'summary.json'),
            'verification_scope': '66 sampled fresh/recorded-action games and six original-next-update recoveries; not exhaustive replay',
            'stages': stages, 'elapsed_seconds': time.perf_counter() - begun,
            'completed_utc': now(), 'learning_success_separate_from_execution': True,
            'public_server_used': False, 'final_reserved_data_used': False, 'promotion_authorized': False}
        receipt['content_hash'] = hashlib.sha256(canonical(receipt)).hexdigest()
        path = ROOT / 'artifacts/reports/f34-policy-reset.json'
        if path.exists():
            raise ValueError('Refusing to replace existing F34 receipt')
        atomic_json(path, receipt)
        saved = REVIEW / 'f34-policy-reset-receipt.json'
        with saved.open('xb') as target:
            target.write(path.read_bytes())
        assert path.read_bytes() == saved.read_bytes()
        for rel, expected in evidence.items():
            if digest(ROOT / rel) != expected:
                raise ValueError('Evidence changed during receipt verification: ' + rel)
        completion = {'status': 'passed', 'receipt': str(path.relative_to(ROOT)),
            'receipt_sha256': digest(path), 'input_hash': input_hash,
            'source_files_verified': len(admitted['source_files']),
            'evidence_files_verified': len(evidence), 'completed_utc': now()}
        atomic_json(REVIEW / 'completion.json', completion)
        atomic_json(REVIEW / 'workflow-exit.json', {'exit_code': 0, **completion})
        print(json.dumps(completion), flush=True)
    except BaseException as exc:
        failure = {'status': 'failed', 'error': repr(exc), 'stages_completed': stages,
            'elapsed_seconds': time.perf_counter() - begun, 'ended_utc': now(),
            'input_hash': input_hash, 'completed_screen': False}
        atomic_json(REVIEW / 'workflow-failure.json', failure)
        atomic_json(REVIEW / 'workflow-exit.json', {'exit_code': 1, **failure})
        raise


if __name__ == '__main__':
    main()
