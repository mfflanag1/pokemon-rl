#!/usr/bin/env python3
"""Persistent F39 driver owner; bounded outer cleanup, no scientific retry."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_teacher_runtime.lifecycle import capture_tree, terminate_tree

REVIEW = ROOT / 'artifacts/reviews/f39-online-continuation'


def main():
    started = time.time()
    plan = json.loads((ROOT / 'contracts/online-continuation-v1/plan.json').read_text())
    stop = datetime.fromisoformat(plan['deadline_utc'].replace('Z', '+00:00')).timestamp()
    if started >= stop:
        raise TimeoutError('F39 launch after hard cutoff forbidden')
    with (REVIEW / 'launcher-claim.json').open('x') as out:
        json.dump({'schema_version': 'f39-launcher-claim-v1', 'pid': os.getpid(), 'started_unix': started}, out)
    child, snapshot, error, cleanup, code = None, None, None, None, 1
    try:
        # Reserve launch evidence before acquiring a live child.
        with (REVIEW / 'driver-launch.json').open('x') as launch, (REVIEW / 'driver.log').open('xb') as log:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/run-online-continuation.py')],
                cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'python'), 'JAX_PLATFORMS': 'cpu',
                    'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            json.dump({'schema_version': 'f39-driver-launch-v1', 'launcher_pid': os.getpid(),
                'driver_pid': child.pid, 'started_unix': started}, launch)
            launch.flush(); os.fsync(launch.fileno())
            while child.poll() is None:
                snapshot = capture_tree(child, snapshot)
                # Driver stops scientific admissions at stop; allow bounded cleanup.
                remaining = stop + 45 - time.time()
                if remaining <= 0:
                    raise TimeoutError('Driver exceeded cutoff plus cleanup grace')
                try:
                    child.wait(timeout=min(2., remaining))
                except subprocess.TimeoutExpired:
                    pass
            code = child.returncode
    except BaseException as exc:
        error = repr(exc)
    finally:
        if child is not None:
            try:
                cleanup = terminate_tree(child, snapshot=snapshot, grace_seconds=10, kill_wait_seconds=5)
            except BaseException as exc:
                errors = [repr(exc)]
                if child.poll() is None:
                    try:
                        child.kill()
                        child.wait(timeout=3)
                    except BaseException as direct:
                        errors.append(repr(direct))
                cleanup = {'status': 'failed', 'errors': errors, 'descendant_cleanup_unresolved': True}
            if cleanup['status'] != 'passed' or cleanup.get('live_descendants_at_cleanup'):
                code = 1
        if error:
            code = 1
        with (REVIEW / 'driver-exit.json').open('x') as out:
            json.dump({'schema_version': 'f39-driver-exit-v1', 'exit_code': code,
                'elapsed_seconds': time.time() - started, 'ended_unix': time.time(),
                'error': error, 'cleanup': cleanup}, out)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
