#!/usr/bin/env python3
"""Persistent owner of the one F36 driver, including early-exit evidence."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'artifacts/reviews/f36-teacher-runtime'


def main():
    started = time.time()
    with (REVIEW / 'launcher-claim.json').open('x') as out:
        json.dump({'schema_version': 'f36-launcher-claim-v1', 'pid': os.getpid(), 'started_unix': started}, out)
    with (REVIEW / 'driver.log').open('xb') as log:
        child = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/run-teacher-runtime.py')],
            cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'python'), 'JAX_PLATFORMS': 'cpu',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, stdout=log, stderr=subprocess.STDOUT)
        with (REVIEW / 'driver-launch.json').open('x') as out:
            json.dump({'schema_version': 'f36-driver-launch-v1', 'launcher_pid': os.getpid(), 'driver_pid': child.pid, 'started_unix': started}, out)
        code = child.wait()
    with (REVIEW / 'driver-exit.json').open('x') as out:
        json.dump({'schema_version': 'f36-driver-exit-v1', 'exit_code': code, 'elapsed_seconds': time.time() - started, 'ended_unix': time.time()}, out)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
