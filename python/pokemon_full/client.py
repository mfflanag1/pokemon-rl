"""Synchronous bounded JSON-lines transport. No server, socket, or hidden state API."""
import json
from pathlib import Path
import select
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


class NodeWorker:
    def __init__(self, timeout=60):
        self.timeout, self.sequence = timeout, 0
        self.errors = tempfile.TemporaryFile(mode='w+t')
        started = time.perf_counter()
        self.process = subprocess.Popen(['node', 'packages/full-showdown/worker.mjs'], cwd=ROOT,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.errors, text=True, bufsize=1)
        try:
            self.hello = self.request('hello')
            self.initialization_seconds = time.perf_counter()-started
        except Exception:
            self.close(force=True)
            raise

    def request(self, op, **payload):
        self.sequence += 1
        self.process.stdin.write(json.dumps({'id': self.sequence, 'op': op, **payload}, allow_nan=False)+'\n')
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not ready:
            raise TimeoutError('Full-Showdown worker timed out; episode is incomplete')
        line = self.process.stdout.readline()
        if not line:
            self.errors.seek(0)
            raise RuntimeError('Full-Showdown worker exited: '+self.errors.read()[-2000:])
        response = json.loads(line)
        if response.get('id') != self.sequence or not response.get('ok'):
            raise RuntimeError(f'Worker command failed: {response}')
        return response['data']

    def close(self, force=False):
        if self.process.poll() is None:
            if not force:
                try:
                    self.request('close')
                    self.process.stdin.close()
                    self.process.wait(timeout=5)
                except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    force = True
            if force and self.process.poll() is None:
                self.process.kill(); self.process.wait(timeout=5)
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.close(force=exc_type is not None)
