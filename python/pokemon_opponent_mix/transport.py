"""Bounded wire transport with retained failure payloads; no silent game skipping."""
import json
import os
from pathlib import Path
import select
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


class Worker:
    def __init__(self, evidence, timeout=60):
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.sequence, self.timeout, self.buffer, self.last = 0, timeout, b'', None
        self.stderr = open(self.evidence / f'worker-{time.time_ns()}.stderr', 'xb')
        self.process = None
        try:
            self.process = subprocess.Popen(['node', 'packages/opponent-mix/worker.mjs'], cwd=ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0)
            self.hello = self.request('hello')
            if self.hello['schema_version'] != 'opponent-mix-worker-v1':
                raise ValueError('Wrong worker version')
        except BaseException as exc:
            try:
                (self.evidence/f'initialization-failure-{time.time_ns()}.json').write_text(json.dumps({'stage':'worker-initialization','error':repr(exc)}))
            finally:
                self.close()
            raise

    def request(self, op, **payload):
        self.sequence += 1
        command = {'id': self.sequence, 'op': op, **payload}
        deadline = time.monotonic() + self.timeout
        try:
            self.process.stdin.write((json.dumps(command, allow_nan=False)+'\n').encode())
            self.process.stdin.flush()
            while b'\n' not in self.buffer:
                remaining = deadline-time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError('Worker response deadline exceeded')
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError('Worker exited before complete response')
                self.buffer += block
                if len(self.buffer) > 4_000_000:
                    raise RuntimeError('Worker response exceeds limit')
            raw, self.buffer = self.buffer.split(b'\n', 1)
            response = json.loads(raw)
            if response.get('id') != self.sequence or response.get('ok') is not True:
                raise RuntimeError(f'Worker rejected command: {response}')
            data = response['data']
            if op in ('start','step','cutoff'):
                if data.get('schema_version') != 'opponent-mix-frame-v1' or set(data.get('players', {})) != {'p1','p2'}:
                    raise ValueError('Wrong dynamic frame version or sides')
                if any(data['players'][side].get('schema_version') != 'full-showdown-player-v2' for side in ['p1','p2']):
                    raise ValueError('Wrong dynamic player version')
            self.last = data
            return self.last
        except BaseException as exc:
            record = {'command': command, 'previous_response': self.last, 'error': repr(exc),
                      'partial_response': self.buffer.decode(errors='replace')[-10000:]}
            (self.evidence / f'failure-{time.time_ns()}.json').write_text(json.dumps(record, indent=2))
            self.close()
            raise

    def close(self):
        if self.process is None:
            self.stderr.close()
            return
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        for stream in [self.process.stdin, self.process.stdout, self.stderr]:
            if stream and not stream.closed:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
