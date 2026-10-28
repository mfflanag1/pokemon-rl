"""Reuse the bounded F4 transport; change only the additive worker entry point."""
from pathlib import Path
import subprocess
import time
from pokemon_full_learning.transport import Worker as BaseWorker, ROOT


class Worker(BaseWorker):
    def __init__(self, evidence, timeout=60):
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.sequence, self.timeout, self.buffer, self.last = 0, timeout, b'', None
        self.stderr = open(self.evidence / f'worker-{time.time_ns()}.stderr', 'xb')
        self.process = subprocess.Popen(['node', 'packages/switch-audit/worker.mjs'], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0)
        try:
            self.hello = self.request('hello')
            if self.hello != {'schema_version': 'switch-audit-worker-v1'}:
                raise ValueError('Wrong audit worker')
        except BaseException:
            self.close()
            raise
