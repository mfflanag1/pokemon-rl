"""Real subprocess checks for additive F36 EOF/constructor cleanup only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_teacher_runtime import transport


PYTHON_WORKER = r'''
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'id': request['id'], 'ok': True, 'data': request['op']}), flush=True)
'''
NODE_WORKER = r'''
const readline = require('node:readline');
const input = readline.createInterface({input: process.stdin});
input.on('line', line => {
  const request = JSON.parse(line);
  process.stdout.write(JSON.stringify({id: request.id, ok: true, data: request.op}) + '\n');
});
'''


class Shutdown(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='f36-eof-')
        self.addCleanup(self.folder.cleanup)
        self.children = []
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        for child, kill in self.children:
            if child.poll() is None:
                kill()
            child.wait(timeout=5)
            for handle in (child.stdin, child.stdout):
                if handle is not None:
                    handle.close()

    def track(self, child):
        self.children.append((child, child.kill))
        return child

    def path(self, name):
        return Path(self.folder.name) / name

    def assert_closed(self, worker):
        self.assertTrue(worker.closed)
        for handle in (worker.process.stdin, worker.process.stdout, worker.stderr):
            self.assertTrue(handle.closed)

    def test_python_and_node_workers_exit_cleanly_on_eof_without_signals(self):
        for name, command in [('python', [sys.executable, '-u', '-c', PYTHON_WORKER]),
                              ('node', ['node', '-e', NODE_WORKER])]:
            with self.subTest(worker=name):
                worker = transport.JsonWorker(command, self.path(name + '.stderr'))
                self.track(worker.process)
                self.assertEqual(worker.request('ready'), 'ready')
                with patch.object(worker.process, 'terminate', wraps=worker.process.terminate) as term, \
                     patch.object(worker.process, 'kill', wraps=worker.process.kill) as kill:
                    self.assertEqual(worker.close(), [])
                    self.assertEqual(worker.close(), [])
                term.assert_not_called()
                kill.assert_not_called()
                self.assertEqual(worker.process.returncode, 0)
                self.assert_closed(worker)

    def test_constructor_failure_after_launch_reaps_child_and_closes_pipes(self):
        real_popen, real_set_blocking = subprocess.Popen, os.set_blocking
        created = []

        def create(*args, **kwargs):
            child = self.track(real_popen(*args, **kwargs))
            created.append((child, kwargs['stderr']))
            return child

        calls = []
        original = OSError(5, 'injected second set_blocking failure')

        def blocking(fd, value):
            calls.append(fd)
            if len(calls) == 2:
                raise original
            return real_set_blocking(fd, value)

        with patch.object(transport.subprocess, 'Popen', create), \
             patch.object(transport.os, 'set_blocking', blocking):
            with self.assertRaises(OSError) as caught:
                transport.JsonWorker([sys.executable, '-u', '-c', PYTHON_WORKER], self.path('init.stderr'))
        self.assertIs(caught.exception, original)
        self.assertEqual(len(created), 1)
        child, stderr = created[0]
        self.assertEqual(child.returncode, 0)
        self.assertTrue(child.stdin.closed and child.stdout.closed and stderr.closed)

    def test_constructor_cleanup_permission_errors_remain_secondary(self):
        real_popen = subprocess.Popen
        created = []
        denied = PermissionError(1, 'injected owned-child signal denial')
        original = OSError(5, 'injected setup failure')

        def create(*args, **kwargs):
            child = self.track(real_popen(*args, **kwargs))
            created.append((child, kwargs['stderr']))

            def deny():
                raise denied

            child.terminate = deny
            child.kill = deny
            return child

        with patch.object(transport.subprocess, 'Popen', create), \
             patch.object(transport.os, 'set_blocking', side_effect=original):
            with self.assertRaises(OSError) as caught:
                transport.JsonWorker([sys.executable, '-c', 'import time; time.sleep(30)'],
                    self.path('denied.stderr'), eof_wait=.05, term_wait=.05, kill_wait=.05)
        self.assertIs(caught.exception, original)
        self.assertTrue(any('PermissionError' in error for error in caught.exception.cleanup_errors))
        child, stderr = created[0]
        self.assertIsNone(child.poll())  # Emergency cleanup uses its saved owned Popen handle.
        self.assertTrue(child.stdin.closed and child.stdout.closed and stderr.closed)


if __name__ == '__main__':
    unittest.main()
