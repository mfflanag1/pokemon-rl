"""Synthetic process trees only: no simulators, native search or games."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from pokemon_teacher_assisted import lifecycle

ROOT = Path(__file__).resolve().parents[1]


def wait_file(path, child=None):
    deadline = time.monotonic() + 5.
    while not path.exists():
        if child is not None and child.poll() is not None:
            raise AssertionError('Fixture exited before readiness: ' + str(child.returncode))
        if time.monotonic() >= deadline:
            raise AssertionError('Fixture readiness timeout: ' + str(path))
        time.sleep(.01)


def start_tree(folder, *, normal=False):
    grandchild = """import os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1], 'grandchild.pid').write_text(str(os.getpid()))
while True: time.sleep(.1)
"""
    worker = f"""import os, pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen([sys.executable, '-c', {grandchild!r}, sys.argv[1]], start_new_session=True)
pathlib.Path(sys.argv[1], 'worker.pid').write_text(str(os.getpid()))
while True: time.sleep(.1)
"""
    stage = f"""import pathlib, subprocess, sys, time
root=pathlib.Path(sys.argv[1]); root.joinpath('stage.ready').write_text('ready')
try:
    while not root.joinpath('go').exists(): time.sleep(.01)
    subprocess.Popen([sys.executable, '-c', {worker!r}, str(root)], start_new_session=True)
    while not ({normal!r} and root.joinpath('normal-exit').exists()): time.sleep(.01)
finally:
    root.joinpath('unwound').write_text('finally ran')
"""
    (folder / '_f35_lifecycle_fixture.py').write_text(stage)
    child = subprocess.Popen([sys.executable, '-m', 'pokemon_teacher_assisted.lifecycle', '--',
        '_f35_lifecycle_fixture', str(folder)], env={**os.environ, 'PYTHONPATH': str(folder) + os.pathsep + str(ROOT / 'python')},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return child


def emergency(child, snapshot):
    if snapshot:
        for previous in snapshot['owned'].values():
            current = lifecycle.process_identity(previous['pid'])
            if lifecycle._same_process(previous, current) and not current['zombie']:
                try: os.kill(previous['pid'], signal.SIGKILL)
                except ProcessLookupError: pass
    if child.poll() is None:
        child.kill()
    child.wait(timeout=3)


class LifecycleChecks(unittest.TestCase):
    def test_kernel_identity_contains_birth_uid_parent_and_group(self):
        identity = lifecycle.process_identity(os.getpid())
        self.assertEqual(identity['pid'], os.getpid())
        self.assertEqual(identity['ppid'], os.getppid())
        self.assertEqual(identity['pgid'], os.getpgrp())
        self.assertEqual(identity['uid'], os.getuid())
        self.assertFalse(identity['zombie'])
        self.assertRegex(identity['birth'], r'^(macos|linux):[0-9]+')

    def test_multisession_tree_unwinds_and_kills_stubborn_descendants_without_touching_sibling(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            child = start_tree(folder)
            sibling = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
            snapshot = None
            try:
                wait_file(folder / 'stage.ready', child)
                snapshot = lifecycle.capture_tree(child)
                self.assertEqual(set(snapshot['owned']), {child.pid})
                (folder / 'go').touch()
                wait_file(folder / 'grandchild.pid', child)
                snapshot = lifecycle.capture_tree(child, snapshot=snapshot)
                worker = int((folder / 'worker.pid').read_text())
                grandchild = int((folder / 'grandchild.pid').read_text())
                self.assertEqual(set(snapshot['owned']), {child.pid, worker, grandchild})
                self.assertEqual(len({p['pgid'] for p in snapshot['owned'].values()}), 3)
                result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.2, kill_wait_seconds=2.)
                self.assertEqual(result['status'], 'passed', result)
                self.assertEqual(result['returncode'], 143)
                self.assertTrue((folder / 'unwound').exists())
                self.assertEqual(result['remaining_pids'], [])
                killed = {s['pid'] for s in result['signals'] if s['signal'] == signal.SIGKILL}
                self.assertEqual(killed, {worker, grandchild})
                self.assertIsNone(sibling.poll())
                self.assertNotIn(sibling.pid, snapshot['owned'])
                self.assertLess(result['elapsed_seconds'], 4.)
            finally:
                emergency(child, snapshot)
                sibling.terminate(); sibling.wait(timeout=3)

    def test_snapshot_retains_detached_children_after_normal_parent_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            child = start_tree(folder, normal=True)
            snapshot = None
            try:
                wait_file(folder / 'stage.ready', child)
                snapshot = lifecycle.capture_tree(child)
                (folder / 'go').touch()
                wait_file(folder / 'grandchild.pid', child)
                snapshot = lifecycle.capture_tree(child, snapshot=snapshot)
                (folder / 'normal-exit').touch()
                self.assertEqual(child.wait(timeout=3), 0)
                snapshot = lifecycle.capture_tree(child, snapshot=snapshot)
                result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.1, kill_wait_seconds=2.)
                self.assertEqual(result['status'], 'passed', result)
                self.assertFalse(result['stage_was_running'])
                self.assertEqual(len(result['live_descendants_at_cleanup']), 2)
                self.assertEqual(result['remaining_pids'], [])
                # The driver must classify these leaked children as a lifecycle
                # defect even though cleanup itself succeeded.
            finally:
                emergency(child, snapshot)

    def test_changed_birth_identity_is_never_signalled(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        snapshot = lifecycle.capture_tree(child)
        try:
            changed = {**snapshot['owned'][child.pid], 'birth': 'reused-pid'}
            with patch.object(lifecycle, 'process_identity', return_value=changed), \
                 patch.object(lifecycle, '_parent_table', return_value={}), \
                 patch.object(lifecycle.os, 'kill') as kill:
                result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.01, kill_wait_seconds=.01)
                kill.assert_not_called()
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['identity_changed_skipped'], [child.pid])
            self.assertIsNone(child.poll())
        finally:
            emergency(child, snapshot)

    def test_foreign_snapshot_and_unbounded_intervals_are_rejected(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        snapshot = lifecycle.capture_tree(child)
        try:
            with self.assertRaises(ValueError):
                lifecycle.terminate_tree(child, snapshot={**snapshot, 'owner_pid': -1})
            with self.assertRaises(ValueError):
                lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=1000.)
            self.assertIsNone(child.poll())
        finally:
            emergency(child, snapshot)


if __name__ == '__main__':
    unittest.main()
