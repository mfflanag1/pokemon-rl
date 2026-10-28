"""F36 runtime checks: synthetic process trees and stub JSON workers only.

No simulator, no native search, no network, no scientific games and no
learner/teacher code are involved. Every synthetic process is tracked and
force-killed by the emergency cleanup registered in setUp.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'python') not in sys.path:
    sys.path.insert(0, str(ROOT / 'python'))

from pokemon_teacher_runtime import lifecycle, transport  # noqa: E402

WORKER_SOURCE = r'''
import json, os, signal, subprocess, sys, time

mode = sys.argv[1] if len(sys.argv) > 1 else 'plain'
if mode in ('stubborn', 'family'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode == 'family':
    source = ("import signal, time\n"
              "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
              "while True: time.sleep(.1)\n")
    # Same process group as this worker: a blanket killpg would have hit it.
    descendant = subprocess.Popen([sys.executable, '-c', source])
    with open(sys.argv[2], 'w') as handle:
        handle.write(str(descendant.pid))
while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    op = request.get('op')
    if op == 'echo':
        payload = {key: value for key, value in request.items() if key not in ('id', 'op')}
        reply = {'id': request['id'], 'ok': True, 'data': payload}
    elif op == 'environment':
        reply = {'id': request['id'], 'ok': True, 'data': {
            'cwd': os.getcwd(), 'pythonpath': os.environ.get('PYTHONPATH'), 'argv': sys.argv}}
    elif op == 'bad-id':
        reply = {'id': request['id'] + 1000, 'ok': True, 'data': None}
    elif op == 'not-ok':
        reply = {'id': request['id'], 'ok': False, 'error': 'refused'}
    elif op == 'huge':
        reply = {'id': request['id'], 'ok': True, 'data': 'x' * 5_000_000}
    elif op == 'sleep':
        time.sleep(30)
        continue
    elif op == 'exit':
        sys.stdout.flush()
        os._exit(7)
    else:
        reply = {'id': request['id'], 'ok': False, 'error': 'unknown op'}
    sys.stdout.write(json.dumps(reply) + '\n')
    sys.stdout.flush()
time.sleep(30)
'''


def wait_for(predicate, *, timeout=10., message='condition'):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for ' + message)
        time.sleep(.01)


def wait_file(path, child=None, timeout=10.):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if child is not None and child.poll() is not None:
            raise AssertionError('Fixture exited before readiness: ' + str(child.returncode))
        if time.monotonic() >= deadline:
            raise AssertionError('Fixture readiness timeout: ' + str(path))
        time.sleep(.01)


class Injector:
    """Deterministic injection of transient/altered identity observations.

    Wraps the real prober so that only the targeted PIDs are affected and all
    other observations stay real; `calls` bounds how many consecutive UNKNOWN
    observations are injected (None means always).
    """

    def __init__(self, real, *, unknown=(), calls=None, replacements=None):
        self.real = real
        self.unknown = set(unknown)
        self.calls = calls
        self.replacements = dict(replacements or {})
        self.counts = {}

    def __call__(self, pid):
        if pid in self.replacements:
            return self.replacements[pid]
        if pid in self.unknown:
            seen = self.counts.get(pid, 0)
            if self.calls is None or seen < self.calls:
                self.counts[pid] = seen + 1
                return lifecycle._unknown(pid, {'probe': 'injected-transient', 'errno': 3})
        return self.real(pid)


class SyntheticBase(unittest.TestCase):
    def setUp(self):
        self.children = []
        self.pids = set()
        self.identities = {}
        self.pid_files = []
        self.workers = []
        self.addCleanup(self.emergency_cleanup)

    def spawn(self, arguments, **kwargs):
        child = subprocess.Popen([sys.executable, *arguments], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True, **kwargs)
        self.children.append(child)
        return child

    def track_pid(self, pid):
        self.pids.add(int(pid))
        observed = lifecycle.process_identity(int(pid))
        if lifecycle.is_live(observed) and observed.get('birth_verified'):
            self.identities[int(pid)] = observed
        return int(pid)

    def track_pid_file(self, path):
        self.pid_files.append(Path(path))

    def read_pid(self, path):
        return self.track_pid(int(Path(path).read_text().strip()))

    def emergency_cleanup(self):
        pids = set(self.pids)
        for child in self.children:
            try:
                snapshot = lifecycle.capture_tree(child)
                self.identities.update(snapshot['owned'])
                pids.update(snapshot['owned'])
            except Exception:
                pass
        for path in self.pid_files:
            try:
                pids.add(int(path.read_text().strip()))
            except (OSError, ValueError):
                pass
        for worker in self.workers:
            if worker.process is not None:
                pids.add(worker.process.pid)
        for pid in pids:
            observation = lifecycle.process_identity(pid)
            if lifecycle._same_process(self.identities.get(pid), observation):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        for worker in self.workers:
            try:
                worker.close(report=False)
            except Exception:
                pass
            if worker.process is not None:
                try:
                    worker.process.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        for child in self.children:
            try:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
            except OSError:
                pass


class IdentityObservation(SyntheticBase):
    def test_live_observation_carries_birth_uid_parent_and_group(self):
        observation = lifecycle.process_identity(os.getpid())
        self.assertTrue(lifecycle.is_live(observation))
        self.assertEqual(observation['pid'], os.getpid())
        self.assertEqual(observation['ppid'], os.getppid())
        self.assertEqual(observation['pgid'], os.getpgrp())
        self.assertEqual(observation['uid'], os.getuid())
        self.assertFalse(observation['zombie'])
        self.assertRegex(observation['birth'], r'^(macos|linux):[0-9]+')

    def test_reaped_child_is_confirmed_gone_not_unknown(self):
        child = self.spawn(['-c', 'pass'])
        before = lifecycle.process_identity(child.pid)
        self.assertEqual(child.wait(timeout=10), 0)
        after = lifecycle.process_identity(child.pid)
        self.assertFalse(lifecycle.is_unknown(after), after)
        # Either confirmed absent, or (vanishingly unlikely in a test) the PID
        # already hosts a different birth identity; never the same process.
        self.assertTrue(lifecycle.is_gone(after) or after['birth'] != before['birth'], after)
        self.assertFalse(lifecycle._same_process(before, after))

    def test_identity_probe_disagreement_is_unknown_with_bounded_diagnostics(self):
        """Exact F35 raise site: libproc declines while the PID is present."""
        stub = types.SimpleNamespace(proc_pidinfo=lambda *args: 0)
        row = types.SimpleNamespace(returncode=0, stdout='R+   1   2 ' + str(os.getuid()) + '\n', stderr='')
        with patch.object(lifecycle, '_libproc', lambda: stub), \
             patch.object(lifecycle, '_ps_probe', lambda pid: (row, {'probe': 'ps', 'returncode': 0,
                 'stdout': row.stdout, 'stderr': ''})):
            observation = lifecycle._macos_identity(os.getpid())
        self.assertTrue(lifecycle.is_unknown(observation), observation)
        self.assertIsNone(observation['birth'])
        self.assertFalse(observation['zombie'])
        self.assertEqual(observation['diagnostic']['libproc']['result'], 0)
        self.assertEqual(observation['diagnostic']['ps']['returncode'], 0)
        self.assertEqual(observation['diagnostic']['kill0']['probe'], 'kill0')
        self.assertLessEqual(len(json.dumps(observation['diagnostic'])), 4000)

    def test_absent_and_zombie_probe_results_are_classified(self):
        stub = types.SimpleNamespace(proc_pidinfo=lambda *args: 0)
        absent = types.SimpleNamespace(returncode=1, stdout='', stderr='')
        zombie = types.SimpleNamespace(returncode=0, stdout='Z+   1   2 7\n', stderr='')
        with patch.object(lifecycle, '_libproc', lambda: stub), \
             patch.object(lifecycle, '_ps_probe', lambda pid: (absent, {'probe': 'ps'})):
            self.assertTrue(lifecycle.is_gone(lifecycle._macos_identity(os.getpid())))
        with patch.object(lifecycle, '_libproc', lambda: stub), \
             patch.object(lifecycle, '_ps_probe', lambda pid: (zombie, {'probe': 'ps'})):
            observation = lifecycle._macos_identity(os.getpid())
        self.assertTrue(observation['zombie'])
        self.assertFalse(observation['birth_verified'])

    def test_probe_exception_becomes_unknown_not_a_raise(self):
        def explode(pid):
            raise OSError(5, 'injected inspection failure')

        with patch.object(lifecycle, 'process_identity', explode):
            observation = lifecycle._observe(os.getpid())
        self.assertTrue(lifecycle.is_unknown(observation))
        self.assertIn('injected inspection failure', observation['diagnostic']['error'])


class SnapshotRefresh(SyntheticBase):
    def test_transient_unknown_preserves_ownership_and_resolves_on_later_poll(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        snapshot = lifecycle.capture_tree(child)
        original = dict(snapshot['owned'][child.pid])
        injector = Injector(lifecycle.process_identity, unknown={child.pid}, calls=2)
        with patch.object(lifecycle, 'process_identity', injector):
            lifecycle.capture_tree(child, snapshot=snapshot)
            # UNKNOWN is not GONE: ownership is preserved and nothing retired.
            self.assertEqual(snapshot['owned'][child.pid]['birth'], original['birth'])
            self.assertIn(child.pid, snapshot['unknown'])
            self.assertEqual(snapshot['retired'], [])
            self.assertEqual(snapshot['last_live'], [])
            lifecycle.capture_tree(child, snapshot=snapshot)
            self.assertEqual(snapshot['unknown'][child.pid]['count'], 2)
        # Healthy work was never aborted by the transient failure.
        self.assertIsNone(child.poll())
        lifecycle.capture_tree(child, snapshot=snapshot)
        self.assertNotIn(child.pid, snapshot['unknown'])
        self.assertEqual(snapshot['last_live'], [child.pid])
        self.assertEqual(snapshot['owned'][child.pid]['birth'], original['birth'])
        self.assertEqual(snapshot['retired'], [])
        result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.3, kill_wait_seconds=2.)
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['unresolved_unknown_pids'], [])

    def test_unknown_telemetry_is_bounded(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        snapshot = lifecycle.capture_tree(child)
        injector = Injector(lifecycle.process_identity, unknown={child.pid})
        with patch.object(lifecycle, 'process_identity', injector), \
             patch.object(lifecycle, '_parent_table', lambda: {}):
            for _poll in range(40):
                lifecycle.capture_tree(child, snapshot=snapshot)
        # One entry per PID with a counter, and a capped event log.
        self.assertEqual(list(snapshot['unknown']), [child.pid])
        self.assertEqual(snapshot['unknown'][child.pid]['count'], 40)
        self.assertLessEqual(len(snapshot['events']), lifecycle._MAX_EVENTS)
        self.assertEqual(snapshot['retired'], [])

    def test_confirmed_exit_is_retired_and_cleanup_passes(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        snapshot = lifecycle.capture_tree(child)
        child.terminate()
        self.assertIsNotNone(child.wait(timeout=10))
        lifecycle.capture_tree(child, snapshot=snapshot)
        self.assertEqual(snapshot['owned'], {})
        self.assertEqual(snapshot['root_ownership'], 'exited')
        self.assertEqual([record['reason'] for record in snapshot['retired']], ['gone'])
        result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.1, kill_wait_seconds=.1)
        self.assertEqual(result['status'], 'passed', result)
        self.assertFalse(result['stage_was_running'])
        self.assertEqual(result['signals'], [])
        self.assertEqual(result['remaining_pids'], [])

    def test_process_table_failure_never_declares_a_process_gone(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        snapshot = lifecycle.capture_tree(child)
        birth = snapshot['owned'][child.pid]['birth']

        def broken():
            raise subprocess.TimeoutExpired('/bin/ps', 2.)

        with patch.object(lifecycle, '_parent_table', broken):
            lifecycle.capture_tree(child, snapshot=snapshot)  # must not raise
        self.assertIsNotNone(snapshot['last_report']['table_error'])
        self.assertIn('TimeoutExpired', snapshot['last_report']['table_error'])
        self.assertEqual(snapshot['owned'][child.pid]['birth'], birth)
        self.assertEqual(snapshot['retired'], [])
        self.assertIsNone(child.poll())

    def test_fast_stage_exit_is_recorded_rather_than_raised(self):
        child = self.spawn(['-c', 'pass'])
        self.assertEqual(child.wait(timeout=10), 0)
        snapshot = lifecycle.capture_tree(child)
        self.assertEqual(snapshot['root_ownership'], 'exited')
        self.assertEqual(snapshot['owned'], {})
        self.assertTrue(lifecycle.is_gone(snapshot['root_observation']))
        result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.1, kill_wait_seconds=.1)
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['signals'], [])
        self.assertEqual(result['handle_signals'], [])
        self.assertEqual(result['returncode'], 0)

    def test_unverifiable_initial_capture_claims_no_ownership_and_retries(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        injector = Injector(lifecycle.process_identity, unknown={child.pid})
        with patch.object(lifecycle, 'process_identity', injector), \
             patch.object(lifecycle, '_parent_table', lambda: {}):
            snapshot = lifecycle.capture_tree(child)  # must not raise
        self.assertEqual(snapshot['root_ownership'], 'unverified')
        self.assertEqual(snapshot['owned'], {})
        self.assertIn(child.pid, snapshot['unknown'])
        self.assertIsNone(child.poll())
        lifecycle.capture_tree(child, snapshot=snapshot)  # later poll retries
        self.assertEqual(snapshot['root_ownership'], 'verified')
        self.assertIn(child.pid, snapshot['owned'])
        self.assertNotIn(child.pid, snapshot['unknown'])
        result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.3, kill_wait_seconds=2.)
        self.assertEqual(result['status'], 'passed', result)

    def test_unverifiable_root_is_reached_through_its_owned_handle(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        injector = Injector(lifecycle.process_identity, unknown={child.pid})
        with patch.object(lifecycle, 'process_identity', injector), \
             patch.object(lifecycle, '_parent_table', lambda: {}):
            snapshot = lifecycle.capture_tree(child)
            with patch.object(lifecycle, '_kill') as kill:
                result = lifecycle.terminate_tree(child, snapshot=snapshot,
                    grace_seconds=1., kill_wait_seconds=1.)
                kill.assert_not_called()  # no PID-number signalling of an unknown
        self.assertTrue(result['handle_signals'])
        self.assertEqual(result['handle_signals'][0]['signal'], signal.SIGTERM)
        self.assertTrue(all(entry['basis'] == 'owned-unreaped-popen-handle'
                            for entry in result['handle_signals']))
        self.assertIsNotNone(child.poll())
        self.assertEqual(result['remaining_pids'], [])
        self.assertEqual(result['unresolved_unknown_pids'], [])
        self.assertEqual(result['status'], 'passed', result)

    def test_validation_rejects_foreign_snapshot_and_unbounded_intervals(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        snapshot = lifecycle.capture_tree(child)
        with self.assertRaises(ValueError):
            lifecycle.terminate_tree(child, snapshot={**snapshot, 'owner_pid': -1})
        with self.assertRaises(ValueError):
            lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=1000.)
        with self.assertRaises(ValueError):
            lifecycle.capture_tree(child.pid)
        with self.assertRaises(ValueError):
            lifecycle.process_identity(0)
        self.assertIsNone(child.poll())


class CleanupBehaviour(SyntheticBase):
    def test_shared_cleanup_deadline_bounds_many_slow_identity_probes(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        snapshot = lifecycle.capture_tree(child)
        original = dict(snapshot['owned'][child.pid])
        for pid in (9000001, 9000002, 9000003):
            snapshot['owned'][pid] = {**original, 'pid': pid}
        stub = types.SimpleNamespace(proc_pidinfo=lambda *args: 0)
        def slow_probe(*args, **kwargs):
            time.sleep(kwargs['timeout'])
            raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
        started = time.monotonic()
        with patch.object(lifecycle, '_libproc', lambda: stub), \
             patch.object(lifecycle, '_presence', lambda pid: ('present', {})), \
             patch.object(lifecycle, '_parent_table', lambda: {}), \
             patch.object(lifecycle.subprocess, 'run', slow_probe), \
             patch.object(lifecycle, '_kill') as kill:
            result = lifecycle.terminate_tree(child, snapshot=snapshot,
                grace_seconds=.15, kill_wait_seconds=.15, poll_seconds=.01)
            kill.assert_not_called()
        self.assertLess(time.monotonic() - started, .65, result)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue({9000001, 9000002, 9000003}.issubset(result['unresolved_unknown_pids']))
        self.assertTrue(set(result['unresolved_unknown_pids']).issubset(result['remaining_pids']))

    def test_unknown_discovered_candidate_survives_parent_exit(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        snapshot = lifecycle.capture_tree(child)
        candidate = 9000004
        injector = Injector(lifecycle.process_identity, unknown={candidate})
        with patch.object(lifecycle, 'process_identity', injector), \
             patch.object(lifecycle, '_parent_table', lambda: {candidate: (child.pid, candidate)}):
            lifecycle.capture_tree(child, snapshot)
        self.assertIn(candidate, snapshot['unknown'])
        self.assertNotIn(candidate, snapshot['owned'])
        child.terminate()
        child.wait(timeout=3)
        with patch.object(lifecycle, 'process_identity', injector), \
             patch.object(lifecycle, '_parent_table', lambda: {}), \
             patch.object(lifecycle, '_kill') as kill:
            result = lifecycle.terminate_tree(child, snapshot=snapshot,
                grace_seconds=.05, kill_wait_seconds=.05, poll_seconds=.01)
            kill.assert_not_called()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['unresolved_unknown_pids'], [candidate])

    def start_tree(self, folder, *, normal=False):
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
        (folder / '_f36_runtime_fixture.py').write_text(stage)
        self.track_pid_file(folder / 'worker.pid')
        self.track_pid_file(folder / 'grandchild.pid')
        return self.spawn(['-m', 'pokemon_teacher_runtime.lifecycle', '--', '_f36_runtime_fixture',
            str(folder)], env={**os.environ,
            'PYTHONPATH': str(folder) + os.pathsep + str(ROOT / 'python')})

    def captured_tree(self, folder):
        child = self.start_tree(folder)
        wait_file(folder / 'stage.ready', child)
        snapshot = lifecycle.capture_tree(child)
        self.assertEqual(set(snapshot['owned']), {child.pid})
        (folder / 'go').touch()
        wait_file(folder / 'grandchild.pid', child)
        snapshot = lifecycle.capture_tree(child, snapshot=snapshot)
        worker = self.read_pid(folder / 'worker.pid')
        grandchild = self.read_pid(folder / 'grandchild.pid')
        self.assertEqual(set(snapshot['owned']), {child.pid, worker, grandchild})
        return child, snapshot, worker, grandchild

    def test_multisession_tree_unwinds_and_kills_stubborn_descendants_sparing_sibling(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            sibling = self.spawn(['-c', 'import time; time.sleep(30)'])
            child, snapshot, worker, grandchild = self.captured_tree(folder)
            self.assertEqual(len({entry['pgid'] for entry in snapshot['owned'].values()}), 3)
            self.assertNotIn(sibling.pid, snapshot['owned'])
            result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.6,
                kill_wait_seconds=2.)
            self.assertEqual(result['status'], 'passed', result)
            self.assertEqual(result['returncode'], 143)
            self.assertTrue((folder / 'unwound').exists())
            self.assertEqual(result['remaining_pids'], [])
            self.assertEqual(result['unresolved_unknown_pids'], [])
            killed = {entry['pid'] for entry in result['signals'] if entry['signal'] == signal.SIGKILL}
            self.assertEqual(killed, {worker, grandchild})
            self.assertIsNone(sibling.poll())
            self.assertLess(result['elapsed_seconds'], 6.)

    def test_persistent_unknown_descendant_fails_closed_and_is_never_signalled(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            child, snapshot, worker, grandchild = self.captured_tree(folder)
            worker_identity = dict(snapshot['owned'][worker])
            injector = Injector(lifecycle.process_identity, unknown={worker})
            with patch.object(lifecycle, 'process_identity', injector):
                result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.6,
                    kill_wait_seconds=1.5)
            self.assertEqual(result['status'], 'failed', result)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['unresolved_unknown_pids'], [worker])
            self.assertIn(worker, result['remaining_pids'])
            self.assertNotIn(worker, {entry['pid'] for entry in result['signals']})
            self.assertEqual(result['handle_signals'], [])
            # The unknown PID was neither signalled nor treated as gone, while
            # cleanup still completed for every verifiable PID.
            current = lifecycle.process_identity(worker)
            self.assertTrue(lifecycle._same_process(worker_identity, current))
            self.assertIn(worker, snapshot['owned'])
            self.assertNotIn(worker, [record['pid'] for record in snapshot['retired']])
            self.assertIn(grandchild, {entry['pid'] for entry in result['signals']
                                       if entry['signal'] == signal.SIGKILL})
            wait_for(lambda: not lifecycle.is_live(lifecycle.process_identity(grandchild))
                     or lifecycle.process_identity(grandchild)['zombie'],
                     timeout=5., message='stubborn grandchild to die')
            self.assertLess(result['elapsed_seconds'], 8.)

    def test_reused_identity_is_never_signalled_or_adopted(self):
        child = self.spawn(['-c', 'import time; time.sleep(30)'])
        snapshot = lifecycle.capture_tree(child)
        original = dict(snapshot['owned'][child.pid])
        reused = {**original, 'birth': 'reused-pid'}
        injector = Injector(lifecycle.process_identity, replacements={child.pid: reused})
        with patch.object(lifecycle, 'process_identity', injector), \
             patch.object(lifecycle, '_parent_table', lambda: {}), \
             patch.object(lifecycle, '_kill') as kill, \
             patch.object(lifecycle, '_send_handle') as handle:
            result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.05,
                kill_wait_seconds=.05)
            kill.assert_not_called()
            handle.assert_not_called()
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(result['reused_pids'], [child.pid])
        self.assertEqual(result['identity_changed_skipped'], [child.pid])
        self.assertEqual(snapshot['root_ownership'], 'contradicted')
        self.assertNotIn(child.pid, snapshot['owned'])
        retired = [record for record in snapshot['retired'] if record['pid'] == child.pid]
        self.assertEqual([record['reason'] for record in retired], ['pid-reused'])
        self.assertEqual(retired[0]['birth'], original['birth'])  # not adopted as the new identity
        self.assertIsNone(child.poll())

    def test_snapshot_retains_detached_children_after_normal_parent_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            child = self.start_tree(folder, normal=True)
            wait_file(folder / 'stage.ready', child)
            snapshot = lifecycle.capture_tree(child)
            (folder / 'go').touch()
            wait_file(folder / 'grandchild.pid', child)
            snapshot = lifecycle.capture_tree(child, snapshot=snapshot)
            self.read_pid(folder / 'worker.pid')
            self.read_pid(folder / 'grandchild.pid')
            (folder / 'normal-exit').touch()
            self.assertEqual(child.wait(timeout=10), 0)
            snapshot = lifecycle.capture_tree(child, snapshot=snapshot)
            result = lifecycle.terminate_tree(child, snapshot=snapshot, grace_seconds=.3,
                kill_wait_seconds=2.)
            self.assertEqual(result['status'], 'passed', result)
            self.assertFalse(result['stage_was_running'])
            self.assertEqual(len(result['live_descendants_at_cleanup']), 2)
            self.assertEqual(result['remaining_pids'], [])

    def test_wrapper_returns_cooperative_code_even_if_unwinding_also_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / '_f36_masking_fixture.py').write_text(
                "import pathlib, sys, time\n"
                "root = pathlib.Path(sys.argv[1])\n"
                "try:\n"
                "    root.joinpath('stage.ready').write_text('ready')\n"
                "    while True: time.sleep(.05)\n"
                "finally:\n"
                "    raise PermissionError(1, 'Operation not permitted')\n")
            child = self.spawn(['-m', 'pokemon_teacher_runtime.lifecycle', '--',
                '_f36_masking_fixture', str(folder)], env={**os.environ,
                'PYTHONPATH': str(folder) + os.pathsep + str(ROOT / 'python')})
            wait_file(folder / 'stage.ready', child)
            child.terminate()
            # The F35 failure mode: a cleanup error raised while unwinding
            # StageTerminated replaced it and the stage exited 1.
            self.assertEqual(child.wait(timeout=10), 143)


class TransportBehaviour(SyntheticBase):
    def setUp(self):
        super().setUp()
        self.folder = Path(tempfile.mkdtemp(prefix='f36-transport-'))
        self.addCleanup(self.remove_folder)
        self.script = self.folder / 'worker.py'
        self.script.write_text(WORKER_SOURCE)
        self.counter = 0

    def remove_folder(self):
        for path in sorted(self.folder.rglob('*'), reverse=True):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            self.folder.rmdir()
        except OSError:
            pass

    def worker(self, mode='plain', *extra, timeout=10., **kwargs):
        self.counter += 1
        stderr_path = self.folder / f'worker-{self.counter}.stderr'
        handle = transport.JsonWorker([sys.executable, '-u', str(self.script), mode, *extra],
            stderr_path, timeout, **kwargs)
        self.workers.append(handle)
        return handle

    def test_json_protocol_ids_payloads_argv_and_environment(self):
        worker = self.worker()
        self.assertEqual(worker.request('echo', value=1), {'value': 1})
        self.assertEqual(worker.sequence, 1)
        self.assertEqual(worker.request('echo', value=[2, 'three']), {'value': [2, 'three']})
        self.assertEqual(worker.sequence, 2)
        environment = worker.request('environment')
        self.assertEqual(environment['cwd'], str(transport.ROOT))
        self.assertEqual(environment['pythonpath'], str(transport.ROOT / 'python'))
        self.assertEqual(environment['argv'][0], str(self.script))
        worker.close()
        self.assertTrue(worker.closed)
        with self.assertRaises(RuntimeError):
            worker.request('echo')

    def test_protocol_violations_timeouts_and_bounds_are_enforced(self):
        with self.assertRaises(RuntimeError):
            self.worker().request('bad-id')
        with self.assertRaises(RuntimeError):
            self.worker().request('not-ok')
        with self.assertRaises(ValueError) as oversized:
            self.worker().request('huge')
        self.assertIn('response size', str(oversized.exception))
        with self.assertRaises(TimeoutError):
            self.worker(timeout=.3).request('sleep')
        with self.assertRaises(RuntimeError) as exited:
            self.worker().request('exit')
        self.assertIn('External worker exited', str(exited.exception))

    def test_close_is_idempotent_bounded_and_limited_to_the_direct_child(self):
        pid_path = self.folder / 'descendant.pid'
        self.track_pid_file(pid_path)
        worker = self.worker('family', str(pid_path))
        self.assertEqual(worker.request('echo', ready=True), {'ready': True})
        wait_file(pid_path)
        descendant = self.read_pid(pid_path)
        started = time.monotonic()
        self.assertEqual(worker.close(), [])
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, worker.term_wait + worker.kill_wait + 2.)
        self.assertIsNotNone(worker.process.poll())
        self.assertEqual(worker.process.returncode, -signal.SIGKILL)
        for handle in (worker.process.stdin, worker.process.stdout, worker.stderr):
            self.assertTrue(handle.closed)
        self.assertEqual(worker.close(), [])  # idempotent
        # Scope: the worker's own descendant shares its process group and is
        # deliberately left to the supervisor's birth-checked cleanup.
        observation = lifecycle.process_identity(descendant)
        self.assertTrue(lifecycle.is_live(observation) and not observation['zombie'], observation)

    def test_cleanup_failure_cannot_replace_a_request_exception(self):
        worker = self.worker('stubborn', timeout=.3, term_wait=.2, kill_wait=.2)
        self.track_pid(worker.process.pid)
        denied = PermissionError(1, 'Operation not permitted')
        with patch.object(worker.process, 'terminate', side_effect=denied), \
             patch.object(worker.process, 'kill', side_effect=denied):
            started = time.monotonic()
            with self.assertRaises(TimeoutError) as caught:
                worker.request('sleep')
            self.assertLess(time.monotonic() - started, 5.)
        self.assertIn('read timed out', str(caught.exception))
        self.assertTrue(any('PermissionError' in error
                            for error in getattr(caught.exception, 'cleanup_errors', [])))
        self.assertTrue(worker.closed)
        for handle in (worker.process.stdin, worker.process.stdout, worker.stderr):
            self.assertTrue(handle.closed)
        self.assertIsNone(worker.process.poll())  # emergency cleanup reaps it

    def test_cleanup_failure_cannot_replace_a_context_exception(self):
        worker = self.worker('stubborn', term_wait=.2, kill_wait=.2)
        self.track_pid(worker.process.pid)
        denied = PermissionError(1, 'Operation not permitted')
        with patch.object(worker.process, 'terminate', side_effect=denied), \
             patch.object(worker.process, 'kill', side_effect=denied):
            with self.assertRaises(RuntimeError) as caught:
                with worker:
                    worker.request('echo', value=1)
                    raise RuntimeError('original stage failure')
        self.assertEqual(str(caught.exception), 'original stage failure')
        self.assertNotIsInstance(caught.exception, transport.WorkerCleanupError)
        self.assertTrue(any('PermissionError' in error
                            for error in getattr(caught.exception, 'cleanup_errors', [])))

    def test_cleanup_failure_on_a_successful_context_is_reported(self):
        worker = self.worker('stubborn', term_wait=.2, kill_wait=.2)
        self.track_pid(worker.process.pid)
        denied = PermissionError(1, 'Operation not permitted')
        with patch.object(worker.process, 'terminate', side_effect=denied), \
             patch.object(worker.process, 'kill', side_effect=denied):
            with self.assertRaises(transport.WorkerCleanupError) as caught:
                with worker:
                    worker.request('echo', value=1)
        self.assertTrue(any('PermissionError' in error for error in caught.exception.errors))
        for handle in (worker.process.stdin, worker.process.stdout, worker.stderr):
            self.assertTrue(handle.closed)


if __name__ == '__main__':
    unittest.main()
