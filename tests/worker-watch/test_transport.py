"""Tests for the diagnostic worker-watch transport.

Stdlib only (no JAX, no Node, no network): every worker is a small Python fake
driven through the real JSON wire, so wire bytes, deadlines, partial-byte
evidence and process-group cleanup are exercised end to end.
"""
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

from pokemon_worker_watch import transport  # noqa: E402
from pokemon_worker_watch.transport import (  # noqa: E402
    DEFAULT_COMMAND, FRAME_SCHEMA, HELLO_SCHEMA, PLAYER_SCHEMA, Worker,
    WorkerCleanupError, WorkerCleanupGroupError, close_workers,
)

POSIX = os.name == 'posix'

FAKE_WORKER = r'''
import json, os, subprocess, sys, time

MODE, HELLO, FRAME, PLAYER = sys.argv[1:5]


def read_line():
    # One byte at a time, so "deaf" mode really leaves the pipe full and
    # never pre-buffers a command it is not going to answer.
    if MODE == 'shared_deadline':
        return sys.stdin.buffer.readline() or None
    buf = b''
    while not buf.endswith(b'\n'):
        try:
            chunk = os.read(0, 1)
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def send(obj):
    data = (json.dumps(obj) + '\n').encode()
    while data:
        data = data[os.write(1, data):]


def player():
    return {'schema_version': PLAYER, 'phase': 'move', 'turn': 1, 'request_id': 'r1'}


def frame():
    return {'schema_version': FRAME, 'players': {'p1': player(), 'p2': player()},
            'outcome': None}


extra = {}
if MODE == 'descendant':
    child = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(120)'],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    extra['child_pid'] = child.pid

while True:
    raw = read_line()
    if raw is None:
        break
    command = json.loads(raw)
    op, cid = command['op'], command['id']
    if op == 'hello':
        hello = {'schema_version': 'wrong-hello-vX' if MODE == 'bad_hello' else HELLO,
                 'mode': MODE, 'pid': os.getpid(), 'pgid': os.getpgid(0),
                 'sid': os.getsid(0)}
        hello.update(extra)
        send({'id': cid, 'ok': True, 'data': hello})
        if MODE == 'eof':
            sys.exit(0)
        if MODE == 'shared_deadline':
            time.sleep(0.65)
        if MODE == 'deaf':
            time.sleep(120)
        continue
    if MODE == 'shared_deadline':
        time.sleep(0.65)
    if MODE == 'malformed':
        os.write(1, b'{"id": not json at all\n')
        continue
    if MODE == 'slow_partial':
        os.write(1, b'{"id": ' + str(cid).encode() + b', "ok": tr')
        time.sleep(120)
    if MODE == 'flood':
        blob = b'x' * 5_000_000
        while blob:
            blob = blob[os.write(1, blob):]
        time.sleep(120)
    if MODE == 'wrong_id':
        send({'id': cid + 1000, 'ok': True, 'data': {'echoed': op}})
        continue
    if MODE == 'not_ok':
        send({'id': cid, 'ok': False, 'error': 'worker refused'})
        continue
    if op == 'echo':
        send({'id': cid, 'ok': True, 'data': {'command': command, 'raw': raw.decode()}})
        continue
    if op in ('start', 'step', 'cutoff'):
        data = frame()
        if MODE == 'bad_frame':
            data['schema_version'] = 'wrong-frame-vX'
        elif MODE == 'bad_sides':
            data['players']['p3'] = data['players'].pop('p2')
        elif MODE == 'bad_player':
            data['players']['p2']['schema_version'] = 'wrong-player-vX'
        send({'id': cid, 'ok': True, 'data': data})
        continue
    send({'id': cid, 'ok': True,
          'data': {'op': op,
                   'payload': {k: v for k, v in command.items() if k not in ('id', 'op')}}})
'''


def fake_command(mode):
    return [sys.executable, '-I', '-c', FAKE_WORKER, mode, HELLO_SCHEMA, FRAME_SCHEMA,
            PLAYER_SCHEMA]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - still a live pid we cannot signal
        return True
    return True


class TransportTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.evidence = self.tmp / 'evidence'

    def spawn(self, mode, timeout=20, evidence=None):
        worker = Worker(evidence or self.evidence, timeout=timeout, command=fake_command(mode))
        self.addCleanup(worker._cleanup)  # non-raising, idempotent
        return worker

    def records(self, kind='failure', evidence=None):
        return sorted((evidence or self.evidence).glob(f'{kind}-*.json'))

    def last_record(self, kind='failure', evidence=None):
        paths = self.records(kind, evidence)
        self.assertTrue(paths, f'no {kind} evidence written')
        return json.loads(paths[-1].read_text())

    # ------------------------------------------------------------- handshake

    def test_hello_handshake_and_dedicated_session(self):
        worker = self.spawn('ok')
        self.assertEqual(worker.hello['schema_version'], HELLO_SCHEMA)
        pid = worker.process.pid
        self.assertEqual(worker.pgid, pid)
        self.assertEqual(worker.hello['pgid'], pid)
        if POSIX:
            self.assertEqual(os.getpgid(pid), pid)
            self.assertEqual(worker.hello['sid'], pid)  # dedicated session, not ours
            self.assertNotEqual(worker.pgid, os.getpgid(0))
        metrics = worker.metrics
        self.assertEqual(metrics['phase'], 'done')
        self.assertEqual(metrics['bytes_written'], metrics['bytes_to_write'])
        self.assertGreater(metrics['bytes_read'], 0)
        self.assertEqual(metrics['pid'], pid)
        self.assertIsNone(metrics['returncode'])
        self.assertFalse(metrics['timed_out'])
        self.assertGreaterEqual(metrics['select_wakeups_read'], 1)
        self.assertEqual(metrics['select_wakeups'],
                         metrics['select_wakeups_read'] + metrics['select_wakeups_write'])
        self.assertGreaterEqual(metrics['elapsed_monotonic'], 0.0)
        self.assertGreaterEqual(metrics['elapsed_wall'], 0.0)

    def test_exact_wire_bytes_and_monotonic_ids(self):
        worker = self.spawn('ok')
        data = worker.request('echo', x=1, y=[1, 2], z=None)
        expected = json.dumps({'id': 2, 'op': 'echo', 'x': 1, 'y': [1, 2], 'z': None},
                              allow_nan=False) + '\n'
        self.assertEqual(data['raw'], expected)
        self.assertEqual(data['command'], {'id': 2, 'op': 'echo', 'x': 1, 'y': [1, 2], 'z': None})
        self.assertIs(worker.last, data)
        self.assertEqual(worker.request('echo')['command']['id'], 3)
        self.assertEqual(worker.sequence, 3)

    def test_frame_validation_accepts_exact_schemas(self):
        worker = self.spawn('ok')
        for op in ('start', 'step', 'cutoff'):
            frame = worker.request(op, payload=op)
            self.assertEqual(frame['schema_version'], FRAME_SCHEMA)
            self.assertEqual(set(frame['players']), {'p1', 'p2'})
            self.assertEqual(frame['players']['p1']['schema_version'], PLAYER_SCHEMA)
            self.assertIs(worker.last, frame)
        # Non-frame ops stay unvalidated, exactly as in the original transport.
        self.assertEqual(worker.request('status', k=5)['payload'], {'k': 5})
        self.assertFalse(self.records('failure'))

    def test_frame_validation_rejects_wrong_versions_and_sides(self):
        for mode, message in (('bad_frame', 'Wrong dynamic frame version or sides'),
                              ('bad_sides', 'Wrong dynamic frame version or sides'),
                              ('bad_player', 'Wrong dynamic player version')):
            with self.subTest(mode=mode):
                evidence = self.tmp / f'ev-{mode}'
                worker = self.spawn(mode, evidence=evidence)
                with self.assertRaises(ValueError) as caught:
                    worker.request('start')
                self.assertEqual(str(caught.exception), message)
                self.assertTrue(worker._closed)
                self.assertTrue(self.records('failure', evidence))

    def test_bad_hello_version_fails_construction_and_reaps(self):
        with self.assertRaises(ValueError) as caught:
            Worker(self.evidence, timeout=20, command=fake_command('bad_hello'))
        self.assertIn('Wrong worker version', str(caught.exception))
        record = self.last_record('initialization-failure')
        self.assertEqual(record['stage'], 'worker-initialization')
        self.assertIn('Wrong worker version', record['error'])
        self.assertIsNotNone(record['cleanup']['returncode'])

    def test_missing_executable_records_initialization_failure(self):
        missing = str(self.tmp / 'no-such-worker-binary')
        with self.assertRaises(FileNotFoundError):
            Worker(self.evidence, timeout=5, command=[missing])
        record = self.last_record('initialization-failure')
        self.assertEqual(record['command'], [missing])
        self.assertIn('stderr', record['cleanup']['closed_streams'])
        self.assertTrue(record['cleanup']['ok'])
        self.assertTrue(list(self.evidence.glob('worker-*.stderr')))

    def test_rejects_non_string_command(self):
        with self.assertRaises(ValueError):
            Worker(self.evidence, command=[])
        with self.assertRaises(ValueError):
            Worker(self.evidence, command=['node', 7])

    # ----------------------------------------------------------- wire faults

    def test_eof_before_response(self):
        worker = self.spawn('eof')
        with self.assertRaises(RuntimeError) as caught:
            worker.request('status')
        self.assertEqual(str(caught.exception), 'Worker exited before complete response')
        record = self.last_record()
        self.assertEqual(record['command'], {'id': 2, 'op': 'status'})
        self.assertEqual(record['previous_response']['schema_version'], HELLO_SCHEMA)
        self.assertEqual(record['partial_response'], '')
        self.assertEqual(record['metrics']['phase'], 'read')
        self.assertEqual(record['metrics']['bytes_written'], record['metrics']['bytes_to_write'])
        self.assertIn('process_group', record['resources'])
        self.assertTrue(record['cleanup']['ok'])
        self.assertEqual(record['cleanup']['returncode'], 0)
        self.assertTrue(worker._closed)

    def test_malformed_response_retains_consumed_line(self):
        worker = self.spawn('malformed')
        with self.assertRaises(ValueError) as caught:
            worker.request('status')
        self.assertIsInstance(caught.exception, json.JSONDecodeError)
        record = self.last_record()
        self.assertIn('not json at all', record['consumed_line'])
        self.assertEqual(record['metrics']['phase'], 'decode')
        self.assertTrue(worker._closed)

    def test_wrong_id_response(self):
        worker = self.spawn('wrong_id')
        with self.assertRaises(RuntimeError) as caught:
            worker.request('status')
        self.assertIn('Worker rejected command', str(caught.exception))
        self.assertIn('1002', str(caught.exception))
        record = self.last_record()
        self.assertEqual(record['metrics']['phase'], 'validate')
        self.assertEqual(record['command']['id'], 2)

    def test_not_ok_response(self):
        worker = self.spawn('not_ok')
        with self.assertRaises(RuntimeError) as caught:
            worker.request('status')
        self.assertIn('Worker rejected command', str(caught.exception))

    def test_oversized_response_is_bounded(self):
        worker = self.spawn('flood')
        with self.assertRaises(RuntimeError) as caught:
            worker.request('status')
        self.assertEqual(str(caught.exception), 'Worker response exceeds limit')
        self.assertGreater(worker.metrics['bytes_read'], transport.RESPONSE_LIMIT)
        self.assertEqual(len(self.last_record()['partial_response']), transport.PARTIAL_TAIL)

    def test_unserializable_payload_never_reaches_the_pipe(self):
        worker = self.spawn('ok')
        with self.assertRaises(ValueError):
            worker.request('echo', bad=float('nan'))
        self.assertEqual(worker.metrics['phase'], 'encode')
        self.assertEqual(worker.metrics['bytes_written'], 0)
        self.assertTrue(worker._closed)

    # -------------------------------------------------------------- deadline

    def test_slow_response_times_out_with_partial_bytes(self):
        worker = self.spawn('slow_partial')
        worker.timeout = 1.0
        started = time.monotonic()
        with self.assertRaises(TimeoutError) as caught:
            worker.request('status')
        elapsed = time.monotonic() - started
        self.assertEqual(str(caught.exception), 'Worker response deadline exceeded')
        self.assertGreaterEqual(elapsed, 1.0)
        metrics = worker.metrics
        self.assertEqual(metrics['phase'], 'read')
        self.assertTrue(metrics['timed_out'])
        self.assertGreater(metrics['bytes_read'], 0)
        self.assertGreaterEqual(metrics['elapsed_monotonic'], 1.0)
        self.assertLess(metrics['elapsed_monotonic'], 10.0)
        record = self.last_record()
        self.assertTrue(record['partial_response'].startswith('{"id": 2, "ok": tr'))
        self.assertNotIn('\n', record['partial_response'])
        self.assertIsNone(record['consumed_line'])
        self.assertIn('pgid', record['resources'])
        self.assertTrue(record['cleanup']['killed_group'])
        self.assertIsNotNone(record['cleanup']['returncode'])

    def test_blocked_stdin_times_out_during_write_without_truncation(self):
        worker = self.spawn('deaf')
        worker.timeout = 1.0
        blob = 'x' * 1_000_000
        started = time.monotonic()
        with self.assertRaises(TimeoutError) as caught:
            worker.request('echo', blob=blob)
        elapsed = time.monotonic() - started
        self.assertEqual(str(caught.exception), 'Worker command write deadline exceeded')
        metrics = worker.metrics
        self.assertEqual(metrics['phase'], 'write')
        self.assertTrue(metrics['timed_out'])
        self.assertGreater(metrics['bytes_to_write'], 1_000_000)
        self.assertGreater(metrics['bytes_written'], 0)          # partial write observed
        self.assertLess(metrics['bytes_written'], metrics['bytes_to_write'])
        self.assertEqual(metrics['bytes_read'], 0)               # read never started
        self.assertEqual(metrics['read_calls'], 0)
        self.assertGreaterEqual(metrics['select_wakeups_write'], 1)
        self.assertGreaterEqual(elapsed, 1.0)
        self.assertLess(elapsed, 20.0)                            # single shared deadline
        record = self.last_record()
        self.assertEqual(record['metrics']['bytes_written'], metrics['bytes_written'])
        self.assertEqual(record['partial_response'], '')
        self.assertTrue(record['cleanup']['ok'])

    def test_successful_partial_writes_preserve_complete_command(self):
        worker = self.spawn('ok')
        real_write = os.write
        with patch.object(transport.os, 'write', side_effect=lambda fd, data: real_write(fd, data[:4096])):
            data = worker.request('echo', blob='x' * 100000)
        self.assertEqual(data['command']['blob'], 'x' * 100000)
        self.assertGreater(worker.metrics['write_calls'], 1)
        self.assertEqual(worker.metrics['bytes_written'], worker.metrics['bytes_to_write'])

    def test_write_and_read_share_one_deadline(self):
        worker = self.spawn('shared_deadline')
        worker.timeout = 1.0
        with self.assertRaisesRegex(TimeoutError, 'response deadline'):
            worker.request('echo', blob='x' * 1000000)
        self.assertEqual(worker.metrics['bytes_written'], worker.metrics['bytes_to_write'])
        self.assertEqual(worker.metrics['phase'], 'read')
        self.assertGreaterEqual(worker.metrics['elapsed_monotonic'], 1.0)
        self.assertLess(worker.metrics['elapsed_monotonic'], 1.3)

    def test_wait_interruption_preserves_encoding_error_and_closes_streams(self):
        worker = self.spawn('ok')
        real_wait = worker.process.wait
        with patch.object(worker.process, 'wait', side_effect=KeyboardInterrupt('cleanup interruption')):
            with self.assertRaises(ValueError) as caught:
                worker.request('echo', bad=float('nan'))
        real_wait(timeout=5)
        self.assertFalse(worker.cleanup_result['ok'])
        self.assertTrue(any('KeyboardInterrupt' in e for e in worker.cleanup_result['errors']))
        self.assertEqual(sorted(worker.cleanup_result['closed_streams']), ['stderr', 'stdin', 'stdout'])
        self.assertTrue(any('Worker cleanup' in note for note in caught.exception.__notes__))
        self.assertIn('ValueError', self.last_record()['error'])

    def test_unexpected_cleanup_failure_preserves_context_exception(self):
        worker = self.spawn('ok')
        worker.close()  # Real process is already reaped; inject only at the guard boundary.
        original = ValueError('primary body failure')
        with patch.object(worker, '_cleanup', side_effect=KeyboardInterrupt('injected')):
            with self.assertRaises(ValueError) as caught:
                with worker:
                    raise original
        self.assertIs(caught.exception, original)
        self.assertFalse(worker.cleanup_result['ok'])
        self.assertTrue(self.records('cleanup-interruption'))

    def test_initialization_cleanup_failure_preserves_bad_hello(self):
        actual = Worker._cleanup
        def interrupted_after_cleanup(worker):
            actual(worker)
            raise KeyboardInterrupt('injected after cleanup')
        with patch.object(Worker, '_cleanup', interrupted_after_cleanup):
            with self.assertRaisesRegex(ValueError, 'Wrong worker version'):
                Worker(self.evidence, command=fake_command('bad_hello'))
        record = self.last_record('initialization-failure')
        self.assertFalse(record['cleanup']['ok'])
        self.assertIn('Wrong worker version', record['error'])

    def test_default_deadline_and_command_are_preserved(self):
        signature = inspect.signature(Worker.__init__)
        self.assertEqual(signature.parameters['timeout'].default, 60)
        self.assertIsNone(signature.parameters['command'].default)
        self.assertEqual(tuple(DEFAULT_COMMAND), ('node', 'packages/worker-watch/worker.mjs'))
        self.assertEqual(transport.RESPONSE_LIMIT, 4_000_000)
        self.assertEqual(transport.PARTIAL_TAIL, 10000)
        self.assertEqual((HELLO_SCHEMA, FRAME_SCHEMA, PLAYER_SCHEMA),
                         ('opponent-mix-worker-v1', 'opponent-mix-frame-v1',
                          'full-showdown-player-v2'))
        worker = self.spawn('ok')
        self.assertEqual(worker.timeout, 20)  # constructor seam only
        self.assertEqual(worker.command, fake_command('ok'))

    # --------------------------------------------------------------- cleanup

    def test_close_is_idempotent(self):
        worker = self.spawn('ok')
        process = worker.process
        first = worker.close()
        self.assertTrue(first['ok'])
        self.assertFalse(first['already_closed'])
        self.assertIsNotNone(first['returncode'])
        self.assertEqual(sorted(first['closed_streams']), ['stderr', 'stdin', 'stdout'])
        for second in (worker.close(), worker.close(), worker._cleanup()):
            self.assertTrue(second['already_closed'])
            self.assertTrue(second['ok'])
            self.assertEqual(second['returncode'], first['returncode'])
        self.assertEqual(len(self.records('cleanup')), 1)  # one real cleanup only
        self.assertTrue(process.stdin.closed and process.stdout.closed)
        self.assertTrue(worker.stderr.closed)
        self.assertFalse(alive(process.pid) and process.poll() is None)
        with self.assertRaises(RuntimeError):
            worker.request('status')

    def test_close_reports_failed_cleanup_and_stays_idempotent(self):
        worker = self.spawn('ok')
        real = worker.process
        self.addCleanup(real.wait)
        self.addCleanup(real.kill)

        class StubProcess:
            pid = real.pid
            stdin, stdout = real.stdin, real.stdout

            def poll(self):
                return None

            def kill(self):
                raise OSError(1, 'Operation not permitted')

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired('stub', timeout)

        worker.process, worker.pgid = StubProcess(), None
        with self.assertRaises(WorkerCleanupError) as caught:
            worker.close()
        result = caught.exception.result
        self.assertFalse(result['ok'])
        self.assertTrue(result['wait_timed_out'])
        self.assertTrue(any('kill' in err for err in result['errors']))
        self.assertIn('residual_group', result)
        self.assertEqual(sorted(result['closed_streams']), ['stderr', 'stdin', 'stdout'])
        with self.assertRaises(WorkerCleanupError) as again:
            worker.close()
        self.assertTrue(again.exception.result['already_closed'])

    def test_close_workers_closes_every_worker(self):
        workers = [self.spawn('ok', evidence=self.tmp / f'ev-{i}') for i in range(3)]
        results = close_workers(workers)
        self.assertEqual(len(results), 3)
        for worker, result in zip(workers, results):
            self.assertTrue(result['ok'])
            self.assertIsNotNone(worker.process.poll())

    def test_close_workers_continues_after_failure_then_aggregates(self):
        attempted = []

        class Failing:
            def __init__(self, tag):
                self.tag = tag

            def close(self):
                attempted.append(self.tag)
                raise WorkerCleanupError(f'boom {self.tag}', {'ok': False, 'errors': ['boom']})

        good = [self.spawn('ok', evidence=self.tmp / f'ev-{i}') for i in range(2)]
        workers = [Failing('a'), good[0], Failing('b'), good[1]]
        with self.assertRaises(WorkerCleanupGroupError) as caught:
            close_workers(workers)
        self.assertEqual(attempted, ['a', 'b'])
        self.assertEqual([index for index, _ in caught.exception.failures], [0, 2])
        self.assertEqual(len(caught.exception.failures), 2)
        for worker in good:  # cleanup continued past both failures
            self.assertTrue(worker._closed)
            self.assertIsNotNone(worker.process.poll())

    def test_context_manager_does_not_mask_request_error(self):
        with self.assertRaises(RuntimeError) as caught:
            with self.spawn('wrong_id') as worker:
                worker.request('status')
        self.assertIn('Worker rejected command', str(caught.exception))
        self.assertTrue(worker._closed)

    @unittest.skipUnless(POSIX and hasattr(os, 'killpg'), 'needs POSIX process groups')
    def test_cleanup_kills_spawned_descendant(self):
        worker = self.spawn('descendant')
        child = worker.hello['child_pid']
        self.addCleanup(lambda: alive(child) and os.kill(child, signal.SIGKILL))
        self.assertTrue(alive(child))
        self.assertEqual(os.getpgid(child), worker.pgid)
        result = worker.close()
        self.assertTrue(result['killed_group'])
        deadline = time.monotonic() + 10.0
        while alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(child), 'descendant survived group cleanup')

    # ------------------------------------------------------------- snapshots

    def test_process_snapshot_is_bounded_and_group_scoped(self):
        worker = self.spawn('ok')
        started = time.monotonic()
        snapshot = worker._resource_snapshot()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, len(transport.PS_VARIANTS) * transport.PS_TIMEOUT + 3.0)
        self.assertEqual(snapshot['pgid'], worker.process.pid)
        self.assertEqual(snapshot['worker_pid'], worker.process.pid)
        self.assertIsNone(snapshot['worker_returncode'])
        group = snapshot['process_group']
        self.assertEqual(group['pgid'], worker.pgid)
        self.assertLessEqual(len(group['attempts']), len(transport.PS_VARIANTS))
        for attempt in group['attempts']:
            self.assertEqual(attempt['argv'][0], 'ps')
            self.assertIn(str(worker.pgid), attempt['argv'])
            for row in attempt.get('rows', []):
                self.assertEqual(row.split()[2], str(worker.pgid))  # owned group only
                self.assertLessEqual(len(row), transport.PS_MAX_ROW)
        if not group['available']:
            self.skipTest(f'no usable ps dialect: {group["attempts"]}')
        self.assertLessEqual(group['members'], transport.PS_MAX_LINES)
        if not group['rows']:
            self.skipTest(f'ps dialect listed no group members: {group["attempts"]}')
        own = next(row.split(None, 7) for row in group['rows'] if row.split()[0] == str(worker.process.pid))
        self.assertEqual(group['fields'][5:7], ['rss_kib', 'cpu_percent'])
        self.assertGreater(int(own[5]), 0)
        self.assertGreaterEqual(float(own[6]), 0)

    def test_snapshot_environment_carries_no_credentials(self):
        self.assertEqual(set(transport.PS_ENV), {'PATH', 'LC_ALL'})
        self.assertLessEqual(transport.PS_TIMEOUT, 1.0)
        self.assertNotIn('-e', transport.PS_FIELDS)
        self.assertNotIn('user', transport.PS_FIELDS)

    @unittest.skipIf(not POSIX or os.geteuid() == 0, 'needs a non-root POSIX user')
    def test_evidence_write_failure_preserves_original_error(self):
        worker = self.spawn('wrong_id')
        mode = self.evidence.stat().st_mode
        os.chmod(self.evidence, 0o500)
        try:
            with self.assertRaises(RuntimeError) as caught:
                worker.request('status')
        finally:
            os.chmod(self.evidence, mode)
        self.assertIn('Worker rejected command', str(caught.exception))
        self.assertIsNotNone(worker.evidence_error)
        self.assertFalse(self.records('failure'))
        self.assertTrue(worker.cleanup_result['ok'])
        self.assertIsNotNone(worker.process.poll())

    def test_lifecycle_leaks_no_threads(self):
        before = threading.active_count()
        worker = self.spawn('slow_partial')
        worker.timeout = 0.75
        with self.assertRaises(TimeoutError):
            worker.request('status')
        with self.spawn('ok', evidence=self.tmp / 'ev-threads') as healthy:
            healthy.request('echo')
        self.assertEqual(threading.active_count(), before)


if __name__ == '__main__':
    unittest.main()
