"""Instrumented, bounded wire transport with retained failure payloads.

Same JSON wire and the same response validation as
``pokemon_opponent_mix.transport`` (newline-delimited objects, monotonic
``id``, ``ok``/``data`` envelope, 4 MB response ceiling, per-op frame/player
schema checks, 60 second default deadline, no retry). The differences are
purely diagnostic and additive:

* the default command is ``node packages/worker-watch/worker.mjs``; the
  command is a *constructor-only* seam used by tests, never actor-controlled
  and never derived from worker/episode data;
* the command write and the response read are both non-blocking and share the
  single request deadline, so a worker that stops reading stdin cannot hang a
  request forever; partial writes/reads are recorded and then raised, never
  truncated or retried;
* every worker gets a dedicated session/process group, cleanup is idempotent,
  closes descriptors and attempts to reap the owned worker *and* its
  descendants, and the request failure record keeps the command, the previous
  response, the partial response, a pre-cleanup resource/process-group
  snapshot and the cleanup result;
* evidence-writing or cleanup problems never replace the original request
  exception.

No threads and no background readers are used: ``select`` alone bounds both
phases, so there is nothing to leak.
"""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

try:  # stdlib on POSIX; absence only degrades the snapshot
    import resource as _resource
except ImportError:  # pragma: no cover - non-POSIX
    _resource = None

ROOT = Path(__file__).resolve().parents[2]

#: Coordinator-owned Node worker. Overridable only through ``Worker(command=...)``.
DEFAULT_COMMAND = ('node', 'packages/worker-watch/worker.mjs')

# Wire/validation constants, identical to the supplied transport. If the
# coordinator's worker.mjs ever declares a different handshake string, this is
# the single line that needs to change.
HELLO_SCHEMA = 'opponent-mix-worker-v1'
FRAME_SCHEMA = 'opponent-mix-frame-v1'
PLAYER_SCHEMA = 'full-showdown-player-v2'
FRAME_OPS = ('start', 'step', 'cutoff')
SIDES = ('p1', 'p2')

READ_CHUNK = 65536
RESPONSE_LIMIT = 4_000_000
PARTIAL_TAIL = 10000
CLEANUP_WAIT = 5.0

# Bounded, read-only process inspection: no shell, no inherited environment,
# no credentials, one second ceiling, output restricted to the owned group.
PS_FIELDS = 'pid=,ppid=,pgid=,stat=,etime=,rss=,pcpu=,comm='
PS_ENV = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LC_ALL': 'C'}
PS_TIMEOUT = 1.0
PS_MAX_LINES = 64
PS_MAX_ROW = 200
# Selection dialects, tried in order until one exits 0 with group members. The
# worker has no controlling terminal (own session), hence BSD's -x first.
PS_VARIANTS = (('-x', '-g'), ('-g',), ('--pgid',))


class WorkerCleanupError(RuntimeError):
    """Raised by ``Worker.close()`` when the bounded cleanup did not succeed."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


class WorkerCleanupGroupError(RuntimeError):
    """Aggregate raised by ``close_workers`` after every worker was attempted."""

    def __init__(self, failures):
        self.failures = list(failures)
        detail = '; '.join(f'{index}: {exc!r}' for index, exc in self.failures)
        super().__init__(f'{len(self.failures)} worker cleanup failure(s): {detail}')


def close_workers(workers):
    """Close every worker, then raise an aggregate if any cleanup failed.

    One failing worker never prevents the remaining workers from being closed.
    """
    results, failures = [], []
    for index, worker in enumerate(list(workers)):
        try:
            results.append(worker.close())
        except BaseException as exc:  # noqa: BLE001 - every worker must be attempted
            failures.append((index, exc))
            results.append(None)
    if failures:
        raise WorkerCleanupGroupError(failures)
    return results


def _safe(call):
    try:
        return call()
    except BaseException as exc:  # noqa: BLE001 - diagnostics must not raise
        return {'error': repr(exc)}


class Worker:
    def __init__(self, evidence, timeout=60, command=None):
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.sequence, self.timeout, self.buffer, self.last = 0, timeout, b'', None
        self.command = list(DEFAULT_COMMAND if command is None else command)
        if not self.command or not all(isinstance(part, str) for part in self.command):
            raise ValueError('Worker command must be a non-empty list of strings')
        self.metrics = None          # metrics of the most recent request
        self.history = []            # bounded tail of per-request metrics
        self.cleanup_result = None
        self.evidence_error = None
        self.pgid = None
        self._closed = False
        self.stderr = open(self.evidence / f'worker-{time.time_ns()}.stderr', 'xb')
        self.process = None
        try:
            # start_new_session: dedicated session *and* process group, so the
            # whole worker subtree can be signalled and inspected by pgid.
            self.process = subprocess.Popen(self.command, cwd=ROOT, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0, start_new_session=True)
            try:
                self.pgid = os.getpgid(self.process.pid)
            except OSError:
                self.pgid = self.process.pid
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            self.hello = self.request('hello')
            if self.hello['schema_version'] != HELLO_SCHEMA:
                raise ValueError('Wrong worker version')
        except BaseException as exc:
            snapshot = _safe(self._resource_snapshot)
            cleanup = self._cleanup_preserving(exc)
            self._write_evidence('initialization-failure', {
                'stage': 'worker-initialization', 'error': repr(exc), 'command': self.command,
                'metrics': self.metrics, 'resources': snapshot, 'cleanup': cleanup})
            raise

    # ------------------------------------------------------------------ wire

    def request(self, op, **payload):
        if self._closed:
            raise RuntimeError('Worker is closed')
        if self.process is None:
            raise RuntimeError('Worker process was never started')
        self.sequence += 1
        command = {'id': self.sequence, 'op': op, **payload}
        metrics = {
            'id': self.sequence, 'op': op, 'phase': 'encode',
            'pid': self.process.pid, 'pgid': self.pgid, 'returncode': None,
            'timeout_seconds': self.timeout, 'timed_out': False,
            'bytes_to_write': 0, 'bytes_written': 0, 'bytes_read': 0,
            'write_calls': 0, 'read_calls': 0,
            'select_wakeups_write': 0, 'select_wakeups_read': 0, 'select_wakeups': 0,
            'buffered_bytes_before': len(self.buffer), 'buffered_bytes_after': len(self.buffer),
            'started_monotonic': time.monotonic(), 'started_wall': time.time(),
            'elapsed_monotonic': 0.0, 'elapsed_wall': 0.0,
        }
        self.metrics = metrics
        self.history = (self.history + [metrics])[-32:]
        deadline = metrics['started_monotonic'] + self.timeout
        raw = None
        try:
            line = (json.dumps(command, allow_nan=False) + '\n').encode()
            metrics['bytes_to_write'] = len(line)
            self._write_all(line, deadline, metrics)
            raw = self._read_line(deadline, metrics)
            metrics['phase'] = 'decode'
            response = json.loads(raw)
            metrics['phase'] = 'validate'
            if response.get('id') != self.sequence or response.get('ok') is not True:
                raise RuntimeError(f'Worker rejected command: {response}')
            data = response['data']
            if op in FRAME_OPS:
                if data.get('schema_version') != FRAME_SCHEMA or set(data.get('players', {})) != set(SIDES):
                    raise ValueError('Wrong dynamic frame version or sides')
                if any(data['players'][side].get('schema_version') != PLAYER_SCHEMA for side in SIDES):
                    raise ValueError('Wrong dynamic player version')
            self.last = data
            metrics['phase'] = 'done'
            self._finalize(metrics)
            return self.last
        except BaseException as exc:
            self._finalize(metrics)
            # Pre-cleanup evidence first: it is the only moment the stalled
            # process group can still be observed.
            snapshot = _safe(self._resource_snapshot)
            cleanup = self._cleanup_preserving(exc)
            self._write_evidence('failure', {
                'command': command, 'previous_response': self.last, 'error': repr(exc),
                'partial_response': self.buffer.decode(errors='replace')[-PARTIAL_TAIL:],
                'consumed_line': (raw.decode(errors='replace')[-PARTIAL_TAIL:]
                                  if isinstance(raw, bytes) else None),
                'worker_command': self.command, 'metrics': metrics,
                'resources': snapshot, 'cleanup': cleanup})
            raise

    def _write_all(self, line, deadline, metrics):
        """Write the whole command or raise; never partial-and-silent, never retried."""
        metrics['phase'] = 'write'
        fd = self.process.stdin.fileno()
        view = memoryview(line)
        while metrics['bytes_written'] < len(line):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                metrics['timed_out'] = True
                raise TimeoutError('Worker command write deadline exceeded')
            ready = select.select([], [fd], [], remaining)[1]
            metrics['select_wakeups_write'] += 1
            metrics['select_wakeups'] += 1
            if not ready:
                metrics['timed_out'] = True
                raise TimeoutError('Worker command write deadline exceeded')
            try:
                written = os.write(fd, view[metrics['bytes_written']:])
            except (BlockingIOError, InterruptedError):
                continue
            metrics['write_calls'] += 1
            if written == 0:
                raise RuntimeError('Worker accepted no command bytes')
            metrics['bytes_written'] += written
        metrics['phase'] = 'written'

    def _read_line(self, deadline, metrics):
        metrics['phase'] = 'read'
        fd = self.process.stdout.fileno()
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                metrics['timed_out'] = True
                raise TimeoutError('Worker response deadline exceeded')
            ready = select.select([fd], [], [], remaining)[0]
            metrics['select_wakeups_read'] += 1
            metrics['select_wakeups'] += 1
            if not ready:
                metrics['timed_out'] = True
                raise TimeoutError('Worker response deadline exceeded')
            try:
                block = os.read(fd, READ_CHUNK)
            except (BlockingIOError, InterruptedError):
                continue
            metrics['read_calls'] += 1
            if not block:
                raise RuntimeError('Worker exited before complete response')
            self.buffer += block
            metrics['bytes_read'] += len(block)
            metrics['buffered_bytes_after'] = len(self.buffer)
            if len(self.buffer) > RESPONSE_LIMIT:
                raise RuntimeError('Worker response exceeds limit')
        raw, self.buffer = self.buffer.split(b'\n', 1)
        metrics['buffered_bytes_after'] = len(self.buffer)
        return raw

    def _finalize(self, metrics):
        metrics['elapsed_monotonic'] = time.monotonic() - metrics['started_monotonic']
        metrics['elapsed_wall'] = time.time() - metrics['started_wall']
        metrics['returncode'] = self.process.poll() if self.process is not None else None

    # ------------------------------------------------------------- evidence

    def _write_evidence(self, kind, record):
        """Best-effort evidence; an I/O problem here never replaces the caller's error."""
        try:
            path = self.evidence / f'{kind}-{time.time_ns()}.json'
            path.write_text(json.dumps(record, indent=2, default=repr))
            return path
        except BaseException as exc:  # noqa: BLE001 - diagnostics must not mask
            self.evidence_error = repr(exc)
            return None

    def _resource_snapshot(self):
        """Read-only, bounded snapshot taken *before* cleanup."""
        snapshot = {
            'monotonic': time.monotonic(), 'wall': time.time(), 'coordinator_pid': os.getpid(),
            'worker_pid': None if self.process is None else self.process.pid,
            'worker_returncode': None if self.process is None else self.process.poll(),
            'pgid': self.pgid, 'buffered_bytes': len(self.buffer),
        }
        if _resource is not None:
            for name, who in (('self', _resource.RUSAGE_SELF), ('children', _resource.RUSAGE_CHILDREN)):
                try:
                    usage = _resource.getrusage(who)
                    snapshot[f'rusage_{name}'] = {'utime': usage.ru_utime, 'stime': usage.ru_stime,
                                                  'maxrss': usage.ru_maxrss}
                except (OSError, ValueError) as exc:
                    snapshot[f'rusage_{name}_error'] = repr(exc)
            try:
                snapshot['rlimit_nofile'] = list(_resource.getrlimit(_resource.RLIMIT_NOFILE))
            except (OSError, ValueError) as exc:
                snapshot['rlimit_nofile_error'] = repr(exc)
        try:
            snapshot['open_descriptors'] = len(os.listdir('/dev/fd'))
        except OSError as exc:
            snapshot['open_descriptors_error'] = repr(exc)
        snapshot['process_group'] = self._group_snapshot()
        return snapshot

    def _group_snapshot(self):
        """`ps` restricted to the owned process group.

        Read-only, no shell, no inherited environment, <=1 s per attempt and
        at most ``len(PS_VARIANTS)`` attempts; rows are filtered on the owned
        pgid so a looser selection dialect can never widen the scope.
        """
        if self.pgid is None:
            return {'available': False, 'reason': 'no owned process group'}
        out = {'pgid': self.pgid, 'available': False, 'attempts': [],
               'fields': ['pid', 'ppid', 'pgid', 'stat', 'etime', 'rss_kib', 'cpu_percent', 'comm']}
        fallback = None
        for flags in PS_VARIANTS:
            argv = ['ps', '-o', PS_FIELDS, *flags, str(self.pgid)]
            attempt = {'argv': argv}
            try:
                done = subprocess.run(argv, cwd='/', env=PS_ENV, stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True, timeout=PS_TIMEOUT, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                attempt['error'] = repr(exc)
                out['attempts'].append(attempt)
                continue
            rows = []
            for raw in done.stdout.splitlines()[:PS_MAX_LINES]:
                fields = raw.split(None, 5)
                if len(fields) >= 3 and fields[2] == str(self.pgid):
                    rows.append(raw.strip()[:PS_MAX_ROW])
            attempt.update(returncode=done.returncode, rows=rows,
                           stderr=done.stderr.strip()[:PS_MAX_ROW],
                           lines=len(done.stdout.splitlines()))
            out['attempts'].append(attempt)
            if done.returncode == 0:
                if rows:
                    out.update(available=True, argv=argv, rows=rows, members=len(rows))
                    return out
                if fallback is None:
                    fallback = argv  # group already gone, or dialect selects nothing
        if fallback is not None:
            out.update(available=True, argv=fallback, rows=[], members=0)
        return out

    # -------------------------------------------------------------- cleanup

    def _cleanup_preserving(self, primary):
        """Retain the in-flight failure even if cleanup itself is interrupted."""
        try:
            result = self._cleanup()
        except BaseException as exc:
            result = self.cleanup_result or {'errors': [], 'closed_streams': []}
            result['errors'].append(f'cleanup interrupted: {exc!r}')
            # A failure outside the individual cleanup guards still must not
            # prevent descriptor closure or silently claim successful cleanup.
            for name in ('stdin', 'stdout', 'stderr'):
                stream = self.stderr if name == 'stderr' else getattr(self.process, name, None)
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                        result['closed_streams'].append(name)
                except BaseException as close_error:
                    result['errors'].append(f'close {name}: {close_error!r}')
            result['ok'] = False
            self._closed = True
            self.cleanup_result = result
            self._write_evidence('cleanup-interruption', result)
        if not result.get('ok', False):
            primary.add_note('Worker cleanup: ' + repr(result.get('errors')))
        return result

    def _cleanup(self):
        """Idempotent bounded cleanup; retain errors and attempt every resource."""
        if self._closed:
            result = dict(self.cleanup_result or {'ok': True, 'errors': []})
            result['already_closed'] = True
            return result
        self._closed = True
        started = time.monotonic()
        result = {'ok': True, 'already_closed': False, 'errors': [], 'steps': [],
                  'pid': None, 'pgid': self.pgid, 'returncode': None, 'killed_group': False,
                  'wait_timed_out': False, 'closed_streams': [], 'reaped': [],
                  'started_wall': time.time()}
        self.cleanup_result = result
        process = self.process
        if process is not None:
            result['pid'] = process.pid
            result['alive_before_cleanup'] = process.poll() is None
            if self.pgid is not None and hasattr(os, 'killpg'):
                # The worker owns its session, so the group signal reaches the
                # worker and every descendant it spawned, and nothing else.
                try:
                    os.killpg(self.pgid, signal.SIGKILL)
                    result['killed_group'] = True
                except ProcessLookupError:
                    result['steps'].append('killpg: no such process group')
                except BaseException as exc:
                    result['errors'].append(f'killpg: {exc!r}')
            if process.poll() is None:
                try:
                    process.kill()
                    result['steps'].append('kill: direct')
                except ProcessLookupError:
                    result['steps'].append('kill: already gone')
                except BaseException as exc:
                    result['errors'].append(f'kill: {exc!r}')
            try:
                result['returncode'] = process.wait(timeout=CLEANUP_WAIT)
            except subprocess.TimeoutExpired as exc:
                result['wait_timed_out'] = True
                result['errors'].append(f'wait: {exc!r}')
            except BaseException as exc:
                result['errors'].append(f'wait: {exc!r}')
            try:
                result['reaped'] = self._reap_group()
            except BaseException as exc:
                result['errors'].append(f'reap: {exc!r}')
        for name in ('stdin', 'stdout', 'stderr'):
            stream = self.stderr if name == 'stderr' else getattr(process, name, None)
            try:
                if stream is not None and not stream.closed:
                    stream.close()
                    result['closed_streams'].append(name)
            except BaseException as exc:
                result['errors'].append(f'close {name}: {exc!r}')
        result['ok'] = not result['errors']
        if not result['ok'] or result['wait_timed_out']:
            # Only pay for `ps` when something actually went wrong.
            result['residual_group'] = _safe(self._group_snapshot)
        result['elapsed_monotonic'] = time.monotonic() - started
        self.cleanup_result = result
        self._write_evidence('cleanup', {'cleanup': result, 'command': self.command,
                                         'recent_requests': self.history})
        return result

    def _reap_group(self):
        """Reap whatever of the owned group is actually our child; bounded, no raises."""
        reaped = []
        if self.pgid is None or not hasattr(os, 'waitpid'):
            return reaped
        for _ in range(64):
            try:
                pid, status = os.waitpid(-self.pgid, os.WNOHANG)
            except (ChildProcessError, OSError):
                break
            if pid == 0:
                break
            reaped.append({'pid': pid, 'status': status})
        return reaped

    def close(self):
        result = self._cleanup()
        if not result.get('ok', True):
            raise WorkerCleanupError(f'Worker cleanup failed: {result.get("errors")}', result)
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self._cleanup_preserving(exc)
        return False
