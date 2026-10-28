"""Bounded JSON-lines transport for the two new offline worker protocols.

Additive F36 replacement for the frozen external transport. The wire protocol
is unchanged: one JSON object per line on stdin/stdout, monotonically
increasing integer request ``id`` echoed by the worker, ``{'ok': True,
'data': ...}`` accepted and anything else rejected, the same per-request
timeout, the same 65536-byte read quantum and 4,000,000-byte response bound,
the same worker argv, cwd and ``PYTHONPATH`` environment, the same
``start_new_session=True`` launch, unbuffered pipes and exclusive stderr file.
No action, payload or accepted-response semantics are altered.

Repaired defect
---------------
The frozen ``close`` signalled the worker's whole process group with
``os.killpg`` and caught only ``ProcessLookupError``. In F35 that raised
``PermissionError: [Errno 1] Operation not permitted`` while a cooperative
``StageTerminated`` was unwinding; the cleanup error replaced the original
exception, so the stage lost its cooperative exit code and exited 1.

This version:

* closes stdin first and allows a bounded clean EOF exit, then terminates only
  the **direct child** through its own owned ``Popen``
  handle - never a blanket process-group kill. An unreaped direct child's PID
  cannot be reused, so the handle is itself the ownership proof.
* **Scope:** subordinate descendants (the worker's own children, including
  processes it starts in other sessions) are intentionally *not* signalled
  here. They are owned and reclaimed by the supervisor's birth-checked process
  tree cleanup (``pokemon_teacher_runtime.lifecycle``), which verifies kernel
  birth identity before signalling individual PIDs.
* is idempotent, always closes all three pipe/file handles, and bounds
  EOF -> TERM -> KILL -> wait, including constructor failures after launch.
* never lets a cleanup failure replace a request or context exception: the
  original exception propagates and cleanup errors are attached as secondary
  diagnostics. A cleanup failure in an otherwise successful context is still
  reported, by raising ``WorkerCleanupError``.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import select
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


class WorkerCleanupError(RuntimeError):
    """Raised when worker cleanup failed and no original exception exists."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__('External worker cleanup failed: ' + '; '.join(self.errors))


def _attach_cleanup_errors(exception, errors):
    """Attach cleanup diagnostics without replacing the original exception."""
    if not errors:
        return
    existing = list(getattr(exception, 'cleanup_errors', []) or [])
    exception.cleanup_errors = existing + list(errors)
    note = 'External worker cleanup also failed: ' + '; '.join(errors)
    add_note = getattr(exception, 'add_note', None)
    if callable(add_note):
        try:
            add_note(note)
        except Exception:
            pass


class JsonWorker:
    def __init__(self, command, stderr_path, timeout=60, *, eof_wait=1., term_wait=3., kill_wait=3.):
        self.sequence = 0
        self.buffer = b''
        self.timeout = timeout
        self.closed = False
        self.process = None
        self.cleanup_errors = []
        self.eof_wait = eof_wait
        self.term_wait = term_wait
        self.kill_wait = kill_wait
        path = Path(stderr_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = path.open('xb')
        try:
            env = {**os.environ, 'PYTHONPATH': str(ROOT / 'python')}
            self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0, start_new_session=True)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
        except BaseException as exc:
            _attach_cleanup_errors(exc, self._shutdown())
            raise

    def request(self, op, **payload):
        if self.closed:
            raise RuntimeError('Closed external worker')
        self.sequence += 1
        deadline = time.monotonic() + self.timeout
        raw = (json.dumps({'id': self.sequence, 'op': op, **payload}, allow_nan=False) + '\n').encode()
        written = 0
        try:
            while written < len(raw):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    raise TimeoutError('External worker write timed out')
                try:
                    n = os.write(self.process.stdin.fileno(), raw[written:])
                except BlockingIOError:
                    continue
                if n == 0:
                    raise RuntimeError('External worker pipe closed')
                written += n
            while b'\n' not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError('External worker read timed out')
                try:
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise RuntimeError('External worker exited ' + str(self.process.poll()))
                self.buffer += chunk
                if len(self.buffer) > 4_000_000:
                    raise ValueError('External worker response size')
            line, self.buffer = self.buffer.split(b'\n', 1)
            response = json.loads(line)
            if response.get('id') != self.sequence or response.get('ok') is not True:
                raise RuntimeError('External worker rejected command: ' + json.dumps(response))
            return response['data']
        except BaseException as exc:
            # The request failure is the scientific evidence; cleanup trouble is
            # only secondary diagnostics and must never replace it.
            _attach_cleanup_errors(exc, self._shutdown())
            raise

    def _shutdown(self):
        """Idempotent bounded cleanup. Returns a list of cleanup error reprs."""
        if self.closed:
            return []
        self.closed = True
        errors = []
        process = self.process

        def close_handle(handle):
            if handle is not None:
                try:
                    handle.close()
                except OSError as exc:
                    errors.append(repr(exc))

        def wait(seconds):
            try:
                process.wait(timeout=seconds)
                return True
            except subprocess.TimeoutExpired:
                return False
            except OSError as exc:
                errors.append(repr(exc))
                return False

        try:
            # Both production worker loops finish naturally when their input
            # closes. This also releases a child launched before setup failed.
            close_handle(getattr(process, 'stdin', None))
            if process is not None and not wait(self.eof_wait):
                for method, seconds in ((process.terminate, self.term_wait),
                                        (process.kill, self.kill_wait)):
                    # Only our own direct child; descendants are supervisor-owned.
                    try:
                        method()
                    except ProcessLookupError:
                        pass
                    except OSError as exc:
                        errors.append(repr(exc))
                    if wait(seconds):
                        break
                else:
                    errors.append('Owned child did not exit within bounded cleanup waits')
        finally:
            for handle in (getattr(process, 'stdin', None), getattr(process, 'stdout', None),
                           self.stderr):
                close_handle(handle)
            self.cleanup_errors = errors
        return errors

    def close(self, *, report=True):
        """Idempotent cleanup. Reports cleanup failure unless report=False.

        With ``report=True`` a cleanup failure raises ``WorkerCleanupError`` so
        it cannot be silently lost; with ``report=False`` the error reprs are
        returned instead (used when an original exception must survive).
        """
        errors = self._shutdown()
        if errors and report:
            raise WorkerCleanupError(errors)
        return errors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()  # a cleanup failure is reported, never swallowed
            return False
        _attach_cleanup_errors(exc, self._shutdown())
        return False
