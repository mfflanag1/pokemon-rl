"""Cooperative stage termination and birth-checked cleanup of owned descendants."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time


class StageTerminated(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__('F35 stage termination requested')


class _BSDInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        'flags', 'status', 'xstatus', 'pid', 'ppid', 'uid', 'gid', 'ruid', 'rgid', 'svuid', 'svgid', 'reserved')]
    _fields_ += [('comm', ctypes.c_char * 16), ('name', ctypes.c_char * 32)]
    _fields_ += [(name, ctypes.c_uint32) for name in ('nfiles', 'pgid', 'jobc', 'tdev', 'tpgid')]
    _fields_ += [('nice', ctypes.c_int32), ('start_sec', ctypes.c_uint64), ('start_usec', ctypes.c_uint64)]


@lru_cache(maxsize=1)
def _libproc():
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    library.proc_pidinfo.restype = ctypes.c_int
    return library


def process_identity(pid):
    """PID plus kernel birth identity; parent/group can change after reparenting."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('Positive process PID required')
    if sys.platform == 'darwin':
        data = _BSDInfo()
        result = _libproc().proc_pidinfo(pid, 3, 0, ctypes.byref(data), ctypes.sizeof(data))
        if result == 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return None
            status = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'stat='],
                capture_output=True, text=True, timeout=1., env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
            if status.returncode == 1 and not status.stdout.strip():
                return None
            if status.returncode == 0 and status.stdout.strip().startswith('Z'):
                return {'pid': pid, 'ppid': 0, 'pgid': 0, 'uid': os.getuid(),
                        'birth': 'unavailable-zombie', 'zombie': True, 'name': '<zombie>'}
            raise RuntimeError('Cannot obtain process birth identity: ' + str(pid))
        if result != ctypes.sizeof(data) or data.pid != pid:
            raise RuntimeError('Unexpected macOS process identity layout')
        return {'pid': pid, 'ppid': int(data.ppid), 'pgid': int(data.pgid), 'uid': int(data.uid),
            'birth': f'macos:{data.start_sec}:{data.start_usec}', 'zombie': data.status == 5,
            'name': bytes(data.comm).decode(errors='replace')}
    if sys.platform.startswith('linux'):
        root = Path('/proc') / str(pid)
        try:
            raw = (root / 'stat').read_text()
            fields = raw.rsplit(')', 1)[1].split()
            uid = root.stat().st_uid
        except FileNotFoundError:
            return None
        return {'pid': pid, 'ppid': int(fields[1]), 'pgid': int(fields[2]), 'uid': uid,
            'birth': 'linux:' + fields[19], 'zombie': fields[0] == 'Z',
            'name': raw.split('(', 1)[1].rsplit(')', 1)[0]}
    raise RuntimeError('Birth-checked cleanup supports macOS and Linux only')


def _same_process(left, right):
    return right is not None and all(left[k] == right[k] for k in ('pid', 'uid', 'birth'))


def _parent_table():
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid='], capture_output=True,
        text=True, timeout=1., check=True, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
    return {int(parts[0]): (int(parts[1]), int(parts[2]))
        for line in result.stdout.splitlines() if len(parts := line.split()) == 3}


def _refresh(snapshot):
    table = _parent_table()
    owned = snapshot['owned']
    live = set()
    for pid, previous in list(owned.items()):
        current = process_identity(pid)
        if current is None or current['zombie']:
            if len(snapshot['retired']) >= 20000:
                raise RuntimeError('Owned process retirement evidence limit exceeded')
            snapshot['retired'].append({**previous, 'retired_epoch': time.time(),
                'reason': 'gone' if current is None else 'zombie'})
            del owned[pid]
        elif _same_process(previous, current):
            live.add(pid)
    while True:
        discovered = []
        for pid, (parent, _group) in table.items():
            if pid in owned or parent not in live:
                continue
            current = process_identity(pid)
            if (current is not None and current['ppid'] == parent
                    and current['uid'] == snapshot['owner_uid']):
                owned[pid] = current
                if not current['zombie']:
                    discovered.append(pid)
        if not discovered:
            break
        live.update(discovered)


def capture_tree(child, snapshot=None):
    """Capture now and merge descendants into the same snapshot on later polls."""
    if not isinstance(child, subprocess.Popen) or child.pid == os.getpid():
        raise ValueError('Expected an owned Popen child')
    if snapshot is None:
        if child.poll() is not None:
            raise ValueError('Cannot establish ownership after the stage exited')
        identity = process_identity(child.pid)
        if identity is None or identity['ppid'] != os.getpid() or identity['uid'] != os.getuid():
            raise ValueError('Stage is not a live child owned by this process')
        snapshot = {'root_pid': child.pid, 'owner_pid': os.getpid(), 'owner_uid': os.getuid(),
            'captured_epoch': time.time(), 'root_identity': identity, 'owned': {child.pid: identity}, 'retired': []}
    elif (snapshot['root_pid'] != child.pid or snapshot['owner_pid'] != os.getpid()
            or snapshot['owner_uid'] != os.getuid() or snapshot['root_identity']['pid'] != child.pid):
        raise ValueError('Snapshot is not owned by this driver and stage')
    _refresh(snapshot)
    return snapshot


def _alive(snapshot, skipped):
    live = []
    for pid, previous in snapshot['owned'].items():
        current = process_identity(pid)
        if current is None or current['zombie']:
            continue
        if not _same_process(previous, current):
            if pid not in skipped:
                skipped.append(pid)
            continue
        live.append(pid)
    return live


def _send(snapshot, pid, signum, signals, skipped):
    previous = snapshot['owned'][pid]
    current = process_identity(pid)
    if current is None or current['zombie']:
        return
    if not _same_process(previous, current):
        if pid not in skipped:
            skipped.append(pid)
        return
    try:
        # Individual PIDs avoid signalling unrelated members of a reused group.
        os.kill(pid, signum)
        signals.append({'pid': pid, 'pgid': current['pgid'], 'birth': current['birth'],
                        'signal': int(signum), 'epoch': time.time()})
    except ProcessLookupError:
        pass


def terminate_tree(child, *, snapshot=None, grace_seconds=10., kill_wait_seconds=5., poll_seconds=.05):
    """Capture before TERM, unwind, then KILL remaining birth-matched descendants.

    The driver must retain the returned evidence and treat a non-passed cleanup as
    a failure. No process is selected by name, and no unowned process group is hit.
    """
    if any(not isinstance(x, (int, float)) or not 0 < x <= 30
           for x in (grace_seconds, kill_wait_seconds, poll_seconds)):
        raise ValueError('Cleanup intervals must be positive and bounded by 30 seconds')
    if snapshot is None:
        snapshot = capture_tree(child)
    if (not isinstance(child, subprocess.Popen) or snapshot['root_pid'] != child.pid
            or snapshot['owner_pid'] != os.getpid() or snapshot['owner_uid'] != os.getuid()
            or snapshot['root_identity']['pid'] != child.pid):
        raise ValueError('Cleanup snapshot is not owned by this driver and stage')
    started, signals, skipped, errors = time.monotonic(), [], [], []
    def refresh():
        try:
            _refresh(snapshot)
        except Exception as exc:
            errors.append(repr(exc))
    refresh()  # descendants in new sessions must be captured before reparenting
    stage_was_running = child.poll() is None
    initial_live = _alive(snapshot, skipped)
    for signum, duration in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, kill_wait_seconds)):
        deadline = time.monotonic() + duration
        sent = set()
        while True:
            refresh()
            try:
                live = _alive(snapshot, skipped)
                # Let Python unwind first; descendants then receive the same TERM.
                order = sorted(live, key=lambda pid: (pid != child.pid, pid))
                for pid in order:
                    if pid not in sent:
                        _send(snapshot, pid, signum, signals, skipped)
                        sent.add(pid)
            except Exception as exc:
                errors.append(repr(exc))
                break
            child.poll()
            if not live or time.monotonic() >= deadline:
                break
            time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
    try:
        remaining = _alive(snapshot, skipped)
    except Exception as exc:
        errors.append(repr(exc))
        remaining = sorted(snapshot['owned'])
    child.poll()
    return {'status': 'passed' if not remaining and not errors and not skipped else 'failed',
        'stage_was_running': stage_was_running, 'initial_live_pids': initial_live,
        'live_descendants_at_cleanup': [pid for pid in initial_live if pid != child.pid],
        'root_pid': child.pid, 'returncode': child.returncode,
        'owned': list(snapshot['owned'].values()), 'retired': snapshot['retired'], 'signals': signals,
        'identity_changed_skipped': skipped, 'remaining_pids': remaining, 'errors': errors,
        'elapsed_seconds': time.monotonic() - started}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == '--':
        arguments.pop(0)
    if not arguments or re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', arguments[0]) is None:
        raise SystemExit('Usage: python -m pokemon_teacher_assisted.lifecycle -- package.module [arguments]')
    terminating = False
    def terminate(signum, _frame):
        nonlocal terminating
        if not terminating:
            terminating = True
            raise StageTerminated(signum)
    previous, old_argv = signal.signal(signal.SIGTERM, terminate), sys.argv
    sys.argv = arguments
    try:
        runpy.run_module(arguments[0], run_name='__main__', alter_sys=True)
    except StageTerminated as exc:
        return 128 + exc.signum
    finally:
        sys.argv = old_argv
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
