"""Cooperative stage termination and birth-checked cleanup of owned descendants.

Additive F36 replacement for the F35 supervisor helpers. The public interface
(`StageTerminated`, `process_identity`, `capture_tree`, `terminate_tree`,
`main`) keeps the F35 signatures and roles. Nothing here selects a process by
name, signals a process group, or signals an unverified or reused PID, and
nothing retries or resamples scientific work.

Repaired defect
---------------
F35 raised ``RuntimeError('Cannot obtain process birth identity: <pid>')`` when
the kernel/inspection probes disagreed for a churning PID. The supervisor
recorded that as a stage wait error and terminated acquisition on an inconclusive observation.

Here every observation is explicitly tri-state:

* ``LIVE``   - a verified kernel birth identity (``pid``/``uid``/``birth``).
* ``GONE``   - *confirmed* absent: ``kill(pid, 0)`` reported ESRCH, ``ps``
  reported the PID absent, ``/proc`` entry absent with ESRCH, or our own
  ``Popen`` handle was already reaped.
* ``UNKNOWN``- a transient identity or process-table read failure, carrying
  bounded structured diagnostics (libproc result/errno, ``ps`` return code and
  truncated output, exception reprs).

``UNKNOWN`` is never ``GONE``, is never signalled, and is never by itself a
reason to abort healthy science: the previously owned identity is preserved and
re-checked on later polls. Unknowns that remain unresolved when bounded cleanup
finishes are reported as *failed* cleanup (fail-closed), never as success.

Observation shape: ``process_identity`` always returns a dict and never
``None``; callers must branch on ``observation['state']`` (helpers `is_live`,
`is_gone`, `is_unknown` are provided). ``'zombie'``, ``'pid'`` and ``'birth'``
keys exist in every state so that existing attribute reads do not explode.

Telemetry is bounded: retirement, unknown-observation and event lists are
capped with overflow counters instead of growing without limit or raising.
"""
from __future__ import annotations

import ctypes
from contextvars import ContextVar
from functools import lru_cache, wraps
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time
import traceback

LIVE = 'live'
GONE = 'gone'
UNKNOWN = 'unknown'

_PROBE_ENV = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}
_PROBE_TIMEOUT = .2
_PROBE_DEADLINE = ContextVar('f36_probe_deadline', default=None)
_TEXT_LIMIT = 200
_MAX_RETIRED = 20000
_MAX_EVENTS = 500
_MAX_UNKNOWN_PIDS = 4096
_MAX_DISCOVERY_ROUNDS = 64
_MIN_SLEEP = .005


def _probe_timeout():
    deadline = _PROBE_DEADLINE.get()
    remaining = _PROBE_TIMEOUT if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Shared process observation deadline reached')
    return min(_PROBE_TIMEOUT, remaining)


def _cleanup_budget(function):
    @wraps(function)
    def bounded(child, **kwargs):
        grace = kwargs.get('grace_seconds', 10.)
        kill = kwargs.get('kill_wait_seconds', 5.)
        if not all(isinstance(x, (int, float)) and 0 < x <= 30 for x in (grace, kill)):
            return function(child, **kwargs)
        token = _PROBE_DEADLINE.set(time.monotonic() + grace + kill)
        try:
            return function(child, **kwargs)
        finally:
            _PROBE_DEADLINE.reset(token)
    return bounded


def _capture_budget(function):
    @wraps(function)
    def bounded(child, snapshot=None):
        deadline = time.monotonic() + .25
        existing = _PROBE_DEADLINE.get()
        if existing is not None:
            deadline = min(deadline, existing)
        token = _PROBE_DEADLINE.set(deadline)
        try:
            return function(child, snapshot)
        finally:
            _PROBE_DEADLINE.reset(token)
    return bounded


class StageTerminated(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__('F36 stage termination requested')


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


def _clip(text):
    text = '' if text is None else str(text)
    return text if len(text) <= _TEXT_LIMIT else text[:_TEXT_LIMIT] + '...<clipped>'


def _live(pid, ppid, pgid, uid, birth, zombie, name, birth_verified=True):
    return {'state': LIVE, 'pid': pid, 'ppid': int(ppid), 'pgid': int(pgid), 'uid': int(uid),
            'birth': birth, 'zombie': bool(zombie), 'name': name, 'birth_verified': birth_verified}


def _gone(pid, evidence):
    return {'state': GONE, 'pid': pid, 'birth': None, 'zombie': False,
            'evidence': evidence, 'epoch': time.time()}


def _unknown(pid, diagnostic):
    return {'state': UNKNOWN, 'pid': pid, 'birth': None, 'zombie': False,
            'diagnostic': diagnostic, 'epoch': time.time()}


def is_live(observation):
    return isinstance(observation, dict) and observation.get('state') == LIVE


def is_gone(observation):
    return isinstance(observation, dict) and observation.get('state') == GONE


def is_unknown(observation):
    return isinstance(observation, dict) and observation.get('state') == UNKNOWN


def _presence(pid):
    """Kernel presence probe: ('gone'|'present'|'unknown', diagnostic)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        return 'gone', {'probe': 'kill0', 'errno': exc.errno}
    except PermissionError as exc:
        # Present but not ours to verify or signal; treat as unverifiable.
        return 'unknown', {'probe': 'kill0', 'errno': exc.errno, 'note': 'signal permission denied'}
    except OSError as exc:
        return 'unknown', {'probe': 'kill0', 'errno': exc.errno, 'error': _clip(repr(exc))}
    return 'present', {'probe': 'kill0', 'errno': 0}


def _ps_probe(pid):
    """Bounded read-only `ps` probe. Returns (completed_or_None, diagnostic)."""
    try:
        done = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'stat=,ppid=,pgid=,uid='],
            capture_output=True, text=True, timeout=_probe_timeout(), env=_PROBE_ENV)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, {'probe': 'ps', 'error': _clip(repr(exc))}
    return done, {'probe': 'ps', 'returncode': done.returncode,
                  'stdout': _clip(done.stdout), 'stderr': _clip(done.stderr)}


def _zombie_from_ps(pid, row):
    """Zombie identity: birth is unreadable, so it is marked unverified.

    A zombie still occupies its PID slot, so it cannot be a reused PID holding
    a different *live* process. It is never signalled; refresh retires it.
    """
    fields = row.split()
    ppid = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else 0
    pgid = int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else 0
    uid = int(fields[3]) if len(fields) > 3 and fields[3].isdigit() else os.getuid()
    return _live(pid, ppid, pgid, uid, 'unavailable-zombie', True, '<zombie>', birth_verified=False)


def _macos_identity(pid):
    data = _BSDInfo()
    try:
        ctypes.set_errno(0)
        result = _libproc().proc_pidinfo(pid, 3, 0, ctypes.byref(data), ctypes.sizeof(data))
        errno_value = ctypes.get_errno()
    except OSError as exc:
        return _unknown(pid, {'probe': 'libproc', 'error': _clip(repr(exc))})
    if result == ctypes.sizeof(data) and data.pid == pid:
        return _live(pid, data.ppid, data.pgid, data.uid,
            f'macos:{data.start_sec}:{data.start_usec}', data.status == 5,
            bytes(data.comm).decode(errors='replace'))
    diagnostic = {'probe': 'libproc', 'result': int(result), 'errno': int(errno_value),
                  'reported_pid': int(data.pid), 'expected_pid': pid}
    if result != 0:
        # F35 raised 'Unexpected macOS process identity layout' here.
        return _unknown(pid, diagnostic)
    presence, presence_diagnostic = _presence(pid)
    if presence == 'gone':
        return _gone(pid, {'libproc': diagnostic, 'kill0': presence_diagnostic})
    row, ps_diagnostic = _ps_probe(pid)
    evidence = {'libproc': diagnostic, 'kill0': presence_diagnostic, 'ps': ps_diagnostic}
    if row is not None:
        text = row.stdout.strip()
        if row.returncode == 1 and not text:
            return _gone(pid, evidence)
        if row.returncode == 0 and text and text.split()[0].startswith('Z'):
            return _zombie_from_ps(pid, text)
    # F35 raised 'Cannot obtain process birth identity' here; that aborted the
    # healthy stage. The condition is transient and now reported as UNKNOWN.
    return _unknown(pid, evidence)


def _linux_identity(pid):
    root = Path('/proc') / str(pid)
    try:
        raw = (root / 'stat').read_text()
        uid = root.stat().st_uid
    except FileNotFoundError as exc:
        presence, presence_diagnostic = _presence(pid)
        if presence == 'gone':
            return _gone(pid, {'probe': 'proc', 'errno': exc.errno, 'kill0': presence_diagnostic})
        return _unknown(pid, {'probe': 'proc', 'errno': exc.errno, 'error': _clip(repr(exc)),
                              'kill0': presence_diagnostic})
    except OSError as exc:
        return _unknown(pid, {'probe': 'proc', 'errno': exc.errno, 'error': _clip(repr(exc))})
    try:
        fields = raw.rsplit(')', 1)[1].split()
        return _live(pid, fields[1], fields[2], uid, 'linux:' + fields[19], fields[0] == 'Z',
            raw.split('(', 1)[1].rsplit(')', 1)[0])
    except (IndexError, ValueError) as exc:
        return _unknown(pid, {'probe': 'proc-parse', 'error': _clip(repr(exc))})


def process_identity(pid):
    """Tri-state observation: LIVE with kernel birth identity, GONE or UNKNOWN.

    PID plus kernel birth identity; parent/group can change after reparenting.
    A transient inspection failure yields an UNKNOWN observation rather than an
    exception, so periodic capture of a churning tree cannot abort the stage.
    """
    if type(pid) is not int or pid <= 0:
        raise ValueError('Positive process PID required')
    if sys.platform == 'darwin':
        return _macos_identity(pid)
    if sys.platform.startswith('linux'):
        return _linux_identity(pid)
    raise RuntimeError('Birth-checked cleanup supports macOS and Linux only')


def _observe(pid):
    """process_identity with an UNKNOWN fallback for unexpected probe errors."""
    try:
        _probe_timeout()
        return process_identity(pid)
    except Exception as exc:  # never let an inspection defect stop the caller
        return _unknown(pid, {'probe': 'process_identity', 'error': _clip(repr(exc))})


def _same_process(left, right):
    return (is_live(left) and is_live(right)
            and all(left[k] == right[k] for k in ('pid', 'uid', 'birth')))


def _parent_table():
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid='], capture_output=True,
        text=True, timeout=_probe_timeout(), check=True, env=_PROBE_ENV)
    return {int(parts[0]): (int(parts[1]), int(parts[2]))
        for line in result.stdout.splitlines() if len(parts := line.split()) == 3}


def _new_report():
    return {'unknown': [], 'reused': [], 'resolved': [], 'retired': [], 'discovered': [],
            'table_error': None, 'events': []}


def _record(snapshot, report, event):
    event.setdefault('epoch', time.time())
    if len(snapshot['events']) < _MAX_EVENTS:
        snapshot['events'].append(event)
    else:
        snapshot['event_overflow'] += 1
    if report is not None and len(report['events']) < _MAX_EVENTS:
        report['events'].append(event)


def _retire(snapshot, snapshot_identity, reason, evidence, report):
    record = {**snapshot_identity, 'retired_epoch': time.time(), 'reason': reason,
              'evidence': evidence}
    if len(snapshot['retired']) < _MAX_RETIRED:
        snapshot['retired'].append(record)
    else:
        snapshot['retired_overflow'] += 1
    if report is not None:
        report['retired'].append({'pid': snapshot_identity.get('pid'), 'reason': reason})
    _record(snapshot, report, {'kind': 'retired', 'pid': snapshot_identity.get('pid'), 'reason': reason})


def _note_unknown(snapshot, report, pid, observation):
    entry = snapshot['unknown'].get(pid)
    if entry is None:
        if len(snapshot['unknown']) >= _MAX_UNKNOWN_PIDS:
            snapshot['unknown_overflow'] += 1
            return
        entry = {'pid': pid, 'first_epoch': time.time(), 'count': 0}
        snapshot['unknown'][pid] = entry
    entry['count'] += 1
    entry['last_epoch'] = time.time()
    entry['last_diagnostic'] = observation.get('diagnostic')
    if pid not in report['unknown']:
        report['unknown'].append(pid)
    _record(snapshot, report, {'kind': 'identity-unknown', 'pid': pid, 'count': entry['count'],
        'diagnostic': observation.get('diagnostic')})


def _clear_unknown(snapshot, report, pid, reason):
    if snapshot['unknown'].pop(pid, None) is not None:
        if pid not in report['resolved']:
            report['resolved'].append(pid)
        _record(snapshot, report, {'kind': 'identity-resolved', 'pid': pid, 'resolution': reason})


def _reconcile_root(snapshot, child, report):
    """Retry unverified root ownership; a reaped handle is authoritative GONE."""
    pid = snapshot['root_pid']
    if child.poll() is not None:
        # We reaped our own direct child, so this PID is retired for certain
        # even when identity inspection is failing.
        previous = snapshot['owned'].pop(pid, None)
        if previous is not None:
            _retire(snapshot, previous, 'gone',
                {'probe': 'popen-wait', 'returncode': child.returncode}, report)
        _clear_unknown(snapshot, report, pid, 'popen-reaped')
        if snapshot['root_ownership'] in ('unverified', 'verified'):
            snapshot['root_ownership'] = 'exited'
            _record(snapshot, report, {'kind': 'root-exited', 'pid': pid,
                'returncode': child.returncode})
        return
    if snapshot['root_ownership'] != 'unverified':
        return
    observation = _observe(pid)
    snapshot['root_observation'] = observation
    if is_unknown(observation):
        _note_unknown(snapshot, report, pid, observation)
        return
    if is_gone(observation):
        snapshot['root_ownership'] = 'exited'
        _clear_unknown(snapshot, report, pid, 'confirmed-gone')
        _record(snapshot, report, {'kind': 'root-gone', 'pid': pid})
        return
    if observation['zombie']:
        snapshot['root_ownership'] = 'exited'
        _clear_unknown(snapshot, report, pid, 'zombie')
        _record(snapshot, report, {'kind': 'root-zombie', 'pid': pid})
        return
    if observation['ppid'] == snapshot['owner_pid'] and observation['uid'] == snapshot['owner_uid']:
        snapshot['root_ownership'] = 'verified'
        snapshot['root_identity'] = observation
        snapshot['owned'].setdefault(pid, observation)
        _clear_unknown(snapshot, report, pid, 'verified')
        _record(snapshot, report, {'kind': 'root-ownership-verified', 'pid': pid})
        return
    # A live process that is not our child now holds the PID: never adopt it.
    snapshot['root_ownership'] = 'contradicted'
    snapshot['reused'][pid] = {'observed': observation, 'epoch': time.time()}
    _clear_unknown(snapshot, report, pid, 'contradicted')
    if pid not in report['reused']:
        report['reused'].append(pid)
    _record(snapshot, report, {'kind': 'root-identity-contradicted', 'pid': pid,
        'observed_ppid': observation['ppid'], 'observed_uid': observation['uid']})


def _refresh(snapshot, child, report):
    """One bounded poll: reconcile owned identities, then discover descendants."""
    snapshot['polls'] += 1
    _reconcile_root(snapshot, child, report)
    table = None
    try:
        table = _parent_table()
        report['table_error'] = None
    except Exception as exc:
        # A table read failure only costs this poll's discovery; owned
        # identities are still reconciled and nothing is declared gone.
        report['table_error'] = _clip(repr(exc))
        _record(snapshot, report, {'kind': 'table-unavailable', 'error': report['table_error']})
    owned, live = snapshot['owned'], set()
    # Unverified candidates survive parent exit; otherwise a vanished parent
    # would erase the only evidence of a potentially orphaned child.
    for pid in list(snapshot['unknown']):
        if pid in owned or pid == snapshot['root_pid']:
            continue
        observation = _observe(pid)
        if is_gone(observation) or (is_live(observation) and observation['zombie']):
            _clear_unknown(snapshot, report, pid, 'candidate-gone')
    for pid, previous in list(owned.items()):
        current = _observe(pid)
        if is_unknown(current):
            _note_unknown(snapshot, report, pid, current)  # keep previous identity, retry later
            continue
        if is_gone(current):
            _clear_unknown(snapshot, report, pid, 'confirmed-gone')
            _retire(snapshot, previous, 'gone', current.get('evidence'), report)
            del owned[pid]
            continue
        if current['zombie']:
            _clear_unknown(snapshot, report, pid, 'zombie')
            _retire(snapshot, previous, 'zombie', current.get('evidence'), report)
            del owned[pid]
            continue
        if _same_process(previous, current):
            owned[pid] = current
            _clear_unknown(snapshot, report, pid, 'verified')
            live.add(pid)
            continue
        # A different live process holds this PID, so the tracked one is gone.
        # The new identity is never adopted as the old process.
        snapshot['reused'][pid] = {'previous_birth': previous.get('birth'),
                                   'observed_birth': current.get('birth'), 'epoch': time.time()}
        if pid not in report['reused']:
            report['reused'].append(pid)
        if pid == snapshot['root_pid']:
            snapshot['root_ownership'] = 'contradicted'
        _clear_unknown(snapshot, report, pid, 'pid-reused')
        _retire(snapshot, previous, 'pid-reused', {'observed_birth': current.get('birth')}, report)
        del owned[pid]
    if table is not None:
        for _round in range(_MAX_DISCOVERY_ROUNDS):
            discovered = []
            for pid, (parent, _group) in table.items():
                if pid in owned or pid in snapshot['reused'] or parent not in live:
                    continue
                current = _observe(pid)
                if is_unknown(current):
                    _note_unknown(snapshot, report, pid, current)
                    _record(snapshot, report, {'kind': 'candidate-unverifiable', 'pid': pid,
                        'parent': parent, 'diagnostic': current.get('diagnostic')})
                    continue
                if (is_live(current) and current['ppid'] == parent
                        and current['uid'] == snapshot['owner_uid']):
                    owned[pid] = current
                    _clear_unknown(snapshot, report, pid, 'candidate-verified')
                    report['discovered'].append(pid)
                    if not current['zombie']:
                        discovered.append(pid)
            if not discovered:
                break
            live.update(discovered)
    snapshot['last_live'] = sorted(live)
    return live


def _new_snapshot(child):
    snapshot = {'root_pid': child.pid, 'owner_pid': os.getpid(), 'owner_uid': os.getuid(),
        'captured_epoch': time.time(), 'root_identity': None, 'root_ownership': 'unverified',
        'root_observation': None, 'owned': {}, 'retired': [], 'retired_overflow': 0,
        'unknown': {}, 'unknown_overflow': 0, 'reused': {}, 'events': [], 'event_overflow': 0,
        'polls': 0, 'last_live': []}
    if child.poll() is not None:
        # Fast stage exit: recorded, never raised, so cleanup still runs.
        snapshot['root_ownership'] = 'exited'
        snapshot['root_observation'] = _gone(child.pid,
            {'probe': 'popen-wait', 'returncode': child.returncode})
        return snapshot
    observation = _observe(child.pid)
    snapshot['root_observation'] = observation
    if is_live(observation) and not observation['zombie']:
        if observation['ppid'] != os.getpid() or observation['uid'] != os.getuid():
            raise ValueError('Stage is not a live child owned by this process')
        snapshot['root_ownership'] = 'verified'
        snapshot['root_identity'] = observation
        snapshot['owned'][child.pid] = observation
    elif is_live(observation) or is_gone(observation):
        snapshot['root_ownership'] = 'exited'  # zombie or confirmed gone
    # UNKNOWN leaves 'unverified': ownership certainty is not claimed and the
    # next poll retries. Cleanup can still reach the stage via its own handle.
    return snapshot


@_capture_budget
def capture_tree(child, snapshot=None):
    """Capture now and merge descendants into the same snapshot on later polls.

    A transient inspection failure never raises: the affected PID becomes an
    UNKNOWN observation with bounded diagnostics while its previously owned
    identity is retained. A stage that exited before or during the first
    capture is recorded (``root_ownership`` ``'exited'``/``'unverified'``)
    rather than raising, without claiming ownership certainty. Polling cannot
    establish absence of descendants that exited or reparented before observation.
    """
    if not isinstance(child, subprocess.Popen) or child.pid == os.getpid():
        raise ValueError('Expected an owned Popen child')
    if snapshot is None:
        snapshot = _new_snapshot(child)
    elif (snapshot['root_pid'] != child.pid or snapshot['owner_pid'] != os.getpid()
            or snapshot['owner_uid'] != os.getuid()
            or (snapshot['root_identity'] is not None
                and snapshot['root_identity']['pid'] != child.pid)):
        raise ValueError('Snapshot is not owned by this driver and stage')
    report = _new_report()
    _refresh(snapshot, child, report)
    snapshot['last_report'] = report
    return snapshot


def _root_needs_handle(snapshot, child):
    """True when the stage is reachable only through our own Popen handle.

    Only for an unverifiable *root*: an unreaped direct child's PID cannot be
    reused, so the handle itself proves ownership and no PID number is guessed.
    Never used for a contradicted/reused identity or for any descendant.
    """
    pid = snapshot['root_pid']
    if pid in snapshot['reused'] or snapshot['root_ownership'] == 'contradicted':
        return False
    if child.poll() is not None:
        return False
    return snapshot['root_ownership'] == 'unverified' or pid in snapshot['unknown']


def _kill(pid, signum):
    """Single-PID signal seam: never a process group and never by name."""
    os.kill(pid, signum)


def _send(snapshot, pid, signum, signals, skipped, errors):
    previous = snapshot['owned'].get(pid)
    if previous is None:
        return
    current = _observe(pid)
    if not is_live(current) or current['zombie']:
        return  # GONE or UNKNOWN identities are never signalled
    if not _same_process(previous, current):
        if pid not in skipped:
            skipped.append(pid)
        return
    try:
        # Individual PIDs avoid signalling unrelated members of a reused group.
        _kill(pid, signum)
        signals.append({'pid': pid, 'pgid': current['pgid'], 'birth': current['birth'],
                        'signal': int(signum), 'epoch': time.time(), 'basis': 'verified-birth'})
        return True
    except ProcessLookupError:
        return True
    except Exception as exc:
        errors.append(repr(exc))
    return False


def _send_handle(child, signum, handle_signals, errors):
    try:
        child.send_signal(signum)
        handle_signals.append({'pid': child.pid, 'signal': int(signum), 'epoch': time.time(),
                               'basis': 'owned-unreaped-popen-handle'})
    except (ProcessLookupError, ValueError):
        pass
    except OSError as exc:
        errors.append(repr(exc))


@_cleanup_budget
def terminate_tree(child, *, snapshot=None, grace_seconds=10., kill_wait_seconds=5., poll_seconds=.05):
    """Capture before TERM, unwind, then KILL remaining birth-matched descendants.

    The driver must retain the returned evidence and treat a non-passed cleanup
    as a failure. No process is selected by name, no process group is signalled,
    and no unknown or reused identity is signalled or assumed gone. Cleanup
    continues across individual inspection failures, retries signalling within
    the fixed TERM and KILL grace windows as identities resolve, and reports
    unresolved unknowns as failed cleanup. It never raises, so it cannot mask an
    original stage error.
    """
    if any(not isinstance(x, (int, float)) or not 0 < x <= 30
           for x in (grace_seconds, kill_wait_seconds, poll_seconds)):
        raise ValueError('Cleanup intervals must be positive and bounded by 30 seconds')
    if snapshot is None:
        snapshot = capture_tree(child)
    if (not isinstance(child, subprocess.Popen) or snapshot['root_pid'] != child.pid
            or snapshot['owner_pid'] != os.getpid() or snapshot['owner_uid'] != os.getuid()
            or (snapshot['root_identity'] is not None
                and snapshot['root_identity']['pid'] != child.pid)):
        raise ValueError('Cleanup snapshot is not owned by this driver and stage')
    started = time.monotonic()
    signals, handle_signals, skipped, errors = [], [], [], []
    report = _new_report()

    def refresh():
        try:
            _refresh(snapshot, child, report)
        except Exception as exc:
            errors.append(repr(exc))

    refresh()  # descendants in new sessions must be captured before reparenting
    stage_was_running = child.poll() is None
    initial_live = list(snapshot['last_live'])
    final_deadline = _PROBE_DEADLINE.get()
    for signum, deadline in ((signal.SIGTERM, final_deadline - kill_wait_seconds),
                             (signal.SIGKILL, final_deadline)):
        sent, handle_sent = set(), False
        while True:
            if time.monotonic() >= deadline:
                break
            refresh()
            try:
                live = list(snapshot['last_live'])
                # Signal the root first, then descendants in the same iteration;
                # this ordering does not guarantee an exclusive unwind interval.
                for pid in sorted(live, key=lambda pid: (pid != child.pid, pid)):
                    # Retry: resolved-late or newly discovered PIDs are signalled
                    # when first seen live; KILL is re-sent while still live.
                    if pid not in sent or signum == signal.SIGKILL:
                        if _send(snapshot, pid, signum, signals, skipped, errors):
                            sent.add(pid)
                if (_root_needs_handle(snapshot, child) and child.pid not in live
                        and (not handle_sent or signum == signal.SIGKILL)):
                    _send_handle(child, signum, handle_signals, errors)
                    handle_sent = True
            except Exception as exc:
                errors.append(repr(exc))
                break
            child.poll()
            outstanding = set(live) | set(snapshot['unknown'])
            if child.poll() is None and snapshot['root_ownership'] != 'verified':
                outstanding.add(snapshot['root_pid'])
            remaining_time = deadline - time.monotonic()
            if not outstanding or remaining_time <= 0:
                break
            time.sleep(min(poll_seconds, remaining_time))
    refresh()
    verified_live = list(snapshot['last_live'])
    unresolved = sorted(snapshot['unknown'])
    # At the shared deadline, fresh inspection may be unavailable. Keep every
    # live-or-unresolved candidate in the remaining inventory, without asserting
    # that an unresolved PID is alive or extending the cleanup budget.
    remaining = set(verified_live) | set(unresolved)
    child.poll()
    if child.poll() is None:
        remaining.add(child.pid)
    passed = (not remaining and not errors and not skipped and not unresolved
              and not snapshot['reused'] and not snapshot['unknown_overflow']
              and report['table_error'] is None)
    return {'status': 'passed' if passed else 'failed',
        'stage_was_running': stage_was_running, 'initial_live_pids': initial_live,
        'live_descendants_at_cleanup': [pid for pid in initial_live if pid != child.pid],
        'root_pid': child.pid, 'root_ownership': snapshot['root_ownership'],
        'returncode': child.returncode,
        'owned': list(snapshot['owned'].values()), 'retired': snapshot['retired'],
        'retired_overflow': snapshot['retired_overflow'], 'signals': signals,
        'handle_signals': handle_signals,
        'identity_changed_skipped': sorted(set(skipped) | set(report['reused'])),
        'reused_pids': sorted(snapshot['reused']),
        'unresolved_unknown_pids': unresolved,
        'verified_live_pids': sorted(verified_live),
        'unknown_overflow': snapshot['unknown_overflow'],
        'unknown_observations': [dict(entry) for entry in snapshot['unknown'].values()],
        'remaining_pids': sorted(remaining), 'errors': errors,
        'table_error': report['table_error'], 'events': snapshot['events'],
        'event_overflow': snapshot['event_overflow'],
        'elapsed_seconds': time.monotonic() - started}


def _chained_termination(exc):
    """Find a StageTerminated in a bounded exception cause/context chain."""
    seen, current = set(), exc
    for _step in range(50):
        if current is None or id(current) in seen:
            return None
        if isinstance(current, StageTerminated):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == '--':
        arguments.pop(0)
    if not arguments or re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', arguments[0]) is None:
        raise SystemExit('Usage: python -m pokemon_teacher_runtime.lifecycle -- package.module [arguments]')
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
    except BaseException as exc:
        # F35 defect: a cleanup error raised while unwinding StageTerminated
        # replaced it, losing the cooperative exit code. Keep the cooperative
        # code and print the whole chain rather than discarding either one.
        stage = _chained_termination(exc)
        if stage is None:
            raise
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        return 128 + stage.signum
    finally:
        sys.argv = old_argv
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
