"""Discarded actual-worker runtime qualification; no scientific training/data reuse."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'artifacts/reviews/f35-teacher-assisted/engineering-preflight-v1/teacher-p2'
CYCLES = 64


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('x') as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')
    os.link(temporary, path)  # Atomic publication, exclusive destination.
    temporary.unlink()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(kind, output):
    from .transport import JsonWorker
    output.mkdir(exist_ok=False, parents=True)
    frame = json.loads((FIXTURE / 'transcript.jsonl').read_text().splitlines()[0])
    envelope = frame['external']
    started = time.monotonic()
    records = []
    if kind == 'replay':
        from .stage import install_overlay
        binding = install_overlay()
        import numpy as np
        from pokemon_teacher_assisted import external, verification
        assert external.JsonWorker is verification.JsonWorker is JsonWorker
        record = json.loads((FIXTURE / 'result.json').read_text())
        transcript = [json.loads(line) for line in (FIXTURE / 'transcript.jsonl').read_text().splitlines()]
        metadata = json.loads((FIXTURE / 'labels.json').read_text())
        with np.load(FIXTURE / 'demonstrations.npz', allow_pickle=False) as arrays:
            with JsonWorker(['node', 'packages/teacher-assisted/external-worker.mjs'], output / 'replay.stderr') as worker:
                result = verification.replay_external(record, transcript, worker,
                    label_arrays=arrays, label_metadata=metadata)
        save(output / 'report.json', {'status': 'passed', 'kind': kind, 'binding': binding,
            'replay': result, 'elapsed_seconds': time.monotonic() - started,
            'scope': 'One existing discarded engineering game replayed with recorded actions; no teacher search or new scientific game.'})
        return
    count = CYCLES if kind == 'churn' else 1
    for i in range(count):
        folder = output / f'{i:03d}'
        folder.mkdir()
        node = teacher = None
        cycle_error = None
        record = {'cycle': i, 'kind': kind, 'started_utc': datetime.now(timezone.utc).isoformat(),
                  'choice_completed': False, 'scientific_game': False, 'accepted_label': False}
        try:
            # Nested construction retains Node cleanup if teacher startup fails.
            with JsonWorker(['node', 'packages/teacher-assisted/external-worker.mjs'], folder / 'node.stderr') as node:
                with JsonWorker([str(ROOT / '.venv-f34-external/bin/python'), '-m', 'pokemon_teacher_assisted.teacher'],
                                folder / 'teacher.stderr') as teacher:
                    pids = {'node': node.process.pid, 'teacher': teacher.process.pid}
                    record.update(pids)
                    save(folder / 'launch.json', {**record})
                    if kind == 'startup':
                        save(output / 'ready.json', {'kind': kind, **pids})
                        time.sleep(15)
                        raise RuntimeError('Expected supervised startup interruption did not arrive')
                    node.request('hello')
                    teacher.request('start', side='p2', search_time_ms=100, seed=1027283600 + i)
                    if kind == 'search':
                        save(output / 'ready.json', {'kind': kind, **pids})
                    choice = teacher.request('choose', input=envelope)
                    record.update(choice_completed=True, choice=choice)
                    save(folder / 'choice.json', {'choice': choice})
                    if kind == 'search':
                        save(output / 'search-returned.json', {'choice': choice})
                        time.sleep(15)
                        raise RuntimeError('Expected supervised search interruption did not arrive')
                    player = frame['frame']['players']['p2']
                    assert choice.get('waiting') is not True
                    assert choice['request_id'] == player['request_id']
                    assert type(choice['action']) is int and player['mask'][choice['action']]
                    teacher.request('close')
                    record.update(request_id=choice['request_id'], action=choice['action'], legal=True)
            assert node.process.poll() == teacher.process.poll() == 0, 'Normal workers must exit cleanly on EOF'
            records.append(record)
        except BaseException as exc:
            cycle_error = exc
            record['error'] = repr(exc)
            raise
        finally:
            for label, worker in [('node', node), ('teacher', teacher)]:
                if worker is not None:
                    record[label + '_exit'] = worker.process.poll()
                    record[label + '_cleanup_errors'] = worker.cleanup_errors
            try:
                save(folder / 'record.json', record)
            except BaseException as exc:
                if cycle_error is None:
                    raise
                cycle_error.add_note('Cycle evidence write also failed: ' + repr(exc))
    save(output / 'report.json', {'status': 'passed', 'kind': kind, 'cycles': len(records),
        'native_search_calls': len(records), 'search_time_ms': 100, 'records': records,
        'elapsed_seconds': time.monotonic() - started, 'scientific_games': 0, 'accepted_labels': 0,
        'scope': 'Fresh worker/search startup and shutdown on a discarded engineering observation. Choices were never executed in a battle or admitted as labels.'})



def cleanup_stage(process, snapshot):
    from . import lifecycle
    try:
        return lifecycle.terminate_tree(process, snapshot=snapshot,
            grace_seconds=1., kill_wait_seconds=3., poll_seconds=.02)
    except BaseException as exc:
        errors = [repr(exc)]
        for method in ('terminate', 'kill'):
            if process.poll() is not None:
                break
            try:
                getattr(process, method)()
                process.wait(timeout=2.)
            except ProcessLookupError:
                pass
            except BaseException as secondary:
                errors.append(method + ': ' + repr(secondary))
        return {'status': 'failed', 'errors': errors, 'cleanup_exception': True,
                'remaining_pids': [process.pid] if process.poll() is None else [],
                'descendant_cleanup_unresolved': True, 'returncode': process.poll()}


def active_search_children(snapshot, teacher_pid):
    from . import lifecycle
    result = []
    for identity in snapshot['owned'].values():
        if identity['ppid'] != teacher_pid or identity.get('zombie'):
            continue
        observed = subprocess.run(['/bin/ps', '-p', str(identity['pid']), '-o', 'command='],
            capture_output=True, text=True, timeout=.2,
            env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
        command = observed.stdout.strip()
        if observed.returncode != 0 or 'resource_tracker' in command:
            continue
        if 'spawn_main' not in command or '--multiprocessing-fork' not in command:
            continue
        current = lifecycle.process_identity(identity['pid'])
        if current is not None and all(current.get(k) == identity[k] for k in ('pid', 'uid', 'birth')):
            result.append({**identity, 'observed_command': command})
    return result


def supervise(kind, output):
    from . import lifecycle
    folder = output / kind
    command = [str(ROOT / '.venv/bin/python'), '-m', 'pokemon_teacher_runtime.lifecycle', '--',
               'pokemon_teacher_runtime.qualification', '--child', kind, '--output', str(folder)]
    begun = time.monotonic()
    snapshot = None
    cleanup = None
    error = None
    interrupted = False
    search_descendants = []
    with (output / f'{kind}.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'python'),
            'PYTHONDONTWRITEBYTECODE': '1', 'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            save(output / f'{kind}-launch.json', {'command': command, 'pid': process.pid,
                'started_utc': datetime.now(timezone.utc).isoformat()})
            while process.poll() is None:
                snapshot = lifecycle.capture_tree(process, snapshot)
                ready_path = folder / 'ready.json'
                if kind in ('startup', 'search') and ready_path.exists():
                    ready = json.loads(ready_path.read_text())
                    if kind == 'search':
                        search_descendants = active_search_children(snapshot, ready['teacher'])
                    if kind == 'startup' or search_descendants:
                        if kind == 'search' and (folder / 'search-returned.json').exists():
                            raise RuntimeError('Search ended before interruption; not qualifying active-search cleanup')
                        interrupted = True
                        cleanup = cleanup_stage(process, snapshot)
                        break
                if time.monotonic() - begun > (180 if kind == 'churn' else 30):
                    raise TimeoutError('Bounded runtime qualification timed out: ' + kind)
                time.sleep(.02)
        except BaseException as exc:
            error = repr(exc)
        finally:
            if cleanup is None:
                cleanup = cleanup_stage(process, snapshot)
            process.poll()
    record = {'kind': kind, 'exit_code': process.returncode, 'error': error,
        'interrupted': interrupted, 'observed_search_descendants': search_descendants,
        'cleanup': cleanup, 'elapsed_seconds': time.monotonic() - begun}
    save(output / f'{kind}-exit.json', record)
    assert error is None, record
    assert cleanup['status'] == 'passed' and cleanup['remaining_pids'] == [], record
    if kind in ('startup', 'search'):
        assert interrupted and process.returncode == 143, record
        assert 'PermissionError' not in (output / f'{kind}.log').read_text(), 'Termination was masked'
    else:
        assert process.returncode == 0, record
        assert json.loads((folder / 'report.json').read_text())['status'] == 'passed'
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', choices=['churn', 'startup', 'search', 'replay'])
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.child:
        child(args.child, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    stages = []
    try:
        for kind in ('churn', 'startup', 'search', 'replay'):
            stages.append(supervise(kind, args.output))
        evidence = {str(p.relative_to(ROOT)): sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
        sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'python/pokemon_teacher_runtime').glob('*.py'))}
        fixture = {str(p.relative_to(ROOT)): sha(p) for p in sorted(FIXTURE.iterdir()) if p.is_file()}
        save(args.output / 'report.json', {'schema_version': 'f36-actual-worker-qualification-v1', 'status': 'passed',
            'stages': stages, 'cycles': CYCLES, 'scientific_games': 0, 'accepted_labels': 0,
            'completed_native_search_calls': CYCLES, 'interrupted_choose_attempts': 1, 'existing_engineering_game_replays': 1,
            'elapsed_seconds': time.monotonic() - begun, 'source_files': sources,
            'fixture_files': fixture, 'evidence_files': evidence,
            'scope': 'Bounded actual Node/teacher/search churn, startup/search termination and one original engineering game replay. No scientific acquisition, learner update, evaluation panel or new policy result.'})
    except BaseException as exc:
        save(args.output / 'failure.json', {'status': 'failed', 'error': repr(exc), 'stages': stages,
            'elapsed_seconds': time.monotonic() - begun})
        raise


if __name__ == '__main__':
    main()
