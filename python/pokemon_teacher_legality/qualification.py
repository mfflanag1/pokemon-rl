"""Fixed discarded engineering qualification for the separate F37 teacher repair."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

from pokemon_teacher_runtime.qualification import cleanup_stage, save, sha

ROOT = Path(__file__).resolve().parents[2]
FAILED = ROOT / 'artifacts/runs/f36-teacher-data-v1/train/8209/058-0'
REPLAY = ROOT / 'artifacts/reviews/f35-teacher-assisted/engineering-preflight-v1/teacher-p2'
FAILED_HASHES = {'result.json': '336ffb178e6d7126b527955c99b64ac82a9d8dbf49be7b893928ec3ea72f9b7e',
    'transcript.jsonl': 'c53334b23f8c4db4c4a5f731ceac17c9a89cdb02516e9f7a72411c511547fc36'}
STAGES = ('historical-request', 'engineered-cases', 'discarded-games', 'recorded-replay')
LIMITS = dict(zip(STAGES, (90., 30., 240., 45.)))
DEADLINE = datetime(2026, 10, 28, 7, 36, 8, tzinfo=timezone.utc).timestamp()
ENGINEERING_GAMES = (
    dict(run_seed=370037, episode_index=0, swap=False, external_side='p1', fp_seed=370101),
    dict(run_seed=370037, episode_index=1, swap=True, external_side='p2', fp_seed=370102),
)
ENGINEERED_CASES = ('no-tera', 'trapped', 'choice-lock', 'forced-switch', 'wait',
    'stale-id', 'malformed-mask', 'empty-legal-root', 'absent-move')


def read(path):
    return json.loads(Path(path).read_text())


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + '\n')
    stream.flush()


@contextmanager
def no_native_search(native):
    """Fail and count any accidental native-search entry during synthetic cases."""
    with patch.object(native, 'get_result_from_mcts',
            side_effect=AssertionError('Deterministic qualification attempted native search')) as calls:
        yield calls
        calls.assert_not_called()


def check_choice(choice, player):
    if not any(player['mask']):
        if choice != {'waiting': True}:
            raise ValueError('Waiting engineering request produced a choice')
        return
    action = choice.get('action')
    if (type(action) is not int or not 0 <= action < 14 or not player['mask'][action]
            or choice.get('request_id') != player['request_id']):
        raise ValueError('Engineering choice violates authoritative request')
    selection = choice['selection']
    expected_kind = 'forced-request' if sum(player['mask']) == 1 else 'native-legal-root'
    if (selection['legal_mask'] != player['mask'] or selection['request_id'] != player['request_id']
            or selection['kind'] != expected_kind or selection['search_calls'] != int(expected_kind == 'native-legal-root')):
        raise ValueError('Engineering choice selection provenance mismatch')


def historical_sequence(teacher, transcript, record, journal, last_move, search_count=None):
    """Fresh choices are discarded; only recorded accepted actions rebuild history."""
    if len(transcript) != 12 or len(record['steps']) != 11 or record['outcome'] is not None:
        raise ValueError('Unexpected preserved failure prefix')
    side = record['external_side']
    rows = []
    for index, bundle in enumerate(transcript):
        player, envelope = bundle['frame']['players'][side], bundle['external']
        before = search_count() if search_count is not None else None
        choice = teacher.choose(envelope, player['mask'], player['request_id'])
        row = {'index': index, 'request_id': player['request_id'], 'choice': choice,
               'fresh_choice_executed': False, 'accepted_label': False}
        append(journal, {'event': 'raw-choice', **row})  # Preserve even a rejected guard result.
        check_choice(choice, player)
        if search_count is not None:
            row['observed_native_search_calls'] = search_count() - before
            if row['observed_native_search_calls'] != int(sum(player['mask']) > 1):
                raise ValueError('Per-request observed native search count mismatch')
        if 'action' in choice:
            teacher.feedback(choice['request_id'], False)
        if index < len(record['steps']):
            step = record['steps'][index]
            submissions = [s for s in step['submitted'] if s['side'] == side]
            if bool(submissions) != bool(any(player['mask'])) or len(submissions) > 1:
                raise ValueError('Historical teacher submission inventory')
            if submissions:
                submitted = submissions[0]
                if (submitted['request_id'] != player['request_id']
                        or submitted['action'] != step['external']['action']
                        or type(submitted['accepted']) is not bool):
                    raise ValueError('Historical accepted feedback coordinates')
                if submitted['accepted']:
                    battle = teacher.adapter.battle
                    decision = step['external']['decision'].removesuffix('-tera').removesuffix('-mega')
                    battle.user.last_selected_move = last_move(battle.user.active.name, decision, battle.turn)
                row['historical_feedback'] = submitted
                row['historical_decision'] = step['external'].get('decision')
        append(journal, {'event': 'validated-discarded-choice', **row})
        rows.append(row)
    final = rows[-1]
    if (final['request_id'] != 12 or final['choice']['action'] != 0
            or final['choice']['selection']['kind'] != 'forced-request'
            or final['choice']['selection']['search_calls'] != 0):
        raise ValueError('Locked-Outrage request was not resolved without search')
    native = sum(row['choice'].get('selection', {}).get('kind') == 'native-legal-root' for row in rows)
    if native < 2:
        raise ValueError('Historical qualification requires ordinary native choices before the forced request')
    return {'requests': len(rows), 'native_choices': native, 'final_request': final,
            'scope': 'Original envelopes in order; fresh alternatives discarded, recorded accepted bookkeeping restored.'}


def fixture():
    for name, expected in FAILED_HASHES.items():
        if sha(FAILED / name) != expected:
            raise ValueError('Preserved F36 failure fixture changed: ' + name)
    return read(FAILED / 'result.json'), [json.loads(line) for line in (FAILED / 'transcript.jsonl').read_text().splitlines()]


def engineered_cases(output, initial):
    """Actual parsed adapter and selector, deterministic roots; no native search/game."""
    from .teacher import AcceptedTeacher, LegalFoulPlayAdapter
    from .selection import LegalSelectionError
    import fp.search.main as native
    base = LegalFoulPlayAdapter('p2', 100, 370201)
    base.ingest(initial['external'])
    original_mask = initial['frame']['players']['p2']['mask']
    move = base.battle.request_json['active'][0]['moves'][0]['id']
    reserve = base.battle.user.reserve[0]
    switch_action = reserve.index + 7
    records = []
    with (output / 'cases.jsonl').open('x') as journal:
        for name in ENGINEERED_CASES:
            adapter = deepcopy(base)
            mask = list(original_mask)
            request = adapter.battle.request_json
            expected_error = name in ('stale-id', 'malformed-mask', 'empty-legal-root')
            roots = [(move + '-tera', 90), (move, 10)]
            expected_action, calls = 0, []
            if name in ('no-tera', 'empty-legal-root'):
                request['active'][0].pop('canTerastallize', None)
                mask[4:8] = [False] * 4
                if name == 'empty-legal-root':
                    roots = [(move + '-tera', 100)]
            elif name == 'trapped':
                request['active'][0]['trapped'] = True
                mask[8:] = [False] * 6
                roots = [('switch ' + reserve.name, 90), (move, 10)]
            elif name == 'choice-lock':
                request['active'][0]['moves'] = [request['active'][0]['moves'][0]]
                mask = [True] + [False] * 13
            elif name == 'forced-switch':
                request.pop('active', None)
                request['forceSwitch'] = [True]
                mask = [i == switch_action for i in range(14)]
                expected_action = switch_action
            elif name == 'absent-move':
                if any(candidate['id'] == 'splash' for candidate in request['active'][0]['moves']):
                    raise AssertionError('Absent-move fixture unexpectedly advertises splash')
                roots = [('splash', 90), (move, 10)]

            def synthetic_search(battle):
                calls.append(True)
                result = SimpleNamespace(total_visits=sum(n for _, n in roots),
                    side_one=[SimpleNamespace(move_choice=choice, visits=n) for choice, n in roots])
                return native.select_move_from_mcts_results([(result, 1., 0)])

            row = {'case': name, 'expected_error': expected_error,
                   'synthetic_root_search_calls': 0, 'scientific_game': False}
            with no_native_search(native) as native_calls, patch.object(native, 'find_best_move', synthetic_search):
                try:
                    if name in ('wait', 'stale-id', 'malformed-mask'):
                        teacher = AcceptedTeacher('p2', adapter=adapter)
                        envelope = dict(initial['external'], protocol=[], pending=name != 'wait')
                        if name == 'wait':
                            result = teacher.choose(envelope, [False] * 14, envelope['request_id'])
                            if result != {'waiting': True}:
                                raise AssertionError('Wait produced an action')
                        else:
                            result = teacher.choose(envelope, [0] * 14 if name == 'malformed-mask' else mask,
                                envelope['request_id'] + int(name == 'stale-id'))
                    else:
                        adapter.set_request_legality(mask, adapter.request_count)
                        result = adapter.choose()
                        check_choice(result, {'mask': mask, 'request_id': adapter.request_count})
                        if result['action'] != expected_action:
                            raise AssertionError('Engineered authoritative choice mismatch')
                        if name == 'absent-move':
                            audit = result['selection']['audit']
                            absent = [candidate for candidate in audit['candidates'] if candidate['choice'] == 'splash']
                            if (len(absent) != 1 or absent[0]['reason'] != 'unmappable_choice'
                                    or absent[0]['allowed'] is not False or absent[0]['action'] is not None
                                    or absent[0]['mass'] != .9 or audit['highest_legal_mass'] != .1
                                    or audit['decision'] != move):
                                raise AssertionError('Absent move was not excluded before the legal threshold')
                    row['result'] = result
                    if expected_error:
                        raise AssertionError('Invalid engineered context was accepted')
                except ValueError as exc:
                    row['error'] = repr(exc)
                    if not expected_error:
                        raise
                    if name == 'empty-legal-root' and not isinstance(exc, LegalSelectionError):
                        raise
                    if name == 'stale-id' and str(exc) != 'Authoritative mask/request differs from teacher envelope':
                        raise
                    if name == 'malformed-mask' and str(exc) != 'Authoritative legal mask must contain exactly 14 booleans':
                        raise
                finally:
                    row['synthetic_root_search_calls'] = len(calls)
                    row['native_search_calls'] = native_calls.call_count
                    append(journal, row)
            if name in ('choice-lock', 'forced-switch', 'wait', 'stale-id', 'malformed-mask') and calls:
                raise AssertionError('Forced/wait/invalid request invoked search')
            if name in ('no-tera', 'trapped', 'empty-legal-root', 'absent-move') and len(calls) != 1:
                raise AssertionError('Engineered selector was not exercised exactly once')
            records.append(row)
    return {'cases': records, 'case_count': len(records),
            'native_search_calls': sum(row['native_search_calls'] for row in records),
            'scope': 'Deterministic synthetic root visits through the actual parsed adapter and legal selector.'}


def child(kind, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if kind in ('historical-request', 'engineered-cases'):
            record, transcript = fixture()
            if kind == 'engineered-cases':
                result = engineered_cases(output, transcript[0])
            else:
                from .teacher import AcceptedTeacher
                from fp.battle.state import LastUsedMove
                import fp.search.main as native
                original = native.find_best_move
                calls = []

                def counted(battle):
                    calls.append(True)
                    return original(battle)

                teacher = AcceptedTeacher('p2', 100, record['fp_seed'])
                with patch.object(native, 'find_best_move', counted), (output / 'choices.jsonl').open('x') as journal:
                    result = historical_sequence(teacher, transcript, record, journal, LastUsedMove, lambda: len(calls))
                if len(calls) != result['native_choices']:
                    raise AssertionError('Observed native calls disagree with declared selections')
                result['observed_native_search_calls'] = len(calls)
                result['fixture_hashes'] = FAILED_HASHES
        else:
            from .stage import install_overlay
            binding = install_overlay()
            from pokemon_teacher_runtime.transport import JsonWorker
            if kind == 'discarded-games':
                from .external import run_game
                games = []
                for index, coordinates in enumerate(ENGINEERING_GAMES):
                    folder = output / str(index)
                    record = run_game(folder, **coordinates, split='development', learner=None,
                        purpose='F37 discarded engineering legality qualification; never training or evaluation',
                        capture_demonstrations=False, max_turns=240, search_time_ms=100,
                        deadline_epoch=min(DEADLINE, time.time() + LIMITS[kind]))
                    if (record['status'] != 'completed' or record['outcome']['kind'] not in ('terminal', 'administrative_cutoff')
                            or record['capture_demonstrations'] is not False
                            or any((folder / n).exists() for n in ('labels.json', 'demonstrations.npz', 'learner.json'))):
                        raise ValueError('Discarded engineering game contract')
                    from . import RESPONSE_JOURNAL
                    responses = [json.loads(line) for line in (folder / RESPONSE_JOURNAL).read_text().splitlines()]
                    transcript = [json.loads(line) for line in (folder / 'transcript.jsonl').read_text().splitlines()]
                    if len(responses) != len(record['steps']):
                        raise ValueError('Engineering raw response inventory')
                    for step, response, bundle in zip(record['steps'], responses, transcript):
                        player = bundle['frame']['players'][coordinates['external_side']]
                        if (response['step_index'] != step['index'] or response['status'] != 'response'
                                or response['raw_response'] != step['external']
                                or response['request_id'] != player['request_id']
                                or response['legal_mask'] != player['mask']
                                or response['external_input_sha256'] != step['external_input_sha256']):
                            raise ValueError('Engineering raw reply/step/request mismatch')
                        check_choice(response['raw_response'], player)
                    games.append({'coordinates': coordinates, 'outcome': record['outcome'],
                        'decisions': record['decisions'], 'seconds': record['seconds'],
                        'raw_responses_checked': len(responses), 'teacher_policy_counts': record['teacher_policy_counts']})
                result = {'games': games, 'discarded_engineering_games': len(games), 'binding': binding}
            else:
                from pokemon_teacher_assisted.verification import replay_external
                record = read(REPLAY / 'result.json')
                transcript = [json.loads(line) for line in (REPLAY / 'transcript.jsonl').read_text().splitlines()]
                with JsonWorker(['node', 'packages/teacher-assisted/external-worker.mjs'], output / 'replay.stderr') as worker:
                    replay = replay_external(record, transcript, worker)
                if worker.process.returncode != 0:
                    raise ValueError('Recorded replay worker did not exit cleanly')
                result = {'replay': replay, 'existing_engineering_game_replays': 1, 'binding': binding,
                    'native_search_calls': 0, 'scope': 'Original recorded actions only; no new labels or search.'}
        save(output / 'report.json', {'status': 'passed', 'kind': kind, 'result': result,
            'scientific_games': 0, 'accepted_labels': 0, 'elapsed_seconds': time.monotonic() - started})
    except BaseException as exc:
        try:
            save(output / 'failure.json', {'status': 'failed', 'kind': kind, 'error': repr(exc),
                'elapsed_seconds': time.monotonic() - started})
        except BaseException as secondary:
            exc.add_note('Failure evidence write failed: ' + repr(secondary))
        raise


def supervise(kind, output):
    from pokemon_teacher_runtime import lifecycle
    runtime = '.venv-f34-external/bin/python' if kind in STAGES[:2] else '.venv/bin/python'
    folder = output / kind
    command = [str(ROOT / runtime), '-m', 'pokemon_teacher_runtime.lifecycle', '--',
        'pokemon_teacher_legality.qualification', '--child', kind, '--output', str(folder)]
    started, snapshot, error, process = time.monotonic(), None, None, None
    cleanup = {'status': 'not-started'}
    with (output / (kind + '.log')).open('xb') as log:
        try:
            if time.time() >= DEADLINE:
                raise TimeoutError('Original goal deadline reached before qualification stage')
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ,
                'PYTHONPATH': str(ROOT / 'python'), 'PYTHONDONTWRITEBYTECODE': '1',
                'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(output / (kind + '-launch.json'), {'pid': process.pid, 'command': command})
            while process.poll() is None:
                snapshot = lifecycle.capture_tree(process, snapshot)
                if time.monotonic() - started >= LIMITS[kind] or time.time() >= DEADLINE:
                    raise TimeoutError('Bounded F37 qualification stage expired: ' + kind)
                time.sleep(.03)
        except BaseException as exc:
            error = repr(exc)
        finally:
            if process is not None:
                cleanup = cleanup_stage(process, snapshot)
    terminal = {'kind': kind, 'exit_code': process.poll() if process else None, 'error': error,
        'cleanup': cleanup, 'elapsed_seconds': time.monotonic() - started}
    save(output / (kind + '-exit.json'), terminal)
    if (error is not None or terminal['exit_code'] != 0 or cleanup['status'] != 'passed'
            or any(cleanup.get(key) for key in ('errors', 'remaining_pids', 'unresolved_unknown_pids',
                'reused_pids', 'unknown_overflow', 'table_error', 'signals', 'handle_signals'))):
        raise RuntimeError('F37 qualification supervision failed: ' + repr(terminal))
    if read(folder / 'report.json').get('status') != 'passed':
        raise ValueError('Missing valid qualification child report')
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--child', choices=STAGES)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.child:
        child(args.child, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    started, stages = time.monotonic(), []
    try:
        inputs = {str(p.relative_to(ROOT)): sha(p) for folder in (FAILED, REPLAY) for p in folder.iterdir() if p.is_file()}
        sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'python/pokemon_teacher_legality').glob('*.py')}
        for kind in STAGES:
            stages.append(supervise(kind, output))
        if any(sha(ROOT / name) != expected for name, expected in {**inputs, **sources}.items()):
            raise ValueError('Qualification input/source drift during execution')
        evidence = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
        save(output / 'report.json', {'schema_version': 'f37-teacher-legality-qualification-v1', 'status': 'passed',
            'stages': stages, 'fixed_engineered_cases': len(ENGINEERED_CASES), 'discarded_engineering_games': 2,
            'historical_failure_prefixes': 1, 'existing_engineering_game_replays': 1,
            'scientific_games': 0, 'accepted_labels': 0, 'source_files': sources,
            'fixture_files': inputs, 'evidence_files': evidence, 'elapsed_seconds': time.monotonic() - started,
            'scope': 'Request legality and bounded engineering only; no learning result or scientific data.'})
    except BaseException as exc:
        try:
            save(output / 'failure.json', {'status': 'failed', 'error': repr(exc), 'stages': stages,
                'elapsed_seconds': time.monotonic() - started})
        except BaseException as secondary:
            exc.add_note('Qualification failure evidence write failed: ' + repr(secondary))
        raise


if __name__ == '__main__':
    main()
