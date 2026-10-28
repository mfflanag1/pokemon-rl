#!/usr/bin/env python3
"""Discarded engineering reproduction of the preserved F37 revival boundary.

No original source/evidence is edited. Historical native search is never rerun:
recorded accepted decisions restore only the adapter's last-selected bookkeeping.
One optional current native search is bounded and discarded at the last boundary.
A separate fresh Showdown session replays only the original 19 action submissions.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
FIXTURE = ROOT / 'artifacts/runs/f37-teacher-data-v1/train/9311/053-1'
ALLOWED_OUTPUT = ROOT / 'artifacts/reviews/f38-teacher-compatibility/reproduction'
DEADLINE = datetime.fromisoformat('2026-10-28T16:03:40+00:00').timestamp()
STAGES = ('inspect', 'native-boundary', 'showdown-prefix')
LIMITS = {'inspect': 30., 'native-boundary': 30., 'showdown-prefix': 45.}
SOURCE_PATHS = (
    'scripts/diagnose-teacher-compatibility.py',
    'python/pokemon_external_benchmark/foul_play.py',
    'python/pokemon_teacher_assisted/teacher.py',
    'python/pokemon_teacher_assisted/verification.py',
    'python/pokemon_teacher_legality/teacher.py',
    'python/pokemon_teacher_legality/selection.py',
    'python/pokemon_teacher_legality/qualification.py',
    'python/pokemon_teacher_runtime/transport.py',
    'python/pokemon_teacher_runtime/lifecycle.py',
    'python/pokemon_teacher_runtime/qualification.py',
    'packages/teacher-assisted/external-worker.mjs',
    'packages/external-benchmark/session.mjs',
    'packages/full-showdown/actions.mjs',
    'packages/external-benchmark/pin.json',
    'artifacts/reviews/f37-teacher-legality-failure/diagnosis-reviewed.md',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def append(stream, value):
    stream.write(canonical(value) + '\n')
    stream.flush()


def same(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError('F38 exact reproduction mismatch: ' + label)


def fixture():
    record = read(FIXTURE / 'result.json')
    transcript = [json.loads(line) for line in (FIXTURE / 'transcript.jsonl').read_text().splitlines()]
    if (record['status'] != 'failed' or record['outcome'] is not None or len(record['steps']) != 19
            or len(transcript) != 20 or record['external_side'] != 'p1'
            or transcript[-1]['external']['request_id'] != 20):
        raise ValueError('Unexpected historical failed-prefix coordinates')
    return record, transcript


def pokemon_snapshot(pokemon):
    return {key: getattr(pokemon, key) for key in ('name', 'index', 'hp', 'max_hp', 'reviving')}


def battle_snapshot(battle):
    return {'request_id': battle.rqid, 'turn': battle.turn, 'force_switch': battle.force_switch,
            'wait': battle.wait, 'own': [pokemon_snapshot(p) for p in [battle.user.active, *battle.user.reserve]],
            'last_selected_move': list(battle.user.last_selected_move),
            'last_used_move': list(battle.user.last_used_move)}


def reconstruct(output):
    from pokemon_teacher_legality.teacher import LegalFoulPlayAdapter
    from fp.battle.state import LastUsedMove
    import fp.search.main as native
    record, transcript = fixture()
    adapter = LegalFoulPlayAdapter(record['external_side'], record['search_time_ms'], record['fp_seed'])
    # This fails rather than silently running native search while ingesting history.
    with patch.object(native, 'find_best_move', side_effect=AssertionError('Historical native search forbidden')) as searches:
        with (output / 'reconstruction.jsonl').open('x') as stream:
            for index, bundle in enumerate(transcript):
                player, envelope = bundle['frame']['players'][record['external_side']], bundle['external']
                adapter.set_request_legality(player['mask'], player['request_id'])
                adapter.ingest(envelope)
                row = {'index': index, 'request_id': player['request_id'], 'phase': player['phase'],
                       'pending': envelope['pending'], 'before_bookkeeping': battle_snapshot(adapter.battle)}
                if index < len(record['steps']):
                    step = record['steps'][index]
                    submissions = [s for s in step['submitted'] if s['side'] == record['external_side']]
                    same(bool(submissions), envelope['pending'], 'historical pending/submission')
                    if len(submissions) > 1:
                        raise ValueError('Duplicate historical teacher submission')
                    if submissions:
                        submitted = submissions[0]
                        same(submitted['request_id'], player['request_id'], 'historical request ID')
                        same(submitted['action'], step['external']['action'], 'historical teacher action')
                        if type(submitted['accepted']) is not bool or not player['mask'][submitted['action']]:
                            raise ValueError('Historical accepted/action bookkeeping invalid')
                        # choose sets this even if rejected; feedback only rolls LastUsedMove back.
                        adapter.last_decided_request = player['request_id']
                        if submitted['accepted']:
                            decision = step['external']['decision'].removesuffix('-tera').removesuffix('-mega')
                            adapter.battle.user.last_selected_move = LastUsedMove(adapter.battle.user.active.name, decision, adapter.battle.turn)
                        row.update(recorded_submission=submitted, recorded_decision=step['external']['decision'])
                    else:
                        same(step['external'], {'waiting': True}, 'historical wait response')
                row['after_bookkeeping'] = battle_snapshot(adapter.battle)
                append(stream, row)
        searches.assert_not_called()
    return adapter, record, transcript


def engine_snapshot(state):
    side = state.side_one
    return {'force_switch': side.force_switch, 'active_index': side.active_index,
            'last_used_move': side.last_used_move,
            'own': [{'id': p.id, 'hp': p.hp, 'maxhp': p.maxhp} for p in side.pokemon],
            'serialized': state.to_string()}


def inspect(output):
    import poke_engine
    from pokemon_external_benchmark.foul_play import decision_to_action
    from fp.modes.base import format_decision
    from fp.search.poke_engine_helpers import battle_to_poke_engine_state
    adapter, record, transcript = reconstruct(output)
    battle = deepcopy(adapter.battle)
    before = battle_snapshot(battle)
    battle.user.update_from_request_json(battle.request_json)
    after = battle_snapshot(battle)
    request = battle.request_json
    player = transcript[-1]['frame']['players'][record['external_side']]
    write(output / 'last-request.json', request)
    write(output / 'last-player.json', player)
    write(output / 'adapter-state.json', {'before_refresh': before, 'after_refresh': after})
    slots = []
    for slot, raw in enumerate(request['side']['pokemon'], 1):
        matches = [p for p in [battle.user.active, *battle.user.reserve] if p.index == slot]
        if len(matches) != 1:
            raise ValueError('Own request/adapter slot identity mismatch')
        p = matches[0]
        # The active slot has no valid switch command; still check its numeric mapping.
        wire = format_decision(battle, 'switch ' + p.name) if not raw['active'] else ['/switch ' + str(slot)]
        action = decision_to_action(wire, request)
        same(action, slot + 7, 'switch action mapping')
        same(p.hp == 0, raw['condition'].startswith('0 fnt'), 'request/adapter fainted state')
        same(p.reviving, raw.get('reviving', False), 'request/adapter reviving state')
        slots.append({'slot': slot, 'request_ident': raw['ident'], 'request_condition': raw['condition'],
                      'request_active': raw['active'], 'request_reviving': raw.get('reviving', False),
                      'adapter': pokemon_snapshot(p), 'decision': 'switch ' + p.name, 'wire': wire,
                      'action': action, 'formatter_checked': not raw['active'], 'authoritative_legal': player['mask'][action]})
    state = battle_to_poke_engine_state(deepcopy(battle))
    write(output / 'engine-state.json', engine_snapshot(state))
    no_revival = deepcopy(battle)
    for p in [no_revival.user.active, *no_revival.user.reserve]:
        p.reviving = False
    for p in no_revival.request_json['side']['pokemon']:
        p['reviving'] = False
    alternative = battle_to_poke_engine_state(no_revival)
    write(output / 'engine-state-revival-flag-cleared.json', engine_snapshot(alternative))
    equivalent = state.to_string() == alternative.to_string()
    same(equivalent, True, 'reviving flag absent from conversion')
    api = {name: [key for key in dir(getattr(poke_engine, name)) if not key.startswith('_')]
           for name in ('State', 'Side', 'Pokemon')}
    api['module_exports'] = [key for key in dir(poke_engine) if not key.startswith('_')]
    write(output / 'engine-api.json', api)
    write(output / 'slot-mapping.json', slots)
    legal = [i for i, allowed in enumerate(player['mask']) if allowed]
    same(legal, [10, 11, 12], 'original revival legal actions')
    same([row['action'] for row in slots if row['adapter']['hp'] == 0], legal, 'fainted target mask')
    return {'requests_reconstructed': len(transcript), 'recorded_steps': len(record['steps']),
            'historical_native_search_calls': 0, 'native_search_calls': 0,
            'legal_actions': legal, 'all_six_request_identity_and_numeric_slots_match': True, 'five_reserve_formatter_slots_match': True,
            'reviving_true_in_request_and_adapter': battle.user.active.reviving,
            'engine_force_switch': state.side_one.force_switch,
            'serialized_state_identical_after_clearing_reviving_only': equivalent,
            'deterministic_get_all_options_available': hasattr(poke_engine, 'get_all_options') or hasattr(state, 'get_all_options'),
            'scope': 'Current request/HP/slot reconstruction; no sampled worlds, search, simulation or labels.'}


def native_boundary(output):
    adapter, record, transcript = reconstruct(output)
    import fp.search.main as native
    original_conversion = native.battle_to_poke_engine_state
    original_search = native.find_best_move
    calls, worlds = [], []
    def capture_conversion(battle):
        state = original_conversion(battle)
        world = {'index': len(worlds), 'adapter': battle_snapshot(battle), 'engine': engine_snapshot(state)}
        write(output / ('sampled-world-' + str(len(worlds)) + '.json'), world)
        worlds.append(world)
        return state
    def counted_search(battle):
        calls.append(True)
        if len(calls) != 1:
            raise AssertionError('More than one final-boundary native search')
        return original_search(battle)
    started = time.monotonic()
    with patch.object(native, 'battle_to_poke_engine_state', capture_conversion), patch.object(native, 'find_best_move', counted_search):
        try:
            choice = adapter.choose()
            result = {'outcome': 'choice', 'choice': choice}
        except Exception as exc:
            result = {'outcome': 'error', 'error_type': type(exc).__name__, 'error': str(exc),
                      'selection_audit': getattr(exc, 'audit', None), 'traceback': traceback.format_exc()}
    write(output / 'raw-native-result.json', result)
    if len(calls) != 1 or result['outcome'] != 'error' or result['error_type'] != 'LegalSelectionError':
        raise ValueError('Unexpected current native-boundary reproduction result')
    audit = result['selection_audit']
    same(sorted(c['action'] for c in audit['candidates']), [9, 13], 'current incompatible roots')
    same(audit['legal_count'], 0, 'current legal root count')
    return {'outcome': 'reproduced-current-incompatibility', 'request_id': 20,
            'native_find_best_move_calls': len(calls), 'serialized_sampled_worlds': len(worlds),
            'configured_search_time_ms': record['search_time_ms'], 'elapsed_search_seconds': time.monotonic() - started,
            'historical_native_search_calls': 0, 'native_result': result,
            'scope': 'One current native search at original boundary only. Unseeded Rust/time-budget search and unreplayed Python sampling RNG history mean this is not an exact stochastic replay. No action submitted, no labels.'}


def showdown_prefix(output):
    from pokemon_teacher_runtime.transport import JsonWorker
    record, transcript = fixture()
    comparisons = []
    with JsonWorker(['node', 'packages/teacher-assisted/external-worker.mjs'], output / 'worker.stderr', timeout=30.) as worker:
        with (output / 'worker-rpc.jsonl').open('x') as stream:
            def request(op, **payload):
                if time.time() >= DEADLINE:
                    raise TimeoutError('Investigation deadline reached')
                append(stream, {'event': 'request', 'op': op, 'payload': payload})
                response = worker.request(op, **payload)
                append(stream, {'event': 'response', 'op': op, 'data': response})
                return response
            hello = request('hello')
            same(hello, record['worker'], 'worker identity')
            bundle = request('start', run_seed=record['run_seed'], episode_index=record['episode_index'],
                             swap=record['swap'], external_side=record['external_side'], split=record['split'], tactical_side=None)
            same(bundle['frame']['provenance'], record['provenance'], 'original game provenance')
            for index, step in enumerate(record['steps']):
                same(bundle, transcript[index], 'full boundary ' + str(index))
                same(step['index'], index, 'step order')
                if bundle['frame']['outcome'] is not None:
                    raise ValueError('Recorded prefix unexpectedly terminal')
                pending = {s for s in ('p1', 'p2') if any(bundle['frame']['players'][s]['mask'])}
                same(sorted(step['actions']), sorted(pending), 'pending action sides')
                same(sorted(step['request_ids']), sorted(pending), 'pending request sides')
                for side in pending:
                    player = bundle['frame']['players'][side]
                    action = step['actions'][side]
                    if type(action) is not int or not 0 <= action < 14 or not player['mask'][action]:
                        raise ValueError('Illegal original prefix action')
                    same(step['request_ids'][side], player['request_id'], 'prefix request ID')
                bundle = request('step', actions=step['actions'], request_ids=step['request_ids'])
                same(bundle['frame'].get('submitted', []), step['submitted'], 'accepted submissions ' + str(index))
                comparisons.append({'index': index, 'full_boundary_matches': True, 'submissions_match': True})
            same(bundle, transcript[-1], 'saved failed final boundary')
            same(bundle['frame']['outcome'], None, 'stop at live failure boundary')
            write(output / 'replayed-final-boundary.json', bundle)
            request('close')
    if worker.process.returncode != 0 or worker.cleanup_errors:
        raise ValueError('Prefix worker did not close cleanly')
    return {'full_boundaries_compared': 20, 'recorded_steps_replayed': 19,
            'comparison_checks': comparisons, 'terminal_outcome': None,
            'stopped_request_id': bundle['external']['request_id'], 'native_search_calls': 0,
            'new_decisions': 0, 'worker_exit_code': worker.process.returncode,
            'scope': 'Original recorded actions only in fresh authoritative Showdown session. Stop at saved request 20; no choice at boundary, full game, labels or learner work.'}


def child(stage, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Bounded investigation child signalled ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = {'inspect': inspect, 'native-boundary': native_boundary, 'showdown-prefix': showdown_prefix}[stage](output)
        write(output / 'report.json', {'status': 'passed', 'stage': stage, 'result': result,
              'scientific_games': 0, 'accepted_labels': 0, 'elapsed_seconds': time.monotonic() - started})
    except BaseException as exc:
        write(output / 'failure.json', {'status': 'failed', 'stage': stage, 'error': repr(exc),
              'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started})
        raise


def supervise(stage, output):
    from pokemon_teacher_runtime import lifecycle
    from pokemon_teacher_runtime.qualification import cleanup_stage
    process, snapshot, error = None, None, None
    started = time.monotonic()
    command = [str(ROOT / '.venv-f34-external/bin/python'), str(Path(__file__).resolve()),
               '--child', stage, '--output', str(output / stage)]
    cleanup = {'status': 'not-started'}
    with (output / (stage + '.log')).open('xb') as log:
        try:
            if time.time() >= DEADLINE:
                raise TimeoutError('Investigation deadline reached')
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'python'),
                'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(output / (stage + '-launch.json'), {'command': command, 'pid': process.pid,
                  'started_utc': datetime.now(timezone.utc).isoformat(), 'limit_seconds': LIMITS[stage]})
            while process.poll() is None:
                snapshot = lifecycle.capture_tree(process, snapshot)
                if time.monotonic() - started >= LIMITS[stage] or time.time() >= DEADLINE:
                    raise TimeoutError('Bounded stage expired: ' + stage)
                time.sleep(.03)
        except BaseException as exc:
            error = repr(exc)
        finally:
            if process is not None:
                cleanup = cleanup_stage(process, snapshot)
    result = {'stage': stage, 'exit_code': process.poll() if process else None, 'error': error,
              'cleanup': cleanup, 'elapsed_seconds': time.monotonic() - started}
    write(output / (stage + '-exit.json'), result)
    if error is not None or result['exit_code'] != 0 or cleanup['status'] != 'passed':
        raise RuntimeError('Bounded stage failed: ' + canonical(result))
    if any(cleanup.get(k) for k in ('errors', 'remaining_pids', 'unresolved_unknown_pids', 'reused_pids', 'unknown_overflow', 'table_error', 'signals', 'handle_signals')):
        raise RuntimeError('Unclean stage termination: ' + stage)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--child', choices=STAGES)
    parser.add_argument('--without-native-search', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if ALLOWED_OUTPUT != output and ALLOWED_OUTPUT not in output.parents:
        raise ValueError('Output must be inside new F38 reproduction directory')
    if args.child:
        child(args.child, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    started, stages = time.monotonic(), []
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in FIXTURE.iterdir() if p.is_file()}
    bindings.update({p: sha(ROOT / p) for p in SOURCE_PATHS})
    pin = read(ROOT / 'packages/external-benchmark/pin.json')
    bindings.update({str(Path('artifacts/external/foul-play') / p): sha(ROOT / 'artifacts/external/foul-play' / p) for p in pin['foul_play']['files']})
    bindings.update({p: sha(ROOT / p) for p in pin['poke_engine']['runtime_files']})
    write(output / 'input-bindings.json', bindings)
    try:
        for stage in STAGES:
            if stage == 'native-boundary' and args.without_native_search:
                continue
            stages.append(supervise(stage, output))
        same({name: sha(ROOT / name) for name in bindings}, bindings, 'input/source final hashes')
        evidence = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
        write(output / 'report.json', {'schema_version': 'f38-teacher-compatibility-reproduction-v1',
              'status': 'passed', 'stages': stages, 'input_files_unchanged': True,
              'input_files': bindings, 'evidence_files': evidence,
              'scientific_games': 0, 'accepted_labels': 0, 'training_steps': 0,
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'Discarded F37 request engineering diagnosis and recorded-action prefix replay only; no repair, labels, data reuse or scientific evaluation.'})
    except BaseException as exc:
        write(output / 'failure.json', {'status': 'failed', 'error': repr(exc), 'stages': stages,
              'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
