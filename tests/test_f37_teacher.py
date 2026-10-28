"""F37 adapter/journal checks with stub search and no new simulator executions."""
from copy import deepcopy
from collections import namedtuple
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_teacher_legality import TEACHER_POLICY, RESPONSE_JOURNAL
from pokemon_teacher_legality.teacher import LegalFoulPlayAdapter, AcceptedTeacher
from pokemon_teacher_legality import external, teacher as teacher_module

FAILED = ROOT / 'artifacts/runs/f36-teacher-data-v1/train/8209/058-0/transcript.jsonl'


def mask(*actions):
    return [i in actions for i in range(14)]


def results(*choices):
    return [(NS(side_one=[NS(move_choice=choice, visits=visits) for choice, visits in choices],
                total_visits=sum(visits for _, visits in choices)), 1., 0)]


def fake_format(battle, decision):
    if decision.startswith('switch '):
        name = decision[7:]
        matches = [p for p in battle.user.reserve if p.name == name]
        if len(matches) != 1:
            raise ValueError('Unknown switch')
        return ['/switch ' + str(matches[0].index), str(battle.rqid)]
    # Match the pinned formatter's missing-move dereference failure; the
    # F37 request prevalidation must prevent reaching it for absent choices.
    name = decision.removesuffix('-tera').removesuffix('-mega')
    if name not in [move['id'] for move in battle.request_json['active'][0]['moves']]:
        raise AttributeError("'NoneType' object has no attribute 'can_z'")
    tera = decision.endswith('-tera')
    return ['/choose move ' + decision.removesuffix('-tera') + (' terastallize' if tera else ''), str(battle.rqid)]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        # Only the adapter boundary is exercised. Search, battle parsing and RNG
        # are not replaced in the separate actual-engineering qualification.
        modules = {name: ModuleType(name) for name in
                   ('fp', 'fp.battle', 'fp.battle.state', 'fp.modes', 'fp.modes.base', 'fp.search', 'fp.search.main')}
        modules['fp'].battle = modules['fp.battle']
        modules['fp'].modes = modules['fp.modes']
        modules['fp'].search = modules['fp.search']
        modules['fp.battle'].state = modules['fp.battle.state']
        modules['fp.modes'].base = modules['fp.modes.base']
        modules['fp.search'].main = modules['fp.search.main']
        modules['fp.battle.state'].LastUsedMove = namedtuple('LastUsedMove', 'pokemon_name move turn')
        modules['fp.modes.base'].format_decision = fake_format
        self.native = modules['fp.search.main']
        self.original_selector = Mock(side_effect=AssertionError('Original selector must be intercepted'))
        self.native.select_move_from_mcts_results = self.original_selector
        self.native.find_best_move = Mock()
        self.modules_patch = patch.dict(sys.modules, modules)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)

    def adapter(self, legal_mask, *, request=None, request_id=12):
        request = deepcopy(request or {'active': [{'moves': [{'id': 'surf'}, {'id': 'protect'}]}]})
        request['rqid'] = request_id
        user = NS(active=NS(name='zekrom'), reserve=[NS(name='pikachu', index=6), NS(name='mew', index=2)],
                  last_selected_move={'previous': True}, update_from_request_json=lambda req: None)
        adapter = LegalFoulPlayAdapter.__new__(LegalFoulPlayAdapter)
        adapter.battle = NS(user=user, opponent=NS(reserve=[], active=NS()), started=True, wait=False,
                           request_json=request, rqid=request_id, turn=8,
                           mode=NS(search_params=lambda b: (8, 12)))
        adapter.pending = True
        adapter.request_count = request_id
        adapter.last_decided_request = -1
        adapter.search_time_ms = 100
        adapter.set_request_legality(legal_mask, request_id)
        return adapter

    def search(self, fixture):
        self.native.find_best_move.side_effect = lambda battle: self.native.select_move_from_mcts_results(fixture)

    def test_preserved_locked_outrage_is_action_zero_without_search_or_rng(self):
        bundle = json.loads(FAILED.read_text().splitlines()[-1])
        request = next(json.loads(line[9:]) for line in reversed(bundle['external']['protocol'])
                       if line.startswith('|request|'))
        adapter = self.adapter(bundle['frame']['players']['p2']['mask'], request=request)
        with patch('pokemon_teacher_legality.teacher.random.choices', side_effect=AssertionError('No forced RNG')):
            response = adapter.choose()
        self.assertEqual((response['action'], response['decision'], response['command']), (0, 'outrage', '/choose move outrage'))
        self.assertEqual(response['selection']['kind'], 'forced-request')
        self.assertEqual(response['selection']['search_calls'], 0)
        self.assertEqual(response['sampled_worlds'], 0)
        self.native.find_best_move.assert_not_called()
        self.assertEqual(adapter.battle.user.last_selected_move.move, 'outrage')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            adapter.choose()

    def test_forced_switch_uses_current_roster_index_not_reserve_list_position(self):
        response = self.adapter(mask(13)).choose()
        self.assertEqual((response['action'], response['decision'], response['command']), (13, 'switch pikachu', '/switch 6'))
        self.native.find_best_move.assert_not_called()

    def test_no_tera_and_trapped_native_mass_is_filtered_before_threshold(self):
        self.search(results(('surf-tera', 80), ('switch pikachu', 12), ('surf', 7), ('protect', 1)))
        response = self.adapter(mask(0, 1)).choose()
        self.assertEqual(response['action'], 0)
        self.assertEqual(response['selection']['kind'], 'native-legal-root')
        self.assertEqual(response['selection']['search_calls'], 1)
        self.assertEqual((response['sampled_worlds'], response['per_world_ms'], response['configured_search_total_ms']), (8, 12, 96))
        self.assertIs(self.native.select_move_from_mcts_results, self.original_selector)
        self.native.find_best_move.assert_called_once()

    def test_absent_native_move_is_audited_before_formatter_none_dereference(self):
        self.search(results(('earthquake', 90), ('surf', 10)))
        response = self.adapter(mask(0, 1)).choose()
        self.assertEqual(response['action'], 0)
        candidates = response['selection']['audit']['candidates']
        absent = next(row for row in candidates if row['choice'] == 'earthquake')
        self.assertEqual(absent['reason'], 'unmappable_choice')
        self.assertFalse(absent['allowed'])
        self.assertIn('absent/ambiguous', absent['detail'])
        self.native.find_best_move.assert_called_once()

    def test_native_selector_failure_restores_binding_and_never_retries_or_commits(self):
        self.search(results(('surf-tera', 100)))
        adapter = self.adapter(mask(0, 1))
        before = deepcopy(adapter.battle.user.last_selected_move)
        with self.assertRaises(Exception) as failure:
            adapter.choose()
        self.assertIsNotNone(getattr(failure.exception, 'audit', None))
        self.assertIs(self.native.select_move_from_mcts_results, self.original_selector)
        self.assertEqual(adapter.battle.user.last_selected_move, before)
        self.assertEqual(adapter.last_decided_request, -1)
        self.native.find_best_move.assert_called_once()

    def test_stale_request_fails_before_search(self):
        adapter = self.adapter(mask(0, 1))
        adapter.set_request_legality(mask(0, 1), 11)
        with self.assertRaisesRegex(ValueError, 'stale'):
            adapter.choose()
        self.native.find_best_move.assert_not_called()

    def test_choice_lock_singleton_uses_advertised_slot_without_other_roster_moves(self):
        response = self.adapter(mask(1), request={'active': [{'moves': [
            {'id': 'surf', 'disabled': True}, {'id': 'protect'}]}]}).choose()
        self.assertEqual((response['action'], response['decision']), (1, 'protect'))
        self.native.find_best_move.assert_not_called()


class FakeAcceptedAdapter:
    def __init__(self):
        self.battle = NS(user=NS(last_selected_move={'old': True}))
        self.pending = False
        self.calls = 0

    def set_request_legality(self, legal_mask, request_id):
        self.mask, self.request_id = legal_mask, request_id

    def ingest(self, envelope):
        self.calls += 1
        self.pending = envelope['pending']

    def choose(self):
        self.battle.user.last_selected_move = {'new': True}
        return {'request_id': self.request_id, 'action': next(i for i, v in enumerate(self.mask) if v)}


class AcceptanceTests(unittest.TestCase):
    def envelope(self, request_id=1, pending=True):
        return {'schema_version': 'f34-external-player-v1', 'side': 'p2',
                'request_id': request_id, 'pending': pending, 'protocol': []}

    def test_wait_is_exact_and_rejection_restores_previous_move(self):
        adapter = FakeAcceptedAdapter()
        teacher = AcceptedTeacher('p2', adapter=adapter)
        self.assertEqual(teacher.choose(self.envelope(pending=False), mask(), 1), {'waiting': True})
        self.assertIsNone(teacher.pending)
        teacher.choose(self.envelope(2), mask(0), 2)
        with self.assertRaisesRegex(ValueError, 'feedback'):
            teacher.choose(self.envelope(3), mask(0), 3)
        self.assertEqual(adapter.calls, 2)
        teacher.feedback(2, False)
        self.assertEqual(adapter.battle.user.last_selected_move, {'old': True})
        teacher.choose(self.envelope(3), mask(0), 3)
        teacher.feedback(3, True)
        self.assertEqual(adapter.battle.user.last_selected_move, {'new': True})

    def test_mask_request_and_foreign_input_fail_before_ingestion(self):
        adapter = FakeAcceptedAdapter()
        teacher = AcceptedTeacher('p2', adapter=adapter)
        fixtures = [(self.envelope(), [1] + [False] * 13, 1),
                    (self.envelope(), mask(0), 2),
                    (self.envelope(pending=False), mask(0), 1),
                    ({**self.envelope(), 'side': 'p1'}, mask(0), 1)]
        for envelope, legal, request_id in fixtures:
            with self.subTest(envelope=envelope, legal=legal, request_id=request_id), self.assertRaises(ValueError):
                teacher.choose(envelope, legal, request_id)
        self.assertEqual(adapter.calls, 0)


class JournalTests(unittest.TestCase):
    def run_failed(self, reply=None, error=None, label_guard=None):
        bundle = json.loads(FAILED.read_text().splitlines()[-1])
        records = []
        class Worker:
            def __init__(self, command, path):
                self.node = command[0] == 'node'
                self.timeout = 60.
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def request(self, op, **payload):
                records.append((self.node, op, payload))
                if self.node:
                    return {} if op == 'hello' else deepcopy(bundle)
                if op == 'choose':
                    if error is not None: raise error
                    return deepcopy(reply)
                return {'started': True}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'game'
            def guard(*args, **kwargs):
                rows = [json.loads(line) for line in (output / RESPONSE_JOURNAL).read_text().splitlines()]
                self.assertEqual(rows[0]['raw_response'], reply)
                if label_guard: label_guard(rows)
                raise ValueError('Injected parent label guard')
            with patch.object(external, 'JsonWorker', Worker), patch.object(external, 'label_candidate', guard):
                with self.assertRaises((ValueError, RuntimeError)):
                    external.run_game(output, run_seed=1850182552, episode_index=58, swap=False,
                        external_side='p2', split='train', fp_seed=1, learner=lambda *args: 0,
                        purpose='mock-journal-only', capture_demonstrations=True)
            journal = [json.loads(line) for line in (output / RESPONSE_JOURNAL).read_text().splitlines()]
            result = json.loads((output / 'result.json').read_text())
        return journal, result, records, bundle

    def test_invalid_raw_reply_survives_before_label_guard_without_saved_step(self):
        reply = {'action': 4, 'request_id': 12, 'teacher_policy': TEACHER_POLICY,
                 'selection': {'kind': 'native-legal-root'}}
        journal, result, records, bundle = self.run_failed(reply=reply)
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]['raw_response']['action'], 4)
        self.assertEqual(journal[0]['external_input_sha256'], external.json_hash(bundle['external']))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['steps'], [])
        choose = next(payload for node, op, payload in records if not node and op == 'choose')
        self.assertEqual(choose, {'input': bundle['external'], 'legal_mask': mask(0), 'request_id': 12})
        self.assertEqual(result['teacher_policy_counts']['native_legal_root'], 1)

    def test_real_stdio_rejection_preserves_complete_error_reply_and_audit(self):
        wire = {'ok': False, 'error': 'no legal root', 'error_type': 'LegalSelectionError',
                'teacher_policy': TEACHER_POLICY, 'selection_audit': {
                    'candidates': [{'choice': 'surf-tera', 'action': 4, 'allowed': False}]},
                'extra_evidence': {'worlds': 8}}
        child = ('import json,sys\n'
                 'request=json.loads(sys.stdin.readline())\n'
                 'reply=' + repr(wire) + '\n'
                 'reply["id"]=request["id"]\n'
                 'print(json.dumps(reply),flush=True)\n')
        with tempfile.TemporaryDirectory() as tmp:
            worker = external.JsonWorker([sys.executable, '-c', child], Path(tmp) / 'worker.stderr')
            try:
                with self.assertRaisesRegex(RuntimeError, 'External worker rejected command') as caught:
                    worker.request('choose', legal_mask=mask(0, 1), request_id=12)
                self.assertEqual(external.rejected_wire_response(caught.exception), {**wire, 'id': 1})
                self.assertTrue(worker.closed)
                self.assertEqual(worker.process.poll(), 0)
            finally:
                worker.close()

    def test_worker_error_reply_and_selector_audit_survive(self):
        wire = {'id': 2, 'ok': False, 'error': 'no legal mass', 'selection_audit': {'candidates': ['surf-tera']}}
        journal, result, _, _ = self.run_failed(error=RuntimeError('External worker rejected command: ' + json.dumps(wire)))
        self.assertEqual(journal[0]['status'], 'error')
        self.assertEqual(journal[0]['raw_error_response'], wire)
        self.assertEqual(result['teacher_policy_counts']['errors'], 1)
        self.assertIn('no legal mass', result['error'])


class TeacherProtocolTests(unittest.TestCase):
    def test_failed_selection_audit_is_in_original_error_reply(self):
        failure = ValueError('No legal positive root mass')
        failure.audit = {'candidates': [{'choice': 'surf-tera', 'action': 4, 'legal': False}]}
        adapter = Mock()
        adapter.choose.side_effect = failure
        commands = [
            {'op': 'start', 'id': 1, 'side': 'p2', 'search_time_ms': 100, 'seed': 9},
            {'op': 'choose', 'id': 2, 'input': {'request_id': 12}, 'legal_mask': mask(0, 1), 'request_id': 12}]
        output = io.StringIO()
        with patch.object(teacher_module, 'AcceptedTeacher', return_value=adapter), \
                patch.object(sys, 'stdin', io.StringIO('\n'.join(json.dumps(c) for c in commands))), \
                patch.object(sys, 'stdout', output), patch.object(sys, 'stderr', io.StringIO()):
            teacher_module.main()
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(replies[1]['id'], 2)
        self.assertFalse(replies[1]['ok'])
        self.assertEqual(replies[1]['selection_audit'], failure.audit)
        self.assertEqual(replies[1]['teacher_policy'], TEACHER_POLICY)
        self.assertEqual(replies[1]['error_type'], 'ValueError')


class OverlayTests(unittest.TestCase):
    def child(self, code):
        return subprocess.run([sys.executable, '-c', code], cwd=ROOT,
            env={**__import__('os').environ, 'PYTHONPATH': str(ROOT / 'python')}, capture_output=True, text=True)

    def test_fresh_overlay_binds_before_consumers_and_rejects_reinstall(self):
        outcome = self.child('from pokemon_teacher_legality.stage import install_overlay\n'
            'r=install_overlay()\n'
            'from pokemon_teacher_assisted import external\n'
            'from pokemon_teacher_runtime.transport import JsonWorker\n'
            'assert external.run_game.__module__=="pokemon_teacher_legality.external"\n'
            'assert external.JsonWorker is JsonWorker\n'
            'try: install_overlay()\n'
            'except RuntimeError: pass\n'
            'else: raise AssertionError("Reinstall accepted")\n')
        self.assertEqual(outcome.returncode, 0, outcome.stderr)

    def test_imported_stage_consumer_is_rejected(self):
        outcome = self.child('from pokemon_teacher_assisted import external\n'
            'from pokemon_teacher_legality.stage import install_overlay\n'
            'try: install_overlay()\n'
            'except RuntimeError: pass\n'
            'else: raise AssertionError("Stale consumer accepted")\n')
        self.assertEqual(outcome.returncode, 0, outcome.stderr)


if __name__ == '__main__':
    unittest.main()
