"""Synthetic qualification-harness tests: no native search, workers or games."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_teacher_legality import qualification as q


def prefix():
    transcript, steps = [], []
    for index in range(12):
        mask = ([False] * 14 if index == 10 else
                [True] + [False] * 13 if index == 11 else [True, True] + [False] * 12)
        transcript.append({'frame': {'players': {'p2': {'mask': mask, 'request_id': index + 1}}},
                           'external': {'request_id': index + 1}})
        if index < 11:
            steps.append({'external': {'action': 0, 'decision': 'original-tera'},
                'submitted': [] if index == 10 else [{'side': 'p2', 'action': 0,
                    'request_id': index + 1, 'accepted': index != 2}]})
    return transcript, {'external_side': 'p2', 'outcome': None, 'steps': steps}


class FakeTeacher:
    def __init__(self):
        self.adapter = SimpleNamespace(battle=SimpleNamespace(turn=8,
            user=SimpleNamespace(active=SimpleNamespace(name='fixture'), last_selected_move='initial')))
        self.feedbacks, self.before = [], []
        self.override = None

    def choose(self, envelope, mask, request_id):
        self.before.append(deepcopy(self.adapter.battle.user.last_selected_move))
        if not any(mask):
            return {'waiting': True}
        self.previous = self.adapter.battle.user.last_selected_move
        self.adapter.battle.user.last_selected_move = 'fresh-unexecuted-choice'
        result = {'action': 0, 'request_id': request_id, 'selection': {'legal_mask': mask,
            'request_id': request_id, 'kind': 'forced-request' if sum(mask) == 1 else 'native-legal-root',
            'search_calls': int(sum(mask) > 1)}}
        if self.override:
            self.override(result, request_id)
        return result

    def feedback(self, request_id, accepted):
        self.feedbacks.append((request_id, accepted))
        if not accepted:
            self.adapter.battle.user.last_selected_move = self.previous


class QualificationChecks(unittest.TestCase):
    def test_deterministic_native_search_guard_measures_zero_and_restores_entrypoint(self):
        original = lambda: 'unpatched'
        native = SimpleNamespace(get_result_from_mcts=original)
        with q.no_native_search(native) as calls:
            self.assertEqual(calls.call_count, 0)
        self.assertIs(native.get_result_from_mcts, original)

    def test_deterministic_native_search_guard_fails_on_accidental_entry(self):
        original = lambda: 'unpatched'
        native = SimpleNamespace(get_result_from_mcts=original)
        with self.assertRaisesRegex(AssertionError, 'attempted native search'):
            with q.no_native_search(native) as calls:
                native.get_result_from_mcts('unexpected search')
        self.assertEqual(calls.call_count, 1)
        self.assertIs(native.get_result_from_mcts, original)

    def test_history_reconciles_only_original_accepted_bookkeeping(self):
        transcript, record = prefix()
        teacher = FakeTeacher()
        result = q.historical_sequence(teacher, transcript, record, io.StringIO(), lambda *x: x)
        self.assertEqual(result['native_choices'], 10)
        self.assertEqual(teacher.feedbacks, [(i, False) for i in [*range(1, 11), 12]])
        self.assertEqual(teacher.before[1], ('fixture', 'original', 8))
        self.assertEqual(teacher.before[3], teacher.before[2])
        self.assertNotIn('fresh-unexecuted-choice', teacher.before)

    def test_raw_invalid_reply_is_preserved_before_guard(self):
        transcript, record = prefix()
        teacher = FakeTeacher()
        teacher.override = lambda result, _: result.update(action=13)
        journal = io.StringIO()
        with self.assertRaisesRegex(ValueError, 'authoritative request'):
            q.historical_sequence(teacher, transcript, record, journal, lambda *x: x)
        rows = [json.loads(line) for line in journal.getvalue().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['event'], 'raw-choice')
        self.assertEqual(rows[0]['choice']['action'], 13)
        self.assertEqual(teacher.feedbacks, [])

    def test_false_forced_search_claim_rejected(self):
        transcript, record = prefix()
        teacher = FakeTeacher()
        def change(result, request_id):
            if request_id == 12:
                result['selection']['search_calls'] = 1
        teacher.override = change
        with self.assertRaisesRegex(ValueError, 'provenance'):
            q.historical_sequence(teacher, transcript, record, io.StringIO(), lambda *x: x)

    def test_native_search_counts_are_independently_observed_per_request(self):
        transcript, record = prefix()
        with self.assertRaisesRegex(ValueError, 'observed native search'):
            q.historical_sequence(FakeTeacher(), transcript, record, io.StringIO(), lambda *x: x, lambda: 0)

    def test_historical_feedback_mismatch_is_rejected(self):
        transcript, record = prefix()
        record['steps'][0]['submitted'][0]['request_id'] = 99
        with self.assertRaisesRegex(ValueError, 'feedback coordinates'):
            q.historical_sequence(FakeTeacher(), transcript, record, io.StringIO(), lambda *x: x)

    def test_missing_historical_teacher_submission_is_rejected(self):
        transcript, record = prefix()
        record['steps'][0]['submitted'] = []
        with self.assertRaisesRegex(ValueError, 'submission inventory'):
            q.historical_sequence(FakeTeacher(), transcript, record, io.StringIO(), lambda *x: x)

    def test_wait_reply_cannot_hide_an_action(self):
        with self.assertRaisesRegex(ValueError, 'Waiting'):
            q.check_choice({'waiting': True, 'action': 0}, {'mask': [False] * 14})

    def test_boolean_action_cannot_pass_as_integer_zero(self):
        with self.assertRaisesRegex(ValueError, 'authoritative'):
            q.check_choice({'action': False, 'request_id': 1}, {'mask': [True] * 14, 'request_id': 1})

    def test_evidence_write_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'record.json'
            q.save(path, {'original': True})
            with self.assertRaises(FileExistsError):
                q.save(path, {'replacement': True})
            self.assertEqual(q.read(path), {'original': True})

    def test_supervisor_observation_exception_still_cleans_and_records_failure(self):
        from pokemon_teacher_runtime import lifecycle
        process = SimpleNamespace(pid=123, returncode=None)
        process.poll = lambda: process.returncode
        def cleanup(child, snapshot):
            child.returncode = 143
            return {'status': 'passed', 'remaining_pids': [], 'errors': []}
        with tempfile.TemporaryDirectory() as folder, patch.object(q.subprocess, 'Popen', return_value=process), \
                patch.object(lifecycle, 'capture_tree', side_effect=RuntimeError('observation unavailable')), \
                patch.object(q, 'cleanup_stage', side_effect=cleanup) as cleaned, patch.object(q, 'DEADLINE', float('inf')):
            output = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'supervision failed'):
                q.supervise('historical-request', output)
            cleaned.assert_called_once_with(process, None)
            terminal = q.read(output / 'historical-request-exit.json')
            self.assertIn('observation unavailable', terminal['error'])
            self.assertEqual(terminal['exit_code'], 143)
            self.assertTrue((output / 'historical-request.log').exists())

    def test_supervisor_rejects_unresolved_cleanup_despite_zero_child_exit(self):
        process = SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
        with tempfile.TemporaryDirectory() as folder, patch.object(q.subprocess, 'Popen', return_value=process), \
                patch.object(q, 'cleanup_stage', return_value={'status': 'passed', 'remaining_pids': [],
                    'unresolved_unknown_pids': [123]}), patch.object(q, 'DEADLINE', float('inf')):
            with self.assertRaisesRegex(RuntimeError, 'supervision failed'):
                q.supervise('historical-request', Path(folder))


if __name__ == '__main__':
    unittest.main()
