"""Tiny fixtures for bounded replay admission, external boundaries and failures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokemon_structured_policy import verification as v


def external_fixture(cutoff=False):
    def player(turn, request, pending):
        return {'phase': 'move' if pending else 'wait', 'turn': turn,
                'request_id': request, 'mask': [pending] + [False] * 13}
    provenance = {'teams': [{'partition': 'development', 'team_key': 'fixture'}]}
    outcome = {'kind': 'administrative_cutoff' if cutoff else 'terminal', 'turns': 2,
        'winner': None if cutoff else 'Full p1', 'rewards': {'p1': 0 if cutoff else 1, 'p2': 0}}
    first = {'frame': {'players': {s: player(1, 1, True) for s in ('p1', 'p2')},
        'outcome': None, 'provenance': provenance}, 'external': {'side': 'p2',
        'request_id': 1, 'pending': True, 'protocol': ['|turn|1']}}
    submitted = [{'side': s, 'action': 0, 'accepted': True, 'feedback': []} for s in ('p1', 'p2')]
    second = {'frame': {'players': {s: player(2, 2, cutoff) for s in ('p1', 'p2')},
        'outcome': None if cutoff else outcome, 'provenance': provenance, 'submitted': submitted},
        'external': {'side': 'p2', 'request_id': 2, 'pending': cutoff, 'protocol': ['|turn|2']}}
    record = {'status': 'completed', 'learner_uses_history': True, 'run_seed': 34,
        'episode_index': 0, 'swap': False, 'external_side': 'p2', 'max_turns': 2,
        'provenance': provenance, 'outcome': outcome,
        'rejected_actions': {'p1': 0, 'p2': 0}, 'illegal_actions': {'p1': 0, 'p2': 0},
        'steps': [{'index': 0, 'turn': 1, 'actions': {'p1': 0, 'p2': 0},
            'request_ids': {'p1': 1, 'p2': 1}, 'submitted': submitted,
            'external_input_sha256': v.external_digest(first['external'])}]}
    final = copy.deepcopy(second)
    final['frame']['outcome'] = outcome
    return record, [first, second], final


class FakeExternalWorker:
    def __init__(self, transcript, final):
        self.transcript, self.final, self.calls = transcript, final, []

    def request(self, op, **payload):
        self.calls.append((op, payload))
        if op == 'hello':
            return {'schema_version': 'fixture'}
        if op == 'start':
            return copy.deepcopy(self.transcript[0])
        if op == 'step':
            return copy.deepcopy(self.transcript[1])
        if op == 'cutoff':
            return copy.deepcopy(self.final)
        raise AssertionError(op)


def game_fixture():
    return {'episode': 0, 'split': 'train', 'swap': False, 'learner_side': 'p1',
        'opponent': 'tactical', 'outcome': {'kind': 'terminal'}, 'provenance': {},
        'decisions': 2, 'phases': {'move': 2}, 'unavailable_choices': 0,
        'trace': [['p1', 1, 0], ['p2', 1, 1]], 'actor_trace': [{'side': 'p1', 'action': 0}],
        'value_mean': .5, 'terminal_value_mse': .25}


class Verification(unittest.TestCase):
    def test_recorded_terminal_actions_compare_every_projected_boundary(self):
        record, transcript, final = external_fixture()
        worker = FakeExternalWorker(transcript, final)
        result = v.replay_external(record, transcript, worker)
        self.assertEqual(result['decisions'], 2)
        self.assertEqual(result['projected_boundaries'], 2)
        self.assertEqual([op for op, _ in worker.calls], ['hello', 'start', 'step'])
        self.assertEqual(worker.calls[-1][1], {'actions': {'p1': 0, 'p2': 0},
                                             'request_ids': {'p1': 1, 'p2': 1}})

    def test_recorded_cutoff_replays_live_joint_boundary_then_cutoff(self):
        record, transcript, final = external_fixture(cutoff=True)
        worker = FakeExternalWorker(transcript, final)
        result = v.replay_external(record, transcript, worker)
        self.assertEqual(result['outcome']['kind'], 'administrative_cutoff')
        self.assertEqual([op for op, _ in worker.calls], ['hello', 'start', 'step', 'cutoff'])
        changed = copy.deepcopy(transcript)
        changed[-1]['frame']['players']['p1']['turn'] = 1
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            v.replay_external(record, changed, FakeExternalWorker(changed, final))

    def test_external_digest_requests_outcome_and_boundary_mutations_reject(self):
        record, transcript, final = external_fixture()
        bad = copy.deepcopy(record)
        bad['steps'][0]['external_input_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'input digest'):
            v.replay_external(bad, transcript, FakeExternalWorker(transcript, final))
        bad = copy.deepcopy(record)
        bad['steps'][0]['request_ids']['p1'] = 2
        with self.assertRaisesRegex(ValueError, 'action/request'):
            v.replay_external(bad, transcript, FakeExternalWorker(transcript, final))
        bad = copy.deepcopy(record)
        bad['outcome']['winner'] = 'Full p2'
        with self.assertRaisesRegex(ValueError, 'outcome'):
            v.replay_external(bad, transcript, FakeExternalWorker(transcript, final))
        changed = copy.deepcopy(transcript)
        changed[0]['external']['protocol'] = ['|turn|2']
        with self.assertRaisesRegex(ValueError, 'boundary'):
            v.replay_external(record, changed, FakeExternalWorker(transcript, final))
        with self.assertRaisesRegex(ValueError, 'boundary count'):
            v.replay_external(record, transcript[:-1], FakeExternalWorker(transcript, final))

    def test_game_check_counts_only_controlled_training_rows(self):
        expected = game_fixture()
        self.assertEqual(v.check_game([{}], expected, expected, 'train'), 1)
        with self.assertRaisesRegex(ValueError, 'row count'):
            v.check_game([{}, {}], expected, expected, 'train')
        development = {**expected, 'split': 'development'}
        self.assertEqual(v.check_game([], development, development, 'development'), 0)
        altered = {**expected, 'actor_trace': [{'side': 'p1', 'action': 1}]}
        with self.assertRaisesRegex(ValueError, 'actor_trace'):
            v.check_game([{}], altered, expected, 'train')

    def test_coverage_is_explicit_and_deadline_preserves_failure_without_execution(self):
        self.assertEqual(v.EXPECTED['game_executions'], 18 + 36 + 12)
        self.assertEqual(v.EXPECTED['sampled_updates'], len(v.SEEDS) * len(v.COHORTS))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (patch.object(v, 'before_deadline', side_effect=TimeoutError('deadline')),
                  patch.object(v, 'Pool') as pool, patch.object(v, 'JsonWorker') as worker):
                with self.assertRaisesRegex(TimeoutError, 'deadline'):
                    v.run(root / 'training', root / 'initial', root / 'final', root / 'verification')
            pool.assert_not_called()
            worker.assert_not_called()
            failure = json.loads((root / 'verification/failure.json').read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertEqual(failure['counts'], {})

    def test_output_cannot_overwrite_or_extend_original_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'outside'):
                v.run(root / 'training', root / 'initial', root / 'final', root / 'training/replay')
            self.assertFalse((root / 'training').exists())


if __name__ == '__main__':
    unittest.main()
