"""Synthetic boundary/negative checks for the bounded F35 verifier."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pokemon_teacher_assisted import verification as v


def external_fixture(cutoff=False):
    def player(turn, request, pending):
        return {'side': 'p1', 'phase': 'move' if pending else 'wait', 'turn': turn,
                'request_id': request, 'mask': [pending] + [False] * 13}
    provenance = {'split': 'development', 'teams': []}
    outcome = {'kind': 'administrative_cutoff' if cutoff else 'terminal', 'turns': 2,
        'winner': None if cutoff else 'Full p1', 'rewards': {'p1': 0 if cutoff else 1, 'p2': 0}}
    first = {'frame': {'players': {s: {**player(1, 1, True), 'side': s} for s in ('p1', 'p2')},
        'observable_history': {s: {'side': s} for s in ('p1', 'p2')},
        'outcome': None, 'provenance': provenance},
        'external': {'side': 'p2', 'request_id': 1, 'pending': True, 'protocol': ['|turn|1']}}
    submitted = [{'side': s, 'action': 0, 'accepted': True, 'feedback': [],
                  'request_id': 1, 'command': 'move 1'} for s in ('p1', 'p2')]
    second = {'frame': {'players': {s: {**player(2, 2, cutoff), 'side': s} for s in ('p1', 'p2')},
        'outcome': None if cutoff else outcome, 'provenance': provenance, 'submitted': submitted},
        'external': {'side': 'p2', 'request_id': 2, 'pending': cutoff, 'protocol': ['|turn|2']}}
    record = {'status': 'completed', 'learner_uses_history': True, 'run_seed': 35,
        'split': 'development', 'episode_index': 0, 'swap': False, 'external_side': 'p2', 'max_turns': 2,
        'provenance': provenance, 'outcome': outcome,
        'rejected_actions': {'p1': 0, 'p2': 0}, 'illegal_actions': {'p1': 0, 'p2': 0},
        'steps': [{'index': 0, 'turn': 1, 'actions': {'p1': 0, 'p2': 0},
            'external': {'action': 0, 'request_id': 1},
            'teacher_feedback': {'acknowledged': True, 'request_id': 1, 'accepted': True},
            'request_ids': {'p1': 1, 'p2': 1}, 'submitted': submitted,
            'external_input_sha256': v.json_hash(first['external'])}]}
    final = deepcopy(second)
    final['frame']['outcome'] = outcome
    final['external']['protocol'] = []
    transcript = [first, second] + ([final] if cutoff else [])
    return record, transcript


class FakeWorker:
    def __init__(self, transcript):
        self.transcript, self.calls = transcript, []

    def request(self, op, **payload):
        self.calls.append((op, payload))
        if op == 'hello': return {'schema_version': 'fixture'}
        if op == 'start': return deepcopy(self.transcript[0])
        if op == 'step': return deepcopy(self.transcript[1])
        if op == 'cutoff': return deepcopy(self.transcript[2])
        raise AssertionError(op)


def game_fixture():
    return {'episode': 0, 'split': 'train', 'swap': False, 'learner_side': 'p1',
        'opponent': 'tactical', 'outcome': {'kind': 'terminal'}, 'provenance': {},
        'decisions': 2, 'phases': {'move': 2}, 'unavailable_choices': 1,
        'trace': [['p1', 1, 0], ['p2', 1, 1]], 'actor_trace': [{'side': 'p1', 'action': 0}],
        'value_mean': .5, 'terminal_value_mse': .25, 'accepted_decisions': 1,
        'accepted_controlled_decisions': 0, 'eligible_learner_rows': 1,
        'rejected_attempts': [{'side': 'p1', 'learner_row_index': 0, 'selected_for_update': True}]}


class VerificationTests(unittest.TestCase):
    def test_external_replay_compares_every_boundary_and_explicit_split(self):
        record, transcript = external_fixture()
        worker = FakeWorker(transcript)
        result = v.replay_external(record, transcript, worker)
        self.assertEqual(result['decisions'], 2)
        self.assertEqual(result['projected_boundaries'], 2)
        self.assertEqual(result['rejected_attempts'], 0)
        self.assertEqual(worker.calls[1][1]['split'], 'development')
        self.assertEqual([op for op, _ in worker.calls], ['hello', 'start', 'step'])

    def test_cutoff_preserves_live_and_post_cutoff_frames(self):
        record, transcript = external_fixture(True)
        worker = FakeWorker(transcript)
        result = v.replay_external(record, transcript, worker)
        self.assertEqual(result['projected_boundaries'], 3)
        self.assertEqual([op for op, _ in worker.calls], ['hello', 'start', 'step', 'cutoff'])
        with self.assertRaisesRegex(ValueError, 'boundary count'):
            v.replay_external(record, transcript[:-1], FakeWorker(transcript))
        bad = deepcopy(transcript)
        bad[1]['frame']['players']['p1']['turn'] = 1
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            v.replay_external(record, bad, FakeWorker(bad))

    def test_teacher_ack_and_request_mutations_reject(self):
        record, transcript = external_fixture()
        for mutate in ('request', 'ack', 'digest', 'outcome'):
            changed = deepcopy(record)
            if mutate == 'request': changed['steps'][0]['request_ids']['p1'] = 2
            if mutate == 'ack': changed['steps'][0]['teacher_feedback']['accepted'] = False
            if mutate == 'digest': changed['steps'][0]['external_input_sha256'] = '0' * 64
            if mutate == 'outcome': changed['outcome']['winner'] = 'Full p2'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                v.replay_external(changed, transcript, FakeWorker(transcript))

    def test_learner_action_log_probability_and_value_are_recomputed(self):
        record, transcript = external_fixture()
        learner = {'action_seed': 9, 'decisions': [{'request_id': 1, 'action': 0, 'log_probability': -.5, 'value': .75}]}
        output = [(np.zeros(4224, np.float32), np.ones(14, bool), 0, -.5, .75)]
        with patch.object(v, 'batched_actor', return_value=output) as actor:
            result = v.replay_external(record, transcript, FakeWorker(transcript),
                learner_record=learner, parameters={}, catalog={})
            self.assertEqual(result['learner_decisions_recomputed'], 1)
            self.assertEqual(actor.call_args.args[3:6], ([9], [0], 2))
            changed = deepcopy(learner); changed['decisions'][0]['value'] = .5
            with self.assertRaisesRegex(ValueError, 'log probability value'):
                v.replay_external(record, transcript, FakeWorker(transcript), learner_record=changed, parameters={}, catalog={})

    def test_rejected_row_selection_is_checked_separately_from_replay(self):
        expected = game_fixture(); actual = deepcopy(expected)
        actual['rejected_attempts'][0]['selected_for_update'] = None
        self.assertEqual(v.check_game([{}], actual, expected, 'train', {'selected_indices': [0]}), 1)
        with self.assertRaisesRegex(ValueError, 'selection attribution'):
            v.check_game([{}], actual, expected, 'train', {'selected_indices': []})
        with self.assertRaisesRegex(ValueError, 'row count'):
            v.check_game([], actual, expected, 'train', {'selected_indices': [0]})
        changed = deepcopy(actual); changed['actor_trace'][0]['action'] = 2
        with self.assertRaisesRegex(ValueError, 'actor_trace'):
            v.check_game([{}], changed, expected, 'train', {'selected_indices': [0]})

    def test_prespecified_counts_and_deadline_failure_are_explicit(self):
        self.assertEqual(v.EXPECTED['game_executions'], 36 + 48 + 18 + 12)
        self.assertEqual(v.EXPECTED['sampled_updates'], len(v.SEEDS) * len(v.ARMS) * len(v.COHORTS))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(v, 'before_deadline', side_effect=TimeoutError('deadline')), patch.object(v, 'Pool') as pool:
                with self.assertRaisesRegex(TimeoutError, 'deadline'):
                    v.run(root / 'training', root / 'external', root / 'teacher', root / 'verification')
            pool.assert_not_called()
            failure = json.loads((root / 'verification/failure.json').read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertEqual(failure['counts'], {})

    def test_fixture_qualification_requires_passed_and_bound_executed_evidence(self):
        teacher, learner = [json.loads((v.ROOT / rel).read_text()) for rel in v.QUALIFICATIONS]
        v.check_fixture_qualifications(teacher, learner)
        for changed in ({}, {'explicit_unavailable_choice_fixture': {'passed': False}},
                        {'explicit_unavailable_choice_fixture': {'passed': True, 'evidence_log': 'missing'}}):
            with self.subTest(record=changed), self.assertRaises(ValueError):
                v.check_fixture_qualifications(teacher, changed)

    def test_output_cannot_mutate_original_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'outside'):
                v.run(root / 'training', root / 'external', root / 'teacher', root / 'teacher/replay')
            self.assertFalse((root / 'teacher').exists())


if __name__ == '__main__':
    unittest.main()
