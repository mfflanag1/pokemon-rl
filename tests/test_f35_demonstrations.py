"""Focused teacher capture, acceptance and split tests; no scientific games."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from pokemon_teacher_assisted.external import label_candidate, accept_label, run_game, public_catalog
from pokemon_teacher_assisted.teacher import AcceptedTeacher
from pokemon_structured_policy.features import empty_history
from pokemon_external_benchmark.foul_play import decision_to_action

ROOT = Path(__file__).resolve().parents[1]


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'artifacts/reviews/f34-policy-reset/external-integration/timing-smoke-001/transcript.jsonl'
        cls.bundle = json.loads(path.read_text().splitlines()[0])
        cls.player = cls.bundle['frame']['players']['p2']
        cls.history = empty_history('p2')  # The old engineering fixture predates observed history.

    def candidate(self):
        player = deepcopy(self.player)
        action = next(i for i, valid in enumerate(player['mask']) if valid)
        return label_candidate(player, deepcopy(self.history), {'action': action, 'request_id': player['request_id']},
            step_index=0, external_input_sha256='a' * 64, catalog=public_catalog())

    def test_real_player_encoding_has_only_supervised_fields(self):
        candidate = self.candidate()
        self.assertEqual(set(candidate), {'features', 'mask', 'action', 'metadata'})
        self.assertEqual(candidate['features'].shape, (4224,))
        self.assertEqual(candidate['features'].dtype, np.float32)
        self.assertEqual(candidate['mask'].dtype, np.bool_)
        self.assertTrue(candidate['mask'][candidate['action']])

    def test_rejected_attempt_has_no_label_and_other_seat_cannot_authorize_it(self):
        candidate = self.candidate()
        row = {'side': 'p2', 'action': candidate['action'], 'request_id': candidate['metadata']['request_id'],
               'accepted': False, 'feedback': ['[Unavailable choice] trapped'], 'command': 'move 1'}
        self.assertIsNone(accept_label(candidate, [row]))
        with self.assertRaises(ValueError):
            accept_label(candidate, [{**row, 'side': 'p1', 'accepted': True}])
        accepted = accept_label(candidate, [{**row, 'accepted': True}])
        self.assertEqual(accepted['metadata']['submission_index'], 0)

    def test_wait_frame_excluded_before_encoding(self):
        player = {'mask': [False] * 14}
        with patch('pokemon_teacher_assisted.external.encode_packed') as encoder:
            self.assertIsNone(label_candidate(player, {}, {'waiting': True}, step_index=0, external_input_sha256='a' * 64))
            encoder.assert_not_called()
            with self.assertRaises(ValueError):
                label_candidate(player, {}, {'action': 0}, step_index=0, external_input_sha256='a' * 64)

    def test_teacher_side_and_request_alignment_are_strict(self):
        player, history = deepcopy(self.player), deepcopy(self.history)
        action = next(i for i, valid in enumerate(player['mask']) if valid)
        with self.assertRaisesRegex(ValueError, 'Foreign'):
            label_candidate(player, {**history, 'side': 'p1'}, {'action': action, 'request_id': player['request_id']},
                            step_index=0, external_input_sha256='a' * 64)
        for decision in [{'action': action, 'request_id': player['request_id'] + 1},
                         {'action': -1, 'request_id': player['request_id']},
                         {'action': True, 'request_id': player['request_id']}]:
            with self.assertRaises(ValueError):
                label_candidate(player, history, decision, step_index=0, external_input_sha256='a' * 64)

    def test_encoder_mask_disagreement_is_not_silently_repaired(self):
        action = next(i for i, valid in enumerate(self.player['mask']) if valid)
        with patch('pokemon_teacher_assisted.external.encode_packed', return_value=(np.zeros(4224, np.float32), np.zeros(14, bool))):
            with self.assertRaisesRegex(ValueError, 'mask differs'):
                label_candidate(self.player, self.history, {'action': action, 'request_id': self.player['request_id']},
                                step_index=0, external_input_sha256='a' * 64)

    def test_mapping_uses_current_move_and_roster_slots(self):
        request = {'active': [{'moves': [{'id': 'surf'}, {'id': 'thunderbolt'}]}]}
        self.assertEqual(decision_to_action(['/choose move thunderbolt'], request), 1)
        self.assertEqual(decision_to_action(['/choose move thunderbolt terastallize'], request), 5)
        self.assertEqual(decision_to_action(['/switch 6'], request), 13)
        with self.assertRaises(ValueError):
            decision_to_action(['/choose move earthquake'], request)

    def test_final_split_rejected_before_output_or_process(self):
        with patch('pokemon_teacher_assisted.external.JsonWorker') as worker:
            with self.assertRaisesRegex(ValueError, 'splits'):
                run_game(ROOT / 'SHOULD_NOT_EXIST_F35', run_seed=1, episode_index=0, swap=False,
                    external_side='p2', split='final', fp_seed=1, learner=None, purpose='test')
            worker.assert_not_called()
        self.assertFalse((ROOT / 'SHOULD_NOT_EXIST_F35').exists())


class FakeAdapter:
    def __init__(self):
        self.battle = SimpleNamespace(user=SimpleNamespace(last_selected_move={'move': 'previous'}))
        self.pending = False
        self.request_count = 0
        self.last_decided_request = -1
        self.ingested = []

    def ingest(self, envelope):
        self.ingested.append((deepcopy(self.battle.user.last_selected_move), envelope))
        self.pending = envelope['pending']
        self.request_count = envelope['request_id']

    def choose(self):
        if self.last_decided_request == self.request_count:
            raise ValueError('Duplicate request')
        self.battle.user.last_selected_move = {'move': 'proposed'}
        self.last_decided_request = self.request_count
        return {'request_id': self.request_count, 'action': 0}


class TeacherFeedbackTests(unittest.TestCase):
    def test_accepted_choice_commits_and_requires_ack_before_ingesting(self):
        adapter = FakeAdapter(); teacher = AcceptedTeacher('p2', adapter=adapter)
        teacher.choose({'pending': True, 'request_id': 1})
        with self.assertRaisesRegex(ValueError, 'feedback'):
            teacher.choose({'pending': True, 'request_id': 2})
        self.assertEqual(len(adapter.ingested), 1)
        teacher.feedback(1, True)
        self.assertEqual(adapter.battle.user.last_selected_move, {'move': 'proposed'})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            teacher.feedback(1, True)

    def test_rejected_choice_restores_before_refreshed_protocol_ingest(self):
        adapter = FakeAdapter(); teacher = AcceptedTeacher('p2', adapter=adapter)
        teacher.choose({'pending': True, 'request_id': 1})
        teacher.feedback(1, False)
        self.assertEqual(adapter.battle.user.last_selected_move, {'move': 'previous'})
        self.assertEqual(adapter.last_decided_request, 1)
        teacher.choose({'pending': True, 'request_id': 2})
        self.assertEqual(adapter.ingested[-1][0], {'move': 'previous'})
        teacher.feedback(2, True)

    def test_wait_needs_no_ack_and_stale_ack_fails_closed(self):
        adapter = FakeAdapter(); teacher = AcceptedTeacher('p2', adapter=adapter)
        self.assertEqual(teacher.choose({'pending': False, 'request_id': 1}), {'waiting': True})
        with self.assertRaises(ValueError): teacher.feedback(1, True)
        teacher.choose({'pending': True, 'request_id': 2})
        with self.assertRaises(ValueError): teacher.feedback(1, True)
        with self.assertRaises(ValueError): teacher.feedback(2, 1)
        teacher.feedback(2, False)


if __name__ == '__main__':
    unittest.main()
