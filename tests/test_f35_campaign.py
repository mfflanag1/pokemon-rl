"""Synthetic complete-trajectory rejection/selection tests; no scientific data."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from pokemon_teacher_assisted.collector import Episode, BaselineEpisode
from pokemon_teacher_assisted.campaign import annotate_selection, optimize

SIDES = ['p1', 'p2']


def frame(request, live=('p1', 'p2'), outcome=None, submitted=()):
    return {'players': {side: {'phase': 'move' if side in live else 'wait',
            'mask': [side in live] * 14, 'request_id': request, 'turn': request} for side in SIDES},
        'observable_history': {}, 'tactical_actions': {'p1': 1, 'p2': 1},
        'baseline_actions': {'p1': 2, 'p2': 2},
        'outcome': outcome, 'submitted': list(submitted),
        'provenance': {'teams': [{'partition': 'train'}, {'partition': 'train'}]}}


def outputs(current, action=0):
    return [(np.zeros(4224, np.float32), np.asarray(current['players'][side]['mask'], bool),
        action if any(current['players'][side]['mask']) else -1, -.3, .4) for side in SIDES]


def submissions(command, rejected=()):
    return [{'side': side, 'action': action, 'accepted': side not in rejected,
        'feedback': ['[Unavailable choice] trapped'] if side in rejected else []}
        for side, action in command[1]['actions'].items()]


def terminal():
    return {'kind': 'terminal', 'winner': 'Full p1', 'turns': 3, 'rewards': {'p1': 1., 'p2': 0.}}


class CampaignSemantics(unittest.TestCase):
    def test_rejected_controlled_row_retained_reward_paid_only_once_and_counter_advances(self):
        episode = Episode(seed=3501, episode=0, max_turns=20)
        current = frame(1)
        episode.receive('start', current)
        command = episode.advance(outputs(current, 9))
        current = frame(2, live=('p1',), submitted=submissions(command, ('p1',)))
        episode.receive('step', current)
        command = episode.advance(outputs(current))
        current = frame(3, submitted=submissions(command))
        episode.receive('step', current)
        command = episode.advance(outputs(current))
        current = frame(4, live=(), outcome=terminal(), submitted=submissions(command))
        episode.receive('step', current)
        self.assertIsNone(episode.advance(outputs(current)))
        rows, game = episode.result
        self.assertEqual(len(rows), 5)
        self.assertEqual(episode.counts, [3, 2])
        self.assertEqual([r['next_index'] for r in rows], [2, 4, 3, -1, -1])
        self.assertEqual([r['reward'] for r in rows], [0., 0., 0., 1., 0.])
        self.assertEqual([r['terminal'] for r in rows], [False, False, False, True, True])
        self.assertEqual(game['unavailable_choices'], 1)
        self.assertEqual(game['accepted_decisions'], 4)
        attempt = game['rejected_attempts'][0]
        self.assertEqual((attempt['side'], attempt['learner_row_index'], attempt['actor_counter']), ('p1', 0, 0))
        self.assertTrue(attempt['controlled'])
        self.assertEqual(attempt['refreshed_request_id'], 2)
        result = annotate_selection([(rows, game)], {'selected_indices': [0, 3], 'collected_rows': 5})
        self.assertEqual(result['rejected_selected_rows'], 1)
        self.assertTrue(attempt['selected_for_update'])

    def test_opponent_rejection_does_not_create_learner_row(self):
        episode = Episode(seed=3501, episode=0, max_turns=20, opponent='tactical')
        current = frame(1)
        episode.receive('start', current)
        command = episode.advance(outputs(current))
        current = frame(2, live=('p2',), submitted=submissions(command, ('p2',)))
        episode.receive('step', current)
        command = episode.advance(outputs(current))
        current = frame(3, live=(), outcome=terminal(), submitted=submissions(command))
        episode.receive('step', current)
        episode.advance(outputs(current))
        rows, game = episode.result
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['reward'], 1)
        self.assertFalse(game['rejected_attempts'][0]['controlled'])
        self.assertIsNone(game['rejected_attempts'][0]['learner_row_index'])
        annotate_selection([(rows, game)], {'selected_indices': [0], 'collected_rows': 1})
        self.assertFalse(game['rejected_attempts'][0]['selected_for_update'])
        self.assertEqual(game['accepted_controlled_decisions'], 1)

    def test_invalid_feedback_or_unrefreshed_request_fails(self):
        for problem in ('feedback', 'request', 'missing', 'action'):
            episode = Episode(seed=3501, episode=0, max_turns=20)
            current = frame(1)
            episode.receive('start', current)
            command = episode.advance(outputs(current))
            current = frame(2, submitted=submissions(command, ('p1',)))
            if problem == 'feedback': current['submitted'][0]['feedback'] = ['[Invalid choice] illegal']
            if problem == 'request': current['players']['p1']['request_id'] = 1
            if problem == 'missing': current['submitted'] = current['submitted'][1:]
            if problem == 'action': current['submitted'][0]['action'] = 2
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                episode.receive('step', current)

    def test_baseline_is_development_only_and_uses_pinned_advice(self):
        with self.assertRaises(ValueError):
            BaselineEpisode(seed=3501, episode=0, max_turns=20, opponent='baseline-v1')
        episode = BaselineEpisode(seed=3501, episode=0, max_turns=20,
            split='development', opponent='baseline-v1')
        current = frame(1)
        for team in current['provenance']['teams']: team['partition'] = 'development'
        episode.receive('start', current)
        command = episode.advance(outputs(current))
        self.assertEqual(command[1]['actions'], {'p1': 0, 'p2': 2})
        current = frame(2, live=(), outcome=terminal(), submitted=submissions(command))
        episode.receive('step', current)
        episode.advance(outputs(current))
        self.assertEqual(episode.result[0], [])
        self.assertEqual(episode.result[1]['opponent'], 'baseline-v1')
        self.assertEqual(len(episode.result[1]['actor_trace']), 1)
        self.assertIsNone(episode.start['tactical_side'])
        self.assertEqual(episode.start['baseline_side'], 'p2')

    def test_control_never_enters_demonstration_optimizer(self):
        with patch('pokemon_teacher_assisted.campaign.original_optimize', return_value='original') as original, \
             patch('pokemon_teacher_assisted.campaign.assisted_optimize', side_effect=AssertionError):
            self.assertEqual(optimize('control', 1, 2, 3, 4, object(), 5, 6), 'original')
            original.assert_called_once_with(1, 2, 3, 4)
        with patch('pokemon_teacher_assisted.campaign.assisted_optimize', return_value='assisted') as assisted:
            self.assertEqual(optimize('assisted', 1, 2, 3, 4, 'dataset', 5, 6), 'assisted')
            assisted.assert_called_once_with(1, 2, 3, 4, dataset='dataset', demo_seed=5, cohort=6, coefficient=.1)

    def test_selection_uses_complete_game_offsets_and_excludes_opponent_attempt(self):
        def game(row, controlled=True):
            return {'rejected_attempts': [{'learner_row_index': row, 'controlled': controlled}]}
        a, b, c = game(0), game(1), game(None, False)
        result = annotate_selection([([{}, {}], a), ([{}, {}], b), ([{}], c)],
            {'selected_indices': [0, 3, 4], 'collected_rows': 5})
        self.assertEqual(result, {'rejected_attempts': 3, 'rejected_controlled_attempts': 2, 'rejected_selected_rows': 2})
        self.assertTrue(a['rejected_attempts'][0]['selected_for_update'])
        self.assertTrue(b['rejected_attempts'][0]['selected_for_update'])
        self.assertFalse(c['rejected_attempts'][0]['selected_for_update'])


if __name__ == '__main__':
    unittest.main()
