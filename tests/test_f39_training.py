"""Focused F39 inherited-state, schedule-role, rejection and fixed-budget checks."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from pokemon_online_continuation import common, campaign
from pokemon_structured_policy.checkpoint import load
from pokemon_structured_policy.common import training_specs as f34_training_specs
from pokemon_teacher_assisted.common import training_specs as f35_training_specs
from pokemon_full_train.storage import ROOT


def plan():
    return common.read(ROOT / 'contracts/online-continuation-v1/plan.json')


class ContinuationTraining(unittest.TestCase):
    def test_all_original_adam_moments_parameters_and_cursors_preserved(self):
        for seed in common.SEEDS:
            with self.subTest(seed=seed):
                parameters, optimizer, _, _, identity = common.load_initial(seed)
                original, old_optimizer, metadata = load(ROOT / 'artifacts/runs/f34-policy-reset-v1' / str(seed) / 'checkpoints/512')
                self.assertEqual(common.state_hash(parameters, optimizer), common.state_hash(original, old_optimizer))
                self.assertEqual(int(optimizer[2]), 4096)
                self.assertFalse(identity['adam_reset'])
                self.assertEqual(identity['f34_metadata'], metadata)
                self.assertTrue(any(np.count_nonzero(np.asarray(x)) for x in optimizer[0].values()))
                self.assertTrue(any(np.count_nonzero(np.asarray(x)) for x in optimizer[1].values()))
                self.assertEqual(sum(metadata['stream_cursors'].values()), metadata['next_episode'])

    def test_seed_inventory_rejects_boolean_and_unplanned_seed(self):
        for bad in (True, '7103', 7104, None):
            with self.subTest(seed=bad), self.assertRaises(ValueError):
                common.load_initial(bad)

    def test_namespaces_fresh_while_episode_cursors_are_inherited(self):
        p = plan()
        for seed in common.SEEDS:
            _, _, _, _, identity = common.load_initial(seed)
            for stream, opponent in common.STREAMS.items():
                cursor = identity['f34_metadata']['stream_cursors'][stream]
                specs = common.training_specs(p, seed, cursor, opponent)
                self.assertEqual([s['episode'] for s in specs], list(range(cursor, cursor + 8)))
                self.assertEqual({s['split'] for s in specs}, {'train'})
                for old_specs in (f34_training_specs, f35_training_specs):
                    old = old_specs(p, seed, cursor, opponent)
                    self.assertNotEqual(specs[0]['seed'], old[0]['seed'])
                    self.assertNotEqual(specs[0]['actor_seed'], old[0]['actor_seed'])
        with self.assertRaises(ValueError):
            common.training_specs(p, 7103, 0, 'baseline-v1')
        with self.assertRaises(ValueError):
            common.training_specs(p, 7103, 0, 'foul_play')

    def test_common_endpoint_panels_pair_every_opponent_without_learner_rows(self):
        p = plan()
        games = 0
        for seed in common.SEEDS:
            for opponent, count in p['evaluation_pairs'].items():
                specs = common.evaluation_specs(p, seed, opponent, 0, count)
                self.assertEqual(len(specs), 2 * count)
                self.assertEqual({s['split'] for s in specs}, {'development'})
                for first, second in zip(specs[::2], specs[1::2]):
                    self.assertEqual(first['episode'], second['episode'])
                    self.assertEqual((first['swap'], second['swap']), (False, True))
                    self.assertEqual((first['learner_side'], second['learner_side']), ('p1', 'p2'))
                self.assertEqual(specs, common.evaluation_specs(p, seed, opponent, 0, count))
                games += 2 * len(specs)
        self.assertEqual(games, 4608)
        with self.assertRaises(ValueError):
            common.evaluation_specs(p, 7103, 'foul_play', 0, 1)

    def test_fixed_budget_namespace_deadline_and_scope_fail_closed(self):
        p = plan()
        common.validate_plan(p)
        for key, bad in [('initial_cohorts_per_seed', 0), ('final_cohorts_per_seed', 769),
                         ('cohorts_per_seed', 512), ('epochs', 1), ('deadline_utc', '2030-01-01T00:00:00Z'),
                         ('sampled_update_cohorts', [0, 255]), ('public_server_used', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.validate_plan({**p, key: bad})
        changed = copy.deepcopy(p)
        changed['namespaces']['training_teams'] = 'f34-training-teams-v1'
        with self.assertRaises(ValueError):
            common.validate_plan(changed)
        with patch.object(common.time, 'time', return_value=0):
            cutoff = common.deadline(p)
        with patch.object(common.time, 'time', return_value=cutoff), self.assertRaises(TimeoutError):
            common.deadline(p)

    def test_rejected_rows_accounted_after_complete_game_selection(self):
        games = []
        for controlled, row, n in [(True, 1, 2), (False, None, 3), (True, 0, 1)]:
            game = {'eligible_learner_rows': n, 'accepted_decisions': 2,
                'decisions': 3, 'unavailable_choices': 1,
                'rejected_attempts': [{'learner_row_index': row, 'controlled': controlled}]}
            games.append(([None] * n, game))
        selection = {'selected_indices': [1, 4], 'collected_rows': 6}
        result = campaign.annotate_selection(games, selection)
        self.assertEqual(result, {'rejected_attempts': 3, 'rejected_controlled_attempts': 2, 'rejected_selected_rows': 1})
        self.assertEqual([g['rejected_attempts'][0]['selected_for_update'] for _, g in games], [True, False, False])
        games[0][1]['rejected_attempts'][0]['learner_row_index'] = 2
        with self.assertRaises(ValueError):
            campaign.annotate_selection(games, selection)

    def test_stream_cap_fails_before_unadmitted_collection(self):
        p = plan()
        p['maximum_games_per_stream'] = 0
        class NeverCollect:
            def collect(self, *_args):
                raise AssertionError('Collection must not execute beyond stream cap')
        with patch.object(campaign, 'deadline'), self.assertRaisesRegex(ValueError, 'stream cap'):
            campaign.collect_cohort(NeverCollect(), {}, p, 7103, 512,
                {key: 10 for key in common.STREAMS}, set())


if __name__ == '__main__':
    unittest.main()
