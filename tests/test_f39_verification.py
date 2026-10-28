"""Focused synthetic verification boundaries; no scientific games or updates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pokemon_online_continuation import verification as v


def fixture():
    row = {'features': np.asarray([1., 2.], np.float32), 'mask': np.asarray([True, False]),
        'action': 0, 'old_logp': 0., 'value': .5, 'reward': 1., 'next_index': -1,
        'bootstrap': 0., 'terminal': True, 'trajectory': 0}
    game = {'episode': 10, 'split': 'train', 'swap': False, 'learner_side': 'p1',
        'opponent': 'tactical', 'outcome': {'kind': 'terminal', 'rewards': {'p1': 1, 'p2': 0}},
        'provenance': {}, 'decisions': 2, 'phases': {'move': 2}, 'unavailable_choices': 0,
        'trace': [['p1', 1, 0]], 'actor_trace': [{'side': 'p1', 'action': 0}],
        'value_mean': .5, 'terminal_value_mse': .25, 'accepted_decisions': 2,
        'accepted_controlled_decisions': 1, 'eligible_learner_rows': 1, 'rejected_attempts': []}
    return [row], game


class VerificationTests(unittest.TestCase):
    def test_original_rows_are_checked_beyond_actions_and_outcomes(self):
        rows, game = fixture()
        h = v.fingerprint(v.prepare_batch(v.merge_rows([(rows, game)])))
        self.assertEqual(v.check_game(rows, game, game, 'train', {'selected_indices': [0]}, h), 1)
        changed = copy.deepcopy(rows)
        changed[0]['features'][0] += 1
        with self.assertRaisesRegex(ValueError, 'complete-game rows'):
            v.check_game(changed, game, game, 'train', {'selected_indices': [0]}, h)
        changed = copy.deepcopy(game)
        changed['actor_trace'][0]['action'] = 1
        with self.assertRaisesRegex(ValueError, 'actor_trace'):
            v.check_game(rows, changed, game, 'train', {'selected_indices': [0]}, h)
        with self.assertRaisesRegex(ValueError, 'requires original'):
            v.check_game(rows, game, game, 'train')

    def test_original_rejected_row_selection_is_required(self):
        rows, game = fixture()
        expected = copy.deepcopy(game)
        rejected = {'learner_row_index': 0, 'selected_for_update': None}
        game['rejected_attempts'] = [rejected]
        expected['rejected_attempts'] = [{**rejected, 'selected_for_update': True}]
        h = v.fingerprint(v.prepare_batch(v.merge_rows([(rows, game)])))
        v.check_game(rows, game, expected, 'train', {'selected_indices': [0]}, h)
        with self.assertRaisesRegex(ValueError, 'attribution'):
            v.check_game(rows, game, expected, 'train', {'selected_indices': []}, h)
        development = {**game, 'split': 'development', 'eligible_learner_rows': 0}
        v.check_game([], development, development, 'development')

    def test_initial_adam_and_cursors_are_exact(self):
        p = {'x': np.asarray([.5], np.float32)}
        opt = ({'x': np.asarray([.1], np.float32)}, {'x': np.asarray([.2], np.float32)}, np.int32(4096))
        meta = {'cohorts': 512, 'stream_cursors': dict.fromkeys(v.STREAMS, 100),
                'next_episode': 300, 'selected_rows': 524288}
        identity = {'f34_metadata': copy.deepcopy(meta)}
        v.check_initial(p, opt, meta, p, opt, identity)
        reset = ({'x': np.zeros(1, np.float32)}, opt[1], opt[2])
        with self.assertRaisesRegex(ValueError, 'Adam moments'):
            v.check_initial(p, reset, meta, p, opt, identity)
        changed = copy.deepcopy(meta)
        changed['stream_cursors']['self_play'] += 1
        with self.assertRaisesRegex(ValueError, 'stream_cursors'):
            v.check_initial(p, opt, changed, p, opt, identity)

    def test_prespecified_coverage_and_deadline_failure(self):
        self.assertEqual(v.EXPECTED['game_executions'], 18 + 48)
        self.assertEqual(v.COHORTS, (512, 767))
        self.assertEqual(v.EXPECTED['checkpoint_comparisons'], 9)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'training/inputs').mkdir(parents=True)
            (root / 'training/inputs/manifest.json').write_text(json.dumps({'plan': {}}))
            with patch.object(v, 'deadline', side_effect=TimeoutError('deadline')), patch.object(v, 'Pool') as pool:
                with self.assertRaisesRegex(TimeoutError, 'deadline'):
                    v.run(root / 'training', root / 'verification')
            pool.assert_not_called()
            self.assertEqual(json.loads((root / 'verification/failure.json').read_text())['counts'], {})

    def test_stochastic_teacher_module_inventory_is_measured_and_scoped(self):
        with patch.dict(v.sys.modules, {'pokemon_teacher_assisted.collector': object(),
                'pokemon_imitation.teacher': object(), 'fp.search.main': object(),
                'pokemon_teacher_assisted.demonstrations': object(), 'poke_engine.native': object()}):
            observed = v.teacher_search_modules()
            self.assertIn('fp.search.main', observed)
            self.assertIn('pokemon_teacher_assisted.demonstrations', observed)
            self.assertIn('poke_engine.native', observed)
            self.assertNotIn('pokemon_teacher_assisted.collector', observed)
            self.assertNotIn('pokemon_imitation.teacher', observed)

    def test_output_cannot_modify_original_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'outside'):
                v.run(root / 'training', root / 'training/replay')
            self.assertFalse((root / 'training').exists())


if __name__ == '__main__':
    unittest.main()
