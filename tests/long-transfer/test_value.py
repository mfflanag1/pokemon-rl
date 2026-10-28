"""Run with: python -m unittest discover -s tests/long-transfer -v."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    'value', Path(__file__).resolve().parents[2] / 'python/pokemon_long_transfer/value.py')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize_value = _MODULE.summarize_value


def game(values, target=1., side='p1', kind='terminal'):
    other = 'p2' if side == 'p1' else 'p1'
    return {'learner_side': side,
            'outcome': {'kind': kind, 'rewards': {side: target, other: 1-target}},
            'trace': [[side, i, 0, v, 'hash'] for i, v in enumerate(values)] +
                     [[other, 0, 1, None, 'hash']]}


class ValueTests(unittest.TestCase):
    def test_unequal_lengths_tie_and_cutoff(self):
        # Game errors: 1, (0 + .25 + 1)/3, 0. Baselines: .25, .25, 0.
        result = summarize_value([game([0]), game([0, .5, 1], 0, 'p2'),
                                  game([.5], .5), game([0]*20, kind='administrative_cutoff')])
        self.assertEqual((result['terminal_games'], result['cutoff_games'],
                          result['terminal_decisions']), (3, 1, 5))
        for key, expected in {'game_mean_mse': 17/36, 'decision_mean_mse': .45,
                              'game_mean_baseline_mse': 1/6,
                              'decision_mean_baseline_mse': .2,
                              'game_mean_improvement': -11/36,
                              'decision_mean_improvement': -.25,
                              'mean_prediction': .4, 'mean_target': .3,
                              'prediction_std': np.sqrt(.14)}.items():
            self.assertAlmostEqual(result[key], expected, msg=key)
        self.assertEqual(result['bins'][0]['count'], 2)
        self.assertEqual(result['bins'][0]['empirical_target'], .5)
        self.assertEqual(result['bins'][0]['mean_mse'], .5)
        json.dumps(result, allow_nan=False)

    def test_edges_and_numpy_scalars(self):
        r = summarize_value([game([np.float64(0), .1, 1.])])
        self.assertEqual([b['count'] for b in r['bins']], [1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(r['bins'][1]['mean_prediction'], .1)
        self.assertAlmostEqual(r['bins'][1]['mean_mse'], .81)
        for i, b in enumerate(r['bins']):
            self.assertEqual((b['lo'], b['hi']), (i/10, (i+1)/10))
            if b['count'] == 0:
                self.assertIsNone(b['mean_prediction'])
                self.assertIsNone(b['empirical_target'])
                self.assertIsNone(b['mean_mse'])
        json.dumps(r, allow_nan=False)

    def test_all_cutoff_and_empty(self):
        for games, count in [([game([1], kind='administrative_cutoff')], 1), ([], 0)]:
            r = summarize_value(games)
            self.assertEqual(r['cutoff_games'], count)
            self.assertEqual(r['terminal_games'], 0)
            self.assertEqual(r['terminal_decisions'], 0)
            for key, value in r.items():
                if key not in ('cutoff_games', 'terminal_games', 'terminal_decisions', 'bins'):
                    self.assertIsNone(value, key)
            self.assertTrue(all(b['count'] == 0 for b in r['bins']))
            json.dumps(r, allow_nan=False)

    def test_constant_half_comparator_including_tie(self):
        r = summarize_value([game([.5], 1), game([.5, .5], 0), game([.5]*4, .5)])
        self.assertEqual(r['game_mean_improvement'], 0)
        self.assertEqual(r['decision_mean_improvement'], 0)
        self.assertAlmostEqual(r['decision_mean_baseline_mse'], 3/28)
        self.assertAlmostEqual(r['game_mean_baseline_mse'], 1/6)

    def test_invalid_predictions_even_in_cutoffs(self):
        for value in [None, True, '0.5', complex(.5), [], np.array([.5]),
                      float('nan'), float('inf'), -float('inf'), -.1, 1.1]:
            for kind in ['terminal', 'administrative_cutoff']:
                with self.subTest(value=repr(value), kind=kind), self.assertRaises(ValueError):
                    summarize_value([game([value], kind=kind)])

    def test_invalid_structure_and_rewards(self):
        base = game([.5])
        invalid = [None, {}, game([])]
        for key, value in [('learner_side', 'p3'), ('trace', None), ('outcome', None)]:
            g = copy.deepcopy(base); g[key] = value; invalid.append(g)
        for field, value in [('kind', 'draw'), ('rewards', {'p1': 1}),
                             ('rewards', {'p1': .2, 'p2': .8}),
                             ('rewards', {'p1': 1, 'p2': 1}),
                             ('rewards', {'p1': True, 'p2': 0}),
                             ('rewards', {'p1': float('nan'), 'p2': 0})]:
            g = copy.deepcopy(base); g['outcome'][field] = value; invalid.append(g)
        for index, value in [(0, 'p3'), (1, None), (2, -1), (2, True), (4, None)]:
            g = copy.deepcopy(base); g['trace'][0][index] = value; invalid.append(g)
        for entry in [[], ['p1', 0], 'abcde']:
            g = copy.deepcopy(base); g['trace'][0] = entry; invalid.append(g)
        g = copy.deepcopy(base); g['trace'][-1][3] = .5; invalid.append(g)
        for g in invalid:
            with self.subTest(game=g), self.assertRaises(ValueError):
                summarize_value([g])


if __name__ == '__main__':
    unittest.main()
