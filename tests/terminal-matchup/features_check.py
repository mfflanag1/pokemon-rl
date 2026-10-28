"""Standalone checks: python tests/matchup/features_check.py."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
import numpy as np
from pokemon_full.features import encode as base_encode
from pokemon_terminal_matchup.features import CHANNELS, EXTRA, FEATURES, encode, extras, validate_catalog
from pokemon_switch_audit.semantic import TYPES


def fixture():
    chart = {a: {d: 1. for d in TYPES} for a in TYPES}
    chart['Fire'].update(Grass=2., Steel=2., Water=.5, Fire=.5)
    chart['Water'].update(Fire=2., Grass=.5)
    chart['Normal']['Ghost'] = 0.
    moves = {
        'flame': {'type': 'Fire', 'category': 'Special', 'base_power': 100},
        'splashhit': {'type': 'Water', 'category': 'Special', 'base_power': 100},
        'tackle': {'type': 'Normal', 'category': 'Physical', 'base_power': 50},
        'rest': {'type': 'Psychic', 'category': 'Status', 'base_power': 0},
    }
    catalog = {'schema_version': 'f17-public-metadata-v1', 'types': list(TYPES),
               'species': {'fox': {'types': ['Fire']}, 'plant': {'types': ['Grass', 'Steel']},
                           'ghost': {'types': ['Ghost']}},
               'moves': moves, 'effectiveness': chart}
    own = [{'ident': str(i), 'species': 'Fox', 'level': 100,
            'condition': '150/200 brn', 'active': i == 0, 'moves': ['flame', 'tackle'],
            'ability': None, 'item': None, 'tera': None, 'tera_type': 'Water',
            'stats': {'atk': 120, 'spa': 240, 'def': 180, 'spd': 300, 'spe': 360},
            'boosts': {}, 'reviving': False} for i in range(6)]
    foe = {'ident': 'public', 'species': 'Plant', 'level': 100, 'condition': '50/100',
           'active': True, 'moves': ['splashhit', 'rest'], 'ability': None, 'item': None,
           'tera': None, 'boosts': {}, 'types': None}
    active_moves = [dict(id=m, pp=10, maxpp=10, disabled=False, **moves[m])
                    for m in ('flame', 'tackle')]
    player = {'schema_version': 'full-showdown-player-v1', 'side': 'p1', 'request_id': 1,
              'turn': 1, 'phase': 'move', 'own': own, 'opponent': [foe] + [None] * 5,
              'active_moves': active_moves, 'weather': None, 'terrain': None,
              'side_conditions': {'own': {}, 'opponent': {}},
              'mask': np.array([True, True, False, False, True, True, False, False,
                                False, True, True, True, True, True], dtype=bool)}
    return player, catalog


class MatchupChecks(unittest.TestCase):
    def setUp(self):
        self.p, self.c = fixture()

    def rows(self):
        return extras(self.p, self.c).reshape(14, 25)

    def test_fire_grass_steel_hand_calculation(self):
        row = self.rows()[0]
        np.testing.assert_allclose(row[:12], [1, 1, 0, 0, 0, 1, 0, .4, .75, 1, 1, .3])
        np.testing.assert_allclose(row[12:18], [.75, 1, 1, .5, 1, .5])

    def test_immunity(self):
        self.p['opponent'][0]['species'] = 'Ghost'
        row = self.rows()[1]
        self.assertEqual(row[9], 1)
        self.assertEqual(row[10], 0)
        self.assertEqual(row[11], 0)

    def test_switch_slot_alignment_and_stable_best(self):
        self.p['own'][2]['condition'] = '10/100'
        self.p['own'][2]['moves'] = ['tackle', 'flame']
        rows = self.rows()
        self.assertAlmostEqual(float(rows[10, 12]), .1)
        self.assertAlmostEqual(float(rows[9, 12]), .75)
        self.assertAlmostEqual(float(rows[10, 11]), .3)
        self.assertEqual(rows[10, 5], 1)
        self.assertTrue(np.all(rows[8] == 0))
        self.p['opponent'] = [None] * 6  # Both pressures tie at zero.
        self.assertEqual(self.rows()[10, 4], 1)
        self.p['own'][2]['moves'] = []
        self.assertTrue(np.all(self.rows()[10, 4:12] == 0))

    def test_own_stats_stage_multipliers(self):
        self.p['own'][0]['boosts'] = {'atk': 2, 'spa': -2, 'def': 6, 'spd': -6, 'spe': 1}
        np.testing.assert_allclose(self.rows()[0, 20:], [.2, .1, .6, .0625, .45])

    def test_tera_stab_and_defense(self):
        self.p['own'][0]['tera_type'] = 'Fire'
        self.assertEqual(self.rows()[4, 8], 1)
        self.p['own'][0]['tera_type'] = 'Water'
        self.p['active_moves'][1] = dict(id='splashhit', pp=10, maxpp=10, disabled=False,
                                        **self.c['moves']['splashhit'])
        rows = self.rows()
        self.assertEqual(rows[5, 8], .75)
        self.assertAlmostEqual(float(rows[0, 18]), .1)  # .4 * .5 * 2/4
        self.assertAlmostEqual(float(rows[4, 18]), .05)  # neutral Water defense
        self.p['own'][0]['tera_type'] = 'Stellar'
        self.assertEqual(self.rows()[4, 8], .75)
        self.assertAlmostEqual(float(self.rows()[4, 18]), .1)
        self.p['own'][0]['tera'] = 'Water'
        self.assertAlmostEqual(float(self.rows()[0, 18]), .05)

    def test_unknown_foe_and_revealed_nonvacuity(self):
        rows = self.rows()
        self.assertGreater(rows[0, 18], 0)
        self.assertEqual(rows[0, 19], 1)
        self.p['opponent'][0]['moves'] = ['rest']
        self.assertEqual(self.rows()[0, 18], 0)
        self.assertEqual(self.rows()[0, 19], 0)
        self.assertEqual(self.rows()[0, 17], .25)
        self.p['opponent'] = [None] * 6
        np.testing.assert_array_equal(self.rows()[0, [9, 10, 11, 15, 16, 17, 18, 19]], 0)

    def test_zero_control_base_equality_and_no_mutation(self):
        before = copy.deepcopy(self.p)
        catalog_before = copy.deepcopy(self.c)
        original, mask = base_encode(self.p)
        zero, zm = encode(self.p, self.c, 'zero')
        matchup, mm = encode(self.p, self.c, 'matchup')
        self.assertEqual((CHANNELS, EXTRA, FEATURES), (25, 350, 1278))
        self.assertEqual(matchup.shape, (1278,))
        self.assertEqual(matchup.dtype, np.float32)
        self.assertTrue(np.isfinite(matchup).all())
        np.testing.assert_array_equal(zero[:928], original)
        np.testing.assert_array_equal(matchup[:928], original)
        np.testing.assert_array_equal(zero[928:], 0)
        np.testing.assert_array_equal(zm, mask)
        np.testing.assert_array_equal(mm, mask)
        np.testing.assert_array_equal(self.p.pop('mask'), before.pop('mask'))
        self.assertEqual(self.p, before)
        self.assertEqual(self.c, catalog_before)

    def test_reject_frames_hidden_stats_and_unknown_arm(self):
        for bad in ({'players': self.p}, dict(self.p, coordinator=True)):
            with self.assertRaises(ValueError):
                extras(bad, self.c)
        self.p['opponent'][0]['stats'] = {'atk': 999}
        with self.assertRaises(ValueError):
            extras(self.p, self.c)
        with self.assertRaises(ValueError):
            encode(fixture()[0], self.c, 'other')

    def test_waiting_and_recharge(self):
        self.p['phase'] = 'wait'
        self.p['mask'][:] = False
        np.testing.assert_array_equal(self.rows(), 0)
        self.p['phase'] = 'move'
        self.p['mask'][0] = True
        self.p['active_moves'][0].update(id='recharge', type='???', category='Status', base_power=0)
        row = self.rows()[0]
        np.testing.assert_array_equal(row[4:9], [0, 0, 1, 0, 0])
        self.assertEqual(row[11], 0)

    def test_invalid_metadata_both_arms(self):
        self.c['moves']['flame']['base_power'] = float('nan')
        for arm in ('zero', 'matchup'):
            with self.assertRaises(ValueError):
                encode(self.p, self.c, arm)
        self.p, self.c = fixture()
        del self.p['own'][0]['stats']['spe']
        for arm in ('zero', 'matchup'):
            with self.assertRaises(ValueError):encode(self.p, self.c, arm)

    def test_explicit_public_types(self):
        self.p['opponent'][0]['types'] = ['Water']
        self.assertEqual(self.rows()[0, 10], .125)
        self.p['opponent'][0]['tera'] = 'Grass'
        self.assertEqual(self.rows()[0, 10], .5)

    def test_full_catalog_admission_and_missing_record(self):
        self.c['species']['missingno'] = {'types':['Bird','Normal']}
        validate_catalog(self.c)
        self.c['moves']['unobserved'] = {'type':'Fire','category':'Special','base_power':float('nan')}
        with self.assertRaises(ValueError):validate_catalog(self.c)
        del self.c['moves']['unobserved']
        del self.c['species']['fox']
        with self.assertRaises(KeyError):self.rows()


if __name__ == '__main__':
    unittest.main()
