"""Focused structured observation tests; no training or evaluation campaign."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from pokemon_structured_policy.features import (
    FEATURES, FIELD_SHAPES, FIELD_SLICES, EVENT_VOCABS, CATEGORICAL_SIZES, MON_NUMERIC_FIELDS,
    category, empty_history, encode, encode_packed, unpack_packed,
)
from pokemon_terminal_matchup.features import encode as frozen_encode
from pokemon_switch_audit.semantic import TYPES


def fixture():
    catalog = {'schema_version': 'f17-public-metadata-v1', 'types': list(TYPES),
               'species': {'pikachu': {'types': ['Electric']}, 'charizard': {'types': ['Fire', 'Flying']}},
               'moves': {'thunderbolt': {'type': 'Electric', 'category': 'Special', 'base_power': 90},
                         'tackle': {'type': 'Normal', 'category': 'Physical', 'base_power': 40}},
               'effectiveness': {a: {b: 1. for b in TYPES} for a in TYPES}}
    own = [{'ident': f'own{i}', 'species': 'Pikachu', 'level': 80, 'condition': '70/200 brn',
            'active': i == 0, 'moves': ['thunderbolt', 'tackle'], 'ability': 'static', 'item': '',
            'tera': None, 'tera_type': 'Water', 'stats': {s: 200 for s in ('atk', 'def', 'spa', 'spd', 'spe')},
            'boosts': {}, 'reviving': False, 'types': ['Electric']} for i in range(6)]
    foe = {'ident': 'observed', 'species': 'Charizard', 'level': 80, 'condition': '60/100',
           'active': True, 'moves': ['tackle'], 'ability': None, 'item': None, 'tera': None,
           'boosts': {}, 'types': None}
    player = {'schema_version': 'full-showdown-player-v2', 'side': 'p1', 'request_id': 1,
              'turn': 7, 'phase': 'move', 'own': own, 'opponent': [foe] + [None] * 5,
              'active_moves': [dict(id=name, pp=10, maxpp=15, disabled=False, **catalog['moves'][name])
                               for name in ('thunderbolt', 'tackle')],
              'weather': None, 'terrain': None, 'side_conditions': {'own': {}, 'opponent': {}},
              'mask': [True, True, False, False, True, True, False, False, False, True, True, True, True, True]}
    history = empty_history('p1')
    history['total_events'] = 1
    history['events'] = [{'ids': [EVENT_VOCABS['event_kind'].index('move'), 2, 1,
                                category('moves', 'Tackle'), category('species', 'Charizard'), 1, 1, 1, 1, 1],
                          'numeric': [6, 0, 0, 0]}]
    history['public_move_counts'][1] = [[category('moves', 'Tackle'), 4]]
    return player, catalog, history


class StructuredFeatures(unittest.TestCase):
    def setUp(self):
        self.player, self.catalog, self.history = fixture()

    def test_exact_f32_prefix_and_lossless_typed_pack(self):
        frozen, old_mask = frozen_encode(self.player, self.catalog, 'matchup')
        tensors, mask = encode(self.player, self.catalog, self.history)
        packed, _ = encode_packed(self.player, self.catalog, self.history)
        self.assertEqual(packed.shape, (4224,))
        self.assertEqual(FEATURES, 4224)
        self.assertEqual(packed.dtype, np.float32)
        np.testing.assert_array_equal(packed[:1278], frozen)
        np.testing.assert_array_equal(mask, old_mask)
        unpacked = unpack_packed(np.stack([packed, packed]))
        for key, shape in FIELD_SHAPES.items():
            self.assertEqual(unpacked[key].shape, (2, *shape))
            np.testing.assert_array_equal(unpacked[key][0], tensors[key])

    def test_unknown_absent_and_canonical_ids(self):
        self.assertEqual(category('species', 'Pikachu'), category('species', 'pikachu'))
        self.assertEqual(category('items', None), 0)
        self.assertEqual(category('items', ''), 1)
        self.assertEqual(category('moves', 'unknown future move'), 0)
        tensors, _ = encode(self.player, self.catalog, self.history)
        self.assertEqual(tensors['item'][0], 1)
        self.assertEqual(tensors['item'][6], 0)
        self.assertTrue(np.all(tensors['species'][7:] == 0))
        self.assertEqual(tensors['moves'][6, 1], 0)
        self.assertEqual(tensors['moves'][0, 2], 1)
        self.assertEqual(CATEGORICAL_SIZES['species'], 1588)
        self.assertEqual(CATEGORICAL_SIZES['move'], 941)

    def test_history_order_age_counts_and_padding(self):
        x, _ = encode(self.player, self.catalog, self.history)
        self.assertAlmostEqual(float(x['event_numeric'][0, 0]), .01)
        self.assertEqual(x['event_mask'].sum(), 1)
        self.assertTrue(np.all(x['event_ids'][1:] == 0))
        self.assertAlmostEqual(float(x['public_move_counts'][1, category('moves', 'tackle')]), np.log(5))
        self.assertEqual(x['public_move_counts'][0].sum(), 0)
        empty, _ = encode(self.player, self.catalog, empty_history('p1'))
        self.assertEqual(empty['event_mask'].sum(), 0)
        self.assertEqual(empty['public_move_counts'].sum(), 0)

    def test_explicit_type_marker_precedes_species_fallback_without_changing_prefix(self):
        column = MON_NUMERIC_FIELDS.index('type_known')
        inferred, mask = encode(self.player, self.catalog, self.history)
        self.assertEqual(inferred['mon_numeric'][6, column], 0)
        self.assertEqual(inferred['mon_numeric'][0, column], 1)
        np.testing.assert_array_equal(inferred['types'][6, :2],
                                      [category('types', 'Fire'), category('types', 'Flying')])
        np.testing.assert_array_equal(inferred['base'], frozen_encode(self.player, self.catalog, 'matchup')[0])
        self.player['opponent'][0]['types'] = ['Water']
        explicit, explicit_mask = encode(self.player, self.catalog, self.history)
        self.assertEqual(explicit['mon_numeric'][6, column], 1)
        self.assertEqual(explicit['types'][6, 0], category('types', 'Water'))
        np.testing.assert_array_equal(explicit_mask, mask)
        np.testing.assert_array_equal(explicit['base'], frozen_encode(self.player, self.catalog, 'matchup')[0])

    def test_missing_public_species_metadata_is_rejected_but_unknown_canonical_id_is_supported(self):
        self.player['opponent'][0]['species'] = 'Offcatalog Species'
        for types in (None, ['Fire']):
            self.player['opponent'][0]['types'] = types
            with self.assertRaisesRegex(ValueError, 'Missing required.*public metadata'):
                encode(self.player, self.catalog, self.history)
        self.catalog['species']['offcatalogspecies'] = {'types': ['Fire']}
        self.player['opponent'][0]['types'] = None
        encoded, _ = encode(self.player, self.catalog, self.history)
        self.assertEqual(encoded['species'][6], 0)
        self.assertEqual(encoded['mon_numeric'][6, MON_NUMERIC_FIELDS.index('type_known')], 0)
        self.assertEqual(encoded['types'][6, 0], category('types', 'Fire'))
        np.testing.assert_array_equal(encoded['base'], frozen_encode(self.player, self.catalog, 'matchup')[0])

    def test_history_refuses_other_player_future_invalid_ids_and_hidden_fields(self):
        mutations = [lambda h: h.update(side='p2'), lambda h: h.update(opponent_private={}),
                     lambda h: h['events'][0]['numeric'].__setitem__(0, 8),
                     lambda h: h['events'][0]['ids'].__setitem__(3, 941),
                     lambda h: h['events'][0].update(raw='private request'),
                     lambda h: h['public_move_counts'][1].append(h['public_move_counts'][1][0])]
        for mutate in mutations:
            history = copy.deepcopy(self.history)
            mutate(history)
            with self.assertRaises(ValueError):
                encode(self.player, self.catalog, history)
        self.player['opponent'][0]['stats'] = {'atk': 999}
        with self.assertRaises(ValueError):
            encode(self.player, self.catalog, self.history)

    def test_wait_terminal_mask_and_cutoff_values_are_not_rewritten(self):
        self.player['phase'] = 'wait'
        self.player['active_moves'] = []
        self.player['mask'] = [False] * 14
        x, mask = encode_packed(self.player, self.catalog, self.history)
        self.assertFalse(mask.any())
        self.assertEqual(x[FIELD_SLICES['event_mask']].sum(), 1)
        np.testing.assert_array_equal(x[:1278], frozen_encode(self.player, self.catalog, 'matchup')[0])

    def test_real_worker_complete_battle_and_episode_reset(self):
        from pokemon_worker_watch.transport import Worker
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads((root / 'artifacts/reviews/f17-switch-audit/public-catalog.json').read_text())
        with tempfile.TemporaryDirectory(prefix='f34-features-') as evidence:
            with Worker(evidence, command=('node', 'packages/structured-policy/worker.mjs')) as worker:
                start = dict(episode_index=3, run_seed=1025262699, split='train', tactical_side='p2')
                frame = worker.request('start', **start)
                opening = copy.deepcopy(frame['observable_history'])
                decisions = 0
                while True:
                    for side in ('p1', 'p2'):
                        x, mask = encode_packed(frame['players'][side], catalog, frame['observable_history'][side])
                        np.testing.assert_array_equal(x[:1278], frozen_encode(frame['players'][side], catalog, 'matchup')[0])
                        np.testing.assert_array_equal(mask, frame['players'][side]['mask'])
                    if frame['outcome']:
                        break
                    actions, request_ids = {}, {}
                    for side in ('p1', 'p2'):
                        player = frame['players'][side]
                        if any(player['mask']):
                            actions[side] = frame['tactical_actions'][side] if side == 'p2' else player['mask'].index(True)
                            request_ids[side] = player['request_id']
                    frame = worker.request('step', actions=actions, request_ids=request_ids)
                    decisions += 1
                    self.assertLess(decisions, 500, 'Engineering fixture unexpectedly failed to terminate')
                self.assertEqual(frame['outcome']['kind'], 'terminal')
                self.assertFalse(any(frame['players']['p1']['mask']))
                self.assertGreater(frame['observable_history']['p1']['total_events'], opening['p1']['total_events'])
                reset = worker.request('start', **start)
                self.assertEqual(reset['observable_history'], opening)


if __name__ == '__main__':
    unittest.main()
