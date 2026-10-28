"""Run directly with python3 -B tests/switch-audit/coverage_check.py."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from pokemon_switch_audit.coverage import summarize


def record(target=0, phase='move', game=0, own='A', foe='B', dest='C', hash_id=0):
    return dict(game=game, phase=phase, target=target, own_species=own,
                foe_species=foe, destination_species=dest if target >= 8 else '',
                feature_hash=f'{hash_id:064x}')


class CoverageTests(unittest.TestCase):
    def test_split_and_family_totals(self):
        rows = [record(0), record(0), record(4, game=1), record(8),
                record(9, 'replace', game=2)]
        summary = summarize(rows, [record()])['train']
        self.assertEqual((summary['rows'], summary['games']), (5, 3))
        families = summary['families']
        self.assertEqual([families[n]['rows'] for n in families], [2, 1, 1, 1])
        normal = families['normal_move']
        self.assertEqual(normal, dict(rows=2, games=1, target_counts={'0': 2},
            own_species_counts={'A': 2}, foe_species_counts={'B': 2},
            destination_species_counts={}, unique_own_foe_pairs=1,
            unique_destination_foe_pairs=0))
        self.assertEqual(families['forced_switch']['destination_species_counts'], {'C': 1})

    def test_same_family_separation(self):
        train = [record(0), record(8)]
        probe = [record(4), record(8, 'replace')]
        families = summarize(train, probe)['probe']['families']
        self.assertEqual(families['tera_move']['seen_own_foe_rows'], 0)
        self.assertEqual(families['forced_switch']['seen_destination_foe_rows'], 0)
        self.assertEqual(families['forced_switch']['unseen_destination_foe_rows'], 1)

    def test_pair_counts(self):
        train = [record(8)]
        probe = [record(8), record(9), record(10, own='X', dest='D'),
                 record(11, foe='Z'), record(0)]
        families = summarize(train, probe)['probe']['families']
        switch = families['voluntary_switch']
        self.assertEqual([switch[k] for k in ('seen_own_foe_rows', 'unseen_own_foe_rows',
            'seen_destination_foe_rows', 'unseen_destination_foe_rows')], [2, 2, 2, 2])
        self.assertEqual(families['normal_move']['unseen_own_foe_rows'], 1)
        self.assertEqual(families['normal_move']['unseen_destination_foe_rows'], 0)

    def test_hash_conflicts(self):
        train = [record(0), record(1), record(2, hash_id=1),
                 record(3, hash_id=2), record(3, hash_id=2)]
        probe = [record(0), record(2, hash_id=1), record(3, hash_id=1),
                 record(4, hash_id=3), record(4, hash_id=3)]
        result = summarize(train, probe)
        for split in ('train', 'probe'):
            self.assertEqual([result[split][k] for k in ('unique_feature_hashes',
                'repeated_hash_groups', 'conflicting_label_groups',
                'rows_in_conflicting_groups')], [3, 2, 1, 2])
        self.assertEqual(result['overlap'], dict(shared_feature_hashes=2,
            shared_feature_rows_in_probe=3, conflicting_shared_hashes=2))
        self.assertEqual(summarize([record()], [record(1)])['overlap']['conflicting_shared_hashes'], 1)

    def test_phase_and_action_classification(self):
        rows = [record(3), record(4), record(7), record(8), record(13),
                record(8, 'replace'), record(13, 'revival')]
        families = summarize(rows, rows)['train']['families']
        self.assertEqual([families[n]['rows'] for n in families], [1, 2, 2, 2])

    def test_validation(self):
        invalid = [None, (), [], [None]]
        for field, values in {
            'game': [True, -1, 1.0, '0'], 'target': [False, -1, 14, 0.0],
            'phase': [None, 'switch', 0], 'own_species': ['', None, 3],
            'foe_species': ['', False], 'destination_species': ['C', None],
            'feature_hash': ['a' * 63, 'A' * 64, 'g' * 64, '0' * 64 + '\n', 0],
        }.items():
            for value in values:
                row = record()
                row[field] = value
                invalid.append([row])
        missing = record()
        del missing['game']
        extra = record()
        extra['extra'] = 1
        invalid.extend([[missing], [extra], [record(0, 'replace')],
                        [dict(record(8), destination_species='')]])
        for rows in invalid:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    summarize(rows, [record()])
                with self.assertRaises(ValueError):
                    summarize([record()], rows)

    def test_nonmutation_and_json(self):
        train, probe = [record(), record(8)], [record(8, 'revival')]
        before = copy.deepcopy((train, probe))
        result = summarize(train, probe)
        self.assertEqual((train, probe), before)
        self.assertEqual(json.loads(json.dumps(result)), result)
        result['train']['families']['normal_move']['target_counts']['0'] = 99
        self.assertEqual((train, probe), before)

    def test_permutation_invariance(self):
        train = [record(0), record(1), record(8, game=1, own='Z'), record(4)]
        probe = [record(9), record(8, 'replace'), record(0, own='X')]
        expected = summarize(train, probe)
        self.assertEqual(summarize(list(reversed(train)), probe[1:] + probe[:1]), expected)


if __name__ == '__main__':
    unittest.main()
