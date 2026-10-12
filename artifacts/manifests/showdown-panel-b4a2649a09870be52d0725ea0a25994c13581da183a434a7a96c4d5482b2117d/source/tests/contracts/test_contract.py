from copy import deepcopy
from pathlib import Path
import unittest

from pokemon_reference.contract import (
    parse_json, canonical, hash_json, validate_artifact, validate_schema,
    load_player_view, make_artifact, action_from_index, action_index,
    action_mask, admissible_actions, vocabulary_index, CONTRACT,
)

GOLDEN = parse_json((CONTRACT.parents[1] / 'tests/fixtures/contract-golden.json').read_bytes())


class ContractTests(unittest.TestCase):
    def test_canonical_vectors(self):
        for vector in GOLDEN['canonical_vectors']:
            value = parse_json(vector['wire'])
            self.assertEqual(canonical(value), vector['canonical'])
            self.assertEqual(hash_json(value), vector['sha256'])

    def test_bad_wire(self):
        for wire in GOLDEN['invalid_wire']:
            with self.subTest(wire=wire), self.assertRaises(Exception):
                parse_json(wire)
        with self.assertRaises(Exception):
            parse_json(b'"\xff"')
        with self.assertRaises(Exception):
            canonical(float('nan'))
        with self.assertRaises(Exception):
            parse_json('[' * 101 + '0' + ']' * 101)

    def test_golden_artifacts_and_roundtrip(self):
        for artifact in GOLDEN['artifacts']:
            validate_artifact(artifact)
            self.assertEqual(parse_json(canonical(artifact)), artifact)
            rebuilt = make_artifact(artifact['artifact_kind'], artifact['manifest_id'], artifact['payload'])
            self.assertEqual(rebuilt, artifact)

    def test_actor_boundary(self):
        for artifact in GOLDEN['artifacts']:
            if artifact['artifact_kind'] == 'player_view':
                view = load_player_view(canonical(artifact), artifact['manifest_id'])
                self.assertNotIn('mechanics_seed', view)
                with self.assertRaises(Exception):
                    load_player_view(canonical(artifact), '0' * 64)
            else:
                with self.assertRaises(Exception):
                    load_player_view(canonical(artifact))

    def test_action_indices_and_request_cases(self):
        for index in range(14):
            self.assertEqual(action_index(action_from_index(index)), index)
        for invalid in [-1, 14, True, '0', 1.0]:
            with self.assertRaises(Exception):
                action_from_index(invalid)
        with self.assertRaises(Exception):
            action_index({'kind': 'pass'})
        for case in GOLDEN['cases']:
            self.assertEqual(admissible_actions(case['request']), case['actions'], case['name'])
            self.assertEqual(action_mask(case['request']), case['mask'], case['name'])
        struggle = next(c for c in GOLDEN['cases'] if c['name'] == 'struggle')
        self.assertEqual(struggle['actions'], [{'kind': 'move', 'slot': 0, 'terastallize': False}])
        self.assertEqual(struggle['mask'], [True] + [False] * 13)

    def test_unknown_zero_and_vocab(self):
        validate_schema({'state': 'known', 'value': 0}, 'known_integer')
        validate_schema({'state': 'unknown'}, 'known_integer')
        for value in [0, None, {'state': 'unknown', 'value': 0}, {'state': 'known'}, {'state': 'known', 'value': True}]:
            with self.assertRaises(Exception):
                validate_schema(value, 'known_integer')
        vocabulary = parse_json((CONTRACT / 'vocabulary.json').read_bytes())
        self.assertEqual(vocabulary_index(vocabulary, 'items', {'state': 'unknown'}), 0)
        self.assertEqual(vocabulary_index(vocabulary, 'items', {'state': 'known', 'value': ''}), 1)
        self.assertGreater(vocabulary_index(vocabulary, 'items', {'state': 'known', 'value': 'leftovers'}), 1)
        with self.assertRaises(Exception):
            vocabulary_index(vocabulary, 'items', {'state': 'known', 'value': 'nonexistentitem'})

    def test_schema_and_integrity_rejections(self):
        view = next(a for a in GOLDEN['artifacts'] if a['artifact_kind'] == 'player_view')
        for key, value in [('schema_version', '2.0.0'), ('schema_version', '1.1.0'), ('content_hash', '0' * 64), ('artifact_id', 'player_view:' + '0' * 64)]:
            changed = deepcopy(view)
            changed[key] = value
            with self.assertRaises(Exception):
                validate_artifact(changed)
        for key, value in [('viewer', 'p2'), ('action_mask', [False] * 14), ('opponent_private_state', {'state': 'unknown', 'value': 0})]:
            payload = deepcopy(view['payload'])
            payload[key] = value
            with self.assertRaises(Exception):
                make_artifact('player_view', view['manifest_id'], payload)
        for action in [{'kind': 'move', 'slot': True, 'terastallize': False}, {'kind': 'switch', 'slot': 6}, {'kind': 'move', 'slot': 0}, {'kind': 'wait'}, {'kind': 'pass', 'slot': 0}]:
            with self.assertRaises(Exception):
                validate_schema(action, 'action')
        with self.assertRaises(Exception):
            validate_schema({'kind': 'showdown-sodium', 'hex': 9007199254740993}, 'seed')


if __name__ == '__main__':
    unittest.main()
