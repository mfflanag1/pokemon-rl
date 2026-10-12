import copy
import json
from pathlib import Path
import unittest

from pokemon_reference.contract import hash_json
from pokemon_reference.dataset_identity import canonical_team, team_key, partition, pair_partition

ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT / 'contracts/dataset-v1/policy.json').read_text())
PAYLOAD = json.loads((ROOT / 'tests/dataset/fixtures/team.json').read_text())['payload']


class IdentityTest(unittest.TestCase):
    def test_metadata_and_seed_do_not_change_group(self):
        p = copy.deepcopy(PAYLOAD)
        p['generator_seed']['hex'] = '0' * 64
        for mon in p['pokemon']:
            mon['name'] = 'A new nickname'; mon['role'] = 'different bookkeeping'; mon['shiny'] = not mon['shiny']
            mon['speciesId'] = 'bookkeeping'
        p['source_team_hash'] = hash_json(p['pokemon'])
        self.assertEqual(team_key(p, POLICY), team_key(PAYLOAD, POLICY))
        self.assertEqual(partition(p, POLICY), partition(PAYLOAD, POLICY))

    def test_each_mechanical_field_and_order_changes_identity(self):
        original = team_key(PAYLOAD, POLICY)
        edits = [lambda p: p['pokemon'][0].update(level=1), lambda p: p['pokemon'][0].update(gender=''),
                 lambda p: p['pokemon'][0]['evs'].update(hp=1), lambda p: p['pokemon'][0]['ivs'].update(hp=1),
                 lambda p: p['ids'][0].update(species='pikachu'), lambda p: p['ids'][0].update(ability='pressure'),
                 lambda p: p['ids'][0].update(item=''), lambda p: p['ids'][0].update(tera_type='normal'),
                 lambda p: p['ids'][1]['moves'].reverse(), lambda p: (p['pokemon'].reverse(), p['ids'].reverse())]
        for edit in edits:
            p = copy.deepcopy(PAYLOAD); edit(p)
            self.assertNotEqual(team_key(p, POLICY), original)

    def test_full_digest_exact_threshold_assignment(self):
        c = canonical_team(PAYLOAD)
        digest = int(hash_json([POLICY['hash_version'], POLICY['salt'], c]), 16)
        expected = 'train' if digest * 10000 < 9000 * 2**256 else 'development' if digest * 10000 < 9500 * 2**256 else 'final'
        self.assertEqual(partition(PAYLOAD, POLICY), expected)
        self.assertEqual(pair_partition(PAYLOAD, PAYLOAD, POLICY), expected)

    def test_mixed_partition_is_rejected(self):
        variants = {}
        for level in range(1, 101):
            p = copy.deepcopy(PAYLOAD); p['pokemon'][0]['level'] = level
            variants.setdefault(partition(p, POLICY), p)
        self.assertGreater(len(variants), 1)
        a, b = list(variants.values())[:2]
        with self.assertRaises(ValueError): pair_partition(a, b, POLICY)

    def test_unknown_mechanical_fields_versions_and_policy_changes_reject(self):
        p = copy.deepcopy(PAYLOAD); p['pokemon'][0]['nature'] = 'Adamant'
        with self.assertRaises(ValueError): canonical_team(p)
        for field,value in [('identity_version','future'),('hash_version','future'),('salt',''),('denominator',0)]:
            policy = copy.deepcopy(POLICY); policy[field] = value
            with self.assertRaises(ValueError): partition(PAYLOAD, policy)
        policy = copy.deepcopy(POLICY); policy['ranges'][0]['upper']=9001
        with self.assertRaises(ValueError): partition(PAYLOAD,policy)
