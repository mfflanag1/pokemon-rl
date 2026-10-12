import copy
from pathlib import Path
import unittest

from pokemon_reference.contract import hash_json, parse_json
from pokemon_reference.mechanics import validate_manifest, validate_bundle, SupportGate, UnsupportedMechanic

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'artifacts/runs/r6-mechanics-v4'


def reseal(value):
    value['content_hash'] = hash_json({k: v for k, v in value.items() if k != 'content_hash'})
    return value


class MechanicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = parse_json((RUN / 'manifest.json').read_bytes())

    def mutate(self, edit, pattern):
        m = copy.deepcopy(self.manifest)
        edit(m)
        with self.assertRaisesRegex(ValueError, pattern):
            validate_manifest(reseal(m), ROOT)

    def test_independent_source_and_bundle_validation(self):
        result = validate_bundle(RUN, ROOT)
        self.assertEqual(result['reference_cases_verified'], 16)
        self.assertEqual(result['custom_engine_gate'], 'blocked-unsupported')

    def test_unsupported_unknown_and_empty_fail_loudly(self):
        gate = SupportGate(self.manifest)
        for keys, message in [(['moves:sleeptalk'], 'ENG-MOVE'), (['moves:invented'], 'Unknown'), ([], 'Empty')]:
            with self.assertRaisesRegex(UnsupportedMechanic, message):
                gate.require(keys)
        with self.assertRaises(UnsupportedMechanic):
            gate.require_profile()

    def test_support_cannot_be_granted_by_editing_json(self):
        self.mutate(lambda m: m['entries'][0].update(support='implemented-tested'), 'cannot certify')

    def test_initial_evidence_overclaim_is_rejected(self):
        def edit(m):
            next(e for e in m['entries'] if not e['observed_count'])['initial_class'] = 'observed-initial'
        self.mutate(edit, 'Initial evidence classification')

    def test_source_fragment_mutation_is_detected_after_resealing(self):
        self.mutate(lambda m: m['entries'][0]['source'].update(fragment_sha256='0' * 64), 'fragment mismatch')

    def test_omitted_initial_entry_is_rejected_after_resealing(self):
        self.mutate(lambda m: m['entries'].remove(next(e for e in m['entries'] if e['key'] == 'items:')), 'Initial corpus coverage')

    def test_unknown_dynamic_kind_is_rejected_after_resealing(self):
        self.mutate(lambda m: m['pathways'][0].update(kind='unclassified-future-call'), 'Unknown pathway')

    def test_narrowed_bound_is_rejected_after_resealing(self):
        self.mutate(lambda m: m['pathways'][0].update(bound='only-one-move'), 'Incorrect pathway bound')

    def test_orphan_path_is_rejected_after_resealing(self):
        def remove_link(m):
            next(e for e in m['entries'] if e['pathways'])['pathways'].pop()
        self.mutate(remove_link, 'Orphan pathway')

    def test_version_and_reachability_overclaim_fail(self):
        self.mutate(lambda m: m.update(schema_version='r6-mechanics-2'), 'Unknown mechanics artifact version')
        self.mutate(lambda m: m['scope'].update(minimal_reachability_proven=True), 'reachability claim')


if __name__ == '__main__':
    unittest.main()
