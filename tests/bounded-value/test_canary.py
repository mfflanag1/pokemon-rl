"""Copied-evidence corruption checks against a real, tiny three-seed trajectory."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from pokemon_bounded import canary, learner
from pokemon_learner.checkpoint import digest, wire


class CanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='m4-unit-', dir=learner.ROOT/'artifacts/runs')
        cls.out = Path(cls.tmp.name)/'run'
        cls.report = canary.run(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rejected(self, mutate, message, rehash=True):
        report = copy.deepcopy(self.report)
        mutate(report)
        if rehash:
            report['content_hash'] = digest({k:v for k,v in report.items() if k!='content_hash'})
        path = self.out/'report.json'
        try:
            path.write_bytes(wire(report)+b'\n')
            with self.assertRaisesRegex(ValueError, message):
                canary.audit(self.out)
        finally:
            path.write_bytes(wire(self.report)+b'\n')

    def test_complete_trace_and_endpoint_replay(self):
        audit = canary.audit(self.out)
        self.assertEqual(audit['replayed_updates'], 24)
        self.assertEqual(audit['training_games'], 768)
        self.assertEqual(audit['checkpoint_pairs'], 6)
        self.assertEqual(audit['terminal_games']+audit['truncated_games'],768)

    def test_omitted_seed_rejected_even_with_fresh_digest(self):
        self.rejected(lambda r:r['records'].pop(), 'Seed roster')

    def test_changed_report_digest(self):
        self.rejected(lambda r:r.update(training_games=769), 'Report digest', rehash=False)

    def test_changed_trace_with_recomputed_digest(self):
        self.rejected(lambda r:r['records'][0]['trace'][0].update(value_max=.99), 'Full update trace')

    def test_false_original_next_metric(self):
        self.rejected(lambda r:r['records'][0]['initial']['original_next'].update(decisions=-1), 'Original next-update')

    def test_false_total(self):
        self.rejected(lambda r:r.update(training_decisions=1), 'Counter accounting')

    def test_false_scope(self):
        self.rejected(lambda r:r.update(strength='qualified'), 'Scope flag')

    def test_admission_drift(self):
        path=self.out/'admission.json';original=path.read_bytes()
        try:
            admission=json.loads(original);admission['runtime']['backend']='gpu'
            path.write_bytes(wire(admission))
            with self.assertRaisesRegex(ValueError,'Admission binding'):
                canary.audit(self.out)
        finally:path.write_bytes(original)


if __name__ == '__main__':unittest.main()
