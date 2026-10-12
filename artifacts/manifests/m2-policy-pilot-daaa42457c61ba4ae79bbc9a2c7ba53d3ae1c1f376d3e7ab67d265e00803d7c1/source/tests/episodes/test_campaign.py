from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest

from pokemon_reference.contract import CONTRACT
from pokemon_reference.corpus import atomic_json, read_json, sealed
from pokemon_reference.episode_campaign import load_context, inspect_episode, validate_campaign

ROOT = CONTRACT.parents[1]


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='r4-python-campaign-')
        cls.directory = Path(cls.temporary.name) / 'campaign'
        subprocess.run(['node', 'packages/episodes/cli.mjs', 'campaign', '--out', str(cls.directory),
                        '--namespace', 'r4-python-acceptance', '--count', '2', '--workers', '2'],
                       cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cls.context = load_context(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_saved_episodes_independently_validate(self):
        report = validate_campaign(self.directory, Path(self.temporary.name) / 'validation', workers=1)
        self.assertEqual(report['episodes'], 2)
        self.assertEqual(report['totals']['decisions'], report['totals']['transitions'])
        self.assertEqual(report['node_replay_receipts_verified'], 2)
        self.assertEqual(sum(report['winners'].values()), 2)

    def test_summary_is_recomputed_from_episode_evidence(self):
        record = deepcopy(self.context['index']['episodes'][0])
        record['summary']['turns'] += 1
        with self.assertRaisesRegex(ValueError, 'summary disagrees'):
            inspect_episode(record, self.context)

    def test_missing_duplicate_index_rejected_even_when_resealed(self):
        path = self.directory / 'index.json'
        original = read_json(path)
        broken = deepcopy(original)
        broken['episodes'][1] = broken['episodes'][0]
        broken.pop('content_hash')
        try:
            atomic_json(path, sealed(broken))
            with self.assertRaisesRegex(ValueError, 'accounting'):
                load_context(self.directory)
        finally:
            atomic_json(path, original)

    def test_compressed_corruption_rejected(self):
        record = self.context['index']['episodes'][0]
        path = self.directory / record['file']
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'x')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                inspect_episode(record, self.context)
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
