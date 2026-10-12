import copy
import gzip
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from pokemon_reference.contract import hash_json, canonical
from pokemon_reference.corpus import read_json, sealed
from pokemon_reference.dataset_identity import partition
from pokemon_reference.dataset_gate import LeakageLedger, checked_file, confined, ActorDataset, POLICY

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = read_json(ROOT / 'tests/dataset/fixtures/team.json')['payload']

class GateTest(unittest.TestCase):
    def test_full_digest_boundary_in_python(self):
        space = 1 << 256; dev = (space * 9000 + 9999) // 10000; final = (space * 9500 + 9999) // 10000
        for n, expected in [(0,'train'),(dev-1,'train'),(dev,'development'),(final-1,'development'),(final,'final'),(space-1,'final')]:
            with patch('pokemon_reference.dataset_identity.hash_json', return_value=f'{n:064x}'):
                self.assertEqual(partition(PAYLOAD, POLICY), expected)

    def test_shared_teams_seeds_episodes_and_mirrors_fail_across_splits(self):
        for kind in ['team', 'seed', 'episode', 'mirror-group']:
            ledger = LeakageLedger(); ledger.add(kind, 'same', 'train', 'pair')
            ledger.add(kind, 'same', 'train', 'pair')
            for split in ['development', 'final']:
                with self.assertRaisesRegex(ValueError, 'Leakage'): ledger.add(kind, 'same', split, 'pair')
        ledger = LeakageLedger(); ledger.add('seed', 'seed', 'train', 'pair', 'mechanics')
        with self.assertRaisesRegex(ValueError, 'Leakage'): ledger.add('seed', 'seed', 'train', 'pair', 'policy')

    def test_boundary_rejects_final_and_unknown_purpose_before_opening_files(self):
        for purpose in ['final', 'evaluation', 'anything']:
            with self.assertRaisesRegex(ValueError, 'sealed'):
                ActorDataset('/nonexistent', expected_hash='0'*64, purpose=purpose)

    def test_changed_bytes_and_path_escape_reject(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root/'file').write_text('different')
            with self.assertRaisesRegex(ValueError, 'changed'): checked_file(root,'file','0'*64)
            with self.assertRaisesRegex(ValueError, 'escapes'): confined(root,'../outside')

    def test_untrusted_resealed_dataset_hash_rejects(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); doc=sealed({'version':'r7-dataset-1','status':'validated','policy_hash':hash_json(POLICY)})
            (root/'dataset.json').write_text(canonical(doc))
            with self.assertRaisesRegex(ValueError, 'Untrusted dataset'):
                ActorDataset(root, expected_hash='0'*64, purpose='training')
