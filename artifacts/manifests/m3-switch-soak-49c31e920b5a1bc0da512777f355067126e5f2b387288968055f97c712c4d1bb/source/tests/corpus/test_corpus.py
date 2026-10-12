from copy import deepcopy
import gzip
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from pokemon_reference.contract import canonical, parse_json, make_artifact, hash_json, CONTRACT
from pokemon_reference.corpus import load_context, inspect_chunk, inventory, coverage_report, digest, sealed

ROOT = CONTRACT.parents[1]


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='pokemon-r3-python-')
        cls.corpus = Path(cls.temporary.name) / 'corpus'
        subprocess.run(['node', 'packages/corpus/cli.mjs', 'generate', '--out', str(cls.corpus),
                        '--namespace', 'r3-python-test', '--count', '12', '--shard-size', '4'],
                       cwd=ROOT, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy_corpus(self, target):
        destination = Path(self.temporary.name) / target
        shutil.copytree(self.corpus, destination)
        return destination

    def mutate_first(self, destination, change):
        context = load_context(destination)
        index = context['index']
        chunk = index['chunks'][0]
        filename = destination / chunk['file']
        lines = gzip.decompress(filename.read_bytes()).splitlines()
        artifact = parse_json(lines[0])
        change(artifact['payload'])
        artifact = make_artifact('team', artifact['manifest_id'], artifact['payload'])
        lines[0] = canonical(artifact).encode('utf-8')
        plain = b'\n'.join(lines) + b'\n'
        compressed = gzip.compress(plain, mtime=0)
        filename.write_bytes(compressed)
        chunk.update(bytes=len(compressed), sha256=digest(compressed), jsonl_bytes=len(plain), jsonl_sha256=digest(plain))
        (destination / 'index.json').write_text(canonical(sealed({k: v for k, v in index.items() if k != 'content_hash'})))

    def test_independent_validation_and_parallel_inventory_agree(self):
        context = load_context(self.corpus)
        self.assertEqual(context['index']['completed'], 12)
        single, parallel = [Path(self.temporary.name) / name for name in ['single', 'parallel']]
        first = inventory(self.corpus, single, workers=1)
        second = inventory(self.corpus, parallel, workers=2)
        self.assertEqual(first['teams'], 12)
        self.assertEqual(first['pokemon'], 72)
        self.assertTrue(second['seed_order_verified'])
        frequencies = parse_json((single / 'frequencies.json').read_bytes())
        self.assertEqual(first['move_slots'], sum(frequencies['moves'].values()))
        self.assertEqual(sum(frequencies['display_forms'].values()), 72)
        self.assertEqual(sum(frequencies['lead_source_species'].values()), 12)
        self.assertTrue(all(n > 0 for n in frequencies['lead_source_species'].values()))
        for name in ['frequencies.json', 'coverage.json', 'joint-sets.jsonl.gz', 'team-counts.jsonl.gz']:
            self.assertEqual(first['files'][name], second['files'][name], name)
        rows = [parse_json(line) for line in gzip.decompress((single / 'joint-sets.jsonl.gz').read_bytes()).splitlines()]
        self.assertEqual(sum(row['count'] for row in rows), 72)
        self.assertTrue(all(hash_json(row['configuration']) == row['set_key'] for row in rows))
        with self.assertRaises(FileExistsError):
            inventory(self.corpus, single)

    def test_tampered_seed_rejected_even_when_all_hashes_are_recomputed(self):
        destination = self.copy_corpus('wrong-seed')
        self.mutate_first(destination, lambda payload: payload['generator_seed'].update(hex='f' * 64))
        context = load_context(destination)
        with self.assertRaisesRegex(ValueError, 'Seed order'):
            inspect_chunk(context['index']['chunks'][0], context)

    def test_wrong_resolved_id_rejected_even_with_valid_envelope(self):
        destination = self.copy_corpus('wrong-id')
        self.mutate_first(destination, lambda payload: payload['ids'][0].update(item='invaliditem'))
        context = load_context(destination)
        with self.assertRaisesRegex(ValueError, 'identifier mismatch'):
            inspect_chunk(context['index']['chunks'][0], context)

    def test_missing_or_reordered_shards_rejected(self):
        destination = self.copy_corpus('missing-shard')
        index = load_context(destination)['index']
        index['chunks'].pop(1)
        (destination / 'index.json').write_text(canonical(sealed({k: v for k, v in index.items() if k != 'content_hash'})))
        with self.assertRaisesRegex(ValueError, 'shard'):
            load_context(destination)

    def test_corrupt_compression_and_catalog_rejected(self):
        destination = self.copy_corpus('corrupt')
        context = load_context(destination)
        chunk = context['index']['chunks'][0]
        (destination / chunk['file']).write_bytes(b'incomplete')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            inspect_chunk(chunk, context)
        (destination / 'catalog.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_context(destination)

    def test_coverage_is_source_conditioned_and_missing_is_not_unreachable(self):
        context = load_context(self.corpus)
        observed = inspect_chunk(context['index']['chunks'][0], context)
        coverage = coverage_report(context['catalog'], observed['frequencies'])
        sources = coverage['comparisons']['source_species']
        self.assertEqual(sources['declared'], 509)
        self.assertLess(sources['observed_declared'], 509)
        self.assertGreater(len(sources['unobserved_declared']), 0)
        self.assertEqual(coverage['comparisons']['source_templates']['declared'], 879)
        self.assertIn('not proof of unreachability', coverage['interpretation'])


if __name__ == '__main__':
    unittest.main()
