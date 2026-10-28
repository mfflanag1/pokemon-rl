"""Real lifecycle-wrapper coverage and exact-exception negatives; no scientific games."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pokemon_online_verification import guard, verification
from pokemon_online_continuation import verification as original

ROOT = Path(__file__).resolve().parents[1]


class GuardTests(unittest.TestCase):
    def run_probe(self, wrapped):
        command = [sys.executable, '-m']
        if wrapped:
            command += ['pokemon_teacher_runtime.lifecycle', '--']
        command += ['pokemon_online_verification.verification', '--probe']
        return subprocess.run(command, cwd=ROOT, env={**os.environ,
            'PYTHONPATH': str(ROOT / 'python'), 'PYTHONDONTWRITEBYTECODE': '1',
            'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            capture_output=True, text=True, timeout=45, check=True)

    def test_actual_direct_and_lifecycle_entrypoints(self):
        direct = json.loads(self.run_probe(False).stdout)
        wrapped = json.loads(self.run_probe(True).stdout)
        self.assertEqual(direct['inventory']['forbidden'], [])
        self.assertEqual(wrapped['inventory']['forbidden'], [])
        self.assertIn('pokemon_teacher_runtime', [m['name'] for m in wrapped['inventory']['allowed_runtime_modules']])
        self.assertEqual(wrapped['inventory']['original_admission_sha256'], guard.ADMISSION_SHA256)

    def test_exact_names_only_reject_all_near_misses(self):
        for name in ['pokemon_teacher_runtime.stage', 'pokemon_teacher_runtime.transport',
                     'pokemon_teacher_runtime.qualification', 'pokemon_teacher_runtime.lifecycle.child',
                     'pokemon_teacher_legality', 'fp', 'poke_engine', 'foul_play.search',
                     'pokemon_teacher_assisted.demonstrations']:
            with self.subTest(name=name), patch.dict(sys.modules, {name: ModuleType(name)}):
                self.assertIn(name, guard.snapshot()['forbidden'])
                with self.assertRaises(ValueError):
                    verification.probe()

    def test_imported_exact_lifecycle_is_source_bound(self):
        importlib.import_module('pokemon_teacher_runtime.lifecycle')
        allowed = {m['name']:m for m in guard.snapshot()['allowed_runtime_modules']}
        self.assertEqual(set(allowed), set(guard.RUNTIME_EXCEPTIONS))
        self.assertTrue(all(len(item['sha256']) == 64 for item in allowed.values()))

    def test_wrong_file_or_origin_is_rejected(self):
        for attr in ['__file__', '__spec__']:
            module = ModuleType('pokemon_teacher_runtime')
            expected = str(ROOT / guard.RUNTIME_EXCEPTIONS[module.__name__])
            module.__file__ = expected
            module.__spec__ = SimpleNamespace(origin=expected)
            setattr(module, attr, '/tmp/unadmitted.py' if attr == '__file__' else SimpleNamespace(origin='/tmp/unadmitted.py'))
            with self.subTest(attr=attr), patch.dict(sys.modules, {module.__name__:module}):
                with self.assertRaisesRegex(ValueError, 'origin mismatch'):
                    guard.snapshot()

    def test_missing_origin_is_rejected(self):
        module = ModuleType('pokemon_teacher_runtime')
        with patch.dict(sys.modules, {module.__name__:module}):
            with self.assertRaisesRegex(ValueError, 'origin mismatch'):
                guard.snapshot()

    def test_admission_or_exception_byte_drift_rejected(self):
        real_digest = guard.digest
        importlib.import_module('pokemon_teacher_runtime.lifecycle')
        for suffix in ['manifest.json', 'lifecycle.py']:
            def changed(path):
                return '0'*64 if str(path).endswith(suffix) else real_digest(path)
            with self.subTest(suffix=suffix), patch.object(guard, 'digest', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'drift|source mismatch'):
                    guard.snapshot()

    def test_guard_restored_on_verifier_failure(self):
        before = original.teacher_search_modules
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(original, 'run', side_effect=ValueError('synthetic failure')):
                with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                    verification.run('unused', Path(temp) / 'output')
        self.assertIs(original.teacher_search_modules, before)

    def test_existing_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            marker = output / 'original.json'
            marker.write_text('original bytes')
            with self.assertRaises(FileExistsError):
                verification.run(ROOT / 'artifacts/runs/f39-online-continuation-v1', output)
            self.assertEqual([p.name for p in output.iterdir()], ['original.json'])
            self.assertEqual(marker.read_text(), 'original bytes')

    def test_success_records_both_guard_snapshots_and_restores_original(self):
        before = original.teacher_search_modules
        def synthetic_run(training_run, output):
            Path(output).mkdir()
            self.assertEqual(original.teacher_search_modules(), [])
            self.assertEqual(original.teacher_search_modules(), [])
            return {'status': 'passed', 'counts': {}}
        with tempfile.TemporaryDirectory() as temp, patch.object(original, 'run', side_effect=synthetic_run):
            output = Path(temp) / 'output'
            result = verification.run('unused', output)
            saved = json.loads((output / 'report.json').read_text())
            self.assertEqual(saved, result)
            self.assertEqual(len(saved['f40_runtime_policy']['snapshots']), 2)
            self.assertFalse(saved['f40_runtime_policy']['production_repeated'])
        self.assertIs(original.teacher_search_modules, before)

    def test_fixed_forbidden_prefixes_retained(self):
        self.assertEqual(guard.FORBIDDEN_MODULE_PREFIXES, original.FORBIDDEN_MODULE_PREFIXES)
        self.assertEqual(set(guard.RUNTIME_EXCEPTIONS), {'pokemon_teacher_runtime','pokemon_teacher_runtime.lifecycle'})


if __name__ == '__main__':
    unittest.main()
