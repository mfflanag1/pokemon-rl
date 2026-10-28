"""F36 transport overlay and orchestration wiring; no games or learner updates."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType
from unittest import mock

from pokemon_teacher_runtime import stage

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_script('f36_checker_test', 'scripts/check-teacher-runtime.py')


class OverlayTests(unittest.TestCase):
    def test_only_six_preserved_stages_are_allowed(self):
        self.assertEqual(len(stage.STAGES), 6)
        with mock.patch.object(stage, 'install_overlay') as install:
            for args in ([], ['os'], ['pokemon_teacher_assisted.preflight']):
                with self.assertRaises(ValueError):
                    stage.main(args)
            install.assert_not_called()

    def test_overlay_precedes_execution_and_argv_is_restored(self):
        old = sys.argv
        events = []
        def run(module, **kwargs):
            events.append(('run', module, list(sys.argv), kwargs))
            raise RuntimeError('synthetic stage failure')
        with mock.patch.object(stage, 'install_overlay', side_effect=lambda: events.append(('install',))), \
                mock.patch.object(stage.runpy, 'run_module', side_effect=run):
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                stage.main(['--', stage.STAGES[0], '--output', 'engineering-only'])
        self.assertEqual(events[0], ('install',))
        self.assertEqual(events[1][1:3], (stage.STAGES[0], [stage.STAGES[0], '--output', 'engineering-only']))
        self.assertIs(sys.argv, old)

    def test_preimported_consumers_are_rejected_before_mutation(self):
        for name in stage.CONSUMERS:
            with mock.patch.dict(sys.modules, {name: object()}):
                with self.assertRaisesRegex(RuntimeError, 'precede stage imports'):
                    stage.install_overlay()

    def test_other_loaded_old_class_alias_is_rejected_without_mutation(self):
        from pokemon_external_benchmark import transport as original
        old_class = original.JsonWorker
        stale = ModuleType('synthetic_stale_importer')
        stale.Alias = old_class
        with mock.patch.dict(sys.modules, {'synthetic_stale_importer': stale}):
            with self.assertRaisesRegex(RuntimeError, 'synthetic_stale_importer.Alias'):
                stage.install_overlay()
        self.assertIs(original.JsonWorker, old_class)

    def test_late_stale_binding_rolls_back_overlay(self):
        from pokemon_external_benchmark import transport as original
        old_class = original.JsonWorker
        with mock.patch.object(stage, '_stale_references', side_effect=[[], ['late_importer.Alias']]):
            with self.assertRaisesRegex(RuntimeError, 'after overlay'):
                stage.install_overlay()
        self.assertIs(original.JsonWorker, old_class)

    def test_actual_external_and_verifier_bind_new_class_local_worker_unchanged(self):
        code = '''
from pokemon_teacher_runtime.stage import install_overlay
from pokemon_worker_watch import transport as local
original_local = local.Worker
metadata = install_overlay()
from pokemon_teacher_runtime.transport import JsonWorker
from pokemon_teacher_assisted import external, verification
from pokemon_external_benchmark import transport
assert external.JsonWorker is verification.JsonWorker is transport.JsonWorker is JsonWorker
assert local.Worker is original_local
assert metadata['implementation'] == 'pokemon_teacher_runtime.transport.JsonWorker'
print('both external consumers use F36 JsonWorker; local Worker unchanged')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT / 'python'), 'JAX_PLATFORMS': 'cpu'},
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('both external consumers', result.stdout)


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_script('f36_driver_test', 'scripts/run-teacher-runtime.py')

    def test_driver_and_checker_bind_the_actual_cli_plan(self):
        original = {'schema_version': 'f35-teacher-assisted-plan-v1', 'seeds': [7103, 8209, 9311]}
        admitted_plan = {**original, 'execution': {'schema_version': 'f36-teacher-runtime-execution-v1'}}
        for module in (self.driver, checker):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                original_path = root / 'contracts/teacher-assisted-v1/plan.json'
                plan_path = root / module.PLAN_RELATIVE
                original_path.parent.mkdir(parents=True)
                plan_path.parent.mkdir(parents=True)
                original_path.write_text(json.dumps(original))
                plan_path.write_text(json.dumps(admitted_plan))
                digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
                manifest = {'plan': copy.deepcopy(admitted_plan), 'source_files': {module.PLAN_RELATIVE: digest}}
                with mock.patch.object(module, 'ROOT', root):
                    module.verify_plan_binding(manifest)
                    missing = copy.deepcopy(manifest)
                    missing['source_files'].clear()
                    with self.assertRaisesRegex(ValueError, 'bound by admission'):
                        module.verify_plan_binding(missing)
                    plan_path.write_text(json.dumps({**admitted_plan, 'unadmitted_change': True}))
                    with self.assertRaisesRegex(ValueError, 'bound by admission'):
                        module.verify_plan_binding(manifest)
                    manifest['source_files'][module.PLAN_RELATIVE] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(ValueError, 'differs from admission'):
                        module.verify_plan_binding(manifest)
                    manifest['plan']['unadmitted_change'] = True
                    with self.assertRaisesRegex(ValueError, 'scientific plan'):
                        module.verify_plan_binding(manifest)

    def test_initial_capture_failure_still_cleans_and_records_exit(self):
        driver = self.driver
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            child = mock.Mock(pid=424242, returncode=None)
            child.poll.side_effect = lambda: child.returncode
            def terminate(process, **kwargs):
                self.assertIs(process, child)
                self.assertIsNone(kwargs['snapshot'])
                child.returncode = 143
                return {'status': 'passed', 'root_pid': child.pid, 'returncode': 143,
                    'live_descendants_at_cleanup': [], 'remaining_pids': [], 'errors': []}
            with mock.patch.object(driver, 'REVIEW', review), \
                    mock.patch.object(driver, 'before_deadline'), \
                    mock.patch.object(driver.subprocess, 'Popen', return_value=child), \
                    mock.patch.object(driver, 'capture_tree', side_effect=RuntimeError('synthetic capture failure')), \
                    mock.patch.object(driver, 'terminate_tree', side_effect=terminate) as cleanup:
                with self.assertRaises(RuntimeError):
                    driver.stage('synthetic', ['-m', stage.STAGES[0]], review / 'report.json')
            cleanup.assert_called_once()
            result = json.loads((review / 'synthetic-exit.json').read_text())
            self.assertEqual(result['exit_code'], 143)
            self.assertIn('synthetic capture failure', result['wait_error'])
            self.assertTrue((review / 'synthetic-cleanup.json').is_file())

    def test_capture_and_helper_failure_fall_back_to_bounded_direct_child_cleanup(self):
        driver = self.driver
        for stubborn in (False, True):
            with self.subTest(stubborn=stubborn), tempfile.TemporaryDirectory() as temporary:
                review = Path(temporary)
                child = mock.Mock(pid=424245, returncode=None)
                child.poll.side_effect = lambda: child.returncode
                def terminate():
                    if not stubborn:
                        child.returncode = 143
                def kill():
                    child.returncode = -9
                def wait(timeout):
                    self.assertEqual(timeout, 3)
                    if child.returncode is None:
                        raise subprocess.TimeoutExpired('synthetic', timeout)
                    return child.returncode
                child.terminate.side_effect = terminate
                child.kill.side_effect = kill
                child.wait.side_effect = wait
                with mock.patch.object(driver, 'REVIEW', review), mock.patch.object(driver, 'before_deadline'), \
                        mock.patch.object(driver.subprocess, 'Popen', return_value=child), \
                        mock.patch.object(driver, 'capture_tree', side_effect=RuntimeError('primary capture error')), \
                        mock.patch.object(driver, 'terminate_tree', side_effect=RuntimeError('secondary helper error')):
                    with self.assertRaises(RuntimeError):
                        driver.stage('synthetic', ['-m', stage.STAGES[0]], review / 'report.json')
                child.terminate.assert_called_once()
                self.assertEqual(child.kill.call_count, int(stubborn))
                result = json.loads((review / 'synthetic-exit.json').read_text())
                cleanup = json.loads((review / 'synthetic-cleanup.json').read_text())
                self.assertIn('primary capture error', result['wait_error'])
                self.assertEqual(result['cleanup_status'], 'failed')
                self.assertIn('secondary helper error', cleanup['errors'][0])
                self.assertFalse(cleanup['direct_child_cleanup']['direct_child_remaining'])
                self.assertFalse(cleanup['direct_child_cleanup']['descendant_cleanup_verified'])
                self.assertTrue(cleanup['descendant_cleanup_unresolved'])

    def test_unresolved_cleanup_never_passes_stage(self):
        driver = self.driver
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            report = review / 'report.json'
            report.write_text('{"status":"passed"}')
            child = mock.Mock(pid=424243, returncode=0)
            child.poll.return_value = 0
            cleanup = {'status': 'failed', 'root_pid': child.pid, 'returncode': 0,
                'live_descendants_at_cleanup': [], 'remaining_pids': [424244],
                'errors': ['unknown descendant identity']}
            with mock.patch.object(driver, 'REVIEW', review), mock.patch.object(driver, 'before_deadline'), \
                    mock.patch.object(driver.subprocess, 'Popen', return_value=child), \
                    mock.patch.object(driver, 'capture_tree', return_value={}), \
                    mock.patch.object(driver, 'terminate_tree', return_value=cleanup):
                with self.assertRaises(RuntimeError):
                    driver.stage('synthetic', ['-m', stage.STAGES[0]], report)
            self.assertEqual(json.loads((review / 'synthetic-exit.json').read_text())['cleanup_status'], 'failed')


class CheckerTests(unittest.TestCase):
    def fixture(self):
        execution = {'schema_version': 'f36-teacher-runtime-execution-v1',
            **{k: {'path': p, 'sha256': str(i) * 64}
               for i, (k, p) in enumerate(checker.BINDINGS.items())}}
        receipt = {'schema_version': 'f36-teacher-runtime-receipt-v1',
            'output_paths': copy.deepcopy(checker.OUTPUT_PATHS),
            'summary': checker.OUTPUT_PATHS['results'] + '/summary.json',
            'execution': execution, 'evidence_files': {v['path']: v['sha256'] for v in execution.values() if isinstance(v, dict)},
            'stages': [{'schema_version': 'f36-stage-v1', 'stage': name, 'command': command}
                       for name, command in checker.expected_commands().items()]}
        manifest = {'plan': {'execution': copy.deepcopy(execution)}}
        completion = {'schema_version': 'f36-completion-v1', 'receipt': 'artifacts/reports/f36-teacher-runtime.json'}
        workflow = {**completion, 'schema_version': 'f36-workflow-exit-v1'}
        driver = {'schema_version': 'f36-driver-exit-v1', 'exit_code': 0}
        return receipt, manifest, completion, workflow, driver

    def test_f36_metadata_is_required_and_preserved_checker_is_reused(self):
        data = self.fixture()
        checker.validate_runtime(*data)
        self.assertIs(checker.validate_completion, checker._frozen.validate_completion)
        for target, key, value in [(0, 'schema_version', 'f35-teacher-assisted-receipt-v1'),
                (0, 'summary', 'artifacts/reviews/f35-teacher-assisted/measured-results-v1/summary.json'),
                (4, 'exit_code', 1)]:
            changed = copy.deepcopy(data)
            changed[target][key] = value
            with self.assertRaises(ValueError):
                checker.validate_runtime(*changed)

    def test_unknown_cleanup_fields_cannot_hide_behind_passed_status(self):
        cleanup = {'status': 'passed', 'unresolved_unknown_pids': [], 'reused_pids': [],
            'handle_signals': [], 'unknown_overflow': 0, 'table_error': None}
        checker.validate_runtime_cleanup(cleanup)
        for key, value in [('unresolved_unknown_pids', [42]), ('reused_pids', [43]),
                ('handle_signals', [{'pid': 44}]), ('unknown_overflow', 1), ('table_error', 'read failure')]:
            with self.subTest(key=key):
                invalid = {**cleanup, key: value}
                with self.assertRaises(ValueError):
                    checker.validate_runtime_cleanup(invalid)
                invalid = dict(cleanup)
                invalid.pop(key)
                with self.assertRaises(ValueError):
                    checker.validate_runtime_cleanup(invalid)

    def test_runtime_qualification_source_hashes_must_match_admission(self):
        data = self.fixture()
        receipt, manifest = data[:2]
        manifest['source_files'] = {p: 'a' * 64 for p in checker.RUNTIME_SOURCES}
        qualification = {'status': 'passed', 'source_files': copy.deepcopy(manifest['source_files'])}
        expected = manifest['plan']['execution']['runtime_qualification']['sha256']
        with mock.patch.dict(checker.BINDINGS, {'runtime_qualification': checker.BINDINGS['runtime_qualification']}, clear=True), \
                mock.patch.object(checker, 'sha', return_value=expected), \
                mock.patch.object(Path, 'read_text', side_effect=lambda *a, **k: json.dumps(qualification)):
            checker.verify_runtime_evidence(receipt, manifest)
            qualification['source_files'][next(iter(checker.RUNTIME_SOURCES))] = 'b' * 64
            with self.assertRaisesRegex(ValueError, 'qualification'):
                checker.verify_runtime_evidence(receipt, manifest)
            qualification.pop('source_files')
            with self.assertRaisesRegex(ValueError, 'qualification'):
                checker.verify_runtime_evidence(receipt, manifest)

    def test_wrong_overlay_path_or_unbound_failure_and_qualification_are_rejected(self):
        for mutate in (
                lambda d: d[0]['stages'][0]['command'].__setitem__(2, 'pokemon_teacher_assisted.lifecycle'),
                lambda d: d[0]['stages'][1]['command'].__setitem__(-1, 'artifacts/runs/f35-teacher-data-v1'),
                lambda d: d[0]['evidence_files'].pop(checker.BINDINGS['failed_predecessor']),
                lambda d: d[0]['evidence_files'].pop(checker.BINDINGS['runtime_qualification'])):
            changed = self.fixture()
            mutate(changed)
            with self.assertRaises(ValueError):
                checker.validate_runtime(*changed)


if __name__ == '__main__':
    unittest.main()
