"""Mocked F39 workflow boundaries; no process, battle, or scientific launch."""
from contextlib import ExitStack
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


driver = load_script('f39_driver_tests', 'run-online-continuation.py')
launcher = load_script('f39_launcher_tests', 'launch-online-continuation.py')


class DriverTests(unittest.TestCase):
    def stage_fixture(self, review, *, code=0, report='passed'):
        child = mock.Mock(pid=424242, returncode=code)
        child.poll.side_effect = lambda: child.returncode
        if report is not None:
            write(review / 'report.json', {'status': report})
        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, value in [('REVIEW', review), ('STOP', 100.), ('deadline', mock.Mock()),
                              ('capture_tree', mock.Mock(return_value={'owned': []}))]:
            stack.enter_context(mock.patch.object(driver, target, value))
        spawn = stack.enter_context(mock.patch.object(driver.subprocess, 'Popen', return_value=child))
        def cleanup(*_args, **_kwargs):
            if child.returncode is None:
                child.returncode = 143
            return {'status': 'passed', 'root_pid': child.pid, 'returncode': child.returncode,
                'live_descendants_at_cleanup': [], 'remaining_pids': [], 'errors': []}
        terminate = stack.enter_context(mock.patch.object(driver, 'terminate_tree', side_effect=cleanup))
        return child, spawn, terminate

    def test_missing_or_failed_report_and_nonzero_exit_rejected(self):
        for status, code in [(None, 0), ('failed', 0), ('passed', 17)]:
            with self.subTest(status=status, code=code), tempfile.TemporaryDirectory() as temp:
                review = Path(temp)
                self.stage_fixture(review, code=code, report=status)
                with self.assertRaises(RuntimeError):
                    driver.stage('synthetic', ['-m', 'pokemon_online_continuation.campaign'], review / 'report.json')
                record = json.loads((review / 'synthetic-exit.json').read_text())
                self.assertEqual(record['exit_code'], code)
                self.assertEqual(record['cleanup_status'], 'passed')

    def test_success_uses_only_generic_lifecycle_and_requested_f39_module(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            _, spawn, _ = self.stage_fixture(review)
            result = driver.stage('synthetic', ['-m', 'pokemon_online_continuation.campaign', '--run', 'dummy'], review / 'report.json')
            command = spawn.call_args.args[0]
            self.assertEqual(command[1:], ['-m', 'pokemon_teacher_runtime.lifecycle', '--',
                'pokemon_online_continuation.campaign', '--run', 'dummy'])
            self.assertFalse(any(x in command for x in ['pokemon_teacher_runtime.stage',
                'pokemon_teacher_legality.stage', 'pokemon_teacher_assisted.demonstrations', 'foul_play']))
            self.assertEqual(result['exit_code'], 0)

    def test_initial_capture_failure_still_cleans_owned_child_and_records_error(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            child, _, terminate = self.stage_fixture(review, code=None)
            with mock.patch.object(driver, 'capture_tree', side_effect=RuntimeError('synthetic capture error')):
                with self.assertRaises(RuntimeError):
                    driver.stage('synthetic', ['-m', 'pokemon_online_continuation.campaign'], review / 'report.json')
            self.assertEqual(child.returncode, 143)
            terminate.assert_called_once()
            record = json.loads((review / 'synthetic-exit.json').read_text())
            self.assertIn('synthetic capture error', record['wait_error'])

    def test_cleanup_exception_uses_bounded_direct_child_fallback_and_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            child, _, _ = self.stage_fixture(review, code=None)
            child.wait.side_effect = [subprocess.TimeoutExpired('synthetic', 3), None]
            child.kill.side_effect = lambda: setattr(child, 'returncode', -9)
            with mock.patch.object(driver.time, 'time', return_value=100.), \
                    mock.patch.object(driver, 'terminate_tree', side_effect=RuntimeError('synthetic cleanup error')):
                with self.assertRaises(RuntimeError):
                    driver.stage('synthetic', ['-m', 'pokemon_online_continuation.campaign'], review / 'report.json')
            self.assertEqual(child.wait.call_args_list, [mock.call(timeout=3), mock.call(timeout=3)])
            child.terminate.assert_called_once()
            child.kill.assert_called_once()
            result = json.loads((review / 'synthetic-cleanup.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['descendant_cleanup_unresolved'])
            self.assertFalse(result['direct_child_cleanup']['direct_child_remaining'])
            self.assertFalse(result['direct_child_cleanup']['descendant_cleanup_verified'])

    def test_stop_boundary_prevents_wait_and_records_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            child, _, _ = self.stage_fixture(review, code=None)
            with mock.patch.object(driver.time, 'time', return_value=100.):
                with self.assertRaises(RuntimeError):
                    driver.stage('synthetic', ['-m', 'pokemon_online_continuation.campaign'], review / 'report.json')
            child.wait.assert_not_called()
            self.assertTrue(json.loads((review / 'synthetic-exit.json').read_text())['deadline_timeout'])

    def test_real_fixed_deadline_guard_precedes_process_creation(self):
        stop = datetime.fromisoformat(driver.PLAN['deadline_utc'].replace('Z', '+00:00')).timestamp()
        self.assertEqual(driver.STOP, stop)
        with mock.patch.object(driver.time, 'time', return_value=stop), \
                mock.patch.object(driver.subprocess, 'Popen') as spawn:
            with self.assertRaises(TimeoutError):
                driver.stage('synthetic', ['-m', 'pokemon_online_continuation.campaign'], Path('unused'))
            spawn.assert_not_called()

    def test_exclusive_workflow_claim_blocks_any_second_stage_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            write(review / 'workflow-claim.json', {'previous': True})
            with mock.patch.object(driver, 'REVIEW', review), mock.patch.object(driver, 'deadline'), \
                    mock.patch.object(driver, 'stage') as stage:
                with self.assertRaises(FileExistsError):
                    driver.main()
                stage.assert_not_called()
            self.assertEqual(json.loads((review / 'workflow-claim.json').read_text()), {'previous': True})

    def test_exclusive_launcher_claim_blocks_duplicate_driver(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            write(review / 'launcher-claim.json', {'previous': True})
            with mock.patch.object(launcher, 'REVIEW', review), mock.patch.object(launcher.time, 'time', return_value=0), mock.patch.object(launcher.subprocess, 'Popen') as spawn:
                with self.assertRaises(FileExistsError):
                    launcher.main()
                spawn.assert_not_called()


    def test_launcher_reserved_evidence_conflict_fails_before_spawn_and_records_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            write(review / 'driver-launch.json', {'previous': True})
            with mock.patch.object(launcher, 'REVIEW', review), mock.patch.object(launcher.time, 'time', return_value=0), \
                    mock.patch.object(launcher.subprocess, 'Popen') as spawn:
                self.assertEqual(launcher.main(), 1)
                spawn.assert_not_called()
            record = json.loads((review / 'driver-exit.json').read_text())
            self.assertIn('FileExistsError', record['error'])
            self.assertIsNone(record['cleanup'])
            self.assertEqual(json.loads((review / 'driver-launch.json').read_text()), {'previous': True})

    def test_launcher_failure_after_spawn_always_cleans_owned_child(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            child = mock.Mock(pid=424242, returncode=None)
            child.poll.side_effect = lambda: child.returncode
            def terminate(*_args, **_kwargs):
                child.returncode = 143
                return {'status': 'passed', 'live_descendants_at_cleanup': []}
            with mock.patch.object(launcher, 'REVIEW', review), mock.patch.object(launcher.time, 'time', return_value=0), \
                    mock.patch.object(launcher.subprocess, 'Popen', return_value=child), \
                    mock.patch.object(launcher.os, 'fsync', side_effect=OSError('synthetic launch evidence failure')), \
                    mock.patch.object(launcher, 'terminate_tree', side_effect=terminate) as cleanup:
                self.assertEqual(launcher.main(), 1)
            cleanup.assert_called_once_with(child, snapshot=None, grace_seconds=10, kill_wait_seconds=5)
            self.assertEqual(child.returncode, 143)
            record = json.loads((review / 'driver-exit.json').read_text())
            self.assertIn('synthetic launch evidence failure', record['error'])
            self.assertEqual(record['exit_code'], 1)

    def test_launcher_outer_deadline_cleans_without_unbounded_wait(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            child = mock.Mock(pid=424242, returncode=None)
            child.poll.side_effect = lambda: child.returncode
            stop = driver.STOP
            def terminate(*_args, **_kwargs):
                child.returncode = 143
                return {'status': 'passed', 'live_descendants_at_cleanup': []}
            with mock.patch.object(launcher, 'REVIEW', review), \
                    mock.patch.object(launcher.time, 'time', side_effect=[0, stop + 45, stop + 46, stop + 46]), \
                    mock.patch.object(launcher.subprocess, 'Popen', return_value=child), \
                    mock.patch.object(launcher, 'capture_tree', return_value={'owned': []}), \
                    mock.patch.object(launcher, 'terminate_tree', side_effect=terminate):
                self.assertEqual(launcher.main(), 1)
            child.wait.assert_not_called()
            self.assertEqual(child.returncode, 143)
            self.assertIn('cutoff plus cleanup grace', json.loads((review / 'driver-exit.json').read_text())['error'])

    def test_launcher_start_at_cutoff_creates_no_claim_or_child(self):
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            with mock.patch.object(launcher, 'REVIEW', review), \
                    mock.patch.object(launcher.time, 'time', return_value=driver.STOP), \
                    mock.patch.object(launcher.subprocess, 'Popen') as spawn:
                with self.assertRaises(TimeoutError):
                    launcher.main()
                spawn.assert_not_called()
            self.assertFalse(list(review.iterdir()))


if __name__ == '__main__':
    unittest.main()
