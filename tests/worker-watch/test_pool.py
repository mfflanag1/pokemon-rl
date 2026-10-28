"""Cleanup must preserve the request error and visit every resource."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace
from pokemon_worker_watch.collector import Pool

class Cleanup(unittest.TestCase):
    def pool(self,path):
        pool=object.__new__(Pool);pool.closed=False;pool.evidence=Path(path)
        pool.executor=Mock();pool.workers=[Mock(),Mock(),Mock()]
        pool.workers[0].close.side_effect=RuntimeError('first cleanup failed')
        pool.feedback_logs=[Mock(closed=False),Mock(closed=False)]
        return pool
    def test_all_resources_attempted_and_normal_close_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            pool=self.pool(d)
            with self.assertRaisesRegex(RuntimeError,'Worker pool cleanup'):pool.close()
            for worker in pool.workers:worker.close.assert_called_once()
            for log in pool.feedback_logs:log.close.assert_called_once()
            self.assertTrue((Path(d)/'pool-cleanup-errors.json').exists())
            pool.close()
            for worker in pool.workers:worker.close.assert_called_once()
    def test_cleanup_does_not_replace_primary_request_error(self):
        with tempfile.TemporaryDirectory() as d:
            pool=self.pool(d);root=ValueError('request root')
            try:
                try:raise root
                except ValueError:
                    pool.close();raise
            except ValueError as exc:
                self.assertIs(exc,root)
                self.assertTrue(any('cleanup' in note for note in exc.__notes__))
            else:self.fail('primary error lost')
            for worker in pool.workers:worker.close.assert_called_once()
            for log in pool.feedback_logs:log.close.assert_called_once()

    def test_failure_evidence_error_does_not_replace_request_error(self):
        with tempfile.TemporaryDirectory() as d:
            pool=self.pool(Path(d)/'missing-parent')
            pool.workers=[SimpleNamespace(close=Mock())]
            root=TimeoutError('original worker deadline')
            pool._requests=Mock(side_effect=root)
            episode=SimpleNamespace(split='train',start={})
            with patch('pokemon_worker_watch.collector.AuditedEpisode',return_value=episode):
                with self.assertRaises(TimeoutError) as caught:pool.collect({},[{}])
            self.assertIs(caught.exception,root)
            self.assertTrue(any('failure evidence' in note for note in root.__notes__))
            pool.workers[0].close.assert_called_once()

if __name__=='__main__':unittest.main()
