import logging
import unittest
import numpy as np
import jax
from pokemon_fixed.campaign import Aggregate, CompileLog, admission, compile_positive_control


class CampaignTests(unittest.TestCase):
    def test_predecessor_and_final_reservation(self):
        a=admission()
        self.assertTrue(a['prior_gates']['final_reservation_unchanged'])
        self.assertEqual(a['plan']['soak_seconds'],7200)

    def test_bounded_aggregation_counts_and_quantiles(self):
        aggregate=Aggregate()
        metrics={k:np.ones((2,4),np.float32) for k in
                 ('loss','actor_loss','value_loss','entropy','approx_kl','clip_fraction','gradient_norm','max_ratio_error')}
        metrics['valid_count']=np.array([[128,128,50,0],[128,128,50,0]])
        report={'decisions':306,'executed_moves':258,'minimum_behavior_probability':.2,'metrics':metrics}
        for _ in range(2000):aggregate.add(report,.002)
        result=aggregate.report()
        self.assertEqual(result['cycles'],2000)
        self.assertEqual(result['decisions'],612000)
        self.assertEqual(result['games'],128000)
        self.assertAlmostEqual(result['complete_cycle_seconds'],4)
        self.assertEqual(len(result['latency_histogram']['counts']),258)
        self.assertEqual(sum(result['latency_histogram']['counts']),2000)
        self.assertLess(.002,result['latency_quantile_upper_bounds_seconds']['0.99'])
        self.assertLess(result['latency_quantile_upper_bounds_seconds']['0.99'],.0022)
        self.assertEqual(result['diagnostics']['loss']['valid_weighted_mean'],1)

    def test_compile_logging_counts_actual_compilation(self):
        log=CompileLog()
        logger=logging.getLogger('jax._src.dispatch');logger.addHandler(log)
        enabled=jax.config.jax_log_compiles
        try:
            jax.config.update('jax_log_compiles',True)
            result=compile_positive_control(log)
            self.assertGreaterEqual(result['first_call_compilation_events'],1)
            self.assertEqual(result['same_shape_repeat_events'],0)
        finally:
            logger.removeHandler(log);jax.config.update('jax_log_compiles',enabled)


if __name__=='__main__':unittest.main()
