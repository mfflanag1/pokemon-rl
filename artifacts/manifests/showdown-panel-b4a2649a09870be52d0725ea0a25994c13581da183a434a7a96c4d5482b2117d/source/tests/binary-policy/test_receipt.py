"""The completion gate must reject a claimed soak outside the policy's support."""
from pathlib import Path
import runpy
import unittest


class ReceiptTests(unittest.TestCase):
    def test_probability_and_ratio_diagnostic_gates(self):
        module=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/seal-binary-policy.py'))
        check=module['validate_soak_diagnostics']
        count=module['node_pass_count']
        for prefix in ('#','ℹ'):
            self.assertEqual(count(f'{prefix} pass 63\n{prefix} fail 0\n'),63)
            with self.assertRaises(ValueError):count(f'{prefix} pass 63\n{prefix} fail 1\n')
        with self.assertRaises(ValueError):count('no test summary')
        good={'minimum_exp_behavior_logp':2**-24,'maximum_update_ratio_error':.21}
        check(good)
        for value in (0.,2**-25,1.1,float('nan'),float('inf')):
            with self.assertRaisesRegex(ValueError,'Behavior probability'):
                check(dict(good,minimum_exp_behavior_logp=value))
        for value in (-1.,2**25,float('nan'),float('inf')):
            with self.assertRaisesRegex(ValueError,'Update ratio'):
                check(dict(good,maximum_update_ratio_error=value))

if __name__=='__main__':unittest.main()
