"""Run directly with Python; six focused behavioral metric checks."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from pokemon_entropy.behavior import describe


class BehaviorCheck(unittest.TestCase):
    def test_switch_opportunities(self):
        p = np.zeros((5, 14))
        m = np.zeros_like(p, dtype=bool)
        m[0, [0, 8]] = True
        p[0, [0, 8]] = [.75, .25]
        m[1, [1, 9]] = True
        p[1, [1, 9]] = [.25, .75]
        m[2, 8] = True
        p[2, 8] = 1
        m[3:, [0, 8]] = True
        p[3:, 8] = 1
        result = describe(p, m, ["move", "move", "move", "replace", "revival"])
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["move_rows"], 3)
        self.assertEqual(result["replacement_rows"], 2)
        self.assertEqual(result["switch_opportunity_rows"], 2)
        self.assertEqual(result["mean_switch_probability"], .5)
        json.dumps(result, allow_nan=False)

    def test_entropy_legal_count(self):
        p = np.zeros((2, 14))
        p[:, :2] = .5
        m = np.zeros_like(p, dtype=bool)
        m[0, :2] = True
        m[1, :4] = True
        result = describe(p, m, ["move", "move"])
        self.assertAlmostEqual(result["mean_entropy"], np.log(2))
        self.assertAlmostEqual(result["mean_normalized_entropy"], .75)

    def test_single_action(self):
        p = np.zeros((1, 14), dtype=int)
        p[0, 3] = 1
        result = describe(p, p.astype(bool), ["move"])
        self.assertEqual(result["mean_entropy"], 0)
        self.assertEqual(result["mean_normalized_entropy"], 0)
        self.assertIsNone(result["mean_switch_probability"])

    def test_empty_and_no_opportunities(self):
        result = describe(np.empty((0, 14)), np.empty((0, 14), bool), [])
        self.assertEqual(result, dict(rows=0, move_rows=0,
                         switch_opportunity_rows=0, replacement_rows=0,
                         mean_switch_probability=None, mean_entropy=None,
                         mean_normalized_entropy=None))
        p = [[0.] * 8 + [1.] + [0.] * 5]
        result = describe(p, np.asarray(p, bool), ("revival",))
        self.assertEqual(result["replacement_rows"], 1)
        self.assertIsNone(result["mean_switch_probability"])

    def test_validation(self):
        p = np.zeros((1, 14))
        p[0, 0] = 1
        m = p.astype(bool)
        cases = [(p.ravel(), m, ["move"]), (p[:, :13], m, ["move"]),
                 (p * .9, m, ["move"]), (p, ~m, ["move"]),
                 (p + np.nan, m, ["move"]), (p.astype(bool), m, ["move"]),
                 (p.astype(complex), m, ["move"]),
                 (p.astype(str), m, ["move"]), (p.astype(object), m, ["move"]),
                 (p, m.astype(int), ["move"]), (p, m, ["wait"]),
                 (p, m, []), (p, m, "move"), (p, m, [1]),
                 (p, np.zeros_like(m), ["move"]),
                 (p, m[:, :13], ["move"]), (p + np.inf, m, ["move"])]
        illegal = p.copy()
        illegal[0, 1] = 1e-12
        cases.append((illegal, m, ["move"]))
        negative = p.copy()
        negative[0, 1] = -1e-12
        cases.append((negative, np.ones_like(m), ["move"]))
        too_large = p.copy()
        too_large[0, 0] += 1e-12
        cases.append((too_large, m, ["move"]))
        for args in cases:
            with self.subTest(dtype=np.asarray(args[0]).dtype, phases=args[2]):
                with self.assertRaises(ValueError):
                    describe(*args)
        accepted = p.copy()
        accepted[0, 0] -= 5e-7
        describe(accepted, m, ["move"])
        rejected = p.copy()
        rejected[0, 0] -= 2e-6
        with self.assertRaises(ValueError):
            describe(rejected, m, ["move"])

    def test_no_mutation(self):
        p = np.zeros((2, 14))
        p[:, [0, 8]] = [.4, .6]
        m = p.astype(bool)
        phases = ["move", "replace"]
        before_p, before_m, before_phases = p.copy(), m.copy(), phases.copy()
        p.setflags(write=False)
        m.setflags(write=False)
        describe(p, m, phases)
        np.testing.assert_array_equal(p, before_p)
        np.testing.assert_array_equal(m, before_m)
        self.assertEqual(phases, before_phases)


if __name__ == "__main__":
    unittest.main()
