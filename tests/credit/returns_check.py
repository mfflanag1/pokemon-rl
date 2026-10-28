"""Direct-run checks: python tests/credit/returns_check.py."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from pokemon_credit.returns import targets


def row(value=.8, reward=0., next_index=-1, bootstrap=0., terminal=False, trajectory=0):
    return dict(value=value, reward=reward, next_index=next_index,
                bootstrap=bootstrap, terminal=terminal, trajectory=trajectory)


class ReturnsCheck(unittest.TestCase):
    def test_hand_computed_terminal(self):
        for outcome in (0., 1.):
            rows = [row(.2, next_index=1), row(.4, next_index=2),
                    row(.6, reward=outcome, terminal=True)]
            for lam in (0., .5, .95, 1.):
                with self.subTest(outcome=outcome, lam=lam):
                    expected = [(1-lam)*.4 + lam*(1-lam)*.6 + lam**2*outcome,
                                (1-lam)*.6 + lam*outcome, outcome]
                    result = targets(rows, lam)
                    np.testing.assert_allclose(result["returns"], expected)
                    np.testing.assert_allclose(result["advantages"], np.array(expected)-[.2,.4,.6])
                    np.testing.assert_array_equal(result["terminal_chain"], [True]*3)
                    np.testing.assert_array_equal(result["distance_to_end"], [2,1,0])

    def test_distant_loss_biased_critic(self):
        rows = [row(next_index=i+1) for i in range(19)] + [row(terminal=True)]
        for lam in (0., .5, .95, 1.):
            result = targets(rows, lam)
            np.testing.assert_allclose(result["returns"],
                                       [.8*(1-lam**d) for d in range(19,-1,-1)], atol=1e-15)

    def test_interleaved_and_shared_label_chains(self):
        rows = [row(next_index=2), row(next_index=3, trajectory=1),
                row(reward=1., terminal=True), row(terminal=True, trajectory=1),
                row(next_index=5), row(terminal=True)]
        result = targets(rows, 1.)
        np.testing.assert_array_equal(result["returns"], [1,0,1,0,0,0])
        np.testing.assert_array_equal(result["distance_to_end"], [1,1,0,0,1,0])

    def test_cutoff_and_no_mutation(self):
        rows = [row(.2, next_index=1), row(.4, bootstrap=.3),
                row(reward=1., terminal=True)]
        before = copy.deepcopy(rows)
        for lam in (0., .5, .95, 1.):
            result = targets(rows, lam)
            np.testing.assert_allclose(result["returns"], [(1-lam)*.4+lam*.3,.3,1])
            np.testing.assert_array_equal(result["terminal_chain"], [False,False,True])
            np.testing.assert_array_equal(result["distance_to_end"], [1,0,0])
        self.assertEqual(rows, before)

    def test_bad_rows(self):
        bad = [None, {}, row(terminal=1), row(terminal="False"),
               row(next_index=True), row(next_index=1.0), row(next_index=-2),
               row(next_index=1), row(next_index=0), row(trajectory=-1),
               row(trajectory=True), row(trajectory=0.0), row(reward=.1),
               row(terminal=True, bootstrap=.1)]
        for field in ("value", "reward", "bootstrap"):
            for value in (float("nan"), float("inf"), -float("inf"), -.1, 1.1, "0", [0]):
                bad.append(row(**{field: value}))
        for invalid in bad:
            with self.subTest(row=invalid), self.assertRaises(ValueError):
                targets([invalid], .95)

    def test_bad_links(self):
        cases = [
            [row(next_index=1), row(trajectory=1)],
            [row(next_index=2), row(next_index=2), row()],
            [row(), row(next_index=0)],
            [row(next_index=1, terminal=True), row()],
            [row(next_index=1, bootstrap=.1), row()],
            [row(next_index=1, reward=.1), row()],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                targets(rows, .95)

    def test_lambda_validation_and_numpy_types(self):
        for lam in (True, np.bool_(False), float("nan"), float("inf"), -.1, 1.1,
                    "0.5", [0.5], np.array(.5), 1j):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                targets([], lam)
        result = targets([row(value=np.float64(.2), terminal=np.bool_(True),
                              trajectory=np.int64(0), next_index=np.int32(-1))], np.float32(.5))
        np.testing.assert_allclose(result["advantages"], [-.2])
        targets([], 0)
        targets([], 1)

    def test_empty_and_dtypes(self):
        for rows in ([], [row()]):
            result = targets(rows, .95)
            self.assertEqual(set(result), {"returns", "advantages", "terminal_chain", "distance_to_end"})
            for name, dtype in (("returns", np.float64), ("advantages", np.float64),
                                ("terminal_chain", np.bool_), ("distance_to_end", np.int64)):
                self.assertEqual(result[name].dtype, np.dtype(dtype))
                self.assertEqual(result[name].shape, (len(rows),))


if __name__ == "__main__":
    unittest.main()
