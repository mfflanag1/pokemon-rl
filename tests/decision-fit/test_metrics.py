"""Hand-computable finite-task metric and gate checks (no task data)."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from pokemon_fit.metrics import gates, summarize


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.contexts = np.repeat(np.arange(16), 2)
        self.targets = np.tile([0, 1], 16)
        self.probabilities = np.eye(14)[self.targets]

    def test_perfect(self):
        summary = summarize(self.probabilities, self.targets, self.contexts)
        expected_context = {"n": 2, "correct": 2, "accuracy": 1.0,
                            "target_probability": 1.0}
        self.assertEqual(summary, {
            "n": 32, "correct": 32, "accuracy": 1.0,
            "mean_target_probability": 1.0,
            "min_context_target_probability": 1.0,
            "contexts": {str(c): expected_context for c in range(16)},
        })
        json.dumps(summary, allow_nan=False)

    def test_uniform_and_smallest_index_ties(self):
        summary = summarize(np.full((32, 14), 1 / 14), self.targets, self.contexts)
        self.assertEqual(summary["correct"], 16)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["mean_target_probability"], 1 / 14)
        self.assertAlmostEqual(summary["min_context_target_probability"], 1 / 14)
        for context in summary["contexts"].values():
            self.assertEqual(context["correct"], 1)
            self.assertAlmostEqual(context["target_probability"], 1 / 14)

    def test_row_and_context_permutations(self):
        p = self.probabilities.copy()
        scores = 0.5 + self.contexts / 64
        p[np.arange(32), self.targets] = scores
        p[:, 13] = 1 - scores
        baseline = summarize(p, self.targets, self.contexts)
        order = np.random.default_rng(37).permutation(32)
        self.assertEqual(baseline, summarize(p[order], self.targets[order],
                                           self.contexts[order]))
        renamed = summarize(p, self.targets, 15 - self.contexts)
        for key in baseline.keys() - {"contexts"}:
            self.assertEqual(baseline[key], renamed[key])
        for c in range(16):
            self.assertEqual(baseline["contexts"][str(c)],
                             renamed["contexts"][str(15 - c)])

    def test_invalid_inputs(self):
        p, t, c = self.probabilities, self.targets, self.contexts
        invalid = [
            (p.ravel(), t, c), (p[:, :13], t, c), (p[:, :, None], t, c),
            (p.astype(str), t, c), (p.astype(complex), t, c),
            (p.astype(bool), t, c), (p, t[:, None], c), (p, t[:-1], c),
            (p, t.astype(float), c), (p, t.astype(bool), c),
            (p, t.astype(str), c), (p, np.full(32, -1), c),
            (p, np.full(32, 14), c), (p, np.full(32, 2**64 - 1, dtype=np.uint64), c),
            (p, t, c[:, None]), (p, t, c[:-1]), (p, t, c.astype(float)),
            (p, t, c.astype(bool)), (p, t, np.full(32, -1)),
            (p, t, np.full(32, 16)), (p, t, np.zeros(32, dtype=int)),
            (p, t, np.concatenate(([1], c[1:]))),
            (p[:0], t[:0], c[:0]), (p * 0.9, t, c),
        ]
        for value in (np.nan, np.inf, -0.01):
            malformed = p.copy()
            malformed[0, 0] = value
            invalid.append((malformed, t, c))
        for inputs in invalid:
            with self.subTest(shapes=[x.shape for x in inputs]):
                with self.assertRaises(ValueError):
                    summarize(*inputs)

    def test_normalization_tolerance(self):
        p = self.probabilities.copy()
        p[:, 0] += 4e-6
        summarize(p, self.targets, self.contexts)
        p[:, 0] += 2e-6
        with self.assertRaises(ValueError):
            summarize(p, self.targets, self.contexts)

    def test_context_gates_override_passing_aggregates(self):
        p = self.probabilities.copy()
        p[:2] = np.eye(14)[[2, 2]]
        summary = summarize(p, self.targets, self.contexts)
        self.assertGreater(summary["accuracy"], 0.9)
        self.assertGreater(summary["mean_target_probability"], 0.9)
        perfect = summarize(self.probabilities, self.targets, self.contexts)
        summaries = {"fit_train": summary, "probe_same": perfect,
                     "probe_variant": perfect}
        plan = {"thresholds": {
            "fit_train": {"minimum_context_accuracy": 0.9,
                          "minimum_context_target_probability": 0.9},
            "probe_same": {"minimum_context_accuracy": 1.0,
                           "minimum_context_target_probability": 1.0},
            "probe_variant": {"minimum_context_accuracy": 0.0,
                              "minimum_context_target_probability": 0.0},
        }}
        result = gates(summaries, plan)
        self.assertEqual(result["fit_train"], dict(passed=False,
                         accuracy_passed=False, probability_passed=False))
        for split in ("probe_same", "probe_variant"):
            self.assertEqual(result[split], dict(passed=True, accuracy_passed=True,
                                                probability_passed=True))
        json.dumps(result, allow_nan=False)
        for field in ("minimum_context_accuracy", "minimum_context_target_probability"):
            changed = copy.deepcopy(plan)
            changed["thresholds"]["probe_same"][field] = 1.01
            checks = gates(summaries, changed)["probe_same"]
            self.assertFalse(checks["passed"])
            self.assertEqual(checks["accuracy_passed"], field != "minimum_context_accuracy")
            self.assertEqual(checks["probability_passed"],
                             field != "minimum_context_target_probability")
        with self.assertRaises(ValueError):
            gates({"fit_train": summary}, plan)
        with self.assertRaises(ValueError):
            gates(dict(summaries, extra=perfect), plan)
        incomplete = copy.deepcopy(summaries)
        del incomplete["fit_train"]["contexts"]["0"]
        with self.assertRaises(ValueError):
            gates(incomplete, plan)


if __name__ == "__main__":
    unittest.main()
