"""Hand-computable checks for deterministic retention diagnostics."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from pokemon_retention.metrics import compare


def endpoint(fresh, supervised):
    return {
        arm: {"cells": {
            f"{panel}/{i}": {
                "n": 16, "target_mass": mass, "greedy_accuracy": 1.0,
                "sampled_accuracy": 0.5, "mean_reward": 0.5,
                "local_value_mse": 0.25,
            }
            for panel in ("train", "variant") for i in range(16)
        }} for arm, mass in (("fresh", fresh), ("supervised", supervised))
    }


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.initial = endpoint(0.25, 0.625)
        self.final = endpoint(0.5, 0.875)
        self.plan = {"thresholds": {
            "minimum_greedy_accuracy": 0.95,
            "minimum_context_mass": 0.625,
            "maximum_context_mass_drop": 0.125,
            "minimum_family_improvement": 0.125,
            "minimum_family_advantage": 0.25,
        }}

    def result(self):
        return compare(self.initial, self.final, self.plan)

    def test_perfect_and_nonmutating_json(self):
        before = copy.deepcopy((self.initial, self.final, self.plan))
        result = self.result()
        self.assertEqual((True, True, True), tuple(result[k] for k in
                         ("retained", "improved", "advantage")))
        self.assertEqual(len(result["cells"]), 32)
        self.assertEqual(result["cells"]["train/0"], {
            "supervised_gain": 0.25, "fresh_gain": 0.25,
            "final_advantage": 0.375, "retained": True,
        })
        self.assertEqual(set(result["families"]), {
            "train/switch", "train/tera", "variant/switch", "variant/tera"})
        for family in result["families"].values():
            self.assertEqual(family, {"supervised_gain": 0.25,
                "final_advantage": 0.375, "improved": True, "advantage": True})
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual((self.initial, self.final, self.plan), before)

    def test_single_context_collapse_passes_family_means(self):
        self.final["supervised"]["cells"]["variant/15"]["target_mass"] = 0.25
        result = self.result()
        self.assertFalse(result["retained"])
        self.assertFalse(result["cells"]["variant/15"]["retained"])
        self.assertTrue(result["improved"])
        self.assertTrue(result["advantage"])
        self.assertEqual(result["families"]["variant/tera"]["supervised_gain"],
                         0.171875)  # (7 * .25 - .375) / 8

    def test_single_family_regression(self):
        for i in range(8):
            self.final["supervised"]["cells"][f"train/{i}"]["target_mass"] = 0.625
        result = self.result()
        self.assertTrue(result["retained"])
        self.assertFalse(result["improved"])
        self.assertFalse(result["advantage"])
        self.assertEqual(result["families"]["train/switch"]["supervised_gain"], 0.0)
        self.assertTrue(result["families"]["train/tera"]["improved"])

    def test_exact_boundaries(self):
        self.final = endpoint(0.5, 0.75)
        for cell in self.final["supervised"]["cells"].values():
            cell["greedy_accuracy"] = 0.95
        self.plan["thresholds"]["minimum_context_mass"] = 0.75
        self.assertTrue(all(self.result()[k] for k in ("retained", "improved", "advantage")))
        self.initial = endpoint(0.25, 0.875)
        self.assertTrue(self.result()["retained"])  # drop exactly .125
        self.initial["supervised"]["cells"]["train/0"]["target_mass"] = 0.8750000000000001
        self.assertFalse(self.result()["retained"])

    def test_each_retention_gate(self):
        for metric, value in (("greedy_accuracy", 0.9499999999999998),
                              ("target_mass", 0.6249999999999999)):
            with self.subTest(metric=metric):
                final = copy.deepcopy(self.final)
                final["supervised"]["cells"]["train/0"][metric] = value
                self.assertFalse(compare(self.initial, final, self.plan)["retained"])

    def test_missing_and_extra_structure(self):
        for where, key in (((), "fresh"), (("fresh",), "cells"),
                           (("fresh", "cells"), "train/0")):
            for extra in (False, True):
                with self.subTest(where=where, extra=extra):
                    initial = copy.deepcopy(self.initial)
                    obj = initial
                    for part in where:
                        obj = obj[part]
                    if extra:
                        obj["unexpected"] = {}
                    else:
                        del obj[key]
                    with self.assertRaises(ValueError):
                        compare(initial, self.final, self.plan)

    def test_invalid_values_in_every_endpoint_and_arm(self):
        for endpoint_index in (0, 1):
            for arm in ("fresh", "supervised"):
                for metric, values in {
                    "n": (0, -1, 16.0, True, "16", 17),
                    **{name: (-0.01, 1.01, float("nan"), float("inf"),
                              -float("inf"), True, "0.5", None)
                       for name in ("target_mass", "greedy_accuracy", "sampled_accuracy",
                                    "mean_reward", "local_value_mse")},
                }.items():
                    for value in values:
                        with self.subTest(endpoint=endpoint_index, arm=arm,
                                          metric=metric, value=value):
                            endpoints = copy.deepcopy([self.initial, self.final])
                            endpoints[endpoint_index][arm]["cells"]["variant/15"][metric] = value
                            with self.assertRaises(ValueError):
                                compare(*endpoints, self.plan)

    def test_invalid_thresholds(self):
        for name in self.plan["thresholds"]:
            for value in (None, True, float("nan"), -0.1, 1.1):
                plan = copy.deepcopy(self.plan)
                plan["thresholds"][name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    compare(self.initial, self.final, plan)
            plan = copy.deepcopy(self.plan)
            del plan["thresholds"][name]
            with self.assertRaises(ValueError):
                compare(self.initial, self.final, plan)


if __name__ == "__main__":
    unittest.main()
