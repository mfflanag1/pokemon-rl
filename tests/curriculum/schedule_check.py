#!/usr/bin/env python3
"""Run directly with Python; no installed package or third-party tests needed."""

import copy
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from pokemon_curriculum.schedule import choose_scenario, quotas, select_indices


def entry(families, content_hash):
    return {"families": families, "content_hash": content_hash, "split": "train",
            "start": {"split": "train", "swap": False}}


class ScheduleChecks(unittest.TestCase):
    def setUp(self):
        self.bank = [entry(["b", "a"], "shared"), entry(["a"], "a2"),
                     entry(["b"], "b1"), entry(["a"], "a1")]

    def test_quotas(self):
        for update, prefix in ((0, 102), (15, 102), (16, 205),
                               (31, 205), (32, 256), (47, 256)):
            self.assertEqual(quotas("curriculum", update),
                             {"official": 1024-prefix, "prefix": prefix})
        for update in range(48):
            self.assertEqual(quotas("official", update), {"official": 1024, "prefix": 0})
        self.assertEqual(sum(quotas("curriculum", u)["prefix"]
                             for _ in (7103, 8209, 9311) for u in range(48)), 27024)
        for update in (True, False, 1.0, "1", None, -1, 48):
            with self.assertRaises(ValueError):
                quotas("official", update)
        for arm in ("dev", "final", "unknown", None):
            with self.assertRaises(ValueError):
                quotas(arm, 0)

    def test_scenario_determinism_and_order(self):
        for seed in range(100):
            result = choose_scenario(self.bank, {"a": 1, "b": 3}, seed)
            reordered = copy.deepcopy(self.bank[::-1])
            for e in reordered:
                e["families"].reverse()
            self.assertEqual(result, choose_scenario(reordered, {"b": 3, "a": 1}, seed))
            self.assertIn(result, {(f, e["content_hash"]) for e in self.bank for f in e["families"]})
            rng = random.Random(seed)
            family = "a" if rng.random() < 0.25 else "b"
            eligible = sorted(e["content_hash"] for e in self.bank
                              if family in e["families"])
            self.assertEqual(result, (family, rng.choice(eligible)))
            modified = copy.deepcopy(self.bank)
            for e in modified:
                e["outcome"] = object()
                e["values"] = object()
                e["start"]["outcome"] = object()
            self.assertEqual(result, choose_scenario(modified, {"a": 1, "b": 3}, seed))
        self.assertEqual(choose_scenario(self.bank, {"a": 1, "b": 0}, 8)[0], "a")

    def test_invalid_banks_and_weights(self):
        banks = [[], self.bank + [self.bank[0]], [entry(["a"], "")], [{}]]
        missing = entry(["a"], "x")
        del missing["families"]
        banks.append([missing])
        for families in (None, "a", ("a",), [], [""], [" "], ["a", "a"], [1], [["a"]]):
            banks.append([entry(families, "x")])
        for path, value in (("split", "dev"), ("split", "final"),
                            ("start.split", "dev"), ("start.split", "final"),
                            ("start.swap", True), ("start.swap", 0),
                            ("start.swap", None)):
            bank = copy.deepcopy(self.bank)
            if path.startswith("start."):
                bank[0]["start"][path.split(".")[1]] = value
            else:
                bank[0][path] = value
            banks.append(bank)
        bank = copy.deepcopy(self.bank)
        del bank[0]["start"]["swap"]
        banks.append(bank)
        for bank in banks:
            with self.assertRaises(ValueError):
                choose_scenario(bank, {"a": 1}, 1)
        for weights in ({"missing": 1}, {}, {"a": 0}, {"a": -1},
                        {"a": float("nan")}, {"a": float("inf")},
                        {"a": True}, {"a": "1"}, {1: 1}):
            with self.assertRaises(ValueError):
                choose_scenario(self.bank, weights, 1)

    def test_overlap_uniform_within_family(self):
        for family in ("a", "b"):
            eligible = sorted(e["content_hash"] for e in self.bank
                              if family in e["families"])
            observed = set()
            for seed in range(256):
                rng = random.Random(seed)
                rng.random()  # Family draw precedes uniform hash selection.
                expected = (family, rng.choice(eligible))
                result = choose_scenario(self.bank, {family: 1}, seed)
                self.assertEqual(result, expected)
                observed.add(result[1])
            self.assertEqual(observed, set(eligible))
            self.assertIn("shared", observed)

    def test_real_bank(self):
        bank = json.loads(Path(__file__).with_name("bank_fixture.json").read_text())
        self.assertEqual(len(bank), 65)
        self.assertEqual(len({e["content_hash"] for e in bank}), 65)
        families = sorted({f for e in bank for f in e["families"]})
        reordered = copy.deepcopy(bank[::-1])
        for e in reordered:
            e["families"].reverse()
        for family in families:
            eligible = sorted(e["content_hash"] for e in bank if family in e["families"])
            observed = set()
            for seed in range(512):
                result = choose_scenario(bank, {family: 1}, seed)
                rng = random.Random(seed)
                rng.random()
                self.assertEqual(result, (family, rng.choice(eligible)))
                self.assertEqual(result, choose_scenario(reordered, {family: 1}, seed))
                observed.add(result[1])
            self.assertEqual(observed, set(eligible))
        weights = dict.fromkeys(families, 1)
        for seed in (0, 7103, 8209, 9311, 2**32 - 1):
            self.assertEqual(choose_scenario(bank, weights, seed),
                             choose_scenario(reordered, dict(reversed(list(weights.items()))), seed))
        for e in bank:
            self.assertEqual(e["split"], "train")
            self.assertEqual(e["start"]["split"], "train")
            self.assertIs(e["start"]["swap"], False)

    def test_seeds(self):
        for seed in (None, True, False, 1.0, "1", -1, 2**32, 2**40):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    choose_scenario(self.bank, {"a": 1}, seed)
                with self.assertRaises(ValueError):
                    select_indices(1500, 1200, 102, seed)
        for seed in (0, 2**32 - 1):
            self.assertEqual(choose_scenario(self.bank, {"a": 1}, seed),
                             choose_scenario(self.bank, {"a": 1}, seed))
            self.assertEqual(select_indices(1500, 1200, 102, seed),
                             select_indices(1500, 1200, 102, seed))

    def test_indices(self):
        for quota in (0, 102, 205, 256, 1024):
            for seed in (7103, 8209, 9311):
                result = select_indices(1500, 1200, quota, seed)
                rng = random.Random(seed)
                expected = [("official", i) for i in rng.sample(range(1500), 1024-quota)]
                expected += [("prefix", i) for i in rng.sample(range(1200), quota)]
                rng.shuffle(expected)
                self.assertEqual(result, expected)
                self.assertEqual(result, select_indices(1500, 1200, quota, seed))
                self.assertEqual(len(result), 1024)
                self.assertEqual(len(set(result)), 1024)
                for stratum, n, count in (("official", 1500, 1024-quota), ("prefix", 1200, quota)):
                    indices = [i for s, i in result if s == stratum]
                    self.assertEqual(len(indices), count)
                    self.assertTrue(all(0 <= i < n for i in indices))
        for slot in range(3):
            for bad in (True, False, -1, 1.0, "1", None):
                args = [1500, 1200, 102]
                args[slot] = bad
                with self.assertRaises(ValueError):
                    select_indices(*args, seed=1)
        for args in ((1500, 1200, 1025), (921, 102, 102), (922, 101, 102)):
            with self.assertRaises(ValueError):
                select_indices(*args, seed=1)
        self.assertEqual(len(select_indices(1024, 0, 0, 1)), 1024)
        self.assertEqual(len(select_indices(0, 1024, 1024, 1)), 1024)


if __name__ == "__main__":
    unittest.main()
