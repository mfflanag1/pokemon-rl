"""Focused unit tests for the F37 legal-root selector.

The module under test is loaded directly so these tests stay independent of
external teacher dependencies and package initialization.

The parity reference below is transcribed from the pinned upstream selector
``artifacts/external/foul-play/fp/search/main.py::select_move_from_mcts_results``
(lines 16-44) rather than from the implementation under test, so an
all-candidates-legal run is checked against the native algorithm, not against a
snapshot of our own code.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import random
import unittest
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "pokemon_teacher_legality"
    / "selection.py"
)
_SPEC = importlib.util.spec_from_file_location("f37_selection_under_test", _MODULE_PATH)
selection = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selection)

LegalSelectionError = selection.LegalSelectionError
select = selection.select_legal_from_mcts_results
MASK_SIZE = selection.MASK_SIZE


# --------------------------------------------------------------------------
# Synthetic MCTS-like objects (no poke-engine, no search, no game)
# --------------------------------------------------------------------------
class FakeOption:
    def __init__(self, move_choice, visits):
        self.move_choice = move_choice
        self.visits = visits


class FakeMctsResult:
    def __init__(self, options, total_visits):
        self.side_one = list(options)
        self.total_visits = total_visits


def world(pairs, chance, index=0, total_visits=100):
    """Build one ``(result, chance, index)`` triple from (choice, visits) pairs."""
    return (
        FakeMctsResult([FakeOption(c, v) for c, v in pairs], total_visits),
        chance,
        index,
    )


ALL_LEGAL = [True] * MASK_SIZE


def mask_with(true_actions):
    mask = [False] * MASK_SIZE
    for action in true_actions:
        mask[action] = True
    return mask


def mapper_from(table, raising=()):
    """Choice -> action mapper; choices in ``raising`` raise ValueError."""

    def action_for_choice(choice):
        if choice in raising:
            raise ValueError("move absent in current request: " + choice)
        if choice not in table:
            raise ValueError("unmapped choice: " + choice)
        return table[choice]

    return action_for_choice


class CountingMapper:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, choice):
        self.calls.append(choice)
        return self.table[choice]


class NoDrawRng:
    """RNG that fails the test if it is consumed at all."""

    def __init__(self):
        self.calls = 0

    def choices(self, population, weights=None):
        self.calls += 1
        raise AssertionError("rng.choices must not be called")


class CountingRng:
    def __init__(self, seed):
        self.inner = random.Random(seed)
        self.calls = []

    def choices(self, population, weights=None):
        self.calls.append((list(population), list(weights)))
        return self.inner.choices(population, weights=weights)


# --------------------------------------------------------------------------
# Reference transcription of the pinned native selector
# --------------------------------------------------------------------------
def native_reference(mcts_results, rng):
    """Returns (decision, considered_policy) exactly as pinned upstream does."""
    final_policy = {}
    for mcts_result, sample_chance, index in mcts_results:
        for s1_option in mcts_result.side_one:
            final_policy[s1_option.move_choice] = final_policy.get(
                s1_option.move_choice, 0
            ) + (sample_chance * (s1_option.visits / mcts_result.total_visits))

    final_policy = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)
    highest_percentage = final_policy[0][1]
    final_policy = [i for i in final_policy if i[1] >= highest_percentage * 0.75]
    choice = rng.choices(final_policy, weights=[p[1] for p in final_policy])[0]
    return choice[0], final_policy


class LegalityGateTest(unittest.TestCase):
    def test_illegal_high_mass_filtered_before_threshold(self):
        # Masses: illegal 0.5, legal 0.3, legal 0.2.
        # Threshold must be 0.75 * 0.3 (highest LEGAL mass) = 0.225, retaining
        # only "mid". Had the threshold used the illegal 0.5 mass (0.375),
        # nothing legal would survive and this call would raise.
        results = [world([("big", 50), ("mid", 30), ("low", 20)], 1.0)]
        mapper = mapper_from({"big": 0, "mid": 1, "low": 2})
        outcome = select(
            results, mask_with([1, 2]), mapper, rng=random.Random(7)
        )
        self.assertEqual(outcome["decision"], "mid")
        audit = outcome["audit"]
        self.assertEqual(audit["highest_legal_mass"], 1.0 * (30 / 100))
        self.assertEqual(audit["threshold"], (1.0 * (30 / 100)) * 0.75)
        self.assertEqual([r["choice"] for r in audit["retained"]], ["mid"])
        reasons = {r["choice"]: r["reason"] for r in audit["candidates"]}
        self.assertEqual(reasons["big"], "illegal_action")
        self.assertEqual(reasons["low"], "below_threshold")
        self.assertIsNone(reasons["mid"])
        self.assertEqual(audit["excluded_count"], 2)
        self.assertAlmostEqual(audit["excluded_mass"], 0.7)
        self.assertEqual(audit["outcome"], "selected")
        self.assertEqual(audit["decision_action"], 1)

    def test_highest_mass_candidate_can_still_win_when_legal(self):
        results = [world([("big", 50), ("mid", 30), ("low", 20)], 1.0)]
        mapper = mapper_from({"big": 0, "mid": 1, "low": 2})
        outcome = select(results, ALL_LEGAL, mapper, rng=random.Random(3))
        self.assertEqual(
            [r["choice"] for r in outcome["audit"]["retained"]], ["big"]
        )
        self.assertEqual(outcome["decision"], "big")

    def test_all_illegal_raises_with_audit_and_never_draws(self):
        results = [world([("a", 60), ("b", 40)], 1.0)]
        mapper = mapper_from({"a": 0, "b": 1})
        rng = NoDrawRng()
        with self.assertRaises(LegalSelectionError) as ctx:
            select(results, mask_with([5]), mapper, rng=rng)
        audit = ctx.exception.audit
        self.assertEqual(rng.calls, 0)
        self.assertEqual(audit["failure"], "no_legal_candidate")
        self.assertEqual(audit["outcome"], "rejected")
        self.assertEqual(audit["legal_count"], 0)
        self.assertEqual(audit["retained"], [])
        self.assertIsNone(audit["threshold"])
        self.assertIsNone(audit["decision"])
        self.assertEqual(
            sorted(r["choice"] for r in audit["candidates"]), ["a", "b"]
        )
        self.assertTrue(
            all(r["reason"] == "illegal_action" for r in audit["candidates"])
        )
        self.assertIsInstance(ctx.exception, ValueError)
        json.dumps(audit)

    def test_zero_mass_legal_candidates_raise_without_drawing(self):
        results = [world([("a", 0), ("b", 0)], 1.0)]
        mapper = mapper_from({"a": 0, "b": 1})
        rng = NoDrawRng()
        with self.assertRaises(LegalSelectionError) as ctx:
            select(results, ALL_LEGAL, mapper, rng=rng)
        audit = ctx.exception.audit
        self.assertEqual(rng.calls, 0)
        self.assertEqual(audit["failure"], "no_positive_legal_mass")
        self.assertEqual(audit["legal_count"], 2)
        self.assertEqual(audit["retained"], [])
        self.assertEqual(audit["excluded_count"], 2)
        self.assertEqual(
            {r["reason"] for r in audit["candidates"]}, {"non_positive_mass"}
        )
        json.dumps(audit)

    def test_zero_chance_world_yields_no_mass_and_no_draw(self):
        results = [world([("a", 50), ("b", 50)], 0.0)]
        mapper = mapper_from({"a": 0, "b": 1})
        with self.assertRaises(LegalSelectionError) as ctx:
            select(results, ALL_LEGAL, mapper, rng=NoDrawRng())
        self.assertEqual(ctx.exception.audit["failure"], "no_positive_legal_mass")


class NativeParityTest(unittest.TestCase):
    def _assert_parity(self, results, seed):
        mine_rng = random.Random(seed)
        ref_rng = random.Random(seed)
        choices = []
        for result, _chance, _index in results:
            for option in result.side_one:
                if option.move_choice not in choices:
                    choices.append(option.move_choice)
        mapper = mapper_from({c: i for i, c in enumerate(choices)})
        outcome = select(results, ALL_LEGAL, mapper, rng=mine_rng)
        reference, considered = native_reference(results, ref_rng)
        self.assertEqual(outcome["decision"], reference)
        # Identical RNG consumption: equal state after exactly one draw.
        self.assertEqual(mine_rng.getstate(), ref_rng.getstate())
        self.assertEqual(
            [r["choice"] for r in outcome["audit"]["retained"]],
            [p[0] for p in considered],
        )
        self.assertEqual(
            [r["mass"] for r in outcome["audit"]["retained"]],
            [p[1] for p in considered],
        )

    def test_all_legal_matches_native_on_randomized_worlds(self):
        generator = random.Random(20261027)
        names = ["m{}".format(i) for i in range(6)]
        for _case in range(40):
            worlds = []
            for index in range(generator.randint(1, 4)):
                picked = generator.sample(names, generator.randint(1, len(names)))
                pairs = [(name, generator.randint(0, 400)) for name in picked]
                total = max(1, sum(v for _n, v in pairs))
                worlds.append(
                    world(pairs, generator.random() + 0.01, index, total)
                )
            if not any(
                option.visits > 0
                for result, _c, _i in worlds
                for option in result.side_one
            ):
                continue
            for seed in (0, 1, 12345):
                self._assert_parity(worlds, seed)

    def test_tie_across_worlds_keeps_native_order_and_parity(self):
        # "a" and "b" accumulate exactly equal mass from mirrored worlds.
        results = [
            world([("a", 60), ("b", 40)], 0.5, 0),
            world([("b", 60), ("a", 40)], 0.5, 1),
        ]
        mapper = mapper_from({"a": 0, "b": 1})
        masses = {}
        for result, chance, _index in results:
            for option in result.side_one:
                masses[option.move_choice] = masses.get(option.move_choice, 0) + (
                    chance * (option.visits / result.total_visits)
                )
        self.assertEqual(masses["a"], masses["b"])

        seen = set()
        for seed in range(60):
            mine_rng = random.Random(seed)
            ref_rng = random.Random(seed)
            outcome = select(results, ALL_LEGAL, mapper, rng=mine_rng)
            reference, considered = native_reference(results, ref_rng)
            self.assertEqual(outcome["decision"], reference)
            self.assertEqual(mine_rng.getstate(), ref_rng.getstate())
            # Stable tie order: aggregation insertion order, "a" before "b".
            self.assertEqual([p[0] for p in considered], ["a", "b"])
            self.assertEqual(
                [r["choice"] for r in outcome["audit"]["retained"]], ["a", "b"]
            )
            seen.add(outcome["decision"])
        self.assertEqual(seen, {"a", "b"})

    def test_single_retained_candidate_still_consumes_one_draw(self):
        results = [world([("a", 100), ("b", 0)], 1.0)]
        mapper = mapper_from({"a": 0, "b": 1})
        mine_rng = random.Random(99)
        ref_rng = random.Random(99)
        outcome = select(results, ALL_LEGAL, mapper, rng=mine_rng)
        reference, _considered = native_reference(results, ref_rng)
        self.assertEqual(outcome["decision"], reference)
        self.assertEqual(mine_rng.getstate(), ref_rng.getstate())
        self.assertNotEqual(mine_rng.getstate(), random.Random(99).getstate())

    def test_weights_are_raw_masses_not_renormalised(self):
        results = [
            world([("a", 50), ("b", 50)], 0.25, 0),
            world([("a", 50), ("b", 50)], 0.25, 1),
        ]
        mapper = mapper_from({"a": 0, "b": 1})
        rng = CountingRng(5)
        outcome = select(results, mask_with([0, 1]), mapper, rng=rng)
        self.assertEqual(len(rng.calls), 1)
        population, weights = rng.calls[0]
        self.assertEqual(population, ["a", "b"])
        self.assertEqual(weights, [0.25, 0.25])
        self.assertAlmostEqual(sum(weights), 0.5)
        self.assertIn(outcome["decision"], population)

    def test_illegal_candidate_mass_is_not_redistributed(self):
        results = [world([("a", 50), ("b", 50)], 1.0)]
        mapper = mapper_from({"a": 0, "b": 1})
        rng = CountingRng(11)
        select(results, mask_with([0]), mapper, rng=rng)
        _population, weights = rng.calls[0]
        self.assertEqual(weights, [0.5])


class AggregationTest(unittest.TestCase):
    def test_duplicate_choice_entries_accumulate(self):
        results = [world([("a", 25), ("a", 25), ("b", 50)], 1.0)]
        mapper = mapper_from({"a": 0, "b": 1})
        outcome = select(results, ALL_LEGAL, mapper, rng=random.Random(1))
        audit = outcome["audit"]
        masses = {r["choice"]: r["mass"] for r in audit["candidates"]}
        self.assertEqual(audit["aggregated_count"], 2)
        self.assertEqual(masses["a"], 0.5)
        self.assertEqual(masses["b"], 0.5)
        self.assertEqual(audit["worlds"][0]["option_count"], 3)

    def test_same_choice_across_worlds_accumulates_with_chance(self):
        results = [
            world([("a", 80), ("b", 20)], 0.75, 0),
            world([("a", 20), ("b", 80)], 0.25, 1),
        ]
        mapper = mapper_from({"a": 0, "b": 1})
        outcome = select(results, ALL_LEGAL, mapper, rng=random.Random(2))
        masses = {r["choice"]: r["mass"] for r in outcome["audit"]["candidates"]}
        self.assertAlmostEqual(masses["a"], 0.75 * 0.8 + 0.25 * 0.2)
        self.assertAlmostEqual(masses["b"], 0.75 * 0.2 + 0.25 * 0.8)
        self.assertEqual(outcome["decision"], "a")

    def test_each_aggregated_choice_is_mapped_exactly_once(self):
        results = [
            world([("a", 50), ("a", 10), ("b", 40)], 0.5, 0),
            world([("a", 50), ("b", 50)], 0.5, 1),
        ]
        mapper = CountingMapper({"a": 0, "b": 1})
        select(results, ALL_LEGAL, mapper, rng=random.Random(4))
        self.assertEqual(mapper.calls, ["a", "b"])


class MappingExclusionTest(unittest.TestCase):
    def test_mapper_value_error_excludes_candidate_with_reason(self):
        results = [world([("unexpressible", 70), ("ok", 30)], 1.0)]
        mapper = mapper_from({"ok": 2}, raising=("unexpressible",))
        outcome = select(results, ALL_LEGAL, mapper, rng=random.Random(8))
        self.assertEqual(outcome["decision"], "ok")
        record = next(
            r for r in outcome["audit"]["candidates"] if r["choice"] == "unexpressible"
        )
        self.assertEqual(record["reason"], "unmappable_choice")
        self.assertFalse(record["allowed"])
        self.assertIsNone(record["action"])
        self.assertIn("current request", record["detail"])
        json.dumps(outcome["audit"])

    def test_mapper_error_on_every_choice_raises_without_drawing(self):
        results = [world([("x", 60), ("y", 40)], 1.0)]
        mapper = mapper_from({}, raising=("x", "y"))
        rng = NoDrawRng()
        with self.assertRaises(LegalSelectionError) as ctx:
            select(results, ALL_LEGAL, mapper, rng=rng)
        self.assertEqual(rng.calls, 0)
        self.assertEqual(ctx.exception.audit["failure"], "no_legal_candidate")

    def test_non_value_error_from_mapper_propagates(self):
        def mapper(choice):
            raise KeyError(choice)

        with self.assertRaises(KeyError):
            select(
                [world([("a", 100)], 1.0)], ALL_LEGAL, mapper, rng=NoDrawRng()
            )

    def test_bad_action_values_are_excluded_with_reasons(self):
        results = [world([("fl", 40), ("bo", 30), ("oor", 20), ("ok", 10)], 1.0)]
        table = {"fl": 1.0, "bo": True, "oor": MASK_SIZE, "ok": 3}

        def mapper(choice):
            return table[choice]

        outcome = select(results, ALL_LEGAL, mapper, rng=random.Random(6))
        self.assertEqual(outcome["decision"], "ok")
        reasons = {r["choice"]: r["reason"] for r in outcome["audit"]["candidates"]}
        self.assertEqual(reasons["fl"], "non_integer_action")
        self.assertEqual(reasons["bo"], "boolean_action")
        self.assertEqual(reasons["oor"], "action_out_of_range")
        self.assertIsNone(reasons["ok"])
        self.assertEqual(outcome["audit"]["legal_count"], 1)
        json.dumps(outcome["audit"])

    def test_negative_action_is_out_of_range(self):
        def mapper(choice):
            return -1 if choice == "neg" else 0

        outcome = select(
            [world([("neg", 90), ("ok", 10)], 1.0)],
            ALL_LEGAL,
            mapper,
            rng=random.Random(0),
        )
        self.assertEqual(outcome["decision"], "ok")
        record = next(
            r for r in outcome["audit"]["candidates"] if r["choice"] == "neg"
        )
        self.assertEqual(record["reason"], "action_out_of_range")


class MalformedInputTest(unittest.TestCase):
    def _expect(self, failure, results, mask, mapper=None, rng=None):
        if mapper is None:
            mapper = mapper_from({"a": 0, "b": 1})
        if rng is None:
            rng = NoDrawRng()
        with self.assertRaises(LegalSelectionError) as ctx:
            select(results, mask, mapper, rng=rng)
        audit = ctx.exception.audit
        self.assertEqual(audit["outcome"], "rejected")
        self.assertEqual(audit["failure"], failure)
        self.assertIn("evidence", audit)
        json.dumps(audit)
        if isinstance(rng, NoDrawRng):
            self.assertEqual(rng.calls, 0)
        return audit

    def _ok_results(self):
        return [world([("a", 60), ("b", 40)], 1.0)]

    def test_mask_wrong_length(self):
        audit = self._expect(
            "mask_wrong_length", self._ok_results(), [True] * (MASK_SIZE - 1)
        )
        self.assertEqual(audit["evidence"]["expected_length"], MASK_SIZE)
        self._expect("mask_wrong_length", self._ok_results(), [True] * (MASK_SIZE + 1))

    def test_mask_non_strict_bools_rejected(self):
        mask = [True] * MASK_SIZE
        mask[2] = 1
        audit = self._expect("mask_not_strict_bool", self._ok_results(), mask)
        self.assertEqual(audit["evidence"]["index"], 2)
        mask[2] = None
        self._expect("mask_not_strict_bool", self._ok_results(), mask)

    def test_mask_all_false_rejected(self):
        self._expect("mask_all_false", self._ok_results(), [False] * MASK_SIZE)

    def test_mask_not_a_sequence(self):
        self._expect("mask_not_sequence", self._ok_results(), "TTTTTTTTTTTTTT")
        self._expect("mask_not_sequence", self._ok_results(), None)

    def test_results_shape_rejected(self):
        self._expect("results_not_sequence", None, ALL_LEGAL)
        self._expect("results_empty", [], ALL_LEGAL)
        self._expect(
            "result_entry_shape",
            [(FakeMctsResult([FakeOption("a", 1)], 1), 1.0)],
            ALL_LEGAL,
        )
        self._expect("result_entry_shape", ["not-a-triple"], ALL_LEGAL)

    def test_world_index_must_be_int(self):
        audit = self._expect(
            "world_index_not_int",
            [world([("a", 60)], 1.0, index="0")],
            ALL_LEGAL,
        )
        self.assertEqual(audit["evidence"]["index"], 0)
        self._expect(
            "world_index_not_int", [world([("a", 60)], 1.0, index=True)], ALL_LEGAL
        )

    def test_chance_must_be_finite_and_non_negative(self):
        for bad in (-0.5, float("nan"), float("inf"), "1.0", None, True):
            self._expect("chance_invalid", [world([("a", 60)], bad)], ALL_LEGAL)

    def test_total_visits_must_be_positive_and_finite(self):
        for bad in (0, -5, float("nan"), float("inf"), "100", None):
            self._expect(
                "total_visits_invalid",
                [world([("a", 60)], 1.0, total_visits=bad)],
                ALL_LEGAL,
            )

    def test_visits_must_be_finite_and_non_negative(self):
        for bad in (-1, float("nan"), float("inf"), "50", None, True):
            audit = self._expect(
                "visits_invalid", [world([("a", bad)], 1.0)], ALL_LEGAL
            )
            self.assertEqual(audit["evidence"]["move_choice"], "a")

    def test_move_choice_must_be_non_empty_string(self):
        for bad in ("", 7, None):
            self._expect("move_choice_invalid", [world([(bad, 50)], 1.0)], ALL_LEGAL)

    def test_missing_attributes_rejected(self):
        class Bare:
            pass

        self._expect("result_missing_total_visits", [(Bare(), 1.0, 0)], ALL_LEGAL)

        class NoSide:
            total_visits = 10

        self._expect("result_missing_side_one", [(NoSide(), 1.0, 0)], ALL_LEGAL)

        class BadSide:
            total_visits = 10
            side_one = "ab"

        self._expect("side_one_not_sequence", [(BadSide(), 1.0, 0)], ALL_LEGAL)
        self._expect("side_one_empty", [world([], 1.0)], ALL_LEGAL)

        class NoVisits:
            move_choice = "a"

        self._expect(
            "visits_invalid",
            [(FakeMctsResult([NoVisits()], 10), 1.0, 0)],
            ALL_LEGAL,
        )

    def test_rng_must_expose_choices(self):
        class Bad:
            choices = 3

        with self.assertRaises(LegalSelectionError) as ctx:
            select(self._ok_results(), ALL_LEGAL, mapper_from({"a": 0}), rng=Bad())
        self.assertEqual(ctx.exception.audit["failure"], "rng_without_choices")
        with self.assertRaises(LegalSelectionError):
            select(self._ok_results(), ALL_LEGAL, mapper_from({"a": 0}), rng=None)

    def test_action_for_choice_must_be_callable(self):
        with self.assertRaises(LegalSelectionError) as ctx:
            select(self._ok_results(), ALL_LEGAL, {"a": 0}, rng=NoDrawRng())
        self.assertEqual(
            ctx.exception.audit["failure"], "action_for_choice_not_callable"
        )

    def test_rng_returning_foreign_choice_is_rejected(self):
        class LyingRng:
            def choices(self, population, weights=None):
                return ["illegalmove"]

        with self.assertRaises(LegalSelectionError) as ctx:
            select(
                self._ok_results(),
                ALL_LEGAL,
                mapper_from({"a": 0, "b": 1}),
                rng=LyingRng(),
            )
        self.assertEqual(
            ctx.exception.audit["failure"], "rng_choices_foreign_choice"
        )

    def test_rng_returning_empty_sample_is_rejected(self):
        class EmptyRng:
            def choices(self, population, weights=None):
                return []

        with self.assertRaises(LegalSelectionError) as ctx:
            select(
                self._ok_results(),
                ALL_LEGAL,
                mapper_from({"a": 0, "b": 1}),
                rng=EmptyRng(),
            )
        self.assertEqual(ctx.exception.audit["failure"], "rng_choices_bad_return")

    def test_keyword_only_rng(self):
        with self.assertRaises(TypeError):
            select(self._ok_results(), ALL_LEGAL, mapper_from({"a": 0}), random)


class DeterminismAndAuditTest(unittest.TestCase):
    def _scenario(self):
        # Masses: a=0.240, b=0.428, c=0.332. With every candidate legal the
        # threshold is 0.321, so "b" and "c" are both retained and the seed
        # genuinely decides the outcome; "a" is always below the threshold.
        return (
            [
                world([("a", 40), ("b", 38), ("c", 22)], 0.6, 0),
                world([("b", 50), ("c", 50)], 0.4, 1),
            ],
            mapper_from({"a": 0, "b": 1, "c": 2}),
        )

    def test_scenario_retains_more_than_one_candidate(self):
        results, mapper = self._scenario()
        audit = select(results, ALL_LEGAL, mapper, rng=random.Random(0))["audit"]
        self.assertEqual([r["choice"] for r in audit["retained"]], ["b", "c"])

    def test_seed_variation_is_observable(self):
        results, mapper = self._scenario()
        seen = {
            select(results, ALL_LEGAL, mapper, rng=random.Random(seed))["decision"]
            for seed in range(40)
        }
        self.assertEqual(seen, {"b", "c"})

    def test_same_seed_gives_same_decision(self):
        results, mapper = self._scenario()
        first = select(results, ALL_LEGAL, mapper, rng=random.Random(42))["decision"]
        for _repeat in range(5):
            again = select(results, ALL_LEGAL, mapper, rng=random.Random(42))
            self.assertEqual(again["decision"], first)

    def test_decision_is_always_a_legal_retained_candidate(self):
        results, mapper = self._scenario()
        mask = mask_with([1, 2])
        for seed in range(25):
            outcome = select(results, mask, mapper, rng=random.Random(seed))
            retained = {r["choice"] for r in outcome["audit"]["retained"]}
            self.assertIn(outcome["decision"], retained)
            self.assertNotIn("a", retained)
            self.assertIn(outcome["audit"]["decision_action"], (1, 2))
            self.assertTrue(mask[outcome["audit"]["decision_action"]])

    def test_audit_is_json_serializable_and_complete(self):
        results, mapper = self._scenario()
        outcome = select(results, mask_with([0, 2]), mapper, rng=random.Random(13))
        audit = outcome["audit"]
        round_tripped = json.loads(json.dumps(audit))
        self.assertEqual(round_tripped["decision"], outcome["decision"])
        for key in (
            "schema",
            "outcome",
            "worlds",
            "candidates",
            "retained",
            "threshold",
            "threshold_fraction",
            "highest_legal_mass",
            "excluded_count",
            "excluded_mass",
            "mask_true_count",
        ):
            self.assertIn(key, audit)
        self.assertEqual(audit["mask_true_count"], 2)
        self.assertEqual(audit["threshold_fraction"], 0.75)
        self.assertEqual({"a", "b", "c"}, {r["choice"] for r in audit["candidates"]})
        for record in audit["candidates"]:
            self.assertTrue(math.isfinite(record["mass"]))
            self.assertIn("allowed", record)
            self.assertIn("insertion_index", record)
        self.assertEqual(
            audit["excluded_count"] + audit["retained_count"],
            audit["aggregated_count"],
        )
        self.assertAlmostEqual(
            audit["excluded_mass"] + audit["retained_mass"], audit["aggregated_mass"]
        )

    def test_candidates_are_ordered_by_descending_mass(self):
        results, mapper = self._scenario()
        audit = select(results, ALL_LEGAL, mapper, rng=random.Random(17))["audit"]
        masses = [r["mass"] for r in audit["candidates"]]
        self.assertEqual(masses, sorted(masses, reverse=True))

    def test_module_imports_only_stdlib(self):
        tree = ast.parse(_MODULE_PATH.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    imported.add("<relative>")
                elif node.module:
                    imported.add(node.module.split(".")[0])
        self.assertLessEqual(imported, {"math", "__future__"}, imported)


class AuditedNumericBoundaryTests(unittest.TestCase):
    def rejected_without_draw(self, results):
        rng = NoDrawRng()
        with self.assertRaises(LegalSelectionError) as caught:
            select(results, ALL_LEGAL, mapper_from({"a": 0, "b": 1}), rng=rng)
        json.dumps(caught.exception.audit, allow_nan=False)
        self.assertEqual(rng.calls, 0)

    def test_finite_weights_with_overflowing_sum(self):
        self.rejected_without_draw([
            world([("a", 1), ("b", 1)], 1.7e308, i, total_visits=2)
            for i in range(2)
        ])

    def test_unrepresentable_integer_numeric_fields(self):
        for field in ("chance", "visits", "total_visits"):
            for magnitude in (10**1000, 10**5000):
                with self.subTest(field=field, digits=magnitude.bit_length()):
                    self.rejected_without_draw([world(
                        [("a", magnitude if field == "visits" else 1)],
                        magnitude if field == "chance" else 1.,
                        total_visits=magnitude if field == "total_visits" else 2)])

    def test_representable_inputs_with_overflowing_ratio(self):
        self.rejected_without_draw([world([("a", 1e308)], 1., total_visits=1e-308)])


if __name__ == "__main__":
    unittest.main()
