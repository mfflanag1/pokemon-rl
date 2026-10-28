"""Checks for ``pokemon_imitation.data`` (game-balanced sampler + metrics).

Run directly, without bytecode caches::

    python -B tests/imitation/data_check.py

The module locates the repository root relative to this file and inserts
``<root>/python`` on ``sys.path`` so no installation step is required.
"""

import json
import math
import pathlib
import sys
import unittest

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from pokemon_imitation import data  # noqa: E402  (path inserted above)


# --------------------------------------------------------------------------- #
# Helpers (not tests)
# --------------------------------------------------------------------------- #
def make_dataset(game_index, seed=0):
    """Build a valid dataset whose rows carry the given game ids."""
    game_index = np.asarray(game_index, dtype=np.int32)
    n = game_index.shape[0]
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, data.FEATURE_DIM)).astype(np.float32)
    mask = rng.random((n, data.ACTION_DIM)) < 0.5
    mask[np.arange(n), rng.integers(0, data.ACTION_DIM, size=n)] = True
    targets = np.array(
        [rng.choice(np.flatnonzero(row)) for row in mask], dtype=np.int32
    )
    return {
        "features": features,
        "mask": mask,
        "targets": targets,
        "game_index": game_index,
    }


def snapshot(mapping):
    return {key: np.array(value, copy=True) for key, value in mapping.items()}


def hand_example():
    """Tiny metrics fixture with a fully hand-derived expectation.

    Row 0 (game 4): legal {0, 1}, p = [.5, .5], target 1.
        argmax tie -> index 0 -> wrong.  nll = ln 2.  mass .5
    Row 1 (game 4): legal {2, 3}, p = [.8, .2], target 2.
        correct.  nll = -ln .8 = ln 1.25.  mass .8
    Row 2 (game 9): legal {5},    p = [1.0],    target 5.
        correct.  nll = 0.  mass 1.0
    """
    probs = np.zeros((3, data.ACTION_DIM), dtype=np.float64)
    masks = np.zeros((3, data.ACTION_DIM), dtype=bool)
    probs[0, 0] = 0.5
    probs[0, 1] = 0.5
    masks[0, [0, 1]] = True
    probs[1, 2] = 0.8
    probs[1, 3] = 0.2
    masks[1, [2, 3]] = True
    probs[2, 5] = 1.0
    masks[2, 5] = True
    targets = np.array([1, 2, 5], dtype=np.int32)
    game_index = np.array([4, 4, 9], dtype=np.int32)  # ids need not be contiguous
    return probs, masks, targets, game_index


class DataCheck(unittest.TestCase):
    # 1 ------------------------------------------------------------------ #
    def test_validate_accepts_valid_and_rejects_boundaries(self):
        single = make_dataset([0])
        self.assertIsNone(data.validate(single))
        sparse = make_dataset([7, 7, 2, 40])  # non-contiguous ids are fine
        before = snapshot(sparse)
        self.assertIsNone(data.validate(sparse))
        for key in before:
            np.testing.assert_array_equal(sparse[key], before[key])

        n = 4
        good = make_dataset([1, 1, 2, 3])

        def variant(**changes):
            copy = dict(good)
            copy.update(changes)
            return copy

        features = good["features"]
        mask = good["mask"]
        targets = good["targets"]
        game_index = good["game_index"]

        bad_mask_row = mask.copy()
        bad_mask_row[1, :] = False
        illegal_target_mask = mask.copy()
        illegal_target_mask[0, :] = True
        illegal_target_mask[0, targets[0]] = False
        nan_features = features.copy()
        nan_features[0, 0] = np.nan
        inf_features = features.copy()
        inf_features[2, 5] = np.inf
        missing = dict(good)
        del missing["mask"]

        cases = {
            "not a dict": [features, mask, targets, game_index],
            "missing key": missing,
            "extra key": variant(labels=np.zeros(n, dtype=np.int32)),
            "features list": variant(features=features.tolist()),
            "features float64": variant(features=features.astype(np.float64)),
            "features width": variant(features=features[:, :-1]),
            "features 1-D": variant(features=features[:, 0]),
            "features nan": variant(features=nan_features),
            "features inf": variant(features=inf_features),
            "empty dataset": {
                "features": np.zeros((0, data.FEATURE_DIM), np.float32),
                "mask": np.zeros((0, data.ACTION_DIM), bool),
                "targets": np.zeros((0,), np.int32),
                "game_index": np.zeros((0,), np.int32),
            },
            "mask uint8": variant(mask=mask.astype(np.uint8)),
            "mask width": variant(mask=mask[:, :-1]),
            "mask empty row": variant(mask=bad_mask_row),
            "targets int64": variant(targets=targets.astype(np.int64)),
            "targets 2-D": variant(targets=targets[:, None]),
            "targets too large": variant(
                targets=np.array([data.ACTION_DIM, 0, 0, 0], np.int32)
            ),
            "targets negative": variant(targets=np.array([-1, 0, 0, 0], np.int32)),
            "target illegal": variant(mask=illegal_target_mask),
            "game_index int64": variant(game_index=game_index.astype(np.int64)),
            "game_index 2-D": variant(game_index=game_index[:, None]),
            "game_index negative": variant(
                game_index=np.array([0, -1, 2, 3], np.int32)
            ),
            "game_index length": variant(game_index=game_index[:-1]),
        }
        for label, bad in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    data.validate(bad)

    # 2 ------------------------------------------------------------------ #
    def test_select_matches_independent_rng_reconstruction(self):
        game_ids = [3, 7, 7, 7, 1, 1, 12]
        dataset = make_dataset(game_ids, seed=5)
        seed, rows = 2024, 11

        batch, indices = data.select(dataset, seed, rows=rows)

        # Independent reconstruction of the specified two-stage stream.
        rng = np.random.default_rng(seed)
        unique_games = sorted(set(game_ids))  # [1, 3, 7, 12]
        game_picks = rng.integers(0, len(unique_games), size=rows)
        expected = []
        for pick in game_picks:
            game = unique_games[int(pick)]
            rows_of_game = [i for i, g in enumerate(game_ids) if g == game]
            offset = int(rng.integers(0, len(rows_of_game)))
            expected.append(rows_of_game[offset])
        expected = np.array(expected, dtype=np.int64)

        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(indices.shape, (rows,))
        np.testing.assert_array_equal(indices, expected)
        for key in data.MODEL_KEYS:
            np.testing.assert_array_equal(batch[key], dataset[key][expected])

        # Same seed -> identical draw.
        batch2, indices2 = data.select(dataset, seed, rows=rows)
        np.testing.assert_array_equal(indices2, indices)
        for key in data.MODEL_KEYS:
            np.testing.assert_array_equal(batch2[key], batch[key])

    # 3 ------------------------------------------------------------------ #
    def test_select_weights_games_equally_despite_unequal_lengths(self):
        lengths = {0: 1, 1: 9, 2: 90}
        game_ids = [g for g, count in lengths.items() for _ in range(count)]
        dataset = make_dataset(game_ids, seed=1)
        rows = 4000

        _, indices = data.select(dataset, 99, rows=rows)
        picked_games = dataset["game_index"][indices]

        for game in lengths:
            share = float(np.mean(picked_games == game))
            # Decision-weighted sampling would give 0.01 / 0.09 / 0.90;
            # game-balanced sampling gives ~1/3 each.
            self.assertGreater(share, 1 / 3 - 0.05, msg=f"game {game}: {share}")
            self.assertLess(share, 1 / 3 + 0.05, msg=f"game {game}: {share}")

        # Within the long game, decisions are actually spread out.
        long_rows = np.unique(indices[picked_games == 2])
        self.assertGreater(long_rows.size, 30)

    # 4 ------------------------------------------------------------------ #
    def test_select_returns_only_legal_model_fields(self):
        dataset = make_dataset([0, 0, 1, 2, 2, 2], seed=3)
        batch, indices = data.select(dataset, 7, rows=16)

        self.assertEqual(set(batch.keys()), set(data.MODEL_KEYS))
        self.assertNotIn("game_index", batch)

        self.assertEqual(batch["features"].dtype, np.float32)
        self.assertEqual(batch["features"].shape, (16, data.FEATURE_DIM))
        self.assertEqual(batch["mask"].dtype, np.bool_)
        self.assertEqual(batch["mask"].shape, (16, data.ACTION_DIM))
        self.assertEqual(batch["targets"].dtype, np.int32)
        self.assertEqual(batch["targets"].shape, (16,))

        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 6))
        self.assertTrue(np.all(batch["mask"].any(axis=1)))
        self.assertTrue(
            np.all(batch["mask"][np.arange(16), batch["targets"]])
        )
        self.assertTrue(np.all(np.isfinite(batch["features"])))

        # The batch itself validates once re-attached to a game index.
        rebuilt = dict(batch)
        rebuilt["game_index"] = dataset["game_index"][indices]
        self.assertIsNone(data.validate(rebuilt))

    # 5 ------------------------------------------------------------------ #
    def test_no_mutation_or_aliasing(self):
        dataset = make_dataset([5, 5, 8, 8, 8], seed=9)
        before = snapshot(dataset)

        batch, indices = data.select(dataset, 11, rows=8)
        for key in before:
            np.testing.assert_array_equal(dataset[key], before[key])
        for key in data.MODEL_KEYS:
            self.assertFalse(np.shares_memory(batch[key], dataset[key]), key)
        self.assertFalse(np.shares_memory(indices, dataset["game_index"]))

        # Writing into the batch must not leak back into the dataset.
        batch["features"][:] = 123.0
        batch["mask"][:] = True
        batch["targets"][:] = 0
        for key in before:
            np.testing.assert_array_equal(dataset[key], before[key])

        probs, masks, targets, game_index = hand_example()
        copies = [np.array(a, copy=True) for a in (probs, masks, targets, game_index)]
        data.summarize(probs, masks, targets, game_index)
        for original, copy in zip((probs, masks, targets, game_index), copies):
            np.testing.assert_array_equal(original, copy)

    # 6 ------------------------------------------------------------------ #
    def test_summarize_matches_hand_calculation(self):
        probs, masks, targets, game_index = hand_example()
        result = data.summarize(probs, masks, targets, game_index)

        ln2 = math.log(2.0)
        ln125 = math.log(1.25)  # -ln 0.8
        expected = {
            "rows": 3,
            "games": 2,
            # decisions weighted equally
            "decision_nll": (ln2 + ln125 + 0.0) / 3,
            "decision_accuracy": 2 / 3,
            "decision_target_mass": (0.5 + 0.8 + 1.0) / 3,
            # per game first (game 4 has two rows, game 9 one), then equal
            "game_nll": ((ln2 + ln125) / 2 + 0.0) / 2,
            "game_accuracy": ((0 + 1) / 2 + 1.0) / 2,
            "game_target_mass": ((0.5 + 0.8) / 2 + 1.0) / 2,
        }
        self.assertEqual(set(result), set(expected))
        self.assertIs(type(result["rows"]), int)
        self.assertIs(type(result["games"]), int)
        for key in expected:
            if key in ("rows", "games"):
                self.assertEqual(result[key], expected[key], key)
            else:
                self.assertIs(type(result[key]), float, key)
                self.assertAlmostEqual(result[key], expected[key], places=12, msg=key)

        # JSON-native round trip
        self.assertEqual(json.loads(json.dumps(result)), result)

        # Preserve positive subnormal targets through logarithms, including
        # extended precision on platforms where longdouble is wider than float64.
        tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
        wide = np.zeros((1, data.ACTION_DIM), dtype=np.longdouble)
        wide[0, 0], wide[0, 1] = 1, tiny
        legal = np.zeros((1, data.ACTION_DIM), dtype=bool)
        legal[0, :2] = True
        wide_result = data.summarize(wide, legal, np.array([1], np.int32), np.array([0], np.int32))
        self.assertAlmostEqual(wide_result["decision_nll"], float(-np.log(tiny)), places=10)
        self.assertAlmostEqual(wide_result["game_nll"], float(-np.log(tiny)), places=10)

        # float32 input is a real float dtype and must be accepted.
        result32 = data.summarize(
            probs.astype(np.float32), masks, targets, game_index
        )
        self.assertAlmostEqual(result32["game_accuracy"], 0.75, places=12)

    # 7 ------------------------------------------------------------------ #
    def test_summarize_rejects_malformed_probabilities(self):
        probs, masks, targets, game_index = hand_example()

        def with_probs(p):
            return (p, masks, targets, game_index)

        off_sum = probs.copy()
        off_sum[0, 1] = 0.4  # row sums to 0.9
        illegal_mass = probs.copy()
        illegal_mass[0, 13] = 1e-9  # illegal slot, sum still within atol
        illegal_mass[0, 0] = 0.5 - 1e-9
        negative = probs.copy()
        negative[0, 0] = 1.1
        negative[0, 1] = -0.1  # sums to 1 but out of range
        nan = probs.copy()
        nan[1, 2] = np.nan
        inf = probs.copy()
        inf[1, 2] = np.inf
        zero_target = probs.copy()
        zero_target[0, 0] = 1.0
        zero_target[0, 1] = 0.0  # target 1 now has zero mass
        one_hot_int = np.zeros_like(probs, dtype=np.int64)
        one_hot_int[np.arange(3), targets] = 1

        bad_mask = masks.copy()
        bad_mask[2, :] = False
        bad_targets = targets.copy()
        bad_targets[0] = 3  # illegal under masks row 0
        bad_games = game_index.copy()
        bad_games[1] = -4

        cases = {
            "list": with_probs(probs.tolist()),
            "int dtype": with_probs(one_hot_int),
            "complex dtype": with_probs(probs.astype(np.complex128)),
            "wrong width": with_probs(probs[:, :-1]),
            "1-D": with_probs(probs[0]),
            "empty": (
                np.zeros((0, data.ACTION_DIM)),
                np.zeros((0, data.ACTION_DIM), bool),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.int32),
            ),
            "row sum off": with_probs(off_sum),
            "illegal nonzero": with_probs(illegal_mass),
            "out of range": with_probs(negative),
            "nan": with_probs(nan),
            "inf": with_probs(inf),
            "zero target mass": with_probs(zero_target),
            "mask dtype": (probs, masks.astype(np.int8), targets, game_index),
            "mask shape": (probs, masks[:, :-1], targets, game_index),
            "mask empty row": (probs, bad_mask, targets, game_index),
            "target illegal": (probs, masks, bad_targets, game_index),
            "target dtype": (probs, masks, targets.astype(np.int64), game_index),
            "game_index negative": (probs, masks, targets, bad_games),
            "game_index dtype": (probs, masks, targets, game_index.astype(np.int64)),
        }
        for label, args in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    data.summarize(*args)

        # Boundary: a row sum inside the tolerance band is accepted.
        near = probs.copy()
        near[0, 0] = 0.5 + 5e-7
        self.assertEqual(data.summarize(near, masks, targets, game_index)["rows"], 3)

    # 8 ------------------------------------------------------------------ #
    def test_select_rejects_invalid_seed_and_rows(self):
        dataset = make_dataset([0, 1, 1], seed=2)

        for bad_seed in (True, False, -1, 2**32, 1.0, "3", None, np.int64(3)):
            with self.subTest(seed=repr(bad_seed)):
                with self.assertRaises(ValueError):
                    data.select(dataset, bad_seed, rows=4)

        for bad_rows in (0, -1, True, 2.0, "4", None, np.int64(4)):
            with self.subTest(rows=repr(bad_rows)):
                with self.assertRaises(ValueError):
                    data.select(dataset, 1, rows=bad_rows)

        # Inclusive/exclusive seed boundaries and the smallest batch.
        _, low = data.select(dataset, 0, rows=1)
        _, high = data.select(dataset, 2**32 - 1, rows=1)
        self.assertEqual(low.shape, (1,))
        self.assertEqual(high.shape, (1,))
        batch, indices = data.select(dataset, 5)  # default rows=64
        self.assertEqual(indices.shape, (64,))
        self.assertEqual(batch["features"].shape[0], 64)


if __name__ == "__main__":
    unittest.main()
