"""Independent sampler checks; run with python -B tests/balanced/sampling_check.py."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np

from pokemon_balanced.sampling import classify, select


def fixture(rows=None):
    # Each tuple explicitly specifies public phase, action, and game.
    if rows is None:
        rows = [(1, 0, 9), (1, 3, 2), (1, 4, 9), (1, 7, 2),
                (1, 8, 9), (1, 13, 2), (2, 8, 9), (3, 13, 2)]
    n = len(rows)
    features = np.zeros((n, 928), dtype=np.float32)
    mask = np.zeros((n, 14), dtype=bool)
    for i, (phase, target, _) in enumerate(rows):
        features[i, 865 + phase] = 1
        features[i, 0] = i
        mask[i, target] = True
    return dict(features=features, mask=mask,
                targets=np.array([r[1] for r in rows], dtype=np.int32),
                game_index=np.array([r[2] for r in rows], dtype=np.int32))


class SamplingChecks(unittest.TestCase):
    def test_semantic_classes(self):
        data = fixture()
        result = classify(data)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(result.dtype, np.dtype(np.int32))
        result[:] = 99
        np.testing.assert_array_equal(classify(data), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_phase_and_target_consistency(self):
        for phase in ([0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0],
                      [0, .5, .5, 0], [0, 2, -1, 0]):
            data = fixture()
            data['features'][0, 865:869] = phase
            for operation in (classify, lambda d: select(d, 0)):
                with self.subTest(phase=phase), self.assertRaises(ValueError):
                    operation(data)
        for phase in (2, 3):
            for target in (0, 7):
                with self.subTest(phase=phase, target=target), self.assertRaises(ValueError):
                    classify(fixture([(phase, target, 0)]))

    def test_all_classes_required(self):
        rows = [(1, 0, 0), (1, 4, 0), (1, 8, 0), (3, 8, 0)]
        for missing in range(4):
            data = fixture(rows[:missing] + rows[missing + 1:])
            self.assertEqual(len(classify(data)), 3)
            with self.assertRaises(ValueError):
                select(data, 1)

    def test_rng_reconstruction_and_shuffle(self):
        # Interleaved, unsorted game IDs and unequal per-game row counts.
        rows = [(p, t, g) for g in (19, 2, 19, 7)
                for p, t in ((1, 2), (1, 6), (1, 11), (3, 12))]
        data = fixture(rows)
        rng = np.random.default_rng(314)
        expected = []
        for code, count in enumerate((32, 16, 8, 8)):
            candidates = list(range(code, len(rows), 4))
            games = sorted({rows[i][2] for i in candidates})
            chosen_games = rng.integers(0, len(games), size=count)
            for chosen in chosen_games:
                eligible = [i for i in candidates if rows[i][2] == games[chosen]]
                expected.append(eligible[int(rng.integers(0, len(eligible)))])
        expected = np.array(expected, dtype=np.int64)[rng.permutation(64)]
        batch, actual = select(data, 314)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(np.int64))
        for key in batch:
            np.testing.assert_array_equal(batch[key], data[key][expected])

    def test_quotas_and_seed_variation(self):
        data = fixture()
        _, first = select(data, 0)
        _, again = select(data, 0)
        _, other = select(data, 2**32 - 1)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(first.shape, (64,))
        for indices in (first, other):
            np.testing.assert_array_equal(
                np.bincount(np.array([0, 0, 1, 1, 2, 2, 3, 3])[indices], minlength=4),
                [32, 16, 8, 8])

    def test_equal_game_weight_with_unequal_counts(self):
        data = fixture([(1, 0, 17)] * 100 + [(1, 0, 4)]
                       + [(1, 4, 0), (1, 8, 0), (2, 8, 0)])
        small_game = 0
        total = 0
        for seed in range(100):
            _, indices = select(data, seed)
            normal = indices[indices <= 100]
            small_game += int((normal == 100).sum())
            total += len(normal)
        self.assertEqual(total, 3200)
        self.assertTrue(.45 < small_game / total < .55)

    def test_model_only_copies_and_no_mutation(self):
        data = fixture()
        before = {key: value.copy() for key, value in data.items()}
        batch, indices = select(data, 8)
        self.assertEqual(set(batch), {'features', 'mask', 'targets'})
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])
        for key, value in batch.items():
            self.assertFalse(np.shares_memory(value, data[key]))
            value[:] = 0
            np.testing.assert_array_equal(data[key], before[key])
        indices[:] = 0
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])

    def test_seed_and_inherited_schema_validation(self):
        for seed in (True, False, np.int64(1), np.uint32(1), 1.0, '1', None, -1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                select(fixture(), seed)
        mutations = [
            lambda d: d.update(extra=np.zeros(1)),
            lambda d: d.pop('game_index'),
            lambda d: d.update(features=d['features'].astype(np.float64)),
            lambda d: d.update(features=d['features'][:, :-1]),
            lambda d: d['features'].__setitem__((0, 0), np.nan),
            lambda d: d.update(mask=d['mask'].astype(np.int32)),
            lambda d: d['mask'].__setitem__(0, False),
            lambda d: d.update(targets=d['targets'].astype(np.int64)),
            lambda d: d['targets'].__setitem__(0, 14),
            lambda d: d['targets'].__setitem__(0, 1),
            lambda d: d['game_index'].__setitem__(0, -1),
            lambda d: d.update(game_index=d['game_index'][:-1]),
        ]
        for mutation in mutations:
            data = fixture()
            mutation(data)
            for operation in (classify, lambda d: select(d, 0)):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    operation(data)


if __name__ == '__main__':
    unittest.main()
