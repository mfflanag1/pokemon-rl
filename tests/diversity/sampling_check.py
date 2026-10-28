"""Run directly with Python; uses only NumPy and the standard library."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from pokemon_balanced.sampling import classify, select
from pokemon_diversity.sampling import PreparedSampler


def dataset():
    records = [(game, code, offset)
               for game, counts in ((91, (1, 3, 2, 4)), (2, (4, 1, 3, 2)),
                                    (7001, (2, 4, 1, 3)))
               for code, count in enumerate(counts) for offset in range(count)]
    # Interleave games and classes without relying on random fixture generation.
    records = records[::2] + records[1::2][::-1]
    n = len(records)
    features = np.arange(n * 928, dtype=np.float32).reshape(n, 928)
    features[:, 865:869] = 0
    targets = np.empty(n, dtype=np.int32)
    for i, (_, code, offset) in enumerate(records):
        features[i, 866 if code < 3 else 867 + offset % 2] = 1
        targets[i] = (offset % 4, 4 + offset % 4, 8 + offset % 6,
                      8 + offset % 6)[code]
    mask = np.zeros((n, 14), dtype=bool)
    mask[np.arange(n), targets] = True
    return dict(features=features, mask=mask, targets=targets,
                game_index=np.array([r[0] for r in records], dtype=np.int32))


class SamplingCheck(unittest.TestCase):
    def assertDrawEqual(self, left, right):
        self.assertEqual(set(left[0]), {"features", "mask", "targets"})
        np.testing.assert_array_equal(left[1], right[1])
        self.assertEqual(left[1].dtype, np.dtype(np.int64))
        for key in left[0]:
            self.assertEqual(left[0][key].dtype, right[0][key].dtype)
            self.assertEqual(left[0][key].tobytes(), right[0][key].tobytes())

    def test_legacy_equality(self):
        data = dataset()
        original = {k: v.copy() for k, v in data.items()}
        sampler = PreparedSampler(data)
        for seed in range(25):
            self.assertDrawEqual(sampler.sample(seed), select(data, seed))
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])

    def test_quotas_and_legal_labels(self):
        data = dataset()
        batch, indices = PreparedSampler(data).sample(41)
        np.testing.assert_array_equal(np.bincount(classify(data)[indices]),
                                      [32, 16, 8, 8])
        self.assertTrue(batch["mask"][np.arange(64), batch["targets"]].all())

    def test_seeds_and_no_revalidation(self):
        data = dataset()
        with patch("pokemon_diversity.sampling.classify", wraps=classify) as checked:
            sampler = PreparedSampler(data)
            self.assertEqual(checked.call_count, 1)
            with patch("pokemon_imitation.data.validate", side_effect=AssertionError), \
                 patch("pokemon_diversity.sampling.classify", side_effect=AssertionError), \
                 patch("numpy.isfinite", side_effect=AssertionError):
                for seed in (0, 2**32 - 1):
                    self.assertDrawEqual(sampler.sample(seed), sampler.sample(seed))
        for seed in (True, False, -1, 2**32, 1.0, np.int64(1), None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                sampler.sample(seed)
        for seed in (0, 2**32 - 1):
            self.assertDrawEqual(sampler.sample(seed), select(data, seed))

    def test_caller_mutation_isolation(self):
        data = dataset()
        sampler = PreparedSampler(data)
        expected = select(data, 8)
        for array in data.values():
            array[...] = 0
        data.clear()
        self.assertDrawEqual(sampler.sample(8), expected)

    def test_output_mutation_isolation(self):
        sampler = PreparedSampler(dataset())
        expected = sampler.sample(19)
        batch, indices = sampler.sample(19)
        for array in batch.values():
            array[...] = 0
        indices[...] = -1
        batch.clear()
        self.assertDrawEqual(sampler.sample(19), expected)

    def test_missing_family(self):
        for code in range(4):
            data = dataset()
            keep = classify(data) != code
            with self.subTest(code=code), self.assertRaises(ValueError):
                PreparedSampler({k: v[keep] for k, v in data.items()})

    def test_malformed_public_phase(self):
        for phase in ([0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0],
                      [0, .5, .5, 0], [0, 0, 1, 0], [0, 0, 0, 1]):
            data = dataset()
            row = int(np.flatnonzero(data["targets"] < 8)[0])
            data["features"][row, 865:869] = phase
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                PreparedSampler(data)

    def test_malformed_schema(self):
        def illegal(data):
            data["mask"][0, data["targets"][0]] = False
            data["mask"][0, (data["targets"][0] + 1) % 14] = True

        mutations = [lambda d: d["features"].__setitem__((0, 0), np.nan),
                     lambda d: d["features"].__setitem__((0, 0), np.inf),
                     lambda d: d.update(features=d["features"][:, :-1]),
                     lambda d: d.update(features=d["features"].astype(np.float64)),
                     lambda d: d["targets"].__setitem__(0, 14), illegal,
                     lambda d: d.update(extra=np.zeros(1)),
                     lambda d: d["game_index"].__setitem__(0, -1)]
        for i, mutate in enumerate(mutations):
            data = dataset()
            mutate(data)
            with self.subTest(case=i), self.assertRaises(ValueError):
                PreparedSampler(data)


if __name__ == "__main__":
    unittest.main()
