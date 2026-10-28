"""Prepared, snapshot-isolated version of the balanced family sampler."""

import numpy as np

from pokemon_balanced.sampling import classify

__all__ = ["PreparedSampler"]


class PreparedSampler:
    """Validate once and retain private arrays and ordered family/game rows.

    Construction and sampling never mutate the caller's data or global RNG.
    Each draw owns its batch and indices; sampling only reads prepared state.
    """

    __slots__ = ("__arrays", "__groups")

    def __init__(self, dataset):
        classes = classify(dataset)
        if any(not (classes == code).any() for code in range(4)):
            raise ValueError("all four action classes are required")
        arrays = {key: value.copy() for key, value in dataset.items()}
        for value in arrays.values():
            value.flags.writeable = False
        groups = []
        for code in range(4):
            rows = np.flatnonzero(classes == code)
            # Stable sorting preserves ascending row positions within each game.
            order = np.argsort(arrays["game_index"][rows], kind="stable")
            rows = rows[order]
            games = arrays["game_index"][rows]
            boundaries = np.flatnonzero(games[1:] != games[:-1]) + 1
            family = tuple(np.split(rows, boundaries))
            for game_rows in family:
                game_rows.flags.writeable = False
            groups.append(family)
        self.__arrays = arrays
        self.__groups = tuple(groups)

    def sample(self, seed):
        """Return the exact legacy 32/16/8/8 draw for a Python integer seed."""
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be a Python int in [0, 2**32)")
        rng = np.random.default_rng(seed)
        indices = np.empty(64, dtype=np.int64)
        cursor = 0
        for family, quota in zip(self.__groups, (32, 16, 8, 8)):
            picks = rng.integers(0, len(family), size=quota)
            for pick in picks:
                rows = family[pick]
                indices[cursor] = rows[rng.integers(0, len(rows))]
                cursor += 1
        indices = indices[rng.permutation(64)]
        batch = {key: self.__arrays[key][indices].copy()
                 for key in ("features", "mask", "targets")}
        return batch, indices
