"""Public-phase stratified, game-balanced F16 sampling."""

import numpy as np

from pokemon_imitation.data import validate

__all__ = ["classify", "select"]


def classify(dataset):
    """Return fresh int32 action classes after schema and phase validation."""
    validate(dataset)
    phases = dataset["features"][:, 865:869]
    if not (((phases == 0) | (phases == 1)).all()
            and (phases.sum(axis=1) == 1).all()
            and (phases[:, 0] == 0).all()):
        raise ValueError("public phase must be exactly one of move, replace, revival")
    forced = (phases[:, 2] == 1) | (phases[:, 3] == 1)
    targets = dataset["targets"]
    if (forced & (targets < 8)).any():
        raise ValueError("replace and revival require a switch target")
    codes = np.zeros(targets.shape, dtype=np.int32)
    codes[targets >= 4] = 1
    codes[targets >= 8] = 2
    codes[forced] = 3
    return codes


def select(dataset, seed):
    """Draw 32/16/8/8 class rows, uniformly by game then row, and shuffle."""
    classes = classify(dataset)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a Python int in [0, 2**32)")
    if any(not (classes == code).any() for code in range(4)):
        raise ValueError("all four action classes are required")

    rng = np.random.default_rng(seed)
    indices = np.empty(64, dtype=np.int64)
    cursor = 0
    for code, quota in enumerate((32, 16, 8, 8)):
        rows = np.flatnonzero(classes == code)
        games = np.unique(dataset["game_index"][rows])
        picks = rng.integers(0, len(games), size=quota)
        for pick in picks:
            game_rows = rows[dataset["game_index"][rows] == games[pick]]
            indices[cursor] = game_rows[rng.integers(0, len(game_rows))]
            cursor += 1
    indices = indices[rng.permutation(64)]
    batch = {key: dataset[key][indices].copy()
             for key in ("features", "mask", "targets")}
    return batch, indices
