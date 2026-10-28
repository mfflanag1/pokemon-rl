"""Game-balanced sampling and descriptive metrics for the imitation dataset.

This module is intentionally self-contained: it depends on NumPy and the
standard library only, performs no I/O, spawns no subprocesses and imports
nothing else from the project.

Dataset layout (see ``validate``)::

    features    float32 (N, 928)  finite
    mask        bool    (N, 14)   at least one legal action per row
    targets     int32   (N,)      in 0..13 and legal under ``mask``
    game_index  int32   (N,)      non-negative; ids need not be contiguous

Every validation failure raises ``ValueError``.  Inputs are never mutated.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ACTION_DIM",
    "DATASET_KEYS",
    "FEATURE_DIM",
    "MAX_SEED",
    "MODEL_KEYS",
    "select",
    "summarize",
    "validate",
]

FEATURE_DIM = 928
ACTION_DIM = 14
MAX_SEED = 2**32  # exclusive upper bound for ``select`` seeds
DATASET_KEYS = frozenset({"features", "mask", "targets", "game_index"})
MODEL_KEYS = ("features", "mask", "targets")  # the only fields a batch may carry
_ROW_SUM_ATOL = 1e-6


# --------------------------------------------------------------------------- #
# Shared validators
# --------------------------------------------------------------------------- #
def _require_array(name, value, dtype, shape):
    """Require an ``np.ndarray`` with exactly ``dtype`` and ``shape``."""
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"{name} must be a numpy.ndarray, got {type(value).__name__}"
        )
    if value.dtype != np.dtype(dtype):
        raise ValueError(
            f"{name} must have dtype {np.dtype(dtype)}, got {value.dtype}"
        )
    if value.shape != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {value.shape}"
        )


def _check_mask(mask, n):
    _require_array("mask", mask, np.bool_, (n, ACTION_DIM))
    if not bool(mask.any(axis=1).all()):
        raise ValueError("mask must mark at least one legal action in every row")


def _check_targets(targets, mask, n):
    _require_array("targets", targets, np.int32, (n,))
    if bool((targets < 0).any()) or bool((targets >= ACTION_DIM).any()):
        raise ValueError(f"targets must lie in 0..{ACTION_DIM - 1}")
    if not bool(mask[np.arange(n), targets].all()):
        raise ValueError("every target must be legal under mask")


def _check_game_index(game_index, n):
    _require_array("game_index", game_index, np.int32, (n,))
    if bool((game_index < 0).any()):
        raise ValueError("game_index must be non-negative")


def _check_python_int(name, value):
    # ``bool`` is a subclass of ``int`` and must be rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be a Python int (not bool), got {type(value).__name__}"
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def validate(dataset):
    """Validate a dataset dict in place-free fashion; raise ``ValueError`` on defect.

    The dict must carry exactly the keys ``features``, ``mask``, ``targets``
    and ``game_index`` with the dtypes/shapes documented in the module
    docstring.  ``N`` (taken from ``features``) must be at least 1.  Nothing
    is returned and nothing is mutated.
    """
    if not isinstance(dataset, dict):
        raise ValueError(
            f"dataset must be a dict, got {type(dataset).__name__}"
        )
    keys = set(dataset.keys())
    if keys != DATASET_KEYS:
        raise ValueError(
            "dataset keys must be exactly "
            f"{sorted(DATASET_KEYS)}, got {sorted(map(str, keys))}"
        )

    features = dataset["features"]
    if not isinstance(features, np.ndarray):
        raise ValueError(
            f"features must be a numpy.ndarray, got {type(features).__name__}"
        )
    if features.dtype != np.dtype(np.float32):
        raise ValueError(f"features must have dtype float32, got {features.dtype}")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"features must have shape (N, {FEATURE_DIM}), got {features.shape}"
        )
    n = features.shape[0]
    if n < 1:
        raise ValueError("dataset must contain at least one row")
    if not bool(np.isfinite(features).all()):
        raise ValueError("features must be finite")

    _check_mask(dataset["mask"], n)
    _check_targets(dataset["targets"], dataset["mask"], n)
    _check_game_index(dataset["game_index"], n)


def select(dataset, seed, rows=64):
    """Draw a game-balanced batch of ``rows`` decisions.

    Sampling is two-stage with ``rng = np.random.default_rng(seed)``:

    1. ``game_picks = rng.integers(0, G, size=rows)`` indexes into the sorted
       unique game ids (``np.unique(game_index)``), i.e. games are chosen
       uniformly *with replacement* regardless of how many decisions they
       contain.
    2. For each pick, in order, ``offset = rng.integers(0, count)`` selects
       one of that game's row indices (ascending order), uniformly.

    Hence every complete game carries equal weight and decisions are weighted
    equally only within their game.  There is no outcome weighting or
    filtering of any kind.

    Returns ``(model_batch, indices)`` where ``model_batch`` holds freshly
    copied ``features``/``mask``/``targets`` of the selected rows (exactly
    those keys, nothing else) and ``indices`` is an ``int64`` array of shape
    ``(rows,)`` with the selected dataset row positions.
    """
    validate(dataset)
    _check_python_int("seed", seed)
    if not 0 <= seed < MAX_SEED:
        raise ValueError(f"seed must satisfy 0 <= seed < 2**32, got {seed}")
    _check_python_int("rows", rows)
    if rows < 1:
        raise ValueError(f"rows must be a positive int, got {rows}")

    game_index = dataset["game_index"]
    games = np.unique(game_index)  # sorted unique ids
    order = np.argsort(game_index, kind="stable")  # rows grouped by game, ascending
    grouped = game_index[order]
    starts = np.searchsorted(grouped, games, side="left")
    ends = np.searchsorted(grouped, games, side="right")

    rng = np.random.default_rng(seed)
    game_picks = rng.integers(0, games.size, size=rows)

    indices = np.empty(rows, dtype=np.int64)
    for i in range(rows):
        k = int(game_picks[i])
        start = int(starts[k])
        count = int(ends[k]) - start
        offset = int(rng.integers(0, count))
        indices[i] = order[start + offset]

    # Integer fancy indexing always allocates new arrays, so the batch never
    # aliases dataset storage.
    model_batch = {key: dataset[key][indices] for key in MODEL_KEYS}
    return model_batch, indices


def summarize(probabilities, masks, targets, game_index):
    """Compute descriptive decision- and game-level metrics.

    ``probabilities`` must be a real floating ``(N, 14)`` array with finite
    entries in ``[0, 1]``, rows summing to 1 within ``atol=1e-6`` (``rtol=0``),
    exactly 0 on illegal actions and strictly positive on each target.
    ``masks``, ``targets`` and ``game_index`` follow ``validate`` semantics.

    Returns a JSON-native dict with keys ``rows``, ``games``, ``decision_nll``,
    ``game_nll``, ``decision_accuracy``, ``game_accuracy``,
    ``decision_target_mass`` and ``game_target_mass``.  NLL uses the natural
    logarithm; accuracy uses greedy ``np.argmax`` (lowest index on ties).
    ``game_*`` values average per game first, then equally across games.
    No input is modified.
    """
    if not isinstance(probabilities, np.ndarray):
        raise ValueError(
            "probabilities must be a numpy.ndarray, "
            f"got {type(probabilities).__name__}"
        )
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise ValueError(
            f"probabilities must have a real floating dtype, got {probabilities.dtype}"
        )
    if probabilities.ndim != 2 or probabilities.shape[1] != ACTION_DIM:
        raise ValueError(
            f"probabilities must have shape (N, {ACTION_DIM}), got {probabilities.shape}"
        )
    n = probabilities.shape[0]
    if n < 1:
        raise ValueError("probabilities must contain at least one row")
    if not bool(np.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite")
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        raise ValueError("probabilities must lie in [0, 1]")

    _check_mask(masks, n)

    probs = probabilities.astype(np.result_type(probabilities.dtype, np.float64))  # never narrow before log
    row_sums = probs.sum(axis=1)
    if not bool((np.abs(row_sums - 1.0) <= _ROW_SUM_ATOL).all()):
        raise ValueError("each probability row must sum to 1 within atol=1e-6")
    if bool((probabilities[~masks] != 0).any()):
        raise ValueError("probabilities on illegal actions must be exactly 0")

    _check_targets(targets, masks, n)
    _check_game_index(game_index, n)

    target_mass = probs[np.arange(n), targets]
    if bool((target_mass <= 0).any()):
        raise ValueError("every target probability must be > 0")

    nll = -np.log(target_mass)
    correct = (np.argmax(probabilities, axis=1) == targets).astype(np.float64)

    games, inverse = np.unique(game_index, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    counts = np.bincount(inverse, minlength=games.size).astype(np.float64)

    def game_mean(values):
        sums = np.zeros(games.size, dtype=values.dtype)
        np.add.at(sums, inverse, values)
        per_game = sums / counts
        return float(per_game.mean())

    return {
        "rows": int(n),
        "games": int(games.size),
        "decision_nll": float(nll.mean()),
        "game_nll": game_mean(nll),
        "decision_accuracy": float(correct.mean()),
        "game_accuracy": game_mean(correct),
        "decision_target_mass": float(target_mass.mean()),
        "game_target_mass": game_mean(target_mass),
    }
