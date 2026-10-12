"""Cutoff-aware paired statistics; see the frozen task contract."""

import math
import numbers

import numpy as np


def _outcomes(value, name):
    """Return a validated integer outcome array."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular integer array") from exc

    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape [N, 2]")
    if not 2 <= array.shape[0] <= 100_000:
        raise ValueError(f"{name} must contain between 2 and 100000 pairs")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must have an integer dtype")
    if not np.isin(array, (-1, 0, 1)).all():
        raise ValueError(f"{name} contains an invalid outcome")
    return array


def summarize(candidate, baseline, *, alpha=0.05, family_size=6):
    candidate = _outcomes(candidate, "candidate")
    baseline = _outcomes(baseline, "baseline")
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline must have matching shapes")

    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real):
        raise ValueError("alpha must be a finite real number")
    try:
        valid_alpha = math.isfinite(alpha) and 0 < alpha < 1
    except (OverflowError, TypeError, ValueError):
        valid_alpha = False
    if not valid_alpha:
        raise ValueError("alpha must be strictly between zero and one")

    if (
        isinstance(family_size, (bool, np.bool_))
        or not isinstance(family_size, (int, np.integer))
        or not 1 <= family_size <= 10_000
    ):
        raise ValueError("family_size must be an integer from 1 to 10000")

    pairs = candidate.shape[0]
    candidate_win = candidate == 1
    candidate_not_loss = candidate != 0
    baseline_win = baseline == 1
    baseline_not_loss = baseline != 0

    outcome_lower = float(candidate_win.mean())
    outcome_upper = float(candidate_not_loss.mean())
    baseline_lower = float(baseline_win.mean())
    baseline_upper = float(baseline_not_loss.mean())
    secured_win_change = float((candidate_win.astype(np.int8) - baseline_win).mean())
    worst_case_improvement = float(
        (candidate_win.astype(np.int8) - baseline_not_loss).mean()
    )
    best_case_improvement = float(
        (candidate_not_loss.astype(np.int8) - baseline_win).mean()
    )

    radius = math.sqrt(
        (math.log(int(family_size)) - math.log(alpha)) / (2 * pairs)
    )

    return {
        "pairs": int(pairs),
        "games": int(2 * pairs),
        "wins": int(candidate_win.sum()),
        "losses": int((candidate == 0).sum()),
        "cutoffs": int((candidate == -1).sum()),
        "outcome_lower": outcome_lower,
        "outcome_upper": outcome_upper,
        "baseline_lower": baseline_lower,
        "baseline_upper": baseline_upper,
        "secured_win_change": secured_win_change,
        "worst_case_improvement": worst_case_improvement,
        "best_case_improvement": best_case_improvement,
        "adjusted_win_lower": float(max(0.0, outcome_lower - radius)),
        "adjusted_improvement_lower": float(
            max(-1.0, worst_case_improvement - 2 * radius)
        ),
    }
