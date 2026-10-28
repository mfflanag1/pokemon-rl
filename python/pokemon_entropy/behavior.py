"""Descriptive metrics for masked, fourteen-action decisions."""

import numpy as np


def describe(probabilities, masks, phases):
    """Validate decisions and return JSON-safe aggregate behavior metrics.

    Switch probability is conditioned on move-phase decisions with both a
    legal move and a legal switch. Entropy normalization counts all legal
    actions, including those assigned zero probability. Inputs are not changed.
    """
    try:
        raw = np.asarray(probabilities)
        legal = np.asarray(masks)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must be rectangular arrays") from exc
    if raw.ndim != 2 or raw.shape[1] != 14:
        raise ValueError("probabilities must have shape (N, 14)")
    if raw.dtype.kind not in "iuf":
        raise ValueError("probabilities must be real numeric values")
    if legal.dtype.kind != "b" or legal.shape != raw.shape:
        raise ValueError("masks must be boolean with shape (N, 14)")
    n = raw.shape[0]
    if (not isinstance(phases, (list, tuple)) or len(phases) != n
            or any(not isinstance(p, str) or p not in
                   {"move", "replace", "revival"} for p in phases)):
        raise ValueError("phases must contain one valid decision phase per row")
    # Check bounds before conversion so narrowing cannot conceal invalid inputs.
    if not np.all(np.isfinite(raw)) or np.any(raw < 0) or np.any(raw > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    p = raw.astype(np.float64, copy=True)
    counts = legal.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("each decision must have a legal action")
    if not np.all(np.isclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("probability rows must sum to one")
    if np.any(raw[~legal] != 0):
        raise ValueError("illegal actions must have exactly zero probability")
    move = np.asarray([phase == "move" for phase in phases], dtype=bool)
    opportunity = move & legal[:, :8].any(axis=1) & legal[:, 8:].any(axis=1)
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive])
    entropy = -terms.sum(axis=1)
    normalized = np.zeros(n, dtype=np.float64)
    multiple = counts > 1
    normalized[multiple] = entropy[multiple] / np.log(counts[multiple])
    return {
        "rows": int(n),
        "move_rows": int(move.sum()),
        "switch_opportunity_rows": int(opportunity.sum()),
        "replacement_rows": int(n - move.sum()),
        "mean_switch_probability": (
            float(p[opportunity, 8:].sum(axis=1).mean())
            if opportunity.any() else None
        ),
        "mean_entropy": float(entropy.mean()) if n else None,
        "mean_normalized_entropy": float(normalized.mean()) if n else None,
    }
