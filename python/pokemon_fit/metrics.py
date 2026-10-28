"""Metrics and per-context gates for the finite-task fitting diagnostic."""

import numpy as np


_SPLITS = ("fit_train", "probe_same", "probe_variant")
_CONTEXTS = {str(c) for c in range(16)}


def summarize(probabilities, targets, contexts):
    """Summarize balanced observations, validating indices before any gather."""
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    contexts = np.asarray(contexts)
    if probabilities.ndim != 2 or probabilities.shape[1] != 14:
        raise ValueError("probabilities must have shape (N, 14)")
    n = probabilities.shape[0]
    if probabilities.dtype.kind not in "fiu":
        raise ValueError("probabilities must be real numeric values")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.all(np.abs(probabilities.sum(axis=1, dtype=np.float64) - 1) <= 5e-6):
        raise ValueError("probability rows must sum to one within 5e-6")
    for name, values, upper in (("targets", targets, 14), ("contexts", contexts, 16)):
        if values.shape != (n,) or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer vector of length N")
        if np.any(values < 0) or np.any(values >= upper):
            raise ValueError(f"{name} values must be in 0..{upper - 1}")
    counts = np.array([np.count_nonzero(contexts == c) for c in range(16)])
    if np.any(counts == 0) or np.any(counts != counts[0]):
        raise ValueError("all 16 contexts must be present with equal row counts")

    target_probability = probabilities[np.arange(n), targets]
    correct = probabilities.argmax(axis=1) == targets
    per_context = {}
    for c in range(16):
        mask = contexts == c
        per_context[str(c)] = {
            "n": int(counts[c]),
            "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()),
            "target_probability": float(target_probability[mask].mean()),
        }
    return {
        "n": int(n),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "mean_target_probability": float(target_probability.mean()),
        "min_context_target_probability": min(
            value["target_probability"] for value in per_context.values()
        ),
        "contexts": per_context,
    }


def gates(summaries, plan):
    """Apply each split's thresholds to every context, never to aggregates."""
    if set(summaries) != set(_SPLITS):
        raise ValueError("summaries must contain exactly the three required splits")
    result = {}
    for split in _SPLITS:
        contexts = summaries[split]["contexts"]
        if set(contexts) != _CONTEXTS:
            raise ValueError("each summary must contain all 16 contexts")
        thresholds = plan["thresholds"][split]
        accuracy_passed = all(
            contexts[c]["accuracy"] >= thresholds["minimum_context_accuracy"]
            for c in _CONTEXTS
        )
        probability_passed = all(
            contexts[c]["target_probability"]
            >= thresholds["minimum_context_target_probability"]
            for c in _CONTEXTS
        )
        result[split] = {
            "passed": bool(accuracy_passed and probability_passed),
            "accuracy_passed": bool(accuracy_passed),
            "probability_passed": bool(probability_passed),
        }
    return result
