"""Independent scalar oracle for the binary-policy-v1 contract.

This module deliberately uses Python integer arithmetic for sampling thresholds.
It is a readable reference, not a vectorized implementation.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import numpy as np


MINIMUM_LEGAL_MASS = np.float32(2.0**-24)
_UINT32_MAX = (1 << 32) - 1


class ScalarPolicy(NamedTuple):
    probabilities: np.ndarray
    logp: np.ndarray
    rare_index: int
    rare_mass: np.float32


def _checked_probability(probability: object) -> np.float32:
    if not isinstance(probability, np.float32):
        raise TypeError("probability must be an actual numpy.float32 scalar")
    if not np.isfinite(probability):
        raise ValueError("probability must be finite")
    if probability < MINIMUM_LEGAL_MASS or probability > np.float32(0.5):
        raise ValueError("probability must be in [2^-24, 0.5]")
    return probability


def dyadic_threshold(probability: object) -> tuple[int, int]:
    """Return high/low uint32 words of the exact integer ``p * 2**64``."""

    probability = _checked_probability(probability)
    numerator, denominator = probability.item().as_integer_ratio()
    scaled = Fraction(numerator, denominator) * (1 << 64)
    if scaled.denominator != 1:  # Defensive: impossible for an in-range float32.
        raise ValueError("probability does not have an integral 64-bit threshold")
    threshold = scaled.numerator
    return threshold >> 32, threshold & _UINT32_MAX


def _checked_word(word: object, name: str) -> int:
    # bool is intentionally excluded even though it subclasses int.
    if type(word) is not int:
        raise TypeError(f"{name} must be a Python int")
    if not 0 <= word <= _UINT32_MAX:
        raise ValueError(f"{name} must be in uint32 range")
    return word


def word_selects_rare(high: object, low: object, probability: object) -> bool:
    """Whether the concatenated high/low word lies below the exact threshold."""

    high = _checked_word(high, "high")
    low = _checked_word(low, "low")
    threshold_high, threshold_low = dyadic_threshold(probability)
    word = (high << 32) | low
    threshold = (threshold_high << 32) | threshold_low
    return word < threshold


def _checked_policy_inputs(
    logits: object, mask: object
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(logits, np.ndarray) or logits.dtype != np.float32:
        raise TypeError("logits must be a numpy.ndarray with dtype float32")
    if logits.shape != (2,):
        raise ValueError("logits must have shape (2,)")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("mask must be a numpy.ndarray with dtype bool")
    if mask.shape != (2,):
        raise ValueError("mask must have shape (2,)")
    return logits, mask


def _rare_mass(logits: np.ndarray) -> tuple[int, np.float32, np.float32]:
    d = np.float32(logits[0] - logits[1])
    rare_index = 1 if d >= np.float32(0.0) else 0
    exponent = np.float32(-np.abs(d))
    exp_value = np.exp(exponent, dtype=np.float32)
    raw = np.float32(exp_value / np.float32(1.0 + exp_value))
    return rare_index, np.maximum(raw, MINIMUM_LEGAL_MASS), raw


def scalar_policy(logits: object, mask: object) -> ScalarPolicy:
    """Evaluate the contract's clipped two-action policy for one state."""

    logits, mask = _checked_policy_inputs(logits, mask)
    legal = int(mask[0]) + int(mask[1])
    probabilities = np.zeros(2, dtype=np.float32)
    logp = np.zeros(2, dtype=np.float32)
    if legal == 0:
        return ScalarPolicy(probabilities, logp, -1, np.float32(0.0))
    if legal == 1:
        only = 0 if mask[0] else 1
        probabilities[only] = np.float32(1.0)
        return ScalarPolicy(probabilities, logp, only, np.float32(0.0))

    rare_index, rare_mass, _ = _rare_mass(logits)
    common_index = 1 - rare_index
    probabilities[rare_index] = rare_mass
    probabilities[common_index] = np.float32(np.float32(1.0) - rare_mass)
    logp[rare_index] = np.log(rare_mass, dtype=np.float32)
    logp[common_index] = np.log1p(np.float32(-rare_mass), dtype=np.float32)
    return ScalarPolicy(probabilities, logp, rare_index, rare_mass)


def selected_logp_derivative(
    logits: object, mask: object, selected_action: object
) -> np.ndarray:
    """Analytic derivative of the selected action's log probability."""

    logits, mask = _checked_policy_inputs(logits, mask)
    if type(selected_action) is not int:
        raise TypeError("selected_action must be a Python int")
    if selected_action not in (0, 1):
        raise ValueError("selected_action must be 0 or 1")
    if not mask[selected_action]:
        raise ValueError("selected_action must be legal")
    if int(mask[0]) + int(mask[1]) < 2:
        return np.zeros(2, dtype=np.float32)

    d = np.float32(logits[0] - logits[1])
    rare_index, rare_mass, raw = _rare_mass(logits)
    # The clipping primitive is constant at equality with the floor as specified.
    if raw <= MINIMUM_LEGAL_MASS:
        return np.zeros(2, dtype=np.float32)
    sign = np.float32(-1.0 if d >= np.float32(0.0) else 1.0)
    dr_dd = np.float32(sign * rare_mass * np.float32(1.0 - rare_mass))
    if selected_action == rare_index:
        derivative_d = np.float32(dr_dd / rare_mass)
    else:
        derivative_d = np.float32(-dr_dd / np.float32(1.0 - rare_mass))
    return np.array([derivative_d, -derivative_d], dtype=np.float32)


# A descriptive alias useful to callers that phrase the operation in sampler terms.
probability_to_threshold_words = dyadic_threshold

