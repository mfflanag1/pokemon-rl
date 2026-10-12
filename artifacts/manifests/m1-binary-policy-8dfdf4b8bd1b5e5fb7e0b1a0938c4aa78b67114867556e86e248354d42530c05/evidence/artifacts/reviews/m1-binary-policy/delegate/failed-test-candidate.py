from __future__ import annotations

import numpy as np
import pytest

from pokemon_binary.reference import (
    MINIMUM_LEGAL_MASS,
    dyadic_threshold,
    scalar_policy,
    selected_logp_derivative,
    word_selects_rare,
)


@pytest.mark.parametrize(
    ("probability", "threshold"),
    [
        (np.float32(0.5), 1 << 63),
        (np.float32(0.25), 1 << 62),
        (np.float32(2.0**-24), 1 << 40),
        (np.nextafter(np.float32(0.5), np.float32(0.0)), (1 << 63) - (1 << 39)),
        (np.nextafter(MINIMUM_LEGAL_MASS, np.float32(1.0)), (1 << 40) + (1 << 17)),
    ],
)
def test_exact_hand_computable_thresholds(probability, threshold):
    high, low = dyadic_threshold(probability)
    assert (high << 32) | low == threshold


def test_word_boundary_is_strict_and_uses_high_low_order():
    probability = np.float32(0.25)
    threshold = 1 << 62
    for word, expected in ((threshold - 1, True), (threshold, False), (threshold + 1, False)):
        assert word_selects_rare(word >> 32, word & 0xFFFFFFFF, probability) is expected


@pytest.mark.parametrize("bad", [0.25, np.float64(0.25), np.float32(np.nan), np.float32(0.0), np.float32(0.75)])
def test_threshold_rejects_invalid_probability(bad):
    with pytest.raises((TypeError, ValueError)):
        dyadic_threshold(bad)


@pytest.mark.parametrize("bad", [True, np.uint32(1), -1, 1 << 32])
def test_word_rejects_non_python_int_or_out_of_range(bad):
    with pytest.raises((TypeError, ValueError)):
        word_selects_rare(bad, 0, np.float32(0.25))


def test_equal_logits_designate_action_one_rare():
    result = scalar_policy(np.array([3.0, 3.0], np.float32), np.array([True, True]))
    assert result.rare_index == 1
    np.testing.assert_array_equal(result.probabilities, np.array([0.5, 0.5], np.float32))
    np.testing.assert_allclose(result.logp, np.log(np.array([0.5, 0.5], np.float32)), rtol=0, atol=0)


def test_mirrored_extremes_clip_the_rare_action():
    mask = np.array([True, True])
    left = scalar_policy(np.array([-100.0, 100.0], np.float32), mask)
    right = scalar_policy(np.array([100.0, -100.0], np.float32), mask)
    assert left.rare_index == 0 and right.rare_index == 1
    assert left.rare_mass == right.rare_mass == MINIMUM_LEGAL_MASS
    np.testing.assert_array_equal(left.probabilities, right.probabilities[::-1])
    assert np.all(np.isfinite(left.logp)) and np.all(np.isfinite(right.logp))


def test_masks_have_contract_values():
    logits = np.array([2.0, -1.0], np.float32)
    one = scalar_policy(logits, np.array([False, True]))
    terminal = scalar_policy(logits, np.array([False, False]))
    np.testing.assert_array_equal(one.probabilities, np.array([0.0, 1.0], np.float32))
    np.testing.assert_array_equal(one.logp, np.zeros(2, np.float32))
    assert terminal.rare_index == -1
    np.testing.assert_array_equal(terminal.probabilities, np.zeros(2, np.float32))


@pytest.mark.parametrize(
    ("logits", "mask"),
    [
        ([0.0, 1.0], np.array([True, True])),
        (np.array([0.0, 1.0]), np.array([True, True])),
        (np.array([0.0], np.float32), np.array([True, True])),
        (np.array([0.0, np.nan], np.float32), np.array([True, True])),
        (np.array([0.0, 1.0], np.float32), np.array([1, 1], np.int32)),
    ],
)
def test_policy_checks_inputs(logits, mask):
    with pytest.raises((TypeError, ValueError)):
        scalar_policy(logits, mask)


@pytest.mark.parametrize("selected", [0, 1])
def test_derivative_has_expected_sign_sum_and_finite_difference(selected):
    logits = np.array([0.7, -0.4], np.float32)
    mask = np.array([True, True])
    derivative = selected_logp_derivative(logits, mask, selected)
    assert derivative[0] == -derivative[1]
    assert bool(derivative[0] > 0) == (selected == 0)
    epsilon = np.float32(1e-3)
    plus = logits.copy()
    minus = logits.copy()
    plus[0] += epsilon
    minus[0] -= epsilon
    finite_difference = (
        scalar_policy(plus, mask).logp[selected] - scalar_policy(minus, mask).logp[selected]
    ) / np.float32(2.0 * epsilon)
    np.testing.assert_allclose(derivative[0], finite_difference, rtol=2e-3, atol=2e-4)


def test_derivative_equal_logits_convention_and_floor_are_explicit():
    mask = np.array([True, True])
    equal = np.array([0.0, 0.0], np.float32)
    np.testing.assert_array_equal(
        selected_logp_derivative(equal, mask, 1), np.array([-0.5, 0.5], np.float32)
    )
    clipped = np.array([100.0, -100.0], np.float32)
    np.testing.assert_array_equal(selected_logp_derivative(clipped, mask, 0), np.zeros(2, np.float32))
    np.testing.assert_array_equal(selected_logp_derivative(clipped, mask, 1), np.zeros(2, np.float32))
