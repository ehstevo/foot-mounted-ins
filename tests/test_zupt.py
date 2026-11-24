"""Analytical and causal checks for SHOE/GLRT stance detection."""

import numpy as np
import pytest

from bootins.aiding.stance_glrt import ShoeStanceDetector, shoe_glrt, shoe_is_stance, shoe_scores, shoe_stance_flags
from bootins.estimation.mechanization import G_NED


SIGMA_A = 2.0
SIGMA_G = 3.0
GRAVITY = np.linalg.norm(G_NED)


def _unit(vector: object) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    return array / np.linalg.norm(array)


def _window(omega: object, specific_force: object, durations: object) -> list[tuple[np.ndarray, np.ndarray, float]]:
    return [(np.asarray(omega) * dt, np.asarray(specific_force) * dt, float(dt)) for dt in durations]


def test_perfect_rest_has_zero_score_in_an_arbitrary_body_orientation():
    force = GRAVITY * _unit([2.0, -1.0, 4.0])

    np.testing.assert_allclose(shoe_glrt(_window(np.zeros(3), force, [0.005, 0.01, 0.02]), SIGMA_A, SIGMA_G), 0.0, atol=1e-12)


def test_constant_rotation_and_force_mismatch_match_closed_form_score_terms():
    direction = _unit([1.0, -2.0, 1.5])
    omega = np.array([0.5, -0.25, 0.1])
    force = GRAVITY * direction + np.array([0.8, -0.4, 0.2])
    expected = np.dot(omega, omega) / SIGMA_G**2 + (np.linalg.norm(force) - GRAVITY) ** 2 / SIGMA_A**2

    np.testing.assert_allclose(shoe_glrt(_window(omega, force, [0.01, 0.02, 0.015]), SIGMA_A, SIGMA_G), expected, atol=1e-12)


def test_stance_threshold_is_strict_at_the_decision_boundary():
    omega = np.array([0.1, 0.2, 0.0])
    window = _window(omega, GRAVITY * _unit([1.0, 0.0, 1.0]), [0.01] * 4)
    threshold = np.dot(omega, omega) / SIGMA_G**2

    assert not shoe_is_stance(window, threshold, SIGMA_A, SIGMA_G)
    assert shoe_is_stance(window, threshold + 1e-12, SIGMA_A, SIGMA_G)


def test_batch_scores_and_flags_remain_causally_aligned_to_trailing_windows():
    rest = _window(np.zeros(3), np.array([0.0, 0.0, GRAVITY]), [0.01] * 4)

    scores = shoe_scores(rest, 3, SIGMA_A, SIGMA_G)
    flags = shoe_stance_flags(rest, 3, 1.0, SIGMA_A, SIGMA_G)

    assert np.isnan(scores[:2]).all()
    np.testing.assert_allclose(scores[2:], 0.0, atol=1e-12)
    np.testing.assert_array_equal(flags, [False, False, True, True])


def test_stateful_detector_matches_batch_and_reset_discards_pre_gap_evidence():
    increments = _window(np.zeros(3), np.array([0.0, 0.0, GRAVITY]), [0.01] * 3)
    detector = ShoeStanceDetector(2, 1.0, SIGMA_A, SIGMA_G)

    flags = [detector.update(increment) for increment in increments]
    np.testing.assert_array_equal(flags, shoe_stance_flags(increments, 2, 1.0, SIGMA_A, SIGMA_G))
    detector.reset()
    assert not detector.update(increments[-1])


def test_detector_refuses_undefined_physical_requests():
    with pytest.raises(ValueError, match="non-empty"):
        shoe_glrt([], SIGMA_A, SIGMA_G)
    with pytest.raises(ValueError, match="window_size"):
        ShoeStanceDetector(0, 1.0, SIGMA_A, SIGMA_G)
