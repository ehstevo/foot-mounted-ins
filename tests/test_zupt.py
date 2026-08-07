"""Tests for the M6 SHOE / GLRT stance detector.

Each oracle builds a window whose score is known in closed form, so the tests
pin the detector math directly instead of only checking vague "low vs high"
behavior. That matters here because the easy mistakes are silent ones: wrong
sample axis, forgetting the dt divide, or comparing against a fixed gravity axis.
"""
import numpy as np
import pytest

from bootins.mechanization import G_NED
from bootins.zupt import shoe_glrt, shoe_is_stance

SIGMA_A = 2.0
SIGMA_G = 3.0
G = np.linalg.norm(G_NED)


def _unit(v):
    """Return v normalized to unit length."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _window_from_rates(omega, f_body, dt_samples):
    """Pack constant per-sample rates into the repo's increment convention."""
    omega = np.asarray(omega, dtype=float)
    f_body = np.asarray(f_body, dtype=float)
    return [
        (omega * dt, f_body * dt, float(dt))
        for dt in np.asarray(dt_samples, dtype=float)
    ]


def test_perfect_rest_in_any_orientation():
    """A planted foot scores zero in any fixed body orientation."""
    u_body = _unit([2.0, -1.0, 4.0])     # arbitrary, non-axis-aligned orientation
    f_rest = G * u_body
    window = _window_from_rates(np.zeros(3), f_rest, [0.005, 0.010, 0.020, 0.015])

    glrt = shoe_glrt(window, SIGMA_A, SIGMA_G)

    np.testing.assert_allclose(glrt, 0.0, atol=1e-12)


def test_constant_rotation_matches_gyro_term():
    """Constant rotation with a fixed gravity vector leaves only the gyro term."""
    u_body = _unit([1.0, 2.0, -1.0])
    f_rest = G * u_body
    omega = np.array([0.5, -0.25, 0.1])
    window = _window_from_rates(omega, f_rest, [0.006, 0.010, 0.014, 0.009])

    glrt = shoe_glrt(window, SIGMA_A, SIGMA_G)
    expected = np.dot(omega, omega) / SIGMA_G**2

    np.testing.assert_allclose(glrt, expected, atol=1e-12)


def test_constant_extra_specific_force_parallel_to_gravity():
    """A constant force parallel to gravity contributes only the accel mismatch."""
    u_body = _unit([-1.0, 1.0, 2.0])
    extra = 0.8
    f_motion = (G + extra) * u_body
    window = _window_from_rates(np.zeros(3), f_motion, [0.010, 0.010, 0.010, 0.010])

    glrt = shoe_glrt(window, SIGMA_A, SIGMA_G)
    expected = extra**2 / SIGMA_A**2

    np.testing.assert_allclose(glrt, expected, atol=1e-12)


def test_constant_extra_specific_force_general_vector():
    """A general constant force vector gives the closed-form norm mismatch."""
    u_body = _unit([1.0, -2.0, 1.5])
    f_motion = G * u_body + np.array([0.8, -0.4, 0.2])
    window = _window_from_rates(np.zeros(3), f_motion, [0.010, 0.020, 0.015, 0.010])

    glrt = shoe_glrt(window, SIGMA_A, SIGMA_G)
    expected = (np.linalg.norm(f_motion) - G)**2 / SIGMA_A**2

    np.testing.assert_allclose(glrt, expected, atol=1e-12)


def test_empty_window_raises():
    with pytest.raises(ValueError, match="non-empty"):
        shoe_glrt([], SIGMA_A, SIGMA_G)


def test_nonpositive_dt_raises():
    window = _window_from_rates(np.zeros(3), G * _unit([0.0, 0.0, 1.0]), [0.010, 0.000])

    with pytest.raises(ValueError, match="dt"):
        shoe_glrt(window, SIGMA_A, SIGMA_G)


def test_nonpositive_noise_scales_raise():
    window = _window_from_rates(np.zeros(3), G * _unit([0.0, 1.0, 0.0]), [0.010, 0.020])

    with pytest.raises(ValueError, match="sigma_a"):
        shoe_glrt(window, 0.0, SIGMA_G)
    with pytest.raises(ValueError, match="sigma_g"):
        shoe_glrt(window, SIGMA_A, 0.0)


def test_near_zero_mean_specific_force_raises():
    # Two equal-and-opposite body-force samples cancel, so there is no single
    # gravity direction the detector can fit to the window.
    window = [
        (np.zeros(3), np.array([G, 0.0, 0.0]) * 0.010, 0.010),
        (np.zeros(3), np.array([-G, 0.0, 0.0]) * 0.010, 0.010),
    ]

    with pytest.raises(ValueError, match="gravity direction is undefined"):
        shoe_glrt(window, SIGMA_A, SIGMA_G)


def test_shoe_is_stance_uses_strict_threshold():
    """Equality with the threshold is non-stance: favor precision over recall."""
    u_body = _unit([1.0, 0.0, 1.0])
    f_rest = G * u_body
    omega = np.array([0.1, 0.2, 0.0])
    window = _window_from_rates(omega, f_rest, [0.010, 0.010, 0.010, 0.010])
    threshold = np.dot(omega, omega) / SIGMA_G**2

    assert not shoe_is_stance(window, threshold, SIGMA_A, SIGMA_G)
    assert shoe_is_stance(window, threshold + 1e-12, SIGMA_A, SIGMA_G)
