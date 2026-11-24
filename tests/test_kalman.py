"""Compact numerical checks for the multivariate Kalman filter."""

import numpy as np

from bootins.foundations.kalman import kalman_predict, kalman_update
from bootins.foundations.kalman_scalar import kalman_predict_scalar, kalman_update_scalar


DT = 0.1
PHI_CV = np.array([[1.0, DT], [0.0, 1.0]])
H_POS = np.array([[1.0, 0.0]])


def test_predict_reduces_to_the_independent_scalar_implementation():
    scalar_mean, scalar_variance = kalman_predict_scalar(-4.0, 10.0, 0.5, 3.0)
    mean, covariance = kalman_predict(
        np.array([-4.0]), np.array([[10.0]]), np.array([[0.5]]), np.array([[3.0]])
    )

    np.testing.assert_allclose(mean, [scalar_mean])
    np.testing.assert_allclose(covariance, [[scalar_variance]])


def test_update_reduces_to_the_independent_scalar_implementation():
    scalar_mean, scalar_variance = kalman_update_scalar(3.0, 5.0, -2.0, 0.5)
    mean, covariance = kalman_update(
        np.array([3.0]), np.array([[5.0]]), np.array([-2.0]), np.array([[1.0]]), np.array([[0.5]])
    )

    np.testing.assert_allclose(mean, [scalar_mean])
    np.testing.assert_allclose(covariance, [[scalar_variance]])


def test_constant_velocity_predict_builds_cross_covariance():
    _, covariance = kalman_predict(
        np.array([1.0, 0.0]), np.diag([1.0, 4.0]), PHI_CV, np.diag([0.01, 0.01])
    )

    np.testing.assert_allclose(covariance, covariance.T)
    np.testing.assert_allclose(covariance[0, 1], DT * 4.0)
    np.testing.assert_allclose(covariance[0, 0], 1.0 + DT**2 * 4.0 + 0.01)


def test_position_measurement_corrects_only_correlated_velocity():
    state, covariance = kalman_predict(
        np.array([1.0, 0.0]), np.diag([1.0, 4.0]), PHI_CV, np.diag([0.01, 0.01])
    )
    corrected, _ = kalman_update(state, covariance, np.array([2.0]), H_POS, np.array([[1.0]]))

    assert corrected[1] > 0.0

    uncoupled_state, uncoupled_covariance = kalman_predict(
        np.array([1.0, 0.0]), np.diag([1.0, 4.0]), np.eye(2), np.diag([0.01, 0.01])
    )
    uncoupled_corrected, _ = kalman_update(
        uncoupled_state, uncoupled_covariance, np.array([2.0]), H_POS, np.array([[1.0]])
    )
    np.testing.assert_allclose(uncoupled_corrected[1], uncoupled_state[1])


def test_update_uses_joseph_covariance_with_symmetric_psd_result():
    prior_covariance = np.array([[2.0, 0.5], [0.5, 4.0]])
    state, covariance = kalman_predict(
        np.array([1.0, 0.0]), prior_covariance, PHI_CV, np.diag([0.01, 0.01])
    )
    _, posterior_covariance = kalman_update(
        state, covariance, np.array([2.0]), H_POS, np.array([[1.0]])
    )
    innovation_covariance = H_POS @ covariance @ H_POS.T + 1.0
    gain = covariance @ H_POS.T @ np.linalg.inv(innovation_covariance)
    short_form = (np.eye(2) - gain @ H_POS) @ covariance

    np.testing.assert_allclose(posterior_covariance, short_form)
    np.testing.assert_allclose(posterior_covariance, posterior_covariance.T)
    assert np.all(np.linalg.eigvalsh(posterior_covariance) >= -1e-12)


def test_measurement_covariance_sets_information_limits():
    state = np.array([1.0, 0.0])
    covariance = np.diag([2.0, 4.0])

    exact_state, exact_covariance = kalman_update(
        state, covariance, np.array([5.0]), H_POS, np.array([[1e-12]])
    )
    weak_state, weak_covariance = kalman_update(
        state, covariance, np.array([5.0]), H_POS, np.array([[1e12]])
    )

    np.testing.assert_allclose(exact_state[0], 5.0, atol=1e-5)
    np.testing.assert_allclose(exact_covariance[0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(weak_state, state, atol=1e-6)
    np.testing.assert_allclose(weak_covariance, covariance, atol=1e-3)
