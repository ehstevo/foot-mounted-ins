"""Compact analytical checks for the scalar Kalman predict/update equations."""

import numpy as np

from bootins.foundations.kalman_scalar import kalman_predict_scalar, kalman_update_scalar


def test_update_matches_closed_form_weighted_mean_and_variance():
    x_hat, variance_hat = kalman_update_scalar(2.0, 4.0, 10.0, 12.0)

    np.testing.assert_allclose(x_hat, 4.0)
    np.testing.assert_allclose(variance_hat, 3.0)


def test_update_adds_precision_and_is_measurement_order_independent():
    initial_mean, initial_variance = 3.0, 4.0
    first_measurement, second_measurement, measurement_variance = 5.0, 1.0, 6.0

    first_mean, first_variance = kalman_update_scalar(
        initial_mean, initial_variance, first_measurement, measurement_variance
    )
    forward_mean, forward_variance = kalman_update_scalar(
        first_mean, first_variance, second_measurement, measurement_variance
    )
    second_mean, second_variance = kalman_update_scalar(
        initial_mean, initial_variance, second_measurement, measurement_variance
    )
    reverse_mean, reverse_variance = kalman_update_scalar(
        second_mean, second_variance, first_measurement, measurement_variance
    )

    np.testing.assert_allclose(1.0 / first_variance, 1.0 / 4.0 + 1.0 / 6.0)
    np.testing.assert_allclose(forward_mean, reverse_mean)
    np.testing.assert_allclose(forward_variance, reverse_variance)


def test_update_has_correct_perfect_and_uninformative_measurement_limits():
    exact_mean, exact_variance = kalman_update_scalar(2.0, 4.0, 10.0, 0.0)
    weak_mean, weak_variance = kalman_update_scalar(2.0, 4.0, 10.0, 1e12)

    np.testing.assert_allclose((exact_mean, exact_variance), (10.0, 0.0), atol=1e-12)
    np.testing.assert_allclose((weak_mean, weak_variance), (2.0, 4.0), atol=1e-6)


def test_update_never_increases_variance_for_positive_measurement_noise():
    _, posterior_variance = kalman_update_scalar(-4.0, 10.0, 7.0, 0.5)

    assert 0.0 < posterior_variance < 10.0


def test_predict_matches_state_scaling_and_process_noise_closed_form():
    mean, variance = kalman_predict_scalar(3.0, 5.0, 2.0, 7.0)

    np.testing.assert_allclose(mean, 6.0)
    np.testing.assert_allclose(variance, 27.0)


def test_repeated_random_walk_predicts_add_process_variance_linearly():
    mean, variance = 1.5, 0.25
    for _ in range(20):
        mean, variance = kalman_predict_scalar(mean, variance, 1.0, 0.5)

    np.testing.assert_allclose(mean, 1.5)
    np.testing.assert_allclose(variance, 10.25)
