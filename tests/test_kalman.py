import numpy as np
import pytest

from bootins.kalman import kalman_update_scalar

# (x_prior, z) pairs reused across the value-driven cases.
CASES = [(1, 3), (2, 4), (5, 3), (-3, 1), (8, 2), (-4, -10)]


@pytest.mark.parametrize("x, z", CASES)
def test_kalman_scalar_symmetric(x, z):
    # Equal trust (P == R) => K = 0.5 => posterior is the exact midpoint,
    # and the variance halves.
    P = R = 4.0
    x_hat, P_hat = kalman_update_scalar(x, P, z, R)
    np.testing.assert_allclose(x_hat, (x + z) / 2)
    np.testing.assert_allclose(P_hat, P / 2)


@pytest.mark.parametrize("x, z", CASES)
def test_kalman_scalar_perfect_measurement(x, z):
    # R = 0 => K = 1 => trust the measurement completely, drop the prior,
    # and collapse the uncertainty to zero.
    P = 4.0
    R = 0.0
    x_hat, P_hat = kalman_update_scalar(x, P, z, R)
    np.testing.assert_allclose(x_hat, z)
    np.testing.assert_allclose(P_hat, 0.0, atol=1e-12)


@pytest.mark.parametrize("x, z", CASES)
def test_kalman_scalar_useless_measurement(x, z):
    # R -> infinity => K -> 0 => ignore the measurement, keep the prior
    # (estimate and variance both essentially unchanged).
    P = 4.0
    R = 1e12
    x_hat, P_hat = kalman_update_scalar(x, P, z, R)
    np.testing.assert_allclose(x_hat, x, atol=1e-6)
    np.testing.assert_allclose(P_hat, P, atol=1e-6)


@pytest.mark.parametrize(
    "P, R",
    [(1, 2), (2, 3), (4, 5), (8, 7), (12, 10)],
)
def test_kalman_scalar_precisions_add(P, R):
    # The fundamental identity: fusing independent estimates ADDS precisions
    # (1/variance). Pins the covariance update to the physics, independent of
    # the exact algebraic form of the code.
    x, z = 3.0, 4.0
    _, P_hat = kalman_update_scalar(x, P, z, R)
    np.testing.assert_allclose(1 / P_hat, 1 / P + 1 / R)


def test_kalman_scalar_order_independence():
    # Static fusion of two independent measurements must not depend on the
    # order they arrive in -- a direct consequence of precisions adding.
    P, R = 4.0, 6.0
    x, z1, z2 = 3.0, 5.0, 1.0

    x_a, P_a = kalman_update_scalar(x, P, z1, R)
    x_a, P_a = kalman_update_scalar(x_a, P_a, z2, R)

    x_b, P_b = kalman_update_scalar(x, P, z2, R)
    x_b, P_b = kalman_update_scalar(x_b, P_b, z1, R)

    np.testing.assert_allclose(x_a, x_b)
    np.testing.assert_allclose(P_a, P_b)


@pytest.mark.parametrize("x, z", CASES)
@pytest.mark.parametrize("P, R", [(1, 2), (4, 4), (10, 0.5), (0.5, 10)])
def test_kalman_scalar_never_increases_uncertainty(x, z, P, R):
    # Q1/Q2 as an invariant: a valid measurement (R > 0) can only shrink the
    # variance, and the gain is always a fraction in [0, 1] -- regardless of
    # what the measurement value actually is.
    K = P / (P + R)
    assert 0.0 <= K <= 1.0
    _, P_hat = kalman_update_scalar(x, P, z, R)
    assert P_hat <= P
