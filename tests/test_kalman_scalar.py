"""Tests for the scalar Kalman filter (M5 Rungs 1-2).

Rung 1 -- the measurement UPDATE: static 1D fusion of two noisy estimates of
one constant (the Kalman gain's DNA). The strongest tests pin the update to
physics / invariants rather than the code's exact algebra: precisions add
(1/P_hat == 1/P + 1/R), fusion order does not matter, and a valid measurement
can only shrink the variance.

Rung 2 -- the PREDICT step: propagating belief forward one step. Predict is
the only place P grows; ``+ Q`` is the drift term. Over many predicts with no
measurement, P = P0 + N*Q -- the scalar shadow of the M4 drift story.
"""

import numpy as np
import pytest

from bootins.kalman_scalar import kalman_update_scalar, kalman_predict_scalar

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


@pytest.mark.parametrize("x, P", [(1, 2), (1, 4), (2, 2), (3, 5), (4, 10)])
def test_kalman_scalar_no_noise(x, P):
    # a = 1, Q = 0 => predict is a no-op: a perfectly-known constant model
    # neither moves the estimate nor loses certainty.
    a = 1.0
    Q = 0.0
    x_prior, P_prior = kalman_predict_scalar(x, P, a, Q)

    np.testing.assert_allclose(x, x_prior)
    np.testing.assert_allclose(P, P_prior)


@pytest.mark.parametrize("x, P", [(1, 2), (1, 4), (2, 2), (3, 5), (4, 10)])
@pytest.mark.parametrize("Q", [1, 2, 3, 4, 5])
def test_kalman_scalar_pure_uncertainty(x, P, Q):
    # a = 1, Q > 0 => estimate unchanged but P_prior = P + Q exactly. Isolates
    # the +Q drift term -- the heart of the predict step.
    a = 1.0
    x_prior, P_prior = kalman_predict_scalar(x, P, a, Q)

    np.testing.assert_allclose(x, x_prior)
    np.testing.assert_allclose(P + Q, P_prior)


@pytest.mark.parametrize("x, P", [(1, 2), (1, 4), (2, 2), (3, 5), (4, 10)])
@pytest.mark.parametrize("a", [0.5, 2, 4, 0.1, 10])
def test_kalman_scalar_scaling(x, P, a):
    # Q = 0 => the dynamics scale the estimate by a and the VARIANCE by a**2
    # (variance carries two copies of the error). Pins the a-appears-squared
    # structure -- the scalar seed of Phi P Phi^T.
    Q = 0.0
    x_prior, P_prior = kalman_predict_scalar(x, P, a, Q)

    np.testing.assert_allclose(x * a, x_prior)
    np.testing.assert_allclose(a**2 * P, P_prior)


@pytest.mark.parametrize("x, P", [(1, 2), (1, 4), (2, 2), (3, 5), (4, 10)])
@pytest.mark.parametrize("a, Q", [(1, 1), (1, 2), (2, 3), (2, 4), (3, 5)])
def test_kalman_scalar_grows_P(x, P, a, Q):
    # Invariant, mirror of never_increases_uncertainty: with |a| >= 1 and
    # Q >= 0, predict can only GROW P. Coasting never buys certainty.
    _, P_prior = kalman_predict_scalar(x, P, a, Q)

    assert P_prior >= P


@pytest.mark.parametrize("P0", [0.0, 1.0, 4.0])
@pytest.mark.parametrize("Q", [0.5, 2.0, 5.0])
def test_kalman_scalar_predict_drifts_linearly(P0, Q):
    # Q2 as an oracle: N predicts with a=1 and no measurement accumulate
    # P = P0 + N*Q -- unbounded, linear drift. The scalar shadow of M4: a
    # bare filter with no aiding loses certainty forever.
    N = 100
    x, P = 0.0, P0
    for _ in range(N):
        x, P = kalman_predict_scalar(x, P, 1.0, Q)
    np.testing.assert_allclose(P, P0 + N * Q)


@pytest.mark.parametrize("x, P", [(1, 2), (1, 4), (2, 2), (3, 5), (4, 10)])
@pytest.mark.parametrize("a, Q", [(1, 1), (1, 2), (2, 3), (2, 4), (3, 5)])
@pytest.mark.parametrize("z, R", [(1, 2), (4, 4), (3, 0.5), (1, 10), (3, 3)])
def test_kalman_scalar_full(x, P, a, Q, z, R):
    # The full cycle: predict grows P, then update shrinks it back below the
    # post-predict prior. Predict-vs-update tug-of-war in miniature.
    x_prior, P_prior = kalman_predict_scalar(x, P, a, Q)
    _, P_hat = kalman_update_scalar(x_prior, P_prior, z, R)

    assert P_hat < P_prior