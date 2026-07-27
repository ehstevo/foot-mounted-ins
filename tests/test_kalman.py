"""Tests for the multivariate Kalman filter (M5 Rung 3).

Two oracles do the heavy lifting:

  * REDUCTION TO SCALAR -- feed 1x1 matrices and demand the matrix code returns
    exactly what the frozen, already-proven scalar filter returns. Any
    transpose / order / shape slip in the promotion shows up immediately.
  * THE CONSTANT-VELOCITY EXAMPLE -- the physical oracle for the whole rung:
    predict builds a position-velocity correlation out of nothing (P[0,1] =
    dt * var_v), and a POSITION-only update then corrects VELOCITY through that
    off-diagonal. The contrast test (no coupling -> no cross-correction) proves
    the off-diagonal is genuinely the mechanism.

The rest pin the invariants (predict grows P, update shrinks it, P stays
symmetric-PSD -- the Joseph-form guarantee) and the R->0 / R->inf limits.
"""

import numpy as np
import pytest

from bootins.kalman import kalman_update, kalman_predict
from bootins.kalman_scalar import kalman_update_scalar, kalman_predict_scalar


# --- Oracle 1: reduction to the frozen scalar filter --------------------------

@pytest.mark.parametrize("a, Q", [(1, 1), (1, 2), (2, 3), (0.5, 4)])
@pytest.mark.parametrize("x, P", [(1, 2), (2, 4), (3, 5), (-4, 10)])
def test_predict_reduces_to_scalar(x, P, a, Q):
    x_s, P_s = kalman_predict_scalar(x, P, a, Q)

    x_v, P_v = kalman_predict(
        np.array([x]), np.array([[P]]), np.array([[a]]), np.array([[Q]])
    )

    np.testing.assert_allclose(x_v[0], x_s)
    np.testing.assert_allclose(P_v[0, 0], P_s)


@pytest.mark.parametrize("z, R", [(1, 2), (4, 4), (3, 0.5), (1, 10)])
@pytest.mark.parametrize("x, P", [(1, 2), (2, 4), (3, 5), (-4, 10)])
def test_update_reduces_to_scalar(x, P, z, R):
    x_s, P_s = kalman_update_scalar(x, P, z, R)

    # H = [[1]]: the measurement IS the (only) state, matching scalar-world.
    x_v, P_v = kalman_update(
        np.array([x]), np.array([[P]]), np.array([z]),
        np.array([[1.0]]), np.array([[R]])
    )

    np.testing.assert_allclose(x_v[0], x_s)
    np.testing.assert_allclose(P_v[0, 0], P_s)


# --- Oracle 2: the constant-velocity example (the point of the rung) ----------

DT = 0.1
PHI_CV = np.array([[1.0, DT],
                   [0.0, 1.0]])   # p_{k+1} = p + v*dt ; v_{k+1} = v
H_POS = np.array([[1.0, 0.0]])    # measure position only


def test_predict_builds_position_velocity_correlation():
    # Start UNCORRELATED (diagonal P): position and velocity independent.
    var_p, var_v = 1.0, 4.0
    x = np.array([1.0, 0.0])
    P = np.diag([var_p, var_v])
    Q = np.diag([0.01, 0.01])     # diagonal -> contributes nothing to P[0,1]

    _, P_prior = kalman_predict(x, P, PHI_CV, Q)

    # The cross-covariance appears from nothing: dt * (prior velocity variance).
    assert P_prior[0, 1] == pytest.approx(DT * var_v)
    # And position variance grew by dt^2 * var_v -- the faster-than-linear term
    # a scalar cannot produce (came entirely from Phi's off-diagonal).
    assert P_prior[0, 0] == pytest.approx(var_p + DT**2 * var_v + Q[0, 0])
    # Predicted covariance is symmetric.
    np.testing.assert_allclose(P_prior, P_prior.T)


def test_position_fix_corrects_velocity():
    x = np.array([1.0, 0.0])          # velocity estimate starts at exactly 0
    P = np.diag([1.0, 4.0])
    Q = np.diag([0.01, 0.01])

    x_prior, P_prior = kalman_predict(x, P, PHI_CV, Q)
    # Measure position HIGHER than predicted -> positive innovation.
    z = np.array([2.0])
    x_hat, _ = kalman_update(x_prior, P_prior, z, H_POS, np.array([[1.0]]))

    # A position-only measurement moved velocity off zero -- through P[0,1].
    assert x_hat[1] > 0.0


def test_no_coupling_means_no_cross_correction():
    # Phi = I (dt = 0): position and velocity never become correlated, so a
    # position fix must leave velocity UNTOUCHED. This isolates the off-diagonal
    # as the cause of cross-correction in the test above.
    x = np.array([1.0, 0.0])
    P = np.diag([1.0, 4.0])
    Q = np.diag([0.01, 0.01])

    x_prior, P_prior = kalman_predict(x, P, np.eye(2), Q)
    assert P_prior[0, 1] == pytest.approx(0.0)

    z = np.array([2.0])
    x_hat, _ = kalman_update(x_prior, P_prior, z, H_POS, np.array([[1.0]]))

    # Velocity gain is zero -> velocity estimate is exactly unchanged.
    assert x_hat[1] == pytest.approx(x_prior[1])


# --- Invariants ---------------------------------------------------------------

def test_predict_grows_covariance():
    # With Phi = I, P- = P + Q, so P- - P = Q is PSD (uncertainty only grows).
    P = np.diag([1.0, 4.0])
    Q = np.diag([0.5, 0.3])
    _, P_prior = kalman_predict(np.array([1.0, 0.0]), P, np.eye(2), Q)

    eigvals = np.linalg.eigvalsh(P_prior - P)
    assert np.all(eigvals >= -1e-12)


def test_update_shrinks_covariance():
    x = np.array([1.0, 0.0])
    P = np.diag([1.0, 4.0])
    x_prior, P_prior = kalman_predict(x, P, PHI_CV, np.diag([0.01, 0.01]))
    _, P_hat = kalman_update(x_prior, P_prior, np.array([2.0]), H_POS, np.array([[1.0]]))

    # Information never increases uncertainty (trace = total variance).
    assert np.trace(P_hat) <= np.trace(P_prior) + 1e-12


def test_posterior_is_symmetric_and_psd():
    # The Joseph-form guarantee: P+ stays symmetric and positive-semidefinite.
    x = np.array([1.0, 0.0])
    P = np.array([[2.0, 0.5],
                  [0.5, 4.0]])
    x_prior, P_prior = kalman_predict(x, P, PHI_CV, np.diag([0.01, 0.01]))
    _, P_hat = kalman_update(x_prior, P_prior, np.array([2.0]), H_POS, np.array([[1.0]]))

    np.testing.assert_allclose(P_hat, P_hat.T)
    assert np.all(np.linalg.eigvalsh(P_hat) >= -1e-12)


def test_joseph_matches_short_form():
    # For the optimal gain the two P+ forms are algebraically equal; a
    # well-conditioned case makes them agree numerically. This proves our
    # Joseph implementation IS the standard update, not a different filter.
    x = np.array([1.0, 0.0])
    P = np.array([[2.0, 0.5],
                  [0.5, 4.0]])
    x_prior, P_prior = kalman_predict(x, P, PHI_CV, np.diag([0.01, 0.01]))
    z, R = np.array([2.0]), np.array([[1.0]])

    _, P_joseph = kalman_update(x_prior, P_prior, z, H_POS, R)

    S = H_POS @ P_prior @ H_POS.T + R
    K = P_prior @ H_POS.T @ np.linalg.inv(S)
    P_short = (np.eye(2) - K @ H_POS) @ P_prior

    np.testing.assert_allclose(P_joseph, P_short)


# --- R -> 0 and R -> inf limits ----------------------------------------------

def test_perfect_measurement_pins_measured_state():
    # R ~ 0: trust the measurement completely -> measured (position) component
    # of the estimate matches z, and its variance collapses toward 0.
    x_prior = np.array([1.0, 0.0])
    P_prior = np.diag([2.0, 4.0])
    z = np.array([5.0])
    x_hat, P_hat = kalman_update(x_prior, P_prior, z, H_POS, np.array([[1e-12]]))

    assert x_hat[0] == pytest.approx(5.0, abs=1e-5)
    assert P_hat[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_useless_measurement_changes_nothing():
    # R -> inf: the measurement carries no information -> estimate and
    # covariance are unchanged.
    x_prior = np.array([1.0, 0.0])
    P_prior = np.diag([2.0, 4.0])
    z = np.array([5.0])
    x_hat, P_hat = kalman_update(x_prior, P_prior, z, H_POS, np.array([[1e12]]))

    np.testing.assert_allclose(x_hat, x_prior, atol=1e-6)
    np.testing.assert_allclose(P_hat, P_prior, atol=1e-3)
