"""Tests for the generic EKF harness (M5 Rung 4).

The toy is a "baby GPS": a 2D constant-velocity target observed by two fixed
range sensors. Range ``r = sqrt((px-sx)^2 + (py-sy)^2)`` is nonlinear in the
state, so there is no matrix ``H`` with ``r = H x`` -- exactly the situation the
EKF exists for, and a 2D dress rehearsal for the M7 GNSS update (GPS *is* range
to known transmitter positions).

Three oracles:

  * REDUCTION TO THE LINEAR KF -- feed a linear ``f``/``h`` (and their constant
    Jacobians) and demand ``ekf_predict``/``ekf_update`` return exactly what the
    proven ``kalman.py`` returns. Proves the harness plumbing; also exercises the
    nonlinear-predict machinery even though this toy's own dynamics are linear.
  * JACOBIAN vs FINITE DIFFERENCES -- the analytic range Jacobian must match a
    numerical ``(h(x+e) - h(x-e)) / 2e``. Independent proof that the hand-derived
    ``H`` really is ``dh/dx`` (catches a sign / index slip cold).
  * RECOVERY -- generate the EXACT measurements a perfect sensor would report for
    a known CV trajectory, start the filter from a wrong guess, and show it is
    pulled onto the truth (physics-run-backwards, like the mechanization oracle).
"""

import numpy as np
import pytest

from bootins.ekf import ekf_predict, ekf_update
from bootins.kalman import kalman_predict, kalman_update


# --- The baby-GPS toy: 2D constant-velocity target, two range sensors ---------

DT = 0.5

# Constant-velocity transition: p_{k+1} = p + v*dt ; v_{k+1} = v. State order is
# [px, py, vx, vy]. This f is LINEAR (f(x) = PHI_CV @ x), so the toy's
# nonlinearity lives entirely in the measurement.
PHI_CV = np.array([
    [1.0, 0.0, DT,  0.0],
    [0.0, 1.0, 0.0, DT ],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# Two fixed sensors on the x-axis. The baseline creates a mirror ambiguity
# across the x-axis (a target and its reflection give identical ranges), so all
# trajectories / guesses below stay on the +y side to keep one branch.
S1 = np.array([0.0, 0.0])
S2 = np.array([10.0, 0.0])


def cv_f(x):
    return PHI_CV @ x


def cv_F_jac(x):
    return PHI_CV        # linear dynamics -> Jacobian is the constant PHI_CV


def range_h(x):
    """Nonlinear measurement: distance from each sensor to the target."""
    p = x[:2]
    return np.array([
        np.linalg.norm(p - S1),
        np.linalg.norm(p - S2),
    ])


def range_H_jac(x):
    """Analytic Jacobian dh/dx (2x4). Each row is the sensor->target unit
    vector in the position columns; the velocity columns are zero (an
    instantaneous range does not depend on velocity)."""
    p = x[:2]
    d1 = p - S1
    d2 = p - S2
    r1 = np.linalg.norm(d1)
    r2 = np.linalg.norm(d2)
    return np.array([
        [d1[0] / r1, d1[1] / r1, 0.0, 0.0],
        [d2[0] / r2, d2[1] / r2, 0.0, 0.0],
    ])


# --- Oracle 1: reduction to the linear KF -------------------------------------

def test_predict_reduces_to_linear_kf():
    # With a linear f and its constant Jacobian, ekf_predict must reproduce the
    # linear KF exactly: x- = PHI x and P- = PHI P PHI^T + Q.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    P = np.diag([1.0, 2.0, 3.0, 4.0])
    Q = np.eye(4) * 0.1

    x_e, P_e = ekf_predict(x, P, cv_f, cv_F_jac, Q)
    x_k, P_k = kalman_predict(x, P, PHI_CV, Q)

    np.testing.assert_allclose(x_e, x_k)
    np.testing.assert_allclose(P_e, P_k)


def test_update_reduces_to_linear_kf():
    # A linear measurement (position only) with a constant Jacobian must make
    # ekf_update reproduce the linear KF update exactly.
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    P = np.diag([2.0, 2.0, 2.0, 2.0])
    R = np.eye(2) * 0.5
    z = np.array([1.5, 2.5])

    x_e, P_e = ekf_update(x, P, z, lambda s: H @ s, lambda s: H, R)
    x_k, P_k = kalman_update(x, P, z, H, R)

    np.testing.assert_allclose(x_e, x_k)
    np.testing.assert_allclose(P_e, P_k)


# --- Oracle 2: analytic Jacobian vs finite differences ------------------------

def test_range_jacobian_matches_finite_difference():
    x = np.array([3.0, 4.0, 1.0, -2.0])   # off-axis, nonzero velocity
    H_analytic = range_H_jac(x)

    eps = 1e-6
    H_fd = np.zeros((2, 4))
    for i in range(4):
        step = np.zeros(4)
        step[i] = eps
        H_fd[:, i] = (range_h(x + step) - range_h(x - step)) / (2 * eps)

    np.testing.assert_allclose(H_analytic, H_fd, atol=1e-7)


def test_range_jacobian_velocity_columns_are_zero():
    # The defining structural fact: range cannot see velocity directly.
    x = np.array([3.0, 4.0, 1.0, -2.0])
    H = range_H_jac(x)
    np.testing.assert_allclose(H[:, 2:], 0.0, atol=1e-12)


# --- Oracle 3: the EKF recovers a known trajectory ----------------------------

def test_ekf_recovers_cv_trajectory():
    # True CV target, kept on the +y side (avoids the two-range mirror).
    x_true = np.array([2.0, 5.0, 0.4, 0.2])

    # Start from a WRONG guess with large uncertainty; noiseless measurements +
    # full observability should pull the estimate onto the truth. (Guess is off
    # the sensors and on the +y side so H is well-defined and on-branch.)
    x_est = np.array([1.0, 2.0, 0.0, 0.0])
    P = np.diag([100.0, 100.0, 100.0, 100.0])
    Q = np.eye(4) * 1e-6
    R = np.eye(2) * 1e-6

    for _ in range(60):
        x_true = PHI_CV @ x_true          # advance truth
        z = range_h(x_true)               # exact measurement a perfect sensor gives

        x_est, P = ekf_predict(x_est, P, cv_f, cv_F_jac, Q)
        x_est, P = ekf_update(x_est, P, z, range_h, range_H_jac, R)

    # Position and (indirectly observed) velocity both converge to truth.
    np.testing.assert_allclose(x_est, x_true, atol=1e-2)


def test_ekf_posterior_is_symmetric_and_psd():
    # Joseph form must keep P symmetric-PSD even through the nonlinear update.
    x = np.array([2.0, 5.0, 0.4, 0.2])
    P = np.diag([4.0, 4.0, 1.0, 1.0])
    z = range_h(np.array([2.3, 5.1, 0.0, 0.0]))   # slightly off -> nonzero innovation

    _, P_hat = ekf_update(x, P, z, range_h, range_H_jac, np.eye(2) * 0.25)

    np.testing.assert_allclose(P_hat, P_hat.T)
    assert np.all(np.linalg.eigvalsh(P_hat) >= -1e-12)
