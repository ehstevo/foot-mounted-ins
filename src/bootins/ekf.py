"""Extended Kalman Filter -- the generic nonlinear harness (M5 Rung 4).

The linear KF in ``kalman.py`` assumed the dynamics and measurement were
matrices: ``x- = Phi x`` and ``y = z - H x``. Real systems are nonlinear --
``x- = f(x)`` and ``z = h(x)`` -- and a Gaussian pushed through a curved
function does NOT stay Gaussian. The EKF's fix, applied identically to both
steps:

  * propagate the STATE with the FULL nonlinear function (no approximation),
  * propagate the COVARIANCE with a LINEARIZATION of that function -- its
    Jacobian, evaluated freshly at the current estimate each step.

    predict:  x- = f(x)              P- = F P F^T + Q ,  F = df/dx |_x
    update:   y  = z - h(x-)         H  = dh/dx |_x-
              S  = H P- H^T + R
              K  = P- H^T S^-1
              x+ = x- + K y
              P+ = (I - K H) P- (I - K H)^T + K R K^T   (Joseph form)

Compared with ``kalman.py`` almost nothing changes: the covariance / gain /
Joseph arithmetic is byte-for-byte identical; only ``Phi`` / ``H`` are now
Jacobians recomputed every step, and the state / innovation use ``f`` / ``h``
directly. That covariance code is INLINED here (not reused from ``kalman.py``)
to keep the whole filter readable in one place.

The catch: the Jacobians are taken at the ESTIMATE, not the truth. Close to
truth the linearization is faithful; far from it the approximation degrades and
the filter can diverge -- the EKF trades the linear KF's optimality guarantee
for the ability to handle nonlinearity. R5's error-state form keeps the
linearized quantity tiny (always near zero) precisely to keep this honest.

``f``, ``h``, ``F_jac``, ``H_jac`` are passed as CALLABLES of the state, so this
harness is model-agnostic: the constant-velocity / range toy in the tests and
the INS mechanization (R5) plug into the exact same two functions.
"""

from collections.abc import Callable

import numpy as np


def ekf_predict(
        x: np.ndarray,
        P: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        F_jac: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
    """One EKF predict step: nonlinear state, linearized covariance.

    Args:
        x:     state estimate, shape (n,).
        P:     state covariance, shape (n, n).
        f:     nonlinear process model, callable ``x -> x_prior`` (n,).
        F_jac: Jacobian ``df/dx``, callable ``x -> (n, n)``.
        Q:     process-noise covariance, shape (n, n).

    The state rides the FULL nonlinear ``f``; only the covariance uses the
    linearized ``Phi = F_jac(x)``. In the linear KF both used the same matrix --
    here they diverge, and that divergence IS the EKF.

    Returns:
        (x_prior, P_prior): predicted estimate (n,) and covariance (n, n).
    """
    x_prior = f(x)                       # FULL nonlinear propagation
    Phi = F_jac(x)                       # linearize about the current estimate
    P_prior = Phi @ P @ Phi.T + Q        # covariance rides the linearized Phi

    return (x_prior, P_prior)


def ekf_update(
        x: np.ndarray,
        P: np.ndarray,
        z: np.ndarray,
        h: Callable[[np.ndarray], np.ndarray],
        H_jac: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
    """One EKF measurement update: nonlinear innovation, linearized gain.

    Args:
        x:     prior state estimate, shape (n,).
        P:     prior state covariance, shape (n, n).
        z:     measurement, shape (m,).
        h:     nonlinear measurement model, callable ``x -> z_pred`` (m,).
        H_jac: Jacobian ``dh/dx``, callable ``x -> (m, n)``.
        R:     measurement-noise covariance, shape (m, m).

    The innovation compares the real measurement with the FULL nonlinear
    prediction ``h(x)``; the gain is built from the linearized ``H = H_jac(x)``.

    Returns:
        (x_hat, P_hat): posterior estimate (n,) and covariance (n, n).
    """
    y = z - h(x)                         # innovation via the FULL nonlinear h
    H = H_jac(x)                         # linearize about the prior estimate
    S = H @ P @ H.T + R                  # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)       # gain (np.linalg.solve is more careful)
    x_hat = x + K @ y

    # Joseph form: symmetric-PSD by construction, robust to a non-optimal K --
    # and the EKF's relinearized gain is ALWAYS slightly non-optimal (it is only
    # optimal for the linearized system, not the true nonlinear one).
    A = np.eye(len(x)) - K @ H
    P_hat = A @ P @ A.T + K @ R @ K.T

    return (x_hat, P_hat)
