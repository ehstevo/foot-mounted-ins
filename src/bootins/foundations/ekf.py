"""Generic extended Kalman-filter implementation for nonlinear systems.

The linear Kalman filter in ``kalman.py`` uses matrix process and measurement
models: ``x- = Phi x`` and ``y = z - H x``. For nonlinear systems,
``x- = f(x)`` and ``z = h(x)``; a Gaussian distribution propagated through a
curved function is not generally Gaussian. The EKF approximates each step by

  * propagating the state with the full nonlinear function, and
  * propagating covariance with a local Jacobian evaluated at the current
    estimate.

    predict:  x- = f(x)              P- = F P F^T + Q ,  F = df/dx |_x
    update:   y  = z - h(x-)         H  = dh/dx |_x-
              S  = H P- H^T + R
              K  = P- H^T S^-1
              x+ = x- + K y
              P+ = (I - K H) P- (I - K H)^T + K R K^T   (Joseph form)

The covariance, gain, and Joseph-form arithmetic have the same mathematical
form as the linear filter. ``Phi`` and ``H`` are now Jacobians recomputed at
each estimate, while the state and innovation use ``f`` and ``h`` directly.

Jacobians are evaluated at the estimate rather than the unknown truth. The
approximation degrades as the estimate moves away from the true state, so an
EKF can diverge. The operational navigation filter uses an error-state EKF to
keep its linearized error near zero.

``f``, ``h``, ``F_jac``, ``H_jac`` are passed as CALLABLES of the state, so this
harness is model-agnostic: its process and measurement models are supplied by
the caller.
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
    """Apply one EKF prediction step with nonlinear state propagation.

    Args:
        x:     state estimate, shape (n,).
        P:     state covariance, shape (n, n).
        f:     nonlinear process model, callable ``x -> x_prior`` (n,).
        F_jac: Jacobian ``df/dx``, callable ``x -> (n, n)``.
        Q:     process-noise covariance, shape (n, n).

    Notes:
        The state uses the full nonlinear model ``f``. Only covariance uses the
        local transition ``Phi = F_jac(x)``; a linear Kalman filter uses the
        same matrix for both operations.

    Returns:
        A pair ``(x_prior, P_prior)`` containing the predicted estimate with
        shape ``(n,)`` and covariance with shape ``(n, n)``.
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
    """Apply one EKF measurement update with a nonlinear innovation.

    Args:
        x:     prior state estimate, shape (n,).
        P:     prior state covariance, shape (n, n).
        z:     measurement, shape (m,).
        h:     nonlinear measurement model, callable ``x -> z_pred`` (m,).
        H_jac: Jacobian ``dh/dx``, callable ``x -> (m, n)``.
        R:     measurement-noise covariance, shape (m, m).

    Notes:
        The innovation compares the measurement with the full nonlinear
        prediction ``h(x)``. The gain uses the local measurement Jacobian
        ``H = H_jac(x)``.

    Returns:
        A pair ``(x_hat, P_hat)`` containing the posterior estimate with shape
        ``(n,)`` and covariance with shape ``(n, n)``.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.
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
