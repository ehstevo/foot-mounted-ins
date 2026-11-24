"""Multivariate linear Kalman-filter reference implementation.

The state ``x`` is an n-vector and the covariance ``P`` is an ``(n, n)`` matrix.
Its off-diagonal entries carry correlations between states, allowing a direct
measurement of one state to correct correlated states. For example, a position
observation can correct velocity or bias estimates through their cross-covariance
with position.

    predict:  x- = Phi x               P- = Phi P Phi^T + Q
    update:   y  = z - H x-            (innovation, in measurement space)
              S  = H P- H^T + R        (innovation covariance)
              K  = P- H^T S^-1         (gain: innovation -> state correction)
              x+ = x- + K y
              P+ = (I - K H) P- (I - K H)^T + K R K^T   (Joseph form)

Relative to the scalar formulation:
  * ``a**2 * P`` becomes the congruence transform ``Phi P Phi^T``. ``Phi``
    appears twice because covariance contains two copies of the error, with the
    second transformed by the transpose.
  * scalar division ``P / (P + R)`` becomes multiplication by ``S^-1``.
  * H, the measurement matrix, is genuinely new -- it maps the (tall) state
    vector into the (short) measurement space so the innovation compares like
    with like. In scalar-world it was silently 1.

The gain's ``P- H^T`` factor is the cross-covariance between state and
measurement. For a direct measurement of state ``i``, it selects column ``i``
of ``P-`` and routes the innovation to every correlated state.

Covariance uses the Joseph form. It preserves symmetry and positive
semidefiniteness algebraically and remains valid when ``K`` is not the exact
optimal gain, such as after an EKF linearization or a modified update. The
shorter ``(I - K H) P-`` form is algebraically equivalent only for the optimal
gain and can lose symmetry or positive semidefiniteness under roundoff.

This module is a generic linear-filter reference. The operational navigation
filter uses the error-state formulation in ``bootins.estimation.esekf``.
"""

import numpy as np


def kalman_predict(
        x: np.ndarray, P: np.ndarray, Phi: np.ndarray, Q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """Apply one multivariate linear Kalman-filter prediction step.

    Propagates the estimate and its covariance forward one step under the
    linear model ``x_{k+1} = Phi x_k + w``, ``w ~ N(0, Q)``, producing the
    *prior* (predicted, pre-measurement) belief the update step consumes.

    Args:
        x:   state estimate, shape (n,).
        P:   state covariance, shape (n, n).
        Phi: state-transition matrix, shape (n, n) (the matrix stand-in for the
             scalar ``a``; for our INS this is ``expm(F * dt) ~ I + F * dt``).
        Q:   process-noise covariance, shape (n, n).

    Notes:
        The process model is ``x_{k+1} = Phi x_k + w`` with ``w ~ N(0, Q)``.
        Its prediction equations are

        x- = Phi x
        P- = Phi P Phi^T + Q      (congruence sandwich, then inject Q)

        Off-diagonal elements of ``Phi`` transfer uncertainty between state
        channels, such as velocity uncertainty entering position over time.

    Returns:
        A pair ``(x_prior, P_prior)`` containing the predicted estimate with
        shape ``(n,)`` and covariance with shape ``(n, n)``.
    """
    x_prior = Phi @ x
    P_prior = Phi @ P @ Phi.T + Q

    return (x_prior, P_prior)


def kalman_update(
        x: np.ndarray, P: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """Apply one multivariate linear Kalman-filter measurement update.

    Fuses a prior estimate ``x`` (covariance ``P``) with a measurement ``z``
    (noise covariance ``R``) that lives in measurement space, related to the
    state by ``z = H x + noise``. The measurement can have fewer components
    than the state (H is generally rectangular, m x n).

    Args:
        x: prior state estimate, shape (n,).
        P: prior state covariance, shape (n, n).
        z: measurement, shape (m,).
        H: measurement matrix mapping state -> measurement space, shape (m, n).
        R: measurement-noise covariance, shape (m, m).

    Notes:
        The measurement model is ``z = H x + noise``. The update equations are
        y = z - H x                   innovation (residual), in measurement space
        S = H P H^T + R               innovation covariance (P projected into
                                      measurement space, plus measurement noise)
        K = P H^T S^-1                gain (maps an innovation back to a state
                                      correction; P H^T is the cross-covariance)
        x+ = x + K y
        P+ = (I - K H) P (I - K H)^T + K R K^T     Joseph form

    Returns:
        A pair ``(x_hat, P_hat)`` containing the posterior estimate with shape
        ``(n,)`` and covariance with shape ``(n, n)``.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance ``S`` is
            singular and cannot be inverted.
    """
    y = z - H @ x                       # innovation, in measurement space
    S = H @ P @ H.T + R                 # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)      # gain (np.linalg.solve is the more careful idiom)
    x_hat = x + K @ y

    # Joseph form: symmetric-PSD by construction, robust to a non-optimal K.
    A = np.eye(len(x)) - K @ H
    P_hat = A @ P @ A.T + K @ R @ K.T

    return (x_hat, P_hat)
