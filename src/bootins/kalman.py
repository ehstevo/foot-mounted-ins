"""Multivariate (matrix) Kalman filter -- the real linear filter for M5 (Rung 3+).

Promotes the frozen scalar filter in ``kalman_scalar.py`` to vectors and
matrices. The state ``x`` is an n-vector; the covariance ``P`` is an n x n
matrix whose OFF-DIAGONALS carry the correlations between states -- and those
off-diagonals are the whole reason the filter can correct quantities it never
directly measured (a position fix nudging velocity, later GNSS nudging biases).
A scalar has no off-diagonal, which is exactly why it could only drift linearly.

    predict:  x- = Phi x               P- = Phi P Phi^T + Q
    update:   y  = z - H x-            (innovation, in measurement space)
              S  = H P- H^T + R        (innovation covariance)
              K  = P- H^T S^-1         (gain: innovation -> state correction)
              x+ = x- + K y
              P+ = (I - K H) P- (I - K H)^T + K R K^T   (Joseph form)

Three things change vs the scalar filter:
  * ``a**2 * P`` becomes the congruence sandwich ``Phi P Phi^T`` -- Phi appears
    twice (covariance has two copies of the error), the second copy transposed.
  * scalar division ``P / (P + R)`` becomes multiplication by ``S^-1``.
  * H, the measurement matrix, is genuinely new -- it maps the (tall) state
    vector into the (short) measurement space so the innovation compares like
    with like. In scalar-world it was silently 1.

The gain's ``P- H^T`` factor is the cross-covariance between state and
measurement: for a measurement of state i it pulls out column i of P-, so the
innovation is routed to EVERY state correlated with the measured one. That is
the mechanism the whole navigation filter rests on.

Covariance uses the Joseph form: symmetric and positive-semidefinite by
construction, and it stays valid even when K is not the exact optimal gain
(EKF relinearization, or a skipped/modified update after NIS gating). The
shorter ``(I - K H) P-`` form is algebraically equal only for the optimal K
and can silently lose symmetry / positive-definiteness under roundoff.
"""

import numpy as np


def kalman_predict(
        x: np.ndarray, P: np.ndarray, Phi: np.ndarray, Q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """One multivariate KF time/predict step.

    Propagates the estimate and its covariance forward one step under the
    linear model ``x_{k+1} = Phi x_k + w``, ``w ~ N(0, Q)``, producing the
    *prior* (predicted, pre-measurement) belief the update step consumes.

    Args:
        x:   state estimate, shape (n,).
        P:   state covariance, shape (n, n).
        Phi: state-transition matrix, shape (n, n) (the matrix stand-in for the
             scalar ``a``; for our INS this is ``expm(F * dt) ~ I + F * dt``).
        Q:   process-noise covariance, shape (n, n).

    Predict form:
        x- = Phi x
        P- = Phi P Phi^T + Q      (congruence sandwich, then inject Q)

    This is the only step that GROWS the covariance -- the matrix form of drift.
    The off-diagonals of Phi are what let uncertainty flow between states
    (e.g. velocity uncertainty leaking into position over dt), giving the
    faster-than-linear growth a scalar could never produce.

    Returns:
        (x_prior, P_prior): predicted estimate (n,) and its covariance (n, n).
    """
    x_prior = Phi @ x
    P_prior = Phi @ P @ Phi.T + Q

    return (x_prior, P_prior)


def kalman_update(
        x: np.ndarray, P: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """One multivariate KF measurement update.

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

    Update form:
        y = z - H x                   innovation (residual), in measurement space
        S = H P H^T + R               innovation covariance (P projected into
                                      measurement space, plus measurement noise)
        K = P H^T S^-1                gain (maps an innovation back to a state
                                      correction; P H^T is the cross-covariance)
        x+ = x + K y
        P+ = (I - K H) P (I - K H)^T + K R K^T     Joseph form

    Returns:
        (x_hat, P_hat): posterior estimate (n,) and covariance (n, n).
    """
    y = z - H @ x                       # innovation, in measurement space
    S = H @ P @ H.T + R                 # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)      # gain (np.linalg.solve is the more careful idiom)
    x_hat = x + K @ y

    # Joseph form: symmetric-PSD by construction, robust to a non-optimal K.
    A = np.eye(len(x)) - K @ H
    P_hat = A @ P @ A.T + K @ R @ K.T

    return (x_hat, P_hat)
