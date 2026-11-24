"""Error-state extended Kalman filter for strapdown navigation.

The filter separates each navigation quantity into a nonlinear nominal state
and a small linearized error state:

    true = nominal + error

  * Nominal (16-vector): the best current estimate, propagated through the
    nonlinear strapdown mechanization.
  * Error (15-vector): a small deviation between nominal and truth. Its mean is
    reset to zero after each update; ``P`` tracks its covariance. Adding a
    deterministic nominal offset does not change covariance, so ``P`` also
    represents uncertainty in the physical navigation state.

The named slices below define the fixed layouts:

    NOMINAL (16):  p(3)  v(3)  q(4, [w,x,y,z] b->n)  b_a(3)  b_g(3)
    ERROR   (15):  dp(3) dv(3) dpsi(3)               db_a(3) db_g(3)

The nominal quaternion requires four components, whereas its multiplicative
attitude error is the three-component small rotation ``dpsi``. This produces a
minimal, unconstrained error state.

Filter operation has two distinct steps:

  * Prediction (typically at IMU rate) de-biases each increment, mechanizes the
    nominal state, and propagates ``P`` with the first-order transition
    ``Phi = I + F * dt``.
  * An update estimates ``dx_hat`` from a measurement, injects it into the
    nominal state, and resets the error-state mean to zero. Resetting keeps the
    linearization point near zero and prevents a correction from being applied
    again on the next update.

Note the two distinct bias operations, easy to conflate:
  * De-bias during prediction: use the current estimate to correct raw IMU
    increments.
  * Inject during an update: improve the bias estimate from a measurement.
"""
from collections.abc import Callable

import numpy as np

from bootins.estimation.mechanization import mechanize_step
from bootins.estimation.error_dynamics import error_dynamics_F
from bootins.frames.quaternion import quat_multiply, normalize, quat_to_dcm

# --- Nominal state layout (16-vector) -----------------------------------------
NOM_P = slice(0, 3)     # position (m, NED)
NOM_V = slice(3, 6)     # velocity (m/s, NED)
NOM_Q = slice(6, 10)    # attitude quaternion [w, x, y, z], body->nav
NOM_BA = slice(10, 13)  # accel bias estimate (m/s^2)
NOM_BG = slice(13, 16)  # gyro bias estimate (rad/s)

# --- Error state layout (15-vector): [dp, dv, dpsi, db_a, db_g] ---------------
ERR_DP = slice(0, 3)     # position error
ERR_DV = slice(3, 6)     # velocity error
ERR_DPSI = slice(6, 9)   # attitude error (small rotation vector, nav frame)
ERR_DBA = slice(9, 12)   # accel bias error
ERR_DBG = slice(12, 15)  # gyro bias error


def inject(nominal: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Inject an estimated error state into a nominal navigation state.

    Args:
        nominal: Current 16-element nominal state in the layout defined by
            ``NOM_*`` slices.
        dx: Estimated 15-element error state in the layout defined by
            ``ERR_*`` slices.

    Returns:
        A new corrected 16-element nominal state. The input ``nominal`` is not
        mutated.

    Notes:
        The additive channels follow ``true = nominal + error``:

        p   += dp,   v += dv,   b_a += db_a,   b_g += db_g
        q    = normalize( dq (x) q ),   dq = [1, 0.5*dpsi]

        Attitude is multiplicative because a rotation error composes rather than
        adds. The small-angle lift ``dpsi -> dq = [1, 0.5 * dpsi]`` is only
        approximately unit and is normalized after multiplication. ``dq``
        multiplies on the left because ``dpsi`` is expressed in the navigation
        frame: ``q_true = dq (x) q_nom``. This differs from a body-frame IMU
        rotation increment, which composes on the right.

        The reset is implicit: the returned nominal has absorbed the estimated
        error, while no nonzero error mean is propagated.
    """
    nominal = np.array(nominal, dtype=float)   # copy: do not mutate the caller
    nominal[NOM_P] += dx[ERR_DP]
    nominal[NOM_V] += dx[ERR_DV]
    delta_q = np.array([1.0, *(0.5 * dx[ERR_DPSI])])
    nominal[NOM_Q] = normalize(quat_multiply(delta_q, nominal[NOM_Q]))
    nominal[NOM_BA] += dx[ERR_DBA]
    nominal[NOM_BG] += dx[ERR_DBG]
    return nominal


def esekf_predict(nominal: np.ndarray, P: np.ndarray,
                  dtheta: np.ndarray, dv: np.ndarray, dt: float,
                  Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict nominal navigation state and error covariance by one IMU increment.

    Args:
        nominal: Current 16-element nominal state.
        P: Current ``(15, 15)`` error-state covariance.
        dtheta: Body-frame angular increment in radians.
        dv: Body-frame specific-force increment in m/s.
        dt: Increment duration in seconds.
        Q: Discrete per-predict ``(15, 15)`` process-noise covariance.

    Returns:
        A pair ``(nominal_new, P_new)`` containing the propagated 16-element
        nominal state and ``(15, 15)`` error covariance.

    Notes:
        The nominal propagation de-biases the raw increments using the current
        bias estimates, then mechanizes ``(p, v, q)`` nonlinearly. The bias
        means are held constant under the random-walk model; ``Q`` grows their
        covariance.

        Covariance propagation builds ``F`` at the nominal operating point,
        using the old attitude that rotates ``dv`` in mechanization. It applies
        the first-order discrete transition ``Phi = I + F * dt``:

            P_new = Phi P Phi^T + Q

        The error-state mean remains zero because it was reset after the prior
        update and ``Phi * 0 = 0``.
    """
    b_a = nominal[NOM_BA]
    b_g = nominal[NOM_BG]
    dtheta_corr = dtheta - b_g * dt    # gyro bias corrects the ANGLE increment
    dv_corr = dv - b_a * dt            # accel bias corrects the VELOCITY increment

    # Propagate the nonlinear nominal state.
    p, v, q = nominal[NOM_P], nominal[NOM_V], nominal[NOM_Q]
    p_new, v_new, q_new = mechanize_step((p, v, q), dtheta_corr, dv_corr, dt)
    # Bias means are constant under the random-walk model.
    nominal_new = np.hstack((p_new, v_new, q_new, b_a, b_g))

    # Propagate the linearized error covariance at the old nominal attitude.
    C_nb = quat_to_dcm(q)
    f_nav = C_nb @ (dv_corr / dt)
    F = error_dynamics_F(C_nb, f_nav)
    Phi = np.eye(P.shape[0]) + F * dt
    P_new = Phi @ P @ Phi.T + Q
    # Remove floating-point skew; a covariance is mathematically symmetric.
    P_new = 0.5 * (P_new + P_new.T)

    return nominal_new, P_new


def esekf_update(nominal: np.ndarray, P: np.ndarray, z: np.ndarray,
                 h: Callable[[np.ndarray], np.ndarray], H: np.ndarray,
                 R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply one error-state EKF measurement update and inject its correction.

    Args:
        nominal: Current 16-element nominal state.
        P: Current ``(15, 15)`` error-state covariance.
        z: Measurement vector with shape ``(m,)``.
        h: Function mapping the nominal state to the predicted measurement with
            shape ``(m,)``.
        H: Measurement Jacobian ``dh / d(dx)`` with shape ``(m, 15)``.
        R: Measurement-noise covariance with shape ``(m, m)``.

    Returns:
        A pair ``(nominal_new, P_new)`` containing the corrected nominal state
        and Joseph-form posterior covariance.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.

    Notes:
        The update estimates the error state, whose prior mean is zero. The
        innovation is nonlinear because it compares the measurement with the
        full nominal prediction:

        y   = z - h(nominal)           innovation (nonlinear, from the nominal)
        S   = H P H^T + R              innovation covariance
        K   = P H^T S^-1              gain
        dx  = 0 + K y                  estimated error (prior mean is zero)
        P+  = (I - K H) P (I - K H)^T + K R K^T    (Joseph form)

        The estimated error is injected into the nominal state. Its reset is
        implicit because no nonzero error mean is carried into the next
        prediction.

        ``h`` maps the 16-element nominal state into measurement space, whereas
        ``H = dh / d(dx)`` maps the 15-element error state. The innovation must
        therefore use ``h(nominal)``, not ``H @ nominal``.
    """
    dx = np.zeros(P.shape[0])
    y = z - h(nominal)                  # innovation, from the NOMINAL (nonlinear)
    S = H @ P @ H.T + R                 # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)      # gain (np.linalg.solve is the more careful idiom)
    dx_hat = dx + K @ y                 # error estimate (prior mean dx is zero)
    nominal_new = inject(nominal, dx_hat)   # inject + implicit reset

    # Joseph form: symmetric-PSD by construction, robust to a non-optimal K.
    A = np.eye(P.shape[0]) - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T
    # Remove floating-point skew; a covariance is mathematically symmetric.
    P_new = 0.5 * (P_new + P_new.T)

    return nominal_new, P_new
