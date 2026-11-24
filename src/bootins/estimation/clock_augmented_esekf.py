"""Clock-augmented error-state extended Kalman filter for GNSS-aided navigation.

The filter extends the 15-element navigation error-state model with the IMU-to-
GNSS clock offset:

    nominal = [p, v, q, b_a, b_g]
    error   = [dp, dv, dpsi, db_a, db_g]

    nominal = [p, v, q, b_a, b_g, tau]      # 17 values
    error   = [dp, dv, dpsi, db_a, db_g, dtau]   # 16 values

The navigation channels use the same nominal mechanization and linearized
navigation dynamics as the non-clock ESEKF. ``tau`` represents IMU time minus
GNSS time in seconds. Its nominal mean is held constant during prediction;
process noise in ``Q`` grows the uncertainty of its random-walk error. A GNSS
measurement update can correct the clock mean through ``dtau``.

The non-clock ESEKF remains a separate implementation for navigation problems
that do not estimate an inter-clock offset.
"""
from collections.abc import Callable

import numpy as np

from bootins.estimation.error_dynamics import error_dynamics_F
from bootins.frames.quaternion import normalize, quat_multiply, quat_to_dcm
from bootins.estimation.mechanization import mechanize_step

# --- Nominal state layout (17-vector) -----------------------------------------
NOM_P = slice(0, 3)     # position (m, NED)
NOM_V = slice(3, 6)     # velocity (m/s, NED)
NOM_Q = slice(6, 10)    # attitude quaternion [w, x, y, z], body->nav
NOM_BA = slice(10, 13)  # accel bias estimate (m/s^2)
NOM_BG = slice(13, 16)  # gyro bias estimate (rad/s)
NOM_TAU = 16            # IMU time minus GNSS time (s)

# --- Error state layout (16-vector): [dp, dv, dpsi, db_a, db_g, dtau] --------
ERR_DP = slice(0, 3)     # position error
ERR_DV = slice(3, 6)     # velocity error
ERR_DPSI = slice(6, 9)   # attitude error (small rotation vector, nav frame)
ERR_DBA = slice(9, 12)   # accel bias error
ERR_DBG = slice(12, 15)  # gyro bias error
ERR_DTAU = 15            # clock-offset error


def inject_clock(nominal: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Inject an estimated navigation-and-clock error into the nominal state.

    Args:
        nominal: Current 17-element nominal state in the layout defined by
            ``NOM_*`` slices and index.
        dx: Estimated 16-element error state in the layout defined by
            ``ERR_*`` slices and index.

    Returns:
        A new corrected 17-element nominal state. The input ``nominal`` is not
        mutated.

    Notes:
        The additive channels follow ``true = nominal + error``:

        p   += dp,   v += dv,   b_a += db_a,   b_g += db_g,   tau += dtau
        q    = normalize( dq (x) q ),   dq = [1, 0.5*dpsi]

        Attitude is multiplicative. The small-angle lift
        ``dpsi -> dq = [1, 0.5 * dpsi]`` is only approximately unit and is
        normalized after multiplication. ``dq`` multiplies on the left because
        ``dpsi`` is expressed in the navigation frame:
        ``q_true = dq (x) q_nom``. Body-frame IMU rotation increments instead
        compose on the right.

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
    nominal[NOM_TAU] += dx[ERR_DTAU]
    return nominal


def esekf_predict_clock(nominal: np.ndarray, P: np.ndarray,
                        dtheta: np.ndarray, dv: np.ndarray, dt: float,
                        Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict navigation, clock-offset, and error covariance by one IMU increment.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        P: Current ``(16, 16)`` error-state covariance.
        dtheta: Body-frame angular increment in radians.
        dv: Body-frame specific-force increment in m/s.
        dt: Increment duration in seconds.
        Q: Discrete per-predict ``(16, 16)`` process-noise covariance.

    Returns:
        A pair ``(nominal_new, P_new)`` containing the propagated 17-element
        nominal state and ``(16, 16)`` error covariance.

    Notes:
        The nonlinear nominal propagation de-biases the raw increments and
        mechanizes ``(p, v, q)``. Bias and clock-offset means are held constant
        under their random-walk models, while ``Q`` grows their covariance.

        The 15-by-15 navigation dynamics matrix is evaluated at the old nominal
        attitude and embedded in a 16-by-16 first-order transition. The clock
        error has identity mean transition:

            P_new = Phi P Phi^T + Q

        The error-state mean remains zero because it was reset after the prior
        update and ``Phi * 0 = 0``.
    """
    b_a = nominal[NOM_BA]
    b_g = nominal[NOM_BG]
    tau = nominal[NOM_TAU]
    dtheta_corr = dtheta - b_g * dt    # gyro bias corrects the ANGLE increment
    dv_corr = dv - b_a * dt            # accel bias corrects the VELOCITY increment

    # Propagate the nonlinear nominal state.
    p, v, q = nominal[NOM_P], nominal[NOM_V], nominal[NOM_Q]
    p_new, v_new, q_new = mechanize_step((p, v, q), dtheta_corr, dv_corr, dt)
    # Bias and clock means are constant under their random-walk models.
    nominal_new = np.hstack((p_new, v_new, q_new, b_a, b_g, tau))

    # Propagate the linearized error covariance at the old nominal attitude.
    C_nb = quat_to_dcm(q)
    f_nav = C_nb @ (dv_corr / dt)
    F = error_dynamics_F(C_nb, f_nav)
    Phi_nav = np.eye(F.shape[0]) + F * dt
    Phi = np.eye(P.shape[0])
    Phi[:F.shape[0], :F.shape[1]] = Phi_nav
    P_new = Phi @ P @ Phi.T + Q
    # Remove floating-point skew; a covariance is mathematically symmetric.
    P_new = 0.5 * (P_new + P_new.T)

    return nominal_new, P_new


def esekf_update_clock(nominal: np.ndarray, P: np.ndarray, z: np.ndarray,
                       h: Callable[[np.ndarray], np.ndarray], H: np.ndarray,
                       R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply one clock-augmented ESEKF measurement update and inject it.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        P: Current ``(16, 16)`` error-state covariance.
        z: Measurement vector with shape ``(m,)``.
        h: Function mapping the nominal state to the predicted measurement with
            shape ``(m,)``.
        H: Measurement Jacobian ``dh / d(dx)`` with shape ``(m, 16)``.
        R: Measurement-noise covariance with shape ``(m, m)``.

    Returns:
        A pair ``(nominal_new, P_new)`` containing the corrected nominal state
        and Joseph-form posterior covariance.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.

    Notes:
        The update estimates the error state, whose prior mean is zero. The
        innovation compares the measurement with the full nominal prediction:

        y   = z - h(nominal)           innovation (nonlinear, from the nominal)
        S   = H P H^T + R              innovation covariance
        K   = P H^T S^-1              gain
        dx  = 0 + K y                  estimated error (prior mean is zero)
        P+  = (I - K H) P (I - K H)^T + K R K^T    (Joseph form)

        The estimated error is injected into the nominal state. Its reset is
        implicit because no nonzero error mean is carried into the next
        prediction.

        ``h`` maps the 17-element nominal state into measurement space, whereas
        ``H = dh / d(dx)`` maps the 16-element error state. The innovation must
        therefore use ``h(nominal)``, not ``H @ nominal``.
    """
    dx = np.zeros(P.shape[0])
    y = z - h(nominal)                  # innovation, from the NOMINAL (nonlinear)
    S = H @ P @ H.T + R                 # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)      # gain (np.linalg.solve is the more careful idiom)
    dx_hat = dx + K @ y                 # error estimate (prior mean dx is zero)
    nominal_new = inject_clock(nominal, dx_hat)   # inject + implicit reset

    # Joseph form: symmetric-PSD by construction, robust to a non-optimal K.
    A = np.eye(P.shape[0]) - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T
    # Remove floating-point skew; a covariance is mathematically symmetric.
    P_new = 0.5 * (P_new + P_new.T)

    return nominal_new, P_new
