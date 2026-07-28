"""
error_state_ekf.py -- the error-state (indirect) EKF, M5 Rung 5.

This ties every earlier piece together: the strapdown mechanization (M3), the
linearized error dynamics F (M4), and the Kalman machinery (M5 R1-R4). It is the
capstone of the estimation module.

The pivot (M4 R2): we do NOT filter the state directly. We split every quantity
into a NOMINAL part plus a small ERROR, and let the filter estimate the error:

    true = nominal + error

  * NOMINAL (16-vector) -- our single best guess, held as plain numbers and
    propagated OPEN-LOOP through the full nonlinear mechanization. It carries the
    answer.
  * ERROR  (15-vector, mean + covariance P) -- the small, ~linear, ~slow
    deviation of the nominal from truth. Its mean is pinned at ZERO between
    updates; all of P's job is to track the uncertainty. Because `nominal` is a
    deterministic offset, P is equally the covariance of the TRUE state.

Layouts (kept as named slices below so a channel can never be miscounted):

    NOMINAL (16):  p(3)  v(3)  q(4, [w,x,y,z] b->n)  b_a(3)  b_g(3)
    ERROR   (15):  dp(3) dv(3) dpsi(3)               db_a(3) db_g(3)

The quaternion needs 4 numbers; its error needs only 3 (a small rotation vector
dpsi), which is exactly why the multiplicative attitude error buys us a minimal,
constraint-free error state.

Two-rate loop:

  * PREDICT (100 Hz, `esekf_predict`): mechanize the nominal one step (after
    DE-BIASING the increments with the current bias estimate), and grow P through
    the linearized Phi = I + F*dt. Two DIFFERENT propagations of two DIFFERENT
    objects -- which is why this cannot reuse the generic single-state
    `ekf_predict` harness (there, one x flows through both f and F_jac; here the
    nominal that f advances and the error that P describes are distinct vectors of
    distinct sizes). The covariance MATH (Phi P Phi^T + Q) is still the R3/R4
    math, just driven by hand.
  * UPDATE (~1 Hz, `esekf_update`): a measurement produces an error estimate
    dx_hat, which we INJECT into the nominal and then RESET to zero -- keeping the
    linearization point permanently near zero (defusing the EKF's Achilles' heel)
    and, crucially, never double-counting a correction (charter pitfall #9).

Note the two distinct bias operations, easy to conflate:
  * DE-BIAS (predict): USE the current estimate to clean the raw increments.
  * INJECT  (update):  IMPROVE the estimate from a measurement.
"""
from collections.abc import Callable

import numpy as np

from bootins.mechanization import mechanize_step
from bootins.error_dynamics import error_dynamics_F
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
    """Fold an estimated error dx into the nominal state (the INJECT step).

    Applies dx as the correction (true = nominal + error), returning a NEW
    nominal centred on the corrected estimate. Every channel is additive EXCEPT
    attitude, which is multiplicative because a rotation error composes, it does
    not add:

        p   += dp,   v += dv,   b_a += db_a,   b_g += db_g
        q    = normalize( dq (x) q ),   dq = [1, 0.5*dpsi]

    The small-angle lift dpsi -> dq = [1, 0.5*dpsi] is the same half-angle as
    q_dot = 0.5 q (x) [0, w]; it is only approximately unit, so we `normalize`.
    dq multiplies on the LEFT (dq (x) q), because our attitude error is defined
    in the NAV frame (M4 R3: q_true = dq (x) q_nom) -- contrast `attitude_update`,
    where the body-frame increment composes on the RIGHT.

    The caller's array is not mutated (we copy up front): inject is a pure
    old -> new map, so the RESET (dx -> 0) is implicit -- the returned nominal
    already absorbs the correction, and the error is simply not carried forward.

    nominal : 16-vector (see layout constants)
    dx      : 15-vector error estimate (see layout constants)
    returns : the corrected 16-vector nominal
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
    """One error-state EKF predict step: nominal nonlinear, covariance linear.

    nominal : current 16-vector nominal state
    P       : current 15x15 error covariance
    dtheta  : body-frame angle increment (rad) for this step
    dv      : body-frame velocity increment (m/s) for this step
    dt      : step duration (s)
    Q       : 15x15 process-noise covariance for the step
    returns : (nominal_new, P_new)

    The two halves are independent:

      Half 1 -- NOMINAL, nonlinear. De-bias the raw increments with the current
        estimates (gyro bias corrects the angle increment, accel bias the
        velocity increment -- M2 R3), then push (p, v, q) through the full
        mechanization. The biases themselves are random walks, so their nominal
        value is unchanged here; only their covariance grows (via Q).

      Half 2 -- ERROR COVARIANCE, linear. Build F at the nominal operating point
        (OLD attitude, the same one mechanization rotates with) and propagate
        P through the discrete transition Phi = I + F*dt:

            P_new = Phi P Phi^T + Q

    The error MEAN is not propagated -- it is zero going in (reset after the last
    update) and Phi*0 = 0 -- so this step only ever touches the nominal and P.
    """
    b_a = nominal[NOM_BA]
    b_g = nominal[NOM_BG]
    dtheta_corr = dtheta - b_g * dt    # gyro bias corrects the ANGLE increment
    dv_corr = dv - b_a * dt            # accel bias corrects the VELOCITY increment

    # --- Half 1: propagate the NOMINAL, nonlinearly ---
    p, v, q = nominal[NOM_P], nominal[NOM_V], nominal[NOM_Q]
    p_new, v_new, q_new = mechanize_step((p, v, q), dtheta_corr, dv_corr, dt)
    # Biases unchanged (random-walk mean); repack in the same slot order.
    nominal_new = np.hstack((p_new, v_new, q_new, b_a, b_g))

    # --- Half 2: propagate the ERROR COVARIANCE, linearly ---
    C_nb = quat_to_dcm(q)              # OLD nominal attitude = linearization point
    f_nav = C_nb @ (dv_corr / dt)      # specific force in NED (what F wants)
    F = error_dynamics_F(C_nb, f_nav)
    Phi = np.eye(P.shape[0]) + F * dt
    P_new = Phi @ P @ Phi.T + Q

    return nominal_new, P_new


def esekf_update(nominal: np.ndarray, P: np.ndarray, z: np.ndarray,
                 h: Callable[[np.ndarray], np.ndarray], H: np.ndarray,
                 R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One error-state EKF measurement update: correct, inject, reset.

    A measurement z (our velocity pseudo-measurement now; ZUPT / GNSS position
    later) refines the belief. The whole update runs on the ERROR, whose mean is
    ZERO going in -- so it reduces to a plain linear Kalman update on dx = 0, and
    the only nonlinear piece is the innovation, which compares z against the FULL
    nominal prediction h(nominal):

        y   = z - h(nominal)           innovation (nonlinear, from the nominal)
        S   = H P H^T + R              innovation covariance
        K   = P H^T S^-1              gain
        dx  = 0 + K y                  estimated error (prior mean is zero)
        P+  = (I - K H) P (I - K H)^T + K R K^T    (Joseph form)

    then INJECT dx into the nominal. The RESET is implicit: dx is never carried
    forward, so the error mean is zero again for the next predict step -- which
    is exactly what keeps the linearization point pinned near zero.

    Note the domains differ: h maps the 16-nominal into measurement space, while
    H = dh/d(dx) maps the 15-error. That is precisely why the innovation must be
    formed with h(nominal), NOT H @ nominal.

    nominal : current 16-vector nominal state
    P       : current 15x15 error covariance
    z       : measurement (m-vector)
    h       : measurement model, callable nominal -> predicted measurement (m,)
    H       : measurement Jacobian dh/d(dx), shape (m, 15)
    R       : measurement-noise covariance, shape (m, m)
    returns : (nominal_new, P_new)
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

    return nominal_new, P_new
