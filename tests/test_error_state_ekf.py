"""Tests for the error-state EKF (M5 Rung 5).

`inject` is the conceptual heart of the rung -- the multiplicative attitude fold
is the one place the correction is not a plain `+=`, so it gets two oracles:

  * ADDITIVE CHANNELS -- with a zero attitude error, every additive slice must
    move by exactly its part of dx, the attitude must be untouched, and the
    caller's array must not be mutated. This pins the index/layout mapping (the
    exact place a swapped bias would hide) and the copy-not-mutate contract.
  * ATTITUDE FOLD -- a small dpsi about a known axis must rotate the nominal by
    that same small rotation, applied on the LEFT (nav-frame error). We check it
    through the DCM built independently as C(dq) @ C(q0) -- pre-multiplication --
    so a right-multiply (dq on the wrong side) fails whenever q0 is a non-trivial
    rotation about a different axis.
"""
import numpy as np

from bootins.error_state_ekf import (
    inject, esekf_predict, esekf_update,
    NOM_P, NOM_V, NOM_Q, NOM_BA, NOM_BG,
)
from bootins.mechanization import mechanize_step
from bootins.frames.quaternion import quat_from_axis_angle, quat_to_dcm


# --- Velocity pseudo-measurement (the R5 scaffold; a real ZUPT is this with z=0)
# h maps the nominal to its velocity; H selects the velocity block of the error.
def _vel_h(nom):
    return nom[NOM_V]


_H_VEL = np.zeros((3, 15))
_H_VEL[:, 3:6] = np.eye(3)

G = 9.80665


def test_inject_additive_channels_and_no_mutation():
    # Nominal with a level (identity) attitude so a zero attitude error leaves q
    # provably unchanged; arbitrary but distinct values elsewhere.
    nominal0 = np.array([
        1.0, 2.0, 3.0,        # p
        4.0, 5.0, 6.0,        # v
        1.0, 0.0, 0.0, 0.0,   # q = identity
        0.10, 0.20, 0.30,     # b_a
        0.01, 0.02, 0.03,     # b_g
    ])
    nominal_before = nominal0.copy()

    dx = np.array([
        0.5, -0.5, 1.0,       # dp
        -1.0, 2.0, 0.25,      # dv
        0.0, 0.0, 0.0,        # dpsi = 0 -> attitude untouched
        0.001, -0.002, 0.003, # db_a
        -0.01, 0.02, -0.03,   # db_g
    ])

    out = inject(nominal0, dx)

    # Each additive channel moved by exactly its slice of dx.
    np.testing.assert_allclose(out[NOM_P], [1.5, 1.5, 4.0])
    np.testing.assert_allclose(out[NOM_V], [3.0, 7.0, 6.25])
    np.testing.assert_allclose(out[NOM_BA], [0.101, 0.198, 0.303])
    np.testing.assert_allclose(out[NOM_BG], [0.0, 0.04, 0.0])
    # Zero attitude error -> identity quaternion is unchanged.
    np.testing.assert_allclose(out[NOM_Q], [1.0, 0.0, 0.0, 0.0])
    # The caller's array must NOT have been mutated (copy-not-mutate contract).
    np.testing.assert_allclose(nominal0, nominal_before)


def test_inject_attitude_is_small_left_rotation():
    # A non-trivial starting attitude ABOUT A DIFFERENT AXIS than the error, so
    # left- vs right-multiply give genuinely different answers.
    q0 = quat_from_axis_angle([0.0, 1.0, 0.0], 0.3)   # 0.3 rad about body y
    nominal0 = np.hstack((
        np.zeros(3), np.zeros(3), q0, np.zeros(3), np.zeros(3),
    ))

    alpha = 1e-4                                       # tiny -> linear dq ~ exact
    dpsi = np.array([alpha, 0.0, 0.0])                 # small tilt about x
    dx = np.zeros(15)
    dx[6:9] = dpsi

    out = inject(nominal0, dx)

    # Independent expected DCM: a LEFT (pre-)multiply of the nominal rotation by
    # the small rotation, dcm(dq (x) q0) = dcm(dq) @ dcm(q0). Built from the exact
    # axis-angle quaternion (not inject's linear form), so agreement to 1e-6
    # confirms the axis, the half-angle, AND the side -- the linear-vs-exact gap
    # is only O(alpha^3) ~ 1e-12 here.
    dq_true = quat_from_axis_angle(dpsi, np.linalg.norm(dpsi))
    C_expected = quat_to_dcm(dq_true) @ quat_to_dcm(q0)

    np.testing.assert_allclose(quat_to_dcm(out[NOM_Q]), C_expected, atol=1e-6)


# --- esekf_predict: the de-bias channels ------------------------------------

def test_predict_debias_removes_the_right_bias_from_each_channel():
    # The runtime guard against the swapped-channel bug: gyro bias must correct
    # the ANGLE increment, accel bias the VELOCITY increment (M2 R3). We bake a
    # KNOWN bias into the raw increments, tell the filter that exact bias, and
    # demand the propagated nominal equals a clean mechanize_step of the
    # bias-free increments. Cross the channels and the corrected increments are
    # wrong -> the nominal disagrees. b_a and b_g are distinct so a swap cannot
    # accidentally still match.
    dt = 0.01
    b_a = np.array([0.10, 0.20, 0.30])
    b_g = np.array([0.01, 0.02, 0.03])

    dtheta_clean = np.array([0.001, -0.002, 0.003])
    dv_clean = np.array([0.05, -0.10, -G]) * dt

    # Raw increments = clean signal + the bias the sensor actually carries.
    dtheta_raw = dtheta_clean + b_g * dt
    dv_raw = dv_clean + b_a * dt

    q0 = quat_from_axis_angle([0.0, 1.0, 0.0], 0.2)   # a non-trivial attitude
    nominal = np.hstack((np.zeros(3), np.zeros(3), q0, b_a, b_g))
    P = np.eye(15) * 0.01
    Q = np.eye(15) * 1e-9

    nominal_new, _ = esekf_predict(nominal, P, dtheta_raw, dv_raw, dt, Q)

    # Independent oracle: mechanize the CLEAN increments directly.
    p_e, v_e, q_e = mechanize_step((np.zeros(3), np.zeros(3), q0),
                                   dtheta_clean, dv_clean, dt)
    np.testing.assert_allclose(nominal_new[NOM_P], p_e, atol=1e-12)
    np.testing.assert_allclose(nominal_new[NOM_V], v_e, atol=1e-12)
    np.testing.assert_allclose(nominal_new[NOM_Q], q_e, atol=1e-12)
    # Biases are random-walk means -> unchanged by predict.
    np.testing.assert_allclose(nominal_new[NOM_BA], b_a)
    np.testing.assert_allclose(nominal_new[NOM_BG], b_g)


def test_predict_covariance_stays_symmetric_psd():
    # P_new = Phi P Phi^T + Q is a congruence sandwich plus a PSD term, so it
    # must stay symmetric and PSD for any valid P, Q.
    dt = 0.01
    nominal = np.hstack((np.zeros(3), np.zeros(3), [1., 0, 0, 0],
                         np.zeros(3), np.zeros(3)))
    P = np.diag(np.arange(1, 16)).astype(float)       # asymmetric-looking but diag
    Q = np.eye(15) * 1e-6

    _, P_new = esekf_predict(nominal, P, np.zeros(3),
                             np.array([0., 0., -G]) * dt, dt, Q)

    np.testing.assert_allclose(P_new, P_new.T)
    assert np.all(np.linalg.eigvalsh(P_new) >= -1e-12)


# --- esekf_update: single-step behaviour ------------------------------------

def test_update_pulls_velocity_toward_measurement_and_shrinks_P():
    # A nominal that thinks it is moving up at 0.5 m/s, measured against a
    # zero-velocity truth (a ZUPT). One update must move velocity toward 0 and
    # shrink the covariance, with a symmetric-PSD posterior (Joseph guarantee).
    nominal = np.hstack((np.zeros(3), [0., 0., 0.5], [1., 0, 0, 0],
                         np.zeros(3), np.zeros(3)))
    P = np.diag([1, 1, 1, 1, 1, 1, 1e-4, 1e-4, 1e-4,
                 1, 1, 1, 1e-6, 1e-6, 1e-6]).astype(float)
    R = np.eye(3) * 1e-4

    nominal_new, P_new = esekf_update(nominal, P, np.zeros(3), _vel_h, _H_VEL, R)

    # Velocity moved toward 0 without overshooting past it.
    assert 0.0 <= nominal_new[NOM_V][2] < 0.5
    # Covariance shrank.
    assert np.trace(P_new) < np.trace(P)
    # Joseph form keeps it symmetric-PSD.
    np.testing.assert_allclose(P_new, P_new.T)
    assert np.all(np.linalg.eigvalsh(P_new) >= -1e-12)


# --- The capstone: the full loop recovers a hidden bias ---------------------

def test_zupt_loop_recovers_vertical_accel_bias():
    # THE R5 oracle. A boot at rest and level with a hidden VERTICAL accel bias.
    # The bias makes the open-loop nominal velocity ramp; repeated zero-velocity
    # updates (ZUPT) pin velocity AND -- through the [dv, db_a] off-diagonal that
    # esekf_predict builds into P -- back out the bias itself. Vertical bias is
    # chosen so there is no tilt/accel-bias confound (horizontal tilt cannot leak
    # into vertical velocity), giving clean, deterministic convergence.
    dt = 0.01
    b_a_true = np.array([0.0, 0.0, 0.1])
    dtheta = np.zeros(3)                               # level, at rest, no gyro bias
    dv = (np.array([0.0, 0.0, -G]) + b_a_true) * dt    # biased specific-force increment

    # Filter starts with the CORRECT p/v/attitude but a WRONG (zero) bias guess.
    nominal = np.hstack((np.zeros(3), np.zeros(3), [1., 0, 0, 0],
                         np.zeros(3), np.zeros(3)))
    P = np.diag([1, 1, 1, 1, 1, 1, 1e-4, 1e-4, 1e-4,
                 1, 1, 1, 1e-6, 1e-6, 1e-6]).astype(float)
    Q = np.eye(15) * 1e-9
    R = np.eye(3) * 1e-6
    z = np.zeros(3)                                    # ZUPT: velocity is known zero

    for _ in range(600):
        nominal, P = esekf_predict(nominal, P, dtheta, dv, dt, Q)
        nominal, P = esekf_update(nominal, P, z, _vel_h, _H_VEL, R)

    # The hidden bias is recovered; velocity is pinned; no false horizontal bias.
    np.testing.assert_allclose(nominal[NOM_BA], b_a_true, atol=1e-6)
    np.testing.assert_allclose(nominal[NOM_V], np.zeros(3), atol=1e-6)
