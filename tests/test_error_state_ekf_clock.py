"""Tests for the clock-augmented error-state EKF (M7 R5).

R5 adds one nominal clock-offset state `tau` and one error-state channel
`dtau`. The navigation channels should keep behaving exactly like the already
proven 15-state filter, while the new clock channel should:

  * inject additively into the nominal
  * stay constant in the nominal mean during predict
  * grow in covariance through the new `Q_tau` process-noise slot
  * reduce exactly to the baseline 15-state filter when the clock channel is
    decoupled from the rest
"""
import numpy as np

from bootins.estimation.esekf import esekf_predict
from bootins.estimation.clock_augmented_esekf import (
    ERR_DTAU,
    NOM_TAU,
    esekf_predict_clock,
    esekf_update_clock,
    inject_clock,
)
from bootins.frames.quaternion import quat_from_axis_angle

G = 9.80665


def _augment_nominal(nominal_base: np.ndarray, tau: float) -> np.ndarray:
    """Append a nominal clock offset to the baseline 16-state nominal."""
    return np.hstack((nominal_base, tau))


def _augment_covariance(P_base: np.ndarray, tau_var: float) -> np.ndarray:
    """Embed a 15x15 covariance inside the augmented 16x16 error covariance."""
    P_clock = np.zeros((16, 16), dtype=float)
    P_clock[:15, :15] = P_base
    P_clock[ERR_DTAU, ERR_DTAU] = tau_var
    return P_clock


def test_inject_clock_adds_dtau_and_does_not_mutate():
    nominal0 = np.array([
        1.0, 2.0, 3.0,        # p
        4.0, 5.0, 6.0,        # v
        1.0, 0.0, 0.0, 0.0,   # q = identity
        0.10, 0.20, 0.30,     # b_a
        0.01, 0.02, 0.03,     # b_g
        0.10,                 # tau
    ])
    nominal_before = nominal0.copy()

    dx = np.array([
        0.5, -0.5, 1.0,       # dp
        -1.0, 2.0, 0.25,      # dv
        0.0, 0.0, 0.0,        # dpsi = 0 -> attitude untouched
        0.001, -0.002, 0.003, # db_a
        -0.01, 0.02, -0.03,   # db_g
        0.01,                 # dtau
    ])

    out = inject_clock(nominal0, dx)

    np.testing.assert_allclose(out[NOM_TAU], 0.11, atol=0.0)
    np.testing.assert_allclose(nominal0, nominal_before, atol=0.0)


def test_predict_clock_leaves_tau_nominal_unchanged():
    dt = 0.01
    b_a = np.array([0.10, 0.20, 0.30])
    b_g = np.array([0.01, 0.02, 0.03])
    tau = 0.10

    dtheta_clean = np.array([0.001, -0.002, 0.003])
    dv_clean = np.array([0.05, -0.10, -G]) * dt

    # Raw increments = clean signal + the bias the sensor actually carries.
    dtheta_raw = dtheta_clean + b_g * dt
    dv_raw = dv_clean + b_a * dt

    q0 = quat_from_axis_angle([0.0, 1.0, 0.0], 0.2)
    nominal = np.hstack((np.zeros(3), np.zeros(3), q0, b_a, b_g, tau))
    P = np.eye(16) * 0.01
    Q = np.eye(16) * 1e-9

    nominal_new, _ = esekf_predict_clock(nominal, P, dtheta_raw, dv_raw, dt, Q)

    np.testing.assert_allclose(nominal_new[NOM_TAU], tau, atol=0.0)


def test_predict_clock_grows_tau_variance():
    dt = 0.01
    tau = 0.25
    tau_var0 = 2.0
    q_tau = 0.05

    nominal = np.hstack((
        np.zeros(3),
        np.zeros(3),
        [1.0, 0.0, 0.0, 0.0],
        np.zeros(3),
        np.zeros(3),
        tau,
    ))
    P = np.zeros((16, 16), dtype=float)
    P[ERR_DTAU, ERR_DTAU] = tau_var0
    Q = np.zeros((16, 16), dtype=float)
    Q[ERR_DTAU, ERR_DTAU] = q_tau

    _, P_new = esekf_predict_clock(
        nominal, P, np.zeros(3), np.array([0.0, 0.0, -G]) * dt, dt, Q
    )

    np.testing.assert_allclose(P_new[ERR_DTAU, ERR_DTAU], tau_var0 + q_tau, atol=0.0)
    np.testing.assert_allclose(P_new[:15, ERR_DTAU], np.zeros(15), atol=0.0)
    np.testing.assert_allclose(P_new[ERR_DTAU, :15], np.zeros(15), atol=0.0)


def test_clock_covariance_steps_remove_floating_point_skew_exactly():
    """Clock-augmented predict and update retain exact stored symmetry too."""
    dt = 0.01
    nominal = np.hstack((
        np.zeros(3), np.zeros(3), [1.0, 0.0, 0.0, 0.0],
        np.zeros(3), np.zeros(3), 0.25,
    ))
    seed = np.arange(1, 16 * 16 + 1, dtype=float).reshape(16, 16) / 100.0
    P = seed @ seed.T + np.diag(np.linspace(0.1, 1.6, 16))
    Q = np.diag(np.linspace(1e-7, 3e-7, 16))
    H = np.zeros((3, 16))
    H[:, 3:6] = np.eye(3)
    R = np.diag([0.01, 0.02, 0.03])

    _, P_predict = esekf_predict_clock(
        nominal, P, np.array([0.013, -0.021, 0.008]),
        np.array([0.15, -0.30, -9.6]) * dt, dt, Q,
    )
    _, P_update = esekf_update_clock(
        nominal, P, np.zeros(3), lambda state: state[3:6], H, R,
    )

    np.testing.assert_array_equal(P_predict, P_predict.T)
    np.testing.assert_array_equal(P_update, P_update.T)


def test_predict_clock_reduces_to_old_predict_on_original_channels():
    dt = 0.01
    b_a = np.array([0.10, 0.20, 0.30])
    b_g = np.array([0.01, 0.02, 0.03])
    tau = 0.10
    tau_var0 = 7.0
    q_tau = 2e-6

    dtheta = np.array([0.001, -0.002, 0.003])
    dv = np.array([0.05, -0.10, -G]) * dt

    q0 = quat_from_axis_angle([0.0, 1.0, 0.0], 0.2)
    nominal_base = np.hstack((np.zeros(3), np.zeros(3), q0, b_a, b_g))
    nominal_clock = _augment_nominal(nominal_base, tau)

    P_base = np.diag(np.arange(1, 16)).astype(float)
    P_clock = _augment_covariance(P_base, tau_var0)

    Q_base = np.eye(15) * 1e-6
    Q_clock = _augment_covariance(Q_base, q_tau)

    nominal_base_new, P_base_new = esekf_predict(
        nominal_base, P_base, dtheta, dv, dt, Q_base
    )
    nominal_clock_new, P_clock_new = esekf_predict_clock(
        nominal_clock, P_clock, dtheta, dv, dt, Q_clock
    )

    np.testing.assert_allclose(
        nominal_clock_new[:nominal_base_new.size], nominal_base_new, atol=1e-12
    )
    np.testing.assert_allclose(P_clock_new[:15, :15], P_base_new, atol=1e-12)
    np.testing.assert_allclose(nominal_clock_new[NOM_TAU], tau, atol=0.0)
    np.testing.assert_allclose(
        P_clock_new[ERR_DTAU, ERR_DTAU], tau_var0 + q_tau, atol=0.0
    )
    np.testing.assert_allclose(P_clock_new[:15, ERR_DTAU], np.zeros(15), atol=0.0)
    np.testing.assert_allclose(P_clock_new[ERR_DTAU, :15], np.zeros(15), atol=0.0)
