"""Tests for the clock-aware GNSS position-aiding helpers (M7 R5).

R5 keeps the GNSS measurement itself as NED position, but the linearized model
adds one new term:

    y ~= dp + v_nom * dtau + noise

So these tests pin the four new ideas introduced by the clock state:

  * the time convention `t_i = t_g + tau`
  * the state-dependent Jacobian column `v_nom`
  * reduction to the old GNSS update when `v_nom = 0`
  * the sign of the clock correction under an along-track residual

There is also one direct NIS oracle so the clock-aware gating path is covered,
not just the plain update.
"""
import numpy as np

from bootins.estimation.esekf import NOM_P as NOM_P_BASE
from bootins.estimation.clock_augmented_esekf import ERR_DP, ERR_DTAU, NOM_P, NOM_TAU, NOM_V
from bootins.aiding.gnss_position import gnss_position_update_ned as gnss_position_update_ned_base
from bootins.aiding.gnss_position_clock import (
    _H_POS,
    gnss_position_innovation_covariance_ned,
    gnss_position_innovation_ned,
    gnss_position_nis_ned,
    gnss_position_update_ned,
    gnss_time_to_imu_time,
)


def _nominal0_base() -> np.ndarray:
    """Simple baseline nominal state."""
    return np.hstack((
        np.zeros(3),
        np.zeros(3),
        [1.0, 0.0, 0.0, 0.0],
        np.zeros(3),
        np.zeros(3),
    ))


def _augment_nominal(nominal_base: np.ndarray, tau: float) -> np.ndarray:
    """Append a nominal clock offset to the baseline GNSS/ESEKF state."""
    return np.hstack((nominal_base, tau))


def _augment_covariance(P_base: np.ndarray, tau_var: float) -> np.ndarray:
    """Embed a 15x15 covariance inside the augmented 16x16 error covariance."""
    P_clock = np.zeros((16, 16), dtype=float)
    P_clock[:15, :15] = P_base
    P_clock[ERR_DTAU, ERR_DTAU] = tau_var
    return P_clock


def _H_pos_local(nominal: np.ndarray) -> np.ndarray:
    """Independent local copy of the clock-aware GNSS position Jacobian."""
    H = np.zeros((3, 16), dtype=float)
    H[:, ERR_DP] = np.eye(3)
    H[:, ERR_DTAU] = nominal[NOM_V]
    return H


def _nis_pos_local(
    nominal: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    R_pos: np.ndarray,
) -> float:
    """Independent local copy of the clock-aware GNSS position NIS oracle."""
    H = _H_pos_local(nominal)
    y = np.asarray(z, dtype=float) - nominal[NOM_P]
    S = H @ P @ H.T + R_pos
    return float(y.T @ np.linalg.inv(S) @ y)


def test_gnss_time_to_imu_time_uses_t_i_equals_t_g_plus_tau():
    np.testing.assert_allclose(gnss_time_to_imu_time(12.3, 0.4), 12.7, atol=0.0)
    np.testing.assert_allclose(gnss_time_to_imu_time(12.3, -0.4), 11.9, atol=0.0)


def test_H_pos_uses_nominal_velocity_in_the_dtau_column():
    nominal = _augment_nominal(_nominal0_base(), tau=0.1)
    nominal[NOM_V] = np.array([1.0, -2.0, 0.5])

    H_expected = _H_pos_local(nominal)
    H = _H_POS(nominal)

    np.testing.assert_allclose(H, H_expected, atol=0.0)


def test_zero_velocity_reduces_to_old_gnss_update_and_leaves_tau_unchanged():
    nominal_base = _nominal0_base()
    nominal_base[NOM_P_BASE] = np.array([1.0, -2.0, 0.5])
    nominal_clock = _augment_nominal(nominal_base, tau=0.2)

    P_base = np.diag([
        4.0, 5.0, 6.0,
        1.0, 2.0, 3.0,
        1e-4, 1e-4, 1e-4,
        1.0, 1.0, 1.0,
        1e-6, 1e-6, 1e-6,
    ]).astype(float)
    tau_var0 = 7.0
    P_clock = _augment_covariance(P_base, tau_var0)

    z = np.array([3.0, -4.0, 2.0], dtype=float)
    R_pos = np.diag([0.5, 0.75, 1.25]).astype(float)

    nominal_base_new, P_base_new = gnss_position_update_ned_base(
        nominal_base, P_base, z, R_pos
    )
    nominal_clock_new, P_clock_new = gnss_position_update_ned(
        nominal_clock, P_clock, z, R_pos
    )

    np.testing.assert_allclose(
        nominal_clock_new[:nominal_base_new.size], nominal_base_new, atol=1e-12
    )
    np.testing.assert_allclose(P_clock_new[:15, :15], P_base_new, atol=1e-12)
    np.testing.assert_allclose(nominal_clock_new[NOM_TAU], 0.2, atol=0.0)
    np.testing.assert_allclose(P_clock_new[ERR_DTAU, ERR_DTAU], tau_var0, atol=0.0)
    np.testing.assert_allclose(P_clock_new[:15, ERR_DTAU], np.zeros(15), atol=0.0)
    np.testing.assert_allclose(P_clock_new[ERR_DTAU, :15], np.zeros(15), atol=0.0)


def test_positive_along_track_residual_increases_tau():
    tau0 = 0.2
    nominal = _augment_nominal(_nominal0_base(), tau=tau0)
    nominal[NOM_V] = np.array([1.0, 0.0, 0.0])

    P = np.zeros((16, 16), dtype=float)
    P[ERR_DTAU, ERR_DTAU] = 1.0

    z = np.array([0.3, 0.0, 0.0], dtype=float)
    R_pos = np.eye(3) * 0.25

    nominal_new, _ = gnss_position_update_ned(nominal, P, z, R_pos)

    expected_dtau = (1.0 / (1.0 + 0.25)) * 0.3
    np.testing.assert_allclose(nominal_new[NOM_TAU], tau0 + expected_dtau, atol=1e-12)
    np.testing.assert_allclose(nominal_new[NOM_P], nominal[NOM_P], atol=1e-12)


def test_gnss_position_innovation_ned_is_measurement_minus_nominal_position():
    nominal = _augment_nominal(_nominal0_base(), tau=0.1)
    nominal[NOM_P] = np.array([1.0, -2.0, 0.5])

    innovation = gnss_position_innovation_ned(nominal, [3.0, -4.0, 2.0])

    np.testing.assert_allclose(innovation, np.array([2.0, -2.0, 1.5]), atol=0.0)


def test_gnss_position_innovation_covariance_ned_matches_manual_formula():
    nominal = _augment_nominal(_nominal0_base(), tau=0.1)
    nominal[NOM_V] = np.array([1.5, -0.5, 0.25])

    P = np.diag([
        4.0, 5.0, 6.0,
        1.0, 2.0, 3.0,
        1e-4, 1e-4, 1e-4,
        1.0, 1.0, 1.0,
        1e-6, 1e-6, 1e-6,
        0.8,
    ]).astype(float)
    P[0, ERR_DTAU] = 0.1
    P[ERR_DTAU, 0] = 0.1
    R_pos = np.diag([0.5, 0.75, 1.25]).astype(float)

    H = _H_pos_local(nominal)
    expected = H @ P @ H.T + R_pos
    innovation_covariance = gnss_position_innovation_covariance_ned(
        nominal, P, R_pos
    )

    np.testing.assert_allclose(innovation_covariance, expected, atol=0.0)


def test_gnss_position_nis_ned_matches_manual_formula():
    nominal = _augment_nominal(_nominal0_base(), tau=0.1)
    nominal[NOM_P] = np.array([1.0, -2.0, 0.5])
    nominal[NOM_V] = np.array([1.5, -0.5, 0.25])

    P = np.diag([
        4.0, 5.0, 6.0,
        1.0, 2.0, 3.0,
        1e-4, 1e-4, 1e-4,
        1.0, 1.0, 1.0,
        1e-6, 1e-6, 1e-6,
        0.8,
    ]).astype(float)
    P[0, ERR_DTAU] = 0.1
    P[ERR_DTAU, 0] = 0.1

    z = np.array([3.0, -4.0, 2.0], dtype=float)
    R_pos = np.diag([0.5, 0.75, 1.25]).astype(float)

    nis_manual = _nis_pos_local(nominal, P, z, R_pos)
    nis = gnss_position_nis_ned(nominal, P, z, R_pos)

    np.testing.assert_allclose(nis, nis_manual, atol=0.0)
