"""Tests for the M6 causal ZUPT-aided EKF runner.

These tests are intentionally about TIME ALIGNMENT and CONTRACT, not threshold
"vibes." The detector math itself is already pinned in `test_zupt.py`; this file
checks that the detector decisions are wired into the EKF at the correct sample,
with the correct startup behavior, and with no accidental updates when stance is
false.
"""
import numpy as np

from bootins.error_state_ekf import NOM_V, esekf_predict, esekf_update
from bootins.mechanization import G_NED
from bootins.zupt_aided import zupt_aided

SIGMA_A = 0.5
SIGMA_G = 0.25
G = np.linalg.norm(G_NED)


def _vel_h(nominal: np.ndarray) -> np.ndarray:
    """Local copy of the ZUPT measurement model used by the manual oracle."""
    return nominal[NOM_V]


_H_VEL = np.zeros((3, 15), dtype=float)
_H_VEL[:, 3:6] = np.eye(3)


def _measurements_from_rates(
    omega_body: np.ndarray,
    f_body: np.ndarray,
    dt_samples: list[float],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Pack constant rates into the repo's `(dtheta, dv, dt)` convention."""
    omega_body = np.asarray(omega_body, dtype=float)
    f_body = np.asarray(f_body, dtype=float)
    return [
        (omega_body * dt, f_body * dt, float(dt))
        for dt in np.asarray(dt_samples, dtype=float)
    ]


def _manual_zupt_loop(
    nominal0: np.ndarray,
    P0: np.ndarray,
    measurements: list[tuple[np.ndarray, np.ndarray, float]],
    flags: np.ndarray,
    Q: np.ndarray,
    R_zupt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Manual oracle: run predict, then optional ZUPT update, one sample at a time."""
    nominal = np.array(nominal0, dtype=float, copy=True)
    P = np.array(P0, dtype=float, copy=True)
    nominals = np.empty((len(measurements), nominal.shape[0]), dtype=float)
    Ps = np.empty((len(measurements), *P.shape), dtype=float)
    z_zupt = np.zeros(3, dtype=float)

    for k, (measurement, flag) in enumerate(zip(measurements, flags, strict=True)):
        dtheta, dv, dt = measurement
        nominal, P = esekf_predict(nominal, P, dtheta, dv, dt, Q)
        if flag:
            nominal, P = esekf_update(nominal, P, z_zupt, _vel_h, _H_VEL, R_zupt)
        nominals[k] = nominal
        Ps[k] = P

    return nominals, Ps


def _nominal0() -> np.ndarray:
    """Initial nominal with a deliberate nonzero velocity so ZUPT has work to do."""
    return np.hstack((
        np.zeros(3),              # position
        [0.2, -0.1, 0.3],         # velocity
        [1.0, 0.0, 0.0, 0.0],     # level attitude
        np.zeros(3),              # accel bias
        np.zeros(3),              # gyro bias
    ))


def _P0() -> np.ndarray:
    """A nontrivial prior covariance so the update can shrink uncertainty."""
    return np.diag([
        1.0, 1.0, 1.0,            # dp
        1.0, 1.0, 1.0,            # dv
        1e-4, 1e-4, 1e-4,         # dpsi
        1.0, 1.0, 1.0,            # db_a
        1e-6, 1e-6, 1e-6,         # db_g
    ]).astype(float)


def test_zupt_aided_window_size_one_matches_manual_predict_then_update():
    """With a 1-sample window, every perfect-rest sample is immediately a ZUPT."""
    measurements = _measurements_from_rates(
        np.zeros(3), np.array([0.0, 0.0, -G]), [0.010, 0.015, 0.020, 0.010]
    )
    Q = np.eye(15) * 1e-9
    R_zupt = np.eye(3) * 1e-4

    flags, nominals, Ps = zupt_aided(
        _nominal0(), _P0(), measurements, Q, R_zupt,
        window_size=1, threshold=1e-6, sigma_a=SIGMA_A, sigma_g=SIGMA_G,
    )

    expected_flags = np.ones(len(measurements), dtype=bool)
    expected_nominals, expected_Ps = _manual_zupt_loop(
        _nominal0(), _P0(), measurements, expected_flags, Q, R_zupt
    )

    np.testing.assert_array_equal(flags, expected_flags)
    np.testing.assert_allclose(nominals, expected_nominals, atol=1e-12)
    np.testing.assert_allclose(Ps, expected_Ps, atol=1e-12)


def test_zupt_aided_waits_for_a_full_window_before_first_update():
    """A 3-sample trailing detector must leave the first two samples update-free."""
    measurements = _measurements_from_rates(
        np.zeros(3), np.array([0.0, 0.0, -G]), [0.010] * 5
    )
    Q = np.eye(15) * 1e-9
    R_zupt = np.eye(3) * 1e-4

    flags, nominals, Ps = zupt_aided(
        _nominal0(), _P0(), measurements, Q, R_zupt,
        window_size=3, threshold=1e-6, sigma_a=SIGMA_A, sigma_g=SIGMA_G,
    )

    expected_flags = np.array([False, False, True, True, True])
    expected_nominals, expected_Ps = _manual_zupt_loop(
        _nominal0(), _P0(), measurements, expected_flags, Q, R_zupt
    )

    np.testing.assert_array_equal(flags, expected_flags)
    np.testing.assert_allclose(nominals, expected_nominals, atol=1e-12)
    np.testing.assert_allclose(Ps, expected_Ps, atol=1e-12)


def test_zupt_aided_never_updates_when_detector_never_fires():
    """If every sample is non-stance, the runner must reduce to predict-only."""
    extra_force = 1.0
    measurements = _measurements_from_rates(
        np.zeros(3), np.array([0.0, 0.0, -(G + extra_force)]), [0.010, 0.020, 0.015, 0.010]
    )
    Q = np.eye(15) * 1e-9
    R_zupt = np.eye(3) * 1e-4

    flags, nominals, Ps = zupt_aided(
        _nominal0(), _P0(), measurements, Q, R_zupt,
        window_size=1, threshold=1.0, sigma_a=SIGMA_A, sigma_g=SIGMA_G,
    )

    expected_flags = np.zeros(len(measurements), dtype=bool)
    expected_nominals, expected_Ps = _manual_zupt_loop(
        _nominal0(), _P0(), measurements, expected_flags, Q, R_zupt
    )

    np.testing.assert_array_equal(flags, expected_flags)
    np.testing.assert_allclose(nominals, expected_nominals, atol=1e-12)
    np.testing.assert_allclose(Ps, expected_Ps, atol=1e-12)


def test_zupt_aided_empty_stream_returns_empty_aligned_outputs():
    """The empty-stream contract should be explicit and shape-stable."""
    flags, nominals, Ps = zupt_aided(
        _nominal0(), _P0(), [], np.eye(15), np.eye(3),
        window_size=1, threshold=1.0, sigma_a=SIGMA_A, sigma_g=SIGMA_G,
    )

    assert flags.shape == (0,)
    assert nominals.shape == (0, 16)
    assert Ps.shape == (0, 15, 15)
