from dataclasses import dataclass

import numpy as np
import pytest

from bootins.estimation.clock_augmented_esekf import ERR_DTAU, NOM_TAU, esekf_predict_clock
from bootins.aiding.gnss_position_clock import gnss_position_update_ned_gated
from bootins.replay.clock_delayed_gnss import (
    TimedGnssFixNed as TimedGnssFixNedR5,
    gnss_delayed_aided as gnss_delayed_aided_r5,
)
from bootins.replay.delayed_gnss import (
    TimedGnssFixNed as TimedGnssFixNedR4,
    gnss_delayed_aided as gnss_delayed_aided_r4,
)
from bootins.estimation.mechanization import G_NED, Increment


@dataclass(frozen=True)
class _TimedFixImu:
    t_fix_i: float
    p_ned: np.ndarray


def _nominal0_clock(tau: float = 0.0) -> np.ndarray:
    return np.hstack((
        np.zeros(3),              # position
        [1.0, 0.0, 0.0],          # 1 m/s north
        [1.0, 0.0, 0.0, 0.0],     # level attitude
        np.zeros(3),              # accel bias
        np.zeros(3),              # gyro bias
        tau,                      # clock offset
    ))


def _P0_clock(tau_var: float = 0.0) -> np.ndarray:
    P = np.zeros((16, 16), dtype=float)
    P[:15, :15] = np.diag([
        4.0, 4.0, 4.0,
        1.0, 1.0, 1.0,
        1e-4, 1e-4, 1e-4,
        1.0, 1.0, 1.0,
        1e-6, 1e-6, 1e-6,
    ]).astype(float)
    P[ERR_DTAU, ERR_DTAU] = tau_var
    return P


def _Q_clock(q_tau: float = 0.0) -> np.ndarray:
    Q = np.zeros((16, 16), dtype=float)
    Q[:15, :15] = np.eye(15) * 1e-6
    Q[ERR_DTAU, ERR_DTAU] = q_tau
    return Q


def _constant_velocity_increments(n_steps: int, dt: float) -> tuple[Increment, ...]:
    dv = -G_NED * dt
    return tuple((np.zeros(3), dv, dt) for _ in range(n_steps))


def _timely_gnss_clock_loop(
    nominal0: np.ndarray,
    P0: np.ndarray,
    increments: tuple[Increment, ...],
    gnss_fixes_i: tuple[_TimedFixImu, ...],
    Q: np.ndarray,
    R_pos: np.ndarray,
    nis_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    nominal = np.array(nominal0, dtype=float, copy=True)
    P = np.array(P0, dtype=float, copy=True)

    nominals = np.empty((len(increments), nominal.shape[0]), dtype=float)
    Ps = np.empty((len(increments), *P.shape), dtype=float)
    next_fix_idx = 0
    t_now = 0.0

    for k, increment in enumerate(increments):
        dtheta, dv, dt = increment
        nominal, P = esekf_predict_clock(nominal, P, dtheta, dv, dt, Q)
        t_now += dt

        while next_fix_idx < len(gnss_fixes_i) and gnss_fixes_i[next_fix_idx].t_fix_i <= t_now:
            nominal, P, _nis, _accepted = gnss_position_update_ned_gated(
                nominal,
                P,
                gnss_fixes_i[next_fix_idx].p_ned,
                R_pos,
                nis_threshold,
            )
            next_fix_idx += 1

        nominals[k] = nominal
        Ps[k] = P

    return nominals, Ps


def test_zero_offset_reduces_to_old_delayed_runner():
    increments = _constant_velocity_increments(n_steps=6, dt=0.1)

    nominal0_clock = _nominal0_clock(tau=0.0)
    P0_clock = _P0_clock(tau_var=0.0)
    Q_clock = _Q_clock(q_tau=0.0)

    nominal0_old = nominal0_clock[:16]
    P0_old = P0_clock[:15, :15]
    Q_old = Q_clock[:15, :15]

    R_pos = np.eye(3) * 0.5
    nis_threshold = 1e6
    p_fix = np.array([0.35, -0.15, 0.0])

    gnss_fix_r4 = TimedGnssFixNedR4(t_fix=0.2, t_arrival=0.5, p_ned=p_fix)
    gnss_fix_r5 = TimedGnssFixNedR5(t_fix_g=0.2, t_arrival_i=0.5, p_ned=p_fix)

    nominals_r4, Ps_r4 = gnss_delayed_aided_r4(
        nominal0_old, P0_old, increments, (gnss_fix_r4,), Q_old, R_pos, nis_threshold, 1.0
    )
    nominals_r5, Ps_r5 = gnss_delayed_aided_r5(
        nominal0_clock, P0_clock, increments, (gnss_fix_r5,), Q_clock, R_pos, nis_threshold, 1.0
    )

    np.testing.assert_allclose(nominals_r5[:, :16], nominals_r4, atol=1e-12)
    np.testing.assert_allclose(Ps_r5[:, :15, :15], Ps_r4, atol=1e-12)
    np.testing.assert_allclose(nominals_r5[:, NOM_TAU], 0.0, atol=0.0)
    np.testing.assert_allclose(Ps_r5[:, ERR_DTAU, :], 0.0, atol=0.0)
    np.testing.assert_allclose(Ps_r5[:, :, ERR_DTAU], 0.0, atol=0.0)


def test_mapped_fix_epoch_uses_t_fix_g_plus_tau():
    increments = _constant_velocity_increments(n_steps=7, dt=0.1)
    nominal0 = _nominal0_clock(tau=0.3)
    P0 = _P0_clock(tau_var=0.0)
    Q = _Q_clock(q_tau=0.0)

    R_pos = np.eye(3) * 0.5
    nis_threshold = 1e6
    p_fix = np.array([0.35, -0.15, 0.0])

    delayed_fix = TimedGnssFixNedR5(t_fix_g=0.2, t_arrival_i=0.6, p_ned=p_fix)
    timely_fix = _TimedFixImu(t_fix_i=0.5, p_ned=p_fix)

    nominals_delayed, Ps_delayed = gnss_delayed_aided_r5(
        nominal0, P0, increments, (delayed_fix,), Q, R_pos, nis_threshold, 1.0
    )
    nominals_timely, Ps_timely = _timely_gnss_clock_loop(
        nominal0, P0, increments, (timely_fix,), Q, R_pos, nis_threshold
    )

    arrival_idx = 5   # dt = 0.1 s, so t = 0.6 s is sample index 5
    assert not np.allclose(nominals_delayed[arrival_idx - 1], nominals_timely[arrival_idx - 1])
    np.testing.assert_allclose(nominals_delayed[arrival_idx:], nominals_timely[arrival_idx:], atol=1e-12)
    np.testing.assert_allclose(Ps_delayed[arrival_idx:], Ps_timely[arrival_idx:], atol=1e-12)


def test_stale_mapped_fix_is_dropped():
    increments = _constant_velocity_increments(n_steps=6, dt=0.1)
    nominal0 = _nominal0_clock(tau=-0.4)
    P0 = _P0_clock(tau_var=0.0)
    Q = _Q_clock(q_tau=0.0)

    R_pos = np.eye(3) * 0.5
    nis_threshold = 1e6
    stale_fix = TimedGnssFixNedR5(
        t_fix_g=0.45,      # raw GNSS time would look recent
        t_arrival_i=0.6,
        p_ned=np.array([0.35, -0.15, 0.0]),
    )

    nominals_stale, Ps_stale = gnss_delayed_aided_r5(
        nominal0, P0, increments, (stale_fix,), Q, R_pos, nis_threshold, 0.2
    )
    nominals_no_fix, Ps_no_fix = gnss_delayed_aided_r5(
        nominal0, P0, increments, (), Q, R_pos, nis_threshold, 0.2
    )

    np.testing.assert_allclose(nominals_stale, nominals_no_fix, atol=1e-12)
    np.testing.assert_allclose(Ps_stale, Ps_no_fix, atol=1e-12)


def test_delayed_outlier_still_gets_gated_out():
    increments = _constant_velocity_increments(n_steps=7, dt=0.1)
    nominal0 = _nominal0_clock(tau=0.3)
    P0 = _P0_clock(tau_var=0.0)
    Q = _Q_clock(q_tau=0.0)

    R_pos = np.eye(3) * 0.25
    outlier_fix = TimedGnssFixNedR5(
        t_fix_g=0.2,
        t_arrival_i=0.6,
        p_ned=np.array([50.0, -50.0, 20.0]),
    )

    nominals_outlier, Ps_outlier = gnss_delayed_aided_r5(
        nominal0, P0, increments, (outlier_fix,), Q, R_pos, 10.0, 1.0
    )
    nominals_no_fix, Ps_no_fix = gnss_delayed_aided_r5(
        nominal0, P0, increments, (), Q, R_pos, 10.0, 1.0
    )

    np.testing.assert_allclose(nominals_outlier, nominals_no_fix, atol=1e-12)
    np.testing.assert_allclose(Ps_outlier, Ps_no_fix, atol=1e-12)


def test_out_of_order_arrival_i_raises():
    increments = _constant_velocity_increments(n_steps=2, dt=0.1)
    nominal0 = _nominal0_clock(tau=0.0)
    P0 = _P0_clock(tau_var=0.0)
    Q = _Q_clock(q_tau=0.0)
    R_pos = np.eye(3) * 0.5

    fix1 = TimedGnssFixNedR5(t_fix_g=0.1, t_arrival_i=0.5, p_ned=np.zeros(3))
    fix2 = TimedGnssFixNedR5(t_fix_g=0.2, t_arrival_i=0.4, p_ned=np.zeros(3))

    with pytest.raises(ValueError, match="t_arrival_i"):
        gnss_delayed_aided_r5(
            nominal0, P0, increments, (fix1, fix2), Q, R_pos, 1e6, 1.0
        )
