"""Tests for the M7 R4 delayed-GNSS rewind-and-replay runner.

R4 is about USING a GNSS fix at the time it belongs to, not the time it
arrives. These tests pin the core delayed-measurement behaviors:

  * zero delay reduces to the ordinary in-order GNSS update loop
  * a delayed fix, once it arrives, catches the live state back up to the
    timely equivalent by rewind-and-replay
  * a fix older than the retained history horizon is dropped
  * R3 NIS gating still applies even when the GNSS fix is delayed
"""
import numpy as np

from bootins.estimation.esekf import esekf_predict
from bootins.aiding.gnss_position import gnss_position_update_ned_gated
from bootins.replay.delayed_gnss import TimedGnssFixNed, gnss_delayed_aided
from bootins.estimation.mechanization import G_NED, Increment


def _nominal0() -> np.ndarray:
    """Constant-velocity level prior used by the timing tests."""
    return np.hstack((
        np.zeros(3),              # position
        [1.0, 0.0, 0.0],          # 1 m/s north
        [1.0, 0.0, 0.0, 0.0],     # level attitude
        np.zeros(3),              # accel bias
        np.zeros(3),              # gyro bias
    ))


def _P0() -> np.ndarray:
    """Moderate prior covariance shared across the delayed-GNSS tests."""
    return np.diag([
        4.0, 4.0, 4.0,
        1.0, 1.0, 1.0,
        1e-4, 1e-4, 1e-4,
        1.0, 1.0, 1.0,
        1e-6, 1e-6, 1e-6,
    ]).astype(float)


def _Q() -> np.ndarray:
    """Small per-step process noise for the timing-only scenarios."""
    return np.eye(15, dtype=float) * 1e-6


def _constant_velocity_increments(n_steps: int, dt: float) -> tuple[Increment, ...]:
    """Level constant-velocity IMU increments: zero rotation, zero net accel."""
    dv = -G_NED * dt
    return tuple((np.zeros(3), dv, dt) for _ in range(n_steps))


def _timely_gnss_loop(
    nominal0: np.ndarray,
    P0: np.ndarray,
    increments: tuple[Increment, ...],
    gnss_fixes: tuple[TimedGnssFixNed, ...],
    Q: np.ndarray,
    R_pos: np.ndarray,
    nis_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference loop: apply each GNSS fix when its measurement epoch is reached."""
    nominal = np.array(nominal0, dtype=float, copy=True)
    P = np.array(P0, dtype=float, copy=True)

    nominals = np.empty((len(increments), nominal.shape[0]), dtype=float)
    Ps = np.empty((len(increments), *P.shape), dtype=float)
    next_fix_idx = 0
    t_now = 0.0

    for k, increment in enumerate(increments):
        dtheta, dv, dt = increment
        nominal, P = esekf_predict(nominal, P, dtheta, dv, dt, Q)
        t_now += dt

        while next_fix_idx < len(gnss_fixes) and gnss_fixes[next_fix_idx].t_fix <= t_now:
            nominal, P, _nis, _accepted = gnss_position_update_ned_gated(
                nominal,
                P,
                gnss_fixes[next_fix_idx].p_ned,
                R_pos,
                nis_threshold,
            )
            next_fix_idx += 1

        nominals[k] = nominal
        Ps[k] = P

    return nominals, Ps


def test_zero_delay_matches_timely_in_order_gnss_loop():
    """No delay should collapse exactly to the ordinary in-order GNSS loop.

    This is the first sanity test for R4: if `t_arrival == t_fix`, the delayed
    runner should not invent any extra behavior. It should reduce to "predict,
    then immediately fuse GNSS at that same epoch."
    """
    increments = _constant_velocity_increments(n_steps=6, dt=0.1)
    nominal0 = _nominal0()
    P0 = _P0()
    Q = _Q()
    R_pos = np.eye(3) * 0.5
    nis_threshold = 1e6
    gnss_fixes = (
        TimedGnssFixNed(
            t_fix=0.2,
            t_arrival=0.2,
            p_ned=np.array([0.35, -0.15, 0.0]),
        ),
    )

    nominals_delayed, Ps_delayed = gnss_delayed_aided(
        nominal0, P0, increments, gnss_fixes, Q, R_pos, nis_threshold, 1.0
    )
    nominals_ref, Ps_ref = _timely_gnss_loop(
        nominal0, P0, increments, gnss_fixes, Q, R_pos, nis_threshold
    )

    np.testing.assert_allclose(nominals_delayed, nominals_ref, atol=1e-12)
    np.testing.assert_allclose(Ps_delayed, Ps_ref, atol=1e-12)


def test_delayed_fix_matches_timely_equivalent_after_arrival():
    """After the delayed fix arrives, rewind-and-replay should catch back up.

    Before arrival, the delayed live state cannot know about the future fix, so
    it SHOULD differ from the timely reference. Once the fix arrives and the
    runner rewinds to `t_fix`, updates there, and replays forward, the live
    state from the arrival epoch onward should match the timely equivalent.
    """
    increments = _constant_velocity_increments(n_steps=6, dt=0.1)
    nominal0 = _nominal0()
    P0 = _P0()
    Q = _Q()
    R_pos = np.eye(3) * 0.5
    nis_threshold = 1e6
    timely_fix = TimedGnssFixNed(
        t_fix=0.2,
        t_arrival=0.2,
        p_ned=np.array([0.35, -0.15, 0.0]),
    )
    delayed_fix = TimedGnssFixNed(
        t_fix=0.2,
        t_arrival=0.5,
        p_ned=np.array([0.35, -0.15, 0.0]),
    )

    nominals_delayed, Ps_delayed = gnss_delayed_aided(
        nominal0, P0, increments, (delayed_fix,), Q, R_pos, nis_threshold, 1.0
    )
    nominals_timely, Ps_timely = _timely_gnss_loop(
        nominal0, P0, increments, (timely_fix,), Q, R_pos, nis_threshold
    )

    arrival_idx = 4   # dt = 0.1 s, so t = 0.5 s arrives at sample index 4
    assert not np.allclose(nominals_delayed[arrival_idx - 1], nominals_timely[arrival_idx - 1])
    np.testing.assert_allclose(
        nominals_delayed[arrival_idx:],
        nominals_timely[arrival_idx:],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        Ps_delayed[arrival_idx:],
        Ps_timely[arrival_idx:],
        atol=1e-12,
    )


def test_too_stale_fix_is_dropped():
    """A fix older than the retained rewind horizon should have no effect.

    This pins the practical real-time policy: if the fix is so old that the
    needed history has already been discarded, the filter must drop it rather
    than pretending it can still rewind correctly.
    """
    increments = _constant_velocity_increments(n_steps=6, dt=0.1)
    nominal0 = _nominal0()
    P0 = _P0()
    Q = _Q()
    R_pos = np.eye(3) * 0.5
    nis_threshold = 1e6
    stale_fix = TimedGnssFixNed(
        t_fix=0.2,
        t_arrival=0.6,
        p_ned=np.array([0.35, -0.15, 0.0]),
    )

    nominals_stale, Ps_stale = gnss_delayed_aided(
        nominal0, P0, increments, (stale_fix,), Q, R_pos, nis_threshold, 0.2
    )
    nominals_no_fix, Ps_no_fix = gnss_delayed_aided(
        nominal0, P0, increments, (), Q, R_pos, nis_threshold, 0.2
    )

    np.testing.assert_allclose(nominals_stale, nominals_no_fix, atol=1e-12)
    np.testing.assert_allclose(Ps_stale, Ps_no_fix, atol=1e-12)


def test_delayed_outlier_still_gets_gated_out():
    """Delayed replay must not bypass the existing R3 NIS gate.

    A GNSS fix that is wildly inconsistent should still reject cleanly even if
    it arrives late. If the gate works, the delayed-aided result matches the
    no-GNSS case exactly.
    """
    increments = _constant_velocity_increments(n_steps=6, dt=0.1)
    nominal0 = _nominal0()
    P0 = _P0()
    Q = _Q()
    R_pos = np.eye(3) * 0.25
    delayed_outlier = TimedGnssFixNed(
        t_fix=0.2,
        t_arrival=0.5,
        p_ned=np.array([50.0, -50.0, 20.0]),
    )

    nominals_outlier, Ps_outlier = gnss_delayed_aided(
        nominal0, P0, increments, (delayed_outlier,), Q, R_pos, 10.0, 1.0
    )
    nominals_no_fix, Ps_no_fix = gnss_delayed_aided(
        nominal0, P0, increments, (), Q, R_pos, 10.0, 1.0
    )

    np.testing.assert_allclose(nominals_outlier, nominals_no_fix, atol=1e-12)
    np.testing.assert_allclose(Ps_outlier, Ps_no_fix, atol=1e-12)
