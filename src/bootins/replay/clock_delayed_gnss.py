"""Provide a clock-aware reference method for delayed GNSS fusion.

Each GNSS position is measured at ``t_fix_g`` on the GNSS-relative timeline
and becomes available at ``t_arrival_i`` on the IMU timeline. When available,
the current clock-offset estimate maps the measurement epoch as
``t_fix_i_hat = t_fix_g + tau_hat``. The method applies the position update to
the first retained posterior at or after that mapped epoch, then replays later
saved IMU increments with the clock-augmented ESEKF.

The first-at-or-after rule is a discrete snap rather than interpolation. This
reference implementation retains only output state histories; it does not
perform host-arrival scheduling, retain per-fix diagnostics, or expose update
statuses. The integrated fusion engine provides those evidence contracts.
"""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from bootins.estimation.clock_augmented_esekf import NOM_TAU, esekf_predict_clock
from bootins.aiding.gnss_position_clock import (
    gnss_position_update_ned_gated,
    gnss_time_to_imu_time,
)
from bootins.estimation.mechanization import Increment


@dataclass(frozen=True)
class TimedGnssFixNed:
    """Represent one delayed local-NED GNSS position observation.

    Attributes:
        t_fix_g: GNSS-relative receiver measurement epoch in s.
        t_arrival_i: IMU-timeline epoch in s when the observation is available
            to the filter.
        p_ned: ``(3,)`` receiver position in m in the fixed local NED frame.

    Notes:
        Measurement time and availability use distinct timelines. The filter's
        current clock estimate maps ``t_fix_g`` only when the fix is processed.
    """
    t_fix_g: float
    t_arrival_i: float
    p_ned: np.ndarray


@dataclass
class _HistoryEntry:
    """Retain one post-increment posterior for delayed replay.

    Attributes:
        sample_idx: Index of the increment producing this posterior and used to
            reconstruct it after an earlier accepted update.
        t: Post-increment IMU timeline epoch in s.
        nominal: Clock-augmented nominal posterior at ``t``.
        P: Clock-augmented error-state covariance at ``t``.
    """
    sample_idx: int
    t: float
    nominal: np.ndarray
    P: np.ndarray


def _validate_gnss_fixes(gnss_fixes: tuple[TimedGnssFixNed, ...]) -> None:
    """Validate GNSS availability ordering on the IMU timeline.

    Args:
        gnss_fixes: Materialized fixes expected in nondecreasing
            ``t_arrival_i`` order.

    Raises:
        ValueError: Availability epochs decrease.

    Notes:
        ``t_fix_g`` and ``t_arrival_i`` use different timelines, so their
        numerical ordering has no common-timeline interpretation.
    """
    last_arrival = float("-inf")
    for fix in gnss_fixes:
        if fix.t_arrival_i < last_arrival:
            raise ValueError(
                "gnss_stream must be sorted by nondecreasing t_arrival_i."
            )
        last_arrival = fix.t_arrival_i


def _trim_history(
    history: list[_HistoryEntry],
    t_now: float,
    buffer_horizon: float,
) -> None:
    """Drop buffered epochs older than the retained rewind horizon (s)."""
    while history and history[0].t < t_now - buffer_horizon:
        history.pop(0)


def _find_update_index(history: list[_HistoryEntry], t_fix: float) -> int | None:
    """Return the first retained posterior at or after a mapped epoch."""
    for idx, entry in enumerate(history):
        if t_fix <= entry.t:
            return idx
    return None


def _process_delayed_gnss_fix(
    history: list[_HistoryEntry],
    increments: tuple[Increment, ...],
    gnss_fix: TimedGnssFixNed,
    Q: np.ndarray,
    R_pos: np.ndarray,
    nis_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one available fix using the current clock estimate, then replay.

    Args:
        history: Retained clock-aware posteriors in IMU-time order.
        increments: Materialized IMU increments addressed by history entries.
        gnss_fix: Available GNSS-relative position observation.
        Q: Per-increment clock-augmented ESEKF process covariance.
        R_pos: ``(3, 3)`` local-NED GNSS position covariance in m².
        nis_threshold: Scalar NIS gate applied by the position update.

    Returns:
        Current clock-aware nominal state and covariance after an accepted
        update and replay, or unchanged values for a stale or NIS-rejected fix.

    Raises:
        ValueError: The mapped measurement epoch is later than the newest
            retained posterior.

    Notes:
        The current posterior's ``tau_hat`` maps ``t_fix_g`` before the update;
        the result then snaps to the first retained posterior at or after that
        epoch. A mapped epoch older than the history horizon cannot be applied.
    """
    tau_hat = history[-1].nominal[NOM_TAU]
    t_fix_i_hat = gnss_time_to_imu_time(gnss_fix.t_fix_g, tau_hat)
    if t_fix_i_hat < history[0].t:
        return history[-1].nominal, history[-1].P

    update_idx = _find_update_index(history, t_fix_i_hat)
    if update_idx is None:
        raise ValueError(
            f"GNSS fix at mapped t_fix_i_hat={t_fix_i_hat} lies after buffered current time "
            f"{history[-1].t}."
        )

    entry = history[update_idx]
    nominal_replay, P_replay, _nis, _accepted = gnss_position_update_ned_gated(
        entry.nominal, entry.P, gnss_fix.p_ned, R_pos, nis_threshold
    )
    history[update_idx] = _HistoryEntry(
        sample_idx=entry.sample_idx,
        t=entry.t,
        nominal=nominal_replay,
        P=P_replay,
    )

    for hist_idx in range(update_idx + 1, len(history)):
        sample_idx = history[hist_idx].sample_idx
        dtheta, dv, dt = increments[sample_idx]
        nominal_replay, P_replay = esekf_predict_clock(
            nominal_replay, P_replay, dtheta, dv, dt, Q
        )
        history[hist_idx] = _HistoryEntry(
            sample_idx=sample_idx,
            t=history[hist_idx].t,
            nominal=nominal_replay,
            P=P_replay,
        )

    return history[-1].nominal, history[-1].P


def gnss_delayed_aided(
    nominal0: np.ndarray,
    P0: np.ndarray,
    increments: Iterable[Increment],
    gnss_stream: Iterable[TimedGnssFixNed],
    Q: np.ndarray,
    R_pos: np.ndarray,
    nis_threshold: float,
    buffer_horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay delayed GNSS observations with a clock-augmented ESEKF.

    Args:
        nominal0: Clock-augmented nominal state before the first IMU increment.
        P0: Clock-augmented error-state covariance before the first increment.
        increments: IMU ``(dtheta_b, dv_b, dt)`` increments, materialized for
            indexed historical replay.
        gnss_stream: Delayed local-NED fixes in nondecreasing
            ``t_arrival_i`` order.
        Q: Per-increment clock-augmented ESEKF process covariance.
        R_pos: ``(3, 3)`` local-NED GNSS position covariance in m².
        nis_threshold: Scalar NIS gate applied by the position update.
        buffer_horizon: Positive retained IMU-history duration (s).

    Returns:
        ``(nominals, Ps)`` with one posterior row per IMU increment. Each row
        includes fixes available by that epoch after any necessary rewind and
        reconstruction.

    Raises:
        ValueError: The history horizon is nonpositive, availability ordering is
            invalid, or a mapped measurement epoch is after current history.

    Notes:
        This is a reference method, not the integrated host-arrival and
        diagnostic pipeline. Its output omits an initial prior row.
    """
    if buffer_horizon <= 0.0:
        raise ValueError("buffer_horizon must be positive.")

    increments = tuple(increments)
    gnss_fixes = tuple(gnss_stream)
    _validate_gnss_fixes(gnss_fixes)

    nominal = np.array(nominal0, dtype=float, copy=True)
    P = np.array(P0, dtype=float, copy=True)

    if not increments:
        empty_nominals = np.empty((0, nominal.shape[0]), dtype=float)
        empty_covariances = np.empty((0, *P.shape), dtype=float)
        return empty_nominals, empty_covariances

    nominals = np.empty((len(increments), nominal.shape[0]), dtype=float)
    Ps = np.empty((len(increments), *P.shape), dtype=float)
    history: list[_HistoryEntry] = []
    next_fix_idx = 0
    t_now = 0.0

    for k, increment in enumerate(increments):
        dtheta, dv, dt = increment
        nominal, P = esekf_predict_clock(nominal, P, dtheta, dv, dt, Q)
        t_now += dt

        history.append(_HistoryEntry(k, t_now, nominal, P))
        _trim_history(history, t_now, buffer_horizon)

        while next_fix_idx < len(gnss_fixes) and gnss_fixes[next_fix_idx].t_arrival_i <= t_now:
            nominal, P = _process_delayed_gnss_fix(
                history,
                increments,
                gnss_fixes[next_fix_idx],
                Q,
                R_pos,
                nis_threshold,
            )
            next_fix_idx += 1

        nominals[k] = nominal
        Ps[k] = P

    return nominals, Ps
