"""Provide a reference delayed-GNSS rewind-and-replay implementation.

This method assumes each GNSS measurement epoch already shares the IMU
propagation timeline. It propagates IMU increments forward, retains recent
posteriors, and, once a GNSS fix becomes available, applies its position update
to the first retained posterior at or after the measurement epoch. It then
replays the saved later IMU increments to reconstruct the current posterior.

The first-at-or-after rule is a discrete time snap, not interpolation. Fixes
older than the retained history are silently left unapplied, and a measurement
epoch after the current posterior is refused. This compact reference method
does not estimate clock offset, schedule host arrivals, retain per-fix
diagnostics, or expose historical evidence; those responsibilities belong to
the clock-aware replay components.
"""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from bootins.estimation.esekf import esekf_predict
from bootins.aiding.gnss_position import gnss_position_update_ned_gated
from bootins.estimation.mechanization import Increment


@dataclass(frozen=True)
class TimedGnssFixNed:
    """Represent a GNSS position observation on a common IMU/GNSS timeline.

    Attributes:
        t_fix: GNSS measurement epoch in s on the same timeline as IMU
            propagation.
        t_arrival: Later availability epoch in s on that same timeline.
        p_ned: ``(3,)`` GNSS position in m in the INS local NED frame.

    Notes:
        This reference model assumes the common timeline rather than estimating
        an IMU-to-GNSS clock relation.
    """
    t_fix: float
    t_arrival: float
    p_ned: np.ndarray


@dataclass
class _HistoryEntry:
    """Retain one IMU posterior needed for delayed measurement replay.

    Attributes:
        sample_idx: Index of the increment whose predict step produced this
            posterior. Replaying entry ``j`` applies
            ``increments[history[j].sample_idx]`` to the corrected previous
            posterior.
        t: Post-increment IMU epoch in s.
        nominal: Nominal posterior at ``t``.
        P: Error-state posterior covariance at ``t``.
    """
    sample_idx: int
    t: float
    nominal: np.ndarray
    P: np.ndarray


def _validate_gnss_fixes(gnss_fixes: tuple[TimedGnssFixNed, ...]) -> None:
    """Validate the common-timeline GNSS availability assumptions.

    Args:
        gnss_fixes: Materialized fixes expected in nondecreasing availability
            order on the shared IMU/GNSS timeline.

    Raises:
        ValueError: A measurement epoch is later than its availability epoch or
            availability timestamps decrease.
    """
    last_arrival = float("-inf")
    for fix in gnss_fixes:
        if fix.t_fix > fix.t_arrival:
            raise ValueError(
                f"GNSS fix has t_fix={fix.t_fix} later than t_arrival={fix.t_arrival}."
            )
        if fix.t_arrival < last_arrival:
            raise ValueError("gnss_stream must be sorted by nondecreasing t_arrival.")
        last_arrival = fix.t_arrival


def _trim_history(
    history: list[_HistoryEntry],
    t_now: float,
    buffer_horizon: float,
) -> None:
    """Drop buffered posteriors older than the permitted rewind horizon (s)."""
    while history and history[0].t < t_now - buffer_horizon:
        history.pop(0)


def _find_update_index(history: list[_HistoryEntry], t_fix: float) -> int | None:
    """Return the first retained posterior at or after a measurement epoch."""
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
    """Apply one available delayed fix and replay later IMU propagation.

    Args:
        history: Retained posteriors ordered by increasing IMU time.
        increments: Materialized IMU increments addressed by each history entry.
        gnss_fix: Available GNSS position observation on the common timeline.
        Q: Per-increment ESEKF process covariance.
        R_pos: ``(3, 3)`` local-NED GNSS position covariance in m².
        nis_threshold: Scalar gate applied by the GNSS position update.

    Returns:
        Current nominal state and error-state covariance after the accepted
        update and replay, or unchanged values for a stale or NIS-rejected fix.

    Raises:
        ValueError: The measurement epoch is after the latest retained
            posterior.

    Notes:
        The update snaps to the first retained posterior at or after ``t_fix``.
        A fix older than the history horizon cannot be applied and leaves the
        current state unchanged. Acceptance replaces the affected historical
        posterior, then rebuilds later entries from their original increments.
    """
    if gnss_fix.t_fix < history[0].t:
        return history[-1].nominal, history[-1].P

    update_idx = _find_update_index(history, gnss_fix.t_fix)
    if update_idx is None:
        raise ValueError(
            f"GNSS fix at t_fix={gnss_fix.t_fix} lies after buffered current time "
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
        nominal_replay, P_replay = esekf_predict(
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
    """Replay common-timeline IMU and delayed GNSS position observations.

    Args:
        nominal0: Nominal state before the first IMU increment.
        P0: Error-state covariance before the first IMU increment.
        increments: IMU ``(dtheta_b, dv_b, dt)`` increments. The method
            materializes them to replay retained intervals by index.
        gnss_stream: Delayed local-NED fixes in nondecreasing ``t_arrival``.
        Q: Per-increment ESEKF process covariance.
        R_pos: ``(3, 3)`` local-NED GNSS position covariance in m².
        nis_threshold: Scalar NIS gate used by the position update.
        buffer_horizon: Positive retained IMU-history duration (s).

    Returns:
        ``(nominals, Ps)`` with one posterior row per IMU increment. Each row
        includes delayed fixes available by that IMU epoch after any necessary
        rewind and replay.

    Raises:
        ValueError: The history horizon is nonpositive, GNSS ordering violates
            the common-timeline contract, or an available fix maps after the
            current retained posterior.

    Notes:
        The returned arrays omit an initial prior row. This method is a compact
        reference for a known common timeline, not a host-arrival scheduler or
        a diagnostic evidence archive.
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
        nominal, P = esekf_predict(nominal, P, dtheta, dv, dt, Q)
        t_now += dt

        history.append(_HistoryEntry(k, t_now, nominal, P))
        _trim_history(history, t_now, buffer_horizon)

        while next_fix_idx < len(gnss_fixes) and gnss_fixes[next_fix_idx].t_arrival <= t_now:
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
