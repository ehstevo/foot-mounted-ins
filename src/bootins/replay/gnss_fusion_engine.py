"""Maintain clock-aware IMU posterior history and fuse delayed GNSS evidence.

The engine advances a clock-augmented ESEKF through saved IMU/ZUPT events and
retains the resulting post-event posteriors. When a GNSS observation is
available, its GNSS-relative measurement time is mapped with the current clock
estimate, snapped to the first applicable retained posterior, and evaluated
there. An accepted update replaces that historical posterior and reconstructs
all later posteriors by replaying their saved events; a NIS-rejected evaluation
leaves the retained history unchanged.

The engine neither merges raw source arrivals nor decides when a future-mapped
fix has exhausted its deferral opportunity. The integrated replay owns those
causal policies and combines optional pre-update evidence with receiver
provenance. Public posterior access returns independent snapshots, while saved
events must remain stable so delayed reconstruction repeats the original IMU,
stance, and coast-covariance inputs.
"""
from dataclasses import dataclass

import numpy as np

from bootins.aiding.gnss_position_clock import (
    gnss_position_innovation_covariance_ned,
    gnss_position_innovation_ned,
    gnss_position_update_ned_gated,
    gnss_time_to_imu_time,
)
from bootins.estimation.clock_augmented_esekf import (
    ERR_DP,
    ERR_DV,
    NOM_V,
    NOM_TAU,
    esekf_predict_clock,
    esekf_update_clock,
)
from bootins.replay.clock_delayed_gnss import TimedGnssFixNed
from bootins.replay.gnss_fusion_models import (
    ClockImuEvent,
    ClockImuPosterior,
    DelayedGnssUpdate,
    DelayedGnssUpdateStatus,
    _TIMELINE_ATOL,
)


def _vel_h(nominal: np.ndarray) -> np.ndarray:
    """Return the velocity predicted by the zero-velocity measurement model."""
    return nominal[NOM_V]


_H_VEL_CLOCK = np.zeros((3, 16), dtype=float)
_H_VEL_CLOCK[:, ERR_DV] = np.eye(3)
_Z_ZUPT = np.zeros(3, dtype=float)


@dataclass(frozen=True)
class _GnssPreUpdateEvaluation:
    """Hold short-lived filter-side evidence from one evaluated GNSS fix.

    Attributes:
        tau_hat: Pre-update clock-offset estimate in s used to map the GNSS
            measurement epoch.
        innovation_ned: ``(3,)`` pre-update local-NED position innovation in m.
        innovation_covariance_ned: ``(3, 3)`` pre-update innovation covariance
            in m².
        prior_position_covariance_ned: ``(3, 3)`` local-NED position block of
            the pre-update error covariance in m².

    Notes:
        This private record is transient. The integrated runner transfers its
        values to :class:`GnssFusionDiagnostic`, which establishes the public
        snapshot ownership boundary.
    """

    tau_hat: float
    innovation_ned: np.ndarray
    innovation_covariance_ned: np.ndarray
    prior_position_covariance_ned: np.ndarray


def _apply_clock_imu_event(
    nominal: np.ndarray,
    P: np.ndarray,
    event: ClockImuEvent,
    Q: np.ndarray,
    R_zupt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one saved IMU event to a clock-aware posterior.

    Args:
        nominal: Clock-augmented nominal state before the event.
        P: Clock-augmented error-state covariance before the event.
        event: Saved IMU increment, causal stance flag, and optional added coast
            covariance.
        Q: Usual per-event process covariance.
        R_zupt: ``(3, 3)`` zero-velocity pseudo-measurement covariance in
            (m/s)².

    Returns:
        Nominal state and covariance after prediction and, for a true stance
        flag, the zero-velocity correction.

    Notes:
        The event predicts with ``Q`` plus any event-specific coast covariance.
        Reusing this transition for forward processing and historical replay
        preserves the recorded ``predict -> optional ZUPT`` sequence exactly.
    """
    dtheta, dv, dt = event.increment

    effective_Q = Q
    if event.additional_process_covariance is not None:
        effective_Q = Q + event.additional_process_covariance
    nominal, P = esekf_predict_clock(nominal, P, dtheta, dv, dt, effective_Q)

    if event.stance_flag:
        nominal, P = esekf_update_clock(
            nominal,
            P,
            _Z_ZUPT,
            _vel_h,
            _H_VEL_CLOCK,
            R_zupt,
        )

    return nominal, P


@dataclass(frozen=True)
class _ClockImuHistoryEntry:
    """Retain one post-event posterior for delayed historical reconstruction.

    Attributes:
        t_i: Post-event local IMU timeline epoch in s.
        event: Stable saved IMU/ZUPT event that reconstructs this row from the
            preceding posterior after an earlier accepted GNSS update.
        nominal: Filter-owned clock-augmented nominal posterior at ``t_i``.
        P: Filter-owned clock-augmented error-state covariance at ``t_i``.

    Notes:
        Accepted GNSS fusion replaces whole entries with reconstructed arrays;
        it does not expose or mutate previously returned public snapshots.
    """

    t_i: float
    event: ClockImuEvent
    nominal: np.ndarray
    P: np.ndarray


class ClockAwareGnssIns:
    """Own a clock-aware INS posterior and replayable IMU/ZUPT history.

    Attributes:
        _nominal: Current filter-owned 17-state clock-augmented nominal
            posterior.
        _P: Current filter-owned ``(16, 16)`` error-state posterior covariance.
        _t_i: Current post-event local IMU timeline epoch in s.
        _Q: Owned per-event ``(16, 16)`` process covariance.
        _R_zupt: Owned ``(3, 3)`` zero-velocity covariance in (m/s)².
        _history: Filter-owned post-event entries in strictly increasing IMU-time
            order. These are replaced during accepted delayed replay.

    Notes:
        The initial prior at ``t_i = 0`` is current state but is not stored in
        ``_history``. Public properties copy current arrays and construct
        independent immutable posterior snapshots.
    """

    def __init__(
        self,
        nominal0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R_zupt: np.ndarray,
    ) -> None:
        """Initialize a post-initialization clock-aware filter at local time zero.

        Args:
            nominal0: Initial 17-state nominal posterior at ``t_i = 0``.
        P0: Initial ``(16, 16)`` error-state covariance at ``t_i = 0``.
            Q: Per-event ``(16, 16)`` process covariance.
            R_zupt: ``(3, 3)`` zero-velocity pseudo-measurement covariance in
                (m/s)².
        """
        self._nominal = np.array(nominal0, dtype=float, copy=True)
        self._P = np.array(P0, dtype=float, copy=True)
        self._t_i = 0.0
        self._Q = np.array(Q, dtype=float, copy=True)
        self._R_zupt = np.array(R_zupt, dtype=float, copy=True)
        self._history: list[_ClockImuHistoryEntry] = []

    @property
    def nominal(self) -> np.ndarray:
        """Return an independent copy of the current 17-state posterior."""
        return self._nominal.copy()

    @property
    def P(self) -> np.ndarray:
        """Return an independent copy of the current ``(16, 16)`` covariance."""
        return self._P.copy()

    @property
    def t_i(self) -> float:
        """Return the current post-event local IMU timeline epoch in s."""
        return self._t_i

    @property
    def posteriors(self) -> tuple[ClockImuPosterior, ...]:
        """Return immutable snapshots of the current reconstructed history.

        Returns:
            One :class:`ClockImuPosterior` per advanced IMU event, in increasing
            local IMU time. The initial prior at time zero is not included.

        Notes:
            An accepted delayed GNSS update replaces historical entries and
            reconstructs later rows. Every access therefore reflects the latest
            corrected history without exposing mutable filter-owned arrays or
            saved :class:`ClockImuEvent` instances.
        """
        return tuple(
            ClockImuPosterior(
                t_i=entry.t_i,
                nominal=entry.nominal,
                P=entry.P,
            )
            for entry in self._history
        )

    def _find_update_index(self, t_fix_i_hat: float) -> int | None:
        """Return the first retained posterior applicable to a mapped epoch.

        A fixed time tolerance avoids placing an otherwise equal epoch on the
        wrong discrete posterior boundary after cancellation of large GNSS times
        and clock offsets.
        """
        for index, entry in enumerate(self._history):
            if entry.t_i + _TIMELINE_ATOL >= t_fix_i_hat:
                return index

        return None


    def advance(self, event: ClockImuEvent) -> None:
        """Advance through one saved IMU/ZUPT event and retain its posterior.

        Args:
            event: Stable real or synthetic IMU event with its causal stance
                flag and optional additional coast covariance.

        Notes:
            The stored history entry is aligned to the posterior after the event
            at the prior local IMU time plus its duration. It uses the same pure
            transition later used for delayed reconstruction.
        """
        nominal_next, P_next = _apply_clock_imu_event(
            self._nominal,
            self._P,
            event,
            self._Q,
            self._R_zupt,
        )

        t_i_next = self._t_i + event.increment[2]

        entry = _ClockImuHistoryEntry(
            t_i=t_i_next,
            event=event,
            nominal=nominal_next,
            P=P_next,
        )

        self._nominal = nominal_next
        self._P = P_next
        self._t_i = t_i_next
        self._history.append(entry)

    def fuse_delayed_gnss(
        self,
        fix: TimedGnssFixNed,
        R_pos: np.ndarray,
        nis_threshold: float,
    ) -> DelayedGnssUpdate:
        """Attempt a delayed GNSS update and return its terminal outcome.

        Args:
            fix: Scheduled local-NED GNSS position observation with GNSS
                measurement time and IMU-timeline availability epoch.
            R_pos: ``(3, 3)`` local-NED GNSS position covariance in m².
            nis_threshold: Scalar normalized-innovation-squared gate.

        Returns:
            Timing and gating outcome. An accepted result replaces the
            applicable posterior and reconstructs later history; a NIS-rejected
            or stale result is a no-op.

        Raises:
            RuntimeError: No IMU posterior exists or the supplied availability
                epoch is later than the current posterior.
            ValueError: The mapped GNSS measurement epoch is after current
                retained history.

        Notes:
            This narrow public method discards pre-update evidence. The
            integrated replay calls the private shared implementation when it
            must retain diagnostics with receiver provenance.
        """
        update, _ = self._fuse_delayed_gnss_with_evaluation(
            fix,
            R_pos,
            nis_threshold,
        )
        return update

    def _fuse_delayed_gnss_with_evaluation(
        self,
        fix: TimedGnssFixNed,
        R_pos: np.ndarray,
        nis_threshold: float,
    ) -> tuple[DelayedGnssUpdate, _GnssPreUpdateEvaluation | None]:
        """Attempt one delayed GNSS update and retain pre-update filter evidence.

        Args:
            fix: Scheduled local-NED GNSS position observation.
            R_pos: ``(3, 3)`` local-NED GNSS position covariance in m².
            nis_threshold: Scalar normalized-innovation-squared gate.

        Returns:
            ``(outcome, evaluation)``. ``evaluation`` is present only when a
            retained posterior was evaluated for an accepted or NIS-rejected
            update; it is ``None`` for a stale outcome.

        Raises:
            RuntimeError: No IMU posterior exists or the original scheduler
                delivery epoch lies after the current posterior.
            ValueError: The mapped GNSS epoch lies after retained history.

        Notes:
            ``fix.t_arrival_i`` can precede the current posterior after bounded
            runner-level deferral, but cannot be later. The pre-update current
            ``tau_hat`` maps the GNSS epoch. An accepted update replaces its
            historical entry and rebuilds every later posterior by applying each
            saved IMU event once; stale and NIS-rejected outcomes are no-ops.
        """
        if not self._history:
            raise RuntimeError(
                "Cannot fuse delayed GNSS before the first IMU posterior."
            )

        if fix.t_arrival_i > self._t_i + _TIMELINE_ATOL:
            raise RuntimeError(
                "Delayed GNSS fix arrival epoch cannot lie after the current "
                "IMU posterior."
            )

        tau_hat = self._nominal[NOM_TAU]

        t_fix_i_hat = gnss_time_to_imu_time(
            t_fix_g=fix.t_fix_g,
            tau_hat=tau_hat,
        )

        if t_fix_i_hat < self._history[0].t_i - _TIMELINE_ATOL:
            return (
                DelayedGnssUpdate(
                    status=DelayedGnssUpdateStatus.STALE,
                    t_fix_i_hat=t_fix_i_hat,
                    t_applied_i=None,
                    nis=None,
                ),
                None,
            )

        index = self._find_update_index(t_fix_i_hat)
        if index is None:
            raise ValueError(
                "Mapped GNSS epoch lies after the current IMU posterior."
            )
        entry = self._history[index]
        evaluation = _GnssPreUpdateEvaluation(
            tau_hat=tau_hat,
            innovation_ned=gnss_position_innovation_ned(
                entry.nominal,
                fix.p_ned,
            ),
            innovation_covariance_ned=gnss_position_innovation_covariance_ned(
                entry.nominal,
                entry.P,
                R_pos,
            ),
            prior_position_covariance_ned=entry.P[ERR_DP, ERR_DP],
        )

        nominal_replay, P_replay, nis, accepted = gnss_position_update_ned_gated(
            nominal=entry.nominal,
            P=entry.P,
            p_gnss_ned=fix.p_ned,
            R_pos=R_pos,
            nis_threshold=nis_threshold,
        )

        if not accepted:
            return (
                DelayedGnssUpdate(
                    status=DelayedGnssUpdateStatus.REJECTED,
                    t_fix_i_hat=t_fix_i_hat,
                    t_applied_i=entry.t_i,
                    nis=nis,
                ),
                evaluation,
            )

        self._history[index] = _ClockImuHistoryEntry(
            t_i=entry.t_i,
            event=entry.event,
            nominal=nominal_replay,
            P=P_replay,
        )
        for history_index in range(index + 1, len(self._history)):
            current_entry = self._history[history_index]
            nominal_replay, P_replay = _apply_clock_imu_event(
                nominal=nominal_replay,
                P=P_replay,
                event=current_entry.event,
                Q=self._Q,
                R_zupt=self._R_zupt,
            )
            self._history[history_index] = _ClockImuHistoryEntry(
                t_i=current_entry.t_i,
                event=current_entry.event,
                nominal=nominal_replay,
                P=P_replay,
            )

        final_entry = self._history[-1]
        self._nominal = final_entry.nominal
        self._P = final_entry.P
        return (
            DelayedGnssUpdate(
                status=DelayedGnssUpdateStatus.ACCEPTED,
                t_fix_i_hat=t_fix_i_hat,
                t_applied_i=entry.t_i,
                nis=nis,
            ),
            evaluation,
        )
