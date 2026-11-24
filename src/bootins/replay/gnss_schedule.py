"""Release host-arrived GNSS evidence at causal processed-IMU epochs.

The scheduler bridges receiver availability on a shared host-monotonic clock
to the discrete posteriors that an IMU navigation loop has already produced.
It assigns a delivered fix an IMU timeline epoch ``t_arrival_i``; this delivery
epoch is neither the receiver's GNSS measurement epoch nor the raw host
timestamp. A later clock-aware fusion stage maps measurement time and decides
whether a delayed update is applicable.

This module preserves receiver records unchanged. It does not transform LLH
coordinates, estimate clock offset, repair timing, or apply navigation updates.
"""
from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from bootins.sensors.gnss_observations import GnssFixLLH


def _require_nonnegative_finite_real(value: object, field: str) -> float:
    """Return one finite, nonnegative real value or raise a clear error."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    value_float = float(value)
    if value_float < 0.0:
        raise ValueError(f"{field} must be nonnegative.")
    return value_float


@dataclass(frozen=True)
class ImuArrivalEpoch:
    """Identify a completed IMU posterior and its host availability time.

    Attributes:
        t_i: Nonnegative IMU integration time in s at a completed posterior.
            It is local to the post-initialization replay timeline.
        host_arrival_monotonic: Nonnegative host-monotonic timestamp in s at
            which the source frame producing that posterior was available.

    Notes:
        ``t_i`` supports filter timing; ``host_arrival_monotonic`` supports
        causal availability. Neither is a GNSS receiver measurement epoch.
    """

    t_i: float
    host_arrival_monotonic: float

    def __post_init__(self) -> None:
        """Reject timestamps that cannot define a causal IMU epoch."""
        _require_nonnegative_finite_real(self.t_i, "t_i")
        _require_nonnegative_finite_real(
            self.host_arrival_monotonic,
            "host_arrival_monotonic",
        )


@dataclass(frozen=True)
class ScheduledGnssFix:
    """Associate one original GNSS fix with its causal delivery posterior.

    Attributes:
        fix: Unmodified receiver observation, including its GNSS measurement
            epoch and host-arrival evidence.
        t_arrival_i: IMU integration time in s of the first completed posterior
            at which this fix became eligible for delivery.
    """

    fix: GnssFixLLH
    t_arrival_i: float


@dataclass(frozen=True)
class GnssScheduleSummary:
    """Count fixes outside the usable causal IMU processing interval.

    Attributes:
        dropped_before_imu_count: Fixes with host arrival strictly before the
            first processed IMU posterior.
        dropped_after_imu_count: Pending fixes remaining when processing ends.

    Notes:
        These counts describe availability, not receiver validity, GNSS
        measurement quality, or the outcome of a filter evaluation.
    """

    dropped_before_imu_count: int
    dropped_after_imu_count: int


class GnssArrivalScheduler:
    """Queue ordered GNSS fixes and release each at its first usable posterior.

    The scheduler is single-use. Submitted records must be ordered by host
    availability; processed IMU epochs must strictly increase in both host and
    local integration time after the first epoch.

    Attributes:
        _pending_fixes: FIFO receiver evidence awaiting a causal IMU posterior.
        _last_submitted_fix_host_arrival: Host timestamp used to enforce GNSS
            submission order, or ``None`` before the first fix.
        _last_imu_epoch: Last processed posterior availability record, or
            ``None`` before the IMU timeline starts.
        _dropped_before_imu_count: Number of pre-timeline unavailable fixes.
        _dropped_after_imu_count: Number of pending fixes at finalization.
        _finished: Whether finalization has closed the scheduler.
    """

    def __init__(self) -> None:
        """Initialize an empty scheduler for one causal IMU timeline."""
        self._pending_fixes: deque[GnssFixLLH] = deque()
        self._last_submitted_fix_host_arrival: float | None = None
        self._last_imu_epoch: ImuArrivalEpoch | None = None
        self._dropped_before_imu_count = 0
        self._dropped_after_imu_count = 0
        self._finished = False

    def submit_fix(self, fix: GnssFixLLH) -> None:
        """Queue one receiver fix in nondecreasing host-arrival order.

        Args:
            fix: Original receiver observation. It remains unchanged, including
                validity evidence and its GNSS measurement epoch.

        Raises:
            RuntimeError: The scheduler has already finished.
            ValueError: The fix's host-arrival timestamp moves backward from a
                previously submitted fix.

        Notes:
            The fix becomes eligible only when a later processed IMU epoch has
            host arrival at least as recent as the fix. No GNSS-time mapping or
            receiver-validity decision occurs here.
        """
        if self._finished:
            raise RuntimeError("GNSS scheduler has already finished.")

        previous_host_arrival = self._last_submitted_fix_host_arrival
        if (
            previous_host_arrival is not None
            and fix.host_arrival_monotonic < previous_host_arrival
        ):
            raise ValueError(
                "GNSS fixes must be submitted in nondecreasing "
                "host-arrival order."
            )

        self._pending_fixes.append(fix)
        self._last_submitted_fix_host_arrival = fix.host_arrival_monotonic

    def release_at_imu_epoch(
        self,
        epoch: ImuArrivalEpoch,
    ) -> tuple[ScheduledGnssFix, ...]:
        """Release fixes first available by one completed IMU posterior.

        Args:
            epoch: Completed posterior's local IMU time and source host-arrival
                timestamp.

        Returns:
            FIFO scheduled fixes with host arrival no later than ``epoch``. Each
            receives ``epoch.t_i`` as its causal delivery time.

        Raises:
            RuntimeError: The scheduler has already finished.
            ValueError: A later IMU epoch does not strictly increase in host
                arrival or local integration time.

        Notes:
            The first call begins the usable IMU timeline. Fixes strictly before
            its host arrival are counted as unavailable and discarded; equality
            remains eligible. All subsequently eligible pending fixes are
            released together, preserving their submitted order.
        """
        if self._finished:
            raise RuntimeError("GNSS scheduler has already finished.")

        previous_epoch = self._last_imu_epoch
        if previous_epoch is None:
            while (
                self._pending_fixes
                and self._pending_fixes[0].host_arrival_monotonic
                < epoch.host_arrival_monotonic
            ):
                self._pending_fixes.popleft()
                self._dropped_before_imu_count += 1
        else:
            if epoch.host_arrival_monotonic <= previous_epoch.host_arrival_monotonic:
                raise ValueError(
                    "IMU epoch host arrivals must be strictly increasing."
                )
            if epoch.t_i <= previous_epoch.t_i:
                raise ValueError("IMU epoch t_i values must be strictly increasing.")

        released_fixes: list[ScheduledGnssFix] = []
        while (
            self._pending_fixes
            and self._pending_fixes[0].host_arrival_monotonic
            <= epoch.host_arrival_monotonic
        ):
            released_fixes.append(
                ScheduledGnssFix(
                    fix=self._pending_fixes.popleft(),
                    t_arrival_i=epoch.t_i,
                )
            )

        self._last_imu_epoch = epoch
        return tuple(released_fixes)

    def finish(self) -> GnssScheduleSummary:
        """Close the scheduler and count fixes after the IMU timeline.

        Returns:
            Counts of fixes unavailable before the first and after the final
            processed IMU posterior.

        Raises:
            RuntimeError: The scheduler is already finished or has not received
                any IMU epoch.
        """
        if self._finished:
            raise RuntimeError("GNSS scheduler has already finished.")
        if self._last_imu_epoch is None:
            raise RuntimeError("GNSS scheduler cannot finish before an IMU epoch.")

        self._dropped_after_imu_count = len(self._pending_fixes)
        self._pending_fixes.clear()
        self._finished = True
        return GnssScheduleSummary(
            dropped_before_imu_count=self._dropped_before_imu_count,
            dropped_after_imu_count=self._dropped_after_imu_count,
        )
