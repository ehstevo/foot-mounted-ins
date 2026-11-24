"""Construct bounded synthetic IMU coasts from observed sequence-gap evidence.

The recovery processor receives factual :class:`TrackedFrame` objects after
decoding and sequence tracking. It validates and owns real increments, then
may create a plan for exactly the policy-permitted number of missing
notifications. The plan repeats held motion rates from the final prior real
increment; it is reconstructed propagation, not an exact recovery of missing
measurements.

This module neither decodes raw transport data nor executes a filter, detects
stance, schedules GNSS, or chooses a runner's uncertainty model. Runners must
record and execute returned coast plans explicitly, reset stateful stance
detectors across the discontinuity, and preserve the observed gap provenance.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
import numpy as np

from bootins.bmu.sequence import GapEvent, SEQUENCE_MODULUS
from bootins.bmu.quality import QualityFlags, assess
from bootins.bmu.stream import TrackedFrame
from bootins.estimation.mechanization import Increment


def _require_positive_int(value: object, field_name: str) -> int:
    """Return one positive integer, rejecting booleans and non-integral values."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _require_frame_index(value: object, field_name: str) -> int:
    """Return one nonnegative raw-frame index."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer.")
    return int(value)


def _validate_gap_event(gap_event: GapEvent) -> None:
    """Reject malformed loss evidence that could not come from ``SequenceTracker``."""
    if not isinstance(gap_event, GapEvent):
        raise ValueError("gap_event must be a GapEvent.")

    previous = gap_event.previous_seq_num
    current = gap_event.current_seq_num
    missing = gap_event.missing_message_count
    if (
        isinstance(previous, bool)
        or not isinstance(previous, Integral)
        or not 0 <= previous < SEQUENCE_MODULUS
        or isinstance(current, bool)
        or not isinstance(current, Integral)
        or not 0 <= current < SEQUENCE_MODULUS
    ):
        raise ValueError("gap_event sequence numbers must be valid u16 values.")

    missing = _require_positive_int(missing, "gap_event.missing_message_count")
    expected_current = (int(previous) + missing + 1) % SEQUENCE_MODULUS
    if int(current) != expected_current:
        raise ValueError("gap_event sequence numbers disagree with its missing count.")


def _owned_increment(increment: object) -> Increment:
    """Return one finite, positive-duration, read-only increment copy."""
    try:
        dtheta, dv, dt = increment
    except (TypeError, ValueError) as error:
        raise ValueError("increment must contain dtheta, dv, and dt.") from error

    try:
        dtheta_owned = np.array(dtheta, dtype=float, copy=True)
        dv_owned = np.array(dv, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("increment vectors must be numeric arrays.") from error

    if dtheta_owned.shape != (3,) or dv_owned.shape != (3,):
        raise ValueError("increment dtheta and dv must each have shape (3,).")
    if not np.all(np.isfinite(dtheta_owned)) or not np.all(np.isfinite(dv_owned)):
        raise ValueError("increment dtheta and dv must be finite.")
    if (
        isinstance(dt, bool)
        or not isinstance(dt, Real)
        or not isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("increment dt must be a positive finite real.")

    dtheta_owned.setflags(write=False)
    dv_owned.setflags(write=False)
    return dtheta_owned, dv_owned, float(dt)


def _quality_flag_names(flags: QualityFlags) -> str:
    """Return stable names for all structural concerns in one measurement."""
    return ", ".join(
        flag.name
        for flag in QualityFlags
        if flag is not QualityFlags.OK and flags & flag
    )


@dataclass(frozen=True)
class ImuGapRecoveryPolicy:
    """Bound notification loss eligible for causal synthetic propagation.

    Attributes:
        max_missing_notifications: Positive maximum number of consecutive BMU
            notifications that one coast plan may represent. The default ``1``
            permits one missing notification only.
    """

    max_missing_notifications: int = 1

    def __post_init__(self) -> None:
        """Validate the bounded-recovery limit."""
        object.__setattr__(
            self,
            "max_missing_notifications",
            _require_positive_int(
                self.max_missing_notifications,
                "max_missing_notifications",
            ),
        )


@dataclass(frozen=True)
class ImuCoastPlan:
    """Describe one explicit synthetic coast between real IMU frames.

    Attributes:
        gap_event: Observed sequence-tracker evidence identifying the missing
            notification count and adjacent sequence numbers.
        increments: Immutable synthetic ``(dtheta_b, dv_b, dt)`` increments.
            Each is constructed from the final increment of the prior real
            frame; none is a received measurement.
        reference_frame_index: Raw-frame index supplying the held increment.
        revealing_frame_index: Raw-frame index whose sequence number revealed
            the preceding loss.
        duration: Total synthetic propagation duration in s.

    Notes:
        The immutable increments and frame indices preserve the distinction
        between actual recorded evidence and a bounded reconstruction used by a
        navigation runner.
    """

    gap_event: GapEvent
    increments: tuple[Increment, ...]
    reference_frame_index: int
    revealing_frame_index: int
    duration: float


@dataclass(frozen=True)
class ImuFrameEvent:
    """Combine factual frame provenance with owned real IMU increments.

    Attributes:
        tracked_frame: Received decoded frame and attached sequence-gap
            evidence. It retains the original raw-frame provenance.
        increments: Immutable owned real decoded ``(dtheta_b, dv_b, dt)``
            increments in measurement order. This tuple never includes a
            synthetic coast increment.

    Notes:
        The recovery processor creates this event only after structural-quality
        assessment. Copying at this boundary prevents later caller mutation of
        the retained increment evidence.
    """

    tracked_frame: TrackedFrame
    increments: tuple[Increment, ...]


def _prepare_real_frame_event(tracked_frame: TrackedFrame) -> ImuFrameEvent:
    """Validate and own real increments while retaining gap provenance.

    Args:
        tracked_frame: One decoded tracked frame from the source stream.

    Returns:
        A factual frame event containing owned immutable real increments.

    Raises:
        ValueError: The object is not a tracked frame or any decoded measurement
            has a structural-quality concern.

    Notes:
        A preceding gap is intentionally retained rather than rejected here,
        because it is evidence required by the bounded coast planner.
    """
    if not isinstance(tracked_frame, TrackedFrame):
        raise ValueError("tracked_frame must be a TrackedFrame.")

    raw_frame = tracked_frame.decoded_frame.raw_frame
    increments: list[Increment] = []
    for measurement_index, measurement in enumerate(
        tracked_frame.decoded_frame.decoded_message.measurements
    ):
        flags = assess(measurement)
        if flags != QualityFlags.OK:
            raise ValueError(
                f"Structural quality flags at frame {raw_frame.frame_index}, "
                f"measurement {measurement_index}: {_quality_flag_names(flags)}."
            )
        increments.append(
            _owned_increment((measurement.dtheta, measurement.dv, measurement.dt))
        )

    return ImuFrameEvent(
        tracked_frame=tracked_frame,
        increments=tuple(increments),
    )


class ImuGapRecoveryProcessor:
    """Maintain causal real-frame context for bounded gap recovery.

    Attributes:
        _recovery_policy: Immutable limit controlling recoverable notification
            loss.
        _previous_real_frame: Most recent structurally valid factual frame, or
            ``None`` before the first frame. Its owned final increment supplies
            any later held-rate coast.
    """

    def __init__(self, recovery_policy: ImuGapRecoveryPolicy) -> None:
        """Initialize the processor before the first factual frame.

        Args:
            recovery_policy: Positive bounded-loss policy for coast planning.

        Raises:
            ValueError: ``recovery_policy`` is not an
                :class:`ImuGapRecoveryPolicy`.
        """
        if not isinstance(recovery_policy, ImuGapRecoveryPolicy):
            raise ValueError("recovery_policy must be an ImuGapRecoveryPolicy.")
        self._recovery_policy = recovery_policy
        self._previous_real_frame: ImuFrameEvent | None = None

    def update(
        self,
        tracked_frame: TrackedFrame,
    ) -> tuple[ImuCoastPlan | ImuFrameEvent, ...]:
        """Process one factual frame and plan any immediately preceding coast.

        Args:
            tracked_frame: Decoded source frame with observed sequence-gap
                evidence, if any.

        Returns:
            A one-item tuple containing the factual frame when contiguous, or a
            two-item tuple ``(coast_plan, factual_frame)`` when bounded loss is
            recoverable. The factual frame always follows its coast plan.

        Raises:
            ValueError: The frame is structurally invalid, a gap has no prior
                real frame, or its evidence cannot support bounded recovery.

        Notes:
            The current frame is validated before planning but does not provide
            synthetic motion values. It becomes the held reference only after a
            successful plan, preserving causal use of prior real evidence.
        """
        current_frame = _prepare_real_frame_event(tracked_frame)
        gap_event = tracked_frame.gap_before
        if gap_event is None:
            self._previous_real_frame = current_frame
            return (current_frame,)

        previous_frame = self._previous_real_frame
        if previous_frame is None:
            raise ValueError(
                "Cannot recover a BMU sequence gap without a prior real frame."
            )

        coast_plan = plan_short_imu_coast(
            gap_event=gap_event,
            previous_increments=previous_frame.increments,
            revealing_frame_increment_count=len(current_frame.increments),
            reference_frame_index=(
                previous_frame.tracked_frame.decoded_frame.raw_frame.frame_index
            ),
            revealing_frame_index=(
                current_frame.tracked_frame.decoded_frame.raw_frame.frame_index
            ),
            recovery_policy=self._recovery_policy,
        )
        self._previous_real_frame = current_frame
        return coast_plan, current_frame


def plan_short_imu_coast(
    gap_event: GapEvent,
    previous_increments: tuple[Increment, ...],
    revealing_frame_increment_count: int,
    reference_frame_index: int,
    revealing_frame_index: int,
    recovery_policy: ImuGapRecoveryPolicy,
) -> ImuCoastPlan:
    """Build one bounded coast from the final increment of a prior real frame.

    Args:
        gap_event: Observed sequence-gap evidence from the revealing frame.
        previous_increments: Owned real increments from the preceding frame.
        revealing_frame_increment_count: Number of real increments in the frame
            that revealed the gap.
        reference_frame_index: Nonnegative raw-frame index of the prior frame.
        revealing_frame_index: Later nonnegative raw-frame index revealing loss.
        recovery_policy: Maximum recoverable notification-loss policy.

    Returns:
        An immutable plan whose synthetic increments repeat one held prior rate.

    Raises:
        ValueError: The policy, gap evidence, frame provenance, packet shape, or
            preceding real increments cannot support bounded recovery.

    Notes:
        The first post-gap frame establishes only packet shape; its increment
        values never influence synthetic motion. The planner assumes a missing
        packet has the same increment count as both real bracketing frames and
        repeats the final prior real increment for every synthetic step.
    """
    if not isinstance(recovery_policy, ImuGapRecoveryPolicy):
        raise ValueError("recovery_policy must be an ImuGapRecoveryPolicy.")
    _validate_gap_event(gap_event)

    if gap_event.missing_message_count > recovery_policy.max_missing_notifications:
        raise ValueError(
            "BMU sequence gap exceeds the bounded recovery policy: "
            f"{gap_event.missing_message_count} missing notification(s), "
            f"limit {recovery_policy.max_missing_notifications}."
        )
    if not isinstance(previous_increments, tuple) or not previous_increments:
        raise ValueError("previous_increments must be a nonempty tuple.")

    revealing_frame_increment_count = _require_positive_int(
        revealing_frame_increment_count,
        "revealing_frame_increment_count",
    )
    if len(previous_increments) != revealing_frame_increment_count:
        raise ValueError(
            "Cannot recover a gap with mismatched bracketing packet sizes: "
            f"previous {len(previous_increments)}, revealing "
            f"{revealing_frame_increment_count}."
        )

    reference_frame_index = _require_frame_index(
        reference_frame_index,
        "reference_frame_index",
    )
    revealing_frame_index = _require_frame_index(
        revealing_frame_index,
        "revealing_frame_index",
    )
    if revealing_frame_index <= reference_frame_index:
        raise ValueError(
            "revealing_frame_index must be greater than reference_frame_index."
        )

    held_increment = _owned_increment(previous_increments[-1])
    synthetic_count = (
        gap_event.missing_message_count * revealing_frame_increment_count
    )
    synthetic_increments = tuple(held_increment for _ in range(synthetic_count))

    return ImuCoastPlan(
        gap_event=gap_event,
        increments=synthetic_increments,
        reference_frame_index=reference_frame_index,
        revealing_frame_index=revealing_frame_index,
        duration=float(sum(increment[2] for increment in synthetic_increments)),
    )
