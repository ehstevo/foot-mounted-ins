"""Establish the strict IMU-integrity boundary for offline replay and analysis.

This module converts source-ordered raw BMU frames into decoded inertial
increments only after sequence tracking and structural measurement assessment.
Canonical replay uses :func:`collect_clean_increments` to reject an entire
capture with confirmed packet loss or malformed measurements. It neither
repairs loss nor infers missing motion.

:func:`select_longest_clean_increment_segment` is deliberately separate: it
partitions observed evidence at gaps and structural concerns for opt-in offline
analysis. Selecting one clean segment does not relax the stricter canonical
navigation-input policy and does not make excluded data trustworthy.
"""
from collections.abc import Iterable
from dataclasses import dataclass

from bootins.bmu.quality import QualityFlags, assess
from bootins.bmu.stream import iter_tracked_frames, TrackedFrame
from bootins.bmu.transport import RawFrame
from bootins.estimation.mechanization import Increment


@dataclass(frozen=True)
class CleanIncrementSegment:
    """Represent one contiguous structurally clean increment run.

    Attributes:
        increments: Immutable real decoded ``(dtheta_b, dv_b, dt)`` increments
            from the selected run. No synthetic reconstruction is included.
        frame_start_index: Inclusive raw-frame index of the first increment.
        frame_end_index: Inclusive raw-frame index of the final increment.
        measurement_start_index: Inclusive measurement index within
            ``frame_start_index``.
        measurement_end_index: Inclusive measurement index within
            ``frame_end_index``.

    Notes:
        A run begins after a confirmed gap or structural-quality concern and
        ends before the next such concern. The fields retain provenance for an
        explicitly selected analysis segment; they do not certify the complete
        original capture.
    """

    increments: tuple[Increment, ...]
    frame_start_index: int
    frame_end_index: int
    measurement_start_index: int
    measurement_end_index: int


def _quality_flag_names(flags: QualityFlags) -> str:
    """Return stable, readable names for every active non-OK quality flag."""
    return ", ".join(
        flag.name
        for flag in QualityFlags
        if flag is not QualityFlags.OK and flags & flag
    )


def validate_clean_tracked_frame(tracked_frame: TrackedFrame) -> None:
    """Validate one tracked frame for strict canonical replay.

    Args:
        tracked_frame: Decoded source frame with sequence-gap provenance.

    Raises:
        ValueError: A confirmed notification gap precedes the frame or any
            decoded measurement has a structural-quality concern.

    Notes:
        Error messages identify the observed raw-frame and measurement indices.
        Decoder and sequence-tracker failures occur before this function and
        intentionally propagate without translation.
    """
    raw_frame = tracked_frame.decoded_frame.raw_frame
    gap_event = tracked_frame.gap_before
    if gap_event is not None:
        raise ValueError(
            f"BMU sequence gap before frame {raw_frame.frame_index}: "
            f"{gap_event.previous_seq_num} -> {gap_event.current_seq_num}; "
            f"{gap_event.missing_message_count} notification(s) missing."
        )

    for measurement_index, measurement in enumerate(
        tracked_frame.decoded_frame.decoded_message.measurements
    ):
        flags = assess(measurement)
        if flags != QualityFlags.OK:
            raise ValueError(
                f"Structural quality flags at frame {raw_frame.frame_index}, "
                f"measurement {measurement_index}: {_quality_flag_names(flags)}."
            )


def collect_clean_increments(raw_frames: Iterable[RawFrame]) -> list[Increment]:
    """Return real IMU increments only when the complete stream is clean.

    Args:
        raw_frames: Complete raw BMU frames in source-arrival order.

    Returns:
        Decoded real ``(dtheta_b, dv_b, dt)`` increments in frame and
        measurement order.

    Raises:
        ValueError: A confirmed sequence gap or structural measurement concern
            invalidates the capture for strict replay, or an upstream decoder or
            sequence tracker rejects the evidence.

    Notes:
        This is a whole-capture preflight boundary. It adds no synthetic
        increments and cannot turn missing or malformed evidence into a usable
        trajectory.
    """
    increments: list[Increment] = []

    for tracked_frame in iter_tracked_frames(raw_frames):
        validate_clean_tracked_frame(tracked_frame)
        for measurement in tracked_frame.decoded_frame.decoded_message.measurements:
            increments.append((measurement.dtheta, measurement.dv, measurement.dt))

    return increments


def select_longest_clean_increment_segment(
    raw_frames: Iterable[RawFrame],
) -> CleanIncrementSegment:
    """Select the first longest contiguous clean run for offline analysis.

    Args:
        raw_frames: Raw BMU frames in source-arrival order.

    Returns:
        The first maximum-length clean segment, including inclusive raw-frame
        and measurement provenance.

    Raises:
        ValueError: No clean increment exists, or framing, decoding, or sequence
            tracking cannot establish reliable segment boundaries.

    Notes:
        A confirmed gap ends the preceding run before the valid frame that
        reveals it. A flagged measurement is excluded and likewise ends the
        current run; later valid measurements in that frame may start a new
        run. Equal-length runs retain the earlier observed segment.
    """
    best_segment: CleanIncrementSegment | None = None
    current_increments: list[Increment] = []
    current_start: tuple[int, int] | None = None
    current_end: tuple[int, int] | None = None

    def finish_current_segment() -> None:
        """Record the open run if it is longer than the prior best run."""
        nonlocal best_segment
        nonlocal current_increments
        nonlocal current_start
        nonlocal current_end

        if not current_increments:
            return

        assert current_start is not None
        assert current_end is not None

        candidate = CleanIncrementSegment(
            increments=tuple(current_increments),
            frame_start_index=current_start[0],
            measurement_start_index=current_start[1],
            frame_end_index=current_end[0],
            measurement_end_index=current_end[1],
        )

        if (
            best_segment is None
            or len(candidate.increments) > len(best_segment.increments)
        ):
            best_segment = candidate

        current_increments = []
        current_start = None
        current_end = None

    for tracked_frame in iter_tracked_frames(raw_frames):
        raw_frame = tracked_frame.decoded_frame.raw_frame
        gap_event = tracked_frame.gap_before
        if gap_event is not None:
            finish_current_segment()

        for measurement_index, measurement in enumerate(
            tracked_frame.decoded_frame.decoded_message.measurements
        ):
            flags = assess(measurement)
            if flags != QualityFlags.OK:
                finish_current_segment()
                continue

            location = (raw_frame.frame_index, measurement_index)

            if current_start is None:
                current_start = location

            current_increments.append((measurement.dtheta, measurement.dv, measurement.dt))
            current_end = location

    finish_current_segment()

    if best_segment is None:
        raise ValueError("No clean increment segment exists")
    return best_segment
