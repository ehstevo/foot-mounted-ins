"""Inspect recorded framed BMU evidence without replaying navigation.

The command counts raw transport frames, successfully decoded notifications,
and individual IMU measurements; reports host-arrival timing separately from
payload ``dt``; and records confirmed sequence gaps and structural quality flags.
It is observational only: it neither reconstructs missing data nor validates
sensor accuracy, initializes a filter, or makes navigation-policy decisions.

An I/O failure prevents a summary and returns an I/O status. A decoding or
sequence failure stops inspection while retaining and reporting the preceding
partial summary.
"""
import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

from bootins.bmu.quality import QualityFlags, assess
from bootins.bmu.sequence import GapEvent, SequenceTracker
from bootins.bmu.stream import track_raw_frame
from bootins.bmu.transport import RawFrame, iter_frames_replay


@dataclass
class InspectionSummary:
    """Mutable facts accumulated while inspecting one finite raw-frame source.

    Host arrival times are diagnostic; only measurements with usable ``dt``
    contribute to IMU-duration totals. A strict frame failure is retained
    rather than discarding the preceding partial summary.

    Attributes:
        raw_frame_count: Complete framed transport records observed.
        decoded_frame_count: Raw frames decoded successfully into notifications.
        measurement_count: IMU measurements from decoded notifications.
        valid_dt_measurement_count: Measurements with finite, positive ``dt``.
        valid_dt_sum: Sum of valid payload integration intervals (s).
        first_host_arrival: First raw-frame host-monotonic arrival (s).
        last_host_arrival: Last raw-frame host-monotonic arrival (s).
        min_host_interarrival: Smallest adjacent host-arrival interval (s).
        max_host_interarrival: Largest adjacent host-arrival interval (s).
        gap_events: ``(frame_index, GapEvent)`` pairs revealed by valid frames.
        missing_message_count: Total notifications inferred missing from gaps.
        flagged_measurement_count: Measurements with any structural quality flag.
        quality_flag_counts: Counts for each asserted structural quality flag.
        failure: Decoder or sequence failure that ended inspection, if any.
    """

    raw_frame_count: int = 0
    decoded_frame_count: int = 0
    measurement_count: int = 0

    valid_dt_measurement_count: int = 0
    valid_dt_sum: float = 0.0

    first_host_arrival: float | None = None
    last_host_arrival: float | None = None
    min_host_interarrival: float | None = None
    max_host_interarrival: float | None = None

    gap_events: list[tuple[int, GapEvent]] = field(default_factory=list)
    missing_message_count: int = 0

    flagged_measurement_count: int = 0
    quality_flag_counts: Counter[QualityFlags] = field(default_factory=Counter)

    failure: str | None = None


def inspect_raw_frames(raw_frames: Iterable[RawFrame]) -> InspectionSummary:
    """Inspect raw frames through decoding, sequence tracking, and assessment.

    Args:
        raw_frames: Finite ordered raw transport frames with host-arrival
            provenance.

    Returns:
        Inspection facts accumulated through the first terminal source or
        decoder/sequence failure.

    Notes:
        This loop owns a ``SequenceTracker`` and calls ``track_raw_frame``
        directly because it must count facts at raw-frame, decoded-notification,
        and measurement layers. A strict decoder or sequence ``ValueError``
        stops further processing while preserving the partial summary. It only
        observes gaps and flags; it does not repair, reject, or navigate data.
    """
    host_arrival_times = []
    summary = InspectionSummary()
    tracker = SequenceTracker()

    for raw_frame in raw_frames:
        summary.raw_frame_count += 1
        host_arrival_times.append(raw_frame.arrival_monotonic)

        try:
            tracked_frame = track_raw_frame(raw_frame, tracker)
        except ValueError as error:
            summary.failure = str(error)
            break

        summary.decoded_frame_count += 1
        if tracked_frame.gap_before is not None:
            summary.gap_events.append(
                (raw_frame.frame_index, tracked_frame.gap_before)
            )
            summary.missing_message_count += tracked_frame.gap_before.missing_message_count

        decoded_frame = tracked_frame.decoded_frame
        for measurement in decoded_frame.decoded_message.measurements:
            summary.measurement_count += 1
            flags = assess(measurement)

            if flags != QualityFlags.OK:
                summary.flagged_measurement_count += 1

                for flag in QualityFlags:
                    if flags & flag:
                        summary.quality_flag_counts[flag] += 1

            invalid_dt_flags = (
                QualityFlags.NONFINITE_DT | QualityFlags.NONPOSITIVE_DT
            )
            if not (flags & invalid_dt_flags):
                summary.valid_dt_measurement_count += 1
                summary.valid_dt_sum += measurement.dt

    host_arrival_times = np.array(host_arrival_times)
    if len(host_arrival_times) > 0:
        summary.first_host_arrival = host_arrival_times[0]
        summary.last_host_arrival = host_arrival_times[-1]

    if len(host_arrival_times) >= 2:
        interarrival_times = np.diff(host_arrival_times)
        summary.min_host_interarrival = np.min(interarrival_times)
        summary.max_host_interarrival = np.max(interarrival_times)

    return summary


def _format_optional(value: float | None, unit: str, precision: int = 3) -> str:
    """Format an optional scalar consistently in the console report."""
    if value is None:
        return "n/a"
    return f"{value:.{precision}f} {unit}"


def print_summary(capture_path: Path, summary: InspectionSummary) -> None:
    """Print an observational report for one capture summary.

    Args:
        capture_path: Source capture identified in the report.
        summary: Accumulated inspection facts to render without modification.
    """
    print("IMU Replay Inspection")
    print("=" * 72)
    print(f"Capture: {capture_path}")
    print()

    print("Transport")
    print(f"  Raw frames observed: {summary.raw_frame_count}")

    host_span = None
    host_mean_interarrival = None
    host_rate = None
    if (
        summary.raw_frame_count >= 2
        and summary.first_host_arrival is not None
        and summary.last_host_arrival is not None
    ):
        host_span = summary.last_host_arrival - summary.first_host_arrival
        if host_span > 0:
            host_mean_interarrival = host_span / (summary.raw_frame_count - 1)
            host_rate = 1.0 / host_mean_interarrival

    print(f"  Host-arrival span: {_format_optional(host_span, 's', 6)}")
    print(
        "  Host inter-arrival min/mean/max: "
        f"{_format_optional(summary.min_host_interarrival, 's', 6)} / "
        f"{_format_optional(host_mean_interarrival, 's', 6)} / "
        f"{_format_optional(summary.max_host_interarrival, 's', 6)}"
    )
    print(
        "  Host notification rate: "
        f"{_format_optional(host_rate, 'Hz', 3)} (diagnostic only)"
    )
    print()

    print("Decoded IMU")
    print(f"  Decoded notifications: {summary.decoded_frame_count}")
    print(f"  Measurements: {summary.measurement_count}")
    print(f"  Valid dt measurements: {summary.valid_dt_measurement_count}")
    print(f"  Valid dt sum: {summary.valid_dt_sum:.6f} s")

    imu_rate = None
    if summary.valid_dt_sum > 0:
        imu_rate = (
            summary.valid_dt_measurement_count / summary.valid_dt_sum
        )
    print(
        "  Effective IMU rate: "
        f"{_format_optional(imu_rate, 'Hz', 3)} (from valid dt only)"
    )
    print()

    print("Sequence Loss")
    print(f"  Confirmed gap events: {len(summary.gap_events)}")
    print(f"  Known missing notifications: {summary.missing_message_count}")
    for frame_index, gap_event in summary.gap_events:
        print(
            f"  Frame {frame_index}: sequence "
            f"{gap_event.previous_seq_num} -> {gap_event.current_seq_num}; "
            f"missing {gap_event.missing_message_count}"
        )
    print()

    print("Structural Quality")
    print(f"  Flagged measurements: {summary.flagged_measurement_count}")
    for flag in QualityFlags:
        if flag == QualityFlags.OK:
            continue
        print(f"  {flag.name}: {summary.quality_flag_counts[flag]}")
    print()

    if summary.failure is None:
        print("Status: COMPLETE")
    else:
        print(f"Status: STOPPED - {summary.failure}")


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect one framed ``.bmu`` capture and print its evidence summary.

    Args:
        argv: Optional command-line arguments. ``None`` uses process arguments.

    Returns:
        ``0`` after complete inspection, ``1`` after a decoder or sequence
        failure with a printed partial summary, or ``2`` when the capture cannot
        be read.
    """
    parser = argparse.ArgumentParser(
        description="Inspect one framed BMU capture before navigation replay."
    )
    parser.add_argument("capture", type=Path, help="Path to a .bmu capture")
    args = parser.parse_args(argv)

    try:
        summary = inspect_raw_frames(iter_frames_replay(args.capture))
    except OSError as error:
        print(f"Could not read {args.capture}: {error}", file=sys.stderr)
        return 2

    print_summary(args.capture, summary)

    if summary.failure is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
