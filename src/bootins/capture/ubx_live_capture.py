"""Capture receiver-native UBX bytes and derive replay evidence artifacts.

Callers provide a minimal synchronous byte reader, allowing capture timing and
artifact behavior to remain testable without receiver hardware. Complete
checksum-validated frames are archived with host-monotonic availability times;
the terminal archive is then exported as a standard UBX stream and projected
into a normalized GNSS table. Raw receiver-native bytes remain distinct from
the later decoded GNSS observations.

The resilient operation preserves one archive across recoverable reader or
framing interruptions and records their provenance through ``CaptureProgress``.
Incomplete byte suffixes are explicitly discarded and reported rather than
made to appear as complete frames.
"""
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Protocol

from bootins.capture.progress import CaptureProgress
from bootins.sensors.gnss_table import write_gnss_fix_table
from bootins.sensors.ubx import UbxStreamFramer, encode_ubx_frame
from bootins.sensors.ubx_archive import (
    iter_ubx_records,
    write_ubx_record,
    write_ubx_stream,
)
from bootins.sensors.ubx_projection import UbxFixProjection, project_ubx_records


_RXM_MESSAGE_CLASS = 0x02
_RXM_RAWX_MESSAGE_ID = 0x15
_RXM_SFRBX_MESSAGE_ID = 0x13


class ByteReader(Protocol):
    """The narrow synchronous serial-read behavior needed by live capture.

    Readers used with an unbounded run must return regularly, such as through a
    positive serial timeout, so the shared stop event can be observed.
    """

    def read(self, size: int) -> bytes:
        """Return up to ``size`` currently available contiguous stream bytes."""


class ClosableByteReader(ByteReader, Protocol):
    """A reconnectable reader whose owner can close one serial session."""

    def close(self) -> None:
        """Release the current reader session and its underlying transport."""


@dataclass(frozen=True)
class UbxLiveCaptureSummary:
    """Summarize one terminal UBX capture and normalized GNSS projection.

    Attributes:
        host_elapsed: Capture duration on the caller-supplied host clock in s.
        nonempty_read_count: Nonempty reader chunks consumed, not a frame count.
        received_byte_count: Raw bytes received in nonempty chunks, including
            incomplete suffixes that may later be discarded.
        archived_frame_count: Complete validated frames written to the native
            archive.
        projection: Terminal NAV-PVT/NAV-TIMEGPS association result derived
            from the synchronized archive.
        completion_status: ``"complete"`` at deadline or ``"stopped"`` after a
            shared stop request.
        stop_reason: Optional terminal stop detail.
        rawx_frame_count: Archived receiver RAWX evidence-frame count.
        sfrbx_frame_count: Archived receiver SFRBX evidence-frame count.
    """

    host_elapsed: float
    nonempty_read_count: int
    received_byte_count: int
    archived_frame_count: int
    projection: UbxFixProjection
    completion_status: str = "complete"
    stop_reason: str | None = None
    rawx_frame_count: int = 0
    sfrbx_frame_count: int = 0


@dataclass
class _UbxCaptureStats:
    """Mutable counts accumulated across one or more UBX reader sessions."""

    nonempty_read_count: int = 0
    received_byte_count: int = 0
    archived_frame_count: int = 0
    rawx_frame_count: int = 0
    sfrbx_frame_count: int = 0


def _require_positive(value: object, field: str) -> None:
    """Reject nonpositive capture-control values before opening an archive."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{field} must be positive.")


def _require_nonnegative(value: object, field: str) -> None:
    """Reject negative or nonfinite bounded-drain controls."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{field} must be nonnegative.")


def _require_reconnect_controls(
    reconnect_initial_delay: object,
    reconnect_max_delay: object,
) -> None:
    """Reject invalid or inverted bounded-exponential retry controls."""
    _require_positive(reconnect_initial_delay, "reconnect_initial_delay")
    _require_positive(reconnect_max_delay, "reconnect_max_delay")
    if reconnect_max_delay < reconnect_initial_delay:
        raise ValueError(
            "reconnect_max_delay must be greater than or equal to "
            "reconnect_initial_delay."
        )


def _format_error(error: BaseException) -> str:
    """Return a concise, nonempty source-failure description."""
    return str(error) or type(error).__name__


def _consume_ubx_chunk(
    framer: UbxStreamFramer,
    chunk: bytes,
    arrival: float,
    archive_file,
    stats: _UbxCaptureStats,
    progress: CaptureProgress | None,
) -> None:
    """Archive every complete validated UBX frame emitted by one reader chunk.

    ``arrival`` timestamps the reader chunk's host availability, not a GNSS
    measurement epoch. A chunk may contain no full frame or several frames.
    """
    if not chunk:
        return
    stats.nonempty_read_count += 1
    stats.received_byte_count += len(chunk)
    for frame in framer.feed(chunk):
        write_ubx_record(archive_file, arrival, encode_ubx_frame(frame))
        stats.archived_frame_count += 1
        if progress is not None:
            progress.record_ubx_frame(
                arrival,
                frame.message_class,
                frame.message_id,
            )
        if (
            frame.message_class == _RXM_MESSAGE_CLASS
            and frame.message_id == _RXM_RAWX_MESSAGE_ID
        ):
            stats.rawx_frame_count += 1
        elif (
            frame.message_class == _RXM_MESSAGE_CLASS
            and frame.message_id == _RXM_SFRBX_MESSAGE_ID
        ):
            stats.sfrbx_frame_count += 1


def _write_derived_ubx_artifacts(
    archive_path: Path,
    ubx_path: Path,
    table_path: Path,
    archived_frame_count: int,
) -> UbxFixProjection:
    """Derive standard UBX and normalized-table artifacts from a synced archive.

    The derived files overwrite their caller-selected paths only after the raw
    archive has been finalized. The projection verifies that the standard-stream
    export preserved the archived complete-frame count.
    """
    exported_frame_count = write_ubx_stream(iter_ubx_records(archive_path), ubx_path)
    if exported_frame_count != archived_frame_count:
        raise RuntimeError(
            "Standard UBX export frame count disagrees with the native archive."
        )
    projection = project_ubx_records(iter_ubx_records(archive_path))
    write_gnss_fix_table(table_path, projection.fixes)
    return projection


def _build_summary(
    *,
    start: float,
    clock: Callable[[], float],
    stats: _UbxCaptureStats,
    projection: UbxFixProjection,
    completion_status: str,
    stop_reason: str | None,
) -> UbxLiveCaptureSummary:
    """Build one immutable result from common single/multi-session facts."""
    return UbxLiveCaptureSummary(
        host_elapsed=clock() - start,
        nonempty_read_count=stats.nonempty_read_count,
        received_byte_count=stats.received_byte_count,
        archived_frame_count=stats.archived_frame_count,
        projection=projection,
        completion_status=completion_status,
        stop_reason=stop_reason,
        rawx_frame_count=stats.rawx_frame_count,
        sfrbx_frame_count=stats.sfrbx_frame_count,
    )


def capture_ubx_stream(
    reader: ByteReader,
    archive_path: str | Path,
    table_path: str | Path,
    *,
    ubx_path: str | Path,
    duration: float | None,
    stop_event: threading.Event | None = None,
    read_size: int = 4096,
    drain_timeout: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
    progress: CaptureProgress | None = None,
) -> UbxLiveCaptureSummary:
    """Capture a finite or explicitly stopped UBX stream into replay artifacts.

    Each complete frame emitted from a given ``read`` call receives that call's
    post-read host-monotonic arrival observation. A bounded drain after the
    requested duration or stop event lets a partially received final frame
    complete rather than misclassifying an intentional stop as corrupted EOF.
    An unbounded run requires the caller to supply ``stop_event``. The native
    archive is synced before its standard-UBX export and replay projection
    write their linked derived artifacts. When supplied, ``progress`` receives
    one event only after each complete native frame is archived successfully.

    Args:
        reader: Synchronous source of contiguous receiver-native byte chunks.
        archive_path: Destination timestamped native-UBX archive, overwritten.
        table_path: Destination normalized GNSS CSV table, overwritten after
            archive finalization.
        ubx_path: Destination header-free standard UBX stream, overwritten after
            archive finalization.
        duration: Positive host-monotonic duration in s, or ``None`` for a run
            controlled by ``stop_event``.
        stop_event: Required shared stop event for an unbounded capture.
        read_size: Positive maximum reader request size in bytes.
        drain_timeout: Nonnegative final partial-frame drain interval in s.
        clock: Host-monotonic clock used for chunk availability timestamps.
        progress: Optional archive-success progress recorder.

    Returns:
        Native capture counts, terminal status, and derived GNSS projection.

    Raises:
        ValueError: controls are invalid, an unbounded run has no stop event,
            strict UBX framing fails, or a final partial frame cannot complete
            during the drain interval.
        OSError: an archive, standard-UBX stream, or table file cannot be
            written or synced.
    """
    if duration is None:
        if stop_event is None:
            raise ValueError("stop_event is required when duration is None.")
    else:
        _require_positive(duration, "duration")
    if isinstance(read_size, bool) or not isinstance(read_size, int) or read_size <= 0:
        raise ValueError("read_size must be a positive integer.")
    _require_nonnegative(drain_timeout, "drain_timeout")

    archive_path = Path(archive_path)
    table_path = Path(table_path)
    ubx_path = Path(ubx_path)
    artifact_paths = {path.resolve() for path in (archive_path, ubx_path, table_path)}
    if len(artifact_paths) != 3:
        raise ValueError("archive_path, ubx_path, and table_path must differ.")
    start = clock()
    if duration is None:
        deadline = None
    else:
        deadline = start + duration
    framer = UbxStreamFramer()
    stats = _UbxCaptureStats()
    completion_status = "complete"
    stop_reason: str | None = None

    with archive_path.open("wb") as archive_file:
        while True:
            if stop_event is not None and stop_event.is_set():
                completion_status = "stopped"
                stop_reason = "stop requested"
                break
            if deadline is not None and clock() >= deadline:
                break
            chunk = reader.read(read_size)
            _consume_ubx_chunk(
                framer,
                chunk,
                clock(),
                archive_file,
                stats,
                progress,
            )

        drain_deadline = clock() + drain_timeout
        while framer.pending_byte_count and clock() < drain_deadline:
            chunk = reader.read(read_size)
            _consume_ubx_chunk(
                framer,
                chunk,
                clock(),
                archive_file,
                stats,
                progress,
            )

        framer.finish()
        archive_file.flush()
        os.fsync(archive_file.fileno())

    projection = _write_derived_ubx_artifacts(
        archive_path,
        ubx_path,
        table_path,
        stats.archived_frame_count,
    )
    return _build_summary(
        start=start,
        clock=clock,
        stats=stats,
        projection=projection,
        completion_status=completion_status,
        stop_reason=stop_reason,
    )


def _wait_for_ubx_retry(
    delay: float,
    *,
    deadline: float | None,
    stop_event: threading.Event | None,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """Wait through one retry backoff while remaining responsive to stop."""
    remaining = delay
    while remaining > 0.0:
        if stop_event is not None and stop_event.is_set():
            return
        if deadline is not None and clock() >= deadline:
            return
        sleep_duration = min(remaining, 0.25)
        if deadline is not None:
            sleep_duration = min(sleep_duration, max(0.0, deadline - clock()))
        if sleep_duration <= 0.0:
            return
        sleep(sleep_duration)
        remaining -= sleep_duration


def _close_reader(reader: ClosableByteReader) -> OSError | None:
    """Close one retry-owned reader, preserving a recoverable close failure."""
    try:
        reader.close()
    except OSError as error:
        return error
    return None


def capture_ubx_resilient_stream(
    reader_factory: Callable[[], ClosableByteReader],
    archive_path: str | Path,
    table_path: str | Path,
    *,
    ubx_path: str | Path,
    duration: float | None,
    stop_event: threading.Event | None = None,
    read_size: int = 4096,
    drain_timeout: float = 2.0,
    reconnect_initial_delay: float,
    reconnect_max_delay: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress: CaptureProgress | None = None,
) -> UbxLiveCaptureSummary:
    """Capture complete UBX frames across recoverable serial interruptions.

    Each serial open owns one independent UBX framer. A reader failure or wire
    framing failure discards only that session's incomplete suffix, closes the
    reader, records the interruption, and retries with bounded exponential
    backoff. Earlier complete frames remain in one native archive, so derived
    UBX/table artifacts are produced exactly once after capture stops.

    Archive writes and final derived-artifact writes remain fatal. Continuing
    after such a failure would claim preservation of raw evidence that the
    process can no longer guarantee.

    Args:
        reader_factory: Factory opening a closeable reader for each session.
        archive_path: Destination shared native-UBX archive, overwritten.
        table_path: Destination normalized GNSS CSV table, written at completion.
        ubx_path: Destination header-free standard UBX stream, written at
            completion.
        duration: Positive host-monotonic duration in s, or ``None`` for a run
            controlled by ``stop_event``.
        stop_event: Required shared stop event for an unbounded capture.
        read_size: Positive maximum reader request size in bytes.
        drain_timeout: Nonnegative partial-frame drain interval in s.
        reconnect_initial_delay: Initial retry backoff in s.
        reconnect_max_delay: Maximum retry backoff in s.
        clock: Host-monotonic clock for arrivals and recovery provenance.
        sleep: Delay function used during retry backoff.
        progress: Optional shared lifecycle and archive-progress recorder.

    Returns:
        Aggregate capture facts and the terminal archive-derived GNSS projection.

    Raises:
        ValueError: Capture, reader, drain, or reconnect controls are invalid.
        OSError: Archive or derived-artifact I/O fails.
        RuntimeError: An internal archive-count consistency check fails.
    """
    if duration is None:
        if stop_event is None:
            raise ValueError("stop_event is required when duration is None.")
    else:
        _require_positive(duration, "duration")
    if isinstance(read_size, bool) or not isinstance(read_size, int) or read_size <= 0:
        raise ValueError("read_size must be a positive integer.")
    _require_nonnegative(drain_timeout, "drain_timeout")
    _require_reconnect_controls(reconnect_initial_delay, reconnect_max_delay)

    archive_path = Path(archive_path)
    table_path = Path(table_path)
    ubx_path = Path(ubx_path)
    artifact_paths = {path.resolve() for path in (archive_path, ubx_path, table_path)}
    if len(artifact_paths) != 3:
        raise ValueError("archive_path, ubx_path, and table_path must differ.")

    recovery_progress = progress if progress is not None else CaptureProgress()
    start = clock()
    deadline = None if duration is None else start + duration
    stats = _UbxCaptureStats()
    recovery_delay = reconnect_initial_delay
    completion_status = "complete"
    stop_reason: str | None = None

    with archive_path.open("wb") as archive_file:
        while True:
            now = clock()
            if stop_event is not None and stop_event.is_set():
                completion_status = "stopped"
                stop_reason = "stop requested"
                recovery_progress.source_finished("gnss", now, "stopped")
                break
            if deadline is not None and now >= deadline:
                recovery_progress.source_finished("gnss", now, "complete")
                break

            recovery_progress.source_connecting("gnss", now)
            try:
                reader = reader_factory()
            except OSError as error:
                interruption_detail = _format_error(error)
            else:
                recovery_progress.source_recording("gnss", clock())
                recovery_delay = reconnect_initial_delay
                framer = UbxStreamFramer()
                interruption_detail: str | None = None
                terminal_status: str | None = None

                while interruption_detail is None:
                    if stop_event is not None and stop_event.is_set():
                        terminal_status = "stopped"
                        break
                    if deadline is not None and clock() >= deadline:
                        terminal_status = "complete"
                        break
                    try:
                        chunk = reader.read(read_size)
                    except OSError as error:
                        interruption_detail = _format_error(error)
                        break
                    try:
                        _consume_ubx_chunk(
                            framer,
                            chunk,
                            clock(),
                            archive_file,
                            stats,
                            progress,
                        )
                    except ValueError as error:
                        interruption_detail = _format_error(error)

                if terminal_status is not None:
                    drain_deadline = clock() + drain_timeout
                    while framer.pending_byte_count and clock() < drain_deadline:
                        try:
                            chunk = reader.read(read_size)
                        except OSError as error:
                            interruption_detail = _format_error(error)
                            break
                        try:
                            _consume_ubx_chunk(
                                framer,
                                chunk,
                                clock(),
                                archive_file,
                                stats,
                                progress,
                            )
                        except ValueError as error:
                            interruption_detail = _format_error(error)
                            break
                    if framer.pending_byte_count:
                        dropped_byte_count = framer.pending_byte_count
                        if interruption_detail is None:
                            interruption_detail = (
                                "discarded "
                                f"{dropped_byte_count} incomplete UBX bytes at capture stop"
                            )
                        else:
                            interruption_detail = (
                                f"{interruption_detail}; discarded "
                                f"{dropped_byte_count} incomplete UBX bytes"
                            )
                elif (
                    interruption_detail is not None and framer.pending_byte_count
                ):
                    interruption_detail = (
                        f"{interruption_detail}; discarded "
                        f"{framer.pending_byte_count} incomplete UBX bytes"
                    )

                close_error = _close_reader(reader)
                if close_error is not None:
                    close_detail = f"reader close failed: {_format_error(close_error)}"
                    if interruption_detail is None:
                        interruption_detail = close_detail
                    else:
                        interruption_detail = f"{interruption_detail}; {close_detail}"

                if terminal_status is not None:
                    if interruption_detail is not None:
                        recovery_progress.source_interrupted(
                            "gnss",
                            clock(),
                            interruption_detail,
                        )
                    completion_status = terminal_status
                    if terminal_status == "stopped":
                        stop_reason = "stop requested"
                    recovery_progress.source_finished(
                        "gnss",
                        clock(),
                        terminal_status,
                    )
                    break

                if interruption_detail is None:
                    raise RuntimeError("GNSS reader session ended without a terminal cause.")

            recovery_progress.source_interrupted(
                "gnss",
                clock(),
                interruption_detail,
            )
            _wait_for_ubx_retry(
                recovery_delay,
                deadline=deadline,
                stop_event=stop_event,
                clock=clock,
                sleep=sleep,
            )
            recovery_delay = min(recovery_delay * 2.0, reconnect_max_delay)

        archive_file.flush()
        os.fsync(archive_file.fileno())

    projection = _write_derived_ubx_artifacts(
        archive_path,
        ubx_path,
        table_path,
        stats.archived_frame_count,
    )
    return _build_summary(
        start=start,
        clock=clock,
        stats=stats,
        projection=projection,
        completion_status=completion_status,
        stop_reason=stop_reason,
    )
