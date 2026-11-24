"""Capture raw BLE notification evidence with host-arrival provenance.

The asynchronous capture operations subscribe to a caller-selected BLE
notification source, timestamp each received payload on a caller-supplied
host-monotonic clock, and append the untouched payload to a raw archive. They
do not decode private BMU data, determine IMU measurement epochs, choose bundle
paths, write a run manifest, or coordinate GNSS acquisition.

The single-session operation reports a terminal session outcome. The resilient
operation retains one archive across recoverable connection or notification
silence interruptions, records those discontinuities through
``CaptureProgress``, and synchronizes the archive only on its terminal return.
Notifications never received are not reconstructed as measurements.
"""
import asyncio
import os
import struct
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path

from bleak import BleakClient
from bleak.exc import BleakError

from bootins.capture.progress import CaptureProgress


@dataclass(frozen=True)
class BleLiveCaptureSummary:
    """Summarize one finite or explicitly stopped raw BLE capture operation.

    ``completion_status`` records the BLE/session outcome independently from
    ``raw_file_sync_status`` so callers can distinguish an incomplete capture
    from a complete capture whose file did not reach durable storage. Host
    arrival timestamps are monotonic-clock observations, not IMU integration
    epochs.

    Attributes:
        host_elapsed: Elapsed host-monotonic time through the final capture
            outcome, before any later caller-side reporting work.
        notification_count: Number of received BLE notifications successfully
            framed and written to the open raw archive. This is not a count of
            inferred missing notifications or decoded IMU measurements.
        bytes_written: Total bytes written to the raw capture, including each
            capture-layer record header.
        first_arrival_monotonic: Host-monotonic arrival observation of the
            first successfully written notification, if any.
        last_arrival_monotonic: Host-monotonic arrival observation of the last
            successfully written notification, if any.
        completion_status: Session outcome such as a completed duration,
            requested stop, notification silence, or BLE error.
        stop_reason: Optional human-readable terminal cause associated with an
            incomplete or failed session.
        raw_file_sync_status: Terminal durability outcome for the raw archive,
            reported separately from BLE/session completion.
        raw_file_sync_error: Optional filesystem error detail when syncing the
            raw file fails.
    """

    host_elapsed: float
    notification_count: int
    bytes_written: int
    first_arrival_monotonic: float | None
    last_arrival_monotonic: float | None
    completion_status: str
    stop_reason: str | None
    raw_file_sync_status: str
    raw_file_sync_error: str | None

    @property
    def recorded_host_arrival_span(self) -> float | None:
        """Return the written-notification host-arrival span in s.

        Returns:
            Difference between last and first archived notification arrival, or
            ``None`` when no notification was written. One notification gives
            ``0.0``.
        """
        if (
            self.first_arrival_monotonic is None
            or self.last_arrival_monotonic is None
        ):
            return None
        return self.last_arrival_monotonic - self.first_arrival_monotonic


@dataclass
class _CaptureStats:
    """Accumulate notifications successfully written to the open archive."""

    notification_count: int = 0
    bytes_written: int = 0
    first_arrival_monotonic: float | None = None
    last_arrival_monotonic: float | None = None


@dataclass(frozen=True)
class _BleSessionResult:
    """Hold the terminal outcome of one connected BLE subscription session."""

    completion_status: str
    stop_reason: str | None
    recording_started: bool
    completed_at: float


def _require_positive(value: object, field: str) -> None:
    """Reject an invalid finite capture timing control before opening a file."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{field} must be positive.")


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


async def _capture_ble_session(
    client: BleakClient,
    notify_uuid: str,
    raw_file,
    capture_stats: _CaptureStats,
    *,
    deadline: float | None,
    silence_timeout: float,
    connect_timeout: float,
    teardown_timeout: float,
    watchdog_period: float,
    stop_event: threading.Event | None,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    progress: CaptureProgress | None,
    on_recording_started: Callable[[float], None] | None = None,
) -> _BleSessionResult:
    """Run one BLE connection and notification subscription against an archive.

    The notification callback timestamps and writes each received raw payload.
    This helper owns neither retry policy nor archive lifecycle: it never opens,
    truncates, flushes, or synchronizes ``raw_file``. A reconnecting caller can
    therefore retain one evidence archive while each session uses a fresh client.
    """
    connected = False
    notifications_started = False
    last_session_arrival: float | None = None
    session_started_at: float | None = None
    write_failure: OSError | None = None
    completion_status = "complete"
    stop_reason: str | None = None
    completed_at: float | None = None

    def notify_handler(_sender, data) -> None:
        nonlocal last_session_arrival, write_failure
        if write_failure is not None:
            return
        arrival = clock()
        payload = bytes(data)
        record = struct.pack("<dH", arrival, len(payload)) + payload
        try:
            raw_file.write(record)
        except OSError as error:
            write_failure = error
            return
        if capture_stats.notification_count == 0:
            capture_stats.first_arrival_monotonic = arrival
        capture_stats.notification_count += 1
        capture_stats.bytes_written += len(record)
        capture_stats.last_arrival_monotonic = arrival
        last_session_arrival = arrival
        if progress is not None:
            progress.record_ble_notification(arrival)

    def record_teardown_failure(operation: str, error: BaseException) -> None:
        nonlocal completion_status, stop_reason
        detail = f"teardown {operation} failed: {_format_error(error)}"
        if completion_status == "complete":
            completion_status = "ble_error"
            stop_reason = detail
        elif stop_reason is None:
            stop_reason = detail
        else:
            stop_reason = f"{stop_reason}; {detail}"

    try:
        await asyncio.wait_for(client.connect(), timeout=connect_timeout)
        connected = True
        await asyncio.wait_for(
            client.start_notify(notify_uuid, notify_handler),
            timeout=connect_timeout,
        )
        notifications_started = True
        session_started_at = clock()
        if on_recording_started is not None:
            on_recording_started(session_started_at)

        while True:
            if write_failure is not None:
                raise write_failure
            if stop_event is not None and stop_event.is_set():
                completion_status = "stopped"
                stop_reason = "stop requested"
                completed_at = clock()
                break

            now = clock()
            if deadline is not None and now >= deadline:
                completed_at = deadline
                break

            sleep_duration = min(watchdog_period, silence_timeout)
            if deadline is not None:
                sleep_duration = min(sleep_duration, deadline - now)
            if sleep_duration <= 0.0:
                break
            await sleep(sleep_duration)

            if write_failure is not None:
                raise write_failure
            now = clock()
            if stop_event is not None and stop_event.is_set():
                completion_status = "stopped"
                stop_reason = "stop requested"
                completed_at = clock()
                break
            if deadline is not None and now >= deadline:
                completed_at = deadline
                break

            if last_session_arrival is None:
                if session_started_at is None:
                    raise RuntimeError("BLE session started without a host timestamp.")
                last_notification_age = now - session_started_at
            else:
                last_notification_age = now - last_session_arrival
            if last_notification_age > silence_timeout:
                completion_status = "notification_silence"
                stop_reason = f"No notifications for {last_notification_age:.1f} s"
                completed_at = now
                break
    except OSError as error:
        if write_failure is error:
            raise
        completion_status = "ble_error"
        stop_reason = _format_error(error)
        completed_at = clock()
    except (BleakError, asyncio.TimeoutError) as error:
        completion_status = "ble_error"
        stop_reason = _format_error(error)
        completed_at = clock()
    finally:
        if notifications_started and client.is_connected:
            try:
                await asyncio.wait_for(
                    client.stop_notify(notify_uuid),
                    timeout=teardown_timeout,
                )
            except (BleakError, asyncio.TimeoutError, OSError) as error:
                record_teardown_failure("stop_notify", error)
        if connected and client.is_connected:
            try:
                await asyncio.wait_for(
                    client.disconnect(),
                    timeout=teardown_timeout,
                )
            except (BleakError, asyncio.TimeoutError, OSError) as error:
                record_teardown_failure("disconnect", error)

    return _BleSessionResult(
        completion_status,
        stop_reason,
        notifications_started,
        clock() if completed_at is None else completed_at,
    )


async def _wait_for_ble_retry(
    delay: float,
    *,
    deadline: float | None,
    stop_event: threading.Event | None,
    watchdog_period: float,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> None:
    """Wait interruptibly through one retry backoff without busy polling."""
    remaining = delay
    while remaining > 0.0:
        if stop_event is not None and stop_event.is_set():
            return
        if deadline is not None and clock() >= deadline:
            return
        sleep_duration = min(remaining, watchdog_period)
        if deadline is not None:
            sleep_duration = min(sleep_duration, max(0.0, deadline - clock()))
        if sleep_duration <= 0.0:
            return
        await sleep(sleep_duration)
        remaining -= sleep_duration


async def capture_ble_stream(
    client: BleakClient,
    notify_uuid: str,
    raw_path: str | Path,
    *,
    duration: float | None,
    silence_timeout: float,
    connect_timeout: float,
    teardown_timeout: float,
    watchdog_period: float,
    stop_event: threading.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    progress: CaptureProgress | None = None,
) -> BleLiveCaptureSummary:
    """Capture one finite BLE notification stream into a raw framed file.

    The supplied client is connected, subscribed, and torn down within this
    call. Each written notification is framed as ``<f64 arrival><u16 length>``
    followed by its uninterpreted raw payload. Arrival observations come from
    ``clock`` and are causal host-time evidence only; they are never used as
    IMU integration intervals. A finite ``duration`` ends normally, while an
    unbounded run requires a caller-owned ``stop_event`` and ends cleanly when
    that event is set. When supplied, ``progress`` receives a notification only
    after its complete raw record is written successfully.

    Args:
        client: Connected-on-demand asynchronous BLE client for one session.
        notify_uuid: Notification characteristic selected by the caller.
        raw_path: Raw archive destination, opened in overwrite mode.
        duration: Positive host-monotonic duration in s, or ``None`` for an
            explicitly stopped capture.
        silence_timeout: Maximum latest-notification age in s.
        connect_timeout: Connect and subscription timeout in s.
        teardown_timeout: Unsubscribe and disconnect timeout in s.
        watchdog_period: Maximum session check interval in s.
        stop_event: Required shared stop event for an unbounded capture.
        clock: Host-monotonic clock used for archive arrival evidence.
        sleep: Awaitable delay used by the watchdog.
        progress: Optional archive-success progress recorder.

    Returns:
        BLE session outcome and raw archive-durability summary.

    Expected BLE failures are recorded in the returned summary, while raw-file
    durability is reported independently. Other programming and filesystem
    failures remain visible to the caller.

    Raises:
        ValueError: A capture timing control is not strictly positive and
            finite, or an unbounded run has no stop event.
        OSError: The raw file cannot be opened, written, flushed, or closed.
    """
    if duration is None:
        if stop_event is None:
            raise ValueError("stop_event is required when duration is None.")
    else:
        _require_positive(duration, "duration")

    for field, value in (
        ("silence_timeout", silence_timeout),
        ("connect_timeout", connect_timeout),
        ("teardown_timeout", teardown_timeout),
        ("watchdog_period", watchdog_period),
    ):
        _require_positive(value, field)

    raw_path = Path(raw_path)
    capture_stats = _CaptureStats()
    start = clock()
    deadline = None if duration is None else start + duration
    raw_file_sync_status = "synced"
    raw_file_sync_error: str | None = None

    with raw_path.open("wb") as raw_file:
        session = await _capture_ble_session(
            client,
            notify_uuid,
            raw_file,
            capture_stats,
            deadline=deadline,
            silence_timeout=silence_timeout,
            connect_timeout=connect_timeout,
            teardown_timeout=teardown_timeout,
            watchdog_period=watchdog_period,
            stop_event=stop_event,
            clock=clock,
            sleep=sleep,
            progress=progress,
        )
        try:
            raw_file.flush()
            os.fsync(raw_file.fileno())
        except OSError as error:
            raw_file_sync_status = "sync_failed"
            raw_file_sync_error = _format_error(error)

    completion_status = session.completion_status
    stop_reason = session.stop_reason
    if completion_status == "complete" and capture_stats.notification_count == 0:
        completion_status = "notification_silence"
        stop_reason = "Capture ended before any notification arrived."

    return BleLiveCaptureSummary(
        host_elapsed=session.completed_at - start,
        notification_count=capture_stats.notification_count,
        bytes_written=capture_stats.bytes_written,
        first_arrival_monotonic=capture_stats.first_arrival_monotonic,
        last_arrival_monotonic=capture_stats.last_arrival_monotonic,
        completion_status=completion_status,
        stop_reason=stop_reason,
        raw_file_sync_status=raw_file_sync_status,
        raw_file_sync_error=raw_file_sync_error,
    )


async def capture_ble_resilient_stream(
    client_factory: Callable[[], BleakClient],
    notify_uuid: str,
    raw_path: str | Path,
    *,
    duration: float | None,
    silence_timeout: float,
    connect_timeout: float,
    teardown_timeout: float,
    watchdog_period: float,
    reconnect_initial_delay: float,
    reconnect_max_delay: float,
    stop_event: threading.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    progress: CaptureProgress | None = None,
) -> BleLiveCaptureSummary:
    """Capture raw BLE evidence across recoverable connection interruptions.

    Each retry receives a newly constructed client while every successfully
    written notification remains in one raw archive. Notification silence,
    BLE connection/subscription failures, and teardown failures are recorded
    as recoverable outages and retried with bounded exponential backoff. The
    function never fabricates the unrecorded interval: callers can retain the
    supplied ``CaptureProgress`` recovery events in capture metadata.

    Args:
        client_factory: Factory creating a fresh BLE client for each attempt.
        notify_uuid: Notification characteristic selected by the caller.
        raw_path: Shared raw archive destination, opened in overwrite mode.
        duration: Positive host-monotonic duration in s, or ``None`` for a
            capture controlled by ``stop_event``.
        silence_timeout: Maximum latest-notification age in s per session.
        connect_timeout: Connect and subscription timeout in s.
        teardown_timeout: Unsubscribe and disconnect timeout in s.
        watchdog_period: Maximum monitoring and retry-check interval in s.
        reconnect_initial_delay: Initial retry backoff in s.
        reconnect_max_delay: Maximum retry backoff in s.
        stop_event: Required shared stop event for an unbounded capture.
        clock: Host-monotonic clock for arrivals, duration, and recovery events.
        sleep: Awaitable delay used during retry backoff.
        progress: Optional shared lifecycle and archive-progress recorder.

    Returns:
        Aggregate archive and terminal completion facts after final sync.

    Raises:
        ValueError: Capture, timeout, or reconnect controls are invalid.
        OSError: The raw archive cannot be opened, written, or synchronized.
        RuntimeError: A session ends without a terminal or recoverable cause.

    Filesystem failures remain fatal because a capture cannot honestly continue
    after losing the ability to preserve its raw evidence. A finite duration or
    a caller-set ``stop_event`` is the only normal terminal condition.
    """
    if duration is None:
        if stop_event is None:
            raise ValueError("stop_event is required when duration is None.")
    else:
        _require_positive(duration, "duration")
    for field, value in (
        ("silence_timeout", silence_timeout),
        ("connect_timeout", connect_timeout),
        ("teardown_timeout", teardown_timeout),
        ("watchdog_period", watchdog_period),
    ):
        _require_positive(value, field)
    _require_reconnect_controls(reconnect_initial_delay, reconnect_max_delay)

    raw_path = Path(raw_path)
    capture_stats = _CaptureStats()
    recovery_progress = progress if progress is not None else CaptureProgress()
    start = clock()
    deadline = None if duration is None else start + duration
    recovery_delay = reconnect_initial_delay
    completion_status = "complete"
    stop_reason: str | None = None
    capture_ended_at: float | None = None

    with raw_path.open("wb") as raw_file:
        while True:
            now = clock()
            if stop_event is not None and stop_event.is_set():
                completion_status = "stopped"
                stop_reason = "stop requested"
                recovery_progress.source_finished("ble", now, "stopped")
                capture_ended_at = now
                break
            if deadline is not None and now >= deadline:
                recovery_progress.source_finished("ble", now, "complete")
                capture_ended_at = deadline
                break

            recovery_progress.source_connecting("ble", now)
            try:
                client = client_factory()
            except (BleakError, asyncio.TimeoutError, OSError) as error:
                session = _BleSessionResult(
                    "ble_error",
                    _format_error(error),
                    False,
                    clock(),
                )
            else:
                session = await _capture_ble_session(
                    client,
                    notify_uuid,
                    raw_file,
                    capture_stats,
                    deadline=deadline,
                    silence_timeout=silence_timeout,
                    connect_timeout=connect_timeout,
                    teardown_timeout=teardown_timeout,
                    watchdog_period=watchdog_period,
                    stop_event=stop_event,
                    clock=clock,
                    sleep=sleep,
                    progress=progress,
                    on_recording_started=lambda host_time: recovery_progress.source_recording(
                        "ble", host_time
                    ),
                )

            now = clock()
            if stop_event is not None and stop_event.is_set():
                completion_status = "stopped"
                stop_reason = "stop requested"
                if session.completion_status not in {"complete", "stopped"}:
                    recovery_progress.source_interrupted(
                        "ble",
                        now,
                        session.stop_reason or session.completion_status,
                    )
                recovery_progress.source_finished("ble", now, "stopped")
                capture_ended_at = session.completed_at
                break
            if deadline is not None and now >= deadline:
                if session.completion_status not in {"complete", "stopped"}:
                    recovery_progress.source_interrupted(
                        "ble",
                        now,
                        session.stop_reason or session.completion_status,
                    )
                recovery_progress.source_finished("ble", now, "complete")
                capture_ended_at = deadline
                break
            if session.completion_status == "complete":
                raise RuntimeError("BLE session ended before the capture deadline.")

            detail = session.stop_reason or session.completion_status
            recovery_progress.source_interrupted("ble", now, detail)
            if session.recording_started:
                recovery_delay = reconnect_initial_delay
            await _wait_for_ble_retry(
                recovery_delay,
                deadline=deadline,
                stop_event=stop_event,
                watchdog_period=watchdog_period,
                clock=clock,
                sleep=sleep,
            )
            recovery_delay = min(recovery_delay * 2.0, reconnect_max_delay)

        raw_file.flush()
        os.fsync(raw_file.fileno())

    return BleLiveCaptureSummary(
        host_elapsed=(clock() if capture_ended_at is None else capture_ended_at) - start,
        notification_count=capture_stats.notification_count,
        bytes_written=capture_stats.bytes_written,
        first_arrival_monotonic=capture_stats.first_arrival_monotonic,
        last_arrival_monotonic=capture_stats.last_arrival_monotonic,
        completion_status=completion_status,
        stop_reason=stop_reason,
        raw_file_sync_status="synced",
        raw_file_sync_error=None,
    )
