"""Record thread-safe live-capture progress and recovery provenance.

The recorder counts only BLE notifications and complete UBX frames successfully
written to their respective open archives. Source loops explicitly report their
connection, recording, interruption, and terminal transitions on a shared
host-monotonic timeline. This module owns no device I/O, timer, terminal
output, stop policy, retry schedule, or inferred missing measurements.
"""
import threading
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal


_RXM_MESSAGE_CLASS = 0x02
_RXM_RAWX_MESSAGE_ID = 0x15
_RXM_SFRBX_MESSAGE_ID = 0x13

CaptureSource = Literal["ble", "gnss"]
CaptureSourceState = Literal[
    "idle",
    "connecting",
    "recording",
    "backoff",
    "complete",
    "stopped",
]
CaptureRecoveryEventKind = Literal[
    "connect_attempt",
    "recording_started",
    "interrupted",
    "complete",
    "stopped",
]

_CAPTURE_SOURCES = frozenset(("ble", "gnss"))
_TERMINAL_SOURCE_STATES = frozenset(("complete", "stopped"))


def _require_arrival(value: object) -> float:
    """Return one finite host-arrival observation or reject it."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("host arrival must be a finite real number.")
    return float(value)


def _require_ubx_byte(value: object, field: str) -> int:
    """Return one UBX class or ID byte or reject it."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFF:
        raise ValueError(f"{field} must be an unsigned byte.")
    return value


def _require_source(value: object) -> CaptureSource:
    """Return one supported capture source name or reject it."""
    if value not in _CAPTURE_SOURCES:
        raise ValueError("source must be 'ble' or 'gnss'.")
    return value


def _require_detail(value: object) -> str | None:
    """Return an optional human-readable source-event detail."""
    if value is not None and not isinstance(value, str):
        raise ValueError("detail must be a string or None.")
    return value


@dataclass(frozen=True)
class CaptureRecoveryEvent:
    """Represent one source lifecycle transition alongside raw evidence.

    ``archive_record_count`` identifies the successfully archived record count
    at the transition. For BLE it is a notification count; for GNSS it is a
    complete UBX-frame count. A reconnect boundary is therefore visible in the
    manifest without making an unavailable source interval look continuous.
    Attributes:
        source: ``"ble"`` or ``"gnss"`` evidence source.
        kind: Lifecycle transition observed by the caller-owned source loop.
        host_time: Host-monotonic transition timestamp in s.
        connection_attempt_count: Attempts started for this source through this
            transition.
        recording_session_count: Successfully started recording sessions through
            this transition.
        archive_record_count: Successfully archived records at this transition:
            BLE notifications for ``"ble"`` or complete UBX frames for ``"gnss"``.
        detail: Optional human-readable interruption reason.
    """

    source: CaptureSource
    kind: CaptureRecoveryEventKind
    host_time: float
    connection_attempt_count: int
    recording_session_count: int
    archive_record_count: int
    detail: str | None


@dataclass(frozen=True)
class CaptureSourceSnapshot:
    """Represent one immutable source-health view in a coordinated capture.

    ``outage_started`` is populated only after a recoverable interruption and
    remains set until that source resumes recording or reaches a terminal
    state. ``last_interruption`` remains available after recovery so periodic
    status and the final manifest retain the evidence that an outage occurred.
    Attributes:
        state: Current source lifecycle state.
        connection_attempt_count: Started connection attempts.
        recording_session_count: Sessions that reached recording state.
        recovery_count: Recoverable interruptions reported by the source loop.
        outage_count: Distinct outage intervals opened by interruptions.
        outage_started: Host-monotonic outage start in s, or ``None`` when the
            source is recording or terminal.
        total_outage_duration: Closed outage duration accumulated in s.
        last_interruption: Most recent reported interruption detail, if any.
    """

    state: CaptureSourceState
    connection_attempt_count: int
    recording_session_count: int
    recovery_count: int
    outage_count: int
    outage_started: float | None
    total_outage_duration: float
    last_interruption: str | None


@dataclass
class _MutableSourceProgress:
    """Thread-protected mutable lifecycle facts for one capture source."""

    state: CaptureSourceState = "idle"
    connection_attempt_count: int = 0
    recording_session_count: int = 0
    recovery_count: int = 0
    outage_count: int = 0
    outage_started: float | None = None
    total_outage_duration: float = 0.0
    last_interruption: str | None = None
    last_transition_time: float | None = None


@dataclass(frozen=True)
class CaptureProgressSnapshot:
    """Represent one consistent immutable view of archived live evidence.

    Arrival fields are host-monotonic seconds and remain ``None`` until their
    respective source has written at least one complete record.

    Attributes:
        ble_notification_count: Successfully archived raw BLE notifications.
        last_ble_arrival: Host arrival in s of the latest archived BLE record,
            or ``None`` before one is archived.
        ubx_frame_count: Successfully archived complete native UBX frames.
        rawx_frame_count: Archived UBX RAWX-frame count.
        sfrbx_frame_count: Archived UBX SFRBX-frame count.
        last_ubx_arrival: Host arrival in s of the latest archived UBX frame,
            or ``None`` before one is archived.
        ble_source: Immutable BLE lifecycle and outage snapshot.
        gnss_source: Immutable GNSS lifecycle and outage snapshot.
        recovery_events: Lifecycle transitions in global report order.
    """

    ble_notification_count: int
    last_ble_arrival: float | None
    ubx_frame_count: int
    rawx_frame_count: int
    sfrbx_frame_count: int
    last_ubx_arrival: float | None
    ble_source: CaptureSourceSnapshot
    gnss_source: CaptureSourceSnapshot
    recovery_events: tuple[CaptureRecoveryEvent, ...]


class CaptureProgress:
    """Collect coordinated BLE/UBX archive facts across execution contexts.

    One mutex makes record and snapshot operations internally consistent. The
    caller owns the meaning and scheduling of source events; this class records
    them without declaring whether an outage corresponds to device packet loss.
    """

    def __init__(self) -> None:
        """Initialize an empty progress record protected by one mutex."""
        self._lock = threading.Lock()
        self._ble_notification_count = 0
        self._last_ble_arrival: float | None = None
        self._ubx_frame_count = 0
        self._rawx_frame_count = 0
        self._sfrbx_frame_count = 0
        self._last_ubx_arrival: float | None = None
        self._source_progress = {
            "ble": _MutableSourceProgress(),
            "gnss": _MutableSourceProgress(),
        }
        self._recovery_events: list[CaptureRecoveryEvent] = []

    def record_ble_notification(self, host_arrival: float) -> None:
        """Record one BLE notification after its raw archive write succeeds.

        Args:
            host_arrival: Host-monotonic callback arrival time in s.
        """
        arrival = _require_arrival(host_arrival)
        with self._lock:
            self._ble_notification_count += 1
            self._last_ble_arrival = arrival

    def record_ubx_frame(
        self,
        host_arrival: float,
        message_class: int,
        message_id: int,
    ) -> None:
        """Record one complete UBX frame after its native archive write succeeds.

        Args:
            host_arrival: Host-monotonic frame availability time in s.
            message_class: Native UBX message-class byte.
            message_id: Native UBX message-identifier byte.
        """
        arrival = _require_arrival(host_arrival)
        message_class = _require_ubx_byte(message_class, "message_class")
        message_id = _require_ubx_byte(message_id, "message_id")

        with self._lock:
            self._ubx_frame_count += 1
            self._last_ubx_arrival = arrival
            if (
                message_class == _RXM_MESSAGE_CLASS
                and message_id == _RXM_RAWX_MESSAGE_ID
            ):
                self._rawx_frame_count += 1
            elif (
                message_class == _RXM_MESSAGE_CLASS
                and message_id == _RXM_SFRBX_MESSAGE_ID
            ):
                self._sfrbx_frame_count += 1

    def source_connecting(
        self,
        source: CaptureSource,
        host_time: float,
    ) -> None:
        """Record one new connection attempt for a source recovery loop.

        Args:
            source: Evidence source beginning an attempt.
            host_time: Host-monotonic transition time in s.

        Raises:
            ValueError: The source, timestamp, or lifecycle transition is invalid.
        """
        source = _require_source(source)
        host_time = _require_arrival(host_time)
        with self._lock:
            progress = self._prepare_source_transition(source, host_time)
            if progress.state in _TERMINAL_SOURCE_STATES:
                raise ValueError(f"{source} source is already terminal.")
            progress.connection_attempt_count += 1
            progress.state = "connecting"
            self._append_recovery_event(source, "connect_attempt", host_time, None)

    def source_recording(self, source: CaptureSource, host_time: float) -> None:
        """Record that a source completed connection setup and is recording.

        Args:
            source: Evidence source entering recording state.
            host_time: Host-monotonic transition time in s.

        Raises:
            ValueError: The source, timestamp, or state transition is invalid.
        """
        source = _require_source(source)
        host_time = _require_arrival(host_time)
        with self._lock:
            progress = self._prepare_source_transition(source, host_time)
            if progress.state != "connecting":
                raise ValueError(f"{source} source must be connecting before recording.")
            progress.recording_session_count += 1
            self._close_outage(progress, host_time)
            progress.state = "recording"
            self._append_recovery_event(source, "recording_started", host_time, None)

    def source_interrupted(
        self,
        source: CaptureSource,
        host_time: float,
        detail: str,
    ) -> None:
        """Record one recoverable source interruption before retry backoff.

        Args:
            source: Evidence source leaving a recording session.
            host_time: Host-monotonic interruption time in s.
            detail: Nonempty source-error or interruption description.

        Raises:
            ValueError: The source, timestamp, detail, or state transition is
                invalid.

        Notes:
            The open outage records source unavailability; it does not count
            unreceived BLE notifications or UBX frames.
        """
        source = _require_source(source)
        host_time = _require_arrival(host_time)
        detail = _require_detail(detail)
        if detail is None or not detail:
            raise ValueError("detail must be a nonempty string for an interruption.")
        with self._lock:
            progress = self._prepare_source_transition(source, host_time)
            if progress.state in _TERMINAL_SOURCE_STATES:
                raise ValueError(f"{source} source is already terminal.")
            if progress.outage_started is None:
                progress.outage_started = host_time
                progress.outage_count += 1
            progress.recovery_count += 1
            progress.last_interruption = detail
            progress.state = "backoff"
            self._append_recovery_event(source, "interrupted", host_time, detail)

    def source_finished(
        self,
        source: CaptureSource,
        host_time: float,
        completion_status: Literal["complete", "stopped"],
    ) -> None:
        """Record a normal duration or requested-stop terminal source state.

        Args:
            source: Evidence source reaching a terminal state.
            host_time: Host-monotonic terminal time in s.
            completion_status: ``"complete"`` at a duration boundary or
                ``"stopped"`` for an operator-requested end.

        Raises:
            ValueError: The source, timestamp, status, or transition is invalid.
        """
        source = _require_source(source)
        host_time = _require_arrival(host_time)
        if completion_status not in _TERMINAL_SOURCE_STATES:
            raise ValueError("completion_status must be 'complete' or 'stopped'.")
        with self._lock:
            progress = self._prepare_source_transition(source, host_time)
            if progress.state in _TERMINAL_SOURCE_STATES:
                raise ValueError(f"{source} source is already terminal.")
            self._close_outage(progress, host_time)
            progress.state = completion_status
            self._append_recovery_event(source, completion_status, host_time, None)

    def recovery_events(self) -> tuple[CaptureRecoveryEvent, ...]:
        """Return lifecycle transitions in their global append order.

        Returns:
            Immutable recovery-event tuple captured under the progress mutex.
        """
        with self._lock:
            return tuple(self._recovery_events)

    def snapshot(self) -> CaptureProgressSnapshot:
        """Return a lock-consistent immutable view for caller-owned reporting.

        Returns:
            Counts, last archived arrivals, source states, and recovery evidence
            observed together under the progress mutex.
        """
        with self._lock:
            return CaptureProgressSnapshot(
                ble_notification_count=self._ble_notification_count,
                last_ble_arrival=self._last_ble_arrival,
                ubx_frame_count=self._ubx_frame_count,
                rawx_frame_count=self._rawx_frame_count,
                sfrbx_frame_count=self._sfrbx_frame_count,
                last_ubx_arrival=self._last_ubx_arrival,
                ble_source=self._snapshot_source("ble"),
                gnss_source=self._snapshot_source("gnss"),
                recovery_events=tuple(self._recovery_events),
            )

    def _prepare_source_transition(
        self,
        source: CaptureSource,
        host_time: float,
    ) -> _MutableSourceProgress:
        """Return one mutable source record after validating event time order."""
        progress = self._source_progress[source]
        if (
            progress.last_transition_time is not None
            and host_time < progress.last_transition_time
        ):
            raise ValueError(f"{source} source transition time moved backward.")
        progress.last_transition_time = host_time
        return progress

    def _close_outage(
        self,
        progress: _MutableSourceProgress,
        host_time: float,
    ) -> None:
        """Accumulate one recoverable outage if the source has one open."""
        if progress.outage_started is not None:
            progress.total_outage_duration += host_time - progress.outage_started
            progress.outage_started = None

    def _append_recovery_event(
        self,
        source: CaptureSource,
        kind: CaptureRecoveryEventKind,
        host_time: float,
        detail: str | None,
    ) -> None:
        """Append one lifecycle transition under the caller-held mutex."""
        progress = self._source_progress[source]
        archive_record_count = (
            self._ble_notification_count if source == "ble" else self._ubx_frame_count
        )
        self._recovery_events.append(
            CaptureRecoveryEvent(
                source=source,
                kind=kind,
                host_time=host_time,
                connection_attempt_count=progress.connection_attempt_count,
                recording_session_count=progress.recording_session_count,
                archive_record_count=archive_record_count,
                detail=detail,
            )
        )

    def _snapshot_source(self, source: CaptureSource) -> CaptureSourceSnapshot:
        """Copy one caller-facing source-health record under the mutex."""
        progress = self._source_progress[source]
        return CaptureSourceSnapshot(
            state=progress.state,
            connection_attempt_count=progress.connection_attempt_count,
            recording_session_count=progress.recording_session_count,
            recovery_count=progress.recovery_count,
            outage_count=progress.outage_count,
            outage_started=progress.outage_started,
            total_outage_duration=progress.total_outage_duration,
            last_interruption=progress.last_interruption,
        )
