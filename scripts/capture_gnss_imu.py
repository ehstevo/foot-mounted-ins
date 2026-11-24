#!/usr/bin/env python3
"""Capture one canonical IMU + GNSS evidence bundle on a shared host clock.

Before running, configure the GNSS receiver for UBX-only output and create the
ignored local BLE configuration:

    cp config/device_example.py config/device_local.py

Fill in the device address and IMU notification UUID in that local file, then
start a continuous capture with:

    python scripts/capture_gnss_imu.py \
        --port /dev/ttyACM0 \
        --label courtyard_square

Omit ``--duration`` for normal operation: ``Ctrl+C`` (SIGINT) or SIGTERM asks
both sources to stop cleanly. For a bounded diagnostic capture, add for example
``--duration 120``. A completed invocation writes a new bundle under
``data/runs/`` (or ``--output-dir``):

    <utc-run-id>[_label]/
      imu.bmu          raw framed BLE notifications
      gnss.ubxlog      host-timestamped receiver-native complete UBX frames
      gnss.ubx         standard concatenated UBX frames for external tools
      gnss_fixes.csv   linked normalized GNSS-fix table
      run.meta.yaml    capture-only manifest

Arguments:
    --port PATH: Required USB serial device for the configured GNSS receiver.
    --address ADDRESS: Optional BLE address overriding ``DEVICE_ADDRESS`` in
        ``config/device_local.py``.
    --duration SECONDS: Optional positive finite capture length. Its absence
        selects continuous operation until a requested stop.
    --label TEXT: Optional short descriptor appended to the generated UTC run
        identifier; unsafe filename characters become underscores.
    --output-dir PATH: Parent directory for canonical run bundles. The default
        is ``data/runs`` relative to the repository root.
    --host-role {laptop,pi}: Capture-host role written to the manifest; the
        default is ``laptop``.
    --silence-timeout SECONDS: Positive BLE inactivity limit before the BLE
        source records a recoverable notification-silence outage and retries.
    --connect-timeout SECONDS: Positive BLE connection and subscription limit.
    --teardown-timeout SECONDS: Positive BLE unsubscribe and disconnect limit.
    --watchdog-period SECONDS: Positive BLE stop/silence polling interval.
    --status-period SECONDS: Positive interval between live capture summaries.
    --baudrate INTEGER: Positive GNSS serial baud rate; defaults to 115200.
    --read-size INTEGER: Positive maximum byte count requested per serial read.
    --read-timeout SECONDS: Positive serial-read timeout. It must remain finite
        so a shared stop request can be observed during continuous operation.
    --drain-timeout SECONDS: Nonnegative time after normal stop for completing
        an already-partial final UBX frame before strict framing is checked.
    --reconnect-initial-delay SECONDS: Positive initial retry delay after a
        recoverable BLE or GNSS interruption.
    --reconnect-max-delay SECONDS: Positive cap for exponential retry delay;
        it must be at least the initial delay.

The same host-monotonic clock timestamps whole BLE notifications and complete
UBX frames; it is availability evidence only. IMU integration remains based on
the payload's ``dt``, while GNSS measurement time remains the receiver's GPS
week/time-of-week. This command captures evidence only: it does not configure
the receiver, decode IMU payloads, select a navigation origin, initialize or
replay a filter, or perform live navigation. Recoverable source outages are
retried and recorded as host-clock provenance; no reconnect interval is
represented as continuous IMU or GNSS evidence.
"""
import argparse
import asyncio
import importlib
import importlib.util
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml
from bleak import BleakClient

from bootins.capture.ble_live_capture import (
    BleLiveCaptureSummary,
    capture_ble_resilient_stream,
)
from bootins.capture.progress import CaptureProgress, CaptureProgressSnapshot
from bootins.capture.ubx_live_capture import (
    ClosableByteReader,
    UbxLiveCaptureSummary,
    capture_ubx_resilient_stream,
)


CONNECT_TIMEOUT = 20.0
DEFAULT_NOTIFICATION_SILENCE_TIMEOUT = 5.0
BLE_TEARDOWN_TIMEOUT = 10.0
DEFAULT_WATCHDOG_PERIOD = 1.0
DEFAULT_STATUS_PERIOD = 5.0
DEFAULT_BAUDRATE = 115200
DEFAULT_READ_SIZE = 4096
DEFAULT_READ_TIMEOUT = 0.1
DEFAULT_DRAIN_TIMEOUT = 2.0
DEFAULT_RECONNECT_INITIAL_DELAY = 1.0
DEFAULT_RECONNECT_MAX_DELAY = 30.0
RUN_CONTRACT_VERSION = 3
IMU_FRAMING = "f64_host_arrival_u16_payload_length_raw_payload_v1"
GNSS_FRAMING = "f64_host_arrival_u32_frame_length_complete_ubx_frame_v1"


def _sanitize_label(raw: str) -> str:
    """Return a filesystem-safe short label for a generated run identifier."""
    return "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in raw
    )


def _positive_float(raw: str) -> float:
    """Parse one strictly positive finite command-line duration."""
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_float(raw: str) -> float:
    """Parse one finite nonnegative command-line drain duration."""
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _positive_int(raw: str) -> int:
    """Parse one strictly positive command-line integer."""
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the canonical coordinated-capture command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="USB GNSS serial device path")
    parser.add_argument(
        "--address",
        help="BLE address; defaults to DEVICE_ADDRESS in config/device_local.py",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        help="finite capture duration; omit to run until SIGINT or SIGTERM",
    )
    parser.add_argument(
        "--label",
        type=_sanitize_label,
        default="",
        help="optional short label appended to the generated run identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "runs",
        help="parent directory for generated canonical run bundles",
    )
    parser.add_argument(
        "--host-role",
        choices=("laptop", "pi"),
        default="laptop",
        help="host platform role recorded in run.meta.yaml",
    )
    parser.add_argument(
        "--silence-timeout",
        type=_positive_float,
        default=DEFAULT_NOTIFICATION_SILENCE_TIMEOUT,
        help="BLE notification silence timeout",
    )
    parser.add_argument(
        "--connect-timeout",
        type=_positive_float,
        default=CONNECT_TIMEOUT,
        help="BLE connect and subscribe timeout",
    )
    parser.add_argument(
        "--teardown-timeout",
        type=_positive_float,
        default=BLE_TEARDOWN_TIMEOUT,
        help="BLE unsubscribe and disconnect timeout",
    )
    parser.add_argument(
        "--watchdog-period",
        type=_positive_float,
        default=DEFAULT_WATCHDOG_PERIOD,
        help="BLE stop and silence polling period",
    )
    parser.add_argument(
        "--status-period",
        type=_positive_float,
        default=DEFAULT_STATUS_PERIOD,
        help="live capture-status reporting interval",
    )
    parser.add_argument("--baudrate", type=_positive_int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--read-size", type=_positive_int, default=DEFAULT_READ_SIZE)
    parser.add_argument(
        "--read-timeout",
        type=_positive_float,
        default=DEFAULT_READ_TIMEOUT,
        help="positive serial-read timeout so a requested stop is observable",
    )
    parser.add_argument(
        "--drain-timeout",
        type=_nonnegative_float,
        default=DEFAULT_DRAIN_TIMEOUT,
        help="post-stop UBX partial-frame drain duration",
    )
    parser.add_argument(
        "--reconnect-initial-delay",
        type=_positive_float,
        default=DEFAULT_RECONNECT_INITIAL_DELAY,
        help="initial delay before retrying a recoverable source interruption",
    )
    parser.add_argument(
        "--reconnect-max-delay",
        type=_positive_float,
        default=DEFAULT_RECONNECT_MAX_DELAY,
        help="maximum bounded-exponential source retry delay",
    )
    args = parser.parse_args(argv)
    if args.reconnect_max_delay < args.reconnect_initial_delay:
        parser.error(
            "--reconnect-max-delay must be greater than or equal to "
            "--reconnect-initial-delay"
        )
    return args


def _load_device_config():
    """Load the ignored local BLE interface configuration when capture starts."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "device_local.py"
    if not config_path.exists():
        raise RuntimeError(
            "missing config/device_local.py; copy config/device_example.py and "
            "supply local device settings."
        )
    specification = importlib.util.spec_from_file_location(
        "device_local",
        config_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load config/device_local.py.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _resolve_address(config, cli_address: str | None) -> str:
    """Use an explicit BLE address or the configured default address."""
    address = cli_address or getattr(config, "DEVICE_ADDRESS", "")
    if not isinstance(address, str) or not address:
        raise RuntimeError(
            "no BLE address; pass --address or set DEVICE_ADDRESS in "
            "config/device_local.py."
        )
    return address


def _resolve_notify_uuid(config) -> str:
    """Return the locally configured IMU notification characteristic UUID."""
    notify_uuid = getattr(config, "IMU_DATA_UUID", "")
    if not isinstance(notify_uuid, str) or not notify_uuid:
        raise RuntimeError("config/device_local.py has no IMU_DATA_UUID.")
    return notify_uuid


def _open_serial(port: str, baudrate: int, timeout: float):
    """Open one USB serial reader without making pyserial an import-time need."""
    try:
        serial = importlib.import_module("serial")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "pyserial is required for USB GNSS capture; install project dependencies."
        ) from error
    return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)


def _make_run_id(started_at: datetime, label: str) -> str:
    """Build a readable UTC run identifier from one capture start instant."""
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    if label:
        run_id = f"{run_id}_{label}"
    return run_id


def _logger_git_commit() -> str:
    """Return the capture software revision or an honest unavailable marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    commit = result.stdout.strip()
    return commit or "unavailable"


def _install_stop_signal_handlers(stop_event: threading.Event) -> Callable[[], None]:
    """Arrange for SIGINT and SIGTERM to request source-level clean shutdown."""
    loop = asyncio.get_running_loop()
    loop_signals: list[signal.Signals] = []
    fallback_signals: list[tuple[signal.Signals, object]] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except (NotImplementedError, RuntimeError):
            try:
                previous_handler = signal.signal(
                    signum,
                    lambda _signum, _frame: stop_event.set(),
                )
            except ValueError:
                continue
            fallback_signals.append((signum, previous_handler))
        else:
            loop_signals.append(signum)

    def restore() -> None:
        """Restore the event loop and process signal state present at startup."""
        for signum in loop_signals:
            loop.remove_signal_handler(signum)
        for signum, previous_handler in fallback_signals:
            signal.signal(signum, previous_handler)

    return restore


async def _capture_sources(
    ble_client_factory: Callable[[], BleakClient],
    notify_uuid: str,
    gnss_reader_factory: Callable[[], ClosableByteReader],
    imu_path: str | Path,
    gnss_archive_path: str | Path,
    gnss_ubx_path: str | Path,
    gnss_table_path: str | Path,
    *,
    duration: float | None,
    silence_timeout: float,
    connect_timeout: float,
    teardown_timeout: float,
    watchdog_period: float,
    stop_event: threading.Event,
    progress: CaptureProgress | None = None,
    read_size: int = DEFAULT_READ_SIZE,
    drain_timeout: float = DEFAULT_DRAIN_TIMEOUT,
    reconnect_initial_delay: float = DEFAULT_RECONNECT_INITIAL_DELAY,
    reconnect_max_delay: float = DEFAULT_RECONNECT_MAX_DELAY,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[BleLiveCaptureSummary, UbxLiveCaptureSummary]:
    """Capture BLE notifications and UBX frames under one stop policy.

    Args:
        ble_client_factory: Factory for a fresh BLE client session.
        notify_uuid: Locally authorized IMU notification characteristic.
        gnss_reader_factory: Factory for a fresh GNSS serial-reader session.
        imu_path: Destination for framed raw BLE notifications.
        gnss_archive_path: Destination for timestamped complete UBX frames.
        gnss_ubx_path: Destination for the derived standard UBX stream.
        gnss_table_path: Destination for the derived normalized GNSS table.
        duration: Optional shared capture duration in seconds.
        silence_timeout: BLE notification inactivity limit in seconds.
        connect_timeout: BLE connection/subscription limit in seconds.
        teardown_timeout: BLE cleanup limit in seconds.
        watchdog_period: BLE stop/silence polling interval in seconds.
        stop_event: Shared request for orderly source shutdown.
        progress: Optional shared source-state accumulator.
        read_size: Maximum requested serial-read size in bytes.
        drain_timeout: Post-stop partial-UBX drain interval in seconds.
        reconnect_initial_delay: Initial source-retry delay in seconds.
        reconnect_max_delay: Maximum exponential source-retry delay in seconds.
        clock: Shared host-monotonic clock for archival provenance.
        sleep: Awaitable sleep implementation used by BLE capture.

    Returns:
        Terminal BLE and GNSS capture summaries, in that order.

    Notes:
        Recoverable BLE and serial interruptions are handled by the individual
        source loops. A fatal archive or configuration exception requests clean
        peer shutdown and is re-raised after the peer reaches terminal cleanup.
        Shared host-clock receipt timestamps are provenance, not sensor-time
        synchronization or continuous evidence across an outage.
    """
    async def capture_ble_source() -> BleLiveCaptureSummary:
        try:
            return await capture_ble_resilient_stream(
                client_factory=ble_client_factory,
                notify_uuid=notify_uuid,
                raw_path=imu_path,
                duration=duration,
                silence_timeout=silence_timeout,
                connect_timeout=connect_timeout,
                teardown_timeout=teardown_timeout,
                watchdog_period=watchdog_period,
                reconnect_initial_delay=reconnect_initial_delay,
                reconnect_max_delay=reconnect_max_delay,
                stop_event=stop_event,
                clock=clock,
                sleep=sleep,
                progress=progress,
            )
        except BaseException:
            stop_event.set()
            raise

    def capture_ubx_source() -> UbxLiveCaptureSummary:
        try:
            return capture_ubx_resilient_stream(
                gnss_reader_factory,
                gnss_archive_path,
                gnss_table_path,
                ubx_path=gnss_ubx_path,
                duration=duration,
                stop_event=stop_event,
                read_size=read_size,
                drain_timeout=drain_timeout,
                reconnect_initial_delay=reconnect_initial_delay,
                reconnect_max_delay=reconnect_max_delay,
                clock=clock,
                progress=progress,
            )
        except BaseException:
            stop_event.set()
            raise

    loop = asyncio.get_running_loop()
    ble_task = asyncio.create_task(capture_ble_source())

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="ubx-capture",
    ) as executor:
        ubx_future = executor.submit(capture_ubx_source)
        ubx_finished = asyncio.Event()

        def report_ubx_completion(_future) -> None:
            loop.call_soon_threadsafe(ubx_finished.set)

        ubx_future.add_done_callback(report_ubx_completion)
        ble_summary: BleLiveCaptureSummary | None = None
        ubx_summary: UbxLiveCaptureSummary | None = None

        try:
            while ble_summary is None or ubx_summary is None:
                if ble_summary is None and ble_task.done():
                    ble_summary = ble_task.result()
                if ubx_summary is None and ubx_finished.is_set():
                    ubx_summary = ubx_future.result()
                if ble_summary is None or ubx_summary is None:
                    await asyncio.sleep(0.01)
        except BaseException:
            stop_event.set()
            await asyncio.gather(ble_task, return_exceptions=True)
            await ubx_finished.wait()
            raise

        if ble_summary is None or ubx_summary is None:
            raise RuntimeError("capture sources ended without summaries.")

        return ble_summary, ubx_summary


def _format_arrival_age(now: float, arrival: float | None) -> str:
    """Describe the age of one optional shared-clock arrival observation."""
    if arrival is None:
        return "none"
    return f"{max(0.0, now - arrival):.1f} s ago"


def _format_source_health(now: float, source) -> str:
    """Format one source's recovery state without treating it as data timing."""
    details = (
        f"{source.state}; attempts {source.connection_attempt_count}, "
        f"sessions {source.recording_session_count}, recoveries {source.recovery_count}"
    )
    if source.outage_started is not None:
        details += f", outage {max(0.0, now - source.outage_started):.1f} s"
    if source.last_interruption is not None:
        details += f", last issue: {source.last_interruption}"
    return details


def _print_progress(
    elapsed: float,
    now: float,
    snapshot: CaptureProgressSnapshot,
) -> None:
    """Print one coherent live-capture status line from a stable snapshot."""
    ble_age = _format_arrival_age(now, snapshot.last_ble_arrival)
    ubx_age = _format_arrival_age(now, snapshot.last_ubx_arrival)
    print(
        f"[{elapsed:7.1f} s] "
        f"BLE: {snapshot.ble_notification_count} notifications, last {ble_age}, "
        f"{_format_source_health(now, snapshot.ble_source)} | "
        f"GNSS: {snapshot.ubx_frame_count} frames "
        f"(RAWX {snapshot.rawx_frame_count}, SFRBX {snapshot.sfrbx_frame_count}), "
        f"last {ubx_age}, {_format_source_health(now, snapshot.gnss_source)}"
    )


async def _await_capture_with_progress(
    capture: Awaitable[tuple[BleLiveCaptureSummary, UbxLiveCaptureSummary]],
    *,
    progress: CaptureProgress,
    status_period: float,
    started: float,
    clock: Callable[[], float],
) -> tuple[BleLiveCaptureSummary, UbxLiveCaptureSummary]:
    """Await coordinated capture while reporting periodic progress snapshots.

    Args:
        capture: Awaitable coordinated BLE and GNSS capture operation.
        progress: Shared capture-state accumulator to snapshot.
        status_period: Console reporting interval in seconds.
        started: Host-monotonic capture start time in seconds.
        clock: Host-monotonic clock used for status timing.

    Returns:
        Terminal BLE and GNSS capture summaries, in that order.

    Raises:
        BaseException: Propagates cancellation or capture failure after
            cancelling the pending coordinated-capture task.
    """
    capture_task = asyncio.create_task(capture)
    try:
        while True:
            done, _ = await asyncio.wait((capture_task,), timeout=status_period)
            if capture_task in done:
                return capture_task.result()

            now = clock()
            _print_progress(now - started, now, progress.snapshot())
    except BaseException:
        if not capture_task.done():
            capture_task.cancel()
            await asyncio.gather(capture_task, return_exceptions=True)
        raise


def _build_manifest(
    *,
    run_id: str,
    label: str,
    started_at: datetime,
    capture_duration: float,
    host_role: str,
    address: str,
    ble_summary: BleLiveCaptureSummary,
    ubx_summary: UbxLiveCaptureSummary,
    progress: CaptureProgressSnapshot,
) -> dict[str, object]:
    """Build capture provenance for one completed canonical evidence bundle.

    Args:
        run_id: Generated UTC identifier for the bundle.
        label: Optional operator-supplied label after filename sanitization.
        started_at: UTC wall-clock capture start instant.
        capture_duration: Host-monotonic elapsed capture duration in seconds.
        host_role: Declared capture-host role.
        address: BLE peer address used for the capture.
        ble_summary: Terminal raw-notification archival summary.
        ubx_summary: Terminal receiver-native archival and projection summary.
        progress: Final source-state and recovery provenance.

    Returns:
        Mapping written unchanged as the capture-only ``run.meta.yaml``
        manifest. It records clock-domain identities and artifact facts, not
        a navigation result or sensor-time alignment.
    """
    projection = ubx_summary.projection
    return {
        "run_contract_version": RUN_CONTRACT_VERSION,
        "run_id": run_id,
        "label": label or None,
        "capture_started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "capture_duration_host_s": capture_duration,
        "capture": {
            "host_role": host_role,
            "logger_git_commit": _logger_git_commit(),
            "device_address": address,
            "source_recovery": {
                "ble": {
                    "state": progress.ble_source.state,
                    "connection_attempt_count": progress.ble_source.connection_attempt_count,
                    "recording_session_count": progress.ble_source.recording_session_count,
                    "recovery_count": progress.ble_source.recovery_count,
                    "outage_count": progress.ble_source.outage_count,
                    "total_outage_duration_host_s": progress.ble_source.total_outage_duration,
                    "last_interruption": progress.ble_source.last_interruption,
                },
                "gnss": {
                    "state": progress.gnss_source.state,
                    "connection_attempt_count": progress.gnss_source.connection_attempt_count,
                    "recording_session_count": progress.gnss_source.recording_session_count,
                    "recovery_count": progress.gnss_source.recovery_count,
                    "outage_count": progress.gnss_source.outage_count,
                    "total_outage_duration_host_s": progress.gnss_source.total_outage_duration,
                    "last_interruption": progress.gnss_source.last_interruption,
                },
                "events": [
                    {
                        "source": event.source,
                        "kind": event.kind,
                        "host_time": event.host_time,
                        "connection_attempt_count": event.connection_attempt_count,
                        "recording_session_count": event.recording_session_count,
                        "archive_record_count": event.archive_record_count,
                        "detail": event.detail,
                    }
                    for event in progress.recovery_events
                ],
            },
        },
        "artifacts": {
            "imu": {
                "path": "imu.bmu",
                "framing": IMU_FRAMING,
                "ble_notification_count": ble_summary.notification_count,
                "bytes_written": ble_summary.bytes_written,
                "completion_status": ble_summary.completion_status,
                "stop_reason": ble_summary.stop_reason,
                "raw_file_sync_status": ble_summary.raw_file_sync_status,
                "raw_file_sync_error": ble_summary.raw_file_sync_error,
            },
            "gnss": {
                "native_archive": {
                    "path": "gnss.ubxlog",
                    "framing": GNSS_FRAMING,
                    "archived_frame_count": ubx_summary.archived_frame_count,
                    "received_byte_count": ubx_summary.received_byte_count,
                    "nonempty_read_count": ubx_summary.nonempty_read_count,
                    "completion_status": ubx_summary.completion_status,
                    "stop_reason": ubx_summary.stop_reason,
                },
                "standard_ubx_stream": {
                    "path": "gnss.ubx",
                    "source": "concatenated_complete_native_archive_frames",
                    "frame_count": ubx_summary.archived_frame_count,
                    "rawx_frame_count": ubx_summary.rawx_frame_count,
                    "sfrbx_frame_count": ubx_summary.sfrbx_frame_count,
                },
                "fix_table": {
                    "path": "gnss_fixes.csv",
                    "normalized_fix_count": len(projection.fixes),
                    "nav_pvt_count": projection.nav_pvt_count,
                    "nav_timegps_count": projection.nav_timegps_count,
                    "unpaired_pvt_count": projection.unpaired_pvt_count,
                    "invalid_time_pvt_count": projection.invalid_time_pvt_count,
                },
            },
        },
        "clock_domains": {
            "host_arrival": "monotonic_seconds",
            "bmu_increment": "delta_t_seconds",
            "bmu_header_timestamp": "coarse_raw_device_timestamp",
            "gnss_fix_epoch": "gps_week_and_time_of_week",
        },
        "provenance": {
            "imu_message_id": "not decoded during capture",
            "wire_endianness": "little",
            "interface_reference": "local device interface configuration",
        },
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Serialize and sync one capture manifest after all capture artifacts exist."""
    with path.open("w", encoding="utf-8") as manifest_file:
        yaml.safe_dump(manifest, manifest_file, sort_keys=False)
        manifest_file.flush()
        os.fsync(manifest_file.fileno())


def _capture_is_healthy(
    ble_summary: BleLiveCaptureSummary,
    ubx_summary: UbxLiveCaptureSummary,
) -> bool:
    """Decide whether a terminal source pair contains usable bundle evidence."""
    normal_statuses = {"complete", "stopped"}
    return (
        ble_summary.completion_status in normal_statuses
        and ubx_summary.completion_status in normal_statuses
        and ble_summary.raw_file_sync_status == "synced"
        and ble_summary.notification_count > 0
        and ubx_summary.archived_frame_count > 0
        and bool(ubx_summary.projection.fixes)
    )


def _print_summary(
    bundle_path: Path,
    ble_summary: BleLiveCaptureSummary,
    ubx_summary: UbxLiveCaptureSummary,
    capture_duration: float,
    progress: CaptureProgressSnapshot,
) -> None:
    """Report artifact locations and capture-layer integrity facts."""
    projection = ubx_summary.projection
    print(f"  bundle: {bundle_path}")
    print(f"  Host elapsed: {capture_duration:.3f} s")
    print(f"  IMU notifications: {ble_summary.notification_count}")
    print(f"  IMU bytes written: {ble_summary.bytes_written}")
    print(f"  IMU terminal status: {ble_summary.completion_status}")
    print(
        "  IMU recovery: "
        f"{progress.ble_source.recovery_count} interruptions across "
        f"{progress.ble_source.recording_session_count} sessions"
    )
    if progress.ble_source.last_interruption is not None:
        print(f"  IMU last recovery issue: {progress.ble_source.last_interruption}")
    print(f"  GNSS archived frames: {ubx_summary.archived_frame_count}")
    print("  GNSS standard UBX stream: gnss.ubx")
    print(f"  GNSS RXM-RAWX frames: {ubx_summary.rawx_frame_count}")
    print(f"  GNSS RXM-SFRBX frames: {ubx_summary.sfrbx_frame_count}")
    print(f"  GNSS normalized fixes: {len(projection.fixes)}")
    print(f"  GNSS terminal status: {ubx_summary.completion_status}")
    print(
        "  GNSS recovery: "
        f"{progress.gnss_source.recovery_count} interruptions across "
        f"{progress.gnss_source.recording_session_count} sessions"
    )
    if progress.gnss_source.last_interruption is not None:
        print(f"  GNSS last recovery issue: {progress.gnss_source.last_interruption}")
    print(f"  manifest: {bundle_path / 'run.meta.yaml'}")


async def main(argv: list[str] | None = None) -> int:
    """Create one finite or operator-stopped coordinated evidence bundle.

    Args:
        argv: Optional command-line arguments. ``None`` uses process arguments.

    Returns:
        ``0`` for a nonempty, synchronized BLE archive and GNSS archive with
        at least one normalized fix; ``1`` for incomplete capture evidence; or
        ``2`` for configuration, runtime, archive, or manifest failures.

    Notes:
        A completed bundle contains raw BLE and receiver-native evidence, their
        derived GNSS artifacts, and a final manifest. SIGINT and SIGTERM request
        orderly source shutdown; they do not make source receipt times a common
        sensor measurement timeline.
    """
    args = _parse_args(argv)
    try:
        config = _load_device_config()
        address = _resolve_address(config, args.address)
        notify_uuid = _resolve_notify_uuid(config)
    except RuntimeError as error:
        print(f"Could not start IMU + GNSS capture: {error}")
        return 2

    started_at = datetime.now(timezone.utc)
    run_id = _make_run_id(started_at, args.label)
    bundle_path = args.output_dir / run_id
    try:
        bundle_path.mkdir(parents=True)
    except FileExistsError:
        print(f"Refusing to overwrite existing run bundle: {bundle_path}")
        return 2
    except OSError as error:
        print(f"Could not create run bundle: {error}")
        return 2

    imu_path = bundle_path / "imu.bmu"
    gnss_archive_path = bundle_path / "gnss.ubxlog"
    gnss_ubx_path = bundle_path / "gnss.ubx"
    gnss_table_path = bundle_path / "gnss_fixes.csv"
    manifest_path = bundle_path / "run.meta.yaml"
    stop_event = threading.Event()
    progress = CaptureProgress()
    restore_signal_handlers = _install_stop_signal_handlers(stop_event)
    host_start = time.monotonic()
    print(f"Capturing IMU + GNSS evidence into {bundle_path}")
    try:
        ble_client_factory = lambda: BleakClient(
            address,
            timeout=args.connect_timeout,
        )
        gnss_reader_factory = lambda: _open_serial(
            args.port,
            args.baudrate,
            args.read_timeout,
        )
        ble_summary, ubx_summary = await _await_capture_with_progress(
            _capture_sources(
                ble_client_factory,
                notify_uuid,
                gnss_reader_factory,
                imu_path,
                gnss_archive_path,
                gnss_ubx_path,
                gnss_table_path,
                duration=args.duration,
                silence_timeout=args.silence_timeout,
                connect_timeout=args.connect_timeout,
                teardown_timeout=args.teardown_timeout,
                watchdog_period=args.watchdog_period,
                stop_event=stop_event,
                progress=progress,
                read_size=args.read_size,
                drain_timeout=args.drain_timeout,
                reconnect_initial_delay=args.reconnect_initial_delay,
                reconnect_max_delay=args.reconnect_max_delay,
                clock=time.monotonic,
            ),
            progress=progress,
            status_period=args.status_period,
            started=host_start,
            clock=time.monotonic,
        )
    except RuntimeError as error:
        print(f"Capture runtime failure: {error}")
        return 2
    except OSError as error:
        print(f"Capture I/O failure: {error}")
        return 2
    except ValueError as error:
        print(f"INCOMPLETE: {error}")
        return 1
    finally:
        restore_signal_handlers()

    capture_duration = time.monotonic() - host_start
    final_progress = progress.snapshot()
    manifest = _build_manifest(
        run_id=run_id,
        label=args.label,
        started_at=started_at,
        capture_duration=capture_duration,
        host_role=args.host_role,
        address=address,
        ble_summary=ble_summary,
        ubx_summary=ubx_summary,
        progress=final_progress,
    )
    try:
        _write_manifest(manifest_path, manifest)
    except OSError as error:
        print(f"Capture manifest failure: {error}")
        return 2

    if _capture_is_healthy(ble_summary, ubx_summary):
        recovery_count = (
            final_progress.ble_source.recovery_count
            + final_progress.gnss_source.recovery_count
        )
        if (
            ble_summary.completion_status == "stopped"
            or ubx_summary.completion_status == "stopped"
        ):
            suffix = "; recovered outages" if recovery_count else ""
            print(f"COMPLETE [operator stop{suffix}]")
        else:
            suffix = " [recovered outages]" if recovery_count else ""
            print(f"COMPLETE{suffix}")
        _print_summary(
            bundle_path,
            ble_summary,
            ubx_summary,
            capture_duration,
            final_progress,
        )
        return 0

    print("INCOMPLETE: bundle has insufficient or undurable capture evidence.")
    _print_summary(
        bundle_path,
        ble_summary,
        ubx_summary,
        capture_duration,
        final_progress,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
