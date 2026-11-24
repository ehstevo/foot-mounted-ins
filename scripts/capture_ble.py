#!/usr/bin/env python3
"""Capture one finite raw BMU BLE stream with a legacy metadata sidecar.

The reusable capture core owns BLE connection, subscription, silence detection,
bounded reconnect retries, teardown, raw framing, and file durability. This
standalone CLI owns the local device configuration, artifact names, legacy
sidecar, terminal reporting, and exit status.

The raw ``.bmu`` file stores one self-delimiting capture-layer record per BLE
notification:

    <f64 arrival_monotonic_s><u16 payload_len><payload_len raw bytes>

Arrival observations are host-time diagnostics only. Navigation integrates the
payload's own IMU increments, never BLE arrival intervals.
"""
import argparse
import asyncio
import importlib.util
import math
import time
from datetime import datetime
from pathlib import Path

import yaml
from bleak import BleakClient

from bootins.capture.ble_live_capture import capture_ble_resilient_stream
from bootins.capture.progress import CaptureProgress


CONNECT_TIMEOUT = 20.0
CAPTURE_SECONDS = 10.0
EXPECTED_RATE_HZ = 25.0
DEFAULT_NOTIFICATION_SILENCE_TIMEOUT = 5.0
BLE_TEARDOWN_TIMEOUT = 10.0
STATUS_UPDATE_PERIOD = 1.0
DEFAULT_RECONNECT_INITIAL_DELAY = 1.0
DEFAULT_RECONNECT_MAX_DELAY = 30.0


def _load_device_config():
    """Load ignored local BLE configuration required by this acquisition CLI."""
    path = Path(__file__).resolve().parents[1] / "config" / "device_local.py"
    if not path.exists():
        raise SystemExit(
            "Missing config/device_local.py.\n"
            "  cp config/device_example.py config/device_local.py\n"
            "then fill in your device's UUIDs (and optionally DEVICE_ADDRESS)."
        )
    spec = importlib.util.spec_from_file_location("device_local", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CFG = _load_device_config()


def _sanitize_label(raw: str) -> str:
    """Validate the optional filename label passed through argparse."""
    if not raw:
        return ""
    sanitized = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in raw
    )
    return sanitized


def _positive_finite_float(raw: str) -> float:
    """Parse a strictly positive finite CLI duration."""
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _parse_args() -> argparse.Namespace:
    """Parse the finite standalone BLE-capture invocation."""
    parser = argparse.ArgumentParser(
        description="Subscribe to the IMU stream and log raw notifications to a "
        "replayable .bmu capture (+ .meta.yaml sidecar).",
    )
    parser.add_argument(
        "address",
        nargs="?",
        default=None,
        help="BLE address of the device. Defaults to cfg.DEVICE_ADDRESS if set.",
    )
    parser.add_argument(
        "-s",
        "--seconds",
        type=_positive_finite_float,
        default=CAPTURE_SECONDS,
        help=f"capture window in seconds (default: {CAPTURE_SECONDS:.0f})",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=_sanitize_label,
        default="",
        help=(
            "optional tag for this run (e.g. a rest pose: 'poseA'); added to "
            "the output filenames and recorded in the sidecar"
        ),
    )
    parser.add_argument(
        "--silence-timeout",
        type=_positive_finite_float,
        default=DEFAULT_NOTIFICATION_SILENCE_TIMEOUT,
        help="Time without notifications before capture times out (s).",
    )
    parser.add_argument(
        "--reconnect-initial-delay",
        type=_positive_finite_float,
        default=DEFAULT_RECONNECT_INITIAL_DELAY,
        help="initial delay before retrying a recoverable BLE interruption (s)",
    )
    parser.add_argument(
        "--reconnect-max-delay",
        type=_positive_finite_float,
        default=DEFAULT_RECONNECT_MAX_DELAY,
        help="maximum bounded-exponential BLE retry delay (s)",
    )
    args = parser.parse_args()
    if args.reconnect_max_delay < args.reconnect_initial_delay:
        parser.error(
            "--reconnect-max-delay must be greater than or equal to "
            "--reconnect-initial-delay"
        )
    return args


def _resolve_address(cli_address: str | None) -> str:
    """Return the explicit or configured BLE address, rejecting an empty one."""
    address = cli_address or getattr(CFG, "DEVICE_ADDRESS", "")
    if not address:
        raise SystemExit(
            "No address. Either pass one:\n"
            "    python scripts/capture_ble.py 80:XX:XX:XX:XX:XX\n"
            "or set DEVICE_ADDRESS in config/device_local.py to the module's address."
        )
    return address


async def main() -> int:
    """Run one finite standalone BLE capture and return a shell exit status.

    The command creates a timestamped raw ``.bmu`` archive and legacy metadata
    sidecar under ``data/``. It returns ``0`` only for a completed, nonempty,
    synchronized archive; ``1`` represents incomplete capture evidence and
    ``2`` represents archive I/O failure. The sidecar records host-arrival and
    recovery facts but does not establish IMU measurement timing or navigation
    validity.
    """
    args = _parse_args()
    address = _resolve_address(args.address)
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start_wall = datetime.now()
    run_basename = start_wall.strftime("%Y%m%d_%H%M%S")
    if args.label:
        run_basename = f"{run_basename}_{args.label}"
    bmu_path = data_dir / f"{run_basename}.bmu"
    meta_path = data_dir / f"{run_basename}.meta.yaml"

    print(f"Capturing from {address} for {args.seconds:.0f}s -> {bmu_path}")
    progress = CaptureProgress()
    try:
        summary = await capture_ble_resilient_stream(
            lambda: BleakClient(address, timeout=CONNECT_TIMEOUT),
            CFG.IMU_DATA_UUID,
            bmu_path,
            duration=args.seconds,
            silence_timeout=args.silence_timeout,
            connect_timeout=CONNECT_TIMEOUT,
            teardown_timeout=BLE_TEARDOWN_TIMEOUT,
            watchdog_period=STATUS_UPDATE_PERIOD,
            reconnect_initial_delay=args.reconnect_initial_delay,
            reconnect_max_delay=args.reconnect_max_delay,
            clock=time.monotonic,
            sleep=asyncio.sleep,
            progress=progress,
        )
    except OSError as error:
        print(f"INCOMPLETE [raw archive]: {error}")
        return 2
    final_progress = progress.snapshot()

    stream_started = summary.notification_count > 0
    meta = {
        "device_address": address,
        "pose_label": args.label or None,
        "start_wall_clock": start_wall.isoformat(timespec="seconds"),
        "stream_started": stream_started,
        "capture_seconds_requested": args.seconds,
        "capture_seconds_elapsed": round(summary.host_elapsed, 3),
        "recorded_host_arrival_span_s": summary.recorded_host_arrival_span,
        "notification_count": summary.notification_count,
        "bytes_written": summary.bytes_written,
        "completion_status": summary.completion_status,
        "raw_file_sync_status": summary.raw_file_sync_status,
        "raw_file_sync_error": summary.raw_file_sync_error,
        "stop_reason": summary.stop_reason,
        "source_recovery": {
            "state": final_progress.ble_source.state,
            "connection_attempt_count": final_progress.ble_source.connection_attempt_count,
            "recording_session_count": final_progress.ble_source.recording_session_count,
            "recovery_count": final_progress.ble_source.recovery_count,
            "outage_count": final_progress.ble_source.outage_count,
            "total_outage_duration_host_s": final_progress.ble_source.total_outage_duration,
            "last_interruption": final_progress.ble_source.last_interruption,
            "events": [
                {
                    "kind": event.kind,
                    "host_time": event.host_time,
                    "connection_attempt_count": event.connection_attempt_count,
                    "recording_session_count": event.recording_session_count,
                    "archive_record_count": event.archive_record_count,
                    "detail": event.detail,
                }
                for event in final_progress.recovery_events
                if event.source == "ble"
            ],
        },
        "record_format": (
            "<dH (f64 arrival_monotonic_s, u16 payload_len) then payload_len "
            "raw bytes"
        ),
    }
    with meta_path.open("w") as meta_file:
        yaml.safe_dump(meta, meta_file, sort_keys=False)

    capture_succeeded = (
        summary.completion_status == "complete" and summary.notification_count > 0
    )
    artifact_durable = summary.raw_file_sync_status == "synced"
    if capture_succeeded and artifact_durable:
        recovery_count = final_progress.ble_source.recovery_count
        suffix = " [recovered outages]" if recovery_count else ""
        print(f"COMPLETE{suffix}")
    elif summary.notification_count == 0:
        print("INCOMPLETE: no notifications were archived.")
    elif not capture_succeeded:
        print(f"INCOMPLETE [{summary.completion_status}]: {summary.stop_reason}")
    else:
        print(f"INCOMPLETE [raw_file_sync]: {summary.raw_file_sync_error}")

    rate = (
        summary.notification_count / summary.host_elapsed
        if summary.host_elapsed > 0.0
        else 0.0
    )
    print()
    print(
        f"Captured {summary.notification_count} notifications, "
        f"{summary.bytes_written} bytes in {summary.host_elapsed:.2f}s"
    )
    print(f"Measured rate: {rate:.1f} notif/s (expected ~{EXPECTED_RATE_HZ:.0f})")
    if summary.notification_count == 0:
        print("FAIL: zero notifications -- wrong IMU_DATA_UUID, or never subscribed.")
    elif abs(rate - EXPECTED_RATE_HZ) <= 0.1 * EXPECTED_RATE_HZ:
        print("PASS: rate within 10% of expected -- capture looks healthy.")
    else:
        print("WARN: rate off by >10% -- dropouts, weak signal, or not streaming at 25 Hz.")
    print(f"  raw : {bmu_path}")
    print(f"  meta: {meta_path}")
    return 0 if capture_succeeded and artifact_durable else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
