#!/usr/bin/env python3
"""Capture receiver-native UBX evidence from a USB serial stream.

The connected receiver must already emit UBX-NAV-PVT and UBX-NAV-TIMEGPS on the
selected port without interleaved NMEA text. The command records host-timestamped
complete UBX frames in a ``.ubxlog`` archive. After successful archival and
durable synchronization, it derives a standard concatenated ``.ubx`` stream for
external tools and a normalized GNSS-fix CSV by pairing compatible PVT and
TIMEGPS records.

Recoverable serial or wire-framing interruptions start a fresh reader and
framer. Any incomplete suffix is recorded as an interruption fact and discarded,
not represented as continuous receiver evidence. Other receiver-native messages
that are archived are not necessarily normalized into fix records. This utility
captures observations; it neither configures the receiver nor establishes
navigation validity or synchronization with another sensor.

Example:
    python scripts/capture_ubx.py --port /dev/ttyACM0 \\
        --archive-output data/receiver.ubxlog \\
        --ubx-output data/receiver.ubx \\
        --table-output data/receiver_fixes.csv
"""
import argparse
import importlib
from math import isfinite
from pathlib import Path

from bootins.capture.progress import CaptureProgress
from bootins.capture.ubx_live_capture import UbxLiveCaptureSummary, capture_ubx_resilient_stream


def _positive_float(value: str) -> float:
    """Parse one command-line positive finite capture duration."""
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_int(value: str) -> int:
    """Parse one command-line positive integer reader setting."""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_float(value: str) -> float:
    """Parse one command-line nonnegative finite drain duration."""
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _open_serial(port: str, baudrate: int, timeout: float):
    """Open one synchronous pyserial port without making it an import-time need."""
    try:
        serial = importlib.import_module("serial")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "pyserial is required for USB GNSS capture; install project dependencies."
        ) from error
    return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)


def _print_summary(
    archive_path: Path,
    ubx_path: Path,
    table_path: Path,
    summary: UbxLiveCaptureSummary,
    progress: CaptureProgress,
) -> None:
    """Report capture integrity and PVT/TIMEGPS association facts."""
    projection = summary.projection
    print(f"  archive: {archive_path}")
    print(f"  standard UBX stream: {ubx_path}")
    print(f"  table:   {table_path}")
    print(f"  host elapsed: {summary.host_elapsed:.3f} s")
    print(f"  nonempty USB reads: {summary.nonempty_read_count}")
    print(f"  received USB bytes: {summary.received_byte_count}")
    print(f"  archived UBX frames: {summary.archived_frame_count}")
    print(f"  RXM-RAWX frames: {summary.rawx_frame_count}")
    print(f"  RXM-SFRBX frames: {summary.sfrbx_frame_count}")
    print(f"  NAV-PVT records: {projection.nav_pvt_count}")
    print(f"  NAV-TIMEGPS records: {projection.nav_timegps_count}")
    print(f"  normalized fixes: {len(projection.fixes)}")
    print(f"  PVT missing TIMEGPS: {projection.unpaired_pvt_count}")
    print(f"  PVT invalid TIMEGPS: {projection.invalid_time_pvt_count}")
    source = progress.snapshot().gnss_source
    print(
        "  recovery: "
        f"{source.recovery_count} interruptions across "
        f"{source.recording_session_count} sessions"
    )


def main(argv: list[str] | None = None) -> int:
    """Capture one finite serial UBX interval and materialize its artifacts.

    Args:
        argv: Optional command-line arguments. ``None`` uses process arguments.

    Returns:
        ``0`` for a nonempty archive with at least one normalized PVT/TIMEGPS
        fix, ``1`` for incomplete receiver evidence, or ``2`` for serial-start
        or archive I/O failure.

    Notes:
        The caller supplies distinct new paths for the native archive, derived
        standard UBX stream, and normalized table. The command reports recovery
        and association facts but does not convert host receipt time into GNSS
        measurement time.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="USB serial device path")
    parser.add_argument(
        "--archive-output",
        required=True,
        type=Path,
        help="new timestamped native UBX archive (.ubxlog)",
    )
    parser.add_argument(
        "--table-output",
        required=True,
        type=Path,
        help="new normalized GNSS fix table (.csv)",
    )
    parser.add_argument(
        "--ubx-output",
        required=True,
        type=Path,
        help="new standard concatenated UBX stream for external tools (.ubx)",
    )
    parser.add_argument("--duration", type=_positive_float, default=60.0)
    parser.add_argument("--baudrate", type=_positive_int, default=115200)
    parser.add_argument("--read-size", type=_positive_int, default=4096)
    parser.add_argument("--read-timeout", type=_positive_float, default=0.1)
    parser.add_argument("--drain-timeout", type=_nonnegative_float, default=2.0)
    parser.add_argument("--reconnect-initial-delay", type=_positive_float, default=1.0)
    parser.add_argument("--reconnect-max-delay", type=_positive_float, default=30.0)
    args = parser.parse_args(argv)
    if args.reconnect_max_delay < args.reconnect_initial_delay:
        parser.error(
            "--reconnect-max-delay must be greater than or equal to "
            "--reconnect-initial-delay"
        )

    archive_path = args.archive_output
    table_path = args.table_output
    ubx_path = args.ubx_output
    if len({path.resolve() for path in (archive_path, ubx_path, table_path)}) != 3:
        parser.error("--archive-output, --ubx-output, and --table-output must differ.")
    for output_path in (archive_path, ubx_path, table_path):
        if output_path.exists():
            parser.error(f"refusing to overwrite existing artifact: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        progress = CaptureProgress()
        summary = capture_ubx_resilient_stream(
            lambda: _open_serial(args.port, args.baudrate, args.read_timeout),
            archive_path,
            table_path,
            ubx_path=ubx_path,
            duration=args.duration,
            read_size=args.read_size,
            drain_timeout=args.drain_timeout,
            reconnect_initial_delay=args.reconnect_initial_delay,
            reconnect_max_delay=args.reconnect_max_delay,
            progress=progress,
        )
    except RuntimeError as error:
        print(f"Could not start USB GNSS capture: {error}")
        return 2
    except OSError as error:
        print(f"USB GNSS I/O failure: {error}")
        return 2
    except ValueError as error:
        print(f"INCOMPLETE: {error}")
        return 1

    if summary.archived_frame_count == 0:
        print("INCOMPLETE: no complete UBX frames were received.")
        _print_summary(archive_path, ubx_path, table_path, summary, progress)
        return 1
    if not summary.projection.fixes:
        print("INCOMPLETE: no PVT/TIMEGPS pair produced a normalized fix.")
        _print_summary(archive_path, ubx_path, table_path, summary, progress)
        return 1

    suffix = " [recovered outages]" if progress.snapshot().gnss_source.recovery_count else ""
    print(f"COMPLETE{suffix}")
    _print_summary(archive_path, ubx_path, table_path, summary, progress)
    if (
        summary.projection.unpaired_pvt_count
        or summary.projection.invalid_time_pvt_count
    ):
        print("WARN: some PVT records lacked usable TIMEGPS evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
