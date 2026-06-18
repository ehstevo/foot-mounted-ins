#!/usr/bin/env python3
"""
ble_capture.py - Rung 3 of device bring-up: subscribe to the IMU stream and log
raw bytes to a replayable capture file.

Rung 2 (ble_inspect.py) proved the device matches the ICD and that the IMU-data
characteristic can notify. This rung opens that firehose for real: subscribe to
notifications and write every payload to disk, faithfully and ICD-agnostically,
so the capture can be replayed later through the parser without this script ever
needing to understand a single byte of the message.

Design:
  * The logger is DUMB. It never parses the payload. Each notification is stored
    as a self-delimiting capture-layer record:
        <f64 arrival_monotonic_s> <u16 payload_len> <payload_len raw bytes>
    Message boundaries are recoverable on replay WITHOUT parsing the ICD header
    (PayLen), which keeps the raw log independent of parser correctness.
  * Arrival time is time.monotonic() -- for diagnostics only. We NEVER integrate
    on arrival time (jitter); mechanization integrates on the payload's own dt.
    The run's wall-clock start goes once into the .meta.yaml sidecar to anchor it.
  * Sanity check is RATE only (count / elapsed vs the expected ~25 notif/s).
    Sequence-number / gap detection needs the header layout (ICD) and belongs to
    the parser, not here.

Output (under data/, which is gitignored):
    data/<timestamp>.bmu        framed raw capture
    data/<timestamp>.meta.yaml  run context (address, start wall-clock, counts)

Run on a Linux machine that passed Rung 2:
    python scripts/ble_capture.py [ADDRESS] [SECONDS]
ADDRESS defaults to cfg.DEVICE_ADDRESS; SECONDS defaults to CAPTURE_SECONDS.

SUCCESS = a .bmu file whose measured notification rate is ~25/s.
"""
import asyncio
import importlib.util
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from bleak import BleakClient
from bleak.exc import BleakError

CONNECT_TIMEOUT = 20.0   # match Rung 2: BLE connect can be slow.
CAPTURE_SECONDS = 10.0   # default capture window; override via argv[2].
EXPECTED_RATE_HZ = 25.0  # BLE notifications/s (each carries 4 IMU measurements).


# NOTE: this is the THIRD copy of _load_device_config (ble_scan, ble_inspect, here).
# Per the parser-design note, a third consumer is the trigger to factor it out into
# a shared helper (e.g. src/bootins/config.py). Flagged for the mentor; duplicated
# for now so this rung stays self-contained until we make that move deliberately.
def _load_device_config():
    """Load config/device_local.py (gitignored) so device-specific UUIDs never
    enter tracked source. Exits with a helpful message if it hasn't been created."""
    path = Path(__file__).resolve().parents[1] / "config" / "device_local.py"
    if not path.exists():
        raise SystemExit(
            "Missing config/device_local.py.\n"
            "  cp config/device.example.py config/device_local.py\n"
            "then fill in your device's UUIDs (and optionally DEVICE_ADDRESS)."
        )
    spec = importlib.util.spec_from_file_location("device_local", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = _load_device_config()


def _resolve_address() -> str:
    """CLI arg wins; otherwise fall back to cfg.DEVICE_ADDRESS. Fail loudly if
    neither is set -- connecting needs a concrete target."""
    address = sys.argv[1] if len(sys.argv) > 1 else getattr(CFG, "DEVICE_ADDRESS", "")
    if not address:
        raise SystemExit(
            "No address. Either pass one:\n"
            "    python scripts/ble_capture.py 80:XX:XX:XX:XX:XX\n"
            "or set DEVICE_ADDRESS in config/device_local.py to the module's address."
        )
    return address


def _resolve_seconds() -> float:
    """Optional argv[2] capture window; defaults to CAPTURE_SECONDS."""
    if len(sys.argv) > 2:
        return float(sys.argv[2])
    return CAPTURE_SECONDS


async def main() -> None:
    """Connect, subscribe to the IMU-data characteristic, stream every notification
    to a framed .bmu for `seconds`, write a .meta.yaml sidecar, and report the rate.

    The notification handler stays deliberately dumb -- it stamps arrival, frames the
    payload, writes it, and counts, but never parses the bytes. That keeps the capture
    independent of (and a check on) the parser. Arrival is time.monotonic(), used for
    diagnostics only: rate here, jitter later. Integration always uses the payload's
    own dt, never arrival time; the wall-clock start lives once in the sidecar just to
    anchor the run in real time. Connection-class failures are caught and explained;
    anything else is a logic bug and is left to raise.
    """
    # Step 1: resolve the invocation FIRST, so a bad address/seconds bails out
    # before we create an empty .bmu file. Anchor data/ at the repo root (like the
    # other scripts) so the capture lands in the same place regardless of cwd.
    address = _resolve_address()
    seconds = _resolve_seconds()

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start_wall = datetime.now()
    run_basename = start_wall.strftime("%Y%m%d_%H%M%S")
    bmu_path = data_dir / f"{run_basename}.bmu"
    meta_path = data_dir / f"{run_basename}.meta.yaml"

    print(f"Capturing from {address} for {seconds:.0f}s -> {bmu_path}")

    # Mutable holder so the nested handler can update counters without `nonlocal`
    # gymnastics. Track BYTES as well as count: bytes ~= 0 with count > 0 is the
    # signature of a handler that's failing silently inside bleak's callback.
    stats = {"count": 0, "bytes": 0}
    elapsed = 0.0

    # Step 2: the notification handler -- stamp, frame, write, count. It NEVER
    # parses `data`; the logger stays ICD-agnostic. Counters bump only AFTER a
    # successful write, so a throwing write can't inflate the rate.
    with open(bmu_path, "wb") as file:

        def notify_handler(_sender, data):
            t_arrival = time.monotonic()
            record = struct.pack("<dH", t_arrival, len(data)) + bytes(data)
            file.write(record)
            stats["count"] += 1
            stats["bytes"] += len(record)

        # Step 3: connect, subscribe, let it flow, unsubscribe. Catch only
        # connection-class failures; logic bugs surface as real tracebacks.
        try:
            async with BleakClient(address, timeout=CONNECT_TIMEOUT) as client:
                await client.start_notify(CFG.IMU_DATA_UUID, notify_handler)
                t0 = time.monotonic()
                await asyncio.sleep(seconds)
                await client.stop_notify(CFG.IMU_DATA_UUID)
                elapsed = time.monotonic() - t0
        except (BleakError, asyncio.TimeoutError) as e:
            print(f"Capture failed: {e}")
            print("  - did the device drop out of range / disconnect mid-capture?")
            print("  - is it connected to another host (phone/Pi)?")
            return

    # Step 4: write the sidecar, then report the rate.
    meta = {
        "device_address": address,
        "start_wall_clock": start_wall.isoformat(timespec="seconds"),
        "capture_seconds_requested": seconds,
        "capture_seconds_elapsed": round(elapsed, 3),
        "notification_count": stats["count"],
        "bytes_written": stats["bytes"],
        # How to decode the .bmu: repeat { read 10 bytes -> unpack "<dH" -> read
        # that many payload bytes }. Arrival time is monotonic, diagnostics only.
        "record_format": "<dH (f64 arrival_monotonic_s, u16 payload_len) then payload_len raw bytes",
    }
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    rate = stats["count"] / elapsed if elapsed > 0 else 0.0
    print()
    print(f"Captured {stats['count']} notifications, {stats['bytes']} bytes in {elapsed:.2f}s")
    print(f"Measured rate: {rate:.1f} notif/s (expected ~{EXPECTED_RATE_HZ:.0f})")
    if stats["count"] == 0:
        print("FAIL: zero notifications -- wrong IMU_DATA_UUID, or never subscribed.")
    elif abs(rate - EXPECTED_RATE_HZ) <= 0.1 * EXPECTED_RATE_HZ:
        print("PASS: rate within 10% of expected -- capture looks healthy.")
    else:
        print("WARN: rate off by >10% -- dropouts, weak signal, or not streaming at 25 Hz.")
    print(f"  raw : {bmu_path}")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    asyncio.run(main())
