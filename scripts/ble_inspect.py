#!/usr/bin/env python3
"""Inspect a connected BLE GATT table against locally authorized capabilities.

The utility connects to a specified or locally configured peer, prints every
discoverable service and characteristic, and compares selected configured
service/data/control characteristics with their available properties. It checks
that the expected data characteristic supports notification or indication and
that the control characteristic supports read or write access.

Run ``python scripts/ble_inspect.py [ADDRESS]`` on a host with Bluetooth access.
When omitted, ``ADDRESS`` comes from ignored ``config/device_local.py``. Connected
GATT discovery is stronger evidence than an advertisement, but it does not prove
that private payload decoding, stream continuity, permissions beyond this session,
or device behavior are correct.
"""
import argparse
import asyncio
from collections.abc import Sequence
import importlib.util
from pathlib import Path

from bleak import BleakClient
from bleak.exc import BleakError

CONNECT_TIMEOUT = 20.0  # BLE connect can be slow; give it headroom over the scan.


def _load_device_config():
    """Load ignored local capability configuration without tracking device values."""
    path = Path(__file__).resolve().parents[1] / "config" / "device_local.py"
    if not path.exists():
        raise SystemExit(
            "Missing config/device_local.py.\n"
            "  cp config/device_example.py config/device_local.py\n"
            "then fill in your device's UUIDs (and optionally DEVICE_ADDRESS)."
        )
    spec = importlib.util.spec_from_file_location("device_local", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the optional explicit BLE peer address."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "address",
        nargs="?",
        help="BLE peer address; defaults to DEVICE_ADDRESS in config/device_local.py",
    )
    return parser.parse_args(argv)


def _resolve_address(address: str | None, config: object) -> str:
    """Return the explicit CLI address or configured fallback target."""
    address = address or getattr(config, "DEVICE_ADDRESS", "")
    if not address:
        raise SystemExit(
            "No address. Either pass one:\n"
            "    python scripts/ble_inspect.py 80:XX:XX:XX:XX:XX\n"
            "or set DEVICE_ADDRESS in config/device_local.py to the module's address."
        )
    return address


async def main(argv: Sequence[str] | None = None) -> None:
    """Connect, print the discoverable GATT table, and report capability checks.

    Service and characteristic identifiers are normalized for comparison because
    backend string casing may vary. Enumeration precedes the compatibility
    verdict so the observed connected-device evidence remains visible. The
    connection context owns the inspection because discovered services are valid
    only while the client remains connected.
    """
    args = _parse_args(argv)
    config = _load_device_config()
    address = _resolve_address(args.address, config)
    print(f"Connecting to {address} (timeout {CONNECT_TIMEOUT:.0f}s)...")
    
    try:
        async with BleakClient(address, timeout=CONNECT_TIMEOUT) as client:
            # enumerate the full GATT table (ground truth)
            service_uuids = set()
            char_props = {}  # uuid(lowercased) -> properties (list[str])
            for service in client.services:
                service_uuids.add(service.uuid.lower())
                print(f"Service: {service.uuid}  ({service.description})")
                for char in service.characteristics:
                    char_props[char.uuid.lower()] = char.properties
                    print(f"    Char: {char.uuid}  {char.properties}  "
                          f"handle={char.handle}  ({char.description})")

            # verify against config (absent / wrong-props / good)
            # Each check records a bool AND prints its own line
            print()
            ok_service = config.SERVICE_UUID.lower() in service_uuids
            if ok_service:
                print("PASS: service present")
            else:
                print("FAIL: service NOT present -- check SERVICE_UUID in device_local.py")

            imu = char_props.get(config.IMU_DATA_UUID.lower())
            ok_imu = bool(imu) and ("notify" in imu or "indicate" in imu)
            if ok_imu:
                print("PASS: IMU data char present and can notify")
            elif imu:
                print(f"FAIL: IMU data char present but cannot notify (has: {imu})")
            else:
                print("FAIL: IMU data char NOT present -- check IMU_DATA_UUID")

            ctrl = char_props.get(config.CONTROL_UUID.lower())
            ok_ctrl = bool(ctrl) and ("read" in ctrl or "write" in ctrl)
            if ok_ctrl:
                print("PASS: control char present and read/write")
            elif ctrl:
                print(f"FAIL: control char present but not read/write (has: {ctrl})")
            else:
                print("FAIL: control char NOT present -- check CONTROL_UUID")

            # one verdict on top of the three checks
            print()
            if ok_service and ok_imu and ok_ctrl:
                print("READY: required GATT service and characteristics are present. Subscribe to")
                print(f"       the IMU char ({config.IMU_DATA_UUID}) to receive notification bytes.")
            else:
                print("NOT READY: see the FAIL line(s) above. A missing service/char means a")
                print("           wrong UUID in device_local.py (or different firmware); an IMU")
                print("           char that can't notify cannot support the required notification stream.")

    except (BleakError, asyncio.TimeoutError) as e:
        # Only connection-class failures land here; logic bugs surface as real
        # tracebacks instead of being misreported as a radio problem.
        print(f"Connect failed: {e}")
        print("  - is the device connected to another host (phone/Pi)?")
        print("  - is it in range / powered?")
        print("  - does it need pairing?")


if __name__ == "__main__":
    asyncio.run(main())
