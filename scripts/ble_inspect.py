#!/usr/bin/env python3
"""
ble_inspect.py - Rung 2 of device bring-up: connect and walk the GATT table.

Rung 1 (ble_scan.py) proved the module is on the air and the stack works. It did
NOT prove the device matches the ICD -- BLE advertisements often omit the service
UUID to fit the 31-byte packet, so "not advertised" told us nothing about what the
device actually exposes. The authoritative service list lives in the device's GATT
table, which we can only read by CONNECTING. That is this rung.

Goal: connect, enumerate every service/characteristic, and VERIFY three things
against config/device_local.py before we write a single byte of parsing code:
  1. the service              (cfg.SERVICE_UUID)   exists on the device,
  2. the IMU data char        (cfg.IMU_DATA_UUID)  exists AND can `notify`,
  3. the control char         (cfg.CONTROL_UUID)   exists AND can read/write.
Output of a good run = the exact characteristics Rung 3 (subscribe + log) will use,
and confidence that device_local.py is correct.

Run on a Linux machine that saw the device in Rung 1:
    python scripts/ble_inspect.py [ADDRESS]
ADDRESS is optional: it defaults to cfg.DEVICE_ADDRESS, and the CLI arg overrides it.
(Use the address Rung 1 printed for the module, e.g. the one named like your unit.)

SUCCESS = all three checks PASS.
"""
import asyncio
import importlib.util
import sys
from pathlib import Path

from bleak import BleakClient
from bleak.exc import BleakError

CONNECT_TIMEOUT = 20.0  # BLE connect can be slow; give it headroom over the scan.


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
            "    python scripts/ble_inspect.py 80:XX:XX:XX:XX:XX\n"
            "or set DEVICE_ADDRESS in config/device_local.py to the module's address."
        )
    return address


async def main() -> None:
    """Connect to the module, enumerate its full GATT table, and verify the three
    characteristics we depend on against config: the service exists, the IMU-data
    char can notify (or indicate), and the control char is read/write. The raw
    enumeration is printed in full before any pass/fail judgement so reality is
    visible first -- when a check fails, you want to eyeball the table before
    trusting the logic. UUID comparisons are lowercased on both sides (bleak's
    casing isn't guaranteed), and the connection context owns the verification
    because `client.services` is only valid while connected.
    """
    address = _resolve_address()
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
            ok_service = CFG.SERVICE_UUID.lower() in service_uuids
            if ok_service:
                print("PASS: service present")
            else:
                print("FAIL: service NOT present -- check SERVICE_UUID in device_local.py")

            imu = char_props.get(CFG.IMU_DATA_UUID.lower())
            ok_imu = bool(imu) and ("notify" in imu or "indicate" in imu)
            if ok_imu:
                print("PASS: IMU data char present and can notify")
            elif imu:
                print(f"FAIL: IMU data char present but cannot notify (has: {imu})")
            else:
                print("FAIL: IMU data char NOT present -- check IMU_DATA_UUID")

            ctrl = char_props.get(CFG.CONTROL_UUID.lower())
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
                print("READY: device matches the ICD. Next is Rung 3 -- subscribe to the")
                print(f"       IMU char ({CFG.IMU_DATA_UUID}) and log raw bytes.")
            else:
                print("NOT READY: see the FAIL line(s) above. A missing service/char means a")
                print("           wrong UUID in device_local.py (or different firmware); an IMU")
                print("           char that can't notify breaks the Rung 3 streaming assumption.")

    except (BleakError, asyncio.TimeoutError) as e:
        # Only connection-class failures land here; logic bugs surface as real
        # tracebacks instead of being misreported as a radio problem.
        print(f"Connect failed: {e}")
        print("  - is the device connected to another host (phone/Pi)?")
        print("  - is it in range / powered?")
        print("  - does it need pairing?")


if __name__ == "__main__":
    asyncio.run(main())
