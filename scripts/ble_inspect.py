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

Run on the Linux box that saw the device in Rung 1:
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
    """Connect to the module and inspect its GATT table.

    Implement these steps:

    1. CONNECT. Use BleakClient as an async context manager so disconnect is
       automatic even on error:
           async with BleakClient(address, timeout=CONNECT_TIMEOUT) as client:
               ...
       Inside the block, `client.is_connected` should be True. If the connect
       itself raises (BleakError / TimeoutError / asyncio.TimeoutError), catch it
       and print a hint -- most common causes: the device is connected to another
       host (phone/Pi), out of range, or needs pairing.

    2. ENUMERATE the GATT table. After connect, bleak has already done service
       discovery; the result is `client.services` (a collection you can iterate).
       For each service, then each characteristic under it, print a readable tree:
           service.uuid, service.description
             char.uuid, char.properties (a list[str] like ['read','notify']),
             char.handle, char.description
       This is the ground truth of what the device offers -- print ALL of it, not
       just the chars you expect, so anything surprising is visible.

    3. VERIFY against config. Build a lookup of {char_uuid(lowercased): properties}
       across all services, and the set of service UUIDs. Then check, printing a
       clear PASS/FAIL line for each:
         (a) cfg.SERVICE_UUID  is among the service UUIDs.
         (b) cfg.IMU_DATA_UUID is present AND its properties include 'notify'
             (some devices use 'indicate' instead -- accept either; that's the
             channel the 100 Hz stream will arrive on in Rung 3).
         (c) cfg.CONTROL_UUID  is present AND its properties include 'read' and/or
             'write'.
       Lowercase both sides before comparing -- bleak's casing isn't guaranteed.

    4. SUMMARIZE. If all three PASS, print that the device matches the ICD and is
       ready for Rung 3 (subscribe to the IMU char + log raw bytes). If any FAIL,
       say which -- a missing service/char means a wrong UUID in device_local.py or
       a different firmware than expected; a present char missing 'notify' means our
       streaming assumption is wrong and we rethink Rung 3.

    Tip: keep the raw enumeration (step 2) and the judgement (step 3) separate --
    print the full table first, THEN the checks. When something is off, you want to
    eyeball reality before trusting the pass/fail logic, same discipline as Rung 1.
    """
    address = _resolve_address()
    print(f"Connecting to {address} (timeout {CONNECT_TIMEOUT:.0f}s)...")
    
    try:
        async with BleakClient(address, timeout=CONNECT_TIMEOUT) as client:
            # --- Step 2: enumerate the full GATT table (ground truth) ---
            service_uuids = set()
            char_props = {}  # uuid(lowercased) -> properties (list[str])
            for service in client.services:
                service_uuids.add(service.uuid.lower())
                print(f"Service: {service.uuid}  ({service.description})")
                for char in service.characteristics:
                    char_props[char.uuid.lower()] = char.properties
                    print(f"    Char: {char.uuid}  {char.properties}  "
                          f"handle={char.handle}  ({char.description})")

            # --- Step 3: verify against config (absent / wrong-props / good) ---
            # Each check records a bool AND prints its own line, so the summary in
            # step 4 can give one verdict without re-deriving any of the logic.
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

            # --- Step 4: one verdict on top of the three checks ---
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
