#!/usr/bin/env python3
"""
ble_scan.py - Rung 1 of device bring-up: can we even SEE the IMU module?

The absolute minimum: ask the OS Bluetooth stack to listen for BLE advertisements
for a few seconds and print what it hears. No connection, no GATT walk, no parsing.
If the module shows up here, everything below us works (adapter, BlueZ, bleak,
permissions, the module's battery + range), and we can move on to connecting.

The module advertises a custom 128-bit service UUID. That UUID is device-specific
and lives in config/device_local.py (gitignored) -- see config/device.example.py.
We use it as a fingerprint to pick the module out of all the other BLE noise nearby.

Run on a Linux box with a Bluetooth adapter (your Ubuntu machine; later the Pi):
    pip install bleak
    cp config/device.example.py config/device_local.py   # then fill in the UUID
    python scripts/ble_scan.py

SUCCESS = the module's address prints with a service-UUID match.
"""
import asyncio
import importlib.util
from pathlib import Path

from bleak import BleakScanner

SCAN_SECONDS = 10.0


def _load_device_config():
    """Load config/device_local.py (gitignored) so device-specific UUIDs never
    enter tracked source. Exits with a helpful message if it hasn't been created."""
    path = Path(__file__).resolve().parents[1] / "config" / "device_local.py"
    if not path.exists():
        raise SystemExit(
            "Missing config/device_local.py.\n"
            "  cp config/device.example.py config/device_local.py\n"
            "then fill in your device's UUIDs from the ICD."
        )
    spec = importlib.util.spec_from_file_location("device_local", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Service UUID is loaded at runtime from the gitignored local config.
SERVICE_UUID = _load_device_config().SERVICE_UUID.lower()


async def main() -> None:
    """Scan for BLE advertisements, print every device heard, and flag any that
    advertise SERVICE_UUID.

    Implement these steps:

    1. Run a scan for SCAN_SECONDS. The call you want is:
           found = await BleakScanner.discover(timeout=SCAN_SECONDS, return_adv=True)
       With return_adv=True this returns a dict:
           { address(str): (device: BLEDevice, adv: AdvertisementData) }
       From each pair you can read:
           device.address              -> the address string
           adv.rssi                    -> signal strength (dBm; closer to 0 = stronger)
           adv.local_name              -> advertised name (may be None)
           adv.service_uuids           -> list[str] of advertised UUIDs (may be empty)

    2. If `found` is empty, the adapter likely isn't up. Print a hint
       (e.g. "bluetoothctl power on") and return.

    3. For each device, decide is_match = (SERVICE_UUID is in the advertised UUIDs,
       lowercased). Print one line per device: address, rssi, name, and a marker
       when it matches.

    4. After the loop, summarize: if any matched, print their address(es) for Rung 2
       (connect). If none matched, print the troubleshooting checks (powered? in
       range? already connected elsewhere? UUID not advertised?).

    Note: some peripherals don't advertise their service UUIDs at all - if so, step 3
    won't flag it, but you may still recognize it by name or by a strong RSSI that
    appears only when the module is near. We confirm by connecting in Rung 2 either way.
    """
    print(f"Scanning {SCAN_SECONDS:.0f}s for BLE advertisements...")
    print(f"Fingerprint (service UUID): {SERVICE_UUID}\n")

    found = await BleakScanner.discover(timeout=SCAN_SECONDS, return_adv=True)

    # Empty dict => the loop never heard a single advertisement, which almost always
    # means the adapter itself isn't usable, not that the room is empty.
    if not found:
        print("No BLE devices heard at all -- the adapter is likely down or blocked.")
        print("  - Powered?    bluetoothctl power on")
        print("  - Blocked?    rfkill list   (then: rfkill unblock bluetooth)")
        print("  - Permission? run as a user in the 'bluetooth' group (or via sudo)")
        return

    # By the time await returned, every advertisement the callbacks collected is here.
    # Print ONE line per device so a module that doesn't advertise its UUID is still
    # visible (recognizable by name or by an RSSI that strengthens as it nears you).
    matches = []
    print(f"Heard {len(found)} device(s):")
    for address, (_device, adv) in found.items():
        advertised = [u.lower() for u in adv.service_uuids]
        is_match = SERVICE_UUID in advertised
        name = adv.local_name or "(no name)"
        marker = "  <== MATCH: advertises our service UUID" if is_match else ""
        print(f"  {address}  rssi={adv.rssi:>4} dBm  {name}{marker}")
        if is_match:
            matches.append(address)

    print()
    if matches:
        print(f"SUCCESS -- {len(matches)} matching device(s). For Rung 2 (connect), use:")
        for addr in matches:
            print(f"  {addr}")
    else:
        print("No device advertised our service UUID. Checks, in order:")
        print("  - Module powered and in range? Look for a strong (less negative) RSSI above.")
        print("  - Already connected to a phone/Pi? A BLE peripheral talks to one host at a")
        print("    time -- disconnect it elsewhere, then rescan.")
        print("  - Not advertising its UUID? If a plausible name / strong RSSI appears above,")
        print("    note that address -- we'll confirm it by connecting in Rung 2.")
        print("  - SERVICE_UUID in config/device_local.py actually matches the ICD?")


if __name__ == "__main__":
    asyncio.run(main())
