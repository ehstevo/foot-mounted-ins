#!/usr/bin/env python3
"""Scan BLE advertisements and identify candidates matching local configuration.

The utility asks the host Bluetooth stack to collect advertisements for a fixed
interval and prints device address, received signal strength, local name, and
advertised service identifiers. It compares advertised services to the
device-specific value loaded at runtime from ignored ``config/device_local.py``;
create that local file from tracked ``config/device_example.py``.

Run on a host with a usable Bluetooth adapter and the required permissions:

    python scripts/ble_scan.py

An advertised-service match identifies a discovery candidate, not proof of GATT
service availability, connection permission, payload compatibility, device
health, or a continuous radio link. Conversely, absence from an advertisement
does not prove a service is absent from the connected device.
"""
import argparse
import asyncio
from collections.abc import Sequence
import importlib.util
from pathlib import Path

from bleak import BleakScanner

SCAN_SECONDS = 10.0


def _load_device_config():
    """Load ignored local discovery configuration without tracking device values."""
    path = Path(__file__).resolve().parents[1] / "config" / "device_local.py"
    if not path.exists():
        raise SystemExit(
            "Missing config/device_local.py.\n"
            "  cp config/device_example.py config/device_local.py\n"
            "then fill in your device's UUIDs from the ICD."
        )
    spec = importlib.util.spec_from_file_location("device_local", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the utility's argument-free command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> None:
    """Scan for ``SCAN_SECONDS`` and report every received advertisement.

    The printed match marker compares a locally configured service identifier to
    service identifiers present in each advertisement. An empty result is an
    adapter or host-environment diagnostic, not proof that no peripherals exist.
    A nonmatch remains inconclusive because peripherals may omit services from
    advertisements; connected GATT inspection provides separate evidence.
    """
    _parse_args(argv)
    service_uuid = _load_device_config().SERVICE_UUID.lower()
    print(f"Scanning {SCAN_SECONDS:.0f}s for BLE advertisements...")
    print(f"Fingerprint (service UUID): {service_uuid}\n")

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
        is_match = service_uuid in advertised
        name = adv.local_name or "(no name)"
        marker = "  <== MATCH: advertises our service UUID" if is_match else ""
        print(f"  {address}  rssi={adv.rssi:>4} dBm  {name}{marker}")
        if is_match:
            matches.append(address)

    print()
    if matches:
        print(f"SUCCESS -- {len(matches)} matching device(s). Inspect connected GATT with:")
        for addr in matches:
            print(f"  {addr}")
    else:
        print("No device advertised our service UUID. Checks, in order:")
        print("  - Module powered and in range? Look for a strong (less negative) RSSI above.")
        print("  - Already connected to a phone/Pi? A BLE peripheral talks to one host at a")
        print("    time -- disconnect it elsewhere, then rescan.")
        print("  - Not advertising its UUID? If a plausible name / strong RSSI appears above,")
        print("    note that address -- connected-device GATT inspection can confirm it.")
        print("  - SERVICE_UUID in config/device_local.py actually matches the ICD?")


if __name__ == "__main__":
    asyncio.run(main())
