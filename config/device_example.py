"""
device.example.py -- TEMPLATE for the device interface constants (safe to commit).

This project talks to a sealed BLE IMU module whose wire-level interface is defined
by a vendor ICD that is not redistributable. The concrete identifiers therefore live
ONLY in a local, gitignored copy of this file.

Setup:
    cp config/device.example.py config/device_local.py
    # then fill device_local.py in with the real values from your device's ICD

config/device_local.py is gitignored and must never be committed.
"""

# Custom GATT service + characteristics (128-bit UUIDs, lowercased for bleak).
SERVICE_UUID = "00000000-0000-0000-0000-000000000000"   # data + control service
IMU_DATA_UUID = "00000000-0000-0000-0000-000000000000"  # IMU data (notify)
CONTROL_UUID = "00000000-0000-0000-0000-000000000000"   # status / control (r/w)

# Pin a specific unit so scans don't grab the wrong one. "" = match by service UUID.
DEVICE_ADDRESS = ""
