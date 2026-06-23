#!/usr/bin/env python3
"""
analyze_rest.py - read a static-rest .bmu capture and report the mean specific
force, so we can map the BMU body axes (x/y/z) to physical up/down + handedness.

This closes M1 Lesson 1: with the boot at rest on one face, the accelerometer
measures specific force f = -g, which points UP with magnitude g. The VN-100
hands us per-sample velocity increments dv ~= f*dt, so the Dt-weighted average

    f_hat = (sum dv) / (sum dt)        [m/s^2]

recovers f. Its magnitude should be ~9.81 (a layout-validation gate); the body
axis carrying the dominant component is the vertical one, and the SIGN tells us
whether that axis points up (+) or down (-) in this pose.

The cross-pose interpretation (which body axis -> which physical direction, and
right- vs left-handedness) is done by hand afterward, since only the operator's
notebook knows which physical face was down for each pose label.

Usage (run where the .bmu files live, with `bootins` importable):
    python scripts/analyze_rest.py data/*.bmu
    python scripts/analyze_rest.py data/20260623_141530_poseA.bmu [more.bmu ...]
"""
import argparse
import struct
from pathlib import Path

import numpy as np

from bootins.bmu import decode

# Capture-layer framing, the exact inverse of ble_capture.py's writer:
#     <f64 arrival_monotonic_s> <u16 payload_len> <payload_len raw bytes>
# (also recorded as `record_format` in each .meta.yaml sidecar). Must stay in
# sync with the writer; when we factor out the config/IO layer this should move
# into a shared module.
FRAME_FORMAT = "<dH"
FRAME_SIZE = struct.calcsize(FRAME_FORMAT)  # 10

AXES = "xyz"

# How much to trim off each end of the window (seconds) to drop the transient
# from placing the boot down and picking it back up.
TRIM_SECONDS = 1.0

# Local gravity for the magnitude sanity check (m/s^2). Good to ~0.05 anywhere
# on Earth; refine to the local value if we want a tighter gate.
G_NOMINAL = 9.81


def iter_payloads(path: Path):
    """Yield each raw notification payload from a framed .bmu file, in order.

    Reverses the capture framing: read the fixed 10-byte frame header, unpack the
    payload length, then read exactly that many payload bytes. A short read at the
    very end means the last record was truncated (capture killed mid-write); we
    stop rather than yield a partial payload.
    """
    with open(path, "rb") as f:
        while True:
            head = f.read(FRAME_SIZE)
            if len(head) < FRAME_SIZE:
                break  # clean EOF, or a truncated trailing header
            _arrival, payload_len = struct.unpack(FRAME_FORMAT, head)
            payload = f.read(payload_len)
            if len(payload) < payload_len:
                break  # truncated trailing payload
            yield payload


def load_measurements(path: Path) -> list:
    """Decode every payload in a .bmu into a flat list of Measurements (each
    notification carries NumMeas of them)."""
    measurements = []
    for payload in iter_payloads(path):
        measurements.extend(decode.decode_message(payload))
    return measurements


def analyze(measurements: list, trim_s: float = TRIM_SECONDS) -> dict:
    """Compute the rest-state summary for one capture.

    Stacks the measurements, trims `trim_s` off each end (the place-down / pick-up
    transients) by selecting on cumulative time, then computes the Dt-weighted mean
    specific force f_hat = sum(dv)/sum(dt): its magnitude should be ~g, and its
    dominant axis + sign give the vertical body axis and which way it points. Also
    returns soft diagnostics -- per-axis specific-force spread, mean gyro rate
    (~0 only if truly still), and mean |q| (~1 regardless of motion).

    Returns a dict consumed by _report:
        n_used, n_total, f_hat, f_norm, dom_axis, dom_sign,
        dv_std, gyro_rate, gyro_rate_norm, quat_mean_norm
    """
    dt = np.array([m.dt for m in measurements])
    dv = np.vstack([m.dv for m in measurements])
    dtheta = np.vstack([m.dtheta for m in measurements])
    quat = np.vstack([m.quat_raw for m in measurements])

    # Select the middle of the window by cumulative time, dropping the place-down
    # and pick-up transients. A boolean mask indexes by *which* samples to keep.
    t = np.cumsum(dt)
    mask = (t >= trim_s) & (t <= t[-1] - trim_s)
    if mask.sum() < 10:                       # too few survived -> keep all
        mask = np.ones_like(t, dtype=bool)
    dt, dv, dtheta, quat = dt[mask], dv[mask], dtheta[mask], quat[mask]

    # The heart: Dt-weighted mean specific force. At rest f = -g, so |f_hat| ~= g
    # and the dominant axis (with its sign) is the vertical body axis.
    f_hat = np.sum(dv, axis=0) / np.sum(dt)
    f_norm = np.linalg.norm(f_hat)
    dom_axis = int(np.argmax(np.abs(f_hat)))
    dom_sign = int(np.sign(f_hat[dom_axis]))

    # Soft diagnostics (motion-sensitive, not gates): per-axis specific-force
    # spread, mean gyro rate (~0 only if still), and mean |q| (~1 either way).
    f_i = dv / dt[:, None]
    dv_std = np.std(f_i, axis=0)
    gyro_rate = np.sum(dtheta, axis=0) / np.sum(dt)
    gyro_rate_norm = np.linalg.norm(gyro_rate)
    quat_mean_norm = np.mean(np.linalg.norm(quat, axis=1))

    return {
        "n_used": dt.shape[0],
        "n_total": t.shape[0],
        "f_hat": f_hat,
        "f_norm": f_norm,
        "dom_axis": dom_axis,
        "dom_sign": dom_sign,
        "dv_std": dv_std,
        "gyro_rate": gyro_rate,
        "gyro_rate_norm": gyro_rate_norm,
        "quat_mean_norm": quat_mean_norm
    }
    


def _report(path: Path, result: dict) -> None:
    """Print a one-pose summary: the specific-force vector, the up/down verdict
    for the dominant body axis, and the soft diagnostics."""
    f_hat = result["f_hat"]
    axis = AXES[result["dom_axis"]]
    direction = "UP" if result["dom_sign"] > 0 else "DOWN"
    f_err = result["f_norm"] - G_NOMINAL

    print(f"\n=== {path.name} ===")
    print(f"  samples used      : {result['n_used']} / {result['n_total']} "
          f"(trimmed {TRIM_SECONDS:.0f}s each end)")
    print(f"  f_hat [m/s^2]     : "
          f"[{f_hat[0]:+8.4f} {f_hat[1]:+8.4f} {f_hat[2]:+8.4f}]")
    print(f"  |f_hat|           : {result['f_norm']:.4f}  "
          f"(expected ~{G_NOMINAL:.2f}, off by {f_err:+.4f})")
    print(f"  -> dominant axis  : body {axis} points {direction} "
          f"(sign {result['dom_sign']:+d})")
    print(f"  stillness std     : "
          f"[{result['dv_std'][0]:.4f} {result['dv_std'][1]:.4f} "
          f"{result['dv_std'][2]:.4f}] m/s^2  (want << {G_NOMINAL:.0f})")
    print(f"  gyro rate norm    : {result['gyro_rate_norm']:.5f} rad/s  (want ~0)")
    print(f"  mean |q|          : {result['quat_mean_norm']:.5f}  (want ~1)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze static-rest .bmu captures for the 6-face body-frame "
                    "characterization (mean specific force per pose).",
    )
    parser.add_argument("files", nargs="+", type=Path, help="one or more .bmu files")
    args = parser.parse_args()

    for path in args.files:
        measurements = load_measurements(path)
        if not measurements:
            print(f"\n=== {path.name} ===\n  (no measurements decoded -- empty or corrupt)")
            continue
        result = analyze(measurements)
        _report(path, result)


if __name__ == "__main__":
    main()
