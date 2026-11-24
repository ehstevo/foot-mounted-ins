#!/usr/bin/env python3
"""Diagnose stationary IMU captures for body-frame specific-force orientation.

At rest, an accelerometer observes specific force opposing gravity. For a known
stationary sensor pose, the dominant signed body-axis component therefore
identifies that pose's vertical axis. The recorded measurements are velocity
increments, so this diagnostic uses the duration-weighted mean

    f_mean = sum(dv) / sum(dt)    [m/s²]

where each ``dv`` is a body-frame specific-force increment. A magnitude near
local gravity supports the stationary-pose assumption, but it is not a complete
calibration or navigation validation. Trimming excludes placement and pickup
transients; mean gyro rate is a stillness diagnostic. Raw quaternion norms are
reported only as a structural check because their component ordering is not
interpreted here.

A single static pose cannot establish every body-axis direction, handedness, or
sensor mounting convention. Those conclusions require independently known poses
and cross-pose interpretation. This is a narrow diagnostic, not a canonical
navigation input path. Its legacy reader silently omits a truncated trailing
record; unlike the strict general transport reader, it does not report that
truncation as a capture-integrity failure.

Examples:
    python scripts/analyze_rest.py data/*.bmu
    python scripts/analyze_rest.py data/rest_pose_a.bmu [more.bmu ...]
"""
import argparse
import struct
from pathlib import Path

import numpy as np

from bootins.bmu import decode

# Legacy standalone BLE-capture framing, inverse to ``capture_ble.py``:
#     <f64 arrival_monotonic_s> <u16 payload_len> <payload_len raw bytes>
# The sidecar records this format as ``record_format``. This diagnostic preserves
# its historical truncated-tail behavior rather than using the strict reader.
FRAME_FORMAT = "<dH"
FRAME_SIZE = struct.calcsize(FRAME_FORMAT)  # 10

AXES = "xyz"

# Time trimmed from each window end to remove placement and pickup transients.
TRIM_SECONDS = 1.0

# Reference gravity magnitude for contextual reporting, in m/s².
G_NOMINAL = 9.81


def iter_payloads(path: Path):
    """Yield complete legacy-capture notification payloads in file order.

    Args:
        path: Framed standalone ``.bmu`` capture to read.

    Yields:
        Complete raw BLE notification payloads without host-arrival timestamps.

    Notes:
        A short trailing header or payload ends iteration without yielding a
        partial notification. This lenient tail handling is specific to this
        diagnostic and is not equivalent to the strict transport reader.
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
    """Decode one legacy capture into its flat measurement sequence.

    Args:
        path: Framed standalone ``.bmu`` capture to decode.

    Returns:
        Measurements from all complete notification payloads, in file order.
    """
    measurements = []
    for payload in iter_payloads(path):
        measurements.extend(decode.decode_message(payload).measurements)
    return measurements


def analyze(measurements: list, trim: float = TRIM_SECONDS) -> dict:
    """Summarize duration-weighted stationary specific force in body axes.

    Args:
        measurements: Decoded increment measurements from one intended-static
            pose, ordered by their payload integration intervals.
        trim: Time removed from both ends of the cumulative-``dt`` window (s).
            The full window is retained when fewer than ten samples would
            remain.

    Returns:
        A report mapping containing sample counts; mean body-frame specific
        force and its magnitude in m/s²; dominant signed body-axis index; the
        per-sample specific-force spread in m/s²; mean angular rate in rad/s;
        and mean raw-quaternion norm.

    Notes:
        The mean is ``sum(dv) / sum(dt)``. Its gravity-scale magnitude and
        dominant axis are informative only when the selected pose is stationary
        and independently known. Quaternion component order is not inferred.
    """
    dt = np.array([m.dt for m in measurements])
    dv = np.vstack([m.dv for m in measurements])
    dtheta = np.vstack([m.dtheta for m in measurements])
    quat = np.vstack([m.quat_raw for m in measurements])

    # Cumulative-time trimming removes placement and pickup transients.
    t = np.cumsum(dt)
    mask = (t >= trim) & (t <= t[-1] - trim)
    if mask.sum() < 10:                       # too few survived -> keep all
        mask = np.ones_like(t, dtype=bool)
    dt, dv, dtheta, quat = dt[mask], dv[mask], dtheta[mask], quat[mask]

    # At rest f = -g, so gravity-scale magnitude and sign identify this pose's
    # vertical body axis.
    f_hat = np.sum(dv, axis=0) / np.sum(dt)
    f_norm = np.linalg.norm(f_hat)
    dom_axis = int(np.argmax(np.abs(f_hat)))
    dom_sign = int(np.sign(f_hat[dom_axis]))

    # Motion-sensitive diagnostics, not acceptance gates.
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
    """Print the stationary-pose summary and non-gating diagnostics.

    Args:
        path: Capture path used as the report label.
        result: Mapping returned by :func:`analyze`.
    """
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
    """Analyze each requested legacy ``.bmu`` capture and print its summary.

    The command exits through ``argparse`` on invalid arguments. For each input,
    an empty decoded measurement sequence is reported and processing continues
    to the next path.
    """
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
