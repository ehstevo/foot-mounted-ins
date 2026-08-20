#!/usr/bin/env python3
"""tune_zupt.py -- sweep SHOE / GLRT window sizes on real walking captures.

This script is the first real-data tuning harness for M6. It:

  * replays one trimmed REST capture to estimate provisional `sigma_a` / `sigma_g`
  * decodes one or more walking/step-hold `.bmu` files
  * sweeps candidate trailing-window lengths `N`
  * reports detector statistics that help us judge chatter, smearing, and
    stance-burst plausibility before we commit to a threshold

It intentionally does NOT choose the threshold for us; it just makes the score
structure visible on real data.
"""
import argparse
import struct
from pathlib import Path

import numpy as np

from bootins import zupt
from bootins.bmu import decode

# Capture-layer framing, the exact inverse of ble_capture.py's writer:
#     <f64 arrival_monotonic_s> <u16 payload_len> <payload_len raw bytes>
FRAME_FORMAT = "<dH"
FRAME_SIZE = struct.calcsize(FRAME_FORMAT)  # 10

DEFAULT_WINDOWS = [5, 7, 9, 11, 13, 15]
CANDIDATE_WINDOWS = [5, 7]
DEFAULT_THRESHOLD = 0.05
CANDIDATE_THRESHOLDS = [1e4, 2e4, 5e4, 1e5, 2e5, 3e5, 5e5, 7e5, 1e6]

# Trim the place-down / pick-up transient off the rest capture before estimating
# the white-noise scales.
TRIM_SECONDS = 0.5


def iter_payloads(path: Path):
    """Yield each raw notification payload from a framed .bmu file, in order."""
    with open(path, "rb") as f:
        while True:
            head = f.read(FRAME_SIZE)
            if len(head) < FRAME_SIZE:
                break
            _arrival, payload_len = struct.unpack(FRAME_FORMAT, head)
            payload = f.read(payload_len)
            if len(payload) < payload_len:
                break
            yield payload


def load_measurements(path: Path) -> list:
    """Decode every payload in a .bmu file into decoded Measurement objects."""
    measurements = []
    for payload in iter_payloads(path):
        measurements.extend(decode.decode_message(payload))
    return measurements


def as_increment(measurement) -> tuple[np.ndarray, np.ndarray, float]:
    """Project a decoded BMU record onto the core INS increment convention."""
    return (measurement.dtheta, measurement.dv, measurement.dt)


def analyze_rest(measurements: list, trim_s: float) -> tuple[np.float64, np.float64]:
    """Estimate provisional accel/gyro white-noise scales from a rest capture.

    Rest is used here to estimate the sample-to-sample JITTER, not the bias.
    We trim the place-down / pick-up transient, convert increments back to
    per-sample means (`omega`, `f`), subtract the per-axis mean, and pool the
    residual standard deviation into one provisional `sigma_g` / `sigma_a`.
    """
    dt_k = np.array([m.dt for m in measurements], dtype=float)
    dv_k = np.vstack([m.dv for m in measurements])
    dtheta_k = np.vstack([m.dtheta for m in measurements])

    t = np.cumsum(dt_k)
    mask = (t >= trim_s) & (t <= t[-1] - trim_s)
    if mask.sum() < 10:
        mask = np.ones_like(t, dtype=bool)
    dt_k, dv_k, dtheta_k = dt_k[mask], dv_k[mask], dtheta_k[mask]

    omega_k = dtheta_k / dt_k[:, None]
    f_k = dv_k / dt_k[:, None]

    omega_centered = omega_k - np.mean(omega_k, axis=0)
    f_centered = f_k - np.mean(f_k, axis=0)
    sigma_g = np.std(omega_centered)
    sigma_a = np.std(f_centered)

    return sigma_a, sigma_g


def _true_run_lengths(flags: np.ndarray, dt_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous-True run lengths in samples and milliseconds."""
    run_lengths_samples = []
    run_lengths_ms = []

    run_len = 0
    run_dt = 0.0
    for flag, dt in zip(flags, dt_k, strict=True):
        if flag:
            run_len += 1
            run_dt += dt
        elif run_len:
            run_lengths_samples.append(run_len)
            run_lengths_ms.append(run_dt * 1000.0)
            run_len = 0
            run_dt = 0.0

    if run_len:
        run_lengths_samples.append(run_len)
        run_lengths_ms.append(run_dt * 1000.0)

    return (
        np.array(run_lengths_samples, dtype=int),
        np.array(run_lengths_ms, dtype=float),
    )


def _fmt_triplet(values: np.ndarray, precision: int = 1) -> str:
    """Format a numeric min/median/max triplet compactly."""
    return (
        f"{np.min(values):.{precision}f}/"
        f"{np.median(values):.{precision}f}/"
        f"{np.max(values):.{precision}f}"
    )


def _summary_block(
    path: Path,
    window_size: int,
    threshold: float,
    sigma_a: float,
    sigma_g: float,
    dt_k: np.ndarray,
    flags: np.ndarray,
    scores: np.ndarray,
    true_step_count: int | None,
) -> str:
    """Build the printed summary for one file / one window size."""
    finite_mask = np.isfinite(scores)
    finite_scores = scores[finite_mask]
    valid_flags = flags[finite_mask]
    startup_ms = (window_size - 1) * np.mean(dt_k) * 1000.0

    lines = [
        f"Results for {path.name}:",
        (
            f"Window Size: {window_size}   Threshold: {threshold}   "
            f"Sample Count: {len(dt_k)}   Duration: {np.sum(dt_k):.3f} s"
        ),
        f"Sigma_a: {sigma_a:.6f}   Sigma_g: {sigma_g:.6f}",
        f"Startup Delay: {startup_ms:.1f} ms",
    ]

    if finite_scores.size == 0:
        lines.append("No finite scores: file shorter than one full window.")
        return "\n".join(lines)

    run_samples, run_ms = _true_run_lengths(flags, dt_k)

    lines.append(
        f"flagged fraction after startup: {np.count_nonzero(valid_flags)}/{valid_flags.size}"
    )
    lines.append(f"Detected stance bursts: {len(run_samples)}")

    if true_step_count is not None:
        lines.append(f"Expected step count: {true_step_count}")
        if true_step_count > 0:
            lines.append(f"bursts / true_step: {len(run_samples) / true_step_count:.3f}")

    if run_samples.size:
        lines.append(
            f"Run Length (samples)[min/med/max] : {_fmt_triplet(run_samples, precision=1)}"
        )
        lines.append(
            f"Run Length (ms)[min/med/max]      : {_fmt_triplet(run_ms, precision=1)}"
        )
    else:
        lines.append("Run Lengths: none (no detected stance bursts)")

    score_q = np.percentile(finite_scores, [5, 50, 95])
    lines.append(f"score[p5/p50/p95] : {score_q}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate provisional SHOE noise scales from a rest capture and sweep "
            "window sizes over one or more walking/step-hold .bmu files."
        ),
    )
    parser.add_argument("files", nargs="+", type=Path, help="one or more .bmu files to analyze")
    parser.add_argument(
        "--rest-file",
        type=Path,
        required=True,
        help="dedicated rest .bmu capture used to estimate sigma_a / sigma_g",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"stance threshold to apply during the sweep (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_WINDOWS,
        help=f"candidate trailing window sizes in samples (default: {DEFAULT_WINDOWS})",
    )
    parser.add_argument(
        "--true-step-counts",
        type=str,
        help="optional comma-separated true step counts aligned 1:1 with positional files",
    )
    args = parser.parse_args()

    true_step_counts = None
    if args.true_step_counts is not None:
        true_step_counts = [int(part) for part in args.true_step_counts.split(",") if part]
        if len(true_step_counts) != len(args.files):
            parser.error("--true-step-counts must have exactly one entry per positional file.")

    rest_measurements = load_measurements(args.rest_file)
    if not rest_measurements:
        print(f"\n=== {args.rest_file.name} ===\n  (no measurements decoded -- empty or corrupt)")
        return

    sigma_a, sigma_g = analyze_rest(rest_measurements, TRIM_SECONDS)

    for file_index, path in enumerate(args.files):
        raw_measurements = load_measurements(path)
        if not raw_measurements:
            print(f"\n=== {path.name} ===\n  (no measurements decoded -- empty or corrupt)")
            continue

        measurements = [as_increment(m) for m in raw_measurements]
        dt_k = np.array([m[2] for m in measurements], dtype=float)
        true_step_count = None if true_step_counts is None else true_step_counts[file_index]

        for window_size in CANDIDATE_WINDOWS:
            for threshold in CANDIDATE_THRESHOLDS:
                scores = zupt.shoe_scores(measurements, window_size, sigma_a, sigma_g)
                flags = zupt.shoe_stance_flags(
                    measurements, window_size, threshold, sigma_a, sigma_g
                )
                print(
                    _summary_block(
                        path, window_size, threshold, sigma_a, sigma_g,
                        dt_k, flags, scores, true_step_count,
                    )
                )
                print()


if __name__ == "__main__":
    main()
