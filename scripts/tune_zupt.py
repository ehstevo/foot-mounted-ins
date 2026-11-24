#!/usr/bin/env python3
"""Explore SHOE/GLRT stance-detector behavior on recorded IMU evidence.

The command first performs strict replay-input preflight on one stationary
``.bmu`` capture and each supplied walking or step-hold capture. It rejects
confirmed notification loss and structural measurement concerns rather than
reporting detector statistics for affected evidence. From the trimmed stationary
capture it estimates pooled sample-to-sample specific-force and angular-rate
noise scales, ``sigma_a`` and ``sigma_g``; these characterize residual jitter,
not sensor bias.

For every requested trailing-window length and detector threshold, the command
reports flagged-sample count/duration, score quantiles, raw contiguous stance
runs, and presentation-only merged stance episodes. Merging brief enclosed
false gaps exposes threshold chatter without modifying the detector flags used
for raw-run statistics. Optional known step counts provide a descriptive ratio,
not ground truth for stance timing.

This is controlled detector analysis, not canonical navigation replay. Threshold
statistics cannot independently identify every physical stance interval, choose
an optimal threshold, or guarantee accurate navigation.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from bootins.aiding import stance_glrt as zupt
from bootins.bmu.transport import iter_frames_replay
from bootins.estimation.mechanization import Increment
from bootins.replay.imu_integrity import collect_clean_increments

DEFAULT_WINDOW_SIZES = [7]
DEFAULT_THRESHOLDS = [4e5, 4.5e5, 5e5, 5.5e5, 6e5]

# Trim placement and pickup transients before estimating rest-capture jitter.
TRIM_SECONDS = 0.5


def load_clean_increments(path: Path) -> list[Increment]:
    """Preflight one capture and return its complete clean increment sequence.

    Args:
        path: Framed ``.bmu`` capture to validate.

    Returns:
        Body-frame ``(dtheta, dv, dt)`` increments in radians, m/s, and seconds.

    Raises:
        ValueError: The capture has a sequence gap or structural concern.
    """
    return collect_clean_increments(iter_frames_replay(path))


def _load_required_clean_increments(path: Path) -> list[Increment]:
    """Return clean increments while refusing an empty capture.

    Args:
        path: Framed ``.bmu`` capture to validate and load.

    Returns:
        Nonempty clean body-frame increment sequence.

    Raises:
        ValueError: The capture is invalid or contains no usable increments.
    """
    increments = load_clean_increments(path)
    if not increments:
        raise ValueError("No usable IMU increments were decoded.")
    return increments


def analyze_rest(increments: list[Increment], trim: float) -> tuple[np.float64, np.float64]:
    """Estimate pooled sample-to-sample noise scales from rest increments.

    Args:
        increments: Nonempty body-frame ``(dtheta, dv, dt)`` tuples in radians,
            m/s, and seconds from a stationary capture.
        trim: Duration removed from both ends of the cumulative-``dt`` interval
            (s).

    Returns:
        ``(sigma_a, sigma_g)`` pooled over centered three-axis specific force
        in m/s² and angular rate in rad/s respectively.

    Notes:
        The calculation converts every increment to its own per-sample mean,
        removes the trimmed per-axis mean, and pools residual standard deviation.
        It estimates short-term jitter only; it does not estimate bias or a
        complete stochastic sensor model.
    """
    dt_k = np.array([increment[2] for increment in increments], dtype=float)
    dv_k = np.vstack([increment[1] for increment in increments])
    dtheta_k = np.vstack([increment[0] for increment in increments])

    t = np.cumsum(dt_k)
    mask = (t >= trim) & (t <= t[-1] - trim)
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
    """Measure raw contiguous detector-flagged run lengths.

    Args:
        flags: Increment-aligned stance decisions, shape ``(N,)``.
        dt_k: Corresponding IMU integration intervals in seconds, shape ``(N,)``.

    Returns:
        Integer sample counts and duration values in milliseconds for every raw
        contiguous ``True`` run.
    """
    run_lengths_samples = []
    run_durations = []

    run_len = 0
    run_dt = 0.0
    for flag, dt in zip(flags, dt_k, strict=True):
        if flag:
            run_len += 1
            run_dt += dt
        elif run_len:
            run_lengths_samples.append(run_len)
            run_durations.append(run_dt * 1000.0)
            run_len = 0
            run_dt = 0.0

    if run_len:
        run_lengths_samples.append(run_len)
        run_durations.append(run_dt * 1000.0)

    return (
        np.array(run_lengths_samples, dtype=int),
        np.array(run_durations, dtype=float),
    )


def _merge_short_false_gaps(flags: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Bridge brief enclosed false gaps for reporting-only stance episodes.

    Args:
        flags: Increment-aligned raw stance decisions, shape ``(N,)``.
        max_gap_samples: Maximum enclosed false-run length to bridge, in samples.

    Returns:
        Copy of ``flags`` with qualifying enclosed false gaps set ``True``.

    Raises:
        ValueError: ``max_gap_samples`` is negative.

    Notes:
        A GLRT may chatter around threshold during a planted-foot interval.
        Merging prevents that fragmentation from inflating episode counts, but
        it does not change raw detector outputs or navigation updates.
    """
    if max_gap_samples < 0:
        raise ValueError("max_gap_samples must be non-negative.")

    merged = np.array(flags, dtype=bool, copy=True)
    i = 0

    while i < len(merged):
        if merged[i]:
            i += 1
            continue

        gap_start = i
        while i < len(merged) and not merged[i]:
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start

        if (
            0 < gap_len <= max_gap_samples
            and gap_start > 0
            and gap_end < len(merged)
            and merged[gap_start - 1]
            and merged[gap_end]
        ):
            merged[gap_start:gap_end] = True

    return merged


def _merged_true_run_lengths(
    flags: np.ndarray,
    dt_k: np.ndarray,
    max_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure reporting-only episode lengths after enclosed-gap bridging.

    Args:
        flags: Increment-aligned raw stance decisions, shape ``(N,)``.
        dt_k: Corresponding IMU integration intervals in seconds, shape ``(N,)``.
        max_gap_samples: Largest enclosed false gap to bridge, in samples.

    Returns:
        Episode sample counts and durations in milliseconds.
    """
    merged_flags = _merge_short_false_gaps(flags, max_gap_samples)
    return _true_run_lengths(merged_flags, dt_k)


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
    """Build detector statistics for one capture/window/threshold candidate.

    Args:
        path: Source capture path used in the summary label.
        window_size: Trailing SHOE/GLRT window length in samples.
        threshold: SHOE/GLRT stance-decision threshold.
        sigma_a: Rest-derived specific-force noise scale in m/s².
        sigma_g: Rest-derived angular-rate noise scale in rad/s.
        dt_k: Increment intervals in seconds, shape ``(N,)``.
        flags: Increment-aligned detector decisions, shape ``(N,)``.
        scores: Increment-aligned SHOE/GLRT scores, shape ``(N,)``.
        true_step_count: Optional independently supplied expected step count.

    Returns:
        Human-readable detector statistics with raw runs and merged episodes.

    Notes:
        Episode merging bridges only short enclosed false gaps for presentation.
        Counts and score quantiles are descriptive and do not validate stance
        truth or select a threshold.
    """
    finite_mask = np.isfinite(scores)
    finite_scores = scores[finite_mask]
    valid_flags = flags[finite_mask]
    valid_dt = dt_k[finite_mask]
    startup_delay = (window_size - 1) * np.mean(dt_k) * 1000.0

    lines = [
        f"Results for {path.name}:",
        (
            f"Window Size: {window_size}   Threshold: {threshold}   "
            f"Sample Count: {len(dt_k)}   Duration: {np.sum(dt_k):.3f} s"
        ),
        f"Sigma_a: {sigma_a:.6f}   Sigma_g: {sigma_g:.6f}",
        f"Startup Delay: {startup_delay:.1f} ms",
    ]

    if finite_scores.size == 0:
        lines.append("No finite scores: file shorter than one full window.")
        return "\n".join(lines)

    flagged_samples = int(np.count_nonzero(valid_flags))
    flagged_duration = float(np.sum(valid_dt[valid_flags]) * 1000.0)
    raw_run_samples, raw_run_durations = _true_run_lengths(flags, dt_k)
    # Gaps shorter than the detector memory are usually threshold chatter within
    # one physical stance, not a full swing phase.
    merge_gap_samples = window_size - 1
    merge_gap_duration = merge_gap_samples * np.mean(dt_k) * 1000.0
    merged_run_samples, _ = _merged_true_run_lengths(flags, dt_k, merge_gap_samples)

    lines.append(
        f"Flagged samples after startup: {flagged_samples}/{valid_flags.size} "
        f"({flagged_samples / valid_flags.size:.3f})"
    )
    lines.append(f"Flagged duration after startup: {flagged_duration:.1f} ms")
    lines.append(f"Raw contiguous stance runs: {len(raw_run_samples)}")
    lines.append(
        "Merged stance episodes "
        f"(bridging gaps <= {merge_gap_samples} samples / {merge_gap_duration:.1f} ms): "
        f"{len(merged_run_samples)}"
    )

    if true_step_count is not None:
        lines.append(f"Expected step count: {true_step_count}")
        if true_step_count > 0:
            lines.append(
                f"raw runs / true_step: {len(raw_run_samples) / true_step_count:.3f}"
            )
            lines.append(
                f"merged episodes / true_step: {len(merged_run_samples) / true_step_count:.3f}"
            )

    if raw_run_samples.size:
        lines.append(
            "Raw Run Length (samples)[min/med/max] : "
            f"{_fmt_triplet(raw_run_samples, precision=1)}"
        )
        lines.append(
            f"Raw Run Length (ms)[min/med/max]      : {_fmt_triplet(raw_run_durations, precision=1)}"
        )
    else:
        lines.append("Raw Run Lengths: none (no detected stance runs)")

    score_q = np.percentile(finite_scores, [5, 50, 95])
    lines.append(f"score[p5/p50/p95] : {score_q}")

    return "\n".join(lines)


def main() -> int:
    """Run a strict SHOE/GLRT parameter sweep over recorded captures.

    Returns:
        ``0`` after reporting every requested candidate, ``1`` when capture
        evidence is rejected, or ``2`` when a capture cannot be read.

    Notes:
        This CLI writes no navigation result or selected profile. It prints
        rest-derived scales and detector statistics for the declared threshold
        and window-size combinations.
    """
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
        "--thresholds",
        "--threshold",
        dest="thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help=(
            "one or more stance thresholds to apply during the sweep "
            f"(default: {DEFAULT_THRESHOLDS})"
        ),
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_WINDOW_SIZES,
        help=(
            "one or more trailing window sizes in samples "
            f"(default: {DEFAULT_WINDOW_SIZES})"
        ),
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

    if any(window_size < 1 for window_size in args.window_sizes):
        parser.error("--window-sizes must contain positive integer sample counts.")
    if any(
        not np.isfinite(threshold) or threshold < 0.0
        for threshold in args.thresholds
    ):
        parser.error("--thresholds must contain finite, non-negative values.")

    current_capture_path = args.rest_file
    try:
        rest_increments = _load_required_clean_increments(current_capture_path)
        walking_inputs: list[tuple[Path, list[Increment]]] = []
        for current_capture_path in args.files:
            walking_inputs.append(
                (
                    current_capture_path,
                    _load_required_clean_increments(current_capture_path),
                )
            )
    except OSError as error:
        print(
            f"Could not read tuning capture {current_capture_path}: {error}",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(
            f"Tuning capture rejected {current_capture_path}: {error}",
            file=sys.stderr,
        )
        return 1

    sigma_a, sigma_g = analyze_rest(rest_increments, TRIM_SECONDS)

    for file_index, (path, clean_increments) in enumerate(walking_inputs):
        dt_k = np.array([increment[2] for increment in clean_increments], dtype=float)
        true_step_count = None if true_step_counts is None else true_step_counts[file_index]

        for window_size in args.window_sizes:
            for threshold in args.thresholds:
                scores = zupt.shoe_scores(clean_increments, window_size, sigma_a, sigma_g)
                flags = zupt.shoe_stance_flags(
                    clean_increments, window_size, threshold, sigma_a, sigma_g
                )
                print(
                    _summary_block(
                        path, window_size, threshold, sigma_a, sigma_g,
                        dt_k, flags, scores, true_step_count,
                    )
                )
                print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
