#!/usr/bin/env python3
"""Compare predict-only and ZUPT-corrected navigation on one walking capture.

This noncanonical diagnostic reads a continuous, structurally clean ``.bmu``
capture and runs two self-contained ESEKF histories over the same IMU increments:
predict-only propagation and causal SHOE/GLRT-triggered zero-velocity updates.
It writes top-down trajectory and multi-panel diagnostic plots plus per-sample
and raw-stance-run CSV summaries under ``/tmp``.

The comparison intentionally uses an identity initial attitude, zero local
position/velocity, and fixed detector/noise/covariance settings. Those choices
make within-script predict-versus-update behavior inspectable, but differ from
the configuration-driven initialization, bounded-coast policy, and artifact
conventions of ``run_zupt_replay.py``. The plotted coordinate labels therefore
do not provide geographic registration, and neither trace establishes trajectory
accuracy without independent reference evidence.

The results can illustrate how the stated ZUPT assumptions affect velocity and
covariance behavior. They cannot independently identify physically correct
stance intervals, select an optimal threshold, or validate navigation accuracy.
"""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from bootins.bmu.decode import Measurement
from bootins.bmu.quality import QualityFlags, assess
from bootins.bmu.stream import iter_tracked_frames
from bootins.bmu.transport import iter_frames_replay
from bootins.estimation.esekf import ERR_DV, NOM_V, esekf_predict, esekf_update
from bootins.aiding.stance_glrt import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    shoe_scores,
    shoe_stance_flags,
)

# Fixed diagnostic detector-noise scales keep this comparison self-contained.
SIGMA_A = 0.01
SIGMA_G = 0.0005

_H_VEL = np.zeros((3, 15), dtype=float)
_H_VEL[:, 3:6] = np.eye(3)


def _vel_h(nominal: np.ndarray) -> np.ndarray:
    """Local ZUPT measurement model: predicted measurement is nominal velocity."""
    return nominal[NOM_V]


def _nominal0() -> np.ndarray:
    """Return the fixed diagnostic initial nominal state.

    Position and velocity begin at zero with an identity attitude and zero bias.
    The identity quaternion is not a claim that the physical sensor is aligned
    to geographic NED; it only supplies a common initial state for comparing
    predict-only and ZUPT-corrected propagation.
    """
    return np.hstack((
        np.zeros(3),              # position
        np.zeros(3),              # velocity
        [1.0, 0.0, 0.0, 0.0],     # level attitude
        np.zeros(3),              # accel bias
        np.zeros(3),              # gyro bias
    ))


def _P0() -> np.ndarray:
    """Return the fixed nontrivial ESEKF prior used by this diagnostic.

    These covariance values support within-script comparison only. They are not
    loaded from, or a substitute for, the canonical navigation profile.
    """
    return np.diag([
        10.0, 10.0, 10.0,         # dp
        10.0, 10.0, 10.0,         # dv
        1e-3, 1e-3, 1e-3,         # dpsi
        1.0, 1.0, 1.0,            # db_a
        1e-6, 1e-6, 1e-6,         # db_g
    ]).astype(float)


def load_measurements(path: Path) -> list[Measurement]:
    """Load one continuous, structurally clean capture as flat measurements.

    Args:
        path: Framed BMU capture to inspect before diagnostic propagation.

    Returns:
        Decoded measurements in source order.

    Raises:
        ValueError: A confirmed sequence gap or structural quality concern is
            encountered.

    Notes:
        This batch diagnostic refuses affected evidence rather than silently
        skipping samples or reconstructing missing IMU increments.
    """
    measurements: list[Measurement] = []
    for tracked_frame in iter_tracked_frames(iter_frames_replay(path)):
        decoded_frame = tracked_frame.decoded_frame
        frame_index = decoded_frame.raw_frame.frame_index
        gap_before = tracked_frame.gap_before

        if gap_before is not None:
            raise ValueError(
                f"Confirmed BMU sequence gap before frame {frame_index}: "
                f"{gap_before.previous_seq_num} -> {gap_before.current_seq_num}; "
                f"{gap_before.missing_message_count} notification(s) missing."
            )
        for measurement_index, measurement in enumerate(
            decoded_frame.decoded_message.measurements
        ):
            flags = assess(measurement)
            if flags != QualityFlags.OK:
                flag_names = " | ".join(
                    flag.name
                    for flag in QualityFlags
                    if flag != QualityFlags.OK and flags & flag
                )
                raise ValueError(
                    "Structurally flagged measurement at "
                    f"frame {frame_index}, measurement index {measurement_index}: "
                    f"{flag_names}."
                )
            measurements.append(measurement)

    return measurements


def as_increment(measurement) -> tuple[np.ndarray, np.ndarray, float]:
    """Project one decoded record to body-frame INS increments.

    Args:
        measurement: Decoded BMU measurement with ``dtheta``, ``dv``, and ``dt``.

    Returns:
        ``(dtheta, dv, dt)`` in radians, m/s, and seconds respectively.
    """
    return (measurement.dtheta, measurement.dv, measurement.dt)


def _pvv_trace(P: np.ndarray) -> float:
    """Return the velocity-error covariance trace from a 15x15 ESEKF P."""
    return float(np.trace(P[ERR_DV, ERR_DV]))


def _run_predict_only(
    increments: list[tuple[np.ndarray, np.ndarray, float]],
    nominal0: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
) -> dict:
    """Propagate the fixed ESEKF model without measurement updates.

    Args:
        increments: Ordered body-frame ``(dtheta, dv, dt)`` tuples in radians,
            m/s, and seconds.
        nominal0: Initial nominal state shared with the ZUPT comparison.
        P0: Initial 15-state error covariance.
        Q: Per-predict process covariance used for this diagnostic.

    Returns:
        Post-predict nominal states, covariances, speed norms, and velocity-
        covariance traces indexed by increment.
    """
    nominal = np.array(nominal0, dtype=float, copy=True)
    P = np.array(P0, dtype=float, copy=True)
    nominals = np.zeros((len(increments), nominal.shape[0]), dtype=float)
    Ps = np.zeros((len(increments), *P.shape), dtype=float)
    speed = np.zeros(len(increments), dtype=float)
    pvv_trace = np.zeros(len(increments), dtype=float)

    for k, (dtheta, dv, dt) in enumerate(increments):
        nominal, P = esekf_predict(nominal, P, dtheta, dv, dt, Q)
        nominals[k] = nominal
        Ps[k] = P
        speed[k] = float(np.linalg.norm(nominal[NOM_V]))
        pvv_trace[k] = _pvv_trace(P)

    return {
        "nominals": nominals,
        "Ps": Ps,
        "speed": speed,
        "pvv_trace": pvv_trace,
    }


def _run_zupt_with_diagnostics(
    increments: list[tuple[np.ndarray, np.ndarray, float]],
    nominal0: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
    R_zupt: np.ndarray,
    window_size: int,
    threshold: float,
    sigma_a: float,
    sigma_g: float,
) -> dict:
    """Run causal ZUPT updates and retain pre/post-update diagnostics.

    Args:
        increments: Ordered body-frame ``(dtheta, dv, dt)`` tuples in radians,
            m/s, and seconds.
        nominal0: Initial nominal state shared with predict-only propagation.
        P0: Initial 15-state error covariance.
        Q: Per-predict process covariance used for this diagnostic.
        R_zupt: Three-axis zero-velocity measurement covariance.
        window_size: Trailing SHOE/GLRT window length in samples.
        threshold: SHOE/GLRT stance-decision threshold.
        sigma_a: Specific-force noise scale used by the detector.
        sigma_g: Angular-rate noise scale used by the detector.

    Returns:
        Per-increment detector, posterior, speed, covariance-trace, and
        ZUPT-correction diagnostics.

    Notes:
        The history distinguishes each pre-update prediction from its optional
        post-update posterior, including repeated updates in a flagged stance.
    """
    increments = tuple(increments)
    dt_k = np.array([increment[2] for increment in increments], dtype=float)
    time = np.cumsum(dt_k)
    scores = shoe_scores(increments, window_size, sigma_a, sigma_g)
    flags = shoe_stance_flags(increments, window_size, threshold, sigma_a, sigma_g)

    nominal = np.array(nominal0, dtype=float, copy=True)
    P = np.array(P0, dtype=float, copy=True)

    nominals_post = np.zeros((len(increments), nominal.shape[0]), dtype=float)
    Ps_post = np.zeros((len(increments), *P.shape), dtype=float)
    speed_pred = np.zeros(len(increments), dtype=float)
    speed_post = np.zeros(len(increments), dtype=float)
    pvv_trace_pred = np.zeros(len(increments), dtype=float)
    pvv_trace_post = np.zeros(len(increments), dtype=float)
    delta_v_update_norm = np.zeros(len(increments), dtype=float)
    delta_pvv_trace = np.zeros(len(increments), dtype=float)
    z_zupt = np.zeros(3, dtype=float)

    for k, (dtheta, dv, dt) in enumerate(increments):
        nominal_pred, P_pred = esekf_predict(nominal, P, dtheta, dv, dt, Q)
        speed_pred[k] = float(np.linalg.norm(nominal_pred[NOM_V]))
        pvv_trace_pred[k] = _pvv_trace(P_pred)

        nominal_post = nominal_pred
        P_post = P_pred
        if flags[k]:
            nominal_post, P_post = esekf_update(
                nominal_pred, P_pred, z_zupt, _vel_h, _H_VEL, R_zupt
            )
            delta_v_update_norm[k] = float(
                np.linalg.norm(nominal_post[NOM_V] - nominal_pred[NOM_V])
            )
            delta_pvv_trace[k] = pvv_trace_pred[k] - _pvv_trace(P_post)

        speed_post[k] = float(np.linalg.norm(nominal_post[NOM_V]))
        pvv_trace_post[k] = _pvv_trace(P_post)
        nominals_post[k] = nominal_post
        Ps_post[k] = P_post

        nominal = nominal_post
        P = P_post

    return {
        "time_s": time,
        "dt_k": dt_k,
        "scores": scores,
        "flags": flags,
        "nominals_post": nominals_post,
        "Ps_post": Ps_post,
        "speed_pred": speed_pred,
        "speed_post": speed_post,
        "pvv_trace_pred": pvv_trace_pred,
        "pvv_trace_post": pvv_trace_post,
        "delta_v_update_norm": delta_v_update_norm,
        "delta_pvv_trace": delta_pvv_trace,
    }


def _build_run_summaries(diagnostics: dict) -> list[dict]:
    """Summarize contiguous detector-flagged runs before episode merging.

    Args:
        diagnostics: Per-increment mapping returned by
            :func:`_run_zupt_with_diagnostics`.

    Returns:
        One mapping per raw contiguous ``True`` stance run, including duration,
        pre/post speed, covariance trace, and accumulated update magnitudes.

    Notes:
        The diagnostic ESEKF applies one ZUPT per flagged sample, so raw runs
        retain the repeated-update behavior that episode merging would hide.
    """
    flags = diagnostics["flags"]
    time = diagnostics["time_s"]
    dt_k = diagnostics["dt_k"]
    speed_pred = diagnostics["speed_pred"]
    speed_post = diagnostics["speed_post"]
    pvv_trace_pred = diagnostics["pvv_trace_pred"]
    pvv_trace_post = diagnostics["pvv_trace_post"]
    delta_v_update_norm = diagnostics["delta_v_update_norm"]
    delta_pvv_trace = diagnostics["delta_pvv_trace"]

    runs = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue

        start = i
        while i + 1 < len(flags) and flags[i + 1]:
            i += 1
        end = i

        runs.append({
            "run_id": len(runs) + 1,
            "start_index": start,
            "end_index": end,
            "start_s": float(time[start]),
            "end_s": float(time[end]),
            "duration_ms": float(np.sum(dt_k[start:end + 1]) * 1000.0),
            "n_flagged_samples": int(end - start + 1),
            "speed_pred_at_run_start": float(speed_pred[start]),
            "speed_post_at_run_end": float(speed_post[end]),
            "pvv_trace_pred_at_run_start": float(pvv_trace_pred[start]),
            "pvv_trace_post_at_run_end": float(pvv_trace_post[end]),
            "sum_delta_v_update_norm_over_run": float(
                np.sum(delta_v_update_norm[start:end + 1])
            ),
            "sum_delta_pvv_trace_over_run": float(
                np.sum(delta_pvv_trace[start:end + 1])
            ),
        })
        i += 1

    return runs


def _write_sample_csv(path: Path, predict_only: dict, diagnostics: dict) -> None:
    """Write increment-aligned predict-only and ZUPT diagnostics to CSV.

    Args:
        path: New per-sample CSV output path.
        predict_only: Mapping returned by :func:`_run_predict_only`.
        diagnostics: Mapping returned by :func:`_run_zupt_with_diagnostics`.
    """
    fieldnames = [
        "time_s",
        "flag",
        "score",
        "speed_predict_only",
        "speed_pred_zupt",
        "speed_post_zupt",
        "pvv_trace_predict_only",
        "pvv_trace_pred_zupt",
        "pvv_trace_post_zupt",
        "delta_v_update_norm",
        "delta_pvv_trace",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(len(diagnostics["time_s"])):
            writer.writerow({
                "time_s": diagnostics["time_s"][k],
                "flag": int(diagnostics["flags"][k]),
                "score": diagnostics["scores"][k],
                "speed_predict_only": predict_only["speed"][k],
                "speed_pred_zupt": diagnostics["speed_pred"][k],
                "speed_post_zupt": diagnostics["speed_post"][k],
                "pvv_trace_predict_only": predict_only["pvv_trace"][k],
                "pvv_trace_pred_zupt": diagnostics["pvv_trace_pred"][k],
                "pvv_trace_post_zupt": diagnostics["pvv_trace_post"][k],
                "delta_v_update_norm": diagnostics["delta_v_update_norm"][k],
                "delta_pvv_trace": diagnostics["delta_pvv_trace"][k],
            })


def _write_run_csv(path: Path, run_summaries: list[dict]) -> None:
    """Write one raw contiguous-stance-run summary row to CSV.

    Args:
        path: New raw-run CSV output path.
        run_summaries: Mappings returned by :func:`_build_run_summaries`.
    """
    if not run_summaries:
        path.write_text("")
        return

    fieldnames = list(run_summaries[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_summaries)


def _print_run_summaries(run_summaries: list[dict]) -> None:
    """Print a compact fixed-width summary of each raw detector run."""
    if not run_summaries:
        print("Raw stance runs: none")
        return

    print("Raw stance runs:")
    print("  id  start_s   end_s   dur_ms   n  |v|_pred0  |v|_postN  trPvv_pred0  trPvv_postN  sum|dv_upd|")
    for run in run_summaries:
        print(
            "  "
            f"{run['run_id']:>2}  "
            f"{run['start_s']:>7.2f}  "
            f"{run['end_s']:>6.2f}  "
            f"{run['duration_ms']:>7.1f}  "
            f"{run['n_flagged_samples']:>2}  "
            f"{run['speed_pred_at_run_start']:>9.4f}  "
            f"{run['speed_post_at_run_end']:>10.4f}  "
            f"{run['pvv_trace_pred_at_run_start']:>11.4e}  "
            f"{run['pvv_trace_post_at_run_end']:>11.4e}  "
            f"{run['sum_delta_v_update_norm_over_run']:>10.4f}"
        )


def _summary_block(
    predict_only: dict,
    diagnostics: dict,
    run_summaries: list[dict],
    path: Path,
) -> None:
    """Print aggregate detector and update facts for one walking capture.

    Args:
        predict_only: Mapping returned by :func:`_run_predict_only`.
        diagnostics: Mapping returned by :func:`_run_zupt_with_diagnostics`.
        run_summaries: Raw contiguous detector-run summaries.
        path: Source capture path used as the report label.
    """
    flags = diagnostics["flags"]
    flagged_count = int(np.count_nonzero(flags))
    dt_k = diagnostics["dt_k"]
    flagged_update_norm = diagnostics["delta_v_update_norm"][flags]
    flagged_pvv_drop = diagnostics["delta_pvv_trace"][flags]

    print(f"Results for {path}:")
    print(f"Sample Count: {len(dt_k)}   Duration: {np.sum(dt_k):.3f} s")
    print(f"Number of flagged stance samples: {flagged_count}")
    print(f"Raw contiguous stance runs: {len(run_summaries)}")
    print(
        "Final position norm from predict-only: "
        f"{np.linalg.norm(predict_only['nominals'][-1][0:3])}"
    )
    print(
        "Final position norm from ZUPT-aided: "
        f"{np.linalg.norm(diagnostics['nominals_post'][-1][0:3])}"
    )

    if run_summaries:
        run_lengths = np.array([run["n_flagged_samples"] for run in run_summaries], dtype=float)
        run_durations = np.array(
            [run["duration_ms"] for run in run_summaries],
            dtype=float,
        )
        print(
            "Flagged samples per raw run [min/med/max]: "
            f"{run_lengths.min():.0f}/{np.median(run_lengths):.1f}/{run_lengths.max():.0f}"
        )
        print(
            "Run duration [ms][min/med/max]: "
            f"{run_durations.min():.1f}/{np.median(run_durations):.1f}/{run_durations.max():.1f}"
        )

    if flagged_count:
        print(
            "Per-flag |dv_update| [min/med/max]: "
            f"{np.min(flagged_update_norm):.4f}/"
            f"{np.median(flagged_update_norm):.4f}/"
            f"{np.max(flagged_update_norm):.4f}"
        )
        print(
            "Per-flag delta trace(Pvv) [min/med/max]: "
            f"{np.min(flagged_pvv_drop):.4e}/"
            f"{np.median(flagged_pvv_drop):.4e}/"
            f"{np.max(flagged_pvv_drop):.4e}"
        )


def plot_position(predict_only: dict, diagnostics: dict, out_path: Path) -> None:
    """Write an unregistered horizontal predict-only/ZUPT comparison plot.

    Args:
        predict_only: Mapping returned by :func:`_run_predict_only`.
        diagnostics: Mapping returned by :func:`_run_zupt_with_diagnostics`.
        out_path: PNG output path.

    Notes:
        The plot compares diagnostic state coordinates only; it does not
        register either path geographically or establish accuracy.
    """
    positions = predict_only["nominals"][:, 0:3]
    positions_zupt = diagnostics["nominals_post"][:, 0:3]
    plt.figure(figsize=(8, 6))

    plt.plot(positions[:, 0], positions[:, 1], label="Predict Only", color="blue")
    plt.plot(positions_zupt[:, 0], positions_zupt[:, 1], label="ZUPT", color="red")
    plt.xlabel("North position (m)")
    plt.ylabel("East position (m)")
    plt.title("Predict Only vs ZUPT 2-D Positions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_diagnostics(predict_only: dict, diagnostics: dict, out_path: Path) -> None:
    """Write detector, velocity, covariance, and update diagnostic panels.

    Args:
        predict_only: Mapping returned by :func:`_run_predict_only`.
        diagnostics: Mapping returned by :func:`_run_zupt_with_diagnostics`.
        out_path: PNG output path.
    """
    time = diagnostics["time_s"]
    finite = np.isfinite(diagnostics["scores"])
    flags = diagnostics["flags"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(time[finite], diagnostics["scores"][finite], color="black", linewidth=1.0)
    axes[0].axhline(DEFAULT_THRESHOLD, color="tab:red", linestyle="--", label="threshold")
    axes[0].fill_between(
        time,
        0.0,
        0.12,
        where=flags,
        step="post",
        color="tab:green",
        alpha=0.25,
        transform=axes[0].get_xaxis_transform(),
        label="stance flags",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("GLRT score")
    axes[0].set_title("Detector score with threshold and raw stance flags")
    axes[0].legend(loc="upper right")

    axes[1].plot(time, predict_only["speed"], label="predict-only |v|", color="tab:blue")
    axes[1].plot(time, diagnostics["speed_pred"], label="ZUPT pre-update |v|", color="tab:orange")
    axes[1].plot(time, diagnostics["speed_post"], label="ZUPT post-update |v|", color="tab:green")
    axes[1].set_ylabel("speed (m/s)")
    axes[1].set_title("Velocity-norm evolution")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, predict_only["pvv_trace"], label="predict-only trace(Pvv)", color="tab:blue")
    axes[2].plot(time, diagnostics["pvv_trace_pred"], label="ZUPT pre-update trace(Pvv)", color="tab:orange")
    axes[2].plot(time, diagnostics["pvv_trace_post"], label="ZUPT post-update trace(Pvv)", color="tab:green")
    axes[2].set_ylabel("trace(Pvv)")
    axes[2].set_title("Velocity covariance evolution")
    axes[2].legend(loc="upper right")

    axes[3].vlines(
        time[flags],
        0.0,
        diagnostics["delta_v_update_norm"][flags],
        color="tab:green",
        alpha=0.55,
    )
    axes[3].scatter(
        time[flags],
        diagnostics["delta_v_update_norm"][flags],
        s=10,
        color="tab:green",
    )
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("|dv update|")
    axes[3].set_title("Velocity correction magnitude at each flagged sample")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Run the fixed-assumption walking comparison and write ``/tmp`` artifacts.

    The positional capture argument is parsed from process arguments. An empty
    decoded capture prints its status and returns; other source or validation
    errors propagate. This diagnostic deliberately has no configuration or
    geographic-output interface.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare predict-only and ZUPT-aided trajectories with "
            "per-sample diagnostics on one real walking .bmu file."
        ),
    )
    parser.add_argument("file", type=Path, help="one .bmu file to analyze")
    args = parser.parse_args()

    raw_measurements = load_measurements(args.file)
    if not raw_measurements:
        print(f"\n=== {args.file.name} ===\n  (no measurements decoded -- empty or corrupt)")
        return

    increments = [as_increment(measurement) for measurement in raw_measurements]
    # Fixed covariances preserve a controlled within-script comparison.
    Q = np.eye(15) * 1e-2
    R_zupt = np.eye(3) * 1e-2
    nominal0 = _nominal0()
    P0 = _P0()

    predict_only = _run_predict_only(increments, nominal0, P0, Q)
    diagnostics = _run_zupt_with_diagnostics(
        increments,
        nominal0,
        P0,
        Q,
        R_zupt,
        DEFAULT_WINDOW_SIZE,
        DEFAULT_THRESHOLD,
        SIGMA_A,
        SIGMA_G,
    )
    run_summaries = _build_run_summaries(diagnostics)

    _summary_block(predict_only, diagnostics, run_summaries, args.file)
    _print_run_summaries(run_summaries)

    stem = args.file.stem
    path_plot = Path("/tmp") / f"{stem}_path.png"
    diag_plot = Path("/tmp") / f"{stem}_zupt_diag.png"
    sample_csv = Path("/tmp") / f"{stem}_zupt_diag.csv"
    run_csv = Path("/tmp") / f"{stem}_zupt_runs.csv"

    plot_position(predict_only, diagnostics, path_plot)
    plot_diagnostics(predict_only, diagnostics, diag_plot)
    _write_sample_csv(sample_csv, predict_only, diagnostics)
    _write_run_csv(run_csv, run_summaries)

    print(f"Saved path plot: {path_plot}")
    print(f"Saved diagnostics plot: {diag_plot}")
    print(f"Saved per-sample diagnostics CSV: {sample_csv}")
    print(f"Saved per-run summary CSV: {run_csv}")


if __name__ == "__main__":
    main()
