"""Replay one framed BMU capture with canonical causal ZUPT aiding.

The command loads one INS + ZUPT YAML profile, uses a real contiguous stationary
interval to level the run-local Forward-Right-Down (FRD) frame and estimate the
initial gyroscope bias, then processes subsequent IMU increments causally. Each
posterior consists of an ESEKF predict step followed by an optional SHOE/GLRT
zero-velocity pseudo-measurement update. The profile supplies the initial and
process covariances, detector settings, initial accelerometer bias, ZUPT
measurement covariance, and bounded-coast uncertainty.

After initialization, one qualifying missing notification may be propagated as
an explicitly recorded synthetic coast interval; the stance detector resets
across that interval and no ZUPT is applied to it. The optional CSV output
contains the aligned nominal/covariance history and schema sidecar. The optional
horizontal plot is a presentation of run-local FRD estimates, not a geographic
registration or navigation result.

ZUPTs constrain motion and associated error states but do not independently
establish absolute position or geographic heading.

Example:
    python scripts/run_zupt_replay.py data/walk.bmu \\
        --config config/ins_zupt/example.yaml \\
        --csv-output results/zupt.csv --plot-output plots/zupt.png
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from bootins.bmu.transport import iter_frames_replay
from bootins.config.ins_zupt import load_zupt_ins_config
from bootins.replay.ins_zupt import ZuptReplayResult, run_zupt_replay
from bootins.reporting.navigation_export import write_esekf_history_csv
from bootins.reporting.trajectory_plots import plot_horizontal_trajectory


def _format_vector(vector: np.ndarray) -> str:
    """Format one navigation vector compactly for a terminal summary."""
    return np.array2string(vector, precision=4, suppress_small=True)


def print_summary(
    replay_result: ZuptReplayResult,
    capture_path: Path,
    config_path: Path,
) -> None:
    """Print initialization, ZUPT, recovery, and terminal-state facts.

    Args:
        replay_result: Completed ESEKF replay with aligned posterior histories.
        capture_path: Source capture path used as the report label.
        config_path: Loaded navigation-profile path used as the report label.
    """
    initialization = replay_result.initialization
    static_level = initialization.static_level
    interval = replay_result.initialization_interval
    update_count = int(np.count_nonzero(replay_result.stance_flags))
    replay_increment_count = replay_result.stance_flags.size

    print(f"INS plus ZUPT replay: {capture_path.name}")
    print(f"Configuration: {config_path.name}")
    print("Navigation frame: run-local Forward-Right-Down")
    print()
    print("Initialization")
    print(f"  Selected stance increments: {interval.increment_count}")
    print(f"  Selected stance duration: {interval.duration:.3f} s")
    print(
        "  Mean specific force_b: "
        f"{_format_vector(static_level.mean_specific_force_b)} m/s^2"
    )
    print(
        "  Estimated gyro bias_b: "
        f"{_format_vector(initialization.gyro_bias_b)} rad/s"
    )
    print("Replay")
    print(f"  Post-init increments: {replay_increment_count}")
    print(f"  Post-init duration: {replay_result.time[-1]:.3f} s")
    if replay_result.coast_plans:
        print(
            "  IMU recovery: "
            f"{len(replay_result.coast_plans)} bounded coast plan(s) "
            "(raw capture contains a gap)"
        )
    else:
        print("  IMU recovery: none")
    print(f"  ZUPT updates: {update_count} / {replay_increment_count}")
    print(
        "  Final position_n: "
        f"{_format_vector(replay_result.nominals[-1, :3])} m"
    )
    print(
        "  Final velocity_n: "
        f"{_format_vector(replay_result.nominals[-1, 3:6])} m/s"
    )


def main() -> int:
    """Run one canonical INS + ZUPT replay and write requested artifacts.

    Returns:
        ``0`` after successful replay and requested outputs, ``1`` when the
        capture or replay evidence is rejected, or ``2`` when configuration,
        capture I/O, CSV export, or plot output fails.

    Notes:
        The CSV export preserves ESEKF posterior history and row-aligned ZUPT
        flags. The optional plot visualizes horizontal run-local FRD position
        only and does not alter or validate the navigation estimate.
    """
    parser = argparse.ArgumentParser(
        description="Replay an INS plus ZUPT trajectory from one .bmu capture.",
    )
    parser.add_argument("capture_path", type=Path, help="Path to one .bmu capture")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to one INS plus ZUPT YAML configuration",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        metavar="PATH",
        help="Optional ESEKF-history CSV output path (also writes a schema sidecar)",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        metavar="PATH",
        help="Optional horizontal-trajectory plot output path",
    )
    args = parser.parse_args()

    try:
        zupt_config = load_zupt_ins_config(args.config)
    except OSError as error:
        print(
            f"Could not read INS plus ZUPT configuration {args.config}: {error}",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(
            f"INS plus ZUPT configuration rejected {args.config}: {error}",
            file=sys.stderr,
        )
        return 2

    try:
        replay_result = run_zupt_replay(
            iter_frames_replay(args.capture_path),
            minimum_stance_duration=zupt_config.minimum_stance_duration,
            startup_search_limit=zupt_config.startup_search_limit,
            initial_accel_bias_b=zupt_config.initial_accel_bias_b,
            P0=zupt_config.P0,
            Q=zupt_config.Q,
            Q_coast=zupt_config.Q_coast,
            R_zupt=zupt_config.R_zupt,
            window_size=zupt_config.window_size,
            threshold=zupt_config.threshold,
            sigma_a=zupt_config.sigma_a,
            sigma_g=zupt_config.sigma_g,
        )
    except OSError as error:
        print(
            f"Could not read capture {args.capture_path}: {error}",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(
            f"INS plus ZUPT replay rejected {args.capture_path}: {error}",
            file=sys.stderr,
        )
        return 1

    print_summary(replay_result, args.capture_path, args.config)

    if args.csv_output is not None:
        try:
            write_esekf_history_csv(
                args.csv_output,
                replay_result.time,
                replay_result.nominals,
                replay_result.covariances,
                zupt_applied=np.concatenate(
                    ([False], replay_result.stance_flags)
                ),
            )
        except (OSError, ValueError) as error:
            print(
                f"Could not write navigation history CSV {args.csv_output}: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"Wrote navigation history CSV: {args.csv_output}")

    if args.plot_output is not None:
        try:
            trajectory_plot = plot_horizontal_trajectory(
                replay_result.nominals[:, :3],
                trajectory_label="INS + ZUPT",
                title=f"INS + ZUPT replay: {args.capture_path.stem}",
            )
            trajectory_plot.axes.figure.savefig(args.plot_output)
        except (OSError, ValueError) as error:
            print(
                f"Could not write trajectory plot {args.plot_output}: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"Wrote trajectory plot: {args.plot_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
