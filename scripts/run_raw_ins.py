"""Replay one framed BMU capture as a canonical unaided INS trajectory.

The command consumes raw ``.bmu`` transport evidence, requires a
caller-declared contiguous static interval, and uses its leveling result to
define a run-local Forward-Right-Down (FRD) navigation frame. It initializes
position and velocity at zero, then mechanizes subsequent IMU increments into
a nominal position, velocity, and attitude history. The requested static
duration must be supported by real, continuous initialization evidence.

After initialization, the shared bounded IMU-recovery policy can propagate one
qualifying missing notification from held prior rates while retaining explicit
coast provenance. This remains reconstructed propagation, not recorded IMU
evidence. The optional CSV export writes the nominal trajectory and a schema
sidecar; the terminal summary reports initialization, coast, position, and
velocity facts.

This is an unaided baseline: it applies neither zero-velocity updates nor GNSS
updates. Its run-local FRD coordinates are not geographic North-East-Down.

Example:
    python scripts/run_raw_ins.py data/walk.bmu \\
        --initialization-duration-s 3.0 --csv-output results/raw_ins.csv
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from bootins.bmu.transport import iter_frames_replay
from bootins.replay.raw_ins import RawINSReplayResult, run_raw_ins_replay
from bootins.reporting.navigation_export import write_raw_ins_trajectory_csv


def _format_vector(vector: np.ndarray) -> str:
    """Format one navigation vector compactly for a terminal summary."""
    return np.array2string(vector, precision=4, suppress_small=True)


def print_summary(replay_result: RawINSReplayResult, capture_path: Path) -> None:
    """Print initialization, recovery, and terminal nominal-state facts.

    Args:
        replay_result: Completed row-aligned raw INS replay result.
        capture_path: Source capture path used as the report label.
    """
    initialization = replay_result.initialization

    print(f"Raw INS replay: {capture_path.name}")
    print("Navigation frame: run-local Forward-Right-Down")
    print()
    print("Initialization")
    print(f"  Increments: {initialization.increment_count}")
    print(f"  Duration: {initialization.duration:.3f} s")
    print(
        "  Mean specific force_b: "
        f"{_format_vector(initialization.mean_specific_force_b)} m/s^2"
    )
    print("Replay")
    print(f"  Increments: {replay_result.time.size - 1}")
    print(f"  Duration: {replay_result.time[-1]:.3f} s")
    if replay_result.coast_plans:
        print(
            "  IMU recovery: "
            f"{len(replay_result.coast_plans)} bounded coast plan(s) "
            "(raw capture contains a gap)"
        )
    else:
        print("  IMU recovery: none")
    print(f"  Final position_n: {_format_vector(replay_result.positions_n[-1])} m")
    print(f"  Final velocity_n: {_format_vector(replay_result.velocities_n[-1])} m/s")


def main() -> int:
    """Run one canonical raw INS replay and optionally export its trajectory.

    Returns:
        ``0`` after successful replay and any requested CSV export, ``1`` when
        replay input or initialization evidence is rejected, or ``2`` when a
        requested export cannot be written.

    Notes:
        The optional export writes a raw nominal-trajectory CSV plus its schema
        sidecar. Terminal output summarizes the same run-local FRD result; it
        does not geographically register or validate that trajectory.
    """
    parser = argparse.ArgumentParser(
        description="Replay a raw, unaided INS trajectory from one .bmu capture.",
    )
    parser.add_argument("capture_path", type=Path, help="Path to one .bmu capture")
    parser.add_argument(
        "--initialization-duration-s",
        dest="initialization_duration",
        type=float,
        required=True,
        metavar="SECONDS",
        help="Caller-declared static initialization duration in seconds",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        metavar="PATH",
        help=(
            "Optional raw nominal-trajectory CSV output "
            "(also writes a schema sidecar)"
        ),
    )
    args = parser.parse_args()

    try:
        replay_result = run_raw_ins_replay(
            iter_frames_replay(args.capture_path),
            args.initialization_duration,
        )
    except ValueError as error:
        print(
            f"Raw INS replay rejected {args.capture_path}: {error}",
            file=sys.stderr,
        )
        return 1

    print_summary(replay_result, args.capture_path)

    if args.csv_output is not None:
        try:
            write_raw_ins_trajectory_csv(
                args.csv_output,
                replay_result.time,
                replay_result.positions_n,
                replay_result.velocities_n,
                replay_result.quaternions_nb,
            )
        except (OSError, ValueError) as error:
            print(
                f"Could not write raw INS trajectory CSV {args.csv_output}: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"Wrote raw INS trajectory CSV: {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
