"""Replay one IMU capture and GNSS-fix table with canonical GNSS + INS + ZUPT aiding.

The command consumes framed ``.bmu`` IMU evidence, a normalized receiver GNSS
fix table, and one GNSS + INS + ZUPT YAML profile. Startup identifies an initial
stationary interval, derives a course from receiver fixes, and uses a terminal
stationary interval to initialize attitude and gyroscope bias. Terminal-static
receiver evidence defines the fixed geographic local-NED origin, while the seed
fix separately defines GNSS-relative time zero.

During replay, IMU increments drive causal ESEKF prediction and optional ZUPT
updates. The clock state maps receiver measurement epochs to the IMU timeline;
host arrival order determines only when a fix becomes available to the causal
scheduler. Delayed updates rewind to an eligible IMU posterior, gate the GNSS
innovation, and replay subsequent IMU propagation. Receiver measurement time,
host availability time, and mapped IMU update time therefore remain distinct.

The optional exports provide aligned geographic-NED state/covariance history,
the associated geographic-origin sidecar, and per-fix fusion diagnostics. PNG
plots and the optional interactive LLH map are presentation artifacts; their
receiver layers preserve source evidence and fusion outcomes. An accepted update
is statistically consistent with the current model, not proof that its receiver
position is correct; rejected, stale, future-dropped, invalid, or unavailable
fixes do not constitute accepted navigation corrections. Bounded synthetic IMU
coasts, when used, remain identified as reconstructed rather than raw evidence.

Example:
    python scripts/run_gnss_zupt_replay.py data/runs/<run>/imu.bmu \\
        --gnss-fix-table data/runs/<run>/gnss_fixes.csv \\
        --config config/gnss_ins_zupt/example.yaml \\
        --csv-output results/aided.csv \\
        --gnss-diagnostics-output results/aided_gnss.csv
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from bootins.bmu.transport import iter_frames_replay
from bootins.estimation.clock_augmented_esekf import NOM_TAU
from bootins.reporting.interactive_maps import (
    FoliumBasemapStyle,
    GnssMapStatus,
    write_llh_trajectory_map,
)
from bootins.frames.geodesy import llh_to_ned, ned_to_llh
from bootins.sensors.gnss_observations import GnssFixLLH
from bootins.reporting.gnss_fusion_export import write_gnss_fusion_diagnostics_csv
from bootins.sensors.gnss_table import load_gnss_fix_table
from bootins.config.gnss_ins_zupt import load_gnss_zupt_config
from bootins.replay.gnss_fusion_models import DelayedGnssUpdateStatus
from bootins.replay.gnss_ins_zupt import GnssZuptReplayResult, run_gnss_zupt_replay
from bootins.reporting.navigation_export import (
    NavigationHistoryLayout,
    write_esekf_history_csv,
    write_geographic_ned_origin_sidecar,
)
from bootins.reporting.trajectory_plots import (
    BasemapStyle,
    HorizontalFrame,
    plot_horizontal_trajectory,
    plot_llh_trajectories,
)


def _format_vector(vector: np.ndarray) -> str:
    """Format one navigation vector compactly for a terminal summary."""
    return np.array2string(vector, precision=4, suppress_small=True)


def _gnss_fixes_for_plot(
    gnss_fixes: tuple[GnssFixLLH, ...],
    replay_result: GnssZuptReplayResult,
) -> tuple[GnssFixLLH, ...]:
    """Select receiver-valid fixes over the aided replay's GNSS-time span.

    Args:
        gnss_fixes: All normalized receiver fixes in host-arrival order.
        replay_result: Completed aided replay defining seed-relative duration.

    Returns:
        Receiver-valid fixes whose raw GNSS epochs overlap the post-seed replay
        interval, regardless of later fusion outcome.

    Raises:
        ValueError: No receiver-valid fix overlaps the plotted interval.

    Notes:
        The seed fix defines GNSS-time zero only. LLH/NED conversion uses the
        distinct terminal-static origin, and selection does not judge whether
        a receiver fix is physically accurate.
    """
    seed_fix = replay_result.startup.seed_fix
    start_t_fix_g = seed_fix.t_fix_g
    end_t_fix_g = start_t_fix_g + replay_result.time[-1]
    selected_fixes = tuple(
        fix
        for fix in gnss_fixes
        if fix.fix_valid and start_t_fix_g <= fix.t_fix_g <= end_t_fix_g
    )
    if not selected_fixes:
        raise ValueError(
            "No receiver-valid GNSS fixes overlap the plotted post-seed interval."
        )
    return selected_fixes


def _gnss_positions_ned_for_plot(
    gnss_fixes: tuple[GnssFixLLH, ...],
    replay_result: GnssZuptReplayResult,
) -> np.ndarray:
    """Convert selected receiver fixes to the static-origin local NED frame.

    Args:
        gnss_fixes: All normalized receiver fixes in host-arrival order.
        replay_result: Completed aided replay providing the fixed LLH origin.

    Returns:
        ``(N, 3)`` receiver positions in metres as North-East-Down coordinates.
    """
    static_origin = replay_result.startup.static_origin
    positions = [
        llh_to_ned(
            fix.latitude,
            fix.longitude,
            fix.height_ellipsoid,
            static_origin.latitude,
            static_origin.longitude,
            static_origin.height_ellipsoid,
        )
        for fix in _gnss_fixes_for_plot(gnss_fixes, replay_result)
    ]
    return np.asarray(positions, dtype=float)


def _gnss_positions_llh_for_plot(
    gnss_fixes: tuple[GnssFixLLH, ...],
    replay_result: GnssZuptReplayResult,
) -> np.ndarray:
    """Return selected receiver positions for LLH map presentation.

    Args:
        gnss_fixes: All normalized receiver fixes in host-arrival order.
        replay_result: Completed aided replay defining the displayed time span.

    Returns:
        ``(N, 3)`` latitude, longitude, and ellipsoidal-height rows, in radians,
        radians, and metres respectively.
    """
    return np.asarray(
        [
            (fix.latitude, fix.longitude, fix.height_ellipsoid)
            for fix in _gnss_fixes_for_plot(gnss_fixes, replay_result)
        ],
        dtype=float,
    )


def _navigation_positions_llh_for_plot(
    replay_result: GnssZuptReplayResult,
) -> np.ndarray:
    """Convert aided local-NED navigation positions to map-ready LLH rows.

    Args:
        replay_result: Completed aided replay with a fixed static LLH origin.

    Returns:
        ``(N, 3)`` latitude, longitude, and ellipsoidal-height rows, in radians,
        radians, and metres respectively.
    """
    static_origin = replay_result.startup.static_origin
    return np.asarray(
        [
            ned_to_llh(
                position_ned,
                static_origin.latitude,
                static_origin.longitude,
                static_origin.height_ellipsoid,
            )
            for position_ned in replay_result.nominals[:, :3]
        ],
        dtype=float,
    )


def _gnss_map_status(status: DelayedGnssUpdateStatus) -> GnssMapStatus:
    """Translate one terminal fusion status to a map-layer category.

    Args:
        status: Delayed GNSS update outcome retained by replay.

    Returns:
        Corresponding presentation-only GNSS map status.
    """
    statuses = {
        DelayedGnssUpdateStatus.ACCEPTED: GnssMapStatus.ACCEPTED,
        DelayedGnssUpdateStatus.REJECTED: GnssMapStatus.REJECTED,
        DelayedGnssUpdateStatus.STALE: GnssMapStatus.STALE,
        DelayedGnssUpdateStatus.FUTURE_DROPPED: GnssMapStatus.FUTURE_DROPPED,
    }
    return statuses[status]


def _gnss_map_evidence_for_plot(
    gnss_fixes: tuple[GnssFixLLH, ...],
    replay_result: GnssZuptReplayResult,
) -> tuple[np.ndarray, tuple[GnssMapStatus, ...], tuple[str, ...]]:
    """Build map positions, fusion categories, and evidence tooltips.

    Args:
        gnss_fixes: All normalized receiver fixes in host-arrival order.
        replay_result: Completed aided replay and per-fix fusion diagnostics.

    Returns:
        ``(positions_llh, statuses, tooltips)`` for the selected receiver-valid
        fixes. LLH rows use radians, radians, and metres.

    Notes:
        Every receiver-valid fix over the seed-relative span remains visible.
        Startup or otherwise nonterminal fixes have no fusion diagnostic and
        are labeled ``NOT_EVALUATED`` rather than omitted or treated as
        rejected. This presentation bridge does not assess physical accuracy.
    """
    diagnostics_by_native_record = {
        diagnostic.fix.native_record_index: diagnostic
        for diagnostic in replay_result.gnss_fusion_diagnostics
    }
    positions: list[tuple[float, float, float]] = []
    statuses: list[GnssMapStatus] = []
    tooltips: list[str] = []
    for fix in _gnss_fixes_for_plot(gnss_fixes, replay_result):
        diagnostic = diagnostics_by_native_record.get(fix.native_record_index)
        if diagnostic is None:
            status = GnssMapStatus.NOT_EVALUATED
            details = ["Fusion outcome: not evaluated"]
        else:
            status = _gnss_map_status(diagnostic.update.status)
            details = [f"Fusion outcome: {status.value.replace('_', ' ')}"]
            if diagnostic.update.nis is not None:
                details.append(f"NIS: {diagnostic.update.nis:.3f}")
            if diagnostic.innovation_ned is not None:
                horizontal_innovation = float(
                    np.linalg.norm(diagnostic.innovation_ned[:2])
                )
                details.append(
                    f"Horizontal innovation: {horizontal_innovation:.3f} m"
                )
            if diagnostic.update.t_applied_i is not None:
                details.append(
                    f"Applied IMU epoch: {diagnostic.update.t_applied_i:.3f} s"
                )
        positions.append((fix.latitude, fix.longitude, fix.height_ellipsoid))
        statuses.append(status)
        tooltips.append(
            "\n".join(
                (
                    f"GPS week: {fix.gps_week}",
                    f"GPS time of week: {fix.gps_tow:.3f} s",
                    f"Horizontal accuracy: {fix.horizontal_accuracy:.3f} m",
                    f"Satellites: {fix.satellite_count}",
                    *details,
                )
            )
        )
    return np.asarray(positions, dtype=float), tuple(statuses), tuple(tooltips)


def print_summary(
    replay_result: GnssZuptReplayResult,
    capture_path: Path,
    gnss_fix_table_path: Path,
    config_path: Path,
) -> None:
    """Print startup, state, and GNSS-delivery facts without full histories.

    Args:
        replay_result: Completed aided replay result and fusion diagnostics.
        capture_path: Source IMU capture path used as the report label.
        gnss_fix_table_path: Source normalized GNSS table used as the label.
        config_path: Loaded aided navigation profile used as the report label.

    Notes:
        Counts describe delivery and current gating outcomes, not independent
        receiver accuracy or trajectory validation.
    """
    startup = replay_result.startup
    initial_interval = startup.initial_interval
    terminal_interval = startup.terminal_interval
    terminal_initialization = startup.terminal_initialization
    terminal_static = terminal_initialization.terminal_static
    course = terminal_initialization.course
    update_count = int(np.count_nonzero(replay_result.stance_flags))
    replay_increment_count = replay_result.stance_flags.size
    accepted_count = sum(
        update.status is DelayedGnssUpdateStatus.ACCEPTED
        for update in replay_result.gnss_updates
    )
    rejected_count = sum(
        update.status is DelayedGnssUpdateStatus.REJECTED
        for update in replay_result.gnss_updates
    )
    stale_count = sum(
        update.status is DelayedGnssUpdateStatus.STALE
        for update in replay_result.gnss_updates
    )
    future_dropped = sum(
        update.status is DelayedGnssUpdateStatus.FUTURE_DROPPED
        for update in replay_result.gnss_updates
    )

    print(f"GNSS + INS + ZUPT replay: {capture_path.name}")
    print(f"GNSS fix table: {gnss_fix_table_path.name}")
    print(f"Configuration: {config_path.name}")
    print("Navigation frame: static-origin local geographic North-East-Down")
    print()
    print("Initialization")
    print(f"  Initial stance increments: {initial_interval.increment_count}")
    print(f"  Initial stance duration: {initial_interval.duration:.3f} s")
    print(f"  GNSS course fixes: {course.fix_count}")
    print(f"  GNSS course: {np.degrees(course.course):.3f} deg clockwise from North")
    print(f"  GNSS course baseline: {course.baseline:.3f} m")
    print(f"  GNSS course cross-track RMS: {course.cross_track_rms:.3f} m")
    print(f"  Terminal stance increments: {terminal_interval.increment_count}")
    print(f"  Terminal stance duration: {terminal_interval.duration:.3f} s")
    print(
        "  Terminal mean specific force_b: "
        f"{_format_vector(terminal_static.static_level.mean_specific_force_b)} m/s^2"
    )
    print(
        "  Terminal estimated gyro bias_b: "
        f"{_format_vector(terminal_static.gyro_bias_b)} rad/s"
    )
    print("  Static GNSS origin: defines local position frame")
    print("  Seed fix: defines GNSS-time zero")
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
    print(
        "  Final clock offset: "
        f"{replay_result.nominals[-1, NOM_TAU]:.6f} s"
    )
    print("GNSS delivery and fusion")
    print(f"  Valid GNSS outcomes: {len(replay_result.gnss_updates)}")
    print(f"  Accepted: {accepted_count}")
    print(f"  NIS rejected: {rejected_count}")
    print(f"  Stale: {stale_count}")
    print(f"  Future-mapped dropped: {future_dropped}")
    print(
        "  Receiver-invalid scheduled fixes: "
        f"{replay_result.invalid_gnss_fix_count}"
    )
    print(
        "  Scheduler drops before first posterior: "
        f"{replay_result.summary.dropped_before_imu_count}"
    )
    print(
        "  Scheduler drops after final posterior: "
        f"{replay_result.summary.dropped_after_imu_count}"
    )


def main() -> int:
    """Run one canonical GNSS + INS + ZUPT replay and requested artifacts.

    Returns:
        ``0`` after successful replay and requested outputs, ``1`` when replay
        evidence is rejected, or ``2`` when configuration/table/capture I/O or
        an output artifact fails.

    Notes:
        The state CSV uses static-origin geographic NED and receives a matching
        LLH-origin sidecar. Fusion diagnostics retain receiver and update facts.
        Static, global-map, and interactive-map outputs only render replay
        evidence; visually plausible paths do not establish navigation accuracy.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Replay a GNSS + INS + ZUPT trajectory from one .bmu capture "
            "and normalized GNSS fix table."
        ),
    )
    parser.add_argument("capture_path", type=Path, help="Path to one .bmu capture")
    parser.add_argument(
        "--gnss-fix-table",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to one normalized GNSS fix-table CSV",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to one GNSS + INS + ZUPT YAML configuration",
    )
    parser.add_argument(
            "--gnss-outage-after",
            type=float,
            help="Seconds relative to filter startup after which GNSS measurements will be excluded",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        metavar="PATH",
        help="Optional clock-aware ESEKF-history CSV output path",
    )
    parser.add_argument(
        "--gnss-diagnostics-output",
        type=Path,
        metavar="PATH",
        help="Optional per-fix GNSS-fusion diagnostics CSV output path",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        metavar="PATH",
        help="Optional geographic horizontal-trajectory plot output path",
    )
    parser.add_argument(
        "--plot-frame",
        choices=("local-ned", "global-map"),
        default="local-ned",
        help="Coordinate presentation for --plot-output (default: local-ned)",
    )
    parser.add_argument(
        "--basemap",
        choices=tuple(style.value for style in BasemapStyle),
        metavar="STYLE",
        help="Tiled background for --plot-frame global-map"
        "Options: satellite (default), streets, light, topographic",
    )
    parser.add_argument(
        "--plot-gnss-fixes",
        action="store_true",
        help=(
            "Overlay receiver-valid GNSS fixes on --plot-output in the "
            "static-origin local NED frame"
        ),
    )
    parser.add_argument(
        "--html-map-output",
        type=Path,
        metavar="PATH.html",
        help=(
            "Optional interactive Folium LLH map with layered receiver-valid "
            "GNSS fusion outcomes"
        ),
    )
    parser.add_argument(
        "--html-map-basemap",
        choices=tuple(style.value for style in FoliumBasemapStyle),
        metavar="STYLE",
        help=(
            "Tiled background for --html-map-output: streets, satellite, light, "
            "or topographic (default: light)"
        ),
    )
    args = parser.parse_args()
    if args.plot_gnss_fixes and args.plot_output is None:
        parser.error("--plot-gnss-fixes requires --plot-output.")
    if args.plot_frame == "global-map" and args.plot_output is None:
        parser.error("--plot-frame global-map requires --plot-output.")
    if args.basemap is not None and args.plot_frame != "global-map":
        parser.error("--basemap requires --plot-frame global-map.")
    if args.html_map_basemap is not None and args.html_map_output is None:
        parser.error("--html-map-basemap requires --html-map-output.")
    if (
        args.csv_output is not None
        and args.gnss_diagnostics_output is not None
        and args.csv_output.resolve() == args.gnss_diagnostics_output.resolve()
    ):
        parser.error(
            "--csv-output and --gnss-diagnostics-output must use different paths."
        )

    try:
        config = load_gnss_zupt_config(args.config)
    except OSError as error:
        print(
            f"Could not read GNSS + INS + ZUPT configuration {args.config}: {error}",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(
            f"GNSS + INS + ZUPT configuration rejected {args.config}: {error}",
            file=sys.stderr,
        )
        return 2

    try:
        gnss_fixes = load_gnss_fix_table(args.gnss_fix_table)
    except OSError as error:
        print(
            f"Could not read GNSS fix table {args.gnss_fix_table}: {error}",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(
            f"GNSS fix table rejected {args.gnss_fix_table}: {error}",
            file=sys.stderr,
        )
        return 2

    try:
        replay_result = run_gnss_zupt_replay(
            iter_frames_replay(args.capture_path),
            gnss_fixes,
            config,
            gnss_outage=args.gnss_outage_after,
        )
    except OSError as error:
        print(f"Could not read capture {args.capture_path}: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(
            f"GNSS + INS + ZUPT replay rejected {args.capture_path}: {error}",
            file=sys.stderr,
        )
        return 1

    print_summary(
        replay_result,
        args.capture_path,
        args.gnss_fix_table,
        args.config,
    )

    if args.csv_output is not None:
        try:
            write_esekf_history_csv(
                args.csv_output,
                replay_result.time,
                replay_result.nominals,
                replay_result.covariances,
                zupt_applied=np.concatenate(([False], replay_result.stance_flags)),
                layout=NavigationHistoryLayout.GEOGRAPHIC_NED_CLOCK,
            )
            static_origin = replay_result.startup.static_origin
            write_geographic_ned_origin_sidecar(
                args.csv_output,
                static_origin.latitude,
                static_origin.longitude,
                static_origin.height_ellipsoid,
            )
        except (OSError, ValueError) as error:
            print(
                f"Could not write navigation history CSV {args.csv_output}: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"Wrote navigation history CSV: {args.csv_output}")
        print(
            "Wrote geographic NED origin sidecar: "
            f"{args.csv_output.with_suffix('.georef.yaml')}"
        )

    if args.gnss_diagnostics_output is not None:
        try:
            write_gnss_fusion_diagnostics_csv(
                args.gnss_diagnostics_output,
                replay_result.gnss_fusion_diagnostics,
            )
        except (OSError, ValueError) as error:
            print(
                "Could not write GNSS fusion diagnostics CSV "
                f"{args.gnss_diagnostics_output}: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"Wrote GNSS fusion diagnostics CSV: {args.gnss_diagnostics_output}")

    if args.plot_output is not None:
        try:
            if args.plot_frame == "local-ned":
                trajectory_plot = plot_horizontal_trajectory(
                    replay_result.nominals[:, :3],
                    frame=HorizontalFrame.GEOGRAPHIC_NED,
                    trajectory_label="GNSS + INS + ZUPT",
                    title=f"GNSS + INS + ZUPT replay: {args.capture_path.stem}",
                )
                if args.plot_gnss_fixes:
                    plot_horizontal_trajectory(
                        _gnss_positions_ned_for_plot(gnss_fixes, replay_result),
                        frame=HorizontalFrame.GEOGRAPHIC_NED,
                        axes=trajectory_plot.axes,
                        trajectory_label="M8T receiver fixes",
                    )
            else:
                basemap = BasemapStyle(
                    args.basemap if args.basemap is not None else "satellite"
                )
                map_trajectories = [
                    (
                        _navigation_positions_llh_for_plot(replay_result),
                        "GNSS + INS + ZUPT",
                    )
                ]
                if args.plot_gnss_fixes:
                    map_trajectories.append(
                        (
                            _gnss_positions_llh_for_plot(gnss_fixes, replay_result),
                            "M8T receiver fixes",
                        )
                    )
                trajectory_plot = plot_llh_trajectories(
                    map_trajectories,
                    basemap=basemap,
                    title=f"GNSS + INS + ZUPT replay: {args.capture_path.stem}",
                )
            trajectory_plot.axes.figure.savefig(args.plot_output, bbox_inches="tight")
        except (OSError, RuntimeError, ValueError) as error:
            print(
                f"Could not write trajectory plot {args.plot_output}: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"Wrote trajectory plot: {args.plot_output}")

    if args.html_map_output is not None:
        try:
            gnss_positions, gnss_statuses, gnss_tooltips = _gnss_map_evidence_for_plot(
                gnss_fixes,
                replay_result,
            )
            write_llh_trajectory_map(
                args.html_map_output,
                _navigation_positions_llh_for_plot(replay_result),
                gnss_positions_llh=gnss_positions,
                gnss_statuses=gnss_statuses,
                gnss_tooltips=gnss_tooltips,
                basemap=FoliumBasemapStyle(
                    args.html_map_basemap
                    if args.html_map_basemap is not None
                    else "light"
                ),
                title=f"GNSS + INS + ZUPT replay: {args.capture_path.stem}",
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(
                f"Could not write interactive trajectory map {args.html_map_output}: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"Wrote interactive trajectory map: {args.html_map_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
