"""Analyze clean IMU increments with overlapping Allan deviation.

The command converts validated body-frame delta measurements to angular-rate
(``dtheta / dt``, rad/s) and specific-force (``dv / dt``, m/s²) histories. It
requires near-uniform IMU integration intervals for the standard discrete
estimator and evaluates gyroscope and accelerometer data on one shared
logarithmic averaging-time grid before writing a two-panel plot.

By default, the entire capture must be continuous and structurally clean. The
explicit ``--longest-contiguous`` option instead analyzes the first longest
clean segment and reports its provenance; it does not repair missing data.
Optional caller-declared log-log power-law fits are descriptive curve summaries,
not noise classification or a complete ESEKF process-covariance derivation.

Example:
    python scripts/analyze_allan.py data/rest.bmu --output plots/allan.png
"""
import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bootins.analysis.allan import AllanDeviationResult, overlapping_allan_deviation
from bootins.reporting.allan_plots import plot_allan_deviation
from bootins.analysis.allan_fit import AllanPowerLawFit, fit_allan_power_law
from bootins.bmu.transport import iter_frames_replay
from bootins.replay.imu_integrity import (
    CleanIncrementSegment,
    collect_clean_increments,
    select_longest_clean_increment_segment,
)

MAX_RELATIVE_SAMPLE_PERIOD_DEPARTURE = 0.002
# Maximum analyzed Allan averaging time, in seconds.
MAX_TAU = 2000.0
# Target number of logarithmically distributed cluster sizes.
GRID_POINT_COUNT = 100


def uniform_sample_period(dt: np.ndarray) -> float:
    """Return the representative period of a sufficiently uniform IMU history.

    Args:
        dt: Positive IMU integration intervals in seconds, shape ``(N,)``.

    Returns:
        Mean IMU sample period in seconds.

    Raises:
        ValueError: If the greatest relative departure from the mean exceeds
            the configured standard-discrete-Allan tolerance.

    Notes:
        Individual intervals recover rate samples, but the discrete Allan
        estimator needs one representative period to map cluster size to
        physical averaging time.
    """
    sample_period = np.mean(dt)
    largest_departure = np.max(np.abs(dt - sample_period)) / sample_period
    if largest_departure > MAX_RELATIVE_SAMPLE_PERIOD_DEPARTURE:
        raise ValueError(
            "IMU sample intervals are not sufficiently uniform for standard "
            f"Allan deviation: maximum relative departure {largest_departure:.3%} "
            f"exceeds {MAX_RELATIVE_SAMPLE_PERIOD_DEPARTURE:.3%}"
        )

    return sample_period


def choose_cluster_sizes(sample_period: float, sample_count: int) -> np.ndarray:
    """Choose a shared logarithmic cluster-size grid for one rate history.

    Args:
        sample_period: Representative IMU period in seconds.
        sample_count: Number of rate samples available.

    Returns:
        Strictly increasing positive integer cluster sizes.

    Raises:
        ValueError: If fewer than two rate samples can form an Allan pair.

    Notes:
        The largest cluster is limited by ``MAX_TAU`` and by the need for an
        adjacent pair of cluster averages.
    """
    max_m = int(min(np.floor(MAX_TAU / sample_period), sample_count // 2))
    if max_m < 1:
        raise ValueError(
            "At least two rate samples are required to form an Allan pair at m = 1."
        )

    cluster_sizes = np.unique(
        np.rint(np.geomspace(1, max_m, GRID_POINT_COUNT)).astype(int)
    )

    return cluster_sizes


def capture_to_rate(
    path: Path,
    *,
    longest_contiguous: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, CleanIncrementSegment | None]:
    """Convert a clean capture or selected segment into rate histories.

    Args:
        path: Framed ``.bmu`` capture containing delta measurements.
        longest_contiguous: Select the first longest clean segment instead of
            requiring the complete capture to be continuous and clean.

    Returns:
        ``(omega, f, sample_period, selected_segment)``, where ``omega`` has
        units rad/s and ``f`` has units m/s², each with shape ``(N, 3)``;
        ``sample_period`` is in seconds; and ``selected_segment`` is ``None``
        for strict whole-capture analysis.

    Raises:
        ValueError: If the selected evidence fails integrity requirements or
            its intervals are unsuitable for standard Allan analysis.

    Notes:
        Every accepted ``dtheta`` and ``dv`` is divided by its own ``dt``
        before the uniform-time check. Segment selection is diagnostic only and
        does not repair gaps or make the result canonical replay input.
    """
    raw_frames = iter_frames_replay(path)
    if longest_contiguous:
        selected_segment = select_longest_clean_increment_segment(raw_frames)
        increments = selected_segment.increments
    else:
        selected_segment = None
        increments = collect_clean_increments(raw_frames)

    dtheta = np.vstack([increment[0] for increment in increments])
    dv = np.vstack([increment[1] for increment in increments])
    dt = np.asarray([increment[2] for increment in increments], dtype=float)

    omega = dtheta / dt[:, None]
    f = dv / dt[:, None]

    sample_period = uniform_sample_period(dt)

    return omega, f, sample_period, selected_segment


def analyze_rates(
    omega: np.ndarray,
    f: np.ndarray,
    sample_period: float,
) -> tuple[AllanDeviationResult, AllanDeviationResult]:
    """Evaluate gyro and accelerometer histories on one shared tau grid.

    Args:
        omega: Body-frame angular-rate history in rad/s, shape ``(N, 3)``.
        f: Body-frame specific-force history in m/s², shape ``(N, 3)``.
        sample_period: Representative IMU sample period in seconds.

    Returns:
        Gyroscope and accelerometer Allan-deviation results sharing identical
        averaging times.
    """
    sample_count = omega.shape[0]

    cluster_sizes = choose_cluster_sizes(sample_period, sample_count)

    allan_omega = overlapping_allan_deviation(omega, sample_period, cluster_sizes)
    allan_f = overlapping_allan_deviation(f, sample_period, cluster_sizes)

    return allan_omega, allan_f


def fit_allan_axes(
    result: AllanDeviationResult,
    *,
    tau_min: float,
    tau_max: float,
    reference_tau: float,
) -> tuple[AllanPowerLawFit, AllanPowerLawFit, AllanPowerLawFit]:
    """Fit all three axes over one caller-declared Allan interval.

    Args:
        result: One three-axis Allan-deviation result.
        tau_min: Inclusive lower fit bound in seconds.
        tau_max: Inclusive upper fit bound in seconds.
        reference_tau: Averaging time in seconds for reported fitted deviation.

    Returns:
        Independent x-, y-, and z-axis power-law fit summaries.
    """
    allan_fit_x = fit_allan_power_law(
        tau=result.tau,
        deviation=result.deviation[:, 0],
        tau_min=tau_min,
        tau_max=tau_max,
        reference_tau=reference_tau,
    )
    allan_fit_y = fit_allan_power_law(
        tau=result.tau,
        deviation=result.deviation[:, 1],
        tau_min=tau_min,
        tau_max=tau_max,
        reference_tau=reference_tau,
    )
    allan_fit_z = fit_allan_power_law(
        tau=result.tau,
        deviation=result.deviation[:, 2],
        tau_min=tau_min,
        tau_max=tau_max,
        reference_tau=reference_tau,
    )

    return (allan_fit_x, allan_fit_y, allan_fit_z)


def print_fit_summary(
    gyro_fits: tuple[AllanPowerLawFit, AllanPowerLawFit, AllanPowerLawFit],
    accel_fits: tuple[AllanPowerLawFit, AllanPowerLawFit, AllanPowerLawFit],
) -> None:
    """Print descriptive three-axis power-law fit summaries.

    Args:
        gyro_fits: Gyroscope fits with deviations in rad/s.
        accel_fits: Accelerometer fits with deviations in m/s².

    Notes:
        Reported slopes and residuals describe only the caller-selected
        intervals; this function does not classify noise or choose filter
        process covariance.
    """
    first_fit = gyro_fits[0]
    print()
    print("Declared Allan power-law fits")
    print(f"  Declared interval: [{first_fit.tau_min:.6g}, {first_fit.tau_max:.6g}] s")
    print(f"  Reference tau: {first_fit.reference_tau:.6g} s")

    for sensor_name, units, fits in (
        ("Gyroscope", "rad/s", gyro_fits),
        ("Accelerometer", "m/s²", accel_fits),
    ):
        print()
        print(f"  {sensor_name} ({units})")
        print("    Axis      Slope  Fitted sigma_A(ref)  Log residual RMS  Points")
        for axis, fit in zip(("x", "y", "z"), fits, strict=True):
            print(
                f"    {axis:>4}  {fit.slope:>9.4f}  "
                f"{fit.reference_deviation:>19.6e}  "
                f"{fit.residual_rms:>16.4f}  {fit.sample_count:>6}"
            )


def main() -> int:
    """Analyze one capture, write its Allan plot, and optionally print fits.

    Returns:
        ``0`` after successful analysis and plot output, ``1`` for invalid
        capture evidence or analysis/rendering failure, or ``2`` for capture
        input or plot-output I/O failure.

    Notes:
        CLI fit bounds choose descriptive power-law summaries only; they do not
        identify a complete stochastic sensor model.
    """
    parser = argparse.ArgumentParser(
        description="Calculate Allan deviation from a .bmu capture."
    )
    parser.add_argument("file", type=Path, help="one .bmu file to analyze")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path for the output image",
    )
    parser.add_argument(
        "--longest-contiguous",
        action="store_true",
        help=(
            "analyze the first longest clean contiguous segment instead of "
            "rejecting gaps or structural concerns"
        ),
    )
    parser.add_argument(
        "--fit-tau-min",
        type=float,
        help="lower averaging-time bound for a declared power-law fit",
    )
    parser.add_argument(
        "--fit-tau-max",
        type=float,
        help="upper averaging-time bound for a declared power-law fit",
    )
    parser.add_argument(
        "--fit-reference-tau",
        type=float,
        help="report fitted deviation at this tau; defaults to 1 s when fitting",
    )
    args = parser.parse_args()

    has_fit_min = args.fit_tau_min is not None
    has_fit_max = args.fit_tau_max is not None
    has_fit_reference = args.fit_reference_tau is not None
    if has_fit_min != has_fit_max:
        parser.error("--fit-tau-min and --fit-tau-max must be supplied together")
    if has_fit_reference and not has_fit_min:
        parser.error("--fit-reference-tau requires both fit bounds")
    fit_requested = has_fit_min

    try:
        omega, f, sample_period, selected_segment = capture_to_rate(
            args.file,
            longest_contiguous=args.longest_contiguous,
        )
        gyro_result, accel_result = analyze_rates(omega, f, sample_period)
    except OSError as error:
        print(f"Could not read Allan capture {args.file}: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"Allan capture rejected {args.file}: {error}", file=sys.stderr)
        return 1

    figure = None
    if selected_segment is None:
        title_suffix = ""
        selection_summary = (
            "  Selection: complete capture "
            "(strictly continuous and structurally clean)\n"
            f"  Selected increments: {omega.shape[0]}\n"
        )
    else:
        title_suffix = " (longest clean contiguous segment)"
        selection_summary = (
            "  Selection: first longest clean contiguous segment\n"
            "  Frame/measurement bounds: "
            f"{selected_segment.frame_start_index}/"
            f"{selected_segment.measurement_start_index} -> "
            f"{selected_segment.frame_end_index}/"
            f"{selected_segment.measurement_end_index} (inclusive)\n"
            f"  Selected increments: {len(selected_segment.increments)}\n"
        )
    gyro_fits: tuple[AllanPowerLawFit, AllanPowerLawFit, AllanPowerLawFit] | None = None
    accel_fits: tuple[AllanPowerLawFit, AllanPowerLawFit, AllanPowerLawFit] | None = None
    if fit_requested:
        reference_tau = (
            1.0 if args.fit_reference_tau is None else args.fit_reference_tau
        )
        try:
            gyro_fits = fit_allan_axes(
                gyro_result,
                tau_min=args.fit_tau_min,
                tau_max=args.fit_tau_max,
                reference_tau=reference_tau,
            )
            accel_fits = fit_allan_axes(
                accel_result,
                tau_min=args.fit_tau_min,
                tau_max=args.fit_tau_max,
                reference_tau=reference_tau,
            )
        except ValueError as error:
            print(f"Could not fit Allan curves for {args.file}: {error}", file=sys.stderr)
            return 1
    try:
        figure, (gyro_axes, accel_axes) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(8.0, 9.0),
            constrained_layout=True,
        )
        figure.suptitle(f"Allan deviation: {args.file.name}{title_suffix}")

        axis_labels = ("x", "y", "z")
        plot_allan_deviation(
            gyro_result,
            axes=gyro_axes,
            y_label="Angular-rate Allan deviation (rad/s)",
            axis_labels=axis_labels,
            title="Gyroscope",
        )
        plot_allan_deviation(
            accel_result,
            axes=accel_axes,
            y_label="Specific-force Allan deviation (m/s²)",
            axis_labels=axis_labels,
            title="Accelerometer",
        )
        gyro_axes.set_xlabel("")
        figure.savefig(args.output, dpi=200)
    except OSError as error:
        print(f"Could not write Allan plot {args.output}: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"Could not render Allan plot {args.output}: {error}", file=sys.stderr)
        return 1
    finally:
        if figure is not None:
            plt.close(figure)

    sample_rate = 1.0 / sample_period
    print(
        f"Analyzed {args.file}\n"
        f"{selection_summary}"
        f"  IMU sample period: {sample_period:.9f} s ({sample_rate:.3f} Hz)\n"
        f"  Allan tau range: {gyro_result.tau[0]:.6g} to "
        f"{gyro_result.tau[-1]:.6g} s ({gyro_result.tau.size} points)\n"
        f"  Wrote plot: {args.output}"
    )
    if gyro_fits is not None:
        assert accel_fits is not None
        print_fit_summary(gyro_fits, accel_fits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
