"""Plot already-computed Allan-deviation results without changing statistics.

These helpers consume :class:`~bootins.analysis.allan.AllanDeviationResult`
arrays. They validate the stricter positive values required by logarithmic axes,
draw one curve per measurement axis, and return caller-owned Matplotlib axes.
They do not load captures, select cluster sizes, fit noise models, display, or
save figures.
"""
from collections.abc import Sequence

from matplotlib.axes import Axes
import numpy as np

from bootins.analysis.allan import AllanDeviationResult


def _validate_log_plot_data(
    deviation_result: AllanDeviationResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, aligned data that can be represented on log-log axes.

    Zero Allan deviation is mathematically valid, but has no representation on
    a logarithmic y-axis. That plot-specific limitation is deliberately kept
    separate from the estimator's broader result contract.
    """
    try:
        tau = np.asarray(deviation_result.tau, dtype=float)
        deviation = np.asarray(deviation_result.deviation, dtype=float)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Allan result must provide numeric tau and deviation arrays."
        ) from error

    if tau.ndim != 1 or tau.size == 0:
        raise ValueError("tau must be a non-empty one-dimensional array.")
    if deviation.ndim != 2 or deviation.shape[0] != tau.size:
        raise ValueError("deviation must have shape (len(tau), axis_count).")
    if deviation.shape[1] == 0:
        raise ValueError("deviation must contain at least one measurement axis.")
    if not np.isfinite(tau).all() or not np.isfinite(deviation).all():
        raise ValueError("tau and deviation must contain only finite values.")
    if np.any(tau <= 0.0) or np.any(np.diff(tau) <= 0.0):
        raise ValueError("tau must be strictly positive and increasing.")
    if np.any(deviation <= 0.0):
        raise ValueError("deviation must be strictly positive for log-scale plotting.")

    return tau, deviation


def _axis_labels(labels: Sequence[str] | None, axis_count: int) -> tuple[str, ...]:
    """Return one non-empty legend label for every measurement axis."""
    if labels is None:
        return tuple(f"axis {index}" for index in range(axis_count))
    if isinstance(labels, str):
        raise ValueError("axis_labels must provide one label per measurement axis.")

    axis_labels = tuple(labels)
    if len(axis_labels) != axis_count or any(
        not isinstance(label, str) or not label for label in axis_labels
    ):
        raise ValueError(
            "axis_labels must provide one non-empty label per measurement axis."
        )
    return axis_labels


def plot_allan_deviation(
    deviation_result: AllanDeviationResult,
    *,
    axes: Axes | None = None,
    y_label: str,
    axis_labels: Sequence[str] | None = None,
    title: str | None = None,
) -> Axes:
    """Plot every measurement axis of one Allan result on log-log axes.

    Args:
        deviation_result: Finite positive plotting data with ``tau`` in s and
            per-axis deviation values in their original physical units.
        axes: Existing axes for composition, or ``None`` to create one.
        y_label: Display label naming the deviation quantity and its units.
        axis_labels: Optional legend label per measurement axis.
        title: Optional axes title.

    Returns:
        The populated caller-provided or newly created axes.

    Raises:
        ValueError: The result cannot be represented on log-log axes or labels
            do not align with its measurement-axis count.

    Notes:
        Zero Allan deviation is statistically valid but intentionally refused
        here because it is not representable on a logarithmic y-axis. The plot
        presents statistics only; it does not infer or fit a sensor-noise model.
    """
    tau, deviation = _validate_log_plot_data(deviation_result)
    labels = _axis_labels(axis_labels, deviation.shape[1])

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots()

    for axis_index, label in enumerate(labels):
        axes.loglog(tau, deviation[:, axis_index], label=label)

    axes.set_xlabel("Averaging time (s)")
    axes.set_ylabel(y_label)
    # Configure each axis explicitly because a Matplotlib style may already
    # have enabled minor grids. Major grids mark equal log-decade intervals
    # without visual clutter from minor values inside each decade.
    axes.xaxis.grid(True, which="major")
    axes.yaxis.grid(True, which="major")
    axes.xaxis.grid(False, which="minor")
    axes.yaxis.grid(False, which="minor")
    if title is not None:
        axes.set_title(title)
    axes.legend()

    return axes
