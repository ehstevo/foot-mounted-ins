"""Compute overlapping Allan deviation for uniformly sampled vector signals.

Callers provide one value per uniform sample interval and choose the integer
cluster sizes to evaluate. The statistic forms overlapping cluster averages and
reports the square root of half their successive-difference variance for each
averaging time. Results retain the input units, for example rad/s for angular
rate or m/s² for specific force.

This module does not load captures, recover rates from IMU increments, choose a
tau grid, plot curves, or identify a complete IMU process-noise model.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class AllanDeviationResult:
    """Represent an overlapping Allan-deviation curve for a vector signal.

    Attributes:
        cluster_sizes: Positive integer sample counts of shape ``(M,)`` used in
            each cluster average.
        tau: Averaging times in s with shape ``(M,)``, equal to
            ``cluster_sizes * sample_period``.
        deviation: Nonnegative Allan deviations of shape ``(M, D)``. Each
            column retains its input signal's physical unit.
    """

    cluster_sizes: np.ndarray
    tau: np.ndarray
    deviation: np.ndarray


def _validate_values(values: object) -> np.ndarray:
    """Return finite two-dimensional sample values as a float array."""
    try:
        values_array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("values must convert to a finite two-dimensional array.") from error

    if values_array.ndim != 2:
        raise ValueError("values must have shape (N, D).")
    if values_array.shape[0] == 0 or values_array.shape[1] == 0:
        raise ValueError("values must contain at least one sample and one axis.")
    if not np.isfinite(values_array).all():
        raise ValueError("values must contain only finite entries.")
    return values_array


def _validate_sample_period(sample_period: object) -> float:
    """Return one finite, strictly positive uniform sample interval."""
    if isinstance(sample_period, bool):
        raise ValueError("sample_period must be finite and positive.")
    try:
        period = float(sample_period)
    except (TypeError, ValueError) as error:
        raise ValueError("sample_period must be finite and positive.") from error
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("sample_period must be finite and positive.")
    return period


def _validate_cluster_sizes(
    cluster_sizes: Sequence[int],
    sample_count: int,
) -> np.ndarray:
    """Return increasing sizes supporting overlapping Allan differences."""
    if isinstance(cluster_sizes, (str, bytes)):
        raise ValueError("cluster_sizes must be a non-empty sequence of integers.")
    try:
        sizes = tuple(cluster_sizes)
    except TypeError as error:
        raise ValueError("cluster_sizes must be a non-empty sequence of integers.") from error

    if not sizes:
        raise ValueError("cluster_sizes must be a non-empty sequence of integers.")
    if any(isinstance(size, bool) or not isinstance(size, Integral) for size in sizes):
        raise ValueError("cluster_sizes must contain only positive integers.")

    sizes_array = np.asarray(sizes, dtype=int)
    if np.any(sizes_array <= 0):
        raise ValueError("cluster_sizes must contain only positive integers.")
    if np.any(np.diff(sizes_array) <= 0):
        raise ValueError("cluster_sizes must be strictly increasing.")
    if np.any(2 * sizes_array > sample_count):
        raise ValueError(
            "Each cluster size must leave at least one adjacent pair of averages."
        )
    return sizes_array


def overlapping_allan_deviation(
    values: np.ndarray,
    sample_period: float,
    cluster_sizes: Sequence[int],
) -> AllanDeviationResult:
    """Compute a per-axis overlapping Allan-deviation curve.

    Args:
        values: Finite samples of shape ``(N, D)`` in fixed physical units, with
            uniformly spaced rows and independent columns.
        sample_period: Positive constant intersample interval in s.
        cluster_sizes: Strictly increasing positive integer cluster counts ``m``.

    Returns:
        Cluster sizes, averaging times ``tau = m * sample_period`` in s, and
        per-axis deviations of shape ``(M, D)``.

    Raises:
        ValueError: Samples are not finite two-dimensional data, the period or
            sizes are invalid, or a requested size lacks an adjacent mean pair.

    Notes:
        Each size uses all contiguous ``m``-sample means and differences between
        means separated by ``m`` samples. Uniform sampling is a caller
        requirement; no resampling or timing repair occurs. Constant offsets,
        including static gravity or a constant bias, cancel in the differences.
    """
    values_array = _validate_values(values)
    period = _validate_sample_period(sample_period)
    sizes = _validate_cluster_sizes(cluster_sizes, values_array.shape[0])

    prefix = np.empty((values_array.shape[0] + 1, values_array.shape[1]), dtype=float)
    prefix[0] = 0.0
    prefix[1:] = np.cumsum(values_array, axis=0)

    tau = sizes * period
    deviation = np.empty((sizes.size, values_array.shape[1]), dtype=float)

    for index, size in enumerate(sizes):
        means = (prefix[size:] - prefix[:-size]) / size
        differences = means[size:] - means[:-size]
        variance = 0.5 * np.mean(differences**2, axis=0)
        deviation[index] = np.sqrt(variance)

    return AllanDeviationResult(
        cluster_sizes=sizes,
        tau=tau,
        deviation=deviation,
    )
