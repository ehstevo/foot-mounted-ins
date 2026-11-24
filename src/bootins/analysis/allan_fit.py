"""Fit declared log-log power-law regions of Allan-deviation curves.

The least-squares result describes a caller-selected positive averaging-time
interval. It summarizes a local Allan-deviation slope and fitted amplitude at
a reference time; it neither identifies a unique noise source nor produces a
complete, validated ESEKF process-noise model.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllanPowerLawFit:
    """Represent one least-squares power-law description of an Allan interval.

    The fitted model is ``deviation ≈ reference_deviation *
    (tau / reference_tau)**slope``. It is descriptive only: this result does
    not classify a noise source or choose a process covariance.

    Attributes:
        slope: Dimensionless log-log exponent of the fitted power law.
        reference_deviation: Fitted deviation at ``reference_tau``, in the input
            curve's physical unit.
        residual_rms: Root-mean-square log-space residual, dimensionless.
        sample_count: Number of selected curve points.
        tau_min: Inclusive lower averaging-time bound in s.
        tau_max: Inclusive upper averaging-time bound in s.
        reference_tau: Averaging time in s where the amplitude is reported.
    """

    slope: float
    reference_deviation: float
    residual_rms: float
    sample_count: int
    tau_min: float
    tau_max: float
    reference_tau: float


def _validate_curve(
    tau: object,
    deviation: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite positive one-dimensional curve data with ordered tau."""
    try:
        tau_array = np.asarray(tau, dtype=float)
        deviation_array = np.asarray(deviation, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("tau and deviation must convert to finite one-dimensional arrays.") from error

    if tau_array.ndim != 1 or deviation_array.ndim != 1:
        raise ValueError("tau and deviation must be one-dimensional arrays.")
    if tau_array.shape != deviation_array.shape:
        raise ValueError("tau and deviation must have the same shape.")
    if tau_array.size == 0:
        raise ValueError("tau and deviation must not be empty.")
    if not np.isfinite(tau_array).all() or not np.isfinite(deviation_array).all():
        raise ValueError("tau and deviation must contain only finite values.")
    if np.any(tau_array <= 0.0) or np.any(deviation_array <= 0.0):
        raise ValueError("tau and deviation must contain only positive values.")
    if np.any(np.diff(tau_array) <= 0.0):
        raise ValueError("tau must be strictly increasing.")

    return tau_array, deviation_array


def _validate_positive_scalar(value: object, field: str) -> float:
    """Return one finite positive scalar with a field-specific error."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and positive.")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite and positive.") from error
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{field} must be finite and positive.")
    return scalar


def fit_allan_power_law(
    tau: object,
    deviation: object,
    *,
    tau_min: object,
    tau_max: object,
    reference_tau: object = 1.0,
) -> AllanPowerLawFit:
    """Fit a caller-declared Allan-deviation interval in log space.

    Args:
        tau: Strictly increasing positive averaging times in s.
        deviation: Positive one-dimensional Allan deviations, in a consistent
            signal unit, aligned with ``tau``.
        tau_min: Inclusive lower fit bound in s.
        tau_max: Inclusive upper fit bound in s.
        reference_tau: Positive in-range time in s for the fitted amplitude.

    Returns:
        Descriptive power-law slope, reference amplitude, and log-residual facts.

    Raises:
        ValueError: Curve data or bounds are invalid, the reference is outside
            the interval, or fewer than three points match the interval.

    Notes:
        The fit is linear least squares in ``log(tau / reference_tau)`` and
        ``log(deviation)``. An apparent slope over a selected interval is not
        evidence that a noise process is unique, stationary, or directly usable
        as a filter-noise parameter.
    """
    tau_array, deviation_array = _validate_curve(tau, deviation)
    minimum = _validate_positive_scalar(tau_min, "tau_min")
    maximum = _validate_positive_scalar(tau_max, "tau_max")
    reference = _validate_positive_scalar(reference_tau, "reference_tau")

    if minimum > maximum:
        raise ValueError("tau_min must not exceed tau_max.")
    if not minimum <= reference <= maximum:
        raise ValueError("reference_tau must lie within [tau_min, tau_max].")

    mask = (tau_array >= minimum) & (tau_array <= maximum)
    selected_tau = tau_array[mask]
    selected_deviation = deviation_array[mask]
    if selected_tau.size < 3:
        raise ValueError("The selected interval must contain at least three points.")

    x = np.log(selected_tau / reference)
    y = np.log(selected_deviation)

    slope, intercept = np.polyfit(x, y, 1)

    fitted_reference_deviation = np.exp(intercept)

    residual = y - (slope * x + intercept)
    residual_rms = np.sqrt(np.mean(residual**2))

    return AllanPowerLawFit(
        slope=float(slope),
        reference_deviation=float(fitted_reference_deviation),
        residual_rms=float(residual_rms),
        sample_count=selected_tau.size,
        tau_min=minimum,
        tau_max=maximum,
        reference_tau=reference,
    )
