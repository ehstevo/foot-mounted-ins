"""Define immutable public evidence records for clock-aware GNSS replay.

The integrated replay retains IMU events and post-event posteriors so an
available GNSS measurement can update its applicable historical epoch and
reconstruct later states. These records make the resulting event, timing, and
diagnostic evidence independently owned by public consumers. They do not
implement scheduling, clock mapping, filtering, or replay.

In particular, a NIS-rejected update is evaluated and therefore can carry
pre-update innovation evidence. A stale or future-dropped update is never
evaluated and must not carry that evidence. Arrays exposed by diagnostic and
posterior records are copied and read-only where the existing ownership
boundary requires it.
"""
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from bootins.estimation.mechanization import Increment
from bootins.sensors.gnss_observations import GnssFixLLH


_TIMELINE_ATOL = 1e-6
_COVARIANCE_ATOL = 1e-12
_COVARIANCE_RTOL = 1e-10

def _owned_finite_array(
    value: object,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray:
    """Copy, validate, and freeze one finite array.

    Args:
        value: Array-like input to copy.
        shape: Required array shape.
        field: Field name included in validation errors.

    Returns:
        An owned read-only ``float`` array with the required shape.

    Raises:
        ValueError: The value cannot form a finite array with ``shape``.
    """
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite array with shape {shape}.") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite array with shape {shape}.")
    array.setflags(write=False)
    return array


def _owned_symmetric_psd_array(
    value: object,
    shape: tuple[int, int],
    field: str,
) -> np.ndarray:
    """Copy, validate, and freeze one covariance array.

    Args:
        value: Array-like covariance input to copy.
        shape: Required square covariance shape.
        field: Field name included in validation errors.

    Returns:
        An owned read-only finite symmetric positive-semidefinite ``float``
        covariance array.

    Raises:
        ValueError: The array has the wrong shape, is nonfinite, asymmetric, or
            materially non-positive-semidefinite.
    """
    covariance = _owned_finite_array(value, shape, field)
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=_COVARIANCE_RTOL,
        atol=_COVARIANCE_ATOL,
    ):
        raise ValueError(f"{field} must be symmetric.")

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.linalg.norm(covariance, ord=np.inf)))
    psd_tolerance = 100.0 * np.finfo(float).eps * scale
    if float(np.min(eigenvalues)) < -psd_tolerance:
        raise ValueError(f"{field} must be positive semidefinite.")

    covariance.setflags(write=False)
    return covariance


class DelayedGnssUpdateStatus(Enum):
    """Classify the terminal outcome of one scheduled delayed GNSS fix.

    ``ACCEPTED`` and ``REJECTED`` denote an update evaluated at a retained
    historical posterior; the latter is a NIS-gated no-op. ``STALE`` denotes a
    mapped epoch before retained history. ``FUTURE_DROPPED`` denotes a mapped
    epoch still later than the filter after its bounded later opportunity.
    Neither unevaluated status has innovation or NIS evidence.
    """

    ACCEPTED = auto()
    REJECTED = auto()
    STALE = auto()
    FUTURE_DROPPED = auto()


@dataclass(frozen=True)
class DelayedGnssUpdate:
    """Describe the terminal timing and gating outcome of one GNSS fix.

    Attributes:
        status: Whether the update was accepted, NIS-rejected, stale, or
            future-dropped.
        t_fix_i_hat: Mapped GNSS measurement epoch in s on the IMU timeline.
        t_applied_i: Snapped retained IMU posterior epoch in s for an evaluated
            accepted or rejected update; ``None`` for stale or future-dropped
            fixes.
        nis: Scalar normalized innovation squared for an evaluated accepted or
            rejected update; ``None`` when no measurement evaluation occurred.

    Notes:
        A NIS-rejected fix is evaluated but leaves the filter state unchanged.
        Stale and future-dropped fixes have no evaluation, applied epoch, or NIS.
    """

    status: DelayedGnssUpdateStatus
    t_fix_i_hat: float
    t_applied_i: float | None
    nis: float | None


@dataclass(frozen=True)
class GnssFusionDiagnostic:
    """Preserve receiver provenance and pre-update GNSS fusion evidence.

    Attributes:
        fix: Original receiver observation. Its ``t_fix_g`` remains the raw
            absolute GNSS epoch and its quality fields remain receiver evidence.
        t_fix_g: Seed-relative GNSS measurement epoch in s provided to the
            clock-aware filter.
        t_arrival_i: Scheduler delivery epoch in s on the local IMU timeline.
        p_gnss_ned: Owned read-only ``(3,)`` receiver position in m in the
            static-origin local NED frame.
        tau_hat: Clock-offset estimate in s used before the terminal outcome.
        update: Terminal mapped-time and gating result for this observation.
        innovation_ned: Owned read-only ``(3,)`` pre-update position innovation
            in m, or ``None`` if the fix was not evaluated.
        innovation_covariance_ned: Owned read-only ``(3, 3)`` pre-update
            innovation covariance in m², or ``None`` if not evaluated.
        prior_position_covariance_ned: Owned read-only ``(3, 3)`` historical
            position block of the pre-update covariance in m², or ``None`` if
            not evaluated.

    Notes:
        Evaluated accepted and NIS-rejected fixes require all three optional
        snapshots. Stale and future-dropped fixes require none, because no
        historical measurement evaluation occurs. ``fix`` is retained as the
        immutable receiver record, while the array fields are independent public
        snapshots rather than filter-owned arrays.
    """

    fix: GnssFixLLH
    t_fix_g: float
    t_arrival_i: float
    p_gnss_ned: np.ndarray
    tau_hat: float
    update: DelayedGnssUpdate
    innovation_ned: np.ndarray | None
    innovation_covariance_ned: np.ndarray | None
    prior_position_covariance_ned: np.ndarray | None

    def __post_init__(self) -> None:
        """Own public snapshots and enforce outcome/evidence consistency."""
        object.__setattr__(
            self,
            "p_gnss_ned",
            _owned_finite_array(self.p_gnss_ned, (3,), "p_gnss_ned"),
        )

        evaluated = self.update.status in {
            DelayedGnssUpdateStatus.ACCEPTED,
            DelayedGnssUpdateStatus.REJECTED,
        }
        snapshots = (
            self.innovation_ned,
            self.innovation_covariance_ned,
            self.prior_position_covariance_ned,
        )
        if evaluated:
            if any(snapshot is None for snapshot in snapshots):
                raise ValueError(
                    "An evaluated GNSS update requires innovation and covariance "
                    "snapshots."
                )
            object.__setattr__(
                self,
                "innovation_ned",
                _owned_finite_array(self.innovation_ned, (3,), "innovation_ned"),
            )
            object.__setattr__(
                self,
                "innovation_covariance_ned",
                _owned_finite_array(
                    self.innovation_covariance_ned,
                    (3, 3),
                    "innovation_covariance_ned",
                ),
            )
            object.__setattr__(
                self,
                "prior_position_covariance_ned",
                _owned_finite_array(
                    self.prior_position_covariance_ned,
                    (3, 3),
                    "prior_position_covariance_ned",
                ),
            )
        elif any(snapshot is not None for snapshot in snapshots):
            raise ValueError(
                "An unevaluated GNSS update cannot include innovation or covariance "
                "snapshots."
            )


@dataclass(frozen=True)
class ClockImuEvent:
    """Save one IMU propagation event for later historical reconstruction.

    Attributes:
        increment: Owned read-only ``(dtheta_b, dv_b, dt)`` IMU increment, with
            body-frame angle and velocity increments and duration in s. It is a
            real measurement for ordinary events and synthetic held-rate motion
            for coast events.
        stance_flag: Causal stance decision paired with the increment. A
            synthetic coast uses ``False`` and therefore cannot trigger a ZUPT.
        additional_process_covariance: Owned read-only ``(16, 16)`` symmetric
            positive-semidefinite covariance added to the usual ``Q`` during
            prediction, or ``None`` for an ordinary real event.

    Notes:
        The event snapshots increment arrays and optional coast covariance so
        replay applies the same saved motion and uncertainty model after a
        delayed accepted GNSS correction. The constructor does not establish
        whether a supplied increment is physically real; its provenance is
        carried by the creating replay path.
    """

    increment: Increment
    stance_flag: bool
    additional_process_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Own immutable copies of the increment and optional covariance."""
        dtheta, dv, dt = self.increment
        dtheta_owned = np.array(dtheta, dtype=float, copy=True)
        dv_owned = np.array(dv, dtype=float, copy=True)
        dtheta_owned.setflags(write=False)
        dv_owned.setflags(write=False)

        object.__setattr__(
            self,
            "increment",
            (dtheta_owned, dv_owned, float(dt)),
        )
        if self.additional_process_covariance is not None:
            object.__setattr__(
                self,
                "additional_process_covariance",
                _owned_symmetric_psd_array(
                    self.additional_process_covariance,
                    (16, 16),
                    "additional_process_covariance",
                ),
            )


@dataclass(frozen=True)
class ClockImuPosterior:
    """Expose an independently owned post-event clock-aware posterior.

    Attributes:
        t_i: Post-event local IMU timeline epoch in s.
        nominal: Owned read-only clock-augmented nominal-state array at ``t_i``.
        P: Owned read-only clock-augmented error-state covariance at ``t_i``.

    Notes:
        A new instance is constructed for each public history access. It is an
        immutable snapshot of the current corrected replay history, not a view
        into a mutable filter-owned historical array.
    """

    t_i: float
    nominal: np.ndarray
    P: np.ndarray

    def __post_init__(self) -> None:
        """Own read-only state and covariance snapshots."""
        nominal = np.array(self.nominal, dtype=float, copy=True)
        covariance = np.array(self.P, dtype=float, copy=True)
        nominal.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "nominal", nominal)
        object.__setattr__(self, "P", covariance)
