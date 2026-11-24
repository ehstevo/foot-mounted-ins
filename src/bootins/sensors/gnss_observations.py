"""Define normalized receiver GNSS position evidence for replay.

This module is the receiver-independent boundary between native GNSS archives
or standardized tables and the later causal GNSS/INS scheduler. Each record
retains WGS-84 LLH, receiver GPS time, host-arrival provenance, and reported
receiver-quality facts. It does not parse native bytes, select a local origin,
choose filter covariance, map GNSS time to the IMU timeline, or accept a
navigation update.

Receiver validity and reported accuracy describe receiver evidence only. They
do not establish geodetic accuracy, measurement-time alignment, or a later
filter's NIS-gating outcome.
"""
from dataclasses import dataclass
from math import isfinite, pi
from numbers import Integral, Real


_GPS_WEEK_DURATION = 604800.0


def _require_nonnegative_integer(value: object, field: str) -> None:
    """Reject non-integral, Boolean, or negative record-index fields."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer.")


def _require_finite_real(value: object, field: str) -> float:
    """Return one finite non-Boolean real value or raise a field-specific error."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


@dataclass(frozen=True)
class GnssFixLLH:
    """Represent one normalized receiver position observation in WGS-84 LLH.

    Attributes:
        gps_week: Nonnegative receiver GPS week for the measurement epoch.
        gps_tow: Receiver time of week in s in ``[0, 604800)``.
        host_arrival_monotonic: Nonnegative host-monotonic availability time in
            s. It is not GNSS measurement time or an IMU propagation epoch.
        latitude: Geodetic WGS-84 latitude in rad within ``[-π/2, π/2]``.
        longitude: Geodetic WGS-84 longitude in rad within ``[-π, π]``.
        height_ellipsoid: WGS-84 ellipsoidal height in m.
        fix_valid: Receiver-reported Boolean validity evidence.
        satellite_count: Nonnegative receiver-reported satellite count.
        horizontal_accuracy: Nonnegative receiver-reported horizontal accuracy
            in m.
        vertical_accuracy: Nonnegative receiver-reported vertical accuracy in m.
        native_record_index: Nonnegative index linking this record to its native
            archive provenance.

    Notes:
        This frozen scalar record cannot be reassigned after construction.
        Construction validates field representation and geographic bounds, but
        not receiver accuracy or stream ordering. A table or projection layer
        establishes host-arrival order. ``fix_valid`` and accuracy fields remain
        receiver facts, not filter covariance or acceptance decisions.
    """

    gps_week: int
    gps_tow: float
    host_arrival_monotonic: float
    latitude: float
    longitude: float
    height_ellipsoid: float
    fix_valid: bool
    satellite_count: int
    horizontal_accuracy: float
    vertical_accuracy: float
    native_record_index: int

    def __post_init__(self) -> None:
        """Validate structural clock, coordinate, and receiver-evidence fields."""
        _require_nonnegative_integer(self.gps_week, "gps_week")
        gps_tow = _require_finite_real(self.gps_tow, "gps_tow")
        if not 0.0 <= gps_tow < _GPS_WEEK_DURATION:
            raise ValueError(
                "gps_tow must lie in [0, one GPS-week duration)."
            )

        host_arrival = _require_finite_real(
            self.host_arrival_monotonic,
            "host_arrival_monotonic",
        )
        if host_arrival < 0.0:
            raise ValueError("host_arrival_monotonic must be nonnegative.")

        latitude = _require_finite_real(self.latitude, "latitude")
        if not -pi / 2.0 <= latitude <= pi / 2.0:
            raise ValueError("latitude must lie in [-pi/2, pi/2].")

        longitude = _require_finite_real(self.longitude, "longitude")
        if not -pi <= longitude <= pi:
            raise ValueError("longitude must lie in [-pi, pi].")

        _require_finite_real(self.height_ellipsoid, "height_ellipsoid")
        if not isinstance(self.fix_valid, bool):
            raise ValueError("fix_valid must be Boolean.")
        _require_nonnegative_integer(self.satellite_count, "satellite_count")

        horizontal_accuracy = _require_finite_real(
            self.horizontal_accuracy,
            "horizontal_accuracy",
        )
        if horizontal_accuracy < 0.0:
            raise ValueError("horizontal_accuracy must be nonnegative.")

        vertical_accuracy = _require_finite_real(
            self.vertical_accuracy,
            "vertical_accuracy",
        )
        if vertical_accuracy < 0.0:
            raise ValueError("vertical_accuracy must be nonnegative.")

        _require_nonnegative_integer(self.native_record_index, "native_record_index")

    @property
    def t_fix_g(self) -> float:
        """Return the continuous GPS measurement epoch in s.

        The epoch is ``gps_week * 604800 + gps_tow``. It is an absolute
        receiver-time quantity; a replay may later form a seed-relative GNSS
        time without altering this evidence record.
        """
        return self.gps_week * _GPS_WEEK_DURATION + self.gps_tow
