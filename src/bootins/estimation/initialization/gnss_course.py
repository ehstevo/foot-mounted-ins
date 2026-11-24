"""GNSS course estimation, heading alignment, and static-origin estimation.

Receiver-valid GNSS positions provide horizontal course-over-ground evidence.
The selected course aligns a terminal static body-to-local-FRD attitude with
geographic NED, subject to the assumption that course over ground represents
the body's forward direction during the selected walk. Static receiver fixes
are separately averaged in ECEF to establish a geographic origin.
"""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from bootins.estimation.initialization.static import (
    StaticZuptInitialization,
    _FORWARD_REFERENCE_B,
    initialize_static_zupt,
)
from bootins.estimation.mechanization import Increment
from bootins.frames.geodesy import ecef_to_llh, llh_to_ecef, llh_to_ned
from bootins.frames.quaternion import dcm_to_quat, quat_to_dcm
from bootins.sensors.gnss_observations import GnssFixLLH

@dataclass(frozen=True)
class GnssCourseEstimate:
    """One qualified horizontal geographic course from GNSS track evidence.

    Attributes:
        direction_ne: Signed unit course direction ``[North, East]`` with shape
            ``(2,)``.
        course: Course clockwise from geographic North in radians, normalized
            to ``[0, 2π)``.
        baseline: First-to-last displacement projected along ``direction_ne``
            in m.
        cross_track_rms: RMS perpendicular distance from the fitted line in m.
        fix_count: Number of receiver-valid fixes in the qualified candidate.

    Notes:
        This frozen dataclass does not make its NumPy array field immutable.
        The result describes observed course over ground, not an independent
        direct measurement of body heading.
    """

    direction_ne: np.ndarray
    course: float
    baseline: float
    cross_track_rms: float
    fix_count: int


@dataclass(frozen=True)
class GnssCourseInitialization:
    """Terminal static facts aligned to a qualified geographic GNSS course.

    ``terminal_static.static_level.q_fb`` maps body axes into an arbitrary-yaw
    local Forward-Right-Down frame. ``q_nb`` maps the same body axes into
    geographic North-East-Down after horizontal alignment with ``course``.

    Attributes:
        terminal_static: Terminal static leveling and gyro-bias estimate.
        course: Qualified GNSS course-over-ground evidence.
        q_nb: Scalar-first body-to-geographic-NED quaternion ``[w, x, y, z]``
            with shape ``(4,)``.

    Notes:
        The course alignment assumes the walking course represents the selected
        body-forward direction. It cannot independently distinguish body yaw
        from sideways or backward walking.
    """

    terminal_static: StaticZuptInitialization
    course: GnssCourseEstimate
    q_nb: np.ndarray


def _fit_course_direction_ne(points_ne: np.ndarray) -> np.ndarray | None:
    """Fit a signed principal horizontal direction to GNSS track points.

    Args:
        points_ne: ``(N, 2)`` horizontal positions ``[North, East]`` in m.

    Returns:
        Signed unit fitted direction ``[North, East]`` with shape ``(2,)``, or
        ``None`` when fewer than two points or zero net displacement leaves its
        sign undefined.

    Notes:
        Principal-component fitting treats North and East coordinate variation
        symmetrically, unlike a regression that designates one axis as
        error-free. PCA determines an unoriented line; temporal first-to-last
        displacement selects its sign. Qualification of baseline and straightness
        is performed separately.
    """
    if points_ne.shape[0] < 2:
        return None

    center_ne = np.mean(points_ne, axis=0)
    centered_positions_ne = points_ne - center_ne
    scatter_matrix = centered_positions_ne.T @ centered_positions_ne
    _eigenvalues, eigenvectors = np.linalg.eigh(scatter_matrix)
    direction_ne = eigenvectors[:, -1]
    net_displacement = points_ne[-1] - points_ne[0]

    if np.isclose(np.linalg.norm(net_displacement), 0.0):
        return None

    # PCA identifies an unoriented axis; time order selects its forward sign.
    if np.dot(direction_ne, net_displacement) < 0:
        direction_ne = -direction_ne

    return direction_ne


def _course_fit_metrics(
    points_ne: np.ndarray,
    direction_ne: np.ndarray,
) -> tuple[float, float]:
    """Measure along-course baseline and cross-track RMS for a fitted direction.

    Args:
        points_ne: ``(N, 2)`` horizontal positions ``[North, East]`` in m.
        direction_ne: Signed unit fitted course direction with shape ``(2,)``.

    Returns:
        ``(baseline, cross_track_rms)`` in m. ``baseline`` is first-to-last
        displacement projected onto the fitted direction.

    Notes:
        Projecting displacement onto the fitted direction prevents lateral GNSS
        noise from inflating the baseline. Cross-track RMS is the root mean
        square perpendicular distance of the points from the fitted line.
    """
    center_ne = np.mean(points_ne, axis=0)
    centered_positions_ne = points_ne - center_ne
    net_displacement = points_ne[-1] - points_ne[0]

    # Use only progress along the fitted direction as the travel baseline.
    baseline = np.dot(direction_ne, net_displacement)

    # Rotating [North, East] by 90 degrees gives its horizontal line normal.
    normal_ne = np.array([-direction_ne[1], direction_ne[0]])
    cross_track_distances = centered_positions_ne @ normal_ne
    cross_track_rms = np.sqrt(np.mean(cross_track_distances**2))

    return float(baseline), float(cross_track_rms)


class GnssCourseSelector:
    """Causally select one qualified straight-walk GNSS course.

    The first valid fix establishes a temporary LLH origin. Subsequent valid
    fixes convert to horizontal North/East points relative to that origin. The
    origin exists only for course geometry; it does not define the navigation
    frame origin. Receiver-invalid fixes reset the candidate so a course is
    never assembled from disjoint evidence.

    A candidate remains open while it lacks enough valid fixes or along-course
    baseline. Once it is long enough, excessive cross-track RMS rejects it and
    starts a fresh candidate at the current valid fix. The first qualified
    estimate is retained permanently and returned on subsequent calls.

    Attributes:
        minimum_fix_count: Minimum number of valid fixes needed for evaluation.
        minimum_baseline: Minimum projected horizontal baseline in m.
        maximum_cross_track_rms: Maximum fitted-line cross-track RMS in m.

    Notes:
        Input configuration is validated at the aided-profile boundary. This
        selector owns only causal candidate accumulation and qualification.
    """

    def __init__(
        self,
        minimum_fix_count: int,
        minimum_baseline: float,
        maximum_cross_track_rms: float,
    ) -> None:
        """Create a selector with caller-supplied course criteria.

        Args:
            minimum_fix_count: Minimum receiver-valid fixes needed for course
                evaluation.
            minimum_baseline: Minimum projected course baseline in m.
            maximum_cross_track_rms: Maximum line-fit cross-track RMS in m.

        Notes:
            Profile loading validates these values before selector construction.
        """
        self.minimum_fix_count = minimum_fix_count
        self.minimum_baseline = minimum_baseline
        self.maximum_cross_track_rms = maximum_cross_track_rms
        self._selected_estimate: GnssCourseEstimate | None = None
        self._fix_count = 0

        # This temporary origin makes only the candidate course track Cartesian.
        # Terminal-static fixes provide the separate navigation origin evidence.
        self._origin_fix: GnssFixLLH | None = None
        self._positions_ne: list[np.ndarray] = []

    def update(self, fix: GnssFixLLH) -> GnssCourseEstimate | None:
        """Consume one GNSS fix and return a course after qualification.

        Args:
            fix: GNSS observation carrying WGS-84 LLH and receiver validity.

        Returns:
            The selected ``GnssCourseEstimate`` once qualified, otherwise
            ``None``. Repeated calls after selection return the same result.

        Notes:
            An invalid fix clears an unselected candidate instead of joining
            track segments across unavailable receiver evidence. A valid fix
            extends the candidate until it qualifies, or restarts it when a
            sufficiently long candidate exceeds the cross-track limit.
        """
        if self._selected_estimate is not None:
            return self._selected_estimate

        if not fix.fix_valid:
            self._origin_fix = None
            self._positions_ne = []
            self._fix_count = 0
            return None

        self._fix_count += 1

        if self._origin_fix is None:
            self._origin_fix = fix
            self._positions_ne.append(np.zeros(2, dtype=float))
            return None

        fix_ned = llh_to_ned(
            fix.latitude,
            fix.longitude,
            fix.height_ellipsoid,
            self._origin_fix.latitude,
            self._origin_fix.longitude,
            self._origin_fix.height_ellipsoid,
        )

        # Heading is a horizontal property; Down is intentionally excluded.
        fix_ne = np.array([fix_ned[0], fix_ned[1]])
        self._positions_ne.append(fix_ne)

        if self._fix_count < self.minimum_fix_count:
            return None

        positions_ne = np.asarray(self._positions_ne, dtype=float)

        fitted_direction_ne = _fit_course_direction_ne(positions_ne)

        if fitted_direction_ne is None:
            return None

        direction_ne = np.array(fitted_direction_ne, dtype=float, copy=True)

        baseline, cross_track_rms = _course_fit_metrics(positions_ne, direction_ne)

        if baseline < self.minimum_baseline:
            return None

        if cross_track_rms > self.maximum_cross_track_rms:
            self._positions_ne.clear()
            self._positions_ne.append(np.zeros(2, dtype=float))
            self._origin_fix = fix
            self._fix_count = 1
            return None

        course = float(np.atan2(direction_ne[1], direction_ne[0]))
        if course < 0:
            course += 2 * np.pi

        course_estimate = GnssCourseEstimate(
            direction_ne=direction_ne,
            course=course,
            baseline=baseline,
            cross_track_rms=cross_track_rms,
            fix_count=self._fix_count,
        )
        self._selected_estimate = course_estimate
        return self._selected_estimate


def initialize_static_gnss(
    increments: Iterable[Increment],
    course: GnssCourseEstimate,
) -> GnssCourseInitialization:
    """Align terminal static leveling with a qualified GNSS course.

    Args:
        increments: Terminal stationary body-frame ``(dtheta, dv, dt)`` IMU
            increments in rad, m/s, and s.
        course: Qualified horizontal GNSS course-over-ground evidence.

    Returns:
        Terminal static facts, course evidence, and a body-to-geographic-NED
        quaternion.

    Raises:
        ValueError: If static initialization rejects the terminal interval.

    Notes:
        The terminal stance supplies gravity leveling and gyro bias. Its local
        FRD attitude is rotated into geographic NED by treating the course's
        signed North/East direction as body forward. This function neither
        establishes an LLH origin nor constructs position/velocity state or
        fuses GNSS observations.
    """
    terminal_static = initialize_static_zupt(increments)

    # Static initialization maps body axes into a local FRD frame.
    C_fb = quat_to_dcm(terminal_static.static_level.q_fb)

    u_N = course.direction_ne[0]
    u_E = course.direction_ne[1]

    # Columns are local Forward, Right, and Down axes expressed in NED.
    C_nf = np.array(
        [
            [u_N, -u_E, 0.0],
            [u_E, u_N, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Compose body -> local FRD -> geographic NED.
    C_nb = C_nf @ C_fb
    q_nb = dcm_to_quat(C_nb)

    return GnssCourseInitialization(
        terminal_static=terminal_static,
        course=course,
        q_nb=q_nb,
    )


@dataclass(frozen=True)
class GnssStaticOrigin:
    """ECEF-averaged WGS-84 origin with its static receiver evidence.

    Attributes:
        latitude: Estimated geodetic latitude in radians.
        longitude: Estimated longitude in radians.
        height_ellipsoid: Estimated ellipsoidal height in m.
        contributing_fixes: Receiver-valid static GNSS fixes used in the ECEF
            average, retained with their original evidence and provenance.

    Notes:
        This estimate does not invent a GNSS epoch, host arrival time, receiver
        quality record, or native-archive identity of its own.
    """

    latitude: float
    longitude: float
    height_ellipsoid: float
    contributing_fixes: tuple[GnssFixLLH, ...]


def estimate_static_gnss_origin(fixes: Iterable[GnssFixLLH]) -> GnssStaticOrigin:
    """Estimate a static WGS-84 origin by averaging receiver positions in ECEF.

    Args:
        fixes: Iterable of receiver-valid GNSS observations with WGS-84 LLH.

    Returns:
        ECEF-averaged geographic origin and the materialized contributing fixes.

    Raises:
        ValueError: If no fix is supplied or any supplied fix is receiver
            invalid.

    Notes:
        The iterable is materialized because it may be one-pass. Latitude and
        longitude are not Cartesian coordinates, so each LLH fix is converted
        to ECEF before averaging and the mean is converted back to LLH. Averaging
        reduces uncorrelated static receiver noise but does not remove shared
        multipath or other systematic bias.
    """
    contributing_fixes = tuple(fixes)
    if not contributing_fixes:
        raise ValueError("Static GNSS origin requires at least one fix.")
    if any(not fix.fix_valid for fix in contributing_fixes):
        raise ValueError("Static GNSS origin requires only receiver-valid fixes.")

    ecef_positions = np.asarray(
        [
            llh_to_ecef(fix.latitude, fix.longitude, fix.height_ellipsoid)
            for fix in contributing_fixes
        ],
        dtype=float,
    )
    mean_llh = ecef_to_llh(np.mean(ecef_positions, axis=0))

    return GnssStaticOrigin(
        latitude=float(mean_llh[0]),
        longitude=float(mean_llh[1]),
        height_ellipsoid=float(mean_llh[2]),
        contributing_fixes=contributing_fixes,
    )
