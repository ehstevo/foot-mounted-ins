"""Clock-aware GNSS position aiding for the augmented navigation ESEKF.

The GNSS observation remains a receiver solved position in fixed local NED, but
its measurement epoch must be mapped to the IMU-relative propagation timeline:

    t_i = t_g + tau

``t_g`` is seed-relative GNSS measurement time, ``t_i`` is IMU-relative time,
and ``tau`` is their estimated offset (IMU time minus GNSS time), all in
seconds. This filter coordinate is not absolute GPS time and is distinct from
host-arrival timing. The mapped measurement epoch is
``t_fix_i_hat = t_fix_g + tau_hat``.

Linearizing position at the mapped epoch gives

    y ~= dp + v_nom * dtau + noise

So the position Jacobian is no longer constant:

    H_pos = [I 0 0 0 0 v_nom]

The velocity-dependent clock column makes the Jacobian state dependent, so
``_H_POS`` is a function rather than a fixed module-level matrix. As in the
non-clock model, cross-covariances can propagate a GNSS position observation to
other navigation states.
"""
import numpy as np

from bootins.estimation.clock_augmented_esekf import (
    ERR_DP,
    ERR_DTAU,
    NOM_P,
    NOM_V,
    esekf_update_clock,
)
from bootins.frames.geodesy import llh_to_ned


def _pos_h(nominal: np.ndarray) -> np.ndarray:
    """GNSS position measurement model: predict the nominal NED position."""
    return nominal[NOM_P]


def _H_POS(nominal: np.ndarray) -> np.ndarray:
    """Return ``H = [I 0 0 0 0 v_nom]`` for clock-aware position aiding."""
    H = np.zeros((3, 16), dtype=float)
    H[:, ERR_DP] = np.eye(3)
    H[:, ERR_DTAU] = nominal[NOM_V]
    return H


def gnss_time_to_imu_time(t_fix_g: float, tau_hat: float) -> float:
    """Map a seed-relative GNSS fix epoch to IMU-relative time.

    Args:
        t_fix_g: GNSS fix measurement time relative to the selected GNSS seed,
            in seconds.
        tau_hat: Estimated IMU-minus-GNSS clock offset in seconds.

    Returns:
        Mapped IMU-relative fix epoch in seconds:
        ``t_fix_i_hat = t_fix_g + tau_hat``.

    Notes:
        This arithmetic does not convert or estimate absolute GPS time. It only
        expresses a seed-relative GNSS measurement epoch on the IMU timeline.
    """
    return t_fix_g + tau_hat


def gnss_position_innovation_ned(
    nominal: np.ndarray,
    p_gnss_ned: np.ndarray,
) -> np.ndarray:
    """Return the local-NED GNSS position innovation ``y = z - h(x)``.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        p_gnss_ned: Receiver solved ``[North, East, Down]`` position in m.

    Returns:
        GNSS position innovation in NED m with shape ``(3,)``.
    """
    return np.asarray(p_gnss_ned, dtype=float) - _pos_h(nominal)


def gnss_position_innovation_covariance_ned(
    nominal: np.ndarray,
    P: np.ndarray,
    R_pos: np.ndarray,
) -> np.ndarray:
    """Return the innovation covariance for one clock-aware GNSS position fix.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        P: Current ``(16, 16)`` error-state covariance.
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².

    Returns:
        Innovation covariance ``S = H P H.T + R_pos`` in NED m² with shape
        ``(3, 3)``.

    Notes:
        ``H`` includes nominal NED velocity in its clock-offset column. This is
        the first-order displacement induced by a clock-offset perturbation.
    """
    H_pos = _H_POS(nominal)
    return H_pos @ P @ H_pos.T + R_pos


def gnss_position_update_ned(
    nominal: np.ndarray,
    P: np.ndarray,
    p_gnss_ned: np.ndarray,
    R_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one clock-aware GNSS position update in local NED coordinates.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        P: Current ``(16, 16)`` error-state covariance.
        p_gnss_ned: Receiver solved ``[North, East, Down]`` position in m,
            expressed in the same fixed local NED frame as ``nominal``.
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².

    Returns:
        A pair ``(nominal_new, P_new)`` containing the posterior 17-element
        nominal state and ``(16, 16)`` error covariance.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.

    Notes:
        The innovation compares local-NED positions. The Jacobian contributes
        ``v_nom`` to the clock-error column, allowing a position observation to
        correct ``dtau`` and any states correlated with it.
    """
    p_gnss_ned = np.asarray(p_gnss_ned, dtype=float)
    H_pos = _H_POS(nominal)
    return esekf_update_clock(nominal, P, p_gnss_ned, _pos_h, H_pos, R_pos)


def gnss_position_update_llh(
    nominal: np.ndarray,
    P: np.ndarray,
    latitude: float,
    longitude: float,
    height_ellipsoid: float,
    origin_latitude: float,
    origin_longitude: float,
    origin_height_ellipsoid: float,
    R_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a clock-aware GNSS position update from a WGS-84 LLH fix.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        P: Current ``(16, 16)`` error-state covariance.
        latitude: GNSS geodetic latitude (rad).
        longitude: GNSS longitude (rad).
        height_ellipsoid: GNSS ellipsoidal height (m).
        origin_latitude: Fixed local-NED-origin geodetic latitude (rad).
        origin_longitude: Fixed local-NED-origin longitude (rad).
        origin_height_ellipsoid: Fixed local-NED-origin ellipsoidal height
            (m).
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².

    Returns:
        A pair ``(nominal_new, P_new)`` from the local-NED update.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.
    """
    p_gnss_ned = llh_to_ned(
        latitude,
        longitude,
        height_ellipsoid,
        origin_latitude,
        origin_longitude,
        origin_height_ellipsoid,
    )
    return gnss_position_update_ned(nominal, P, p_gnss_ned, R_pos)


def gnss_position_nis_ned(
    nominal: np.ndarray,
    P: np.ndarray,
    p_gnss_ned: np.ndarray,
    R_pos: np.ndarray,
) -> float:
    """Compute normalized innovation squared for a clock-aware GNSS position.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        P: Current ``(16, 16)`` error-state covariance.
        p_gnss_ned: Receiver solved ``[North, East, Down]`` position in m.
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².

    Returns:
        Dimensionless scalar ``y.T @ inv(S) @ y``.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.

    Notes:
        The NIS includes position, covariance, receiver uncertainty, and the
        current velocity-dependent clock Jacobian. It is a consistency statistic,
        not an independent assessment of receiver truth.
    """
    y = gnss_position_innovation_ned(nominal, p_gnss_ned)
    S = gnss_position_innovation_covariance_ned(nominal, P, R_pos)
    nis = float(y.T @ np.linalg.inv(S) @ y)

    return nis


def gnss_position_update_ned_gated(
    nominal: np.ndarray,
    P: np.ndarray,
    p_gnss_ned: np.ndarray,
    R_pos: np.ndarray,
    nis_threshold: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fuse a local-NED GNSS position only when its NIS passes the gate.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        P: Current ``(16, 16)`` error-state covariance.
        p_gnss_ned: Receiver solved ``[North, East, Down]`` position in m.
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².
        nis_threshold: Acceptance limit; equality is accepted.

    Returns:
        ``(nominal_out, P_out, nis, accepted)``. Rejected updates return the
        original ``nominal`` and ``P`` objects, with ``accepted`` set to
        ``False``.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.

    Notes:
        Rejection leaves the filter state unchanged, preventing an inconsistent
        GNSS observation from moving the nominal state or shrinking covariance.
    """
    nis = gnss_position_nis_ned(nominal, P, p_gnss_ned, R_pos)

    if nis > nis_threshold:
        return nominal, P, nis, False
    nominal_new, P_new = gnss_position_update_ned(nominal, P, p_gnss_ned, R_pos)
    return nominal_new, P_new, nis, True


def gnss_position_update_llh_gated(
    nominal: np.ndarray,
    P: np.ndarray,
    latitude: float,
    longitude: float,
    height_ellipsoid: float,
    origin_latitude: float,
    origin_longitude: float,
    origin_height_ellipsoid: float,
    R_pos: np.ndarray,
    nis_threshold: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Apply a gated clock-aware GNSS position update from a WGS-84 LLH fix.

    Args:
        nominal: Current 17-element nominal navigation-and-clock state.
        P: Current ``(16, 16)`` error-state covariance.
        latitude: GNSS geodetic latitude (rad).
        longitude: GNSS longitude (rad).
        height_ellipsoid: GNSS ellipsoidal height (m).
        origin_latitude: Fixed local-NED-origin geodetic latitude (rad).
        origin_longitude: Fixed local-NED-origin longitude (rad).
        origin_height_ellipsoid: Fixed local-NED-origin ellipsoidal height
            (m).
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².
        nis_threshold: Acceptance limit for the NIS gate; equality is accepted.

    Returns:
        ``(nominal_out, P_out, nis, accepted)`` from the gated local-NED update.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.
    """
    p_gnss_ned = llh_to_ned(
        latitude,
        longitude,
        height_ellipsoid,
        origin_latitude,
        origin_longitude,
        origin_height_ellipsoid,
    )
    return gnss_position_update_ned_gated(
        nominal, P, p_gnss_ned, R_pos, nis_threshold
    )
