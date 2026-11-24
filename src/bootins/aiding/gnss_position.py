"""Loosely coupled GNSS position aiding in a fixed local NED frame.

The receiver supplies a solved geographic position rather than raw
pseudoranges. A caller converts that observation to the fixed local
North-East-Down frame used by the navigation state, then this module applies a
position measurement to the 15-state navigation ESEKF.

The measurement model assumes:

  * Loosely coupled solved GNSS positions, not raw pseudoranges.
  * Position-only aiding; GNSS velocity and Doppler are not modeled here.
  * A measurement epoch already consistent with the current IMU state. The
    clock-augmented counterpart handles estimated GNSS/IMU timing offset.
  * No rigid antenna-to-foot lever arm. A non-rigid antenna/body-point mismatch
    is unmodeled and must be represented as measurement uncertainty in ``R_pos``.

Under those assumptions the cleanest measurement model is:

    z_gnss = p_true^n + noise

with the nominal/error split

    p_true^n = p_nominal^n + delta p

so the error-state update sees

    y = z - h(nominal)
      = z - p_nominal^n
      ~= delta p + noise

The position measurement directly observes only ``delta p`` through the
Jacobian

    H_pos = [ I  0  0  0  0 ]

Velocity, attitude, and bias corrections are indirect: they arise through the
cross-covariances already present in ``P``. A GNSS position measurement therefore
does not correct only position in the posterior state.

The normalized innovation squared (NIS) tests consistency before fusion:

    y   = z - h(nominal)
    S   = H P H^T + R
    NIS = y^T S^-1 y

The gate rejects a fix when this statistic exceeds the caller-chosen threshold.
"""
import numpy as np

from bootins.estimation.esekf import ERR_DP, NOM_P, esekf_update
from bootins.frames.geodesy import llh_to_ned


def _pos_h(nominal: np.ndarray) -> np.ndarray:
    """GNSS position measurement model: predict the nominal NED position."""
    return nominal[NOM_P]


_H_POS = np.zeros((3, 15), dtype=float)
_H_POS[:, ERR_DP] = np.eye(3)


def gnss_position_update_ned(
    nominal: np.ndarray,
    P: np.ndarray,
    p_gnss_ned: np.ndarray,
    R_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one loose-coupled GNSS position update in local NED coordinates.

    Args:
        nominal: Current 16-element nominal ESEKF state.
        P: Current ``(15, 15)`` error-state covariance.
        p_gnss_ned: Receiver solved position ``[North, East, Down]`` in m,
            expressed in the same fixed local NED frame as ``nominal``.
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².

    Returns:
        A pair ``(nominal_new, P_new)`` containing the posterior nominal state
        and ``(15, 15)`` error covariance.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.

    Notes:
        The measurement model is ``h(nominal) = p_nominal^n`` with
        ``H_pos = [I 0 0 0 0]``. ``esekf_update`` injects any indirect
        velocity, attitude, or bias corrections supported by ``P``.
    """
    p_gnss_ned = np.asarray(p_gnss_ned, dtype=float)
    return esekf_update(nominal, P, p_gnss_ned, _pos_h, _H_POS, R_pos)


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
    """Apply a loose-coupled GNSS position update from a WGS-84 LLH fix.

    Args:
        nominal: Current 16-element nominal ESEKF state.
        P: Current ``(15, 15)`` error-state covariance.
        latitude: GNSS geodetic latitude (rad).
        longitude: GNSS longitude (rad).
        height_ellipsoid: GNSS ellipsoidal height (m).
        origin_latitude: Fixed local-NED-origin geodetic latitude (rad).
        origin_longitude: Fixed local-NED-origin longitude (rad).
        origin_height_ellipsoid: Fixed local-NED-origin ellipsoidal height
            (m).
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².

    Returns:
        A pair ``(nominal_new, P_new)`` from the local-NED position update.

    Raises:
        numpy.linalg.LinAlgError: If the innovation covariance is singular and
            cannot be inverted.

    Notes:
        The LLH observation is converted to the INS's fixed local NED frame
        before fusion. The geographic origin remains explicit and is not
        inferred from the nominal navigation state.
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
    """Compute normalized innovation squared for a local-NED GNSS position.

    Args:
        nominal: Current 16-element nominal ESEKF state.
        P: Current ``(15, 15)`` error-state covariance.
        p_gnss_ned: Receiver solved ``[North, East, Down]`` position in m.
        R_pos: ``(3, 3)`` GNSS position-noise covariance in NED m².

    Returns:
        Dimensionless scalar ``y.T @ inv(S) @ y``.

    Raises:
        numpy.linalg.LinAlgError: If ``S = H P H.T + R_pos`` is singular and
            cannot be inverted.

    Notes:
        Small NIS indicates that the residual is plausible under the declared
        innovation covariance. Large NIS indicates inconsistency with that
        uncertainty model; it does not by itself identify the source of error.
    """
    y = np.asarray(p_gnss_ned, dtype=float) - _pos_h(nominal)
    S = _H_POS @ P @ _H_POS.T + R_pos
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
        nominal: Current 16-element nominal ESEKF state.
        P: Current ``(15, 15)`` error-state covariance.
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
        Rejection is intentionally a filter no-op: an inconsistent observation
        must neither move the nominal state nor shrink its covariance.
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
    """Apply a gated GNSS position update from a WGS-84 LLH fix.

    Args:
        nominal: Current 16-element nominal ESEKF state.
        P: Current ``(15, 15)`` error-state covariance.
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
