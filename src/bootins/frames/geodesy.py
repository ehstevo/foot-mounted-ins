"""WGS-84 geodesy and fixed-origin local-NED coordinate conversions.

GNSS observations are geodetic WGS-84 positions

    (lat, lon, h)

whereas the navigation state uses a fixed local North-East-Down (NED) frame. The
conversion chain is

    (lat, lon, h)  <->  p^e (ECEF, meters)  <->  p^n (local NED, meters)

Why the ECEF middle step exists:
  * `lat` / `lon` are ANGLES, not Cartesian meters
  * meters-per-radian vary with latitude
  * subtracting two geodetic tuples is NOT a Euclidean displacement

ECEF fixes that: convert each geodetic point to a 3-D Cartesian position in
meters, subtract the chosen origin there, then rotate that relative vector into
the local North-East-Down axes.

The local NED basis is fixed at its chosen origin. Accordingly,
``ecef_to_ned_matrix`` depends on the origin latitude and longitude, not the
current GNSS point or origin height. Height changes a point's location along
the local normal but not the tangent-plane orientation.

We use the precise name "LLH" in the function names because the third geodetic
coordinate in these formulas is ellipsoidal height `h`. Common "LLA" aliases are
also provided because that shorthand appears throughout GNSS/INS literature.
"""
import numpy as np

# WGS-84 reference ellipsoid parameters.
WGS84_A = 6378137.0                    # semi-major axis (equatorial radius), m
WGS84_F = 1.0 / 298.257223563          # flattening, dimensionless
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)   # first eccentricity squared, dimensionless
WGS84_B = WGS84_A * (1.0 - WGS84_F)    # semi-minor axis (polar radius), m
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2) # second eccentricity squared, dimensionless


def _prime_vertical_radius(latitude: float) -> float:
    """Return the WGS-84 prime-vertical radius `N(phi)` at latitude `phi`.

    ``N(phi)`` is the latitude-dependent scale factor used to transform
    geodetic WGS-84 coordinates into ECEF Cartesian coordinates.

    Args:
        latitude: Geodetic latitude (rad).
    """
    sin_lat = np.sin(latitude)
    return float(WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2))


def llh_to_ecef(
    latitude: float,
    longitude: float,
    height_ellipsoid: float,
) -> np.ndarray:
    """Convert a WGS-84 geodetic position to ECEF Cartesian coordinates.

    Args:
        latitude: Geodetic latitude (rad).
        longitude: Longitude (rad).
        height_ellipsoid: Height above the WGS-84 reference ellipsoid (m).

    Returns:
        ECEF position ``[x, y, z]`` in m with shape ``(3,)``.

    Notes:
        Geodetic latitude is measured relative to the ellipsoid normal, not the
        Earth-center radial direction. The ``z`` expression consequently uses
        ``N(phi) * (1 - e²)`` rather than the ``x`` and ``y`` scale factor.
    """
    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)
    sin_lon = np.sin(longitude)
    cos_lon = np.cos(longitude)

    N_phi = _prime_vertical_radius(latitude)

    x = (N_phi + height_ellipsoid) * cos_lat * cos_lon
    y = (N_phi + height_ellipsoid) * cos_lat * sin_lon
    z = (N_phi * (1.0 - WGS84_E2) + height_ellipsoid) * sin_lat

    return np.array([x, y, z], dtype=float)


def ecef_to_ned_matrix(
    origin_latitude: float,
    origin_longitude: float,
) -> np.ndarray:
    """Return the DCM from ECEF vectors to a fixed local NED frame.

    Args:
        origin_latitude: Geodetic latitude of the local-frame origin (rad).
        origin_longitude: Longitude of the local-frame origin (rad).

    Returns:
        ``(3, 3)`` DCM ``C^n_e`` mapping ECEF vectors to local NED vectors.

    Notes:
        The rows are the local unit basis vectors resolved in ECEF coordinates:

        row 0 = north_hat^e
        row 1 = east_hat^e
        row 2 = down_hat^e

    so, by the repo's frame convention,

        v^n = C^n_e @ v^e

        Height does not appear because it changes origin position, not the
        tangent-plane orientation at a given latitude and longitude.
    """
    sin_origin_latitude = np.sin(origin_latitude)
    cos_origin_latitude = np.cos(origin_latitude)
    sin_origin_longitude = np.sin(origin_longitude)
    cos_origin_longitude = np.cos(origin_longitude)

    north_hat_e = np.array(
        [
            -sin_origin_latitude * cos_origin_longitude,
            -sin_origin_latitude * sin_origin_longitude,
            cos_origin_latitude,
        ],
        dtype=float,
    )
    east_hat_e = np.array(
        [-sin_origin_longitude, cos_origin_longitude, 0.0],
        dtype=float,
    )
    down_hat_e = np.array(
        [
            -cos_origin_latitude * cos_origin_longitude,
            -cos_origin_latitude * sin_origin_longitude,
            -sin_origin_latitude,
        ],
        dtype=float,
    )

    return np.vstack((north_hat_e, east_hat_e, down_hat_e))


def ecef_to_ned(
    p_ecef: np.ndarray,
    p0_ecef: np.ndarray,
    origin_latitude: float,
    origin_longitude: float,
) -> np.ndarray:
    """Express an ECEF position relative to an origin in local NED coordinates.

    Args:
        p_ecef: ECEF point position in m with shape ``(3,)``.
        p0_ecef: ECEF position of the local NED origin in m with shape ``(3,)``.
        origin_latitude: Geodetic latitude of the origin (rad).
        origin_longitude: Longitude of the origin (rad).

    Returns:
        Local NED displacement from the origin in m with shape ``(3,)``.

    Notes:
        The operation must first subtract in Cartesian ECEF coordinates and
        then rotate the resulting displacement:

        1. subtract the origin in ECEF  -> a true Cartesian displacement
        2. rotate that displacement     -> local North / East / Down components

        Rotating absolute ECEF positions would be incorrect because NED is a
        local displacement frame rather than an Earth-centered absolute frame.
    """
    p_ecef = np.asarray(p_ecef, dtype=float)
    p0_ecef = np.asarray(p0_ecef, dtype=float)
    C_ne = ecef_to_ned_matrix(origin_latitude, origin_longitude)
    return C_ne @ (p_ecef - p0_ecef)


def llh_to_ned(
    latitude: float,
    longitude: float,
    height_ellipsoid: float,
    origin_latitude: float,
    origin_longitude: float,
    origin_height_ellipsoid: float,
) -> np.ndarray:
    """Convert a WGS-84 geodetic point to fixed-origin local NED coordinates.

    Args:
        latitude: Point geodetic latitude (rad).
        longitude: Point longitude (rad).
        height_ellipsoid: Point ellipsoidal height (m).
        origin_latitude: Local-frame origin geodetic latitude (rad).
        origin_longitude: Local-frame origin longitude (rad).
        origin_height_ellipsoid: Local-frame origin ellipsoidal height (m).

    Returns:
        Local NED displacement from the origin in m with shape ``(3,)``.
    """
    p_ecef = llh_to_ecef(latitude, longitude, height_ellipsoid)
    p0_ecef = llh_to_ecef(
        origin_latitude,
        origin_longitude,
        origin_height_ellipsoid,
    )
    return ecef_to_ned(p_ecef, p0_ecef, origin_latitude, origin_longitude)


def ned_to_ecef(
    p_ned: np.ndarray,
    origin_latitude: float,
    origin_longitude: float,
    origin_height_ellipsoid: float,
) -> np.ndarray:
    """Convert a fixed-origin local NED displacement to absolute ECEF.

    Args:
        p_ned: ``[North, East, Down]`` displacement in m with shape ``(3,)``.
        origin_latitude: Geodetic latitude of the fixed NED origin (rad).
        origin_longitude: Longitude of the fixed NED origin (rad).
        origin_height_ellipsoid: Ellipsoidal height of the fixed NED origin
            (m).

    Returns:
        ECEF position in m with shape ``(3,)``.

    Notes:
        The inverse DCM is the transpose of the origin-fixed ECEF-to-NED DCM
        because that DCM is orthonormal.
    """
    p0_ecef = llh_to_ecef(
        origin_latitude,
        origin_longitude,
        origin_height_ellipsoid,
    )
    C_ne = ecef_to_ned_matrix(origin_latitude, origin_longitude)
    return p0_ecef + C_ne.T @ np.asarray(p_ned, dtype=float)


def ecef_to_llh(p_ecef: np.ndarray) -> np.ndarray:
    """Convert an ECEF Cartesian position to WGS-84 geodetic LLH.

    Args:
        p_ecef: ECEF ``[x, y, z]`` position in m with shape ``(3,)``.

    Returns:
        ``[latitude, longitude, ellipsoidal_height]`` in radians, radians, and
        m, respectively, with shape ``(3,)``.

    Raises:
        ValueError: If the ECEF position is the Earth-centered origin, which
            does not define a unique geodetic position.

    Notes:
        The nonpolar path uses Bowring's auxiliary-angle form. At either pole,
        longitude is conventionally zero and a dedicated height expression
        avoids division by a vanishing horizontal radius or cosine of latitude.
    """
    x, y, z = np.asarray(p_ecef, dtype=float)
    horizontal_radius = float(np.hypot(x, y))

    if horizontal_radius == 0.0:
        if z == 0.0:
            raise ValueError("ECEF origin does not define a unique LLH position.")
        return np.array(
            [
                np.copysign(np.pi / 2.0, z),
                0.0,
                abs(z) - WGS84_B,
            ],
            dtype=float,
        )

    lon = float(np.arctan2(y, x))
    aux_angle = float(np.arctan2(z * WGS84_A, horizontal_radius * WGS84_B))
    sin_aux = np.sin(aux_angle)
    cos_aux = np.cos(aux_angle)
    lat = float(
        np.arctan2(
            z + WGS84_EP2 * WGS84_B * sin_aux**3,
            horizontal_radius - WGS84_E2 * WGS84_A * cos_aux**3,
        )
    )

    N_phi = _prime_vertical_radius(lat)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    if abs(cos_lat) >= abs(sin_lat):
        h = horizontal_radius / cos_lat - N_phi
    else:
        h = z / sin_lat - N_phi * (1.0 - WGS84_E2)

    return np.array([lat, lon, h], dtype=float)


def ned_to_llh(
    p_ned: np.ndarray,
    origin_latitude: float,
    origin_longitude: float,
    origin_height_ellipsoid: float,
) -> np.ndarray:
    """Convert a fixed-origin local NED displacement to WGS-84 LLH.

    Args:
        p_ned: ``[North, East, Down]`` displacement in m with shape ``(3,)``.
        origin_latitude: Geodetic latitude of the fixed NED origin (rad).
        origin_longitude: Longitude of the fixed NED origin (rad).
        origin_height_ellipsoid: Ellipsoidal height of the fixed NED origin
            (m).

    Returns:
        ``[latitude, longitude, ellipsoidal_height]`` in radians, radians, and
        m, respectively, with shape ``(3,)``.

    Notes:
        This is the inverse of ``llh_to_ned`` for a shared origin, subject to
        floating-point roundoff.
    """
    return ecef_to_llh(
        ned_to_ecef(
            p_ned,
            origin_latitude,
            origin_longitude,
            origin_height_ellipsoid,
        )
    )


# Common shorthand aliases. Many texts use "LLA" when the third coordinate is
# ellipsoidal height; the primary names retain the more precise ``LLH`` term.
lla_to_ecef = llh_to_ecef
lla_to_ned = llh_to_ned
ecef_to_lla = ecef_to_llh
ned_to_lla = ned_to_llh
