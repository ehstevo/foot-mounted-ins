"""Reference-frame and WGS-84 geometry checks."""

import numpy as np

from bootins.frames import geodesy


def test_llh_to_ecef_matches_ellipsoid_cardinal_points():
    np.testing.assert_allclose(geodesy.llh_to_ecef(0.0, 0.0, 0.0), [geodesy.WGS84_A, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(geodesy.llh_to_ecef(0.0, np.pi / 2.0, 0.0), [0.0, geodesy.WGS84_A, 0.0], atol=1e-9)
    np.testing.assert_allclose(geodesy.llh_to_ecef(np.pi / 2.0, 0.0, 0.0), [0.0, 0.0, geodesy.WGS84_B], atol=1e-9)


def test_ecef_to_ned_basis_has_the_expected_equatorial_orientation():
    matrix = geodesy.ecef_to_ned_matrix(0.0, 0.0)

    np.testing.assert_allclose(matrix, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)


def test_ecef_and_ned_are_inverse_for_a_known_equatorial_displacement():
    origin = np.array([geodesy.WGS84_A, 0.0, 0.0])
    position = origin + np.array([3.0, 4.0, 5.0])

    ned = geodesy.ecef_to_ned(position, origin, 0.0, 0.0)

    np.testing.assert_allclose(ned, [5.0, 4.0, -3.0], atol=1e-12)
    np.testing.assert_allclose(geodesy.ned_to_ecef(ned, 0.0, 0.0, 0.0), position, atol=1e-9)


def test_llh_ecef_llh_round_trip_recovers_a_nontrivial_position():
    llh = np.array([0.7, -1.1, 125.0])

    np.testing.assert_allclose(geodesy.ecef_to_llh(geodesy.llh_to_ecef(*llh)), llh, atol=1e-8)


def test_ned_llh_ned_round_trip_preserves_a_fixed_origin_displacement():
    origin = (0.7, -1.1, 125.0)
    displacement = np.array([25.0, -10.0, 3.0])

    np.testing.assert_allclose(
        geodesy.llh_to_ned(*geodesy.ned_to_llh(displacement, *origin), *origin),
        displacement,
        atol=1e-6,
    )


def test_height_and_latitude_signs_match_ned_convention():
    origin = (0.8, -1.1, 125.0)

    assert geodesy.llh_to_ned(0.8, -1.1, 135.0, *origin)[2] < 0.0
    assert geodesy.llh_to_ned(0.8 + 1e-6, -1.1, 125.0, *origin)[0] > 0.0
