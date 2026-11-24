"""Independent convention checks for quaternion frame rotations."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bootins.frames import euler, quaternion


def test_identity_and_normalization_define_the_unit_quaternion_boundary():
    np.testing.assert_allclose(quaternion.quat_to_dcm([1.0, 0.0, 0.0, 0.0]), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(quaternion.normalize([2.0, -1.0, 0.5, 4.0])), 1.0)
    with pytest.raises(ValueError):
        quaternion.normalize(np.zeros(4))


def test_axis_angle_dcm_matches_scipy_for_a_tilted_rotation():
    axis = np.array([1.0, -2.0, 0.5])
    angle = 0.9
    q = quaternion.quat_from_axis_angle(axis, angle)

    np.testing.assert_allclose(
        quaternion.quat_to_dcm(q),
        Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix(),
        atol=1e-12,
    )


def test_dcm_quaternion_round_trip_covers_general_and_half_turn_branches():
    for axis, angle in (([1.0, 2.0, -3.0], -np.pi / 3), ([0.0, 1.0, 0.0], np.pi)):
        matrix = quaternion.quat_to_dcm(quaternion.quat_from_axis_angle(axis, angle))
        np.testing.assert_allclose(quaternion.quat_to_dcm(quaternion.dcm_to_quat(matrix)), matrix, atol=1e-12)


def test_euler_and_quaternion_constructions_share_the_body_to_navigation_convention():
    angles = (0.3, -0.5, 1.1)

    np.testing.assert_allclose(
        quaternion.quat_to_dcm(quaternion.euler_to_quat(*angles)),
        euler.euler_to_dcm(*angles),
        atol=1e-12,
    )


def test_hamilton_product_matches_dcm_composition_order():
    first = quaternion.quat_from_axis_angle([0.0, 0.0, 1.0], 0.7)
    second = quaternion.quat_from_axis_angle([1.0, 0.0, 0.0], -0.4)

    np.testing.assert_allclose(
        quaternion.quat_to_dcm(quaternion.quat_multiply(first, second)),
        quaternion.quat_to_dcm(first) @ quaternion.quat_to_dcm(second),
        atol=1e-12,
    )


def test_quaternion_rate_is_tangent_to_the_unit_sphere():
    q = quaternion.normalize([2.0, -1.0, 0.5, 4.0])
    qdot = quaternion.q_dot(q, np.array([0.2, -0.3, 0.4]))

    assert np.dot(q, qdot) == pytest.approx(0.0, abs=1e-12)
