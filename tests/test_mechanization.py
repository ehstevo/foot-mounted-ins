"""Physical and reference checks for strapdown mechanization."""

import numpy as np
from scipy.spatial.transform import Rotation

from bootins.estimation import mechanization
from bootins.frames import quaternion


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_attitude_update_composes_body_increments_on_the_right():
    start = quaternion.quat_from_axis_angle([0.0, 0.0, 1.0], np.pi / 2.0)
    increment = np.array([np.pi / 2.0, 0.0, 0.0])

    updated = mechanization.attitude_update(start, increment)
    expected = quaternion.quat_to_dcm(start) @ quaternion.quat_to_dcm(
        quaternion.quat_from_axis_angle([1.0, 0.0, 0.0], np.pi / 2.0)
    )

    np.testing.assert_allclose(quaternion.quat_to_dcm(updated), expected, atol=1e-12)


def test_attitude_update_matches_scipy_for_noncommuting_steps():
    increments = [np.array([np.pi / 6.0, 0.0, 0.0]), np.array([0.0, np.pi / 4.0, 0.0]), np.array([0.0, 0.0, np.pi / 3.0])]
    q = IDENTITY
    reference = Rotation.identity()
    for increment in increments:
        q = mechanization.attitude_update(q, increment)
        reference = reference * Rotation.from_rotvec(increment)

    np.testing.assert_allclose(quaternion.quat_to_dcm(q), reference.as_matrix(), atol=1e-12)


def test_stationary_velocity_update_cancels_gravity_in_any_attitude():
    q = quaternion.quat_from_axis_angle([1.0, -2.0, 0.5], 0.8)
    dv = -quaternion.quat_to_dcm(q).T @ mechanization.G_NED * 0.01

    np.testing.assert_allclose(mechanization.velocity_update(np.zeros(3), q, dv, 0.01), np.zeros(3), atol=1e-12)


def test_velocity_update_rotates_body_specific_force_before_integrating():
    q = quaternion.quat_from_axis_angle([0.0, 0.0, 1.0], np.pi / 2.0)

    velocity = mechanization.velocity_update(np.zeros(3), q, np.array([1.0, 0.0, 0.0]), 0.0)

    np.testing.assert_allclose(velocity, [0.0, 1.0, 0.0], atol=1e-12)


def test_mechanize_step_uses_trapezoidal_position_integration():
    state = (np.zeros(3), np.zeros(3), IDENTITY)
    gravity = np.zeros(3)
    original_velocity_update = mechanization.velocity_update
    try:
        mechanization.velocity_update = lambda velocity, _q, _dv, _dt: velocity + np.array([2.0, 0.0, 0.0])
        position, velocity, _ = mechanization.mechanize_step(state, np.zeros(3), np.zeros(3), 0.5)
    finally:
        mechanization.velocity_update = original_velocity_update

    np.testing.assert_allclose(velocity, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(position, [0.5, 0.0, 0.0])


def test_mechanize_returns_the_initial_state_and_each_posterior():
    states = mechanization.mechanize(
        (np.zeros(3), np.zeros(3), IDENTITY),
        [(np.zeros(3), -mechanization.G_NED * 0.01, 0.01)] * 3,
    )

    assert len(states) == 4
    np.testing.assert_allclose(states[-1][0], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(states[-1][1], np.zeros(3), atol=1e-12)
