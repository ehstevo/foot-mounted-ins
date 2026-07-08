"""
test_mechanization.py -- tests for the strapdown mechanization stages.

Same property/oracle discipline as the frame tests. The recurring trap here is
that a CONSTANT-spin (fixed-axis) test cannot catch a left-vs-right multiply bug,
because same-axis rotations commute (q ⊗ Δq == Δq ⊗ q when the axes are
parallel). So we ALSO test with an increment whose axis differs from the current
attitude's axis -- there left != right, and the side gets pinned -- and with a
multi-step sequence of DIFFERENT-axis increments (real, non-commuting motion)
against scipy.

Comparisons are done on DCMs, not raw quaternions, to sidestep the double-cover
sign ambiguity (q and -q are the same rotation). The one exception is
test_constant_spin_integrates: positive small steps about a fixed axis stay on
the same branch as quat_from_axis_angle(axis, total), so a raw-quaternion
compare is unambiguous there.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from bootins import mechanization
from bootins.frames import quaternion

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_zero_rotation():
    """The zero-rotation guard: a Δθ of zero leaves attitude unchanged (the
    stationary-boot / stance case that would otherwise be a 0/0 axis normalize).
    Checked from identity AND a tilted attitude so it isn't just the trivial case."""
    np.testing.assert_allclose(mechanization.attitude_update(IDENTITY, np.zeros(3)), IDENTITY, atol=1e-12)
    q = quaternion.quat_from_axis_angle([1, 1, 0], 0.7)
    np.testing.assert_allclose(mechanization.attitude_update(q, np.zeros(3)), q, atol=1e-12)


def test_constant_spin_integrates():
    """Headline test -- proves stepping ACCUMULATES: N small increments of a
    constant spin about a fixed axis equal one single rotation through the total
    angle. The bound is essentially exact (atol=1e-12) because same-axis rotations
    compose exactly (half-angles just add), so there is no commutation error to
    loosen it. (A CHANGING axis would break that exactness -- see the scipy test.)"""
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)
    total_angle = 1.2
    N = 100
    dtheta_step = (total_angle / N) * axis
    q = IDENTITY
    for _ in range(N):
        q = mechanization.attitude_update(q, dtheta_step)

    single_rotation = quaternion.quat_from_axis_angle(axis, total_angle)
    np.testing.assert_allclose(q, single_rotation, atol=1e-12)


def test_composes_on_body_right():
    """Oracle that pins the SIDE: the body increment must multiply on the RIGHT,
    q_new = q ⊗ Δq. We start from an attitude about z and apply an increment about
    x -- DIFFERENT axes, so left and right give different answers. The independent
    expected value is the DCM product C^n_b(q) @ C^n_b(Δq) (the homomorphism
    dcm(q ⊗ Δq) = dcm(q) @ dcm(Δq), proven in test_quaternion). A wrong-sided
    implementation (Δq ⊗ q) would yield dcm(Δq) @ dcm(q) and fail here."""
    q_start = quaternion.quat_from_axis_angle([0, 0, 1], np.pi / 2)   # yaw 90 about z
    dtheta = (np.pi / 2) * np.array([1.0, 0.0, 0.0])                  # roll 90 about x

    q_new = mechanization.attitude_update(q_start, dtheta)

    dq = quaternion.quat_from_axis_angle([1, 0, 0], np.pi / 2)
    expected_dcm = quaternion.quat_to_dcm(q_start) @ quaternion.quat_to_dcm(dq)
    np.testing.assert_allclose(quaternion.quat_to_dcm(q_new), expected_dcm, atol=1e-12)


def test_matches_scipy_sequence():
    """Oracle over a multi-step, CHANGING-axis trajectory -- the real, non-commuting
    motion the other tests don't reach. Feed the same ordered rotation vectors to
    attitude_update and to scipy, and the accumulated attitudes must agree.

    scipy note: a body-frame (intrinsic) increment composes on the RIGHT, so the
    scipy reference is R_total = R_total * R_step (right-multiply), matching our
    q_new = q ⊗ Δq. Each step is built with Rotation.from_rotvec (axis-angle from a
    rotation vector -- the exact analog of quat_from_axis_angle). Compared as DCMs
    to dodge the double cover."""
    steps = [
        (np.pi / 6) * np.array([1.0, 0.0, 0.0]),
        (np.pi / 4) * np.array([0.0, 1.0, 0.0]),
        (np.pi / 3) * np.array([0.0, 0.0, 1.0]),
        0.5 * np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5]),
    ]

    q = IDENTITY
    R_ref = Rotation.identity()
    for dtheta in steps:
        q = mechanization.attitude_update(q, dtheta)
        R_ref = R_ref * Rotation.from_rotvec(dtheta)   # body increment on the RIGHT

    np.testing.assert_allclose(quaternion.quat_to_dcm(q), R_ref.as_matrix(), atol=1e-12)


def test_stays_unit_norm():
    """Property: attitude_update renormalizes, so |q| stays 1 over many steps -- the
    quaternion's cheap defense against drifting off the unit sphere (vs a DCM's
    six-constraint re-orthonormalization)."""
    q = IDENTITY
    dtheta_step = 0.01 * np.array([0.3, -0.7, 0.5])
    for _ in range(500):
        q = mechanization.attitude_update(q, dtheta_step)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# velocity_update
# ---------------------------------------------------------------------------

def test_velocity_rest_cancels_any_attitude():
    """The headline physics/sign test: a boot AT REST does not accelerate, in ANY
    orientation. At rest the accelerometer reads specific force = -g_grav (points
    UP), so its body-frame increment is dv_b = C^b_n · (-g_ned) · dt (the up vector
    expressed in body coords, where C^b_n = C^n_b transposed). Feed that in and the
    two velocity terms must cancel exactly:

        C^n_b · dv_b + g_ned·dt = C^n_b · C^b_n·(-g_ned)·dt + g_ned·dt
                                = -g_ned·dt + g_ned·dt = 0

    A flipped gravity sign would instead give -2·g_ned·dt (the fatal 2g 'launch').
    Testing every attitude proves the cancellation is a frame-independent physical
    fact, not an artifact of q = identity."""
    dt = 0.01
    for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -2, 0.5]):
        for angle in (0.0, 0.6, 2.0):
            q = quaternion.quat_from_axis_angle(ax, angle)
            C_bn = quaternion.quat_to_dcm(q).T                  # nav -> body
            dv_rest = C_bn @ (-mechanization.G_NED) * dt        # what a resting accel reads
            v_new = mechanization.velocity_update(np.zeros(3), q, dv_rest, dt)
            np.testing.assert_allclose(v_new, np.zeros(3), atol=1e-12)


def test_velocity_free_fall():
    """The gravity term in isolation. In free fall the accelerometer is blind to
    gravity and reads nothing (dv = 0, M2 Rung 1), so the g_ned·dt term is all that
    survives -- the boot accelerates DOWNWARD (+z in NED) at exactly g. This is the
    mirror image of the rest test: there the specific-force term cancelled gravity;
    here it is absent, so gravity acts alone."""
    dt = 0.01
    v_new = mechanization.velocity_update(np.zeros(3), IDENTITY, np.zeros(3), dt)
    np.testing.assert_allclose(v_new, mechanization.G_NED * dt, atol=1e-12)


def test_velocity_rotates_dv():
    """Oracle for the rotation term, with gravity ZEROED so only C^n_b·dv survives.
    Compared against scipy's independent active rotation of the same vector -- this
    pins that C^n_b is the active body->nav transform with the right convention (a
    transpose/convention error the physics tests above would not catch). scipy is
    scalar-LAST, so q=[w,x,y,z] is handed over reordered as [x,y,z,w]."""
    q = quaternion.quat_from_axis_angle([1, -2, 0.5], 0.9)
    dv = np.array([0.3, -1.1, 2.0])
    v_new = mechanization.velocity_update(np.zeros(3), q, dv, 1.0, gravity=np.zeros(3))
    expected = Rotation.from_quat([*q[1:], q[0]]).apply(dv)
    np.testing.assert_allclose(v_new, expected, atol=1e-12)