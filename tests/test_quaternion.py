"""
test_quaternion.py -- tests for the unit-quaternion <-> DCM/Euler conversions.

Same property/oracle discipline as the other frame tests, with two
quaternion-specific lessons baked into the choices:

  * To test that a quaternion MEANS the intended rotation, build it from an angle
    by an INDEPENDENT path and compare (here: euler_to_dcm from the known angle,
    and scipy). Round-tripping a quaternion through another quaternion library
    never checks the half-angle encoding -- both sides would agree on garbage.
  * Principal-axis rotations zero out the cross-coupling entries (2(xy±wz), ...),
    so TILTED axes are included to exercise all nine entries of quat_to_dcm.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bootins.frames import quaternion
from bootins.frames import euler

ANGLES_RAD = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]

# First three are principal axes; last two are tilted so x, y, z are all nonzero
# and every (incl. cross-coupling) entry of quat_to_dcm gets exercised.
AXES = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0], [1.0, -2.0, 0.5]]

# (phi, theta, psi) triples, pitch kept off +/-90 deg so the round-trip is defined.
# Non-principal so a wrong compose order (qx⊗qy⊗qz vs qz⊗qy⊗qx) can't hide.
EULER_TRIPLES = [(0.0, 0.0, 0.0), (0.3, -0.5, 1.1),
                 (-np.pi / 3, np.pi / 6, np.pi / 4), (0.9, 0.2, -2.0)]


def test_identity():
    """The identity quaternion is no rotation."""
    np.testing.assert_allclose(quaternion.quat_to_dcm([1, 0, 0, 0]), np.identity(3), atol=1e-12)


def test_normalize():
    """normalize puts any nonzero quaternion on the unit sphere; zero is an error."""
    q = quaternion.normalize(np.array([2.0, -1.0, 0.5, 4.0]))
    np.testing.assert_allclose(np.linalg.norm(q), 1.0)
    with pytest.raises(ValueError):
        quaternion.normalize(np.zeros(4))


def test_output_is_valid_dcm():
    """Property: quat_to_dcm yields an orthonormal matrix with det = +1 (SO(3))."""
    for ax in AXES:
        for a in ANGLES_RAD:
            C = quaternion.quat_to_dcm(quaternion.quat_from_axis_angle(ax, a))
            np.testing.assert_approx_equal(np.linalg.det(C), 1.0)
            np.testing.assert_allclose(C.T @ C, np.identity(3), atol=1e-12)


def test_matches_scipy():
    """Oracle: our C^n_b equals scipy's matrix for the same rotation. scipy is
    scalar-LAST, so q=[w,x,y,z] is handed over reordered as [x,y,z,w]."""
    for ax in AXES:
        for a in ANGLES_RAD:
            q = quaternion.quat_from_axis_angle(ax, a)
            C = quaternion.quat_to_dcm(q)
            C_sci = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
            np.testing.assert_allclose(C, C_sci, atol=1e-12)


def test_agrees_with_euler():
    """Cross-rung oracle: a single-axis quaternion equals the SAME rotation built
    INDEPENDENTLY from the known angle via euler_to_dcm (a principal-axis rotation
    maps the angle straight into one Euler slot). The angle -- not the quaternion
    -- is the source of truth, so this catches a half-angle/sign error in
    quat_from_axis_angle or quat_to_dcm, which a quat->quat test cannot."""
    principal = [((1, 0, 0), lambda a: (a, 0.0, 0.0)),    # roll about x
                 ((0, 1, 0), lambda a: (0.0, a, 0.0)),    # pitch about y
                 ((0, 0, 1), lambda a: (0.0, 0.0, a))]    # yaw about z
    for axis, to_euler in principal:
        for a in ANGLES_RAD:
            q = quaternion.quat_from_axis_angle(axis, a)
            np.testing.assert_allclose(quaternion.quat_to_dcm(q),
                                       euler.euler_to_dcm(*to_euler(a)), atol=1e-12)


def test_quat_to_euler():
    """quat_to_euler recovers the angle of a single-axis rotation into the right
    slot (kept off the 0/pi/2/pi wrap points so the comparison is unambiguous)."""
    principal = [((1, 0, 0), 0),   # roll  -> phi
                 ((0, 1, 0), 1),   # pitch -> theta
                 ((0, 0, 1), 2)]   # yaw   -> psi
    for axis, slot in principal:
        for a in [np.pi / 6, np.pi / 4, np.pi / 3]:
            e = quaternion.quat_to_euler(quaternion.quat_from_axis_angle(axis, a))
            np.testing.assert_allclose(e[slot], a, atol=1e-12)


def test_quat_multiply_pins():
    """Concrete, hand-computed products pin the dot/cross transcription and the
    scalar-first ordering. i⊗j = +k vs j⊗i = -k also re-proves anticommutativity
    (the sign flip is the whole 'rotations depend on order' story)."""
    q = [0.0, 0.6, 0.0, 0.8]
    np.testing.assert_allclose(quaternion.quat_multiply([1, 0, 0, 0], q), q, atol=1e-12)  # identity ⊗ q = q
    np.testing.assert_allclose(quaternion.quat_multiply([0, 1, 0, 0], [0, 0, 1, 0]),
                               [0, 0, 0, 1], atol=1e-12)                                  # i ⊗ j = +k
    np.testing.assert_allclose(quaternion.quat_multiply([0, 0, 1, 0], [0, 1, 0, 0]),
                               [0, 0, 0, -1], atol=1e-12)                                 # j ⊗ i = −k


def test_quat_multiply_composes():
    """Oracle: ⊗ composes rotations, so dcm(p ⊗ q) == dcm(p) @ dcm(q) over every
    pair of our test rotations. Compare DCMs, not raw quaternions, to sidestep the
    double-cover sign ambiguity (p⊗q and -(p⊗q) are the SAME rotation). Building
    the quaternions from quat_from_axis_angle keeps full float64 precision, so the
    tolerance stays tight (1e-12) -- hand-typed 0.707s would force a loose 1e-3."""
    quats = [quaternion.quat_from_axis_angle(ax, a) for ax in AXES for a in ANGLES_RAD]
    for p in quats:
        for q in quats:
            lhs = quaternion.quat_to_dcm(quaternion.quat_multiply(p, q))
            rhs = quaternion.quat_to_dcm(p) @ quaternion.quat_to_dcm(q)
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_q_dot_matches_analytic_spin():
    """Analytic oracle: for a CONSTANT spin at rate `rate` about a fixed unit axis
    n̂, the trajectory and its exact derivative are known in closed form:
        q(t)  = [cos(rate*t/2),  sin(rate*t/2) n̂]
        q̇(t)  = (rate/2) [−sin(rate*t/2),  cos(rate*t/2) n̂]
    Feed q(t) and the body rate ω = rate·n̂ to q_dot and compare. Deriving q̇ from
    the angle (not from q_dot itself) is what pins the ½ factor AND the right-side
    multiply -- a q̇-vs-q̇ check could not."""
    for ax in AXES:
        n = np.asarray(ax, float) / np.linalg.norm(ax)
        for rate in [0.5, 2.0, -1.3]:
            for t in [0.0, 0.37, 1.1]:
                a = rate * t
                q = np.array([np.cos(a / 2), *(np.sin(a / 2) * n)])
                expected = (rate / 2) * np.array([-np.sin(a / 2), *(np.cos(a / 2) * n)])
                np.testing.assert_allclose(quaternion.q_dot(q, rate * n), expected, atol=1e-12)


def test_euler_to_quat_matches_dcm():
    """Oracle: euler_to_quat and euler_to_dcm must describe the SAME rotation, so
    dcm(euler_to_quat(a)) == euler_to_dcm(a). Compared as DCMs (double-cover-proof)
    and over NON-principal triples -- this is the test that catches a reversed
    compose order (qx⊗qy⊗qz would fail here, qz⊗qy⊗qx passes)."""
    for a in EULER_TRIPLES:
        np.testing.assert_allclose(quaternion.quat_to_dcm(quaternion.euler_to_quat(*a)),
                                   euler.euler_to_dcm(*a), atol=1e-12)


def test_euler_quat_round_trip():
    """quat_to_euler inverts euler_to_quat off the gimbal singularity (pitch != +/-90)."""
    for a in EULER_TRIPLES:
        np.testing.assert_allclose(quaternion.quat_to_euler(quaternion.euler_to_quat(*a)),
                                   a, atol=1e-12)