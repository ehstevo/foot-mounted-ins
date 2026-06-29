"""
test_euler.py -- tests for the Euler-angle <-> DCM conversions.

Same two-flavor split as test_rotations.py:
  * PROPERTY tests (round-trip, valid-DCM) check that the result is *some* valid
    rotation and that the two functions invert each other. Necessary but NOT
    sufficient -- a wrong-but-self-consistent convention passes them.
  * ORACLE test (scipy) pins the actual numbers against an independent source,
    so it catches a convention/sign error. Every direction needs one oracle.

Convention note: `euler_to_dcm` returns `C^n_b` (body->nav), which equals scipy's
ACTIVE matrix for the same angles -- so the oracle compares WITHOUT a transpose
(unlike test_rotations.py, where the passive `rot_*` needed `.T`). The transpose
lives inside `euler_to_dcm` instead.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from bootins.frames import euler

# (phi, theta, psi) triples for the invertible tests. Pitch is kept AWAY from
# +/-90 deg: the round-trip is only well-defined off the gimbal-lock singularity.
ANGLES_RAD = [(0.0, 0.0, 0.0), (0.1, np.pi / 6, np.pi / 4,), (-np.pi / 3, np.pi / 3, np.pi / 4)]

# Gimbal-lock pair: pitch pinned at +90 deg. At theta = +pi/2 the DCM depends
# only on (phi - psi), so these two DISTINCT (roll, yaw) inputs -- both with
# phi - psi = pi/3 -- collapse to the SAME orientation. Gimbal lock made tangible.
ANGLES_GIMBAL_LOCK = [(np.pi / 3, np.pi / 2, 0.0), (0.0, np.pi / 2, -np.pi / 3)]

def test_round_trip():
    """dcm_to_euler inverts euler_to_dcm off the singularity."""
    for a in ANGLES_RAD:
        np.testing.assert_allclose(euler.dcm_to_euler(euler.euler_to_dcm(*a)), a, atol=1e-12)


def test_output_is_valid_dcm():
    """Property: euler_to_dcm yields an orthonormal matrix with det = +1 (SO(3))."""
    for a in ANGLES_RAD:
        C = euler.euler_to_dcm(*a)
        np.testing.assert_allclose(C.T @ C, np.identity(3), atol=1e-12)
        np.testing.assert_approx_equal(np.linalg.det(C), 1)


def test_matches_scipy():
    """Oracle that pins the convention: our C^n_b equals scipy's ACTIVE 'ZYX'
    matrix (uppercase = intrinsic; angle order [psi, theta, phi] = Z, Y, X).
    No transpose -- body->nav already matches scipy's active form."""
    for a in ANGLES_RAD:
        C = euler.euler_to_dcm(*a)
        C_sci = Rotation.from_euler('ZYX', [a[2], a[1], a[0]]).as_matrix()
        np.testing.assert_allclose(C, C_sci, atol=1e-12)


def test_gimbal_lock():
    """Demonstrate the singularity: two different (roll, yaw) pairs at pitch = 90
    deg produce the identical DCM, so the orientation cannot distinguish them."""
    a1 = ANGLES_GIMBAL_LOCK[0]
    a2 = ANGLES_GIMBAL_LOCK[1]

    C1 = euler.euler_to_dcm(*a1)
    C2 = euler.euler_to_dcm(*a2)
    np.testing.assert_allclose(C1, C2, atol=1e-12)
