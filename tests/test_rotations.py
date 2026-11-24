"""
test_rotations.py -- tests for the elementary rotation matrices.

Two flavors of test, and the distinction matters:
  * PROPERTY tests (zero, orthogonality, det, round-trip) check that each matrix
    is *some* valid rotation. They are necessary but NOT sufficient -- a rotation
    in the wrong direction satisfies every one of them, so they cannot catch a
    sign error.
  * ORACLE tests (known_vector, scipy bridge) pin the actual numbers against an
    independent source of truth, so they DO catch sign/direction bugs. In GNC a
    sign error is silent and fatal, so every function needs at least one oracle.

Tolerance note: when an expected entry is 0.0, `assert_allclose`'s default
rtol is useless (rtol * 0 == 0), so an explicit `atol` is required.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from bootins.frames import rotations

# Representative angles (radians), including 0, small, negative, and > pi/2.
ANGLES_RAD = [0.0, 0.1, np.pi / 6, np.pi / 4, -np.pi / 3, 2.0]

# (scipy axis letter, our function) -- one row per elementary rotation.
AXES = [("x", rotations.rot_x), ("y", rotations.rot_y), ("z", rotations.rot_z)]


def test_zero():
    """A zero rotation is the identity for every axis."""
    for _, rot in AXES:
        np.testing.assert_allclose(rot(0.0), np.identity(3), atol=1e-12)


def test_orthogonality():
    """R^T R = I -- the matrix preserves lengths and angles (it is in O(3))."""
    for _, rot in AXES:
        for a in ANGLES_RAD:
            np.testing.assert_allclose(rot(a).T @ rot(a), np.identity(3), atol=1e-12)


def test_det():
    """det = +1 -- a proper rotation, not a reflection (so R is in SO(3))."""
    for _, rot in AXES:
        for a in ANGLES_RAD:
            np.testing.assert_allclose(np.linalg.det(rot(a)), 1.0)


def test_round_trip():
    """The inverse rotation is the negated angle: rot(a) @ rot(-a) = I."""
    for _, rot in AXES:
        for a in ANGLES_RAD:
            np.testing.assert_allclose(rot(a) @ rot(-a), np.identity(3), atol=1e-12)


def test_known_vector_z():
    """Oracle: rotate the FRAME +90 deg about z. A vector along +x_ref, [1,0,0],
    reads as [0,-1,0] in the rotated frame -- the old +x now lies along the new
    -y. (Note the argument is RADIANS: np.pi/2, not 90.)"""
    v_ref = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotations.rot_z(np.pi / 2) @ v_ref,
                               np.array([0.0, -1.0, 0.0]), atol=1e-12)


def test_matches_scipy_passive():
    """Oracle that pins the SIGN of every axis: our passive matrices equal scipy's
    active matrices transposed (active <-> passive are transposes). This is the
    test that catches a flipped rot_y; the property tests above cannot."""
    for axis, rot in AXES:
        for a in ANGLES_RAD:
            expected = Rotation.from_euler(axis, a).as_matrix().T
            np.testing.assert_allclose(rot(a), expected, atol=1e-12)


def test_skew_matches_cross():
    """skew is the cross-product operator written as a matrix.

    Property: skew(v) is antisymmetric (skew(v).T == -skew(v)) -- necessary but,
    like the rotation property tests above, blind to a sign flip on its own.
    Oracle:   skew(v) @ w reproduces np.cross(v, w), an INDEPENDENT source of
    truth that pins all six off-diagonal signs. Iterating all pairs includes the
    v == w case, so v x v = 0 is exercised too (atol required for that zero).
    """
    vecs = np.array([
        [0.0, 1.0, 2.0],
        [2.0, 3.0, 4.0],
        [-2.0, -4.0, -6.0],
        [-3.0, 1.0, 6.0],
        [4.0, -3.0, 0.0],
    ])
    for v in vecs:
        np.testing.assert_allclose(rotations.skew(v).T, -rotations.skew(v), atol=1e-12)
        for w in vecs:
            np.testing.assert_allclose(rotations.skew(v) @ w, np.cross(v, w), atol=1e-12)