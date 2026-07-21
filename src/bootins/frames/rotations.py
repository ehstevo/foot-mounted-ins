"""
rotations.py -- elementary (single-axis) rotation matrices.

These are the building blocks of attitude: every orientation we will use is a
product of these three. They follow the project convention fixed in
`docs/01_frames_and_conventions.md`:

  * PASSIVE rotations (coordinate transforms). The frame rotates, the vector
    stays put; the matrix re-expresses the SAME vector's coordinates in the
    rotated frame. For a rotation by `a` about axis k:

        v_rotated = rot_k(a) @ v_reference

    i.e. each function returns the DCM `C^rotated_reference`.
  * Angles are in RADIANS (our SI boundary -- convert deg->rad at the I/O edge,
    never in here).
  * Right-handed, right-hand-rule positive sense about the named axis.

Each matrix is orthonormal with det = +1 (an element of SO(3)), so its inverse
is its transpose: `rot_k(a).T == rot_k(-a)`.

Aliases: rot_x = R1, rot_y = R2, rot_z = R3 in the usual strapdown notation.

Sign note (the classic trap): because the cyclic axis order is x -> y -> z -> x,
the y-rotation mixes z INTO x, so its off-diagonal minus sign sits in the
OPPOSITE corner from the x- and z-rotations. Writing R2 "by symmetry" with the
other two silently gives you rot_y(-a).

Convention bridge: scipy's `Rotation` is ACTIVE, so a scipy matrix is the
transpose of ours -- `rot_k(a) == Rotation.from_euler(k, a).as_matrix().T`.
"""
import numpy as np


def rot_x(angle_rad: float) -> np.ndarray:
    """Passive rotation about the x-axis (R1). Returns C^rotated_reference (3x3)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,   s],
        [0.0,  -s,   c],
    ])


def rot_y(angle_rad: float) -> np.ndarray:
    """Passive rotation about the y-axis (R2). Returns C^rotated_reference (3x3).

    Note the minus sign sits top-right / plus bottom-left -- mirror of rot_x/rot_z.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [  c, 0.0,  -s],
        [0.0, 1.0, 0.0],
        [  s, 0.0,   c],
    ])


def rot_z(angle_rad: float) -> np.ndarray:
    """Passive rotation about the z-axis (R3). Returns C^rotated_reference (3x3)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [  c,   s, 0.0],
        [ -s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric (cross-product) matrix of a 3-vector: skew(v) @ b == v x b.

    A cross product is linear in its second argument, so "cross with v" is a
    linear map -- i.e. a matrix -- and this is that matrix. It is antisymmetric
    (skew(v).T == -skew(v)). Used to write small rotations and cross products as
    matrices in the error-state dynamics (see error_dynamics.py):

            |  0   -v_z   v_y |
    skew =  |  v_z   0   -v_x |
            | -v_y   v_x   0  |
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])