"""Elementary passive direction-cosine matrices and cross-product matrices.

These matrices are the elementary building blocks for attitude transformations.
They use the following coordinate-transform convention:

  * Passive rotations: the frame rotates while the physical vector remains
    fixed. For rotation ``a`` about axis ``k``:

        v_rotated = rot_k(a) @ v_reference

    Each function returns the DCM ``C^rotated_reference``.
  * Angles are in radians; unit conversion belongs at the I/O boundary.
  * Right-handed, right-hand-rule positive sense about the named axis.

Each matrix is orthonormal with det = +1 (an element of SO(3)), so its inverse
is its transpose: `rot_k(a).T == rot_k(-a)`.

``R1``, ``R2``, and ``R3`` are standard strapdown aliases for ``rot_x``,
``rot_y``, and ``rot_z``, respectively.

The ``y``-rotation mixes ``z`` into ``x``, so its off-diagonal signs occupy the
opposite corners from the ``x`` and ``z`` matrices. Writing ``R2`` by apparent
symmetry with the other two yields ``rot_y(-a)``.

SciPy's ``Rotation`` matrices are active, so their matrices are the transpose of
these passive DCMs: ``rot_k(a) == Rotation.from_euler(k, a).as_matrix().T``.
"""
import numpy as np


def rot_x(angle: float) -> np.ndarray:
    """Return the passive x-axis rotation DCM ``R1``.

    Args:
        angle: Right-hand-rule rotation angle (rad).

    Returns:
        ``(3, 3)`` DCM ``C^rotated_reference``.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,   s],
        [0.0,  -s,   c],
    ])


def rot_y(angle: float) -> np.ndarray:
    """Return the passive y-axis rotation DCM ``R2``.

    Args:
        angle: Right-hand-rule rotation angle (rad).

    Returns:
        ``(3, 3)`` DCM ``C^rotated_reference``.

    Notes:
        The negative sign is in the upper-right element and the positive sign
        in the lower-left element, opposite the corresponding sign placement in
        ``rot_x`` and ``rot_z``.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [  c, 0.0,  -s],
        [0.0, 1.0, 0.0],
        [  s, 0.0,   c],
    ])


def rot_z(angle: float) -> np.ndarray:
    """Return the passive z-axis rotation DCM ``R3``.

    Args:
        angle: Right-hand-rule rotation angle (rad).

    Returns:
        ``(3, 3)`` DCM ``C^rotated_reference``.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [  c,   s, 0.0],
        [ -s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def skew(v: np.ndarray) -> np.ndarray:
    """Construct the skew-symmetric cross-product matrix of a 3-vector.

    Args:
        v: Vector with shape ``(3,)``.

    Returns:
        ``(3, 3)`` matrix such that ``skew(v) @ b == v x b``.

    Notes:
        A cross product is linear in its second argument, so ``skew(v)`` is the
        matrix representation of the map ``b -> v x b``. It is antisymmetric:
        ``skew(v).T == -skew(v)``.

        It represents small rotations and cross products in the error-state
        dynamics:

            |  0   -v_z   v_y |
    skew =  |  v_z   0   -v_x |
            | -v_y   v_x   0  |
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
