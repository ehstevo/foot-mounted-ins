"""Hamilton scalar-first unit-quaternion attitude representation.

A unit quaternion is the propagated attitude state. It has one redundant scalar
component constrained by ``|q| = 1`` and no Euler-angle singularity. In contrast,
a DCM has six orthonormality constraints. Euler angles and DCMs are derived
representations used for display, coordinate transformation, and conversion.

Conventions:
  * Hamilton scalar-first order: ``q = [w, x, y, z]``, where ``w`` is scalar and
    ``(x, y, z)`` is vector. SciPy uses scalar-last ``[x, y, z, w]`` and must be
    reordered at that boundary.
  * A unit quaternion is axis-angle encoded:
    ``q = [cos(a / 2), sin(a / 2) n_hat]`` for angle ``a`` about unit axis
    ``n_hat``.
  * ``quat_to_dcm`` returns ``C^n_b``, mapping body coordinates to navigation
    coordinates.

Dependencies are one-directional: ``rotations <- euler <- quaternion``. Both
quaternion/Euler conversions live here to avoid an import cycle.
"""
import numpy as np

from bootins.frames import euler


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Construct a unit quaternion from an axis-angle rotation.

    Args:
        axis: Nonzero rotation axis with shape ``(3,)``. It need not be unit
            length.
        angle: Rotation angle in radians.

    Returns:
        Scalar-first unit quaternion ``[w, x, y, z]`` with shape ``(4,)``.

    Notes:
        The encoding is ``q = [cos(a / 2), sin(a / 2) n_hat]``, with normalized
        axis ``n_hat``. The half-angle arises because the quaternion sandwich
        ``q ⊗ v ⊗ q*`` contains two quaternion factors that combine to the full
        rotation angle.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)])


def normalize(q: np.ndarray) -> np.ndarray:
    """Scale a quaternion onto the unit sphere.

    Args:
        q: Quaternion array with any leading scale.

    Returns:
        Quaternion with unit Euclidean norm.

    Raises:
        ValueError: If ``q`` has zero norm.

    Notes:
        Normalization restores a propagated quaternion to the unit sphere, where
        it represents a pure rotation. A zero quaternion has no direction to
        preserve and is rejected rather than converted to a silent NaN.
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("cannot normalize a zero quaternion")
    return q / norm


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a body-to-navigation DCM.

    Args:
        q: Unit scalar-first quaternion ``[w, x, y, z]``.

    Returns:
        Body-to-navigation DCM ``C^n_b`` with shape ``(3, 3)``.

    Notes:
        The entries below are the matrix form of the quaternion sandwich
        ``v' = q ⊗ v ⊗ q*``. Expanding the Hamilton product gives

        ``v' = (w² - |u|²) v + 2(u·v) u + 2w (u × v)``, ``u = (x, y, z)``.

        Collecting the identity, outer-product, and skew terms yields the DCM.
        The function assumes ``|q| = 1``; a non-unit input produces a matrix that
        scales as well as rotates and is not a valid DCM.
    """
    (w, x, y, z) = q
    C_nb = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x**2 + y**2)],
    ])
    return C_nb


def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """Convert a body-to-navigation DCM to a scalar-first unit quaternion.

    Args:
        C: Finite ``(3, 3)`` body-to-navigation rotation matrix ``C^n_b``.

    Returns:
        Scalar-first unit quaternion with shape ``(4,)``. ``q`` and ``-q`` are
        globally equivalent rotations, so either sign may be returned.

    Raises:
        ValueError: If ``C`` is not finite or does not have shape ``(3, 3)``.

    Notes:
        The caller is responsible for supplying a proper rotation matrix; this
        function validates only shape and finiteness. It solves first for the
        largest quaternion component, then recovers the others from off-diagonal
        DCM products. That choice avoids division by a near-zero component for
        rotations near 180 degrees.
    """
    C = np.asarray(C, dtype=float)

    if C.shape != (3, 3):
        raise ValueError("DCM must have shape (3, 3).")
    if not np.isfinite(C).all():
        raise ValueError("DCM must contain only finite values.")

    trace = float(np.trace(C))

    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([
            0.25 * scale,
            (C[2, 1] - C[1, 2]) / scale,
            (C[0, 2] - C[2, 0]) / scale,
            (C[1, 0] - C[0, 1]) / scale,
        ])
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        q = np.array([
            (C[2, 1] - C[1, 2]) / scale,
            0.25 * scale,
            (C[0, 1] + C[1, 0]) / scale,
            (C[0, 2] + C[2, 0]) / scale,
        ])
    elif C[1, 1] > C[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        q = np.array([
            (C[0, 2] - C[2, 0]) / scale,
            (C[0, 1] + C[1, 0]) / scale,
            0.25 * scale,
            (C[1, 2] + C[2, 1]) / scale,
        ])
    else:
        scale = 2.0 * np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        q = np.array([
            (C[1, 0] - C[0, 1]) / scale,
            (C[0, 2] + C[2, 0]) / scale,
            (C[1, 2] + C[2, 1]) / scale,
            0.25 * scale,
        ])

    return normalize(q)

def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Convert a unit quaternion to Z-Y-X Euler angles for display.

    Args:
        q: Unit scalar-first body-to-navigation quaternion.

    Returns:
        ``(phi, theta, psi)`` roll, pitch, and yaw angles in radians.

    Notes:
        Conversion routes through ``dcm_to_euler(quat_to_dcm(q))`` so the Euler
        convention has one implementation. It inherits Euler extraction's
        pitch ``±90°`` gimbal singularity.
    """
    return euler.dcm_to_euler(quat_to_dcm(q))


def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Convert intrinsic Z-Y-X Euler angles to a unit quaternion.

    Args:
        phi: Roll angle in radians.
        theta: Pitch angle in radians.
        psi: Yaw angle in radians.

    Returns:
        Scalar-first unit body-to-navigation quaternion with shape ``(4,)``.

    Notes:
        The quaternion product order is the reverse of the Euler angle-list
        order: ``qz ⊗ qy ⊗ qx``. ``euler_to_dcm`` forms
        ``C^b_n = rot_x(phi) @ rot_y(theta) @ rot_z(psi)`` then transposes it,
        which reverses its product order:

        ``C^n_b = rot_z(psi).T @ rot_y(theta).T @ rot_x(phi).T``.

        Each per-axis quaternion generates the corresponding ``C^n_b`` factor,
        and ``quat_multiply`` preserves product order:
        ``dcm(a ⊗ b) = dcm(a) @ dcm(b)``.
    """
    qx = quat_from_axis_angle([1, 0, 0], phi)
    qy = quat_from_axis_angle([0, 1, 0], theta)
    qz = quat_from_axis_angle([0, 0, 1], psi)
    return quat_multiply(qz, quat_multiply(qy, qx))


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the scalar-first Hamilton product ``p ⊗ q``.

    Args:
        p: Scalar-first quaternion with shape ``(4,)``.
        q: Scalar-first quaternion with shape ``(4,)``.

    Returns:
        Scalar-first Hamilton product with shape ``(4,)``.

    Notes:
        Quaternion products compose rotations right-to-left: ``p ⊗ q`` applies
        ``q`` first and then ``p``. Thus
        ``quat_to_dcm(p ⊗ q) = quat_to_dcm(p) @ quat_to_dcm(q)``. This operation
        underlies both body-frame attitude propagation and quaternion kinematics.

        For ``p = (p0, pv)`` and ``q = (q0, qv)``, Hamilton multiplication is

        ``p ⊗ q = (p0*q0 - pv·qv, p0*qv + q0*pv + pv × qv)``.

        The cross-product term makes the product noncommutative; rotation order
        matters whenever the vector parts are not parallel.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p0, pv = p[0], p[1:]
    q0, qv = q[0], q[1:]
    w = p0*q0 - np.dot(pv, qv)
    v = p0*qv + q0*pv + np.cross(pv, qv)
    return np.array([w, *v])


def q_dot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Evaluate body-rate quaternion kinematics ``q̇ = 1/2 q ⊗ [0, ω]``.

    Args:
        q: Current scalar-first body-to-navigation quaternion.
        omega: Body-frame angular rate in rad/s with shape ``(3,)``.

    Returns:
        Quaternion derivative with shape ``(4,)``.

    Notes:
        ``q̇`` is a derivative rather than an orientation: it is generally
        non-unit and tangent to the unit-quaternion sphere. A first-order rate
        integration would use ``normalize(q + q̇ * dt)``.

        ``omega`` is packed into the pure quaternion ``[0, omega]`` only to
        apply the Hamilton product. It multiplies on the right because the rate
        is expressed in the body frame; a navigation-frame rate would multiply
        on the left. The factor ``1/2`` is the differential limit of the
        axis-angle half-angle relation.
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return 0.5 * quat_multiply(q, [0, *omega])
