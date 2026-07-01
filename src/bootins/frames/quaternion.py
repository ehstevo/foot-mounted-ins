"""
quaternion.py -- unit-quaternion attitude representation (Hamilton, scalar-first).

A quaternion is the project's ATTITUDE STATE: the thing we store and
propagate through time. It is the engineering sweet spot between the two forms
we already built and broke -- it carries only ONE redundant number with ONE
cheap constraint (|q| = 1, restored by `normalize` in a single divide, vs a
DCM's six-constraint re-orthonormalization), and it has NO singularity (vs
Euler's gimbal lock). That is why the charter keeps attitude as a quaternion and
uses Euler/DCM only as derived views.

Conventions:
  * Hamilton convention, SCALAR-FIRST: q = [w, x, y, z]; w is the scalar part,
    (x, y, z) the vector part. !! scipy is scalar-LAST [x, y, z, w] -- reorder at
    that boundary. Mixing the two is the quaternion version of the NED/ENU sin.
  * A unit quaternion is axis-angle packed: q = [cos(a/2), sin(a/2)*n̂] for a
    rotation of angle `a` about unit axis n̂ (see `quat_from_axis_angle`).
  * `quat_to_dcm` returns C^n_b (body->nav), matching `C^n_b = dcm(q)`.

Dependency layering is one-directional: rotations <- euler <- quaternion. This
module may import `euler`; `euler` must NOT import this one (that would cycle).
The two cross-conversions (quat<->euler) therefore both live here.
"""
import numpy as np

from bootins.frames import euler


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion for a rotation of `angle` (radians) about `axis`.

    The defining encoding of this rung: q = [cos(a/2), sin(a/2)*n̂], where n̂ is
    `axis` normalized to unit length. The half-angle is what the two-sided
    rotation q⊗v⊗q* demands -- each factor of q carries a/2 and they recombine to
    a full a (see `quat_to_dcm`). `axis` need not be unit; it is normalized here.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)])


def normalize(q: np.ndarray) -> np.ndarray:
    """Scale `q` back onto the unit sphere (|q| = 1).

    This is the one-divide repair that keeps a propagated quaternion a valid
    rotation -- the quaternion's answer to a DCM drifting off SO(3). A zero
    quaternion has no direction to preserve, so it is a programming error we
    surface loudly rather than return a silent NaN.
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("cannot normalize a zero quaternion")
    return q / norm


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> C^n_b (body->nav DCM), 3x3 float64.

    These nine entries ARE the sandwich rotation v' = q⊗v⊗q* written as a matrix.
    Expanding that product with the Hamilton rule -- whose scalar part is a DOT
    product and whose vector part is a CROSS product -- gives
        v' = (w^2 - |u|^2) v + 2(u·v) u + 2w (u x v),     u = (x, y, z)
    and collecting the I, u·u^T, and skew-[u]x pieces yields the matrix below.

    Assumes |q| = 1 (call `normalize` first if unsure): a non-unit q produces a
    matrix that scales as well as rotates, i.e. not a valid DCM.
    """
    (w, x, y, z) = q
    C_nb = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x**2 + y**2)],
    ])
    return C_nb


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Unit quaternion -> (phi, theta, psi) Z-Y-X Euler angles (rad), for DISPLAY.

    Routed through the DCM (`dcm_to_euler . quat_to_dcm`) so the Euler convention
    lives in exactly one place (euler.py). Inherits that extraction's pitch =
    +/-90 deg gimbal singularity -- fine, because these are for display only,
    never propagated.
    """
    return euler.dcm_to_euler(quat_to_dcm(q))


def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Z-Y-X Euler angles (rad) -> unit quaternion (the inverse of quat_to_euler).

    Compose three single-axis quaternions -- but in the REVERSE of the angle-list
    axis order: qz ⊗ qy ⊗ qx, NOT qx ⊗ qy ⊗ qz. Here is exactly why. euler_to_dcm
    builds C^b_n = rot_x(phi) @ rot_y(theta) @ rot_z(psi) and then TRANSPOSES to
    C^n_b, and transposing a product reverses its order:
        C^n_b = rot_z(psi).T @ rot_y(theta).T @ rot_x(phi).T
    Each quat_to_dcm(q_k) is that same per-axis C^n_b factor, and quat_multiply is
    a homomorphism (dcm(a ⊗ b) = dcm(a) @ dcm(b), with NO reversal). Matching the
    two forces the quaternion product into the already-reversed order qz ⊗ qy ⊗ qx.
    """
    qx = quat_from_axis_angle([1, 0, 0], phi)
    qy = quat_from_axis_angle([0, 1, 0], theta)
    qz = quat_from_axis_angle([0, 0, 1], psi)
    return quat_multiply(qz, quat_multiply(qy, qx))


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p ⊗ q (scalar-first) -- the composition of rotations.

    Reads right-to-left like a DCM product: p ⊗ q means "do q FIRST, then p"
    (so quat_to_dcm(p ⊗ q) == quat_to_dcm(p) @ quat_to_dcm(q)). This is the ⊗
    behind the attitude update q_new = q ⊗ Δq and the rate law q̇ = ½ q ⊗ [0, ω].

    Splitting each into scalar/vector parts p = (p0, pv), q = (q0, qv), the
    imaginary-unit rules i²=j²=k²=-1 collapse the 16-term expansion to
        p ⊗ q = ( p0*q0 − pv·qv ,  p0*qv + q0*pv + pv x qv )
    The DOT term is the fingerprint of the squares (i²=-1); the CROSS term is the
    fingerprint of anticommutativity (ij=-ji). That cross term is why ⊗ does NOT
    commute: p⊗q − q⊗p = 2(pv x qv), nonzero exactly when the axes aren't
    parallel -- the algebra of "rotations depend on order."
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p0, pv = p[0], p[1:]
    q0, qv = q[0], q[1:]
    w = p0*q0 - np.dot(pv, qv)
    v = p0*qv + q0*pv + np.cross(pv, qv)
    return np.array([w, *v])


def q_dot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Attitude kinematics q̇ = ½ q ⊗ [0, ω] -- the equation mechanization integrates.

    Given the current attitude q (body->nav) and the BODY-frame angular rate omega
    (rad/s), returns the time-derivative q̇. NOTE q̇ is a RATE, not a rotation: it is
    the 'velocity' of the attitude (generally non-unit, and tangent to the |q|=1
    sphere). You integrate it to dead-reckon attitude from gyro rates, e.g.
    q(t+dt) ≈ normalize(q + q̇·dt).

    omega is packed into a PURE quaternion [0, ω] (zero scalar) only so the Hamilton
    product is defined -- it is angular velocity, not an orientation. It sits on the
    RIGHT because a body-frame rate composes on the body side (a nav-frame rate would
    be ½ [0, ω] ⊗ q instead). The ½ is the half-angle surviving the dt->0 limit of
    the discrete update q_new = q ⊗ Δq.
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return 0.5 * quat_multiply(q, [0, *omega])
