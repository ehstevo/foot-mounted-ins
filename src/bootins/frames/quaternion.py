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
    """Z-Y-X Euler angles (rad) -> unit quaternion. Deferred to Rung 4.

    Planned form: compose three axis-angle quaternions in the same Z-Y-X order as
    `euler_to_dcm`, i.e. qz(psi) ⊗ qy(theta) ⊗ qx(phi). Needs `quat_multiply`
    (the Hamilton product), which arrives with the Rung 4 rate kinematics.
    """
    raise NotImplementedError("euler_to_quat lands in Rung 4 (needs quat_multiply)")
