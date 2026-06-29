"""
euler.py -- Euler-angle <-> DCM conversions (Z-Y-X yaw-pitch-roll).

Euler angles are the project's HUMAN-READABLE attitude format: three numbers
(roll phi, pitch theta, yaw psi) instead of a 9-entry DCM. They are for DISPLAY
and I/O only -- never for propagation -- because they hit a singularity at
pitch = +/-90 deg (gimbal lock; see `dcm_to_euler`). The state we actually carry
and integrate is the quaternion; this module just translates to and
from the form a human reads.

Conventions:
  * Order: Z-Y-X intrinsic -- yaw about z, then pitch about the NEW y, then roll
    about the NEWEST x. The aerospace standard; it puts gimbal lock at +/-90 deg
    pitch, a pose a walker rarely visits.
  * Angles (phi, theta, psi) = (roll, pitch, yaw), in RADIANS.
  * `euler_to_dcm` returns `C^n_b` (body->nav), matching the stored-attitude
    convention `C^n_b = dcm(q)`. `dcm_to_euler` expects that same `C^n_b`.

Convention bridge: a body->nav DCM equals scipy's ACTIVE rotation matrix for the
same angles, so (unlike the passive `rot_*` in rotations.py) these compare to
scipy WITHOUT a transpose:
    euler_to_dcm(phi,theta,psi) == Rotation.from_euler('ZYX',[psi,theta,phi]).as_matrix()
"""
import numpy as np

from bootins.frames import rotations


def euler_to_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """Z-Y-X Euler angles (radians) -> `C^n_b` (body->nav DCM), 3x3 float64.

    Build the nav->body coordinate transform as the intrinsic Z-Y-X composition
    of the passive elementary rotations, then transpose to get body->nav:

        C^b_n = rot_x(phi) @ rot_y(theta) @ rot_z(psi)   # nav->body
        C^n_b = (C^b_n).T                                # body->nav  (returned)

    The transpose is what flips us from the passive `rot_*` convention into the
    body->nav (= scipy-active) convention -- see the module docstring.
    """
    C_bn = rotations.rot_x(phi) @ rotations.rot_y(theta) @ rotations.rot_z(psi)
    C_nb = C_bn.T
    return C_nb


def dcm_to_euler(C: np.ndarray) -> tuple[float, float, float]:
    """`C^n_b` (body->nav DCM) -> (phi, theta, psi) = (roll, pitch, yaw), radians.

    Pitch reads off a single entry; roll and yaw come from RATIOS of entries
    (via atan2, which also fixes the quadrant). The indices are those of `C^n_b`
    -- the transpose of the entries you'd use on a `C^b_n` matrix:

        theta = asin(-C[2,0])
        phi   = atan2(C[2,1], C[2,2])
        psi   = atan2(C[1,0], C[0,0])

    GIMBAL LOCK: at theta = +/-90 deg, cos(theta) = 0 makes C[2,1]=C[2,2]=0 and
    C[1,0]=C[0,0]=0, so both atan2 calls become atan2(0, 0) -- roll and yaw are
    no longer separable (only their combination survives in the matrix). This is
    the unavoidable singularity of ANY 3-angle representation, and the reason we
    propagate quaternions, not Euler angles.
    """
    theta = np.asin(-C[2, 0])
    phi = np.atan2(C[2, 1], C[2, 2])
    psi = np.atan2(C[1, 0], C[0, 0])
    return (phi, theta, psi)
