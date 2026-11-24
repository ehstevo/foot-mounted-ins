"""Z-Y-X yaw-pitch-roll Euler-angle and DCM conversions.

Euler angles provide a compact human-readable attitude representation:
roll ``phi``, pitch ``theta``, and yaw ``psi``. They are appropriate for display
and interchange, but not propagation, because they are singular at pitch
``±90°``. Navigation propagation uses unit quaternions instead.

Conventions:
  * Intrinsic Z-Y-X: yaw about ``z``, then pitch about the rotated ``y``, then
    roll about the final rotated ``x``.
  * ``(phi, theta, psi) = (roll, pitch, yaw)`` are in radians.
  * ``euler_to_dcm`` returns ``C^n_b``, which maps body coordinates to
    navigation coordinates; ``dcm_to_euler`` expects the same convention.

For the same angles, the body-to-navigation DCM equals SciPy's active rotation
matrix, unlike the passive elementary matrices in ``rotations.py``:

    euler_to_dcm(phi, theta, psi) == Rotation.from_euler(
        'ZYX', [psi, theta, phi]
    ).as_matrix()
"""
import numpy as np

from bootins.frames import rotations


def euler_to_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """Convert intrinsic Z-Y-X Euler angles to a body-to-navigation DCM.

    Args:
        phi: Roll angle in radians.
        theta: Pitch angle in radians.
        psi: Yaw angle in radians.

    Returns:
        The ``(3, 3)`` float64 body-to-navigation DCM ``C^n_b``.

    Notes:
        The function first forms the passive navigation-to-body transform from
        elementary rotations, then transposes it:

        C^b_n = rot_x(phi) @ rot_y(theta) @ rot_z(psi)   # nav->body
        C^n_b = (C^b_n).T                                # body->nav  (returned)

        The transpose changes from the passive ``rot_*`` convention to the
        body-to-navigation convention.
    """
    C_bn = rotations.rot_x(phi) @ rotations.rot_y(theta) @ rotations.rot_z(psi)
    C_nb = C_bn.T
    return C_nb


def dcm_to_euler(C: np.ndarray) -> tuple[float, float, float]:
    """Convert a body-to-navigation DCM to intrinsic Z-Y-X Euler angles.

    Args:
        C: Body-to-navigation DCM ``C^n_b`` with shape ``(3, 3)``.

    Returns:
        A tuple ``(phi, theta, psi)`` of roll, pitch, and yaw in radians.

    Notes:
        Pitch uses a single matrix entry, while roll and yaw use ``atan2`` to
        recover their quadrants:

        theta = asin(-C[2,0])
        phi   = atan2(C[2,1], C[2,2])
        psi   = atan2(C[1,0], C[0,0])

        At ``theta = ±90°``, ``cos(theta) = 0`` makes both ``atan2`` arguments
        vanish. Roll and yaw are then inseparable because only their combination
        survives in the DCM. This gimbal lock is inherent to any three-angle
        representation.
    """
    theta = np.asin(-C[2, 0])
    phi = np.atan2(C[2, 1], C[2, 2])
    psi = np.atan2(C[1, 0], C[0, 0])
    return (phi, theta, psi)
