"""
mechanization.py -- strapdown INS mechanization in the local-level NED frame.

Mechanization is dead reckoning done carefully: each step consumes one
(Δθ, Δv, Δt) triple from the VN-100 (body frame, already pre-integrated by the
device) and propagates the navigation state forward. The three stages form a
cascade, and the ORDER is forced -- velocity needs the current attitude to rotate
the body-frame Δv into NED, so attitude leads:

    ATTITUDE  q  ── integrate Δθ ───────────────────►  q_new      (this file)
    VELOCITY  v  ── rotate Δv to nav, then add gravity ► v_new     (Rung 3, TODO)
    POSITION  p  ── integrate v ────────────────────►  p_new      (Rung 4, TODO)

Conventions (from the charter / M1):
  * Attitude q = [w, x, y, z], Hamilton scalar-first, body->nav
    (C^n_b = quaternion.quat_to_dcm(q)).
  * Nav frame is NED; down is +z; g_ned = [0, 0, +9.81].
  * Δθ is a body-frame rotation vector in RADIANS (deg->rad happens once, at the
    parser boundary -- charter pitfall #2).

Earth rotation (ω_ie) and transport rate (ω_en) are NEGLECTED for the MVP (a
charter decision justified by their magnitude vs the VN-100 noise floor at
walking speed; to be quantified later in M3, then restored in a rigor pass).
"""
import numpy as np

from bootins.frames import quaternion

G_NED = np.array([0.0, 0.0, 9.80665])   # local gravity vector, NED (down = +z)


def attitude_update(q: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """Propagate attitude one step by a body-frame rotation increment.

    This is branch B of the dual-attitude pipeline: open-loop gyro dead-reckoning
    (no aiding). Short-term it should track the device's factory quaternion;
    long-term it will drift -- and seeing that drift is what motivates the EKF.

    Δθ is itself an axis-angle (angle = |Δθ|, axis = Δθ/|Δθ|), so the incremental
    quaternion is Δq = [cos(|Δθ|/2), sin(|Δθ|/2)·Δθ̂] via quat_from_axis_angle.
    Because Δθ is a BODY-frame increment it composes on the RIGHT (M1 Rung 4:
    C^n_b' = C^n_b @ C^b_b'), then we renormalize to stay on the unit sphere:

        q_new = normalize( q ⊗ Δq )

    No Δt argument: Δθ is ALREADY the integrated angle for the step (the VN-100
    pre-integrates), so the attitude stage never sees Δt -- it enters only in the
    velocity and position stages.

    q      : current attitude [w, x, y, z], body->nav (C^n_b)
    dtheta : body-frame rotation vector for this step (rad), = ∫ω dt
    returns: q_new, normalized
    """
    # Guard the stationary-boot / ZUPT-stance case: a near-zero |Δθ| would make the
    # axis normalization inside quat_from_axis_angle a 0/0. No rotation -> no change.
    angle = np.linalg.norm(dtheta)
    if np.isclose(angle, 0.0):
        return q
    dq = quaternion.quat_from_axis_angle(dtheta, angle)   # axis is Δθ (normalized inside), angle = |Δθ|
    q_new = quaternion.quat_multiply(q, dq)               # body increment on the RIGHT
    return quaternion.normalize(q_new)


def velocity_update(v, q, dv, dt, gravity=G_NED):
    """One velocity step: rotate the body-frame Δv into NED and add gravity.

    v       : current velocity, NED (m/s)
    q       : attitude to rotate Δv with, [w,x,y,z] body->nav
    dv      : body-frame specific-force increment for this step (m/s)
    dt      : step duration (s) -- gravity's contribution is g·dt
    gravity : gravity vector in NED (default G_NED)
    returns : v_new, NED
    """
    v_new = v + quaternion.quat_to_dcm(q) @ dv + gravity * dt
    return v_new