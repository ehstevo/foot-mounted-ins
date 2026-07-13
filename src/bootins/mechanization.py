"""
mechanization.py -- strapdown INS mechanization in the local-level NED frame.

Mechanization is dead reckoning done carefully: each step consumes one
(Δθ, Δv, Δt) triple from the VN-100 (body frame, already pre-integrated by the
device) and propagates the navigation state forward. The three stages form a
cascade, and the ORDER is forced -- velocity needs the current attitude to rotate
the body-frame Δv into NED, so attitude leads:

    ATTITUDE  q  ── integrate Δθ ───────────────────►  q_new
    VELOCITY  v  ── rotate Δv to nav, then add gravity ► v_new
    POSITION  p  ── integrate v ────────────────────►  p_new

Conventions (from the charter / M1):
  * Attitude q = [w, x, y, z], Hamilton scalar-first, body->nav
    (C^n_b = quaternion.quat_to_dcm(q)).
  * Nav frame is NED; down is +z; g_ned = [0, 0, +9.81].
  * Δθ is a body-frame rotation vector in RADIANS (deg->rad happens once, at the
    parser boundary -- charter pitfall #2).

Earth rotation (ω_ie) and transport rate (ω_en) are NEGLECTED for the MVP 
(Earth ≈ 15 °/hr ≈ the gyro bias floor, transport ≈ 0.045 °/hr, both dwarfed
per-stride by ZUPT).
"""
from collections.abc import Iterable

import numpy as np

from bootins.frames import quaternion

G_NED = np.array([0.0, 0.0, 9.80665])   # local gravity vector, NED (down = +z)

# Type aliases for the nav state and one IMU increment. All vectors are float64
# ndarrays; dt is seconds.
State = tuple[np.ndarray, np.ndarray, np.ndarray]   # (p [m, NED], v [m/s, NED], q [w,x,y,z] b->n)
Measurement = tuple[np.ndarray, np.ndarray, float]  # (dtheta [rad, body], dv [m/s, body], dt [s])


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


def velocity_update(v: np.ndarray, q: np.ndarray, dv: np.ndarray,
                    dt: float, gravity: np.ndarray = G_NED) -> np.ndarray:
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


def position_update(p: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """One position step: integrate velocity over the step.

    p  : current position, NED (m)
    v  : velocity to integrate, NED (m/s) -- the loop passes ½(v_old+v_new) for
         trapezoidal integration (see Rung 5)
    dt : step duration (s)
    returns: p_new, NED
    """
    return p + v * dt


def mechanize_step(state: State, dtheta: np.ndarray, dv: np.ndarray, dt: float) -> State:
    """Advance the nav state (p, v, q) by one IMU increment.

    state : (p, v, q) -- position & velocity in NED, attitude body->nav
    dtheta, dv, dt : one measurement triple (body frame, radians / m/s / s)
    returns: the new (p, v, q)
    """
    p, v, q = state
    q_new = attitude_update(q, dtheta)
    v_new = velocity_update(v, q, dv, dt)
    p_new = position_update(p, 0.5 * (v + v_new), dt)
    return (p_new, v_new, q_new)


def mechanize(state0: State, measurements: Iterable[Measurement]) -> list[State]:
    """Run mechanize_step over a sequence of measurements.

    state0       : initial (p, v, q)
    measurements : iterable of (dtheta, dv, dt) triples
    returns      : list of states, one per step, starting with state0
    """
    states = [state0]
    state = state0
    for m in measurements:
        state = mechanize_step(state, *m)
        states.append(state)
    return states