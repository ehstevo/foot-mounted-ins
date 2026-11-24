"""Strapdown INS mechanization in a local-level NED frame.

Each step consumes one body-frame ``(Δθ, Δv, Δt)`` triple, with coning- and
sculling-corrected increments already integrated by the IMU. The navigation
state propagates through the following ordered cascade. Velocity needs the
current attitude to rotate body-frame ``Δv`` into NED, so attitude leads:

    ATTITUDE  q  ── integrate Δθ ───────────────────►  q_new
    VELOCITY  v  ── rotate Δv to nav, then add gravity ► v_new
    POSITION  p  ── integrate v ────────────────────►  p_new

Conventions:
  * ``q = [w, x, y, z]`` is a Hamilton scalar-first body-to-navigation
    quaternion, with ``C^n_b = quaternion.quat_to_dcm(q)``.
  * The navigation frame is North-East-Down (NED), so Down is positive and
    ``g_ned = [0, 0, +9.80665] m/s²``.
  * ``Δθ`` is a body-frame rotation vector in radians. Unit conversion occurs
    at the input boundary.

This local-level model omits Earth rotation ``ω_ie`` and transport rate
``ω_en``. It is therefore not a full Earth-rate strapdown mechanization; those
terms must be added for applications where their accumulated effect is material.
"""
from collections.abc import Iterable

import numpy as np

from bootins.frames import quaternion

G_NED = np.array([0.0, 0.0, 9.80665])   # Local gravity vector in NED; Down is positive.

# Navigation state and IMU increment types. Vectors are float64 ndarrays and
# each duration is in seconds.
State = tuple[np.ndarray, np.ndarray, np.ndarray]  # (p [m, NED], v [m/s, NED], q [w,x,y,z] b->n)
Increment = tuple[np.ndarray, np.ndarray, float]  # (dtheta [rad, body], dv [m/s, body], dt [s])


def attitude_update(q: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """Propagate attitude by one body-frame rotation increment.

    Args:
        q: Current scalar-first ``[w, x, y, z]`` unit quaternion representing
            body-to-navigation attitude.
        dtheta: Body-frame rotation vector for this step in radians.

    Returns:
        The normalized body-to-navigation quaternion after the increment.

    Notes:
        ``Δθ`` is axis-angle data: its magnitude is the rotation angle and its
        normalized direction is the axis. The increment quaternion is
        ``Δq = [cos(|Δθ| / 2), sin(|Δθ| / 2) Δθ̂]``. Because ``Δθ`` is expressed
        in the body frame, it composes on the right:

        q_new = normalize( q ⊗ Δq )

        No ``dt`` argument is required because ``Δθ`` is already the integrated
        angle for the step. Duration enters the velocity and position stages.
    """
    # A near-zero rotation has no defined axis; its attitude update is identity.
    angle = np.linalg.norm(dtheta)
    if np.isclose(angle, 0.0):
        return q
    dq = quaternion.quat_from_axis_angle(dtheta, angle)   # `dtheta` supplies the axis and `angle` its norm.
    q_new = quaternion.quat_multiply(q, dq)               # Body-frame increment composes on the right.
    return quaternion.normalize(q_new)


def velocity_update(v: np.ndarray, q: np.ndarray, dv: np.ndarray,
                    dt: float, gravity: np.ndarray = G_NED) -> np.ndarray:
    """Propagate NED velocity from a body-frame specific-force increment.

    Args:
        v: Current NED velocity in m/s.
        q: Scalar-first body-to-navigation attitude quaternion used to rotate
            ``dv``.
        dv: Body-frame specific-force increment in m/s.
        dt: Increment duration in seconds.
        gravity: NED gravity vector in m/s². The default has positive Down.

    Returns:
        Updated NED velocity in m/s.

    Notes:
        The IMU measures specific force, so gravity is restored after rotating
        ``dv`` into the navigation frame.
    """
    v_new = v + quaternion.quat_to_dcm(q) @ dv + gravity * dt
    return v_new


def position_update(p: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Propagate NED position by integrating velocity over one interval.

    Args:
        p: Current NED position in m.
        v: NED velocity in m/s to integrate over the interval.
        dt: Integration duration in seconds.

    Returns:
        Updated NED position in m.

    Notes:
        ``mechanize_step`` supplies the average of its old and new velocity,
        giving trapezoidal position integration.
    """
    return p + v * dt


def mechanize_step(state: State, dtheta: np.ndarray, dv: np.ndarray, dt: float) -> State:
    """Advance a local-NED navigation state by one IMU increment.

    Args:
        state: ``(p, v, q)`` containing NED position in m, NED velocity in
            m/s, and a scalar-first body-to-navigation quaternion.
        dtheta: Body-frame angular increment in radians.
        dv: Body-frame specific-force increment in m/s.
        dt: Increment duration in seconds.

    Returns:
        The updated ``(p, v, q)`` navigation state in the same conventions.
    """
    p, v, q = state
    q_new = attitude_update(q, dtheta)
    v_new = velocity_update(v, q, dv, dt)
    p_new = position_update(p, 0.5 * (v + v_new), dt)
    return (p_new, v_new, q_new)


def mechanize(state0: State, increments: Iterable[Increment]) -> list[State]:
    """Mechanize an IMU increment sequence from an initial state.

    Args:
        state0: Initial ``(p, v, q)`` navigation state.
        increments: Iterable of body-frame ``(dtheta, dv, dt)`` triples.

    Returns:
        A list containing ``state0`` followed by one propagated state per input
        increment.
    """
    states = [state0]
    state = state0
    for increment in increments:
        state = mechanize_step(state, *increment)
        states.append(state)
    return states
