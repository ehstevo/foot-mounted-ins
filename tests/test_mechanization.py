"""
test_mechanization.py -- tests for the strapdown mechanization stages.

Same property/oracle discipline as the frame tests. The recurring trap here is
that a CONSTANT-spin (fixed-axis) test cannot catch a left-vs-right multiply bug,
because same-axis rotations commute (q ⊗ Δq == Δq ⊗ q when the axes are
parallel). So we ALSO test with an increment whose axis differs from the current
attitude's axis -- there left != right, and the side gets pinned -- and with a
multi-step sequence of DIFFERENT-axis increments (real, non-commuting motion)
against scipy.

Comparisons are done on DCMs, not raw quaternions, to sidestep the double-cover
sign ambiguity (q and -q are the same rotation). The one exception is
test_constant_spin_integrates: positive small steps about a fixed axis stay on
the same branch as quat_from_axis_angle(axis, total), so a raw-quaternion
compare is unambiguous there.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from bootins import mechanization
from bootins.frames import quaternion

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_zero_rotation():
    """The zero-rotation guard: a Δθ of zero leaves attitude unchanged (the
    stationary-boot / stance case that would otherwise be a 0/0 axis normalize).
    Checked from identity AND a tilted attitude so it isn't just the trivial case."""
    np.testing.assert_allclose(mechanization.attitude_update(IDENTITY, np.zeros(3)), IDENTITY, atol=1e-12)
    q = quaternion.quat_from_axis_angle([1, 1, 0], 0.7)
    np.testing.assert_allclose(mechanization.attitude_update(q, np.zeros(3)), q, atol=1e-12)


def test_constant_spin_integrates():
    """Headline test -- proves stepping ACCUMULATES: N small increments of a
    constant spin about a fixed axis equal one single rotation through the total
    angle. The bound is essentially exact (atol=1e-12) because same-axis rotations
    compose exactly (half-angles just add), so there is no commutation error to
    loosen it. (A CHANGING axis would break that exactness -- see the scipy test.)"""
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)
    total_angle = 1.2
    N = 100
    dtheta_step = (total_angle / N) * axis
    q = IDENTITY
    for _ in range(N):
        q = mechanization.attitude_update(q, dtheta_step)

    single_rotation = quaternion.quat_from_axis_angle(axis, total_angle)
    np.testing.assert_allclose(q, single_rotation, atol=1e-12)


def test_composes_on_body_right():
    """Oracle that pins the SIDE: the body increment must multiply on the RIGHT,
    q_new = q ⊗ Δq. We start from an attitude about z and apply an increment about
    x -- DIFFERENT axes, so left and right give different answers. The independent
    expected value is the DCM product C^n_b(q) @ C^n_b(Δq) (the homomorphism
    dcm(q ⊗ Δq) = dcm(q) @ dcm(Δq), proven in test_quaternion). A wrong-sided
    implementation (Δq ⊗ q) would yield dcm(Δq) @ dcm(q) and fail here."""
    q_start = quaternion.quat_from_axis_angle([0, 0, 1], np.pi / 2)   # yaw 90 about z
    dtheta = (np.pi / 2) * np.array([1.0, 0.0, 0.0])                  # roll 90 about x

    q_new = mechanization.attitude_update(q_start, dtheta)

    dq = quaternion.quat_from_axis_angle([1, 0, 0], np.pi / 2)
    expected_dcm = quaternion.quat_to_dcm(q_start) @ quaternion.quat_to_dcm(dq)
    np.testing.assert_allclose(quaternion.quat_to_dcm(q_new), expected_dcm, atol=1e-12)


def test_matches_scipy_sequence():
    """Oracle over a multi-step, CHANGING-axis trajectory -- the real, non-commuting
    motion the other tests don't reach. Feed the same ordered rotation vectors to
    attitude_update and to scipy, and the accumulated attitudes must agree.

    scipy note: a body-frame (intrinsic) increment composes on the RIGHT, so the
    scipy reference is R_total = R_total * R_step (right-multiply), matching our
    q_new = q ⊗ Δq. Each step is built with Rotation.from_rotvec (axis-angle from a
    rotation vector -- the exact analog of quat_from_axis_angle). Compared as DCMs
    to dodge the double cover."""
    steps = [
        (np.pi / 6) * np.array([1.0, 0.0, 0.0]),
        (np.pi / 4) * np.array([0.0, 1.0, 0.0]),
        (np.pi / 3) * np.array([0.0, 0.0, 1.0]),
        0.5 * np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5]),
    ]

    q = IDENTITY
    R_ref = Rotation.identity()
    for dtheta in steps:
        q = mechanization.attitude_update(q, dtheta)
        R_ref = R_ref * Rotation.from_rotvec(dtheta)   # body increment on the RIGHT

    np.testing.assert_allclose(quaternion.quat_to_dcm(q), R_ref.as_matrix(), atol=1e-12)


def test_stays_unit_norm():
    """Property: attitude_update renormalizes, so |q| stays 1 over many steps -- the
    quaternion's cheap defense against drifting off the unit sphere (vs a DCM's
    six-constraint re-orthonormalization)."""
    q = IDENTITY
    dtheta_step = 0.01 * np.array([0.3, -0.7, 0.5])
    for _ in range(500):
        q = mechanization.attitude_update(q, dtheta_step)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# velocity_update
# ---------------------------------------------------------------------------

def test_velocity_rest_cancels_any_attitude():
    """The headline physics/sign test: a boot AT REST does not accelerate, in ANY
    orientation. At rest the accelerometer reads specific force = -g_grav (points
    UP), so its body-frame increment is dv_b = C^b_n · (-g_ned) · dt (the up vector
    expressed in body coords, where C^b_n = C^n_b transposed). Feed that in and the
    two velocity terms must cancel exactly:

        C^n_b · dv_b + g_ned·dt = C^n_b · C^b_n·(-g_ned)·dt + g_ned·dt
                                = -g_ned·dt + g_ned·dt = 0

    A flipped gravity sign would instead give -2·g_ned·dt (the fatal 2g 'launch').
    Testing every attitude proves the cancellation is a frame-independent physical
    fact, not an artifact of q = identity."""
    dt = 0.01
    for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -2, 0.5]):
        for angle in (0.0, 0.6, 2.0):
            q = quaternion.quat_from_axis_angle(ax, angle)
            C_bn = quaternion.quat_to_dcm(q).T                  # nav -> body
            dv_rest = C_bn @ (-mechanization.G_NED) * dt        # what a resting accel reads
            v_new = mechanization.velocity_update(np.zeros(3), q, dv_rest, dt)
            np.testing.assert_allclose(v_new, np.zeros(3), atol=1e-12)


def test_velocity_free_fall():
    """The gravity term in isolation. In free fall the accelerometer is blind to
    gravity and reads nothing (dv = 0, M2 Rung 1), so the g_ned·dt term is all that
    survives -- the boot accelerates DOWNWARD (+z in NED) at exactly g. This is the
    mirror image of the rest test: there the specific-force term cancelled gravity;
    here it is absent, so gravity acts alone."""
    dt = 0.01
    v_new = mechanization.velocity_update(np.zeros(3), IDENTITY, np.zeros(3), dt)
    np.testing.assert_allclose(v_new, mechanization.G_NED * dt, atol=1e-12)


def test_velocity_rotates_dv():
    """Oracle for the rotation term, with gravity ZEROED so only C^n_b·dv survives.
    Compared against scipy's independent active rotation of the same vector -- this
    pins that C^n_b is the active body->nav transform with the right convention (a
    transpose/convention error the physics tests above would not catch). scipy is
    scalar-LAST, so q=[w,x,y,z] is handed over reordered as [x,y,z,w]."""
    q = quaternion.quat_from_axis_angle([1, -2, 0.5], 0.9)
    dv = np.array([0.3, -1.1, 2.0])
    v_new = mechanization.velocity_update(np.zeros(3), q, dv, 1.0, gravity=np.zeros(3))
    expected = Rotation.from_quat([*q[1:], q[0]]).apply(dv)
    np.testing.assert_allclose(v_new, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# position_update
# ---------------------------------------------------------------------------

def test_position_constant_velocity():
    """Analytic oracle: at constant velocity, integrating N steps of dt must land
    at the exact closed-form displacement v·(N·dt). Constant v makes every step's
    v·dt exact (no changing-velocity error), so the bound stays at atol=1e-12."""
    p = np.zeros(3)
    v = np.array([1.0, -0.7, 0.9])
    N = 100
    dt = 0.01
    for _ in range(N):
        p = mechanization.position_update(p, v, dt)
    np.testing.assert_allclose(p, v * (N * dt), atol=1e-12)


def test_position_zero_velocity():
    """Property: zero velocity leaves position unchanged, from several starting
    positions (not just the origin)."""
    p_values = ([0, 0, 0], [1.0, 2.0, 2.0], [-1.0, 2.5, -1.5], [3.0, -1.0, 2.0])
    dt = 0.01
    for p in p_values:
        p_new = mechanization.position_update(p, np.zeros(3), dt)
        np.testing.assert_allclose(p_new, p, atol=1e-12)


# ---------------------------------------------------------------------------
# full mechanization loop -- the trajectory-simulator "test oracle"
# ---------------------------------------------------------------------------

def _scipy_to_quat(R):
    """scipy Rotation -> our scalar-first [w,x,y,z] (scipy is scalar-last)."""
    x, y, z, w = R.as_quat()
    return np.array([w, x, y, z])


def simulate(p0, v0, a_nav, R0, omega_body, dt, N):
    """Inverse-problem trajectory oracle: author a KNOWN-truth trajectory and emit
    the exact IMU increments a perfect sensor riding it would produce.

    Truth is constant nav acceleration + constant body-rate spin -- chosen because
    it is EXACTLY representable by the discrete mechanization (trapezoidal position
    is exact for linear velocity), so a correct loop recovers it to machine
    precision, leaving a bug nowhere to hide:
        v(t) = v0 + a_nav·t
        p(t) = p0 + v0·t + ½·a_nav·t²
        R(t) = R0 * exp(omega_body·t)         (body-frame spin, right-composed)

    Increments per step (see the Rung-5b derivation):
        Δθ_k  = rotvec( R_k⁻¹ · R_{k+1} )          (exact body increment)
        Δv_bk = C^b_n(R_k) · (a_nav - g_ned) · dt  (specific force, body frame)

    scipy generates the attitude truth so the oracle is INDEPENDENT of our
    quaternion module. Returns (truth_states, measurements) where truth_states[k]
    is (p, v, R) and measurements[k] is (dtheta, dv, dt).
    """
    truth = []
    for k in range(N + 1):
        t = k * dt
        p = p0 + v0 * t + 0.5 * a_nav * t**2
        v = v0 + a_nav * t
        R = R0 * Rotation.from_rotvec(omega_body * t)
        truth.append((p, v, R))

    f_nav = a_nav - mechanization.G_NED          # specific force in nav = a − g
    measurements = []
    for k in range(N):
        R_k = truth[k][2]
        dtheta = (R_k.inv() * truth[k + 1][2]).as_rotvec()   # body increment
        dv = R_k.inv().apply(f_nav) * dt                     # C^b_n · f_nav · dt
        measurements.append((dtheta, dv, dt))
    return truth, measurements


def test_mechanize_recovers_trajectory():
    """THE end-to-end test: feed the exact increments of a known translating-AND-
    spinning trajectory through the full loop and recover position, velocity, and
    attitude at every step to machine precision. This validates the whole cascade
    at once -- gravity sign, the frame transforms, the body/nav bookkeeping, the
    trapezoidal position, and the state threading. Attitude is compared as a DCM
    (double-cover-proof) against scipy's independent truth."""
    p0 = np.zeros(3)
    v0 = np.array([1.0, 0.5, -0.2])
    a_nav = np.array([0.3, -0.4, 0.6])
    omega_body = np.array([0.5, -0.3, 0.8])
    R0 = Rotation.from_rotvec([0.2, -0.5, 0.1])
    dt = 0.01
    N = 200

    truth, meas = simulate(p0, v0, a_nav, R0, omega_body, dt, N)
    states = mechanization.mechanize((p0, v0, _scipy_to_quat(R0)), meas)

    assert len(states) == N + 1
    for k in range(N + 1):
        p_t, v_t, R_t = truth[k]
        p_m, v_m, q_m = states[k]
        np.testing.assert_allclose(p_m, p_t, atol=1e-9)
        np.testing.assert_allclose(v_m, v_t, atol=1e-9)
        np.testing.assert_allclose(quaternion.quat_to_dcm(q_m), R_t.as_matrix(), atol=1e-9)


def test_mechanize_rest_stays_put():
    """A tilted boot AT REST must not drift over a full run: no accel, no velocity,
    no rotation -- but a non-level attitude, so gravity is spread across all three
    body axes. The specific-force and gravity terms must cancel every step through
    the whole loop, or the boot 'launches'. The full-loop version of the Rung-3
    rest test."""
    p0 = np.zeros(3)
    v0 = np.zeros(3)
    a_nav = np.zeros(3)
    omega_body = np.zeros(3)
    R0 = Rotation.from_rotvec([0.3, -0.7, 0.2])
    dt = 0.01
    N = 100

    truth, meas = simulate(p0, v0, a_nav, R0, omega_body, dt, N)
    states = mechanization.mechanize((p0, v0, _scipy_to_quat(R0)), meas)

    p_final, v_final, q_final = states[-1]
    np.testing.assert_allclose(p_final, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(v_final, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(quaternion.quat_to_dcm(q_final), R0.as_matrix(), atol=1e-9)