"""
test_drift.py -- M4 Rung 1: watch a bare INS drift, and pin the drift LAWS.

A strapdown INS is open-loop dead reckoning: it integrates whatever it is given
with no feedback to catch errors. A constant sensor bias therefore does not
average out -- it accumulates, and because the state is a CASCADE of integrals
(attitude -> velocity -> position, M3), an error entering early is integrated
several times on its way to position. That is why the drift SHAPE depends on
which sensor the bias sits in:

    accel bias b_a:  δp = ½ · b_a · t²                        -> t²  (2 integrals)
    gyro  bias b_g:  δp = (1/6) · |f| · b_g · t³              -> t³  (3 integrals)

The gyro law's |f| is the SENSED SPECIFIC FORCE magnitude (≈ g = 9.81 at rest),
NOT the gravity we re-add in the nav frame: a tilt error δψ = b_g·t mis-rotates
the up-pointing specific-force vector, leaking |f|·δψ into the HORIZONTAL axes.
The t³ eventually dominates the t² no matter how small b_g is -- this is why
gyro quality drives long-term navigation, and why ZUPT (which pins t back to one
footstep ~1 s every stance) is so effective.

Tolerance is itself a lesson here:
  * The accel law is EXACT under the discrete update (attitude stays identity, v
    ramps linearly, trapezoidal integration is exact for linear v) -> atol 1e-12.
  * The gyro law is a LINEARIZED, leading-order approximation (sin α ≈ α) and the
    discrete triple-integration differs by O(dt/T) ~ 1% -> a loose rtol, NOT 1e-12.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from bootins import mechanization
from test_mechanization import simulate

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _rest_measurements(dt, N):
    """Exact increments a perfect IMU produces for a LEVEL boot AT REST for N steps.

    Reuses the trajectory oracle with a trivial (stationary, level) trajectory, so
    every Δθ = 0 and every Δv = C^b_n·(−g)·Δt (the specific force holding the boot
    up). Mechanizing these unperturbed must yield zero drift; each test then
    corrupts ONE sensor and checks the resulting drift against its closed-form law.
    """
    _, meas = simulate(np.zeros(3), np.zeros(3), np.zeros(3),
                       Rotation.identity(), np.zeros(3), dt, N)
    return meas


def test_accel_bias_grows_quadratically():
    """A constant accel bias drives position error δp = ½ · b_a · t² (EXACT).

    The bias adds b_a·dt to every Δv. Attitude never moves (Δθ = 0), so C^n_b = I
    and the error passes straight down the velocity->position integrals with no
    gravity coupling. Velocity ramps perfectly linearly, and trapezoidal
    integration is exact for linear velocity, so this matches to machine precision.
    """
    dt, N = 0.01, 100
    b_a = 1e-3                                    # scalar -> same bias on all 3 axes
    meas = [(dth, dv + b_a * dt, dt) for (dth, dv, dt) in _rest_measurements(dt, N)]
    states = mechanization.mechanize((np.zeros(3), np.zeros(3), IDENTITY), meas)

    T = N * dt
    np.testing.assert_allclose(states[-1][0], 0.5 * b_a * T**2 * np.ones(3), atol=1e-12)


def test_gyro_bias_tilts_gravity_into_horizontal():
    """A gyro bias tilts gravity into horizontal: δp ≈ (1/6)·g·b_g·t³ (APPROXIMATE).

    Bias is placed on the NORTH (x) axis ONLY, so the answer is a clean 1-D
    prediction: a North-axis tilt leaks the up-pointing specific force into EAST
    (confirmed empirically; East-axis bias would leak into −North, and yaw into
    nothing -- see test_yaw_bias_causes_no_drift).

    The law is leading-order (sin α ≈ α) and the discrete triple-integral differs
    by O(dt/T) ~ 1.5% at this dt, so the tolerance is a loose rtol -- NOT the
    accel test's 1e-12. atol is also required: the desired North and Down
    components are 0, and rtol·0 = 0 would reject the harmless second-order
    g(1−cos α) ≈ 4e-7 term that shows up on Down (the M1 rtol-vs-zero trap).
    """
    dt, N = 0.01, 100
    b_g = np.array([1e-3, 0.0, 0.0])             # bias about NORTH (x) only
    g = mechanization.G_NED[2]                   # scalar magnitude, not the vector
    meas = [(dth + b_g * dt, dv, dt) for (dth, dv, dt) in _rest_measurements(dt, N)]
    states = mechanization.mechanize((np.zeros(3), np.zeros(3), IDENTITY), meas)

    T = N * dt
    expected_east = (1 / 6) * g * b_g[0] * T**3
    np.testing.assert_allclose(states[-1][0], [0.0, expected_east, 0.0], rtol=3e-2, atol=1e-6)


def test_yaw_bias_causes_no_drift():
    """A yaw (Down-axis) gyro bias produces NO horizontal drift at rest (EXACT zero).

    Rotating the vertical specific-force vector ABOUT the vertical axis does not
    tilt it, so there is no gravity-into-horizontal leak. This is the M3-thread-1
    insight made concrete -- heading/yaw is decoupled from position at rest, which
    is exactly why ZUPT is weak on yaw and heading needs its own aiding (M6/M7).
    Because there is no coupling term at all, this is exact -> a tight atol.
    """
    dt, N = 0.01, 100
    b_g = np.array([0.0, 0.0, 1e-3])             # bias about DOWN (yaw)
    meas = [(dth + b_g * dt, dv, dt) for (dth, dv, dt) in _rest_measurements(dt, N)]
    states = mechanization.mechanize((np.zeros(3), np.zeros(3), IDENTITY), meas)

    np.testing.assert_allclose(states[-1][0], np.zeros(3), atol=1e-9)
