"""
test_error_dynamics.py -- M4 Rung 3: prove the F matrix reproduces the drift laws.

This is the payoff test for error_dynamics_F. In Rung 1 we watched the full
NONLINEAR mechanization engine drift and pinned two closed-form laws
(test_drift.py). Here we show the LINEAR error model carries the same physics.

The linear error model is δẋ = F·δx. At level rest F is CONSTANT (C_nb = I,
f_nav = -g·ẑ don't move), so the exact finite-time solution is the matrix
exponential -- the state-transition matrix:

    δx(T) = Φ·δx(0),    Φ = e^(F·T)

Because F is nilpotent (F⁴ = 0), that exponential is a FINITE polynomial:

    e^(F·T) = I + F·T + (1/2!)·F²T² + (1/3!)·F³T³

and the drift laws are just individual terms of it. A lone accel bias reaches
δp through F² (2 hops), a lone gyro bias through F³ (3 hops):

    accel: δp = −½·b_a·T²          ← the ½  is the 1/2! of the exponential
    gyro:  δp = −(1/6)·|f|·b_g·T³  ← the 1/6 is the 1/3! of the exponential

So the "2 integrals → t²" / "3 integrals → t³" cascade of Rung 1 is literally
e^(F·T) unpacked, one power of F per integration.

EXACT vs APPROXIMATE (the contrast worth savoring): test_drift.py matched the
gyro law only to ~1.5% (rtol=3e-2), because the nonlinear DISCRETE engine carries
discretization + sin α ≈ α error. Here the linear model propagated by expm matches
to MACHINE PRECISION (atol=1e-12) -- because the linear model and the closed-form
law are the SAME object (the exact continuous solution of δẋ = F·δx).

Why no separate "F vs the nonlinear engine" test: test_drift.py already proved
engine ≈ law (loosely); this proves F = law (exactly); together they give
F ≈ engine, i.e. F is the correct linearization of our mechanization.

Sign note: the error state is δ = true − nominal. In test_drift.py we measured the
NOMINAL's drift (truth = 0), which was +East for a North gyro bias; the error state
is its negative, hence the minus signs in the expected values below.
"""
import numpy as np
from scipy.linalg import expm

from bootins.error_dynamics import error_dynamics_F
from bootins.mechanization import G_NED

# Operating point: level boot at rest. C_nb = I, and the specific force is minus
# gravity (it points UP, ≈ [0,0,-g] in NED -- the M2 R1 fact). F is constant here,
# so a single Φ = e^(F·T) propagates the whole window exactly.
C_nb = np.eye(3)
f_nav = -G_NED
T = 1.0                      # 1 s window (matches test_drift.py's N·dt = 100·0.01)
F = error_dynamics_F(C_nb, f_nav)
Phi = expm(F * T)


def test_accel_bias():
    """A constant accel bias drives δp = −½·b_a·T² (the F² / 1-over-2! term).

    b_a sits in the b_a slot (indices 9:12) of the initial error state; Φ carries
    it two hops (b_a → δv → δp). With C_nb = I the three axes are decoupled, so we
    can bias all three at once and check element-wise. Exact -> tight atol.
    """
    b_a_vec = np.array([1e-3, 2e-3, -1e-3])

    delta_x0 = np.zeros(15)        # error state at t=0: all zero except the bias
    delta_x0[9:12] = b_a_vec

    delta_p = (Phi @ delta_x0)[0:3]

    np.testing.assert_allclose(delta_p, -0.5 * b_a_vec * T**2, atol=1e-12)


def test_gyro_bias():
    """A North gyro bias drives δp = −(1/6)·g·b_g·T³ into EAST (the F³ / 1-over-3! term).

    b_g sits in the b_g slot (indices 12:15); Φ carries it three hops
    (b_g → δψ → δv → δp). Bias on North (x) only, so the answer is a clean 1-D
    prediction on East, with North and Down zero (atol covers those zeros).
    """
    b_g_vec = np.array([1e-3, 0.0, 0.0])

    delta_x0 = np.zeros(15)
    delta_x0[12:15] = b_g_vec

    g = G_NED[2]
    delta_p = (Phi @ delta_x0)[0:3]

    np.testing.assert_allclose(delta_p, [0.0, -(1 / 6) * g * b_g_vec[0] * T**3, 0.0],
                               atol=1e-12)


def test_yaw_bias_null():
    """A DOWN (yaw) gyro bias produces NO position drift -- the yaw-null, from F.

    The δv←δψ block is −[f^n×], and f^n ≈ [0,0,-g] has a dead third column, so a
    yaw error (δψ on Down) maps to zero velocity error -> zero δp. This is the same
    yaw-null test_drift.py saw in the nonlinear engine, now falling straight out of
    the matrix structure (why ZUPT is weak on heading -> heading needs its own aiding).
    """
    b_g_vec = np.array([0.0, 0.0, 1e-3])

    delta_x0 = np.zeros(15)
    delta_x0[12:15] = b_g_vec

    delta_p = (Phi @ delta_x0)[0:3]

    np.testing.assert_allclose(delta_p, np.zeros(3), atol=1e-12)
