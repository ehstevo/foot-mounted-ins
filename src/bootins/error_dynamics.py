"""
error_dynamics.py -- the linearized error-state dynamics (the F matrix), M4 Rung 3.

Where mechanization.py propagates the NOMINAL state (p, v, q) forward, this module
holds the filter's model of how the small ERRORS in that state evolve:

    δẋ = F · δx    (+ G·w -- process noise, added in a later rung)

with the 15-state error vector (charter / M4 Rung 2):

    δx = [ δp(3), δv(3), δψ(3), b_a(3), b_g(3) ]
           pos    vel    tilt   accel   gyro
                                bias    bias

F is a 5x5 grid of 3x3 blocks. Only FOUR blocks are nonzero -- each is one
coupling we derived by perturbing a mechanization equation and keeping first
order (M4 Rung 3):

    δṗ = δv                          -> [δp, δv]  = +I
    δv̇ = −[f^n×]·δψ − C^n_b·b_a       -> [δv, δψ] = −skew(f_nav),  [δv, b_a] = −C^n_b
    δψ̇ = −C^n_b·b_g                   -> [δψ, b_g] = −C^n_b

The two bias rows are zero: biases are modeled as random walks, driven only by
noise that lives in G, not F -- the M2 R3 bias-vs-noise split is exactly the wall
between F and G.

Read the nonzero blocks as the drift cascade of M4 Rung 1:

    b_g →(−C) δψ →(−[f^n×]) δv →(I) δp     3 hops -> t³   (gyro-bias law)
    b_a →(−C) δv →(I) δp                   2 hops -> t²   (accel-bias law)

F is nilpotent (F⁴ = 0 -- the longest chain is 3 hops), so e^(FΔt) is a finite
polynomial in Δt -> bare-INS drift is polynomial, not oscillatory or bounded.

Earth rate / transport rate (ω_in) are NEGLECTED for the MVP, which is precisely
why the [δψ, δψ] block is zero (no tilt self-coupling); that block fills in during
the rigor pass, and is where Earth rate would leak into yaw (M3 Thread 1).

Conventions: C_nb = C^n_b (body->nav); f_nav = specific force resolved in NED
(m/s²), which at the call site is C_nb @ (Δv/Δt).
"""
import numpy as np

from bootins.frames import rotations


def error_dynamics_F(C_nb: np.ndarray, f_nav: np.ndarray) -> np.ndarray:
    """Assemble the 15x15 error-state dynamics matrix F at one operating point.

    C_nb  : nominal attitude DCM C^n_b (3x3), body->nav (from quaternion.quat_to_dcm(q))
    f_nav : specific force resolved in NED (3-vector, m/s²) = C_nb @ (Δv/Δt)
    returns: F (15x15, float64) -- the linearization δẋ = F·δx about the nominal.
    """
    # Block offsets into δx = [δp, δv, δψ, b_a, b_g] (each block is 3 wide).
    P, V, PSI, BA, BG = 0, 3, 6, 9, 12
    F = np.zeros((15, 15))

    # δṗ = δv -- position error just accumulates velocity error (first integral).
    F[P:P + 3, V:V + 3] = np.eye(3)

    # δv̇ = −[f^n×]·δψ − C^n_b·b_a
    #   [δv, δψ]: a tilt mis-rotates the up-pointing specific force, leaking it into
    #             the horizontal. The minus is the [a×]b = −[b×]a flip. Its dead
    #             bottom row (f^n ≈ [0,0,−g]) is the yaw-null from Rung 1.
    #   [δv, b_a]: accel bias, rotated body->nav.
    F[V:V + 3, PSI:PSI + 3] = -rotations.skew(f_nav)
    F[V:V + 3, BA:BA + 3] = -C_nb

    # δψ̇ = −C^n_b·b_g -- gyro bias, rotated body->nav, drives tilt (top of cascade).
    F[PSI:PSI + 3, BG:BG + 3] = -C_nb

    return F
