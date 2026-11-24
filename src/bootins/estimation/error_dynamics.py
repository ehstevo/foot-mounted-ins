"""Continuous-time linearized error dynamics for the navigation ESEKF.

Where ``mechanization`` propagates the nonlinear nominal state ``(p, v, q)``,
this module models the evolution of its small error state:

    δẋ = F · δx + G · w

The 15-element error vector is

    δx = [ δp(3), δv(3), δψ(3), b_a(3), b_g(3) ]
           pos    vel    tilt   accel   gyro
                                bias    bias

``F`` is a 5-by-5 grid of 3-by-3 blocks. This implementation populates four
nonzero blocks obtained by first-order perturbation of the mechanization
equations:

    δṗ = δv                          -> [δp, δv]  = +I
    δv̇ = −[f^n×]·δψ − C^n_b·b_a       -> [δv, δψ] = −skew(f_nav),  [δv, b_a] = −C^n_b
    δψ̇ = −C^n_b·b_g                   -> [δψ, b_g] = −C^n_b

The bias rows are zero because bias is modeled as a random walk driven by the
process-noise term ``G · w``, rather than deterministic error dynamics.

The nonzero blocks form a bias-driven drift cascade:

    b_g →(−C) δψ →(−[f^n×]) δv →(I) δp     3 hops -> t³   (gyro-bias law)
    b_a →(−C) δv →(I) δp                   2 hops -> t²   (accel-bias law)

For this simplified continuous-time model, ``F`` is nilpotent (``F⁴ = 0``) because
the longest nonzero coupling chain has three hops. Consequently, ``exp(F Δt)``
is a finite polynomial in ``Δt``. This observation does not extend unchanged to
models with additional feedback or Earth-rate terms.

Earth rotation and transport rate are omitted, so the ``[δψ, δψ]`` block is zero.
They introduce attitude self-coupling in a more complete Earth-rate error model.

Conventions: ``C_nb = C^n_b`` maps body to navigation coordinates, and
``f_nav`` is specific force resolved in NED in m/s². At the call site,
``f_nav = C_nb @ (Δv / Δt)``.
"""
import numpy as np

from bootins.frames import rotations


def error_dynamics_F(C_nb: np.ndarray, f_nav: np.ndarray) -> np.ndarray:
    """Assemble the continuous-time 15-by-15 ESEKF dynamics matrix.

    Args:
        C_nb: Nominal body-to-navigation DCM ``C^n_b`` with shape ``(3, 3)``.
        f_nav: Specific force resolved in NED in m/s² with shape ``(3,)``.

    Returns:
        The float64 ``(15, 15)`` matrix ``F`` that linearizes
        ``δẋ = F · δx`` about the supplied nominal operating point.

    Notes:
        Bias random-walk excitation enters the associated process-noise model,
        not this deterministic matrix.
    """
    # Block offsets into δx = [δp, δv, δψ, b_a, b_g]; each block is three wide.
    P, V, PSI, BA, BG = 0, 3, 6, 9, 12
    F = np.zeros((15, 15))

    # Position error accumulates velocity error.
    F[P:P + 3, V:V + 3] = np.eye(3)

    # δv̇ = −[f^n×]·δψ − C^n_b·b_a
    #   [δv, δψ]: a tilt mis-rotates specific force into horizontal acceleration.
    #             The sign follows [a×] b = −[b×] a. At level rest, the zero
    #             bottom row represents the yaw-null direction.
    #   [δv, b_a]: accelerometer bias expressed in navigation coordinates.
    F[V:V + 3, PSI:PSI + 3] = -rotations.skew(f_nav)
    F[V:V + 3, BA:BA + 3] = -C_nb

    # Gyro bias, rotated from body to navigation coordinates, drives attitude error.
    F[PSI:PSI + 3, BG:BG + 3] = -C_nb

    return F
