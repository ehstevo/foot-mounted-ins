"""Batch reference implementation of causal ZUPT-aided ESEKF sequencing.

The SHOE/GLRT detector uses a trailing window of IMU increments to determine
whether the foot is plausibly in stance. A detected stance supports the
zero-velocity pseudo-measurement ``z_k = 0`` in the navigation frame.

One IMU sample at index k plays two distinct roles:

  1. Process input: ``(dtheta_k, dv_k, dt_k)`` advances the nominal state and
     covariance through `esekf_predict`.
  2. Stance evidence: the trailing window ending at ``k`` may certify a
     pseudo-measurement at that same time:

         z_k = 0   (velocity in NED is known zero)

The temporal ordering is therefore

    predict with sample k
    -> evaluate the trailing stance window ending at k
    -> if stance, apply the ZUPT update at k

This compact helper materializes the input because ``shoe_stance_flags`` scores
causal trailing windows over a complete sequence before the ESEKF loop begins.
The stance decisions are causal in time, but execution is batch-oriented and is
not a live replay implementation.

The canonical replay path owns raw-frame validation, initialization, and its
own post-initialization detector. Keeping this helper separate isolates the
``predict -> stance decision -> optional ZUPT`` timing contract.
"""
from collections.abc import Iterable

import numpy as np

from bootins.estimation.esekf import NOM_V, esekf_predict, esekf_update
from bootins.estimation.mechanization import Increment
from bootins.aiding.stance_glrt import shoe_stance_flags


def _vel_h(nominal: np.ndarray) -> np.ndarray:
    """ZUPT measurement model: the predicted measurement is nominal velocity."""
    return nominal[NOM_V]


_H_VEL = np.zeros((3, 15), dtype=float)
_H_VEL[:, 3:6] = np.eye(3)


def zupt_aided(
    nominal0: np.ndarray,
    P0: np.ndarray,
    increments: Iterable[Increment],
    Q: np.ndarray,
    R_zupt: np.ndarray,
    window_size: int,
    threshold: float,
    sigma_a: float,
    sigma_g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process one increment sequence with causal ZUPT timing.

    Args:
        nominal0: Initial 16-element nominal ESEKF state.
        P0: Initial ``(15, 15)`` error-state covariance.
        increments: IMU ``(dtheta, dv, dt)`` increments. The function
            materializes this iterable for batch detector scoring.
        Q: Per-predict ``(15, 15)`` process-noise covariance.
        R_zupt: ``(3, 3)`` velocity pseudo-measurement covariance.
        window_size: Number of trailing increments in each SHOE/GLRT window.
        threshold: Strict upper bound for a stance score.
        sigma_a: Specific-force noise scale used by the detector.
        sigma_g: Angular-rate noise scale used by the detector.

    Returns:
        A tuple ``(flags, nominals, Ps)`` aligned one-to-one with the input
        increments. ``nominals[k]`` and ``Ps[k]`` are posteriors after
        prediction with increment ``k`` and, when ``flags[k]`` is true, its
        ZUPT update.
    """
    increments = tuple(increments)
    flags = shoe_stance_flags(
        increments, window_size, threshold, sigma_a, sigma_g
    )

    nominal = np.array(nominal0, dtype=float, copy=True)
    P = np.array(P0, dtype=float, copy=True)

    if not increments:
        empty_nominals = np.empty((0, nominal.shape[0]), dtype=float)
        empty_covariances = np.empty((0, *P.shape), dtype=float)
        return flags, empty_nominals, empty_covariances

    nominals = np.empty((len(increments), nominal.shape[0]), dtype=float)
    Ps = np.empty((len(increments), *P.shape), dtype=float)
    z_zupt = np.zeros(3, dtype=float)

    for k, (flag, increment) in enumerate(zip(flags, increments, strict=True)):
        dtheta, dv, dt = increment
        nominal, P = esekf_predict(nominal, P, dtheta, dv, dt, Q)

        # The trailing window ends at sample k, so a detected stance constrains
        # the JUST-predicted state at k, not the prior state at k-1.
        if flag:
            nominal, P = esekf_update(nominal, P, z_zupt, _vel_h, _H_VEL, R_zupt)

        nominals[k] = nominal
        Ps[k] = P

    return flags, nominals, Ps
