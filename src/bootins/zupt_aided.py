"""zupt_aided.py -- M6 Rung 2: causal wiring of stance detection into the EKF.

M6 Rung 1 built the SHOE / GLRT detector: from a trailing window of IMU
increments, decide whether the foot is plausibly in stance. The missing piece is
to turn that detector decision into an actual ZERO-VELOCITY UPDATE in the
error-state EKF.

One IMU sample at index k plays two distinct roles:

  1. PROCESS INPUT: `(dtheta_k, dv_k, dt_k)` advances the nominal state and
     covariance through `esekf_predict`.
  2. STANCE EVIDENCE: the trailing window ending at k may certify that a
     pseudo-measurement exists at that SAME time:

         z_k = 0   (velocity in NED is known zero)

The order is therefore forced:

    predict with sample k
    -> evaluate the trailing stance window ending at k
    -> if stance, apply the ZUPT update at k

This module implements that causal loop in a small offline runner. It is not yet
the live real-time path; it uses the already-written `shoe_stance_flags`, which
scores the whole measurement sequence causally and returns one stance flag per
sample.
"""
from collections.abc import Iterable

import numpy as np

from bootins.error_state_ekf import NOM_V, esekf_predict, esekf_update
from bootins.mechanization import Measurement
from bootins.zupt import shoe_stance_flags


def _vel_h(nominal: np.ndarray) -> np.ndarray:
    """ZUPT measurement model: the predicted measurement is nominal velocity."""
    return nominal[NOM_V]


_H_VEL = np.zeros((3, 15), dtype=float)
_H_VEL[:, 3:6] = np.eye(3)


def zupt_aided(
    nominal0: np.ndarray,
    P0: np.ndarray,
    measurements: Iterable[Measurement],
    Q: np.ndarray,
    R_zupt: np.ndarray,
    window_size: int,
    threshold: float,
    sigma_a: float,
    sigma_g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the offline causal ZUPT-aided EKF loop over one measurement stream.

    nominal0, P0
        Prior nominal state and covariance before the first IMU sample.
    measurements
        IMU increment stream. We materialize it because the current detector API
        needs the trailing-window view of the whole sequence.
    Q
        Per-step process-noise covariance for `esekf_predict`.
    R_zupt
        Measurement-noise covariance for the velocity pseudo-measurement.
    window_size, threshold, sigma_a, sigma_g
        SHOE / GLRT detector parameters passed straight through.

    returns
        `(flags, nominals, Ps)` where each entry is aligned 1:1 with the input
        samples. `nominals[k]` and `Ps[k]` are the POSTERIOR state/covariance
        after processing sample k: after predict, and after the optional ZUPT
        update if `flags[k]` is true.
    """
    measurements = tuple(measurements)
    flags = shoe_stance_flags(
        measurements, window_size, threshold, sigma_a, sigma_g
    )

    nominal = np.array(nominal0, dtype=float, copy=True)
    P = np.array(P0, dtype=float, copy=True)

    if not measurements:
        empty_nominals = np.empty((0, nominal.shape[0]), dtype=float)
        empty_covariances = np.empty((0, *P.shape), dtype=float)
        return flags, empty_nominals, empty_covariances

    nominals = np.empty((len(measurements), nominal.shape[0]), dtype=float)
    Ps = np.empty((len(measurements), *P.shape), dtype=float)
    z_zupt = np.zeros(3, dtype=float)

    for k, (flag, measurement) in enumerate(zip(flags, measurements, strict=True)):
        dtheta, dv, dt = measurement
        nominal, P = esekf_predict(nominal, P, dtheta, dv, dt, Q)

        # The trailing window ends at sample k, so a detected stance constrains
        # the JUST-predicted state at k, not the prior state at k-1.
        if flag:
            nominal, P = esekf_update(nominal, P, z_zupt, _vel_h, _H_VEL, R_zupt)

        nominals[k] = nominal
        Ps[k] = P

    return flags, nominals, Ps
