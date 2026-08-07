"""zupt.py -- M6 Rung 1: SHOE / GLRT stance scoring on IMU windows.

The EKF already has the zero-velocity UPDATE scaffold. M6's first new task is
deciding when a short window of IMU data looks enough like a planted foot that
the pseudo-measurement `v = 0` is safe to apply.

For one sample k in a stance window, the idealized model is:

    omega_k ~= 0
    f_k     ~= g * u

where `omega_k` is body angular rate, `f_k` is body specific force, `g` is the
local gravity magnitude, and `u` is the UNKNOWN unit gravity direction in the
body frame for that particular window. The VN-100 gives angle/velocity
INCREMENTS, so the detector first recovers per-sample means:

    omega_k = dtheta_k / dt_k
    f_k     = dv_k / dt_k

Assuming white Gaussian gyro/accel noise with standard deviations `sigma_g` and
`sigma_a`, the SHOE / GLRT test statistic is:

    T = (1/N) * sum_k( ||omega_k||^2 / sigma_g^2
                     + ||f_k - g*u_hat||^2 / sigma_a^2 )

with the best-fit body-frame gravity direction estimated FROM THE WINDOW:

    u_hat = mean(f_k) / ||mean(f_k)||

Small T means "stance-like"; large T means "moving". Threshold tuning belongs
outside this module. Following the charter, equality is treated conservatively:
`T == threshold` counts as NON-STANCE, because a false ZUPT is worse than a
missed one.
"""
from collections.abc import Sequence

import numpy as np

from bootins.mechanization import G_NED, Measurement

G = np.linalg.norm(G_NED)


def _stack_window(window: Sequence[Measurement]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a measurement window into float64 arrays with consistent shapes.

    The repo's IMU convention is "one sample = one (dtheta, dv, dt) triple". The
    GLRT works on a WINDOW of such samples, so this helper stacks them into:

      dtheta : shape (N, 3)
      dv     : shape (N, 3)
      dt     : shape (N,)

    Any mismatch is a caller bug, so we fail loud with a shape-specific message
    instead of letting NumPy broadcast something surprising.
    """
    if not window:
        raise ValueError("SHOE / GLRT needs a non-empty measurement window.")

    dtheta = np.asarray([sample[0] for sample in window], dtype=float)
    dv = np.asarray([sample[1] for sample in window], dtype=float)
    dt = np.asarray([sample[2] for sample in window], dtype=float)

    if dtheta.ndim != 2 or dtheta.shape[1] != 3:
        raise ValueError("dtheta samples must stack to shape (N, 3).")
    if dv.shape != dtheta.shape:
        raise ValueError("dv samples must match dtheta shape (N, 3).")
    if dt.ndim != 1 or dt.shape[0] != dtheta.shape[0]:
        raise ValueError("dt must supply one scalar per sample.")

    return dtheta, dv, dt


def _validate_window_size(window_size: int) -> None:
    """Validate the trailing-window length for stream-level scoring.

    A zero- or negative-length window has no physical meaning for the detector,
    so we reject it explicitly instead of relying on slicing side effects.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1 sample.")


def shoe_glrt(window: Sequence[Measurement], sigma_a: float, sigma_g: float,
              gravity: float = G) -> float:
    """Return the SHOE / GLRT score for one window of IMU increments.

    window  : sequence of (dtheta [rad], dv [m/s], dt [s]) samples
    sigma_a : accel white-noise standard deviation, m/s²
    sigma_g : gyro white-noise standard deviation, rad/s
    gravity : local gravity magnitude, m/s²
    returns : scalar GLRT score T; smaller means more stance-like

    ValueError is raised when the inputs make the test statistic undefined or
    physically meaningless: empty window, bad shapes, nonpositive dt, nonpositive
    noise scales, or near-zero mean specific force (no well-defined gravity
    direction to fit).
    """
    if sigma_a <= 0.0:
        raise ValueError("sigma_a must be positive.")
    if sigma_g <= 0.0:
        raise ValueError("sigma_g must be positive.")
    if gravity <= 0.0:
        raise ValueError("gravity must be positive.")

    dtheta, dv, dt = _stack_window(window)

    if np.any(dt <= 0.0):
        raise ValueError("All dt samples must be positive.")

    omega = dtheta / dt[:, None]
    f = dv / dt[:, None]

    f_mean = np.mean(f, axis=0)
    f_mean_norm = np.linalg.norm(f_mean)
    if np.isclose(f_mean_norm, 0.0):
        raise ValueError("Mean specific force is near zero; gravity direction is undefined.")
    u_hat = f_mean / f_mean_norm

    gyro_term = np.sum(omega**2, axis=1) / sigma_g**2
    accel_term = np.sum((f - gravity * u_hat)**2, axis=1) / sigma_a**2
    return float(np.mean(gyro_term + accel_term))


def shoe_is_stance(window: Sequence[Measurement], threshold: float,
                   sigma_a: float, sigma_g: float,
                   gravity: float = G) -> bool:
    """Classify a window as stance/non-stance using a conservative threshold.

    `threshold` is a detector tuning parameter chosen by the caller. We require
    it to be non-negative, then apply the project-safe decision rule:

        stance iff T < threshold

    Equality is classified as non-stance to bias toward precision over recall.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")

    return shoe_glrt(window, sigma_a, sigma_g, gravity) < threshold


def shoe_scores(measurements: Sequence[Measurement], window_size: int,
                sigma_a: float, sigma_g: float,
                gravity: float = G) -> np.ndarray:
    """Return one causal GLRT score per sample in a measurement stream.

    The score at index i is computed from the trailing window that ENDS at i:

        measurements[i-window_size+1 : i+1]

    so the output stays aligned 1:1 with the input stream. The first
    `window_size - 1` samples do not yet have a full window; their scores are
    returned as NaN to make that start-up region explicit.
    """
    _validate_window_size(window_size)

    scores = np.full(len(measurements), np.nan, dtype=float)

    for i in range(window_size - 1, len(measurements)):
        window = measurements[i - window_size + 1:i + 1]
        scores[i] = shoe_glrt(window, sigma_a, sigma_g, gravity)

    return scores


def shoe_stance_flags(measurements: Sequence[Measurement], window_size: int,
                      threshold: float, sigma_a: float, sigma_g: float,
                      gravity: float = G) -> np.ndarray:
    """Return one causal stance/non-stance flag per input measurement.

    This is the stream-level wrapper around `shoe_scores`: compute the trailing
    GLRT score at each sample, then apply the same conservative threshold rule

        stance iff score < threshold

    Any startup NaNs (before the first full window exists) are classified as
    non-stance, which is the safe direction for ZUPT.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")

    scores = shoe_scores(measurements, window_size, sigma_a, sigma_g, gravity)
    return np.isfinite(scores) & (scores < threshold)
