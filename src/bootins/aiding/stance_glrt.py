"""SHOE/GLRT foot-stance detection from trailing IMU increment windows.

The detector determines whether a short interval of IMU data is sufficiently
consistent with a stationary foot to support the zero-velocity pseudo-measurement
``v = 0``. It classifies motion evidence only; it does not independently measure
position, heading, or attitude.

For one sample k in a stance window, the idealized model is:

    omega_k ~= 0
    f_k     ~= g * u

where ``omega_k`` is body-frame angular rate in rad/s, ``f_k`` is body-frame
specific force in m/s², ``g`` is local gravity magnitude in m/s², and ``u`` is
the unknown unit specific-force direction in the body frame for that window.
The IMU supplies angle and velocity increments, so the detector first recovers
per-sample mean rates:

    omega_k = dtheta_k / dt_k
    f_k     = dv_k / dt_k

Assuming white Gaussian gyroscope and accelerometer noise with standard
deviations ``sigma_g`` and ``sigma_a``, the dimensionless SHOE/GLRT statistic is

    T = (1/N) * sum_k( ||omega_k||^2 / sigma_g^2
                     + ||f_k - g*u_hat||^2 / sigma_a^2 )

with the best-fit body-frame specific-force direction estimated from the window:

    u_hat = mean(f_k) / ||mean(f_k)||

Small ``T`` is stance-like; large ``T`` is motion-like. The strict decision rule
``T < threshold`` classifies equality as non-stance. This conservative choice
prioritizes avoiding false ZUPTs, which can impose zero velocity during real
motion and reduce covariance unjustifiably.
"""
from collections import deque
from collections.abc import Sequence
from numbers import Integral

import numpy as np

from bootins.estimation.mechanization import G_NED, Increment

G = np.linalg.norm(G_NED)

# Default SHOE detector settings. Keeping these named constants prevents callers
# from silently duplicating detector parameters.
DEFAULT_WINDOW_SIZE = 7
DEFAULT_THRESHOLD = 5e5


class ShoeStanceDetector:
    """Stateful causal SHOE detector for one IMU increment stream.

    The detector retains only its most recent ``window_size`` increments. Each
    ``update`` call returns the stance decision for the newly received
    increment's trailing window; it returns ``False`` until that window is
    full. Instances own independent trailing-window state and must be reset or
    reconstructed at each independent replay or live-data segment.

    Attributes:
        window_size: Positive number of IMU increments in the trailing window.
        threshold: Nonnegative, strict upper bound for a stance score.
        sigma_a: Specific-force noise standard deviation in m/s².
        sigma_g: Angular-rate noise standard deviation in rad/s.
        gravity: Local gravity magnitude in m/s².

    Notes:
        Reset before synthetic IMU coasting or any discontinuity in observed
        IMU evidence. A post-gap score must not combine real increments from
        opposite sides of unobserved motion.
    """

    def __init__(
        self,
        window_size: int,
        threshold: float,
        sigma_a: float,
        sigma_g: float,
        gravity: float = G,
    ) -> None:
        """Create a detector with validated causal-window configuration.

        Args:
            window_size: Positive trailing-window length in samples.
            threshold: Finite nonnegative strict stance threshold.
            sigma_a: Specific-force white-noise standard deviation in m/s².
            sigma_g: Angular-rate white-noise standard deviation in rad/s.
            gravity: Local gravity magnitude in m/s².

        Raises:
            ValueError: If a configuration value is outside its supported
                physical or structural range.
        """
        _validate_window_size(window_size)
        _validate_threshold(threshold)
        _validate_positive_value(sigma_a, "sigma_a")
        _validate_positive_value(sigma_g, "sigma_g")
        _validate_positive_value(gravity, "gravity")

        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.sigma_a = float(sigma_a)
        self.sigma_g = float(sigma_g)
        self.gravity = float(gravity)
        self._trailing_window: deque[Increment] = deque(maxlen=self.window_size)

    def update(self, increment: Increment) -> bool:
        """Ingest one increment and return its causal stance decision.

        Args:
            increment: Body-frame ``(dtheta, dv, dt)`` increment containing an
                angular increment in rad, specific-force increment in m/s, and
                duration in seconds.

        Returns:
            ``True`` when the complete trailing window passes ``T < threshold``;
            ``False`` until the window is full or when it fails the test.

        Raises:
            ValueError: If a complete trailing window contains values for which
                the SHOE statistic is undefined or physically invalid.

        Notes:
            The increment is appended before scoring, so the decision is aligned
            with the window ending at this increment. Incomplete startup history
            is conservatively non-stance.
        """
        self._trailing_window.append(increment)
        if len(self._trailing_window) < self.window_size:
            return False
        return shoe_is_stance(
            tuple(self._trailing_window),
            self.threshold,
            self.sigma_a,
            self.sigma_g,
            self.gravity,
        )


    def reset(self) -> None:
        """Discard trailing evidence after an IMU-evidence discontinuity.

        This clears the causal window. Subsequent real samples must accumulate a
        full fresh window before another stance decision is possible.
        """
        self._trailing_window.clear()


def _stack_window(window: Sequence[Increment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack one increment window into consistently shaped float64 arrays.

    Args:
        window: Nonempty sequence of ``(dtheta, dv, dt)`` IMU increments.

    Returns:
        ``(dtheta, dv, dt)`` arrays with shapes ``(N, 3)``, ``(N, 3)``, and
        ``(N,)``, respectively.

    Raises:
        ValueError: If the window is empty or its increment fields cannot form
            the required sample-aligned arrays.
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
    """Validate the positive integer trailing-window length for SHOE scoring.

    A zero- or negative-length window has no physical meaning, while fractional
    lengths cannot identify a discrete sample window. Reject both before deque
    construction or batch slicing reaches an implementation-specific error.
    """
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, Integral)
        or window_size < 1
    ):
        raise ValueError("window_size must be a positive integer number of samples.")


def _validate_threshold(threshold: float) -> None:
    """Validate the finite, nonnegative SHOE decision threshold."""
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative.")


def _validate_positive_value(value: float, name: str) -> None:
    """Validate one finite, strictly positive physical detector parameter."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


def shoe_glrt(window: Sequence[Increment], sigma_a: float, sigma_g: float,
              gravity: float = G) -> float:
    """Compute the SHOE/GLRT score for one IMU increment window.

    Args:
        window: Nonempty sequence of body-frame ``(dtheta, dv, dt)`` increments
            in rad, m/s, and s.
        sigma_a: Specific-force white-noise standard deviation in m/s².
        sigma_g: Angular-rate white-noise standard deviation in rad/s.
        gravity: Local gravity magnitude in m/s².

    Returns:
        Dimensionless SHOE/GLRT score ``T``; smaller values are more stance-like.

    Raises:
        ValueError: If the window is empty or malformed, a duration is
            nonpositive, a physical scale is invalid, or the mean specific force
            does not define a body-frame direction.

    Notes:
        The unknown body-frame specific-force direction is fitted from the same
        window. This makes the stationary model orientation agnostic while
        retaining sensitivity to varying specific force and angular rate.
    """
    _validate_positive_value(sigma_a, "sigma_a")
    _validate_positive_value(sigma_g, "sigma_g")
    _validate_positive_value(gravity, "gravity")

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


def shoe_is_stance(window: Sequence[Increment], threshold: float,
                   sigma_a: float, sigma_g: float,
                   gravity: float = G) -> bool:
    """Classify an IMU window using the strict SHOE threshold rule.

    Args:
        window: Nonempty sequence of body-frame IMU increments.
        threshold: Finite nonnegative decision threshold.
        sigma_a: Specific-force white-noise standard deviation in m/s².
        sigma_g: Angular-rate white-noise standard deviation in rad/s.
        gravity: Local gravity magnitude in m/s².

    Returns:
        ``True`` exactly when ``shoe_glrt(...) < threshold``.

    Raises:
        ValueError: If the threshold or SHOE-statistic inputs are invalid.

    Notes:
        The strict comparison is intentional:

        stance iff T < threshold

        Equality is non-stance, favoring detector precision over recall.
    """
    _validate_threshold(threshold)

    return shoe_glrt(window, sigma_a, sigma_g, gravity) < threshold


def shoe_scores(increments: Sequence[Increment], window_size: int,
                sigma_a: float, sigma_g: float,
                gravity: float = G) -> np.ndarray:
    """Compute one causal SHOE/GLRT score for each input increment.

    Args:
        increments: Sequence of body-frame ``(dtheta, dv, dt)`` IMU increments.
        window_size: Positive trailing-window length in samples.
        sigma_a: Specific-force white-noise standard deviation in m/s².
        sigma_g: Angular-rate white-noise standard deviation in rad/s.
        gravity: Local gravity magnitude in m/s².

    Returns:
        Float64 array with shape ``(len(increments),)``. Entries before the first
        complete window are ``NaN``; subsequent entries are dimensionless scores.

    Raises:
        ValueError: If ``window_size`` or a complete-window SHOE calculation is
            invalid.

    Notes:
        The score at index ``i`` uses the trailing window ending at ``i``:

        increments[i-window_size+1 : i+1]

        The output is one-to-one aligned with the input stream. The first
        ``window_size - 1`` samples have no complete causal window.
    """
    _validate_window_size(window_size)

    scores = np.full(len(increments), np.nan, dtype=float)

    for i in range(window_size - 1, len(increments)):
        window = increments[i - window_size + 1:i + 1]
        scores[i] = shoe_glrt(window, sigma_a, sigma_g, gravity)

    return scores


def shoe_stance_flags(increments: Sequence[Increment], window_size: int,
                      threshold: float, sigma_a: float, sigma_g: float,
                      gravity: float = G) -> np.ndarray:
    """Compute one causal stance flag for each input increment.

    Args:
        increments: Sequence of body-frame ``(dtheta, dv, dt)`` IMU increments.
        window_size: Positive trailing-window length in samples.
        threshold: Finite nonnegative decision threshold.
        sigma_a: Specific-force white-noise standard deviation in m/s².
        sigma_g: Angular-rate white-noise standard deviation in rad/s.
        gravity: Local gravity magnitude in m/s².

    Returns:
        Boolean array with shape ``(len(increments),)`` aligned one-to-one with
        the input increments.

    Raises:
        ValueError: If detector configuration or a complete-window SHOE
            calculation is invalid.

    Notes:
        This batch wrapper has the same causal decisions as repeatedly calling
        ``ShoeStanceDetector.update`` on a fresh detector instance. The decision
        rule remains

        stance iff score < threshold

        The startup region before the first full window is non-stance.
    """
    detector = ShoeStanceDetector(window_size, threshold, sigma_a, sigma_g, gravity)

    return np.asarray(
        [detector.update(increment) for increment in increments],
        dtype=bool,
    )
