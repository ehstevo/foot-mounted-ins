"""Scalar linear Kalman-filter reference implementation.

The filter stripped to a single scalar state, where every quantity (estimate,
variance, gain) is just a number, so the structure is naked:

    predict:  x_prior = a * x            P_prior = a**2 * P + Q
    update:   K = P / (P + R)            x_hat = x_prior + K * (z - x_prior)
                                         P_hat = (1 - K) * P

Prediction increases variance through process uncertainty; measurement updates
reduce it through information from the observation. The relative magnitudes of
``Q`` and ``R`` determine that balance.

This module is a compact scalar reference. Both this and ``kalman.py`` implement
linear filters; their distinction is scalar versus multivariate state dimension,
not linear versus nonlinear dynamics. The operational navigation filter is the
error-state EKF in ``bootins.estimation.esekf``.
"""


def kalman_update_scalar(
        x_prior: float, P_prior: float, z: float, R: float
    ) -> tuple[float, float]:
    """Apply one scalar Kalman-filter measurement update.

    Fuses a prior estimate ``x_prior`` (variance ``P_prior``) with a
    measurement ``z`` (noise variance ``R``) into the minimum-variance
    posterior. The posterior is always at least as certain as the prior
    (``P_hat <= P_prior``), regardless of the measurement value.

    Args:
        x_prior: Prior scalar estimate.
        P_prior: Prior estimate variance.
        z: Scalar measurement of the estimated quantity.
        R: Measurement-noise variance.

    Returns:
        A pair ``(x_hat, P_hat)`` containing the posterior estimate and
        posterior variance.

    Notes:
        The update is

        K   = P / (P + R)          gain: fraction of the way to move toward z
        y   = z - x_prior          innovation (residual)
        x_hat = x_prior + K * y
        P_hat = (1 - K) * P_prior

        ``P_hat <= P_prior`` for nonnegative variances, independently of the
        observed measurement value.
    """
    K = P_prior / (P_prior + R)
    innovation = z - x_prior
    x_hat = x_prior + K * innovation
    P_hat = (1 - K) * P_prior

    return (x_hat, P_hat)


def kalman_predict_scalar(
        x: float, P: float, a: float, Q: float
    ) -> tuple[float, float]:
    """Apply one scalar Kalman-filter prediction step.

    Propagates the estimate and its uncertainty forward one step under the
    linear model ``x_{k+1} = a * x_k + w``, ``w ~ N(0, Q)``, producing the
    *prior* (predicted, pre-measurement) belief the update step consumes.

    Args:
        x: Current scalar estimate.
        P: Current estimate variance.
        a: Scalar state-transition coefficient. ``a = 1`` represents a
            constant or random-walk mean, ``a < 1`` a decaying quantity, and
            ``a > 1`` an unstable quantity.
        Q: Process-noise variance.

    Returns:
        A pair ``(x_prior, P_prior)`` containing the predicted estimate and
        predicted variance.

    Notes:
        The process model is ``x_{k+1} = a * x_k + w`` with
        ``w ~ N(0, Q)``. Its prediction equations are

    Predict form:
        x_prior = a * x            estimate rides the dynamics
        P_prior = a**2 * P + Q     uncertainty scaled by a**2 (variance has two
                                   copies of the error), then INJECTED with Q

        This step admits uncertainty from unmodeled dynamics. Without ``+ Q``,
        a filter can become overconfident while propagating without
        measurements.
    """
    x_prior = a * x
    P_prior = a**2 * P + Q

    return (x_prior, P_prior)
