"""Scalar (1-D) Kalman filter -- the teaching foundation for M5 (Rungs 1-2).

The filter stripped to a single scalar state, where every quantity (estimate,
variance, gain) is just a number, so the structure is naked:

    predict:  x_prior = a * x            P_prior = a**2 * P + Q
    update:   K = P / (P + R)            x_hat = x_prior + K * (z - x_prior)
                                         P_hat = (1 - K) * P

Predict GROWS the variance (drift); update SHRINKS it (information). The whole
filter is the tug-of-war between Q and R.

This module is a FROZEN pedagogical artifact -- it establishes the concepts,
it is not the real filter. The actual navigation filter is MULTIVARIATE (state
and covariance are vectors/matrices, with an H matrix mapping the state into
measurement space) and lives in ``kalman.py``. Note the axis of separation is
DIMENSIONALITY (scalar vs multivariate), NOT linearity: both this and the
multivariate KF are linear; the linear-vs-nonlinear split is KF vs EKF, later.
"""


def kalman_update_scalar(
        x_prior: float, P_prior: float, z: float, R: float
    ) -> tuple[float, float]:
    """One scalar Kalman filter measurement update.

    Fuses a prior estimate ``x_prior`` (variance ``P_prior``) with a
    measurement ``z`` (noise variance ``R``) into the minimum-variance
    posterior. The posterior is always at least as certain as the prior
    (``P_hat <= P_prior``), regardless of the measurement value.

    Kalman form:
        K   = P / (P + R)          gain: fraction of the way to move toward z
        y   = z - x_prior          innovation (residual)
        x_hat = x_prior + K * y
        P_hat = (1 - K) * P_prior

    Returns:
        (x_hat, P_hat): posterior estimate and its variance.
    """
    K = P_prior / (P_prior + R)
    innovation = z - x_prior
    x_hat = x_prior + K * innovation
    P_hat = (1 - K) * P_prior

    return (x_hat, P_hat)


def kalman_predict_scalar(
        x: float, P: float, a: float, Q: float
    ) -> tuple[float, float]:
    """One scalar KF time/predict step.

    Propagates the estimate and its uncertainty forward one step under the
    linear model ``x_{k+1} = a * x_k + w``, ``w ~ N(0, Q)``, producing the
    *prior* (predicted, pre-measurement) belief the update step consumes.

    ``a`` = state-transition coefficient (the scalar stand-in for Phi): how
    the deterministic dynamics map this state forward. ``a = 1`` models a
    constant / random walk, ``a < 1`` a decaying (Gauss-Markov) quantity,
    ``a > 1`` an unstable one. ``Q`` = process-noise variance: the model's
    admitted incompleteness.

    Predict form:
        x_prior = a * x            estimate rides the dynamics
        P_prior = a**2 * P + Q     uncertainty scaled by a**2 (variance has two
                                   copies of the error), then INJECTED with Q

    This is the only step that GROWS P -- the mathematical form of drift.
    Without ``+ Q`` the filter would think coasting is free, grow overconfident,
    and stop trusting measurements.

    Returns:
        (x_prior, P_prior): predicted estimate and its (larger) variance.
    """
    x_prior = a * x
    P_prior = a**2 * P + Q

    return (x_prior, P_prior)