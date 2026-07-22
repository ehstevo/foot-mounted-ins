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