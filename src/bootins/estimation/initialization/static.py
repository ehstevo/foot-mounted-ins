"""Static inertial alignment, gyro-bias estimation, and stance selection.

A declared stationary IMU interval supplies mean body-frame specific force for
gravity leveling and mean body-frame angular rate for gyro-bias estimation. The
resulting attitude maps the body frame into a run-local Forward-Right-Down
(FRD) frame, whose horizontal orientation is a convention rather than geographic
North/East. Geographic NED alignment requires independent course evidence.

The implemented mechanization omits Earth rotation. Consequently, the static
mean angular rate is interpreted as gyro bias rather than separated into gyro
bias and Earth-rate components.
"""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from bootins.estimation.mechanization import Increment
from bootins.frames.quaternion import dcm_to_quat


_FORWARD_REFERENCE_B = np.array([0.0, -1.0, 0.0])

@dataclass(frozen=True)
class StaticLevelInitialization:
    """Static-interval facts and the resulting body-to-run-frame attitude.

    ``f`` denotes the INS-only run-local Forward-Right-Down frame: ``+x`` is
    forward, ``+y`` is right, and ``+z`` is Down. ``q_fb`` maps body axes into
    that frame. Its horizontal orientation is a local convention, not a claim
    of geographic North/East alignment.

    Attributes:
        q_fb: Scalar-first ``[w, x, y, z]`` unit quaternion mapping body to
            run-local FRD.
        mean_specific_force_b: Time-weighted mean body-frame specific force in
            m/s² with shape ``(3,)``.
        duration: Total duration of the static interval (s).
        increment_count: Number of IMU increments contributing to the mean.

    Notes:
        The dataclass prevents field rebinding, but its NumPy array fields are
        not independently immutable.
    """

    q_fb: np.ndarray
    mean_specific_force_b: np.ndarray
    duration: float
    increment_count: int


@dataclass(frozen=True)
class StaticZuptInitialization:
    """Static-level facts plus the gyro bias estimated from that interval.

    ``gyro_bias_b`` is a body-frame angular-rate bias in rad/s under the
    Earth-rate-omitted mechanization model.

    Attributes:
        static_level: Gravity-leveling result from the same static interval.
        gyro_bias_b: Mean body-frame angular rate in rad/s with shape ``(3,)``.

    Notes:
        The frozen dataclass does not make the nested NumPy arrays immutable.
    """

    static_level: StaticLevelInitialization
    gyro_bias_b: np.ndarray


@dataclass(frozen=True)
class StanceInitializationInterval:
    """The first detected contiguous stance interval that satisfies startup.

    ``start_index`` is inclusive and ``replay_start_index`` is exclusive, so
    slicing the original increment sequence by these indices retains the whole
    duration-crossing increment in initialization and starts replay at the
    following increment.

    Attributes:
        start_index: Inclusive index of the first flagged increment.
        replay_start_index: Exclusive index immediately after the retained
            duration-crossing increment.
        duration: Sum of ``dt`` across the selected contiguous stance (s).
        increment_count: Number of selected contiguous stance increments.
    """

    start_index: int
    replay_start_index: int
    duration: float
    increment_count: int


def _validate_stance_selection_config(
    minimum_stance_duration: float,
    startup_search_limit: float,
) -> tuple[float, float]:
    """Return validated finite timing requirements for stance initialization."""
    try:
        minimum_stance_duration = float(minimum_stance_duration)
        startup_search_limit = float(startup_search_limit)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Stance-initialization timing requirements must be finite and positive."
        ) from error

    if (
        not np.isfinite(minimum_stance_duration)
        or minimum_stance_duration <= 0.0
    ):
        raise ValueError("Minimum stance duration must be finite and positive.")
    if not np.isfinite(startup_search_limit) or startup_search_limit <= 0.0:
        raise ValueError("Startup search limit must be finite and positive.")
    if minimum_stance_duration > startup_search_limit:
        raise ValueError(
            "Minimum stance duration cannot exceed the startup search limit."
        )

    return minimum_stance_duration, startup_search_limit


class StanceInitializationSelector:
    """Causally select the first startup stance suitable for initialization.

    Each ``update`` receives one clean IMU increment and its already-computed
    Boolean stance flag. The selector owns timing, raw-contiguity, and stream
    indexing only; it neither detects stance nor retains increments for the
    physical initialization calculation. It returns ``None`` until the first
    qualifying stance is complete, then transitions permanently to selected.

    Attributes:
        minimum_stance_duration: Required contiguous flagged duration (s).
        startup_search_limit: Maximum elapsed input duration considered (s).

    Notes:
        The selector retains only timing and indexing state. The caller owns the
        increments required for physical static initialization.
    """

    def __init__(
        self,
        minimum_stance_duration: float,
        startup_search_limit: float,
    ) -> None:
        """Create a selector with validated startup timing requirements.

        Args:
            minimum_stance_duration: Required contiguous stance duration (s).
            startup_search_limit: Maximum elapsed input duration considered
                during startup (s).

        Raises:
            ValueError: If either duration is nonfinite or nonpositive, or if
                the minimum stance duration exceeds the search limit.
        """
        (
            self.minimum_stance_duration,
            self.startup_search_limit,
        ) = _validate_stance_selection_config(
            minimum_stance_duration,
            startup_search_limit,
        )
        self._elapsed_dt = 0.0
        self._stance_dt = 0.0
        self._stance_count = 0
        self._next_index = 0
        self._stance_start_index = 0
        self._selected_interval: StanceInitializationInterval | None = None

    def update(
        self,
        increment: Increment,
        stance_flag: bool,
    ) -> StanceInitializationInterval | None:
        """Process one increment and return a selected interval when ready.

        Args:
            increment: IMU ``(dtheta, dv, dt)`` increment; ``dt`` must be a
                finite positive scalar in seconds.
            stance_flag: Scalar Python or NumPy Boolean causal stance decision.

        Returns:
            A completed ``StanceInitializationInterval`` when this increment
            first satisfies the duration requirement; otherwise ``None``.

        Raises:
            RuntimeError: If an interval was already selected.
            ValueError: If the flag is not Boolean, ``dt`` is unusable, or the
                search limit is exceeded.

        Notes:
            A false flag clears the current candidate. The duration-crossing
            increment remains in initialization, so the following increment is
            the first available for replay or estimation.
        """
        if self._selected_interval is not None:
            raise RuntimeError("Initialization interval already found.")
        if not isinstance(stance_flag, (bool, np.bool_)):
            raise ValueError("Stance flag must be a Boolean scalar.")

        try:
            dt = float(increment[2])
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "Static-initialization increment must provide a scalar dt."
            ) from error
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "Static-initialization dt must be finite and positive."
            )

        current_index = self._next_index
        self._next_index += 1
        self._elapsed_dt += dt
        if self._elapsed_dt > self.startup_search_limit:
            raise ValueError(
                "No qualifying contiguous stance interval completed within "
                "the startup search limit."
            )

        if not stance_flag:
            self._stance_count = 0
            self._stance_dt = 0.0
            return None

        if self._stance_count == 0:
            self._stance_start_index = current_index
        self._stance_count += 1
        self._stance_dt += dt

        if self._stance_dt >= self.minimum_stance_duration:
            self._selected_interval = StanceInitializationInterval(
                start_index=self._stance_start_index,
                replay_start_index=self._next_index,
                duration=self._stance_dt,
                increment_count=self._stance_count,
            )
            return self._selected_interval

        return None


def initialize_static_level(
    increments: Iterable[Increment],
) -> StaticLevelInitialization:
    """Estimate run-local FRD leveling from one declared stationary interval.

    Args:
        increments: Iterable of body-frame ``(dtheta, dv, dt)`` IMU increments.
            ``dv`` has units m/s and ``dt`` has units s.

    Returns:
        Static leveling facts and the body-to-run-local-FRD quaternion.

    Raises:
        ValueError: If no increment is supplied, an increment has invalid ``dv``
            shape or finite values, a duration is invalid, the mean specific
            force has no direction, or the body-forward reference is vertical.

    Notes:
        ``sum(dv) / sum(dt)`` is the time-weighted mean body specific force. At
        rest, its direction defines body Up; its negative defines body Down.
        A fixed body-forward reference is projected into the horizontal plane to
        complete the run-local FRD basis. This establishes only a local yaw
        convention, not geographic heading. The interval is selected outside
        this function and is not reclassified here.
    """
    total_dv_b = np.zeros(3, dtype=float)
    duration = 0.0
    increment_count = 0

    for _dtheta, dv, dt in increments:
        dv = np.asarray(dv, dtype=float)
        dt = float(dt)

        if dv.shape != (3,):
            raise ValueError("Static-initialization dv must have shape (3,).")
        if not np.isfinite(dv).all():
            raise ValueError("Static-initialization dv must be finite.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "Static-initialization dt must be finite and positive."
            )

        total_dv_b += dv
        duration += dt
        increment_count += 1

    if increment_count == 0:
        raise ValueError("Static initialization requires at least one increment.")

    mean_specific_force_b = total_dv_b / duration
    mean_force_norm = np.linalg.norm(mean_specific_force_b)
    if np.isclose(mean_force_norm, 0.0):
        raise ValueError(
            "Static initialization requires a nonzero mean specific-force vector."
        )

    up_b = mean_specific_force_b / mean_force_norm
    down_b = -up_b

    forward_horizontal_b = (
        _FORWARD_REFERENCE_B
        - np.dot(_FORWARD_REFERENCE_B, down_b) * down_b
    )
    forward_horizontal_norm = np.linalg.norm(forward_horizontal_b)
    if np.isclose(forward_horizontal_norm, 0.0):
        raise ValueError(
            "Boot-forward reference is vertical; cannot define run-local forward."
        )
    forward_b = forward_horizontal_b / forward_horizontal_norm
    right_b = np.cross(down_b, forward_b)
    right_b /= np.linalg.norm(right_b)

    C_fb = np.vstack((forward_b, right_b, down_b))
    q_fb = dcm_to_quat(C_fb)

    return StaticLevelInitialization(
        q_fb=q_fb,
        mean_specific_force_b=mean_specific_force_b,
        duration=duration,
        increment_count=increment_count,
    )


def initialize_static_zupt(
    increments: Iterable[Increment],
) -> StaticZuptInitialization:
    """Estimate static leveling and body-frame gyro bias for one stance interval.

    Args:
        increments: Iterable of stationary body-frame ``(dtheta, dv, dt)`` IMU
            increments in rad, m/s, and s.

    Returns:
        Static leveling result and mean body-frame gyro bias in rad/s.

    Raises:
        ValueError: If an angular increment has invalid shape or finite values,
            or if static-level initialization rejects the interval.

    Notes:
        Under the Earth-rate-omitted model, ``sum(dtheta) / sum(dt)`` estimates
        gyro bias. Accelerometer bias is not initialized because stationary data
        alone cannot separate horizontal accelerometer bias from leveling tilt.
    """
    increments = tuple(increments)
    total_dtheta_b = np.zeros(3, dtype=float)

    for dtheta, _dv, _dt in increments:
        dtheta = np.asarray(dtheta, dtype=float)

        if dtheta.shape != (3,):
            raise ValueError("Static-ZUPT initialization dtheta must have shape (3,).")
        if not np.isfinite(dtheta).all():
            raise ValueError("Static-ZUPT initialization dtheta must be finite.")

        total_dtheta_b += dtheta

    static_level = initialize_static_level(increments)

    gyro_bias_b = total_dtheta_b / static_level.duration

    return StaticZuptInitialization(
        static_level=static_level,
        gyro_bias_b=gyro_bias_b,
    )


def select_first_qualified_stance(
    increments: Sequence[Increment],
    stance_flags: np.ndarray,
    minimum_stance_duration: float,
    startup_search_limit: float,
) -> StanceInitializationInterval:
    """Select the first qualifying contiguous stance from batch evidence.

    Args:
        increments: Sequence of IMU ``(dtheta, dv, dt)`` increments.
        stance_flags: One-dimensional Boolean causal stance decisions aligned
            one-to-one with ``increments``.
        minimum_stance_duration: Required contiguous flagged duration (s).
        startup_search_limit: Maximum elapsed input duration considered (s).

    Returns:
        First ``StanceInitializationInterval`` that satisfies the requirements.

    Raises:
        ValueError: If flags are not a matching one-dimensional Boolean array,
            selector input is invalid, or no interval qualifies before the
            search limit.

    Notes:
        A false flag resets the candidate; reporting-only merging is not used.
        The batch wrapper has the same selection semantics as sequential
        ``StanceInitializationSelector.update`` calls.
    """
    selector = StanceInitializationSelector(
        minimum_stance_duration,
        startup_search_limit,
    )

    flags = np.asarray(stance_flags)
    if flags.ndim != 1 or flags.dtype != np.bool_:
        raise ValueError("Stance flags must be a one-dimensional Boolean array.")
    if flags.size != len(increments):
        raise ValueError("Stance flags must contain one entry per increment.")

    for flag, increment in zip(flags, increments, strict=True):
        interval = selector.update(increment, flag)
        if interval is not None:
            return interval

    raise ValueError(
        "No qualifying contiguous stance interval completed within the "
        "startup search limit."
    )
