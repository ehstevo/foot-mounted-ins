"""Load the strict YAML profile for INS propagation with ZUPT aiding.

This module converts a source-agnostic profile into the startup, SHOE detector,
and ESEKF settings used by the INS-plus-ZUPT replay. The profile contains
initial nominal quantities and *discrete per-increment* covariance matrices;
it does not convert continuous-time noise densities or establish a physical
noise model. Capture source, device connection, logging, and evolving filter
posteriors belong to other layers.

The parser enforces the exact public YAML shape, finite scalar/vector entries,
and the documented nonnegative or positive constraints. It does not determine
whether a chosen detector threshold or covariance is scientifically suitable
for a particular sensor, gait, or trajectory.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import yaml


_STATE_GROUPS = (
    "position",
    "velocity",
    "attitude",
    "accel_bias",
    "gyro_bias",
)


@dataclass(frozen=True)
class ZuptINSConfig:
    """Store typed configuration for one INS-plus-ZUPT replay profile.

    Attributes:
        minimum_stance_duration: Required contiguous startup stance duration in
            s.
        startup_search_limit: Maximum startup search duration in s. It must be
            at least ``minimum_stance_duration``.
        initial_accel_bias_b: ``(3,)`` initial nominal accelerometer bias in
            the body frame, in m/s².
        P0: ``(15, 15)`` diagonal initial error-state covariance. Its ordered
            three-axis groups are position (m²), velocity ((m/s)²), attitude
            (rad²), accelerometer bias ((m/s²)²), and gyro bias ((rad/s)²).
        Q: ``(15, 15)`` diagonal process covariance added once per real IMU
            increment in the same error-state order and units as ``P0``. It is
            a configured discrete covariance, not a continuous-time spectral
            density or a value scaled here by ``dt``.
        Q_coast: ``(15, 15)`` additional discrete covariance for each synthetic
            coast increment. Only position, velocity, and attitude groups may
            be nonzero; bias groups are required to be zero.
        R_zupt: ``(3, 3)`` diagonal covariance of the zero-velocity
            pseudo-measurement in (m/s)².
        window_size: Positive number of trailing IMU increments in each SHOE
            detector window.
        threshold: Nonnegative SHOE GLRT stance-decision threshold.
        sigma_a: Positive accelerometer-rate noise scale in m/s².
        sigma_g: Positive gyroscope-rate noise scale in rad/s.

    Notes:
        Arrays are newly constructed parser outputs but remain mutable NumPy
        arrays. They are configuration inputs, distinct from the filter's
        evolving posterior covariance history.
    """

    minimum_stance_duration: float
    startup_search_limit: float
    initial_accel_bias_b: np.ndarray
    P0: np.ndarray
    Q: np.ndarray
    Q_coast: np.ndarray
    R_zupt: np.ndarray
    window_size: int
    threshold: float
    sigma_a: float
    sigma_g: float


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    """Return one YAML mapping after rejecting missing or wrong-shaped nodes."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration field '{field}' must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Configuration mapping '{field}' must use string keys.")
    return value


def _require_exact_keys(
    mapping: Mapping[str, object],
    expected_keys: tuple[str, ...],
    field: str,
) -> None:
    """Reject missing or misspelled keys at one configuration mapping."""
    actual_keys = set(mapping)
    expected_key_set = set(expected_keys)
    missing_keys = sorted(expected_key_set - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_key_set)
    if missing_keys or unexpected_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing {', '.join(missing_keys)}")
        if unexpected_keys:
            details.append(f"unexpected {', '.join(unexpected_keys)}")
        raise ValueError(
            f"Configuration mapping '{field}' has {'; '.join(details)} keys."
        )


def _require_finite_float(value: object, field: str) -> float:
    """Return a scalar finite configuration value with field-specific errors."""
    if isinstance(value, bool):
        raise ValueError(f"Configuration field '{field}' must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Configuration field '{field}' must be a finite number."
        ) from error
    if not np.isfinite(number):
        raise ValueError(f"Configuration field '{field}' must be a finite number.")
    return number


def _require_positive_float(value: object, field: str) -> float:
    """Return one strictly positive finite configuration value."""
    number = _require_finite_float(value, field)
    if number <= 0.0:
        raise ValueError(f"Configuration field '{field}' must be positive.")
    return number


def _require_nonnegative_float(value: object, field: str) -> float:
    """Return one finite configuration value that may be zero."""
    number = _require_finite_float(value, field)
    if number < 0.0:
        raise ValueError(f"Configuration field '{field}' must be nonnegative.")
    return number


def _require_positive_int(value: object, field: str) -> int:
    """Return one positive integer configuration value, rejecting Booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"Configuration field '{field}' must be a positive integer.")
    return int(value)


def _require_covariance_group(
    value: object,
    field: str,
    *,
    strictly_positive: bool,
) -> np.ndarray:
    """Return one finite three-axis covariance diagonal from YAML sequence data."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"Configuration field '{field}' must be a three-element sequence."
        )
    if len(value) != 3:
        raise ValueError(
            f"Configuration field '{field}' must contain exactly three values."
        )

    diagonal = np.array(
        [_require_finite_float(entry, f"{field}[{index}]") for index, entry in enumerate(value)],
        dtype=float,
    )
    if strictly_positive and np.any(diagonal <= 0.0):
        raise ValueError(f"Configuration field '{field}' must contain positive values.")
    if not strictly_positive and np.any(diagonal < 0.0):
        raise ValueError(
            f"Configuration field '{field}' must contain nonnegative values."
        )
    return diagonal


def _require_finite_vector(value: object, field: str) -> np.ndarray:
    """Return one owned finite three-axis vector from YAML sequence data."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"Configuration field '{field}' must be a three-element sequence."
        )
    if len(value) != 3:
        raise ValueError(
            f"Configuration field '{field}' must contain exactly three values."
        )
    return np.array(
        [_require_finite_float(entry, f"{field}[{index}]") for index, entry in enumerate(value)],
        dtype=float,
    )


def _load_state_covariance(
    filter_config: Mapping[str, object],
    field_name: str,
) -> np.ndarray:
    """Construct one ordered diagonal error-state covariance from YAML groups.

    Args:
        filter_config: Validated top-level ``filter`` mapping.
        field_name: Exact covariance field name within ``filter_config``.

    Returns:
        ``(15, 15)`` diagonal covariance ordered by the five three-axis error
        state groups.

    Raises:
        ValueError: The field is not an exact mapping of three-element finite,
            nonnegative state-group diagonals.

    Notes:
        The groups concatenate as position, velocity, attitude, accelerometer
        bias, and gyro bias. Their physical units follow the corresponding error
        state quantity rather than a universal covariance unit.
    """
    field = f"filter.{field_name}"
    group_config = _require_mapping(filter_config[field_name], field)
    _require_exact_keys(group_config, _STATE_GROUPS, field)
    diagonal = np.concatenate([
        _require_covariance_group(
            group_config[group_name],
            f"{field}.{group_name}",
            strictly_positive=False,
        )
        for group_name in _STATE_GROUPS
    ])
    return np.diag(diagonal)


def _load_coast_process_covariance(filter_config: Mapping[str, object]) -> np.ndarray:
    """Load additional synthetic-coast covariance with zero bias entries.

    Args:
        filter_config: Validated top-level ``filter`` mapping.

    Returns:
        ``(15, 15)`` diagonal covariance used only for synthetic coast
        increments.

    Raises:
        ValueError: No navigation-state coast uncertainty is positive or a bias
            coast entry is nonzero.

    Notes:
        At least one position, velocity, or attitude entry must be positive.
        This verifies the coast-specific schema restriction, not that the
        selected uncertainty magnitude is physically calibrated.
    """
    field_name = "coast_process_covariance_diagonal"
    covariance = _load_state_covariance(filter_config, field_name)
    diagonal = np.diag(covariance)
    if not np.any(diagonal[:9] > 0.0):
        raise ValueError(
            "Configuration field "
            f"'filter.{field_name}' must contain at least one positive position, "
            "velocity, or attitude value."
        )
    if np.any(diagonal[9:] != 0.0):
        raise ValueError(
            "Configuration field "
            f"'filter.{field_name}' must use zero accel_bias and gyro_bias values."
        )
    return covariance


def _parse_config(config: object) -> ZuptINSConfig:
    """Validate one loaded YAML object and construct typed navigation settings.

    Args:
        config: In-memory YAML-decoded object with the exact public profile
            schema.

    Returns:
        Typed INS-plus-ZUPT startup, detector, and covariance settings.

    Raises:
        ValueError: Required mappings or keys are absent, values violate scalar
            or dimensional constraints, or startup stance duration exceeds its
            search limit.

    Notes:
        The parser requires exact ``startup``, ``detector``, and ``filter``
        mappings and constructs diagonal covariance arrays. It does not apply
        the settings or assess their tuning adequacy.
    """
    root = _require_mapping(config, "root")
    _require_exact_keys(root, ("startup", "detector", "filter"), "root")

    startup = _require_mapping(root["startup"], "startup")
    _require_exact_keys(
        startup,
        ("minimum_stance_duration", "startup_search_limit"),
        "startup",
    )
    minimum_stance_duration = _require_positive_float(
        startup["minimum_stance_duration"],
        "startup.minimum_stance_duration",
    )
    startup_search_limit = _require_positive_float(
        startup["startup_search_limit"],
        "startup.startup_search_limit",
    )
    if minimum_stance_duration > startup_search_limit:
        raise ValueError(
            "Configuration field 'startup.minimum_stance_duration' cannot exceed "
            "'startup.startup_search_limit'."
        )

    detector = _require_mapping(root["detector"], "detector")
    _require_exact_keys(
        detector,
        ("window_size", "threshold", "sigma_a", "sigma_g"),
        "detector",
    )
    window_size = _require_positive_int(detector["window_size"], "detector.window_size")
    threshold = _require_nonnegative_float(
        detector["threshold"],
        "detector.threshold",
    )
    sigma_a = _require_positive_float(detector["sigma_a"], "detector.sigma_a")
    sigma_g = _require_positive_float(detector["sigma_g"], "detector.sigma_g")

    filter_config = _require_mapping(root["filter"], "filter")
    _require_exact_keys(
        filter_config,
        (
            "initial_accel_bias_b",
            "initial_covariance_diagonal",
            "process_covariance_diagonal",
            "coast_process_covariance_diagonal",
            "zupt_measurement_covariance_diagonal",
        ),
        "filter",
    )
    initial_accel_bias_b = _require_finite_vector(
        filter_config["initial_accel_bias_b"],
        "filter.initial_accel_bias_b",
    )
    P0 = _load_state_covariance(filter_config, "initial_covariance_diagonal")
    Q = _load_state_covariance(filter_config, "process_covariance_diagonal")
    Q_coast = _load_coast_process_covariance(filter_config)
    R_zupt = np.diag(
        _require_covariance_group(
            filter_config["zupt_measurement_covariance_diagonal"],
            "filter.zupt_measurement_covariance_diagonal",
            strictly_positive=True,
        )
    )

    return ZuptINSConfig(
        minimum_stance_duration=minimum_stance_duration,
        startup_search_limit=startup_search_limit,
        initial_accel_bias_b=initial_accel_bias_b,
        P0=P0,
        Q=Q,
        Q_coast=Q_coast,
        R_zupt=R_zupt,
        window_size=window_size,
        threshold=threshold,
        sigma_a=sigma_a,
        sigma_g=sigma_g,
    )


def load_zupt_ins_config(path: str | Path) -> ZuptINSConfig:
    """Load and validate one INS-plus-ZUPT YAML profile.

    Args:
        path: Filesystem path to a YAML document with the exact public profile
            schema.

    Returns:
        Typed source-agnostic startup, detector, and covariance settings.

    Raises:
        OSError: The profile cannot be opened.
        ValueError: YAML parsing fails or the document violates the required
            mapping, dimensional, or scalar-value contract.

    Notes:
        The loader validates representation and declared constraints only. It
        does not select covariances from sensor data or validate navigation
        performance.
    """
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Could not parse INS plus ZUPT configuration '{config_path}': {error}"
        ) from error

    return _parse_config(config)
