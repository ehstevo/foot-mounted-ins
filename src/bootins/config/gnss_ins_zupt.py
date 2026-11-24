"""Load the strict YAML profile for clock-aware GNSS, INS, and ZUPT replay.

This module turns one source-agnostic profile into GNSS-aided startup, SHOE,
and clock-augmented ESEKF settings. It constructs *discrete per-increment*
covariance matrices; it neither converts continuous-time noise densities nor
establishes that a configuration is scientifically calibrated. Capture source,
receiver validity, GNSS outage selection, logging, and evolving posterior
covariances belong to sensor and replay layers rather than this schema.

The parser enforces exact YAML keys, finite entries, dimensions, scalar domains,
and stated ordering constraints. It does not assess GNSS accuracy, measurement
time alignment, trajectory quality, or filter tuning.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import yaml


_VECTOR_STATE_GROUPS = (
    "position",
    "velocity",
    "attitude",
    "accel_bias",
    "gyro_bias",
)
_CLOCK_STATE_GROUP = "clock_offset"
_STATE_GROUPS = (*_VECTOR_STATE_GROUPS, _CLOCK_STATE_GROUP)


@dataclass(frozen=True)
class GnssZuptConfig:
    """Store typed configuration for one clock-aware GNSS/INS/ZUPT replay.

    Attributes:
        initial_stance_duration: Required initial diagnostic stance duration in
            s.
        initial_stance_search_limit: Maximum search duration for the initial
            stance in s.
        terminal_stance_duration: Required authoritative terminal stance
            duration in s.
        terminal_stance_search_limit: Maximum terminal-stance search duration
            in s.
        course_minimum_fix_count: Positive count of receiver-valid fixes needed
            before GNSS course qualification.
        course_minimum_baseline: Minimum fitted along-course baseline in m.
        course_maximum_cross_track_rms: Maximum fitted course cross-track RMS in
            m.
        initial_accel_bias_b: ``(3,)`` initial nominal accelerometer bias in
            the body frame, in m/s².
        P0: ``(16, 16)`` diagonal initial error-state covariance. Ordered groups
            are position (m²), velocity ((m/s)²), attitude (rad²), accelerometer
            bias ((m/s²)²), gyro bias ((rad/s)²), and clock offset (s²).
        Q: ``(16, 16)`` diagonal covariance added once per real IMU increment in
            the same order and units as ``P0``. It is discrete configuration
            input, not a continuous-time noise density or a ``dt`` scaling rule.
        Q_coast: ``(16, 16)`` additional covariance for a synthetic coast
            increment. Only position, velocity, and attitude groups may be
            nonzero; bias and clock-offset entries are required to be zero.
        R_zupt: ``(3, 3)`` diagonal zero-velocity pseudo-measurement covariance
            in (m/s)².
        R_gnss: ``(3, 3)`` diagonal local-NED GNSS position covariance in m².
        nis_threshold: Positive scalar gate for GNSS normalized innovation
            squared evaluations.
        window_size: Positive number of trailing IMU increments in each SHOE
            detector window.
        threshold: Nonnegative SHOE GLRT stance-decision threshold.
        sigma_a: Positive accelerometer-rate noise scale in m/s².
        sigma_g: Positive gyroscope-rate noise scale in rad/s.

    Notes:
        Receiver-reported validity and any run-level GNSS outage are replay
        evidence or controls, not fields in this profile. Parser-created arrays
        are mutable configuration inputs, not views of filter history.
    """

    initial_stance_duration: float
    initial_stance_search_limit: float
    terminal_stance_duration: float
    terminal_stance_search_limit: float
    course_minimum_fix_count: int
    course_minimum_baseline: float
    course_maximum_cross_track_rms: float
    initial_accel_bias_b: np.ndarray
    P0: np.ndarray
    Q: np.ndarray
    Q_coast: np.ndarray
    R_zupt: np.ndarray
    R_gnss: np.ndarray
    nis_threshold: float
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
    """Return one scalar finite value with field-specific errors."""
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
    """Return one strictly positive finite value."""
    number = _require_finite_float(value, field)
    if number <= 0.0:
        raise ValueError(f"Configuration field '{field}' must be positive.")
    return number


def _require_nonnegative_float(value: object, field: str) -> float:
    """Return one finite value that may be zero."""
    number = _require_finite_float(value, field)
    if number < 0.0:
        raise ValueError(f"Configuration field '{field}' must be nonnegative.")
    return number


def _require_positive_int(value: object, field: str) -> int:
    """Return one positive integer, rejecting Boolean values."""
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
        [
            _require_finite_float(entry, f"{field}[{index}]")
            for index, entry in enumerate(value)
        ],
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
    """Construct ordered clock-aware diagonal covariance from YAML groups.

    Args:
        filter_config: Validated top-level ``filter`` mapping.
        field_name: Exact covariance field name within ``filter_config``.

    Returns:
        ``(16, 16)`` diagonal covariance ordered by five three-axis state groups
        followed by clock offset.

    Raises:
        ValueError: The field is not an exact mapping of three-element finite,
            nonnegative vector groups and one nonnegative clock scalar.

    Notes:
        The parser concatenates position, velocity, attitude, accelerometer
        bias, gyro bias, and clock offset. Units follow the corresponding error
        state quantity.
    """
    field = f"filter.{field_name}"
    group_config = _require_mapping(filter_config[field_name], field)
    _require_exact_keys(group_config, _STATE_GROUPS, field)
    vector_entries = [
        _require_covariance_group(
            group_config[group_name],
            f"{field}.{group_name}",
            strictly_positive=False,
        )
        for group_name in _VECTOR_STATE_GROUPS
    ]
    clock_entry = _require_nonnegative_float(
        group_config[_CLOCK_STATE_GROUP],
        f"{field}.{_CLOCK_STATE_GROUP}",
    )
    diagonal = np.concatenate([*vector_entries, [clock_entry]])
    return np.diag(diagonal)


def _load_coast_process_covariance(filter_config: Mapping[str, object]) -> np.ndarray:
    """Load synthetic-coast covariance with zero bias and clock entries.

    Args:
        filter_config: Validated top-level ``filter`` mapping.

    Returns:
        ``(16, 16)`` diagonal covariance used only for synthetic coast
        increments.

    Raises:
        ValueError: No navigation-state coast uncertainty is positive or a bias
            or clock coast entry is nonzero.

    Notes:
        At least one position, velocity, or attitude diagonal entry must be
        positive. This enforces the documented schema restriction without
        claiming a physical uncertainty calibration.
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
            f"'filter.{field_name}' must use zero accel_bias, gyro_bias, and "
            "clock_offset values."
        )
    return covariance


def parse_gnss_zupt_config(config: object) -> GnssZuptConfig:
    """Validate one loaded document into typed aided-navigation configuration.

    Args:
        config: In-memory YAML-decoded document with the exact public profile
            schema.

    Returns:
        Typed GNSS-aided startup, detector, and covariance settings.

    Raises:
        ValueError: Required mappings or keys are absent, values violate scalar
            or dimensional constraints, or a stance duration exceeds its search
            limit.

    Notes:
        Separating parsing from file I/O lets callers validate an in-memory
        resolved candidate with the same contract as a YAML-loaded profile.
        The parser does not make receiver-validity, outage, timing-alignment, or
        tuning decisions.
    """
    root = _require_mapping(config, "root")
    _require_exact_keys(root, ("startup", "detector", "filter"), "root")

    startup = _require_mapping(root["startup"], "startup")
    _require_exact_keys(
        startup,
        (
            "initial_stance_duration",
            "initial_stance_search_limit",
            "terminal_stance_duration",
            "terminal_stance_search_limit",
            "course_minimum_fix_count",
            "course_minimum_baseline",
            "course_maximum_cross_track_rms",
        ),
        "startup",
    )
    initial_stance_duration = _require_positive_float(
        startup["initial_stance_duration"],
        "startup.initial_stance_duration",
    )
    initial_stance_search_limit = _require_positive_float(
        startup["initial_stance_search_limit"],
        "startup.initial_stance_search_limit",
    )
    terminal_stance_duration = _require_positive_float(
        startup["terminal_stance_duration"],
        "startup.terminal_stance_duration",
    )
    terminal_stance_search_limit = _require_positive_float(
        startup["terminal_stance_search_limit"],
        "startup.terminal_stance_search_limit",
    )
    if initial_stance_duration > initial_stance_search_limit:
        raise ValueError(
            "Configuration field 'startup.initial_stance_duration' cannot exceed "
            "'startup.initial_stance_search_limit'."
        )
    if terminal_stance_duration > terminal_stance_search_limit:
        raise ValueError(
            "Configuration field 'startup.terminal_stance_duration' cannot exceed "
            "'startup.terminal_stance_search_limit'."
        )
    course_minimum_fix_count = _require_positive_int(
        startup["course_minimum_fix_count"],
        "startup.course_minimum_fix_count",
    )
    course_minimum_baseline = _require_positive_float(
        startup["course_minimum_baseline"],
        "startup.course_minimum_baseline",
    )
    course_maximum_cross_track_rms = _require_nonnegative_float(
        startup["course_maximum_cross_track_rms"],
        "startup.course_maximum_cross_track_rms",
    )

    detector = _require_mapping(root["detector"], "detector")
    _require_exact_keys(
        detector,
        ("window_size", "threshold", "sigma_a", "sigma_g"),
        "detector",
    )
    window_size = _require_positive_int(detector["window_size"], "detector.window_size")
    threshold = _require_nonnegative_float(detector["threshold"], "detector.threshold")
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
            "gnss_measurement_covariance_diagonal",
            "nis_threshold",
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
    R_gnss = np.diag(
        _require_covariance_group(
            filter_config["gnss_measurement_covariance_diagonal"],
            "filter.gnss_measurement_covariance_diagonal",
            strictly_positive=True,
        )
    )
    nis_threshold = _require_positive_float(
        filter_config["nis_threshold"],
        "filter.nis_threshold",
    )

    return GnssZuptConfig(
        initial_stance_duration=initial_stance_duration,
        initial_stance_search_limit=initial_stance_search_limit,
        terminal_stance_duration=terminal_stance_duration,
        terminal_stance_search_limit=terminal_stance_search_limit,
        course_minimum_fix_count=course_minimum_fix_count,
        course_minimum_baseline=course_minimum_baseline,
        course_maximum_cross_track_rms=course_maximum_cross_track_rms,
        initial_accel_bias_b=initial_accel_bias_b,
        P0=P0,
        Q=Q,
        Q_coast=Q_coast,
        R_zupt=R_zupt,
        R_gnss=R_gnss,
        nis_threshold=nis_threshold,
        window_size=window_size,
        threshold=threshold,
        sigma_a=sigma_a,
        sigma_g=sigma_g,
    )


def load_gnss_zupt_config_document(path: str | Path) -> object:
    """Load one GNSS-aided YAML document without interpreting its settings.

    Args:
        path: Filesystem path to a YAML document.

    Returns:
        The YAML-decoded object without schema validation or type conversion.

    Raises:
        OSError: The document cannot be opened.
        ValueError: YAML parsing fails.

    Notes:
        This boundary supports callers that resolve a document before passing it
        to :func:`parse_gnss_zupt_config`. Callers needing typed settings should
        use :func:`load_gnss_zupt_config` instead.
    """
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Could not parse GNSS, INS, and ZUPT configuration "
            f"'{config_path}': {error}"
        ) from error


def load_gnss_zupt_config(path: str | Path) -> GnssZuptConfig:
    """Load and validate one GNSS-aided YAML profile.

    Args:
        path: Filesystem path to a YAML document with the exact public schema.

    Returns:
        Typed source-agnostic GNSS-aided navigation configuration.

    Raises:
        OSError: The profile cannot be opened.
        ValueError: YAML parsing fails or the document violates the required
            mapping, dimensional, or scalar-value contract.
    """
    return parse_gnss_zupt_config(load_gnss_zupt_config_document(path))
