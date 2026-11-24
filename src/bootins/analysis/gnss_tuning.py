"""Build reproducible, bounded GNSS-aiding configuration sweep evidence.

This module validates declarative parameter plans, enumerates candidates,
resolves each candidate against an owned configuration copy, and summarizes
completed replay outcomes. The allowed paths intentionally isolate selected
filter covariance and NIS-gate settings so a candidate records exactly what
varied from its baseline.

The analysis does not read captures, run navigation, write artifacts, establish
an optimal configuration, or turn descriptive terminal metrics into accuracy
or performance rankings.
"""
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from itertools import product
import re
from pathlib import Path

import numpy as np
import yaml

from bootins.estimation.clock_augmented_esekf import NOM_TAU, NOM_V
from bootins.replay.gnss_fusion_models import DelayedGnssUpdateStatus
from bootins.replay.gnss_ins_zupt import GnssZuptReplayResult


_STATE_GROUPS = (
    "position",
    "velocity",
    "attitude",
    "accel_bias",
    "gyro_bias",
    "clock_offset",
)
_COVARIANCE_FIELDS = (
    "initial_covariance_diagonal",
    "process_covariance_diagonal",
)
_ALLOWED_PATHS = frozenset(
    {
        *(
            f"filter.{covariance_field}.{state_group}"
            for covariance_field in _COVARIANCE_FIELDS
            for state_group in _STATE_GROUPS
        ),
        "filter.gnss_measurement_covariance_diagonal",
        "filter.nis_threshold",
    }
)
_CANDIDATE_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


class GnssSweepMode(str, Enum):
    """Choose independent or Cartesian candidate enumeration.

    ``ONE_AT_A_TIME`` varies one declared path relative to the unchanged
    baseline. ``CARTESIAN`` creates every combination of selected values.
    """

    ONE_AT_A_TIME = "one-at-a-time"
    CARTESIAN = "cartesian"


@dataclass(frozen=True)
class GnssSweepParameter:
    """Represent one allowed configuration path and declared candidate values.

    Attributes:
        name: Stable human-readable candidate-name prefix.
        path: Tuple of YAML mapping keys at one allow-listed filter setting.
        values: Caller-declared replacement values in deterministic order.
    """

    name: str
    path: tuple[str, ...]
    values: tuple[object, ...]


@dataclass(frozen=True)
class GnssSweepPlan:
    """Represent a validated declarative enumeration of filter candidates.

    Attributes:
        mode: Independent or Cartesian expansion rule.
        parameters: Distinct allowed paths and candidate values to expand.
    """

    mode: GnssSweepMode
    parameters: tuple[GnssSweepParameter, ...]


@dataclass(frozen=True)
class GnssSweepOverride:
    """Represent one derived value applied at an allow-listed profile path.

    Attributes:
        path: Tuple of YAML keys locating the controlled setting.
        value: Candidate replacement value copied into a resolved profile.
    """

    path: tuple[str, ...]
    value: object


@dataclass(frozen=True)
class GnssSweepCandidate:
    """Represent one named reproducible resolved-profile derivation request.

    Attributes:
        name: Deterministic candidate identity, including ``"baseline"``.
        overrides: Exact controlled settings differing from the base document.
    """

    name: str
    overrides: tuple[GnssSweepOverride, ...]


@dataclass(frozen=True)
class GnssNisStatistics:
    """Represent descriptive NIS distribution facts for one update class.

    Attributes:
        count: Number of terminal updates with an available NIS value.
        minimum: Minimum NIS, or ``None`` when ``count`` is zero.
        median: Median NIS, or ``None`` when ``count`` is zero.
        percentile_95: Linear 95th-percentile NIS, or ``None`` when empty.
        maximum: Maximum NIS, or ``None`` when ``count`` is zero.

    NIS is dimensionless. These statistics describe available accepted or
    NIS-rejected update evidence; they do not rank candidate accuracy.
    """

    count: int
    minimum: float | None
    median: float | None
    percentile_95: float | None
    maximum: float | None


@dataclass(frozen=True)
class GnssSweepSummary:
    """Represent a compact, non-ranking summary of one completed replay.

    Attributes:
        candidate_name: Deterministic identity of the replayed candidate.
        overrides: Controlled base-profile changes for this candidate.
        duration: Final replay time in s.
        final_position_ned: Immutable final position ``(north, east, down)`` in
            m in the replay's static-origin geographic NED frame.
        final_velocity_ned: Immutable final velocity ``(north, east, down)`` in
            m/s in the same NED frame.
        final_clock_offset: Final nominal GNSS-to-IMU clock-offset estimate in s.
        accepted_count: GNSS updates accepted after evaluation.
        rejected_count: GNSS updates evaluated then NIS-rejected.
        stale_count: Updates too old for available IMU history.
        future_dropped_count: Updates whose mapped time was beyond evaluation.
        invalid_fix_count: Receiver-invalid observations excluded before fusion.
        dropped_before_imu_count: Scheduler outcomes dropped before IMU replay.
        dropped_after_imu_count: Scheduler outcomes remaining after IMU replay.
        accepted_nis: Statistics for accepted update NIS values.
        rejected_nis: Statistics for NIS-rejected update values.

    Counts distinguish outcome mechanisms. Final state and NIS summaries are
    descriptive diagnostics, not a position-accuracy comparison or ranking.
    """

    candidate_name: str
    overrides: tuple[GnssSweepOverride, ...]
    duration: float
    final_position_ned: np.ndarray
    final_velocity_ned: np.ndarray
    final_clock_offset: float
    accepted_count: int
    rejected_count: int
    stale_count: int
    future_dropped_count: int
    invalid_fix_count: int
    dropped_before_imu_count: int
    dropped_after_imu_count: int
    accepted_nis: GnssNisStatistics
    rejected_nis: GnssNisStatistics


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    """Return one string-keyed YAML mapping with field-specific refusal."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"GNSS sweep field '{field}' must be a mapping with string keys.")
    return value


def _require_exact_keys(
    mapping: Mapping[str, object],
    expected_keys: tuple[str, ...],
    field: str,
) -> None:
    """Reject missing or unexpected names in one sweep-plan mapping."""
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
            f"GNSS sweep mapping '{field}' has {'; '.join(details)} keys."
        )


def _require_sequence(value: object, field: str) -> Sequence[object]:
    """Return a non-string sequence for one declared YAML collection."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"GNSS sweep field '{field}' must be a sequence.")
    return value


def _parse_parameter(value: object, index: int) -> GnssSweepParameter:
    """Validate one parameter declaration in a loaded sweep plan."""
    field = f"parameters[{index}]"
    mapping = _require_mapping(value, field)
    _require_exact_keys(mapping, ("name", "path", "values"), field)

    name = mapping["name"]
    if not isinstance(name, str) or not _CANDIDATE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"GNSS sweep field '{field}.name' must match "
            "[A-Za-z][A-Za-z0-9_-]*."
        )

    path_string = mapping["path"]
    if not isinstance(path_string, str) or not path_string:
        raise ValueError(f"GNSS sweep field '{field}.path' must be a nonempty string.")
    path = tuple(path_string.split("."))
    if path_string not in _ALLOWED_PATHS:
        raise ValueError(
            f"GNSS sweep field '{field}.path' is not an allowed tuning path: "
            f"{path_string}."
        )

    values = tuple(_require_sequence(mapping["values"], f"{field}.values"))
    if not values:
        raise ValueError(f"GNSS sweep field '{field}.values' must not be empty.")
    return GnssSweepParameter(name=name, path=path, values=values)


def parse_gnss_sweep_plan(document: object) -> GnssSweepPlan:
    """Validate one loaded YAML document into a sweep declaration.

    Args:
        document: YAML-decoded mapping containing expansion mode and parameters.

    Returns:
        Validated mode and distinct allow-listed parameter declarations.

    Raises:
        ValueError: The schema, mode, parameter names, controlled paths, or
            candidate-value collections are invalid.
    """
    root = _require_mapping(document, "root")
    _require_exact_keys(root, ("mode", "parameters"), "root")
    try:
        mode = GnssSweepMode(root["mode"])
    except (TypeError, ValueError) as error:
        allowed_modes = ", ".join(mode.value for mode in GnssSweepMode)
        raise ValueError(
            "GNSS sweep field 'mode' must be one of: " f"{allowed_modes}."
        ) from error

    parameters = tuple(
        _parse_parameter(value, index)
        for index, value in enumerate(_require_sequence(root["parameters"], "parameters"))
    )
    if not parameters:
        raise ValueError("GNSS sweep field 'parameters' must not be empty.")
    names = [parameter.name for parameter in parameters]
    if len(names) != len(set(names)):
        raise ValueError("GNSS sweep parameter names must be unique.")
    paths = [parameter.path for parameter in parameters]
    if len(paths) != len(set(paths)):
        raise ValueError("GNSS sweep parameter paths must be unique.")
    return GnssSweepPlan(mode=mode, parameters=parameters)


def load_gnss_sweep_plan(path: str | Path) -> GnssSweepPlan:
    """Load and validate one declarative GNSS filter-sweep YAML plan.

    Args:
        path: Filesystem path to the sweep-plan YAML document.

    Returns:
        Validated declarative sweep plan.

    Raises:
        OSError: The plan cannot be opened.
        ValueError: YAML parsing or plan validation fails.
    """
    plan_path = Path(path)
    try:
        with plan_path.open(encoding="utf-8") as plan_file:
            document = yaml.safe_load(plan_file)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Could not parse GNSS sweep plan '{plan_path}': {error}"
        ) from error
    return parse_gnss_sweep_plan(document)


def _candidate_count(plan: GnssSweepPlan) -> int:
    """Return baseline plus the exact number of declared derived candidates."""
    value_counts = tuple(len(parameter.values) for parameter in plan.parameters)
    if plan.mode is GnssSweepMode.ONE_AT_A_TIME:
        return 1 + sum(value_counts)
    return 1 + int(np.prod(value_counts, dtype=object))


def generate_gnss_sweep_candidates(
    plan: GnssSweepPlan,
    *,
    max_candidates: int,
) -> tuple[GnssSweepCandidate, ...]:
    """Expand one declarative plan deterministically within its size bound.

    Args:
        plan: Validated expansion mode and allowed parameter values.
        max_candidates: Positive upper bound including the unchanged baseline.

    Returns:
        Baseline first, followed by deterministic independent or Cartesian
        candidate overrides.

    Raises:
        ValueError: The candidate limit is invalid or the expansion exceeds it.

    Notes:
        One-at-a-time candidates change exactly one declared field. Cartesian
        candidates apply one selected value at every declared path. Enumeration
        is reproducible but does not evaluate or rank a candidate.
    """
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise ValueError("max_candidates must be a positive integer.")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer.")
    candidate_count = _candidate_count(plan)
    if candidate_count > max_candidates:
        raise ValueError(
            f"GNSS sweep expands to {candidate_count} candidates, exceeding "
            f"the maximum of {max_candidates}."
        )

    candidates = [GnssSweepCandidate(name="baseline", overrides=())]
    if plan.mode is GnssSweepMode.ONE_AT_A_TIME:
        for parameter in plan.parameters:
            for value_index, value in enumerate(parameter.values, start=1):
                candidates.append(
                    GnssSweepCandidate(
                        name=f"{parameter.name}-{value_index}",
                        overrides=(GnssSweepOverride(parameter.path, value),),
                    )
                )
    else:
        value_indices = product(
            *(range(len(parameter.values)) for parameter in plan.parameters)
        )
        for indices in value_indices:
            overrides = tuple(
                GnssSweepOverride(parameter.path, parameter.values[value_index])
                for parameter, value_index in zip(plan.parameters, indices, strict=True)
            )
            name = "__".join(
                f"{parameter.name}-{value_index + 1}"
                for parameter, value_index in zip(plan.parameters, indices, strict=True)
            )
            candidates.append(GnssSweepCandidate(name=name, overrides=overrides))
    return tuple(candidates)


def resolve_gnss_sweep_candidate(
    base_document: object,
    candidate: GnssSweepCandidate,
) -> dict[str, object]:
    """Resolve one candidate into an owned deep copy of a base YAML document.

    Args:
        base_document: Parsed base configuration mapping.
        candidate: Deterministic allow-listed overrides to apply.

    Returns:
        Independent full configuration mapping with the candidate values copied
        into their declared paths.

    Raises:
        ValueError: The base document is not a mapping or lacks a controlled
            candidate path.

    Notes:
        Resolution preserves all uncontrolled base settings and does not itself
        validate the resulting navigation configuration or run a replay.
    """
    root = _require_mapping(base_document, "base configuration")
    resolved = deepcopy(dict(root))
    for override in candidate.overrides:
        target: MutableMapping[str, object] = resolved
        for key in override.path[:-1]:
            try:
                child = target[key]
            except KeyError as error:
                path = ".".join(override.path)
                raise ValueError(
                    f"Base configuration is missing sweep path '{path}'."
                ) from error
            if not isinstance(child, MutableMapping):
                path = ".".join(override.path)
                raise ValueError(
                    f"Base configuration path '{path}' does not lead to a mapping."
                )
            target = child
        final_key = override.path[-1]
        if final_key not in target:
            path = ".".join(override.path)
            raise ValueError(f"Base configuration is missing sweep path '{path}'.")
        target[final_key] = deepcopy(override.value)
    return resolved


def _nis_statistics(values: list[float], field: str) -> GnssNisStatistics:
    """Return descriptive statistics after requiring complete finite NIS evidence."""
    if not values:
        return GnssNisStatistics(0, None, None, None, None)
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite NIS values.")
    return GnssNisStatistics(
        count=array.size,
        minimum=float(np.min(array)),
        median=float(np.median(array)),
        percentile_95=float(np.percentile(array, 95.0)),
        maximum=float(np.max(array)),
    )


def summarize_gnss_sweep_candidate(
    candidate: GnssSweepCandidate,
    replay_result: GnssZuptReplayResult,
) -> GnssSweepSummary:
    """Summarize one completed replay without assigning an accuracy ranking.

    Args:
        candidate: Identity and controlled overrides used for the replay.
        replay_result: Completed GNSS-aided replay result and aligned diagnostics.

    Returns:
        Outcome counts, available-NIS statistics, and final nominal NED state.

    Raises:
        ValueError: Replay diagnostics are misaligned, required NIS evidence is
            absent, or final time and nominal history are unusable.

    Notes:
        Only accepted and NIS-rejected updates have evaluated NIS values. Stale
        and future-dropped outcomes remain distinct counts rather than zero-NIS
        observations. This is a terminal descriptive summary, not a trajectory
        accuracy evaluation.
    """
    updates = tuple(replay_result.gnss_updates)
    diagnostics = tuple(replay_result.gnss_fusion_diagnostics)
    if len(updates) != len(diagnostics) or any(
        update is not diagnostic.update
        for update, diagnostic in zip(updates, diagnostics, strict=True)
    ):
        raise ValueError("GNSS replay outcomes and diagnostics must remain aligned.")

    accepted_nis: list[float] = []
    rejected_nis: list[float] = []
    counts = {status: 0 for status in DelayedGnssUpdateStatus}
    for update in updates:
        counts[update.status] += 1
        if update.status is DelayedGnssUpdateStatus.ACCEPTED:
            if update.nis is None:
                raise ValueError("Accepted GNSS update is missing NIS evidence.")
            accepted_nis.append(float(update.nis))
        elif update.status is DelayedGnssUpdateStatus.REJECTED:
            if update.nis is None:
                raise ValueError("Rejected GNSS update is missing NIS evidence.")
            rejected_nis.append(float(update.nis))

    time = np.asarray(replay_result.time, dtype=float)
    nominals = np.asarray(replay_result.nominals, dtype=float)
    if time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time)):
        raise ValueError("GNSS replay time must be a nonempty finite one-dimensional array.")
    if nominals.ndim != 2 or nominals.shape[0] != time.size or nominals.shape[1] <= NOM_TAU:
        raise ValueError("GNSS replay nominals do not align with replay time.")
    if not np.all(np.isfinite(nominals)):
        raise ValueError("GNSS replay nominals must be finite.")

    if (
        not isinstance(candidate.name, str)
        or not _CANDIDATE_NAME_PATTERN.fullmatch(candidate.name)
    ):
        raise ValueError("candidate_name must match [A-Za-z][A-Za-z0-9_-]*.")
    duration = float(time[-1])
    if duration < 0.0:
        raise ValueError("GNSS replay duration must be nonnegative.")

    final_position_ned = np.array(nominals[-1, :3], dtype=float, copy=True)
    final_velocity_ned = np.array(nominals[-1, NOM_V], dtype=float, copy=True)
    final_position_ned.setflags(write=False)
    final_velocity_ned.setflags(write=False)

    accepted_count = counts[DelayedGnssUpdateStatus.ACCEPTED]
    rejected_count = counts[DelayedGnssUpdateStatus.REJECTED]
    accepted_nis_statistics = _nis_statistics(accepted_nis, "accepted GNSS updates")
    rejected_nis_statistics = _nis_statistics(rejected_nis, "rejected GNSS updates")
    if accepted_nis_statistics.count != accepted_count:
        raise ValueError("accepted NIS count must match accepted_count.")
    if rejected_nis_statistics.count != rejected_count:
        raise ValueError("rejected NIS count must match rejected_count.")

    return GnssSweepSummary(
        candidate_name=candidate.name,
        overrides=candidate.overrides,
        duration=duration,
        final_position_ned=final_position_ned,
        final_velocity_ned=final_velocity_ned,
        final_clock_offset=float(nominals[-1, NOM_TAU]),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        stale_count=counts[DelayedGnssUpdateStatus.STALE],
        future_dropped_count=counts[DelayedGnssUpdateStatus.FUTURE_DROPPED],
        invalid_fix_count=int(replay_result.invalid_gnss_fix_count),
        dropped_before_imu_count=int(
            replay_result.summary.dropped_before_imu_count
        ),
        dropped_after_imu_count=int(replay_result.summary.dropped_after_imu_count),
        accepted_nis=accepted_nis_statistics,
        rejected_nis=rejected_nis_statistics,
    )
