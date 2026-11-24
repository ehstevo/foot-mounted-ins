"""Serialize computed navigation histories as portable CSV evidence artifacts.

This module receives already-computed nominal states, error covariances, and
optional ZUPT-row flags. It writes one initialized row-zero state followed by
posterior rows, with a YAML sidecar declaring the locked state layout,
coordinate frame, covariance encoding, and row alignment. It does not read
captures, run an estimator, infer missing states, or make navigation decisions.
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import yaml


class NavigationHistoryLayout(Enum):
    """Select one locked nominal/error-state export convention.

    ``LOCAL_FRD`` represents run-local Forward-Right-Down navigation.
    ``GEOGRAPHIC_NED_CLOCK`` represents a static-origin geographic
    North-East-Down state with the clock-offset component. The sidecar records
    the exact names and ordering for the chosen layout.
    """

    LOCAL_FRD = "local_frd"
    GEOGRAPHIC_NED_CLOCK = "geographic_ned_clock"


@dataclass(frozen=True)
class _HistorySchema:
    """One private fixed CSV/sidecar convention selected by public layout."""

    schema_version: int
    nav_frame: str
    nominal_column_names: tuple[str, ...]
    error_state_names: tuple[str, ...]

    @property
    def nominal_state_dimension(self) -> int:
        """Return the nominal-state dimension defined by this schema."""
        return len(self.nominal_column_names)

    @property
    def error_state_dimension(self) -> int:
        """Return the error-state dimension defined by this schema."""
        return len(self.error_state_names)


_LOCAL_FRD_SCHEMA = _HistorySchema(
    schema_version=1,
    nav_frame="run-local Forward-Right-Down",
    nominal_column_names=(
        "p_forward_m",
        "p_right_m",
        "p_down_m",
        "v_forward_mps",
        "v_right_mps",
        "v_down_mps",
        "q_nb_w",
        "q_nb_x",
        "q_nb_y",
        "q_nb_z",
        "b_a_b_x_mps2",
        "b_a_b_y_mps2",
        "b_a_b_z_mps2",
        "b_g_b_x_radps",
        "b_g_b_y_radps",
        "b_g_b_z_radps",
    ),
    error_state_names=(
        "dp_forward",
        "dp_right",
        "dp_down",
        "dv_forward",
        "dv_right",
        "dv_down",
        "dpsi_forward",
        "dpsi_right",
        "dpsi_down",
        "dba_b_x",
        "dba_b_y",
        "dba_b_z",
        "dbg_b_x",
        "dbg_b_y",
        "dbg_b_z",
    ),
)

_GEOGRAPHIC_NED_CLOCK_SCHEMA = _HistorySchema(
    schema_version=3,
    nav_frame="static-origin local geographic North-East-Down",
    nominal_column_names=(
        "p_north_m",
        "p_east_m",
        "p_down_m",
        "v_north_mps",
        "v_east_mps",
        "v_down_mps",
        "q_nb_w",
        "q_nb_x",
        "q_nb_y",
        "q_nb_z",
        "b_a_b_x_mps2",
        "b_a_b_y_mps2",
        "b_a_b_z_mps2",
        "b_g_b_x_radps",
        "b_g_b_y_radps",
        "b_g_b_z_radps",
        "tau_s",
    ),
    error_state_names=(
        "dp_north",
        "dp_east",
        "dp_down",
        "dv_north",
        "dv_east",
        "dv_down",
        "dpsi_north",
        "dpsi_east",
        "dpsi_down",
        "dba_b_x",
        "dba_b_y",
        "dba_b_z",
        "dbg_b_x",
        "dbg_b_y",
        "dbg_b_z",
        "dtau",
    ),
)

_SCHEMAS = {
    NavigationHistoryLayout.LOCAL_FRD: _LOCAL_FRD_SCHEMA,
    NavigationHistoryLayout.GEOGRAPHIC_NED_CLOCK: _GEOGRAPHIC_NED_CLOCK_SCHEMA,
}

_RAW_INS_TRAJECTORY_COLUMNS = (
    "time_s",
    "p_forward_m",
    "p_right_m",
    "p_down_m",
    "v_forward_mps",
    "v_right_mps",
    "v_down_mps",
    "q_fb_w",
    "q_fb_x",
    "q_fb_y",
    "q_fb_z",
)


def _schema_for_layout(layout: NavigationHistoryLayout) -> _HistorySchema:
    """Return one locked schema or reject an unsupported export convention."""
    if not isinstance(layout, NavigationHistoryLayout):
        raise ValueError("layout must be a NavigationHistoryLayout value.")
    return _SCHEMAS[layout]


def _covariance_column_names(schema: _HistorySchema) -> tuple[str, ...]:
    """Return lower-triangle covariance headers for one selected schema."""
    lower_rows, lower_cols = np.tril_indices(schema.error_state_dimension)
    return tuple(
        f"P_{schema.error_state_names[row]}_{schema.error_state_names[col]}"
        for row, col in zip(lower_rows, lower_cols, strict=True)
    )


def _csv_column_names(schema: _HistorySchema) -> tuple[str, ...]:
    """Return the complete fixed CSV header for one selected schema."""
    return (
        "time_s",
        *schema.nominal_column_names,
        "zupt_applied",
        *_covariance_column_names(schema),
    )


def _as_finite_array(values: np.ndarray, name: str) -> np.ndarray:
    """Return ``values`` as finite float data or raise a clear schema error."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to a float array.") from error

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validated_zupt_flags(
    zupt_applied: np.ndarray | None,
    sample_count: int,
) -> np.ndarray:
    """Return row-aligned ZUPT flags, defaulting honestly to no updates."""
    if zupt_applied is None:
        return np.zeros(sample_count, dtype=bool)

    flags = np.asarray(zupt_applied)
    if flags.shape != (sample_count,):
        raise ValueError(
            "zupt_applied must have one value for every exported state row."
        )

    if flags.dtype == bool:
        validated_flags = flags
    elif np.issubdtype(flags.dtype, np.integer) and np.all(
        (flags == 0) | (flags == 1)
    ):
        validated_flags = flags.astype(bool)
    else:
        raise ValueError("zupt_applied must contain only boolean or 0/1 values.")

    if validated_flags[0]:
        raise ValueError("zupt_applied must be false for the initialized row zero.")
    return validated_flags


def _validate_history(
    time: np.ndarray,
    nominals: np.ndarray,
    covariances: np.ndarray,
    zupt_applied: np.ndarray | None,
    schema: _HistorySchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize one selected-layout ESEKF history."""
    validated_time = _as_finite_array(time, "time")
    validated_nominals = _as_finite_array(nominals, "nominals")
    validated_covariances = _as_finite_array(covariances, "covariances")

    if validated_time.ndim != 1 or validated_time.size == 0:
        raise ValueError("time must be a nonempty one-dimensional array.")
    if np.any(np.diff(validated_time) <= 0.0):
        raise ValueError("time must be strictly increasing.")

    sample_count = validated_time.size
    if validated_nominals.shape != (sample_count, schema.nominal_state_dimension):
        raise ValueError(
            "nominals must have shape "
            f"({sample_count}, {schema.nominal_state_dimension})."
        )
    if validated_covariances.shape != (
        sample_count,
        schema.error_state_dimension,
        schema.error_state_dimension,
    ):
        raise ValueError(
            "covariances must have shape "
            f"({sample_count}, {schema.error_state_dimension}, "
            f"{schema.error_state_dimension})."
        )
    if not np.allclose(
        validated_covariances,
        np.swapaxes(validated_covariances, 1, 2),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("covariances must be symmetric.")

    validated_flags = _validated_zupt_flags(zupt_applied, sample_count)
    return (
        validated_time,
        validated_nominals,
        validated_covariances,
        validated_flags,
    )


def _write_schema_sidecar(
    output_path: Path,
    layout: NavigationHistoryLayout,
    schema: _HistorySchema,
) -> None:
    """Write selected schema facts beside one ESEKF-history CSV artifact."""
    sidecar_path = output_path.with_suffix(".meta.yaml")
    sidecar = {
        "schema_version": schema.schema_version,
        "artifact_type": "esekf_navigation_history",
        "csv_filename": output_path.name,
        "nav_frame": schema.nav_frame,
        "nominal_state_columns": list(schema.nominal_column_names),
        "error_state_order": list(schema.error_state_names),
        "covariance_storage": "lower_triangle_row_major",
        "covariance_dimension": schema.error_state_dimension,
        "row_zero": "initialized prior at replay time zero",
        "later_rows": "posterior after the corresponding IMU increment",
        "zupt_applied_alignment": (
            "row zero is false; later rows align with posterior epochs"
        ),
    }
    if layout is not NavigationHistoryLayout.LOCAL_FRD:
        sidecar["state_layout"] = layout.value

    with sidecar_path.open("w", encoding="utf-8") as sidecar_file:
        yaml.safe_dump(sidecar, sidecar_file, sort_keys=False)


def write_esekf_history_csv(
    output_path: str | Path,
    time: np.ndarray,
    nominals: np.ndarray,
    covariances: np.ndarray,
    *,
    zupt_applied: np.ndarray | None = None,
    layout: NavigationHistoryLayout = NavigationHistoryLayout.LOCAL_FRD,
) -> None:
    """Write one complete nominal/error-covariance history and YAML sidecar.

    Args:
        output_path: Destination CSV path, overwritten with this artifact.
        time: Strictly increasing replay times in s with initialized row zero.
        nominals: Finite nominal-state rows matching the selected layout.
        covariances: Finite symmetric error-state covariance matrices aligned
            one-for-one with ``time``.
        zupt_applied: Optional Boolean rows aligned with posterior epochs.
        layout: Locked local-FRD or geographic-NED-clock schema.

    Raises:
        ValueError: Histories are nonfinite, misaligned, wrong-shaped, or have
            nonsymmetric covariance; layout or ZUPT flags are invalid.
        OSError: The CSV or sidecar cannot be written.

    Notes:
        The default stores a 16-state nominal / 15-state error local-FRD
        history; geographic clock-aware replay uses 17 / 16 states. Only the
        lower triangle of each symmetric covariance is serialized. Omitting
        ZUPT flags writes false for every row; it does not infer their history.
    """
    output_path = Path(output_path)
    schema = _schema_for_layout(layout)
    (
        validated_time,
        validated_nominals,
        validated_covariances,
        validated_flags,
    ) = _validate_history(
        time,
        nominals,
        covariances,
        zupt_applied,
        schema,
    )

    lower_rows, lower_cols = np.tril_indices(schema.error_state_dimension)
    covariance_lower = validated_covariances[:, lower_rows, lower_cols]
    table = np.column_stack(
        (
            validated_time,
            validated_nominals,
            validated_flags.astype(float),
            covariance_lower,
        )
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=",".join(_csv_column_names(schema)),
        comments="",
        fmt="%.17g",
    )
    _write_schema_sidecar(output_path, layout, schema)


def write_raw_ins_trajectory_csv(
    output_path: str | Path,
    time: np.ndarray,
    positions_frd: np.ndarray,
    velocities_frd: np.ndarray,
    quaternions_fb: np.ndarray,
) -> None:
    """Write a raw-INS nominal trajectory and its distinct YAML sidecar.

    Args:
        output_path: Destination CSV path, overwritten with this artifact.
        time: Strictly increasing local-replay times in s, including row zero.
        positions_frd: ``(N, 3)`` Forward-Right-Down positions in m.
        velocities_frd: ``(N, 3)`` Forward-Right-Down velocities in m/s.
        quaternions_fb: ``(N, 4)`` body-to-run-local-frame attitude quaternions.

    Raises:
        ValueError: Inputs are nonfinite, misaligned, wrongly shaped, or time
            does not increase strictly.
        OSError: The CSV or sidecar cannot be written.

    Notes:
        Raw replay has no ESEKF covariance or bias state, so this artifact is
        deliberately distinct from :func:`write_esekf_history_csv`. Row zero is
        the initialized prior; later rows are mechanized IMU posteriors.
    """
    validated_time = _as_finite_array(time, "time")
    validated_positions = _as_finite_array(positions_frd, "positions_frd")
    validated_velocities = _as_finite_array(velocities_frd, "velocities_frd")
    validated_quaternions = _as_finite_array(quaternions_fb, "quaternions_fb")

    if validated_time.ndim != 1 or validated_time.size == 0:
        raise ValueError("time must be a nonempty one-dimensional array.")
    if np.any(np.diff(validated_time) <= 0.0):
        raise ValueError("time must be strictly increasing.")

    sample_count = validated_time.size
    for values, name, dimension in (
        (validated_positions, "positions_frd", 3),
        (validated_velocities, "velocities_frd", 3),
        (validated_quaternions, "quaternions_fb", 4),
    ):
        if values.shape != (sample_count, dimension):
            raise ValueError(
                f"{name} must have shape ({sample_count}, {dimension})."
            )

    output_path = Path(output_path)
    table = np.column_stack(
        (
            validated_time,
            validated_positions,
            validated_velocities,
            validated_quaternions,
        )
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=",".join(_RAW_INS_TRAJECTORY_COLUMNS),
        comments="",
        fmt="%.17g",
    )
    sidecar = {
        "schema_version": 1,
        "artifact_type": "raw_ins_trajectory",
        "csv_filename": output_path.name,
        "nav_frame": "run-local Forward-Right-Down",
        "state_columns": list(_RAW_INS_TRAJECTORY_COLUMNS[1:]),
        "row_zero": "initialized prior at replay time zero",
        "later_rows": "mechanized posterior after the corresponding IMU increment",
    }
    with output_path.with_suffix(".meta.yaml").open("w", encoding="utf-8") as sidecar_file:
        yaml.safe_dump(sidecar, sidecar_file, sort_keys=False)


def write_geographic_ned_origin_sidecar(
    navigation_csv_path: str | Path,
    latitude: float,
    longitude: float,
    height_ellipsoid: float,
) -> None:
    """Write the fixed LLH origin sidecar for one geographic-NED CSV.

    Args:
        navigation_csv_path: Existing or planned aided-navigation CSV path.
        latitude: WGS-84 geodetic origin latitude (rad).
        longitude: WGS-84 geodetic origin longitude (rad).
        height_ellipsoid: WGS-84 ellipsoidal origin height (m).

    Raises:
        ValueError: The origin is nonfinite, not three-component, or has invalid
            radian latitude or longitude bounds.
        OSError: The adjacent geographic-reference sidecar cannot be written.

    Notes:
        The sidecar records the terminal-static ECEF-averaged origin already
        used by aided replay. It enables geographic presentation but does not
        change the navigation history or establish receiver antenna position as
        foot-position truth.
    """
    origin = _as_finite_array(
        (latitude, longitude, height_ellipsoid),
        "geographic NED origin",
    )
    if origin.shape != (3,):
        raise ValueError("geographic NED origin must contain three values.")
    if abs(origin[0]) > np.pi / 2.0:
        raise ValueError("geographic NED origin latitude must be in radians.")
    if abs(origin[1]) > np.pi:
        raise ValueError("geographic NED origin longitude must be in radians.")

    navigation_csv_path = Path(navigation_csv_path)
    sidecar = {
        "schema_version": 1,
        "artifact_type": "geographic_ned_origin",
        "csv_filename": navigation_csv_path.name,
        "nav_frame": "static-origin local geographic North-East-Down",
        "origin_source": "terminal static receiver-valid GNSS fixes, ECEF average",
        "latitude_rad": float(origin[0]),
        "longitude_rad": float(origin[1]),
        "height_ellipsoid_m": float(origin[2]),
    }
    with navigation_csv_path.with_suffix(".georef.yaml").open(
        "w",
        encoding="utf-8",
    ) as sidecar_file:
        yaml.safe_dump(sidecar, sidecar_file, sort_keys=False)
