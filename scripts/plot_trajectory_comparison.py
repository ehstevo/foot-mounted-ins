"""Write an auditable interactive map from exported navigation artifacts.

The command is presentation-only postprocessing: it reads exported navigation
state CSVs, normalized receiver fix tables, and an RTKLIB geodetic ``.pos``
solution without running, tuning, or modifying a replay. A schema-versioned
manifest declares all map traces, colors, visibility, and bounds behavior.

Geographic-NED state exports are converted through their matching LLH-origin
sidecar. Run-local Forward-Right-Down (FRD) exports require a manifest-declared
presentation registration: exactly one PPK epoch or receiver-valid GNSS-table
epoch anchors row zero, and a clockwise-from-North heading rotates Forward/Right
into North/East before LLH conversion. That registration changes only displayed
coordinates; it does not alter navigation estimates, establish IMU/GNSS timing,
or demonstrate trajectory accuracy.

The resulting HTML map layers navigation estimates, receiver observations, and
selected PPK quality rows for visual comparison. PPK and receiver layers remain
evidence rather than rigid foot truth.

Example:
    python scripts/plot_trajectory_comparison.py \\
        config/trajectory_comparison/example.yaml \\
        --html-output plots/trajectory_comparison.html
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, pi, radians
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

from bootins.reporting.interactive_maps import (
    FoliumBasemapStyle,
    LlhMapTrace,
    write_llh_trajectory_comparison_map,
)
from bootins.frames.geodesy import ned_to_llh
from bootins.sensors.gnss_observations import GnssFixLLH
from bootins.sensors.gnss_table import load_gnss_fix_table


_GPS_EPOCH = datetime(1980, 1, 6)
_GPS_WEEK_DURATION_S = 7 * 24 * 60 * 60
_RTKLIB_POSITION_COLUMNS = 15
_LOCAL_FRD_FRAME = "run-local Forward-Right-Down"
_GEOGRAPHIC_NED_FRAME = "static-origin local geographic North-East-Down"


@dataclass(frozen=True)
class PpkSolutionPoint:
    """One parsed RTKLIB geodetic solution row and retained quality evidence.

    Attributes:
        gps_week: GPST week number.
        gps_tow: GPST time of week in seconds.
        latitude: Geodetic latitude in radians.
        longitude: Geodetic longitude in radians.
        height_ellipsoid: Ellipsoidal height in metres.
        quality: RTKLIB integer solution-quality code as reported.
        satellite_count: RTKLIB-reported satellite count.
        std_north: Reported North coordinate standard deviation in metres.
        std_east: Reported East coordinate standard deviation in metres.
        std_down: RTKLIB ``sdu`` source field (vertical/up-coordinate standard
            deviation in metres), retained under its existing field name.
        covariance_ne: RTKLIB ``sdne`` source covariance field in m².
        covariance_eu: RTKLIB ``sdeu`` source covariance field in m².
        covariance_un: RTKLIB ``sdun`` source covariance field in m².
        age: RTKLIB-reported solution age.
        ambiguity_ratio: RTKLIB-reported ambiguity ratio.

    Notes:
        The record preserves RTKLIB evidence but does not infer that any quality
        code is sufficient for the scientific comparison at hand.
    """

    gps_week: int
    gps_tow: float
    latitude: float
    longitude: float
    height_ellipsoid: float
    quality: int
    satellite_count: int
    std_north: float
    std_east: float
    std_down: float
    covariance_ne: float
    covariance_eu: float
    covariance_un: float
    age: float
    ambiguity_ratio: float

    @property
    def t_fix_g(self) -> float:
        """Return the continuous GPST epoch in seconds."""
        return self.gps_week * _GPS_WEEK_DURATION_S + self.gps_tow


@dataclass(frozen=True)
class _StateTrajectory:
    """One exported navigation trajectory with its declared source frame.

    Attributes:
        frame: Schema-declared run-local FRD or static-origin geographic-NED
            navigation frame.
        time: Strictly increasing replay times in seconds, shape ``(N,)``.
        positions: Position rows in metres, shape ``(N, 3)``, expressed in
            ``frame``.
    """

    frame: str
    time: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class _RegistrationAnchor:
    """One manifest-declared geographic anchor for local-FRD row zero.

    Attributes:
        latitude: Anchor geodetic latitude in radians.
        longitude: Anchor geodetic longitude in radians.
        height_ellipsoid: Anchor ellipsoidal height in metres.

    Notes:
        This registration anchor is a display transform input, not a correction
        applied to the source navigation history.
    """

    latitude: float
    longitude: float
    height_ellipsoid: float


def _parse_gpst_timestamp(date_text: str, time_text: str, context: str) -> tuple[int, float]:
    """Parse one RTKLIB GPST calendar timestamp to GPS week/time of week.

    Args:
        date_text: RTKLIB calendar-date field.
        time_text: RTKLIB time-of-day field.
        context: Source context included in validation errors.

    Returns:
        ``(gps_week, gps_tow)`` for the parsed GPST instant.

    Raises:
        ValueError: The timestamp is invalid or predates the GPS epoch.
    """
    try:
        timestamp = datetime.fromisoformat(
            f"{date_text.replace('/', '-')}T{time_text}"
        )
    except ValueError as error:
        raise ValueError(f"{context}: invalid GPST timestamp.") from error

    elapsed = (timestamp - _GPS_EPOCH).total_seconds()
    if elapsed < 0.0:
        raise ValueError(f"{context}: GPST timestamp predates the GPS epoch.")
    gps_week, gps_tow = divmod(elapsed, _GPS_WEEK_DURATION_S)
    return int(gps_week), float(gps_tow)


def load_rtklib_pos(path: Path) -> tuple[PpkSolutionPoint, ...]:
    """Load a strict RTKLIB geodetic WGS-84 ``.pos`` solution.

    Args:
        path: RTKLIB text solution with documented GPST geodetic position rows.

    Returns:
        Strictly GPST-ordered parsed solution points with latitude/longitude in
        radians, ellipsoidal height in metres, and retained quality evidence.

    Raises:
        OSError: The solution cannot be read.
        ValueError: Header, row count, ordering, range, or numeric validation
            fails.

    Notes:
        The accepted format has 15 whitespace-separated data columns. Quality
        codes and covariance evidence remain attached to every point; parsing
        does not decide which PPK quality classes are physically trustworthy.
    """
    solution_path = Path(path)
    points: list[PpkSolutionPoint] = []
    found_header = False
    with solution_path.open(encoding="utf-8") as solution_file:
        for line_number, line in enumerate(solution_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                if (
                    "GPST" in stripped
                    and "latitude(deg)" in stripped
                    and "longitude(deg)" in stripped
                ):
                    found_header = True
                continue
            if not found_header:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: missing RTKLIB position header."
                )

            columns = stripped.split()
            if len(columns) != _RTKLIB_POSITION_COLUMNS:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: expected "
                    f"{_RTKLIB_POSITION_COLUMNS} columns."
                )
            try:
                gps_week, gps_tow = _parse_gpst_timestamp(
                    columns[0],
                    columns[1],
                    f"RTKLIB solution line {line_number}",
                )
                point = PpkSolutionPoint(
                    gps_week=gps_week,
                    gps_tow=gps_tow,
                    latitude=radians(float(columns[2])),
                    longitude=radians(float(columns[3])),
                    height_ellipsoid=float(columns[4]),
                    quality=int(columns[5]),
                    satellite_count=int(columns[6]),
                    std_north=float(columns[7]),
                    std_east=float(columns[8]),
                    std_down=float(columns[9]),
                    covariance_ne=float(columns[10]),
                    covariance_eu=float(columns[11]),
                    covariance_un=float(columns[12]),
                    age=float(columns[13]),
                    ambiguity_ratio=float(columns[14]),
                )
            except ValueError as error:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: invalid numeric field."
                ) from error

            numeric_values = (
                point.gps_tow,
                point.latitude,
                point.longitude,
                point.height_ellipsoid,
                point.std_north,
                point.std_east,
                point.std_down,
                point.covariance_ne,
                point.covariance_eu,
                point.covariance_un,
                point.age,
                point.ambiguity_ratio,
            )
            if not all(isfinite(value) for value in numeric_values):
                raise ValueError(
                    f"RTKLIB solution line {line_number}: values must be finite."
                )
            if not -pi / 2.0 <= point.latitude <= pi / 2.0:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: latitude is out of range."
                )
            if not -pi <= point.longitude <= pi:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: longitude is out of range."
                )
            if point.quality < 1 or point.quality > 6:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: quality must lie in [1, 6]."
                )
            if point.satellite_count < 0:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: satellite count must be nonnegative."
                )
            if min(point.std_north, point.std_east, point.std_down) < 0.0:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: standard deviations must be nonnegative."
                )
            if points and point.t_fix_g <= points[-1].t_fix_g:
                raise ValueError(
                    f"RTKLIB solution line {line_number}: GPST epochs must increase strictly."
                )
            points.append(point)

    if not found_header:
        raise ValueError("RTKLIB solution is missing its position header.")
    if not points:
        raise ValueError("RTKLIB solution contains no position rows.")
    return tuple(points)


def _load_yaml_mapping(path: Path, description: str) -> dict[str, Any]:
    """Load one YAML document that must contain a top-level mapping.

    Args:
        path: YAML file to load.
        description: Artifact name used in validation errors.

    Returns:
        Parsed top-level YAML mapping.

    Raises:
        OSError: The file cannot be read.
        ValueError: YAML parsing fails or the document is not a mapping.
    """
    with Path(path).open(encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    if not isinstance(data, dict):
        raise ValueError(f"{description} must contain one YAML mapping.")
    return data


def _require_string(value: object, field: str) -> str:
    """Return one nonempty manifest string or reject an ambiguous value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string.")
    return value


def _require_boolean(value: object, field: str, default: bool) -> bool:
    """Return an optional manifest Boolean without accepting numeric aliases."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be Boolean.")
    return value


def _resolve_manifest_path(manifest_path: Path, value: object, field: str) -> Path:
    """Resolve one required manifest-relative path and reject absolute values.

    Args:
        manifest_path: Source manifest whose directory defines relative paths.
        value: Candidate relative path value from the manifest.
        field: Manifest-field name used in validation errors.

    Returns:
        Normalized path relative to the manifest's parent directory.

    Raises:
        ValueError: The value is not a nonempty relative path string.
    """
    raw_path = Path(_require_string(value, field))
    if raw_path.is_absolute():
        return raw_path
    return manifest_path.parent / raw_path


def _load_state_trajectory(path: Path) -> _StateTrajectory:
    """Load one exported navigation CSV and its schema sidecar.

    Args:
        path: Navigation CSV artifact path.

    Returns:
        Strictly replay-time-ordered position history with its declared frame.

    Raises:
        OSError: The CSV or required schema sidecar cannot be read.
        ValueError: Artifact/schema identity, frame, columns, values, or time
            ordering are invalid.

    Notes:
        Positions remain in their source frame. Geographic-NED conversion or
        local-FRD presentation registration occurs only after this boundary.
    """
    sidecar_path = path.with_suffix(".meta.yaml")
    sidecar = _load_yaml_mapping(sidecar_path, "Navigation CSV schema sidecar")
    if sidecar.get("csv_filename") != path.name:
        raise ValueError(
            f"Navigation CSV schema sidecar {sidecar_path}: csv_filename does not match {path.name}."
        )

    artifact_type = sidecar.get("artifact_type")
    frame = sidecar.get("nav_frame")
    if artifact_type == "esekf_navigation_history":
        if frame == _LOCAL_FRD_FRAME:
            position_columns = ("p_forward_m", "p_right_m", "p_down_m")
        elif frame == _GEOGRAPHIC_NED_FRAME:
            position_columns = ("p_north_m", "p_east_m", "p_down_m")
        else:
            raise ValueError(
                f"Navigation CSV schema sidecar {sidecar_path}: unsupported nav_frame."
            )
    elif artifact_type == "raw_ins_trajectory" and frame == _LOCAL_FRD_FRAME:
        position_columns = ("p_forward_m", "p_right_m", "p_down_m")
    else:
        raise ValueError(
            f"Navigation CSV schema sidecar {sidecar_path}: unsupported artifact type/frame."
        )

    time_history: list[float] = []
    positions: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = ("time_s", *position_columns)
        if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in required_columns
        ):
            raise ValueError(
                f"Navigation CSV {path}: missing required time/position columns."
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"Navigation CSV {path} row {row_number}: too many values."
                )
            try:
                time = float(row["time_s"])
                position = tuple(float(row[column]) for column in position_columns)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Navigation CSV {path} row {row_number}: invalid time or position."
                ) from error
            if not isfinite(time) or not all(isfinite(value) for value in position):
                raise ValueError(
                    f"Navigation CSV {path} row {row_number}: values must be finite."
                )
            if time_history and time <= time_history[-1]:
                raise ValueError(
                    f"Navigation CSV {path} row {row_number}: time_s must increase strictly."
                )
            time_history.append(time)
            positions.append(position)
    if not time_history:
        raise ValueError(f"Navigation CSV {path}: contains no trajectory rows.")
    return _StateTrajectory(
        frame=frame,
        time=np.asarray(time_history, dtype=float),
        positions=np.asarray(positions, dtype=float),
    )


def _load_geographic_ned_origin(path: Path, navigation_csv_path: Path) -> tuple[float, float, float]:
    """Load the LLH origin sidecar associated with a geographic-NED CSV.

    Args:
        path: Geographic-NED origin sidecar path.
        navigation_csv_path: State CSV that the sidecar must identify.

    Returns:
        ``(latitude_rad, longitude_rad, height_ellipsoid_m)`` fixed origin.

    Raises:
        OSError: The sidecar cannot be read.
        ValueError: Artifact identity, frame, coordinates, or units are invalid.
    """
    sidecar = _load_yaml_mapping(path, "Geographic NED origin sidecar")
    if sidecar.get("artifact_type") != "geographic_ned_origin":
        raise ValueError(f"Geographic NED origin sidecar {path}: unsupported artifact type.")
    if sidecar.get("csv_filename") != navigation_csv_path.name:
        raise ValueError(
            f"Geographic NED origin sidecar {path}: csv_filename does not match "
            f"{navigation_csv_path.name}."
        )
    if sidecar.get("nav_frame") != _GEOGRAPHIC_NED_FRAME:
        raise ValueError(f"Geographic NED origin sidecar {path}: unsupported nav_frame.")
    try:
        origin = (
            float(sidecar["latitude_rad"]),
            float(sidecar["longitude_rad"]),
            float(sidecar["height_ellipsoid_m"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Geographic NED origin sidecar {path}: invalid origin.") from error
    if not all(isfinite(value) for value in origin):
        raise ValueError(f"Geographic NED origin sidecar {path}: origin must be finite.")
    if abs(origin[0]) > pi / 2.0 or abs(origin[1]) > pi:
        raise ValueError(f"Geographic NED origin sidecar {path}: origin must use radians.")
    return origin


def _as_llh_from_geographic_ned(
    positions_ned: np.ndarray,
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Convert fixed-origin geographic-NED rows to LLH map positions.

    Args:
        positions_ned: ``(N, 3)`` North-East-Down positions in metres.
        origin: Fixed ``(latitude_rad, longitude_rad, height_m)`` LLH origin.

    Returns:
        ``(N, 3)`` latitude, longitude, ellipsoidal-height rows in radians,
        radians, and metres respectively.
    """
    return np.asarray(
        [ned_to_llh(position, *origin) for position in positions_ned],
        dtype=float,
    )


def _resolve_ppk_anchor(
    points: tuple[PpkSolutionPoint, ...],
    epoch_text: object,
) -> _RegistrationAnchor:
    """Resolve a manifest-declared exact PPK epoch to a display anchor.

    Args:
        points: Parsed PPK solution points ordered by GPST.
        epoch_text: Manifest ``registration.anchor_ppk_gpst`` value.

    Returns:
        Geographic anchor from the declared PPK solution epoch.

    Raises:
        ValueError: The value is malformed or does not identify an exact PPK
            epoch in ``points``.
    """
    epoch_value = _require_string(epoch_text, "registration.anchor_ppk_gpst")
    gps_week, gps_tow = _parse_gpst_timestamp(
        epoch_value.split("T", maxsplit=1)[0],
        epoch_value.split("T", maxsplit=1)[1] if "T" in epoch_value else "",
        "registration.anchor_ppk_gpst",
    )
    target_time = gps_week * _GPS_WEEK_DURATION_S + gps_tow
    for point in points:
        if abs(point.t_fix_g - target_time) <= 1e-6:
            return _RegistrationAnchor(
                latitude=point.latitude,
                longitude=point.longitude,
                height_ellipsoid=point.height_ellipsoid,
            )
    raise ValueError(
        "registration.anchor_ppk_gpst does not identify an exact PPK epoch."
    )


def _require_anchor_gps_week(value: object) -> int:
    """Return one nonnegative GNSS-anchor GPS week from a manifest scalar."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "registration.anchor_gnss_fix.gps_week must be a nonnegative integer."
        )
    return value


def _require_anchor_gps_tow(value: object) -> float:
    """Return one valid GNSS-anchor time-of-week from a manifest scalar."""
    if isinstance(value, bool):
        raise ValueError(
            "registration.anchor_gnss_fix.gps_tow must be a finite number."
        )
    try:
        gps_tow = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "registration.anchor_gnss_fix.gps_tow must be a finite number."
        ) from error
    if not isfinite(gps_tow) or not 0.0 <= gps_tow < _GPS_WEEK_DURATION_S:
        raise ValueError(
            "registration.anchor_gnss_fix.gps_tow must lie in [0, one GPS-week duration)."
        )
    return gps_tow


def _resolve_gnss_anchor(
    manifest_path: Path,
    value: object,
) -> _RegistrationAnchor:
    """Resolve a manifest-declared exact receiver-valid fix to an anchor.

    Args:
        manifest_path: Source manifest whose directory resolves the table path.
        value: ``registration.anchor_gnss_fix`` mapping from the manifest.

    Returns:
        Geographic anchor from the selected receiver-valid GNSS-table row.

    Raises:
        OSError: The declared GNSS table cannot be read.
        ValueError: The mapping is invalid or does not select exactly one valid
            receiver fix at the declared GPS week/time of week.
    """
    if not isinstance(value, dict):
        raise ValueError("registration.anchor_gnss_fix must be a YAML mapping.")
    table_path = _resolve_manifest_path(
        manifest_path,
        value.get("path"),
        "registration.anchor_gnss_fix.path",
    )
    gps_week = _require_anchor_gps_week(value.get("gps_week"))
    gps_tow = _require_anchor_gps_tow(value.get("gps_tow"))
    matching_fixes = tuple(
        fix
        for fix in load_gnss_fix_table(table_path)
        if fix.gps_week == gps_week and abs(fix.gps_tow - gps_tow) <= 1e-6
    )
    if not matching_fixes:
        raise ValueError(
            "registration.anchor_gnss_fix does not identify an exact GNSS-table epoch."
        )
    if len(matching_fixes) != 1:
        raise ValueError(
            "registration.anchor_gnss_fix identifies multiple GNSS-table rows."
        )
    fix: GnssFixLLH = matching_fixes[0]
    if not fix.fix_valid:
        raise ValueError(
            "registration.anchor_gnss_fix must identify a receiver-valid GNSS-table row."
        )
    return _RegistrationAnchor(
        latitude=fix.latitude,
        longitude=fix.longitude,
        height_ellipsoid=fix.height_ellipsoid,
    )


def _local_frd_to_llh(
    positions_frd: np.ndarray,
    anchor: _RegistrationAnchor,
    heading_clockwise_from_north: object,
) -> np.ndarray:
    """Apply one declared local-FRD-to-geographic-NED display registration.

    Args:
        positions_frd: ``(N, 3)`` run-local Forward-Right-Down positions in m.
        anchor: Declared LLH position for row-zero presentation.
        heading_clockwise_from_north: Declared Forward-to-North heading
            clockwise from North (degrees).

    Returns:
        ``(N, 3)`` registered LLH rows in radians, radians, and metres.

    Raises:
        ValueError: The declared heading is absent or not finite.

    Notes:
        This rotates Forward/Right into North/East and anchors the displayed
        trace. It neither changes the source state history nor estimates an
        attitude, timing alignment, or navigation correction.
    """
    try:
        heading = float(heading_clockwise_from_north)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "registration.heading_deg_clockwise_from_north must be finite."
        ) from error
    if not isfinite(heading):
        raise ValueError("registration.heading_deg_clockwise_from_north must be finite.")
    heading = radians(heading % 360.0)
    cosine = np.cos(heading)
    sine = np.sin(heading)
    positions_ned = np.column_stack(
        (
            cosine * positions_frd[:, 0] - sine * positions_frd[:, 1],
            sine * positions_frd[:, 0] + cosine * positions_frd[:, 1],
            positions_frd[:, 2],
        )
    )
    origin = (anchor.latitude, anchor.longitude, anchor.height_ellipsoid)
    return _as_llh_from_geographic_ned(positions_ned, origin)


def _trace_style(
    trace: dict[str, Any],
    index: int,
) -> tuple[str, str, bool, bool]:
    """Read one trace's presentation-only naming, color, and layer controls."""
    label = _require_string(trace.get("label"), f"traces[{index}].label")
    color = _require_string(trace.get("color"), f"traces[{index}].color")
    visible = _require_boolean(trace.get("visible"), f"traces[{index}].visible", True)
    include_in_bounds = _require_boolean(
        trace.get("include_in_bounds"),
        f"traces[{index}].include_in_bounds",
        True,
    )
    return (
        label,
        color,
        visible,
        include_in_bounds,
    )


def _ppk_positions(
    points: tuple[PpkSolutionPoint, ...],
    quality: object,
    index: int,
) -> np.ndarray:
    """Select declared PPK quality rows without changing their source order.

    Args:
        points: Parsed PPK solution points ordered by GPST.
        quality: Manifest quality selector: ``all``, ``fixed``, or ``float``.
        index: Trace index used in validation errors.

    Returns:
        ``(N, 3)`` LLH rows in radians, radians, and metres.

    Raises:
        ValueError: The selector is unsupported or selects no PPK rows.

    Notes:
        Selecting a quality code controls only map presentation; it does not
        conceal which source quality class a selected trace represents.
    """
    quality_name = _require_string(quality, f"traces[{index}].quality")
    quality_codes = {"all": None, "fixed": 1, "float": 2}
    if quality_name not in quality_codes:
        raise ValueError(
            f"traces[{index}].quality must be one of: all, fixed, float."
        )
    selected = [
        point
        for point in points
        if quality_codes[quality_name] is None or point.quality == quality_codes[quality_name]
    ]
    if not selected:
        raise ValueError(
            f"traces[{index}].quality selected no PPK points from the solution."
        )
    return np.asarray(
        [
            (point.latitude, point.longitude, point.height_ellipsoid)
            for point in selected
        ],
        dtype=float,
    )


def _build_map_traces(manifest_path: Path) -> tuple[str, FoliumBasemapStyle, tuple[LlhMapTrace, ...]]:
    """Resolve one comparison manifest into map-ready LLH traces.

    Args:
        manifest_path: Schema-versioned trajectory-comparison YAML manifest.

    Returns:
        Map title, selected Folium basemap style, and ordered LLH traces.

    Raises:
        OSError: A declared manifest, CSV, sidecar, table, or PPK artifact cannot
            be read.
        ValueError: Manifest structure, artifact schema, trace selection, or
            required presentation registration is invalid.

    Notes:
        Manifest paths are relative to the manifest. Local-FRD traces need
        exactly one declared PPK or receiver-fix anchor and a declared heading;
        geographic-NED traces use their own exported origin sidecar. All such
        transformations are presentation-only.
    """
    manifest_path = Path(manifest_path)
    manifest = _load_yaml_mapping(manifest_path, "Trajectory comparison manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("Trajectory comparison manifest schema_version must be 1.")
    title = _require_string(manifest.get("title"), "title")
    try:
        basemap = FoliumBasemapStyle(_require_string(manifest.get("basemap"), "basemap"))
    except ValueError as error:
        styles = ", ".join(style.value for style in FoliumBasemapStyle)
        raise ValueError(f"basemap must be one of: {styles}.") from error
    raw_traces = manifest.get("traces")
    if not isinstance(raw_traces, list) or not raw_traces:
        raise ValueError("traces must be a nonempty YAML list.")

    ppk_points: tuple[PpkSolutionPoint, ...] | None = None

    def get_ppk_points() -> tuple[PpkSolutionPoint, ...]:
        nonlocal ppk_points
        if ppk_points is None:
            ppk_path = _resolve_manifest_path(
                manifest_path,
                manifest.get("ppk_solution"),
                "ppk_solution",
            )
            ppk_points = load_rtklib_pos(ppk_path)
        return ppk_points

    traces: list[LlhMapTrace] = []
    for index, raw_trace in enumerate(raw_traces):
        if not isinstance(raw_trace, dict):
            raise ValueError(f"traces[{index}] must be a YAML mapping.")
        (
            label,
            color,
            visible,
            include_in_bounds,
        ) = _trace_style(raw_trace, index)
        kind = _require_string(raw_trace.get("kind"), f"traces[{index}].kind")
        if kind == "ppk":
            positions_llh = _ppk_positions(
                get_ppk_points(),
                raw_trace.get("quality"),
                index,
            )
        elif kind == "gnss_fix_table":
            fix_path = _resolve_manifest_path(
                manifest_path,
                raw_trace.get("path"),
                f"traces[{index}].path",
            )
            positions_llh = np.asarray(
                [
                    (fix.latitude, fix.longitude, fix.height_ellipsoid)
                    for fix in load_gnss_fix_table(fix_path)
                    if fix.fix_valid
                ],
                dtype=float,
            )
            if positions_llh.size == 0:
                raise ValueError(f"traces[{index}] selected no receiver-valid GNSS fixes.")
        elif kind == "state_csv":
            state_path = _resolve_manifest_path(
                manifest_path,
                raw_trace.get("path"),
                f"traces[{index}].path",
            )
            state_trajectory = _load_state_trajectory(state_path)
            if state_trajectory.frame == _GEOGRAPHIC_NED_FRAME:
                origin_path_value = raw_trace.get("geographic_ned_origin_path")
                origin_path = (
                    state_path.with_suffix(".georef.yaml")
                    if origin_path_value is None
                    else _resolve_manifest_path(
                        manifest_path,
                        origin_path_value,
                        f"traces[{index}].geographic_ned_origin_path",
                    )
                )
                positions_llh = _as_llh_from_geographic_ned(
                    state_trajectory.positions,
                    _load_geographic_ned_origin(origin_path, state_path),
                )
            elif state_trajectory.frame == _LOCAL_FRD_FRAME:
                registration = raw_trace.get("registration")
                if not isinstance(registration, dict):
                    raise ValueError(
                        f"traces[{index}].registration is required for local-FRD state CSVs."
                    )
                has_ppk_anchor = "anchor_ppk_gpst" in registration
                has_gnss_anchor = "anchor_gnss_fix" in registration
                if has_ppk_anchor == has_gnss_anchor:
                    raise ValueError(
                        f"traces[{index}].registration must declare exactly one of "
                        "anchor_ppk_gpst or anchor_gnss_fix."
                    )
                anchor = (
                    _resolve_ppk_anchor(
                        get_ppk_points(),
                        registration["anchor_ppk_gpst"],
                    )
                    if has_ppk_anchor
                    else _resolve_gnss_anchor(
                        manifest_path,
                        registration["anchor_gnss_fix"],
                    )
                )
                positions_llh = _local_frd_to_llh(
                    state_trajectory.positions,
                    anchor,
                    registration.get("heading_deg_clockwise_from_north"),
                )
            else:
                raise ValueError(f"traces[{index}] has an unsupported navigation frame.")
        else:
            raise ValueError(
                f"traces[{index}].kind must be one of: ppk, gnss_fix_table, state_csv."
            )

        traces.append(
            LlhMapTrace(
                label=label,
                positions_llh=positions_llh,
                color=color,
                visible=visible,
                include_in_bounds=include_in_bounds,
            )
        )
    return title, basemap, tuple(traces)


def main() -> int:
    """Build one interactive map from a declared comparison manifest.

    Returns:
        ``0`` after writing the requested HTML map, ``1`` when manifest or
        artifact validation fails, or ``2`` when an input or output cannot be
        accessed.

    Notes:
        The command reads already-produced artifacts and writes only the map.
        It does not rerun a filter, modify source files, or claim accuracy from
        display registration.
    """
    parser = argparse.ArgumentParser(
        description="Overlay exported navigation, receiver, and RTKLIB PPK trajectories.",
    )
    parser.add_argument(
        "manifest_path",
        type=Path,
        help="Path to one trajectory-comparison YAML manifest",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        required=True,
        metavar="PATH.html",
        help="Interactive Folium map artifact to write",
    )
    args = parser.parse_args()

    try:
        title, basemap, traces = _build_map_traces(args.manifest_path)
        write_llh_trajectory_comparison_map(
            args.html_output,
            traces,
            basemap=basemap,
            title=title,
        )
    except OSError as error:
        print(f"Could not access comparison artifact: {error}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as error:
        print(f"Trajectory comparison rejected: {error}", file=sys.stderr)
        return 1

    print(f"Wrote interactive trajectory comparison: {args.html_output}")
    for trace in traces:
        print(f"  {trace.label}: {np.asarray(trace.positions_llh).shape[0]} positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
