"""Serialize terminal GNSS-fusion diagnostics as portable evidence artifacts.

This module receives already-computed receiver-valid diagnostic records and
writes their fixed CSV rows with a self-describing YAML sidecar. Each row
preserves receiver timing and quality facts, GNSS-to-IMU timing estimates, and
the terminal fusion outcome. It does not read captures, run replay, select
filter settings, or reinterpret an outcome as measurement truth.
"""
import csv
from collections.abc import Iterable
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

from bootins.replay.gnss_fusion_models import GnssFusionDiagnostic


GNSS_FUSION_DIAGNOSTIC_SCHEMA_VERSION = 1

_ARTIFACT_NAME = "gnss_fusion_diagnostics"
_NAVIGATION_FRAME = "static-origin local geographic North-East-Down"
_NED_AXES = ("n", "e", "d")
_LOWER_TRIANGLE_INDICES = (
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
)


def _lower_triangle_column_names(prefix: str) -> tuple[str, ...]:
    """Return lower-triangle NED covariance headers including the diagonal."""
    return tuple(
        f"{prefix}_{_NED_AXES[row]}{_NED_AXES[column]}_m2"
        for row, column in _LOWER_TRIANGLE_INDICES
    )


def _csv_column_names() -> tuple[str, ...]:
    """Return the complete fixed header for terminal GNSS fusion diagnostics."""
    return (
        "gps_week",
        "gps_tow_s",
        "host_arrival_monotonic_s",
        "native_record_index",
        "satellite_count",
        "horizontal_accuracy_m",
        "vertical_accuracy_m",
        "t_fix_g_s",
        "t_arrival_i_s",
        "tau_hat_s",
        "t_fix_i_hat_s",
        "t_applied_i_s",
        "status",
        "nis",
        "p_gnss_n_m",
        "p_gnss_e_m",
        "p_gnss_d_m",
        "innovation_n_m",
        "innovation_e_m",
        "innovation_d_m",
        *_lower_triangle_column_names("S"),
        *_lower_triangle_column_names("P_dp"),
    )


def _sidecar_payload() -> dict[str, object]:
    """Return the fixed YAML metadata needed to interpret one diagnostics CSV."""
    return {
        "schema_version": GNSS_FUSION_DIAGNOSTIC_SCHEMA_VERSION,
        "artifact": _ARTIFACT_NAME,
        "navigation_frame": _NAVIGATION_FRAME,
        "row_semantics": "one terminal receiver-valid GNSS fusion outcome",
        "time_conventions": {
            "gps_week": "raw receiver GPS week",
            "gps_tow_s": "raw receiver GPS time-of-week",
            "host_arrival_monotonic_s": "raw host arrival time",
            "t_fix_g_s": "seed-relative GNSS measurement epoch",
            "t_arrival_i_s": "scheduler delivery epoch on the IMU timeline",
            "tau_hat_s": "clock offset used at the terminal fusion attempt",
            "t_fix_i_hat_s": "mapped GNSS epoch on the IMU timeline",
            "t_applied_i_s": "snapped IMU posterior epoch when evaluated",
        },
        "evaluation_fields": {
            "available_for_statuses": ["ACCEPTED", "REJECTED"],
            "unavailable_for_statuses": ["STALE", "FUTURE_DROPPED"],
            "unavailable_csv_encoding": "empty field",
        },
        "covariance_encoding": {
            "frame": _NAVIGATION_FRAME,
            "order": "lower triangle including diagonal",
            "axes": list(_NED_AXES),
            "innovation_covariance": "S = H P H^T + R",
            "prior_position_covariance": "snapped pre-update P[dp, dp]",
        },
        "columns": list(_csv_column_names()),
    }


def _validated_diagnostics(
    diagnostics: Iterable[GnssFusionDiagnostic],
) -> tuple[GnssFusionDiagnostic, ...]:
    """Return one complete sequence of receiver-valid diagnostic records."""
    records = tuple(diagnostics)
    for index, diagnostic in enumerate(records):
        if not isinstance(diagnostic, GnssFusionDiagnostic):
            raise ValueError(
                "diagnostics must contain only GnssFusionDiagnostic records; "
                f"entry {index} is invalid."
            )
        if not diagnostic.fix.fix_valid:
            raise ValueError(
                "diagnostics must contain only receiver-valid fixes; "
                f"entry {index} is invalid."
            )
    return records


def _format_float(value: float) -> str:
    """Return one finite diagnostic scalar with precision-preserving formatting."""
    return format(float(value), ".17g")


def _optional_float(value: float | None) -> str:
    """Return an empty cell only for one unavailable diagnostic scalar."""
    return "" if value is None else _format_float(value)


def _lower_triangle_values(matrix: np.ndarray) -> tuple[str, ...]:
    """Return one NED covariance lower triangle in the schema's fixed order."""
    return tuple(
        _format_float(matrix[row, column])
        for row, column in _LOWER_TRIANGLE_INDICES
    )


def _diagnostic_row(diagnostic: GnssFusionDiagnostic) -> tuple[str, ...]:
    """Flatten one validated terminal GNSS outcome into one fixed CSV row."""
    fix = diagnostic.fix
    evaluation_available = diagnostic.innovation_ned is not None
    if evaluation_available:
        if (
            diagnostic.innovation_covariance_ned is None
            or diagnostic.prior_position_covariance_ned is None
        ):
            raise ValueError("Evaluated diagnostic is missing covariance evidence.")
        evaluation_values = (
            *(_format_float(value) for value in diagnostic.innovation_ned),
            *_lower_triangle_values(diagnostic.innovation_covariance_ned),
            *_lower_triangle_values(diagnostic.prior_position_covariance_ned),
        )
    else:
        evaluation_values = ("",) * 15

    return (
        str(fix.gps_week),
        _format_float(fix.gps_tow),
        _format_float(fix.host_arrival_monotonic),
        str(fix.native_record_index),
        str(fix.satellite_count),
        _format_float(fix.horizontal_accuracy),
        _format_float(fix.vertical_accuracy),
        _format_float(diagnostic.t_fix_g),
        _format_float(diagnostic.t_arrival_i),
        _format_float(diagnostic.tau_hat),
        _format_float(diagnostic.update.t_fix_i_hat),
        _optional_float(diagnostic.update.t_applied_i),
        diagnostic.update.status.name,
        _optional_float(diagnostic.update.nis),
        *(_format_float(value) for value in diagnostic.p_gnss_ned),
        *evaluation_values,
    )


def _write_text_atomically(output_path: Path, content: str) -> None:
    """Replace one text artifact atomically after writing a complete temp file."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _csv_content(diagnostics: tuple[GnssFusionDiagnostic, ...]) -> str:
    """Serialize all validated rows into one complete CSV text artifact."""
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_csv_column_names())
    writer.writerows(_diagnostic_row(diagnostic) for diagnostic in diagnostics)
    return buffer.getvalue()


def write_gnss_fusion_diagnostics_csv(
    output_path: str | Path,
    diagnostics: Iterable[GnssFusionDiagnostic],
) -> None:
    """Write a GNSS-fusion diagnostics CSV and its adjacent YAML sidecar.

    Args:
        output_path: Destination CSV path, atomically replaced after complete
            CSV serialization and followed by an adjacent ``.meta.yaml`` sidecar.
        diagnostics: Receiver-valid terminal fusion outcomes in source order.

    Raises:
        ValueError: A record is not a receiver-valid ``GnssFusionDiagnostic`` or
            its diagnostic fields cannot satisfy the fixed row convention.
        OSError: A temporary or final CSV/sidecar artifact operation fails.

    Notes:
        The sequence may be empty when no receiver-valid post-startup fix reached
        a terminal outcome. Accepted and NIS-rejected rows include pre-update
        innovation and covariance evidence; stale and future-dropped rows use
        empty cells for unavailable evaluation quantities, never invented zeros.
        The sidecar distinguishes raw receiver GPS/host times from seed-relative
        GNSS and IMU-timeline fields. Rows document filter handling, not receiver
        accuracy or navigation truth.
    """
    output_path = Path(output_path)
    validated_diagnostics = _validated_diagnostics(diagnostics)
    sidecar = _sidecar_payload()
    sidecar["csv_filename"] = output_path.name
    csv_content = _csv_content(validated_diagnostics)
    sidecar_content = yaml.safe_dump(sidecar, sort_keys=False)

    _write_text_atomically(output_path, csv_content)
    _write_text_atomically(
        output_path.with_suffix(".meta.yaml"),
        sidecar_content,
    )
