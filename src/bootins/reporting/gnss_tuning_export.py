"""Write portable GNSS-sweep summaries, profiles, and provenance artifacts.

These writers serialize already-computed candidate definitions and terminal
replay summaries. They preserve controlled overrides and caller-supplied input
paths but do not enumerate candidates, replay captures, rank filter settings,
or establish that a configuration improves navigation accuracy.
"""
import csv
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterable, Mapping

import yaml

from bootins.analysis.gnss_tuning import GnssSweepSummary


GNSS_SWEEP_SUMMARY_SCHEMA_VERSION = 1
GNSS_SWEEP_MANIFEST_SCHEMA_VERSION = 1


def _csv_column_names() -> tuple[str, ...]:
    """Return the fixed one-row-per-completed-candidate CSV schema."""
    return (
        "candidate",
        "overrides_json",
        "resolved_profile",
        "gnss_diagnostics_csv",
        "duration_s",
        "final_position_n_m",
        "final_position_e_m",
        "final_position_d_m",
        "final_velocity_n_mps",
        "final_velocity_e_mps",
        "final_velocity_d_mps",
        "final_clock_offset_s",
        "accepted_count",
        "nis_rejected_count",
        "stale_count",
        "future_dropped_count",
        "receiver_invalid_count",
        "scheduler_dropped_before_imu_count",
        "scheduler_dropped_after_imu_count",
        "accepted_nis_count",
        "accepted_nis_min",
        "accepted_nis_median",
        "accepted_nis_p95",
        "accepted_nis_max",
        "rejected_nis_count",
        "rejected_nis_min",
        "rejected_nis_median",
        "rejected_nis_p95",
        "rejected_nis_max",
    )


def _format_float(value: float) -> str:
    """Format one finite scalar without reducing diagnostic precision."""
    return format(float(value), ".17g")


def _optional_float(value: float | None) -> str:
    """Render an unavailable distribution statistic as one empty CSV cell."""
    return "" if value is None else _format_float(value)


def _overrides_json(summary: GnssSweepSummary) -> str:
    """Render a compact stable mapping from each tuned path to its value."""
    overrides = {
        ".".join(override.path): override.value
        for override in summary.overrides
    }
    try:
        return json.dumps(
            overrides,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Candidate '{summary.candidate_name}' has non-serializable overrides."
        ) from error


def _summary_row(summary: GnssSweepSummary) -> tuple[str, ...]:
    """Flatten one completed candidate into the locked summary schema."""
    accepted_nis = summary.accepted_nis
    rejected_nis = summary.rejected_nis
    return (
        summary.candidate_name,
        _overrides_json(summary),
        f"profiles/{summary.candidate_name}.yaml",
        f"diagnostics/{summary.candidate_name}.csv",
        _format_float(summary.duration),
        *(_format_float(value) for value in summary.final_position_ned),
        *(_format_float(value) for value in summary.final_velocity_ned),
        _format_float(summary.final_clock_offset),
        str(summary.accepted_count),
        str(summary.rejected_count),
        str(summary.stale_count),
        str(summary.future_dropped_count),
        str(summary.invalid_fix_count),
        str(summary.dropped_before_imu_count),
        str(summary.dropped_after_imu_count),
        str(accepted_nis.count),
        _optional_float(accepted_nis.minimum),
        _optional_float(accepted_nis.median),
        _optional_float(accepted_nis.percentile_95),
        _optional_float(accepted_nis.maximum),
        str(rejected_nis.count),
        _optional_float(rejected_nis.minimum),
        _optional_float(rejected_nis.median),
        _optional_float(rejected_nis.percentile_95),
        _optional_float(rejected_nis.maximum),
    )


def _validate_summaries(
    summaries: Iterable[GnssSweepSummary],
) -> tuple[GnssSweepSummary, ...]:
    """Return a complete nonempty, uniquely named sequence of summaries."""
    records = tuple(summaries)
    if not records:
        raise ValueError("GNSS sweep summary requires at least one candidate.")
    for index, summary in enumerate(records):
        if not isinstance(summary, GnssSweepSummary):
            raise ValueError(
                "GNSS sweep summaries must contain only GnssSweepSummary records; "
                f"entry {index} is invalid."
            )
    names = [summary.candidate_name for summary in records]
    if len(names) != len(set(names)):
        raise ValueError("GNSS sweep summary candidate names must be unique.")
    return records


def _write_text_atomically(output_path: Path, content: str) -> None:
    """Replace one artifact only after a complete temporary write and sync."""
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


def _summary_csv_content(summaries: tuple[GnssSweepSummary, ...]) -> str:
    """Serialize the fixed candidate-summary rows before artifact replacement."""
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_csv_column_names())
    writer.writerows(_summary_row(summary) for summary in summaries)
    return buffer.getvalue()


def _summary_sidecar(output_path: Path) -> dict[str, object]:
    """Describe the fixed summary columns and deliberate lack of auto-ranking."""
    return {
        "schema_version": GNSS_SWEEP_SUMMARY_SCHEMA_VERSION,
        "artifact": "gnss_filter_sweep_summary",
        "csv_filename": output_path.name,
        "row_semantics": "one completed candidate replay",
        "candidate_profile_convention": "profiles/<candidate>.yaml",
        "gnss_diagnostics_convention": "diagnostics/<candidate>.csv",
        "nis_statistics": {
            "population": "terminal accepted or NIS-rejected GNSS updates",
            "percentile_95": "linear 95th percentile",
            "unavailable_csv_encoding": "empty field",
        },
        "ranking": "none; this artifact reports diagnostics only",
        "columns": list(_csv_column_names()),
    }


def write_gnss_sweep_summary_csv(
    output_path: str | Path,
    summaries: Iterable[GnssSweepSummary],
) -> None:
    """Write candidate summaries and a self-describing adjacent YAML sidecar.

    Args:
        output_path: Destination summary CSV path, atomically replaced together
            with an adjacent ``.meta.yaml`` schema sidecar.
        summaries: Nonempty, uniquely named completed-candidate summaries.

    Raises:
        ValueError: Summaries are empty, mistyped, or use duplicate candidate
            names.
        OSError: A temporary or final CSV/sidecar artifact operation fails.

    Notes:
        Each row is descriptive terminal replay evidence. Empty NIS statistics
        serialize as empty fields; this is unavailable evidence, not zero NIS.
        The sidecar explicitly records that the artifact performs no ranking.
    """
    output_path = Path(output_path)
    validated_summaries = _validate_summaries(summaries)
    csv_content = _summary_csv_content(validated_summaries)
    sidecar_content = yaml.safe_dump(_summary_sidecar(output_path), sort_keys=False)
    _write_text_atomically(output_path, csv_content)
    _write_text_atomically(output_path.with_suffix(".meta.yaml"), sidecar_content)


def write_resolved_gnss_sweep_profile(
    output_path: str | Path,
    document: Mapping[str, object],
) -> None:
    """Write one resolved full candidate profile as a YAML artifact.

    Args:
        output_path: Destination YAML path, atomically replaced by the resolved
            profile.
        document: Full resolved configuration mapping.

    Raises:
        ValueError: ``document`` is not a mapping.
        OSError: The temporary or final profile write fails.

    Notes:
        The writer serializes its supplied resolved copy and never mutates the
        local source profile or evaluates navigation behavior.
    """
    if not isinstance(document, Mapping):
        raise ValueError("Resolved GNSS sweep profile must be a mapping.")
    output_path = Path(output_path)
    _write_text_atomically(output_path, yaml.safe_dump(dict(document), sort_keys=False))


def write_gnss_sweep_manifest(
    output_path: str | Path,
    *,
    capture_path: str | Path,
    gnss_fix_table_path: str | Path,
    base_config_path: str | Path,
    plan_path: str | Path,
    summaries: Iterable[GnssSweepSummary],
) -> None:
    """Write source paths and candidate-artifact links for one sweep.

    Args:
        output_path: Destination YAML manifest path, atomically replaced.
        capture_path: Source IMU-capture path recorded as caller-supplied text.
        gnss_fix_table_path: Normalized GNSS-table path recorded as supplied.
        base_config_path: Base navigation-profile path recorded as supplied.
        plan_path: Sweep-plan path recorded as supplied.
        summaries: Nonempty uniquely named completed-candidate summaries.

    Raises:
        ValueError: Candidate summaries are empty, mistyped, or duplicated.
        OSError: The temporary or final manifest write fails.

    Notes:
        The manifest links expected profile and diagnostics paths for each
        candidate. It records artifact provenance, not a performance ranking or
        confirmation that every linked path is scientifically comparable.
    """
    validated_summaries = _validate_summaries(summaries)
    output_path = Path(output_path)
    manifest = {
        "schema_version": GNSS_SWEEP_MANIFEST_SCHEMA_VERSION,
        "artifact": "gnss_filter_sweep",
        "inputs": {
            "capture": str(capture_path),
            "gnss_fix_table": str(gnss_fix_table_path),
            "base_config": str(base_config_path),
            "plan": str(plan_path),
        },
        "candidates": [
            {
                "name": summary.candidate_name,
                "resolved_profile": f"profiles/{summary.candidate_name}.yaml",
                "gnss_diagnostics": f"diagnostics/{summary.candidate_name}.csv",
            }
            for summary in validated_summaries
        ],
    }
    _write_text_atomically(output_path, yaml.safe_dump(manifest, sort_keys=False))
