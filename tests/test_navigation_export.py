"""Scientific alignment checks for navigation-history export artifacts."""

import csv

import numpy as np
import pytest
import yaml

from bootins.reporting.navigation_export import (
    NavigationHistoryLayout,
    write_esekf_history_csv,
    write_geographic_ned_origin_sidecar,
    write_raw_ins_trajectory_csv,
)


def _history(clock_aware: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.array([0.0, 0.01])
    nominal_dimension = 17 if clock_aware else 16
    error_dimension = 16 if clock_aware else 15
    nominals = np.arange(2 * nominal_dimension, dtype=float).reshape(2, nominal_dimension) / 10.0
    covariance = np.repeat(np.eye(error_dimension)[None, :, :], 2, axis=0)
    covariance[1, 1, 0] = covariance[1, 0, 1] = 0.25
    return time, nominals, covariance


def test_local_frd_export_preserves_state_covariance_and_zupt_row_alignment(tmp_path):
    time, nominals, covariance = _history()
    path = tmp_path / "history.csv"

    write_esekf_history_csv(path, time, nominals, covariance, zupt_applied=np.array([False, True]))

    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    schema = yaml.safe_load((tmp_path / "history.meta.yaml").read_text(encoding="utf-8"))
    assert [row["zupt_applied"] for row in rows] == ["0", "1"]
    assert float(rows[1]["p_forward_m"]) == pytest.approx(nominals[1, 0])
    assert float(rows[1]["P_dp_right_dp_forward"]) == pytest.approx(0.25)
    assert schema["nav_frame"] == "run-local Forward-Right-Down"
    assert schema["covariance_storage"] == "lower_triangle_row_major"


def test_clock_aware_export_preserves_geographic_ned_and_clock_state_alignment(tmp_path):
    time, nominals, covariance = _history(clock_aware=True)
    path = tmp_path / "aided.csv"

    write_esekf_history_csv(
        path, time, nominals, covariance, zupt_applied=np.array([False, True]),
        layout=NavigationHistoryLayout.GEOGRAPHIC_NED_CLOCK,
    )

    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    schema = yaml.safe_load((tmp_path / "aided.meta.yaml").read_text(encoding="utf-8"))
    assert float(rows[1]["p_north_m"]) == pytest.approx(nominals[1, 0])
    assert float(rows[1]["tau_s"]) == pytest.approx(nominals[1, 16])
    assert schema["state_layout"] == "geographic_ned_clock"
    assert schema["nav_frame"] == "static-origin local geographic North-East-Down"


def test_history_export_refuses_misaligned_time_or_nonsymmetric_covariance(tmp_path):
    time, nominals, covariance = _history()
    with pytest.raises(ValueError, match="strictly increasing"):
        write_esekf_history_csv(tmp_path / "time.csv", [0.0, 0.0], nominals, covariance)

    covariance[0, 1, 0] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        write_esekf_history_csv(tmp_path / "covariance.csv", time, nominals, covariance)


def test_raw_and_geographic_sidecars_document_their_actual_coordinate_references(tmp_path):
    raw_path = tmp_path / "raw.csv"
    write_raw_ins_trajectory_csv(
        raw_path, np.array([0.0, 0.1]), np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.2, 0.3]]),
    )
    aided_path = tmp_path / "aided.csv"
    write_geographic_ned_origin_sidecar(aided_path, np.radians(39.478), np.radians(-76.137), -21.8)

    raw_schema = yaml.safe_load((tmp_path / "raw.meta.yaml").read_text(encoding="utf-8"))
    georef = yaml.safe_load((tmp_path / "aided.georef.yaml").read_text(encoding="utf-8"))
    assert raw_schema["artifact_type"] == "raw_ins_trajectory"
    assert raw_schema["nav_frame"] == "run-local Forward-Right-Down"
    assert georef["artifact_type"] == "geographic_ned_origin"
    assert georef["latitude_rad"] == pytest.approx(np.radians(39.478))
