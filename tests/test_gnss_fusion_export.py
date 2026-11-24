"""Tests for the fixed GNSS-fusion diagnostic export schema."""
import csv

import numpy as np
import pytest
import yaml

from bootins.reporting.gnss_fusion_export import (
    GNSS_FUSION_DIAGNOSTIC_SCHEMA_VERSION,
    _csv_column_names,
    _sidecar_payload,
    write_gnss_fusion_diagnostics_csv,
)
from bootins.sensors.gnss_observations import GnssFixLLH
from bootins.replay.gnss_fusion_models import (
    DelayedGnssUpdate,
    DelayedGnssUpdateStatus,
    GnssFusionDiagnostic,
)


def _fix(*, valid: bool = True) -> GnssFixLLH:
    """Return one normalized source fix for a diagnostic-export row."""
    return GnssFixLLH(
        gps_week=2300,
        gps_tow=1000.25,
        host_arrival_monotonic=12.5,
        latitude=0.71,
        longitude=-1.09,
        height_ellipsoid=124.0,
        fix_valid=valid,
        satellite_count=12 if valid else 0,
        horizontal_accuracy=1.25,
        vertical_accuracy=2.5,
        native_record_index=17,
    )


def _accepted_diagnostic() -> GnssFusionDiagnostic:
    """Return one evaluated diagnostic with asymmetric off-diagonal values."""
    return GnssFusionDiagnostic(
        fix=_fix(),
        t_fix_g=0.25,
        t_arrival_i=0.3,
        p_gnss_ned=np.array([1.0, -2.0, 0.5]),
        tau_hat=-0.01,
        update=DelayedGnssUpdate(
            status=DelayedGnssUpdateStatus.ACCEPTED,
            t_fix_i_hat=0.24,
            t_applied_i=0.25,
            nis=1.5,
        ),
        innovation_ned=np.array([0.1, -0.2, 0.3]),
        innovation_covariance_ned=np.array([
            [4.0, 1.0, 2.0],
            [1.0, 5.0, 3.0],
            [2.0, 3.0, 6.0],
        ]),
        prior_position_covariance_ned=np.array([
            [7.0, 8.0, 9.0],
            [8.0, 10.0, 11.0],
            [9.0, 11.0, 12.0],
        ]),
    )


def _future_dropped_diagnostic(*, valid: bool = True) -> GnssFusionDiagnostic:
    """Return one unevaluated terminal timing-drop diagnostic."""
    return GnssFusionDiagnostic(
        fix=_fix(valid=valid),
        t_fix_g=0.5,
        t_arrival_i=0.55,
        p_gnss_ned=np.array([2.0, -3.0, 1.0]),
        tau_hat=0.02,
        update=DelayedGnssUpdate(
            status=DelayedGnssUpdateStatus.FUTURE_DROPPED,
            t_fix_i_hat=0.52,
            t_applied_i=None,
            nis=None,
        ),
        innovation_ned=None,
        innovation_covariance_ned=None,
        prior_position_covariance_ned=None,
    )


def test_gnss_fusion_diagnostic_header_is_fixed_and_has_no_duplicates():
    """One terminal-fix row has all source, timing, and evaluation columns."""
    columns = _csv_column_names()

    assert columns[:7] == (
        "gps_week",
        "gps_tow_s",
        "host_arrival_monotonic_s",
        "native_record_index",
        "satellite_count",
        "horizontal_accuracy_m",
        "vertical_accuracy_m",
    )
    assert columns[7:14] == (
        "t_fix_g_s",
        "t_arrival_i_s",
        "tau_hat_s",
        "t_fix_i_hat_s",
        "t_applied_i_s",
        "status",
        "nis",
    )
    assert columns[-12:] == (
        "S_nn_m2",
        "S_en_m2",
        "S_ee_m2",
        "S_dn_m2",
        "S_de_m2",
        "S_dd_m2",
        "P_dp_nn_m2",
        "P_dp_en_m2",
        "P_dp_ee_m2",
        "P_dp_dn_m2",
        "P_dp_de_m2",
        "P_dp_dd_m2",
    )
    assert len(columns) == len(set(columns))


def test_gnss_fusion_diagnostic_sidecar_documents_schema_and_empty_policy():
    """The sidecar fixes frame, timing, covariance, and unavailable-value meaning."""
    payload = _sidecar_payload()

    assert payload["schema_version"] == GNSS_FUSION_DIAGNOSTIC_SCHEMA_VERSION
    assert payload["artifact"] == "gnss_fusion_diagnostics"
    assert payload["navigation_frame"] == "static-origin local geographic North-East-Down"
    assert payload["evaluation_fields"]["unavailable_csv_encoding"] == "empty field"
    assert payload["covariance_encoding"]["order"] == "lower triangle including diagonal"
    assert payload["columns"] == list(_csv_column_names())


def test_writer_serializes_evaluated_and_unevaluated_diagnostics(tmp_path):
    """One CSV preserves exact evidence while leaving absent evaluation blank."""
    output_path = tmp_path / "diagnostics.csv"

    write_gnss_fusion_diagnostics_csv(
        output_path,
        (_accepted_diagnostic(), _future_dropped_diagnostic()),
    )

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    with (tmp_path / "diagnostics.meta.yaml").open(encoding="utf-8") as sidecar_file:
        sidecar = yaml.safe_load(sidecar_file)

    assert list(rows[0]) == list(_csv_column_names())
    assert rows[0]["gps_week"] == "2300"
    assert rows[0]["tau_hat_s"] == "-0.01"
    assert rows[0]["status"] == "ACCEPTED"
    assert rows[0]["innovation_e_m"] == "-0.20000000000000001"
    assert rows[0]["S_nn_m2"] == "4"
    assert rows[0]["S_en_m2"] == "1"
    assert rows[0]["S_dd_m2"] == "6"
    assert rows[0]["P_dp_en_m2"] == "8"
    assert rows[1]["status"] == "FUTURE_DROPPED"
    assert rows[1]["t_applied_i_s"] == ""
    assert rows[1]["nis"] == ""
    assert rows[1]["innovation_n_m"] == ""
    assert rows[1]["S_nn_m2"] == ""
    assert rows[1]["P_dp_dd_m2"] == ""
    assert sidecar["csv_filename"] == "diagnostics.csv"
    assert sidecar["columns"] == list(_csv_column_names())


def test_writer_supports_empty_sequences_and_rejects_invalid_source_evidence(tmp_path):
    """No valid outcomes remains exportable, but invalid-fix records do not."""
    empty_output_path = tmp_path / "empty.csv"
    write_gnss_fusion_diagnostics_csv(empty_output_path, ())

    with empty_output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows == []

    with pytest.raises(ValueError, match="receiver-valid"):
        write_gnss_fusion_diagnostics_csv(
            tmp_path / "invalid.csv",
            (_future_dropped_diagnostic(valid=False),),
        )
