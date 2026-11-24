"""Focused tests for normalized GNSS-table replay evidence."""

import csv

import pytest

from bootins.sensors.gnss_observations import GnssFixLLH
from bootins.sensors.gnss_table import load_gnss_fix_table, write_gnss_fix_table


def _fix(*, arrival: float = 417.5, valid: bool = True) -> GnssFixLLH:
    return GnssFixLLH(
        gps_week=2300, gps_tow=12345.25 + arrival, host_arrival_monotonic=arrival,
        latitude=0.7, longitude=-1.1, height_ellipsoid=123.0, fix_valid=valid,
        satellite_count=14, horizontal_accuracy=0.8, vertical_accuracy=1.2,
        native_record_index=int(arrival),
    )


def test_table_round_trip_preserves_order_quality_and_radian_coordinates(tmp_path):
    path = tmp_path / "fixes.csv"
    fixes = (_fix(valid=False), _fix(arrival=417.8))

    write_gnss_fix_table(path, fixes)
    loaded = load_gnss_fix_table(path)

    assert [fix.host_arrival_monotonic for fix in loaded] == [417.5, 417.8]
    assert not loaded[0].fix_valid
    assert loaded[0].latitude == pytest.approx(0.7)
    assert loaded[0].longitude == pytest.approx(-1.1)


def test_loader_accepts_an_empty_evidence_table(tmp_path):
    path = tmp_path / "fixes.csv"
    write_gnss_fix_table(path, ())

    assert load_gnss_fix_table(path) == ()


def test_loader_rejects_an_incompatible_schema(tmp_path):
    path = tmp_path / "fixes.csv"
    with path.open("w", newline="", encoding="utf-8") as table:
        csv.writer(table).writerow(["wrong", "header"])

    with pytest.raises(ValueError, match="header"):
        load_gnss_fix_table(path)


def test_loader_rejects_backward_host_arrival_evidence(tmp_path):
    path = tmp_path / "fixes.csv"
    with pytest.raises(ValueError, match="nondecreasing"):
        write_gnss_fix_table(path, (_fix(arrival=2.0), _fix(arrival=1.0)))
