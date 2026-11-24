"""Focused tests for immutable normalized GNSS observation evidence."""

from dataclasses import FrozenInstanceError

import pytest

from bootins.sensors.gnss_observations import GnssFixLLH


def _fix(**overrides: object) -> GnssFixLLH:
    fields: dict[str, object] = {
        "gps_week": 2300, "gps_tow": 12345.25, "host_arrival_monotonic": 417.5,
        "latitude": 0.7, "longitude": -1.1, "height_ellipsoid": 123.0,
        "fix_valid": True, "satellite_count": 14, "horizontal_accuracy": 0.8,
        "vertical_accuracy": 1.2, "native_record_index": 42,
    }
    fields.update(overrides)
    return GnssFixLLH(**fields)


def test_fix_preserves_raw_clock_and_receiver_evidence():
    fix = _fix(fix_valid=False, satellite_count=9, horizontal_accuracy=2.5, native_record_index=11)

    assert fix.t_fix_g == pytest.approx(2300 * 604800.0 + 12345.25)
    assert not fix.fix_valid
    assert fix.satellite_count == 9
    assert fix.horizontal_accuracy == pytest.approx(2.5)
    assert fix.native_record_index == 11


def test_fix_is_immutable_after_normalization():
    with pytest.raises(FrozenInstanceError):
        _fix().gps_tow = 5.0


def test_fix_rejects_invalid_clock_or_coordinate_evidence():
    with pytest.raises(ValueError, match="gps_tow"):
        _fix(gps_tow=604800.0)
    with pytest.raises(ValueError, match="latitude"):
        _fix(latitude=float("inf"))


def test_fix_rejects_invalid_quality_or_archive_evidence():
    with pytest.raises(ValueError, match="fix_valid"):
        _fix(fix_valid=1)
    with pytest.raises(ValueError, match="native_record_index"):
        _fix(native_record_index=-1)
