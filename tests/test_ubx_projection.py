"""Tests for source-specific UBX archive projection into normalized fixes."""
import struct
from math import pi

import pytest

from bootins.sensors.ubx import UbxFrame, encode_ubx_frame
from bootins.sensors.ubx_archive import UbxCaptureRecord
from bootins.sensors.ubx_projection import project_ubx_records


def _pvt_record(
    record_index: int,
    arrival: float,
    tow: int,
    *,
    fix_valid: bool = True,
) -> UbxCaptureRecord:
    """Build one valid archived NAV-PVT record with distinctive position fields."""
    payload = bytearray(92)
    struct.pack_into("<I", payload, 0, tow)
    payload[20] = 3
    payload[21] = 0x01 if fix_valid else 0x00
    payload[23] = 12
    struct.pack_into("<iii", payload, 24, -900_000_000, 450_000_000, 123_000)
    struct.pack_into("<II", payload, 40, 800, 1_200)
    return UbxCaptureRecord(
        record_index=record_index,
        host_arrival_monotonic=arrival,
        raw_frame=encode_ubx_frame(UbxFrame(0x01, 0x07, bytes(payload))),
    )


def _timegps_record(
    record_index: int,
    arrival: float,
    tow: int,
    *,
    fractional_tow: int = 0,
    valid: int = 0x03,
) -> UbxCaptureRecord:
    """Build one archived NAV-TIMEGPS record with caller-controlled validity."""
    payload = bytearray(16)
    struct.pack_into("<Iih", payload, 0, tow, fractional_tow, 2300)
    payload[11] = valid
    return UbxCaptureRecord(
        record_index=record_index,
        host_arrival_monotonic=arrival,
        raw_frame=encode_ubx_frame(UbxFrame(0x01, 0x20, bytes(payload))),
    )


def test_projection_pairs_same_tow_evidence_and_retains_pvt_host_arrival():
    """A valid same-iTOW pair yields one receiver-independent normalized fix."""
    projection = project_ubx_records(
        (
            _timegps_record(0, 10.0, 12_345, fractional_tow=-250_000),
            _pvt_record(1, 10.1, 12_345),
        )
    )

    assert projection.nav_pvt_count == 1
    assert projection.nav_timegps_count == 1
    assert projection.unpaired_pvt_count == 0
    assert projection.invalid_time_pvt_count == 0
    assert len(projection.fixes) == 1
    fix = projection.fixes[0]
    assert fix.gps_week == 2300
    assert fix.gps_tow == pytest.approx(12.34475)
    assert fix.host_arrival_monotonic == pytest.approx(10.1)
    assert fix.native_record_index == 1
    assert fix.fix_valid
    assert fix.satellite_count == 12
    assert fix.longitude == pytest.approx(-pi / 2.0)
    assert fix.latitude == pytest.approx(pi / 4.0)


def test_projection_pairs_pvt_that_precedes_its_matching_timegps_record():
    """Archive order need not require TIMEGPS to arrive before its PVT message."""
    projection = project_ubx_records(
        (
            _pvt_record(0, 10.0, 12_345, fix_valid=False),
            _timegps_record(1, 10.1, 12_345),
        )
    )

    assert len(projection.fixes) == 1
    assert projection.fixes[0].native_record_index == 0
    assert projection.fixes[0].host_arrival_monotonic == pytest.approx(10.0)
    assert not projection.fixes[0].fix_valid


def test_projection_reports_missing_and_invalid_time_evidence_without_fabricating_fixes():
    """PVT data without usable GPS-clock evidence remains visible in diagnostics."""
    projection = project_ubx_records(
        (
            _pvt_record(0, 10.0, 1_000),
            _pvt_record(1, 11.0, 2_000),
            _timegps_record(2, 11.1, 2_000, valid=0x01),
        )
    )

    assert projection.fixes == ()
    assert projection.nav_pvt_count == 2
    assert projection.nav_timegps_count == 1
    assert projection.unpaired_pvt_count == 1
    assert projection.invalid_time_pvt_count == 1


def test_projection_ignores_unknown_valid_ubx_messages_but_rejects_corruption():
    """Unknown evidence is retained in archive while malformed evidence is strict."""
    unknown = UbxCaptureRecord(
        record_index=0,
        host_arrival_monotonic=1.0,
        raw_frame=encode_ubx_frame(UbxFrame(0x06, 0x01, b"\x00")),
    )

    assert project_ubx_records((unknown,)).fixes == ()

    corrupted = UbxCaptureRecord(
        record_index=1,
        host_arrival_monotonic=2.0,
        raw_frame=b"\xb5\x62\x01\x07\x00\x00\x08\x18",
    )
    with pytest.raises(ValueError, match="record 1.*checksum"):
        project_ubx_records((corrupted,))
