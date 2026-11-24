"""Focused UBX framing and navigation-message decoding checks."""

import struct
from math import pi

import pytest

from bootins.sensors.ubx import (
    UbxFrame,
    UbxStreamFramer,
    decode_nav_pvt,
    decode_nav_timegps,
    decode_ubx_frame,
    encode_ubx_frame,
    ubx_checksum,
)


EMPTY_NAV_PVT = b"\xb5\x62\x01\x07\x00\x00\x08\x19"
PAYLOAD_NAV_PVT = b"\xb5\x62\x01\x07\x02\x00\xaa\xbb\x6f\x40"


def test_complete_frame_decode_encode_round_trip_preserves_native_bytes():
    frame = decode_ubx_frame(PAYLOAD_NAV_PVT)

    assert frame == UbxFrame(0x01, 0x07, b"\xaa\xbb")
    assert encode_ubx_frame(frame) == PAYLOAD_NAV_PVT
    assert ubx_checksum(b"\x01\x07\x00\x00") == (0x08, 0x19)


def test_complete_decoder_refuses_corruption_without_resynchronizing():
    with pytest.raises(ValueError, match="checksum"):
        decode_ubx_frame(PAYLOAD_NAV_PVT[:-1] + b"\x41")
    with pytest.raises(ValueError, match="synchronization"):
        decode_ubx_frame(b"\x00" + EMPTY_NAV_PVT[1:])


def test_stream_framer_reassembles_every_frame_boundary_and_multiframe_chunk():
    for split in (1, 2, 5, 6, 8, 9):
        framer = UbxStreamFramer()
        assert framer.feed(PAYLOAD_NAV_PVT[:split]) == ()
        assert framer.feed(PAYLOAD_NAV_PVT[split:]) == (UbxFrame(0x01, 0x07, b"\xaa\xbb"),)
        framer.finish()

    framer = UbxStreamFramer()
    assert framer.feed(EMPTY_NAV_PVT + PAYLOAD_NAV_PVT) == (
        UbxFrame(0x01, 0x07, b""), UbxFrame(0x01, 0x07, b"\xaa\xbb"),
    )
    framer.finish()


def test_stream_framer_retains_partial_evidence_until_completion_or_explicit_eof():
    framer = UbxStreamFramer()
    assert framer.feed(PAYLOAD_NAV_PVT[:6]) == ()
    assert framer.pending_byte_count == 6
    with pytest.raises(ValueError, match="incomplete"):
        framer.finish()


def test_nav_pvt_decoding_scales_position_quality_and_invalid_fix_evidence():
    payload = bytearray(92)
    struct.pack_into("<I", payload, 0, 123_456)
    payload[20], payload[21], payload[23] = 3, 0x01, 14
    struct.pack_into("<iii", payload, 24, -900_000_000, 450_000_000, 123_456)
    struct.pack_into("<II", payload, 40, 800, 1_200)

    observation = decode_nav_pvt(UbxFrame(0x01, 0x07, bytes(payload)))

    assert observation.tow == 123_456
    assert observation.fix_valid
    assert observation.longitude == pytest.approx(-pi / 2.0)
    assert observation.latitude == pytest.approx(pi / 4.0)
    assert observation.height_ellipsoid == pytest.approx(123.456)
    assert observation.horizontal_accuracy == pytest.approx(0.8)


def test_nav_timegps_preserves_signed_fraction_and_validity_evidence():
    payload = bytearray(16)
    struct.pack_into("<Iih", payload, 0, 123_456, -250_000, 2_300)
    payload[11] = 0x07

    timing = decode_nav_timegps(UbxFrame(0x01, 0x20, bytes(payload)))

    assert (timing.tow, timing.fractional_tow, timing.gps_week) == (123_456, -250_000, 2_300)
    assert timing.tow_valid and timing.week_valid and timing.leap_seconds_valid


def test_navigation_decoders_require_exact_message_identity_and_payload_layout():
    with pytest.raises(ValueError, match="expected"):
        decode_nav_pvt(UbxFrame(0x02, 0x07, b"\x00" * 92))
    with pytest.raises(ValueError, match="exactly"):
        decode_nav_timegps(UbxFrame(0x01, 0x20, b"\x00" * 15))
