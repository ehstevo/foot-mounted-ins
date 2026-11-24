"""Contract tests for the RawFrame-to-DecodedMessage provenance bridge.

The bridge intentionally knows no private BMU wire layout. Decoder behavior is
mocked here so these tests prove object retention and error propagation without
duplicating the decoder's protocol tests.
"""
from dataclasses import FrozenInstanceError

import pytest

from bootins.bmu import provenance
from bootins.bmu.decode import DecodedMessage, MessageInfo
from bootins.bmu.transport import RawFrame


def _decoded_message() -> DecodedMessage:
    """Return a minimal approved decoder result for bridge-only tests."""
    return DecodedMessage(
        message_info=MessageInfo(
            sequence_num=12,
            device_timestamp_raw=34,
        ),
        measurements=[],
    )


def test_decode_raw_frame_retains_input_and_decoder_result(monkeypatch):
    """The bridge forwards raw bytes once and retains both source objects.

    Identity assertions matter here: copying either object could break the
    direct audit path from a decoded batch back to its captured notification.
    """
    raw_frame = RawFrame(
        frame_index=7,
        arrival_monotonic=12.5,
        payload=b"arbitrary raw payload",
    )
    decoded_message = _decoded_message()
    seen_payloads = []

    def fake_decode_message(payload: bytes) -> DecodedMessage:
        seen_payloads.append(payload)
        return decoded_message

    monkeypatch.setattr(provenance, "decode_message", fake_decode_message)

    decoded_frame = provenance.decode_raw_frame(raw_frame)

    assert seen_payloads == [raw_frame.payload]
    assert decoded_frame.raw_frame is raw_frame
    assert decoded_frame.decoded_message is decoded_message


def test_decode_raw_frame_propagates_decoder_failure_unchanged(monkeypatch):
    """Malformed-payload handling remains the decoder's responsibility."""
    raw_frame = RawFrame(
        frame_index=0,
        arrival_monotonic=1.0,
        payload=b"bad payload",
    )
    decoder_error = ValueError("decoder rejected payload")

    def fake_decode_message(payload: bytes) -> DecodedMessage:
        raise decoder_error

    monkeypatch.setattr(provenance, "decode_message", fake_decode_message)

    with pytest.raises(ValueError) as raised:
        provenance.decode_raw_frame(raw_frame)

    assert raised.value is decoder_error


def test_decoded_frame_is_immutable():
    """A frame's raw/decoded provenance association cannot be reassigned."""
    decoded_frame = provenance.DecodedFrame(
        raw_frame=RawFrame(
            frame_index=0,
            arrival_monotonic=1.0,
            payload=b"payload",
        ),
        decoded_message=_decoded_message(),
    )

    with pytest.raises(FrozenInstanceError):
        decoded_frame.raw_frame = RawFrame(
            frame_index=1,
            arrival_monotonic=2.0,
            payload=b"replacement",
        )
