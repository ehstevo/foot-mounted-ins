"""Contract tests for the source-neutral decoded-frame stream.

The stream is tested with mocked decoded frames so it proves composition with
``SequenceTracker`` without repeating private wire-layout or decoder behavior.
"""
from dataclasses import FrozenInstanceError

import pytest

from bootins.bmu import stream
from bootins.bmu.decode import DecodedMessage, MessageInfo
from bootins.bmu.provenance import DecodedFrame
from bootins.bmu.sequence import GapEvent, SequenceTracker
from bootins.bmu.transport import RawFrame


def _raw_frame(frame_index: int) -> RawFrame:
    """Return a minimal raw frame with identity-distinct provenance."""
    return RawFrame(
        frame_index=frame_index,
        arrival_monotonic=float(frame_index),
        payload=f"payload-{frame_index}".encode(),
    )


def _decoded_frame(raw_frame: RawFrame, sequence_num: int) -> DecodedFrame:
    """Return a decoder result with known message information for stream tests."""
    return DecodedFrame(
        raw_frame=raw_frame,
        decoded_message=DecodedMessage(
            message_info=MessageInfo(
                sequence_num=sequence_num,
                device_timestamp_raw=0,
            ),
            measurements=[],
        ),
    )


def test_track_raw_frame_attaches_gap_to_current_valid_frame(monkeypatch):
    """The later valid frame reports loss observed between it and its predecessor."""
    first_raw_frame = _raw_frame(0)
    current_raw_frame = _raw_frame(1)
    decoded_by_frame = {
        0: _decoded_frame(first_raw_frame, 120),
        1: _decoded_frame(current_raw_frame, 123),
    }
    monkeypatch.setattr(
        stream,
        "decode_raw_frame",
        lambda raw_frame: decoded_by_frame[raw_frame.frame_index],
    )
    tracker = SequenceTracker()

    first_tracked_frame = stream.track_raw_frame(first_raw_frame, tracker)
    current_tracked_frame = stream.track_raw_frame(current_raw_frame, tracker)

    assert first_tracked_frame.decoded_frame.raw_frame is first_raw_frame
    assert first_tracked_frame.gap_before is None
    assert current_tracked_frame.decoded_frame.raw_frame is current_raw_frame
    assert current_tracked_frame.gap_before == GapEvent(
        previous_seq_num=120,
        current_seq_num=123,
        missing_message_count=2,
    )


def test_iter_tracked_frames_accepts_any_iterable_and_handles_wrap(monkeypatch):
    """A generator source receives one fresh tracker with normal u16 wrap behavior."""
    raw_frames = [_raw_frame(0), _raw_frame(1)]
    decoded_by_frame = {
        0: _decoded_frame(raw_frames[0], 65535),
        1: _decoded_frame(raw_frames[1], 0),
    }
    monkeypatch.setattr(
        stream,
        "decode_raw_frame",
        lambda raw_frame: decoded_by_frame[raw_frame.frame_index],
    )

    tracked_frames = list(
        stream.iter_tracked_frames(raw_frame for raw_frame in raw_frames)
    )

    assert [frame.decoded_frame.raw_frame for frame in tracked_frames] == raw_frames
    assert [frame.gap_before for frame in tracked_frames] == [None, None]


def test_track_raw_frame_propagates_decoder_failure_unchanged(monkeypatch):
    """Decoder validation remains the decoder's responsibility in this layer."""
    raw_frame = _raw_frame(0)
    decoder_error = ValueError("decoder rejected payload")

    def fake_decode_raw_frame(frame: RawFrame) -> DecodedFrame:
        raise decoder_error

    monkeypatch.setattr(stream, "decode_raw_frame", fake_decode_raw_frame)

    with pytest.raises(ValueError) as raised:
        stream.track_raw_frame(raw_frame, SequenceTracker())

    assert raised.value is decoder_error


def test_tracked_frame_is_immutable():
    """A decoded frame and its associated gap cannot be reassigned."""
    raw_frame = _raw_frame(0)
    tracked_frame = stream.TrackedFrame(
        decoded_frame=_decoded_frame(raw_frame, 1),
        gap_before=None,
    )

    with pytest.raises(FrozenInstanceError):
        tracked_frame.gap_before = GapEvent(1, 3, 1)
