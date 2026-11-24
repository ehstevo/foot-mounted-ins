"""Known-contract tests for framed BMU capture replay.

The packer below uses the independently documented ``<dH`` capture framing,
not the production module's constants. That prevents a shared framing mistake
from making both the reader and its tests agree incorrectly.
"""
from dataclasses import FrozenInstanceError
import struct

import pytest

from bootins.bmu.transport import RawFrame, iter_frames_replay


_CAPTURE_FRAME_FORMAT = "<dH"
_CAPTURE_FRAME_SIZE = struct.calcsize(_CAPTURE_FRAME_FORMAT)


def _pack_frame(arrival_monotonic_s: float, payload: bytes) -> bytes:
    """Build one complete raw capture record from known transport values."""
    return (
        struct.pack(_CAPTURE_FRAME_FORMAT, arrival_monotonic_s, len(payload))
        + payload
    )


def test_iter_frames_replay_preserves_complete_frames(tmp_path):
    """Complete records retain their arrival provenance, order, and raw bytes.

    The deliberately arbitrary payloads prove this layer does not impose IMU
    message semantics or invoke the decoder.
    """
    first_payload = b"\x00\xffnot a decoded IMU message"
    second_payload = b"\x10\x20\x30"
    capture_path = tmp_path / "two_frames.bmu"
    capture_path.write_bytes(
        _pack_frame(17.125, first_payload)
        + _pack_frame(17.245, second_payload)
    )

    frames = list(iter_frames_replay(capture_path))

    assert len(frames) == 2
    assert all(isinstance(frame, RawFrame) for frame in frames)
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert [frame.arrival_monotonic for frame in frames] == pytest.approx(
        [17.125, 17.245]
    )
    assert [frame.payload for frame in frames] == [first_payload, second_payload]


def test_iter_frames_replay_empty_file_yields_no_frames(tmp_path):
    """A clean empty capture is a valid empty stream, not a framing failure."""
    capture_path = tmp_path / "empty.bmu"
    capture_path.write_bytes(b"")

    assert list(iter_frames_replay(capture_path)) == []


def test_iter_frames_replay_rejects_truncated_header(tmp_path):
    """EOF partway through a record header must not look like clean EOF."""
    capture_path = tmp_path / "truncated_header.bmu"
    capture_path.write_bytes(b"\x00" * (_CAPTURE_FRAME_SIZE - 1))

    with pytest.raises(ValueError, match="header"):
        list(iter_frames_replay(capture_path))


def test_iter_frames_replay_rejects_truncated_payload(tmp_path):
    """A header's declared payload length must be present in full at EOF."""
    capture_path = tmp_path / "truncated_payload.bmu"
    declared_payload_size = 5
    capture_path.write_bytes(
        struct.pack(_CAPTURE_FRAME_FORMAT, 3.5, declared_payload_size)
        + b"\xaa\xbb"
    )

    with pytest.raises(ValueError, match="payload"):
        list(iter_frames_replay(capture_path))


def test_raw_frame_is_immutable():
    """Captured provenance cannot be accidentally changed after it is yielded."""
    frame = RawFrame(
        frame_index=0,
        arrival_monotonic=1.0,
        payload=b"original payload",
    )

    with pytest.raises(FrozenInstanceError):
        frame.payload = b"mutated payload"
