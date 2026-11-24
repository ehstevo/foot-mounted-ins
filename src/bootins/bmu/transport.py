"""Read framed raw BMU capture evidence without decoding device payloads.

Each :class:`RawFrame` represents one recorded BLE notification: file order,
a host-monotonic arrival timestamp, and untouched payload bytes. This transport
layer does not interpret device payloads. Decoding, structural quality,
sequence integrity, and navigation remain separate so later decoder work can
operate on the original captured evidence.

Transport integrity establishes whether the capture framing is complete; it
does not establish measurement validity or scientific accuracy.
"""
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


# This framing is the exact inverse of the raw-capture record writer.
FRAME_FORMAT = "<dH"
FRAME_SIZE = struct.calcsize(FRAME_FORMAT)


@dataclass(frozen=True)
class RawFrame:
    """Represent one complete notification from a framed BMU capture.

    Attributes:
        frame_index: Zero-based deterministic record order within the capture.
        arrival_monotonic: Host-monotonic receipt timestamp (s) recorded when
            the notification was received. It is source availability evidence,
            not a device measurement time.
        payload: Original notification bytes, retained untouched and not decoded
            at this transport boundary.

    Notes:
        This frozen record preserves capture provenance. It does not establish
        payload validity, sequence continuity, or IMU measurement quality.
    """

    frame_index: int
    arrival_monotonic: float
    payload: bytes


def iter_frames_replay(path: str | Path) -> Iterator[RawFrame]:
    """Yield complete raw frames from a framed capture in file order.

    Args:
        path: Filesystem path to the raw BMU capture.

    Yields:
        Complete provenance-preserving raw frames in deterministic file order.

    Raises:
        OSError: The capture cannot be opened or read.
        ValueError: EOF occurs within a frame header or its declared payload.

    Notes:
        Clean EOF before a new frame begins ends iteration normally. A truncated
        tail is refused so incomplete transport evidence cannot appear to be a
        shorter clean capture. Payload decoding and measurement integrity occur
        in later layers.
    """
    frame_index = 0
    capture_path = Path(path)

    with capture_path.open("rb") as capture_file:
        while True:
            header = capture_file.read(FRAME_SIZE)
            if not header:
                return
            if len(header) != FRAME_SIZE:
                raise ValueError(
                    f"Truncated frame header at index {frame_index}: "
                    f"got {len(header)} B, expected {FRAME_SIZE} B."
                )

            arrival, payload_len = struct.unpack(FRAME_FORMAT, header)
            payload = capture_file.read(payload_len)
            if len(payload) != payload_len:
                raise ValueError(
                    f"Truncated payload for frame {frame_index}: "
                    f"got {len(payload)} B, expected {payload_len} B."
                )

            yield RawFrame(
                frame_index=frame_index,
                arrival_monotonic=arrival,
                payload=payload,
            )
            frame_index += 1
