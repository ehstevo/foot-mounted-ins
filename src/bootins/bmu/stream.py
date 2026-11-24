"""Compose raw-frame decoding with causal BMU sequence-loss tracking.

This source-neutral layer turns each recorded or live-supplied
:class:`~bootins.bmu.transport.RawFrame` into a decoded
:class:`TrackedFrame`. The revealing valid frame carries any preceding forward
gap evidence because its sequence number first establishes the loss. It does
not decide navigation recovery, structural quality, or source connection
policy.

The iterable helper owns one tracker for a finite offline segment. A stateful
live caller instead retains its own tracker and invokes :func:`track_raw_frame`
once per arriving frame, including after any explicit sequence reset.
"""
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from bootins.bmu.provenance import DecodedFrame, decode_raw_frame
from bootins.bmu.sequence import GapEvent, SequenceTracker
from bootins.bmu.transport import RawFrame


@dataclass(frozen=True)
class TrackedFrame:
    """Combine one decoded frame with loss evidence revealed by its arrival.

    Attributes:
        decoded_frame: Raw-frame provenance and decoded semantic measurement
            evidence for the current valid notification.
        gap_before: Unambiguous forward-loss evidence preceding this frame, or
            ``None`` when the sequence is contiguous or begins a new segment.

    Notes:
        ``gap_before`` belongs to this revealing frame because its device
        sequence number first establishes that notifications were absent. The
        current frame remains factual usable evidence; later replay policy
        determines whether and how to handle the discontinuity.
    """

    decoded_frame: DecodedFrame
    gap_before: GapEvent | None


def track_raw_frame(raw_frame: RawFrame, tracker: SequenceTracker) -> TrackedFrame:
    """Decode one raw frame and attach any preceding loss it reveals.

    Args:
        raw_frame: Complete raw transport record in source-arrival order.
        tracker: Stateful sequence tracker for the current continuous source
            segment.

    Returns:
        Decoded factual frame with any unambiguous preceding ``GapEvent``.

    Raises:
        RuntimeError: Private BMU decoder definitions are unavailable.
        ValueError: Payload decoding or sequence validation rejects the frame.

    Notes:
        The caller owns tracker lifetime, allowing a live source to preserve
        continuity across callbacks or reset at a known reconnect.
    """
    decoded_frame = decode_raw_frame(raw_frame)
    seq_num = decoded_frame.decoded_message.message_info.sequence_num
    gap_before = tracker.update(seq_num)
    return TrackedFrame(
        decoded_frame=decoded_frame,
        gap_before=gap_before,
    )


def iter_tracked_frames(raw_frames: Iterable[RawFrame]) -> Iterator[TrackedFrame]:
    """Yield tracked frames from one finite synchronous raw-frame iterable.

    Args:
        raw_frames: Raw transport records in observed source order.

    Yields:
        Decoded factual frames with any gap evidence attached to their revealing
        frames.

    Raises:
        RuntimeError: Private decoder definitions are unavailable.
        ValueError: Payload decoding or sequence validation rejects a frame.

    Notes:
        Each call creates a new sequence tracker and therefore represents one
        offline segment. A live caller must retain a tracker and call
        :func:`track_raw_frame` for each arriving notification.
    """
    seq_tracker = SequenceTracker()

    for raw_frame in raw_frames:
        yield track_raw_frame(raw_frame, seq_tracker)

    
