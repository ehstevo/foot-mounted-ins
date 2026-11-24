"""Detect missing BMU notifications from device-assigned u16 sequence numbers.

The tracker accepts sequence numbers that have already been extracted from
valid BMU headers. It neither parses raw payloads nor repairs missing IMU
increments; it records the loss evidence needed by a later replay or live
runner to handle the resulting navigation discontinuity honestly.
"""
from dataclasses import dataclass


# BMU SeqNum is an unsigned 16-bit counter. Half the range is the largest
# unambiguous forward advance: a larger modular change could be backward.
SEQUENCE_MODULUS = 65536
HALF_SEQUENCE_RANGE = SEQUENCE_MODULUS // 2


@dataclass(frozen=True)
class GapEvent:
    """Evidence that complete BMU notifications were absent between two headers.

    Attributes:
        previous_seq_num: Last accepted unsigned 16-bit BMU sequence number
            before the observed gap.
        current_seq_num: First accepted unsigned 16-bit sequence number after
            the gap, including possible modulo wraparound.
        missing_message_count: Number of absent notification sequence values
            implied by the unambiguous forward modular advance.

    Notes:
        This is loss evidence, not reconstructed IMU data or a navigation
        recovery decision.
    """

    previous_seq_num: int
    current_seq_num: int
    missing_message_count: int


class SequenceTracker:
    """Track one continuous, ordered BMU sequence-number segment.

    The first observation establishes a baseline. Later observations either
    advance contiguously, produce a ``GapEvent``, or raise ``ValueError`` when
    their ordering has no unambiguous forward-loss interpretation. A known
    disconnect or BMU reset must call ``reset()`` before processing the new
    segment, rather than inventing a gap across unrelated counters.

    Attributes:
        _last_seq_num: Last accepted unsigned 16-bit sequence number, or
            ``None`` before a segment begins or after ``reset``.
    """

    def __init__(self):
        """Initialize a tracker with no established sequence segment."""
        self._last_seq_num: int | None = None

    def update(self, seq_num: int) -> GapEvent | None:
        """Consume one sequence number and report unambiguous forward loss.

        Args:
            seq_num: Current unsigned 16-bit device sequence number.

        Returns:
            ``None`` for the first or a contiguous observation; otherwise a
            ``GapEvent`` describing the missing forward notification count.

        Raises:
            ValueError: The value is outside the unsigned 16-bit range, is a
                duplicate, is a half-range ambiguity, or is backward/out of
                order relative to the established segment.

        Notes:
            Modular subtraction treats ``65535 -> 0`` as contiguous. Advances
            larger than half the unsigned range have no unambiguous forward-loss
            interpretation and are refused rather than misreported as loss.
        """
        if not 0 <= seq_num < SEQUENCE_MODULUS:
            raise ValueError(
                f"Sequence number {seq_num} is outside u16 range "
                f"[0, {SEQUENCE_MODULUS})."
            )

        if self._last_seq_num is None:
            self._last_seq_num = seq_num
            return None

        advance = (seq_num - self._last_seq_num) % SEQUENCE_MODULUS
        if advance == 1:
            self._last_seq_num = seq_num
            return None
        if advance == 0:
            raise ValueError(f"Duplicate BMU sequence number {seq_num}.")
        if advance == HALF_SEQUENCE_RANGE:
            raise ValueError(
                f"Ambiguous half-range BMU sequence advance from "
                f"{self._last_seq_num} to {seq_num}."
            )
        if advance > HALF_SEQUENCE_RANGE:
            raise ValueError(
                f"Backward or out-of-order BMU sequence observation: "
                f"{self._last_seq_num} to {seq_num}."
            )

        gap_event = GapEvent(
            previous_seq_num=self._last_seq_num,
            current_seq_num=seq_num,
            missing_message_count=advance - 1,
        )
        self._last_seq_num = seq_num
        return gap_event

    def reset(self) -> None:
        """Discard continuity before a known disconnect or device reset.

        The next sequence number starts a new segment. No gap is inferred across
        the explicitly known discontinuity.
        """
        self._last_seq_num = None
