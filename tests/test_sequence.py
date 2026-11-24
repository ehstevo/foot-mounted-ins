"""Known-contract tests for BMU notification sequence tracking.

The tracker receives device-assigned u16 sequence numbers one at a time. These
tests pin its loss accounting, wrap behavior, and refusal to fabricate a
forward gap from duplicate or backward observations.
"""
import pytest

from bootins.bmu.sequence import GapEvent, SequenceTracker


def test_first_sequence_zero_establishes_baseline_then_advances():
    """Zero is a valid u16 sequence value, not an absent baseline sentinel."""
    tracker = SequenceTracker()

    assert tracker.update(0) is None
    assert tracker.update(1) is None


def test_contiguous_sequences_do_not_emit_gap_events():
    """An advance of one is normal packet continuity."""
    tracker = SequenceTracker()

    assert tracker.update(120) is None
    assert tracker.update(121) is None
    assert tracker.update(122) is None


def test_forward_gap_reports_exact_missing_message_count():
    """A forward jump reports the BMU messages absent between observations."""
    tracker = SequenceTracker()

    assert tracker.update(120) is None
    event = tracker.update(123)

    assert event == GapEvent(
        previous_seq_num=120,
        current_seq_num=123,
        missing_message_count=2,
    )
    assert tracker.update(124) is None


def test_sequence_wrap_is_contiguous():
    """The u16 transition from 65535 to 0 is one normal advance."""
    tracker = SequenceTracker()

    assert tracker.update(65535) is None
    assert tracker.update(0) is None


def test_gap_across_sequence_wrap_reports_missing_messages():
    """Modular arithmetic reports losses on either side of the wrap boundary."""
    tracker = SequenceTracker()

    assert tracker.update(65534) is None
    event = tracker.update(1)

    assert event == GapEvent(
        previous_seq_num=65534,
        current_seq_num=1,
        missing_message_count=2,
    )


def test_duplicate_sequence_is_rejected_without_advancing_baseline():
    """A duplicate cannot be misreported as a negative-size or huge forward gap."""
    tracker = SequenceTracker()

    assert tracker.update(8) is None
    with pytest.raises(ValueError):
        tracker.update(8)
    assert tracker.update(9) is None


@pytest.mark.parametrize("current_seq_num", [99, 32768])
def test_backward_or_ambiguous_sequence_is_rejected(current_seq_num):
    """Backward and half-range jumps have no unambiguous forward loss meaning."""
    tracker = SequenceTracker()

    assert tracker.update(100 if current_seq_num == 99 else 0) is None
    with pytest.raises(ValueError):
        tracker.update(current_seq_num)


@pytest.mark.parametrize("seq_num", [-1, 65536])
def test_out_of_range_sequence_is_rejected(seq_num):
    """Only unsigned-16-bit device sequence values are valid observations."""
    with pytest.raises(ValueError):
        SequenceTracker().update(seq_num)


def test_reset_starts_a_new_unknown_sequence_segment():
    """A known reconnect/reset discards prior continuity rather than guessing loss."""
    tracker = SequenceTracker()

    assert tracker.update(120) is None
    tracker.reset()
    assert tracker.update(3) is None
    assert tracker.update(4) is None
