"""Tests for strict and opt-in clean-increment capture boundaries."""
import numpy as np
import pytest

from bootins.replay import imu_integrity as ins_replay_input
from bootins.bmu.decode import DecodedMessage, Measurement, MessageInfo
from bootins.bmu.provenance import DecodedFrame
from bootins.bmu.sequence import GapEvent
from bootins.bmu.stream import TrackedFrame
from bootins.bmu.transport import RawFrame


def _measurement(identifier: float, dt: float = 0.01) -> Measurement:
    """Return one valid measurement whose dtheta x-value identifies its source."""
    return Measurement(
        dt=dt,
        dtheta=np.array([identifier, 0.0, 0.0]),
        dv=np.array([0.0, identifier, 0.0]),
        quat_raw=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def _tracked_frame(
    frame_index: int,
    sequence_num: int,
    measurements: list[Measurement],
    *,
    gap_before: GapEvent | None = None,
) -> TrackedFrame:
    """Return one tracked frame with controlled decoded measurements."""
    raw_frame = RawFrame(
        frame_index=frame_index,
        arrival_monotonic=float(frame_index),
        payload=b"test payload",
    )
    return TrackedFrame(
        decoded_frame=DecodedFrame(
            raw_frame=raw_frame,
            decoded_message=DecodedMessage(
                message_info=MessageInfo(
                    sequence_num=sequence_num,
                    device_timestamp_raw=0,
                ),
                measurements=measurements,
            ),
        ),
        gap_before=gap_before,
    )


def _mock_tracked_frames(monkeypatch, tracked_frames: list[TrackedFrame]) -> object:
    """Make the selector receive one prescribed tracked-frame stream."""
    raw_frames = object()

    def fake_iter_tracked_frames(observed_raw_frames):
        assert observed_raw_frames is raw_frames
        return iter(tracked_frames)

    monkeypatch.setattr(
        ins_replay_input,
        "iter_tracked_frames",
        fake_iter_tracked_frames,
    )
    return raw_frames


def _increment_identifiers(segment: ins_replay_input.CleanIncrementSegment) -> list[float]:
    """Return the unique test identifiers from a selected increment sequence."""
    return [float(increment[0][0]) for increment in segment.increments]


def test_selector_returns_complete_clean_capture_with_actual_provenance(monkeypatch):
    """A clean source remains one segment even when its frame indices are nonzero."""
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [
            _tracked_frame(14, 30, [_measurement(1.0), _measurement(2.0)]),
            _tracked_frame(15, 31, [_measurement(3.0)]),
        ],
    )

    segment = ins_replay_input.select_longest_clean_increment_segment(raw_frames)

    assert _increment_identifiers(segment) == [1.0, 2.0, 3.0]
    assert segment.frame_start_index == 14
    assert segment.measurement_start_index == 0
    assert segment.frame_end_index == 15
    assert segment.measurement_end_index == 0


def test_selector_begins_a_new_run_at_current_valid_frame_after_gap(monkeypatch):
    """The gap-revealing frame is usable evidence for the next candidate run."""
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [
            _tracked_frame(10, 40, [_measurement(1.0), _measurement(2.0)]),
            _tracked_frame(
                13,
                43,
                [_measurement(3.0)],
                gap_before=GapEvent(40, 43, 2),
            ),
            _tracked_frame(14, 44, [_measurement(4.0), _measurement(5.0)]),
        ],
    )

    segment = ins_replay_input.select_longest_clean_increment_segment(raw_frames)

    assert _increment_identifiers(segment) == [3.0, 4.0, 5.0]
    assert (segment.frame_start_index, segment.measurement_start_index) == (13, 0)
    assert (segment.frame_end_index, segment.measurement_end_index) == (14, 1)


def test_selector_splits_within_one_frame_and_excludes_flagged_measurement(monkeypatch):
    """A bad measurement does not contaminate either adjacent clean run."""
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [
            _tracked_frame(
                20,
                60,
                [
                    _measurement(1.0),
                    _measurement(2.0, dt=np.nan),
                    _measurement(3.0),
                    _measurement(4.0),
                ],
            ),
        ],
    )

    segment = ins_replay_input.select_longest_clean_increment_segment(raw_frames)

    assert _increment_identifiers(segment) == [3.0, 4.0]
    assert (segment.frame_start_index, segment.measurement_start_index) == (20, 2)
    assert (segment.frame_end_index, segment.measurement_end_index) == (20, 3)


def test_selector_retains_first_run_when_longest_runs_tie(monkeypatch):
    """Strictly-greater replacement makes a tie deterministic and auditable."""
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [
            _tracked_frame(100, 80, [_measurement(1.0), _measurement(2.0)]),
            _tracked_frame(
                104,
                84,
                [_measurement(3.0), _measurement(4.0)],
                gap_before=GapEvent(80, 84, 3),
            ),
        ],
    )

    segment = ins_replay_input.select_longest_clean_increment_segment(raw_frames)

    assert _increment_identifiers(segment) == [1.0, 2.0]
    assert (segment.frame_start_index, segment.measurement_start_index) == (100, 0)
    assert (segment.frame_end_index, segment.measurement_end_index) == (100, 1)


def test_selector_rejects_capture_without_a_clean_measurement(monkeypatch):
    """Empty or fully flagged evidence cannot become a synthetic segment."""
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [_tracked_frame(2, 10, [_measurement(1.0, dt=np.nan)])],
    )

    with pytest.raises(ValueError, match="No clean increment segment"):
        ins_replay_input.select_longest_clean_increment_segment(raw_frames)


def test_selector_propagates_strict_source_errors_unchanged(monkeypatch):
    """Untrusted framing/decoder/sequence evidence is not silently segmented."""
    raw_frames = object()
    source_error = ValueError("duplicate sequence number")

    def raise_source_error(_raw_frames):
        raise source_error

    monkeypatch.setattr(ins_replay_input, "iter_tracked_frames", raise_source_error)

    with pytest.raises(ValueError) as raised:
        ins_replay_input.select_longest_clean_increment_segment(raw_frames)

    assert raised.value is source_error
