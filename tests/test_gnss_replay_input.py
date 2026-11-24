"""Focused tests for factual causal GNSS and IMU input merging."""

import numpy as np
import pytest

from bootins.bmu.decode import DecodedMessage, Measurement, MessageInfo
from bootins.bmu.provenance import DecodedFrame
from bootins.bmu.stream import TrackedFrame
from bootins.bmu.transport import RawFrame
from bootins.replay import gnss_arrivals as gnss_replay_input
from bootins.bmu.sequence import GapEvent
from bootins.sensors.gnss_observations import GnssFixLLH
from bootins.replay.gnss_arrivals import iter_gnss_imu_arrivals


def _tracked_frame(*, gap_before: GapEvent | None = None) -> TrackedFrame:
    """Build one provenance-bearing tracked frame for arrival-record tests."""
    measurement = Measurement(
        dt=0.010,
        dtheta=np.array([0.1, 0.2, 0.3]),
        dv=np.array([1.0, 2.0, 3.0]),
        quat_raw=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    return TrackedFrame(
        decoded_frame=DecodedFrame(
            raw_frame=RawFrame(
                frame_index=7,
                arrival_monotonic=12.5,
                payload=b"frame payload",
            ),
            decoded_message=DecodedMessage(
                message_info=MessageInfo(
                    sequence_num=42,
                    device_timestamp_raw=123,
                ),
                measurements=[measurement],
            ),
        ),
        gap_before=gap_before,
    )


def _tracked_frame_for_raw_frame(raw_frame: RawFrame) -> TrackedFrame:
    """Return one valid decoded frame retaining the supplied raw provenance."""
    return TrackedFrame(
        decoded_frame=DecodedFrame(
            raw_frame=raw_frame,
            decoded_message=DecodedMessage(
                message_info=MessageInfo(
                    sequence_num=raw_frame.frame_index,
                    device_timestamp_raw=0,
                ),
                measurements=[],
            ),
        ),
        gap_before=None,
    )


def _raw_frame(frame_index: int, arrival_monotonic: float) -> RawFrame:
    """Build one raw-frame source event with controlled host timing."""
    return RawFrame(
        frame_index=frame_index,
        arrival_monotonic=arrival_monotonic,
        payload=f"raw frame {frame_index}".encode(),
    )


def _gnss_fix(host_arrival_monotonic: float, native_record_index: int) -> GnssFixLLH:
    """Build one receiver-valid GNSS event with controlled host timing."""
    return GnssFixLLH(
        gps_week=0,
        gps_tow=float(native_record_index),
        host_arrival_monotonic=host_arrival_monotonic,
        latitude=0.0,
        longitude=0.0,
        height_ellipsoid=0.0,
        fix_valid=True,
        satellite_count=8,
        horizontal_accuracy=1.0,
        vertical_accuracy=1.5,
        native_record_index=native_record_index,
    )


def test_arrival_merge_preserves_gap_evidence_for_runner_policy(monkeypatch):
    """A tracked frame retains gap evidence without a merge-layer refusal."""
    raw_frame = _raw_frame(7, 2.0)
    gap_event = GapEvent(40, 42, 1)
    tracked_frame = _tracked_frame(gap_before=gap_event)

    def fake_track_raw_frame(
        observed_raw_frame: RawFrame,
        _tracker: object,
    ) -> TrackedFrame:
        assert observed_raw_frame is raw_frame
        return tracked_frame

    monkeypatch.setattr(gnss_replay_input, "track_raw_frame", fake_track_raw_frame)

    events = list(iter_gnss_imu_arrivals([raw_frame], []))

    assert len(events) == 1
    assert events[0] is tracked_frame
    assert events[0].gap_before is gap_event


def test_arrival_merge_defers_structural_quality_policy_to_runner(monkeypatch):
    """A structurally concerning frame remains factual input for the runner."""
    raw_frame = _raw_frame(7, 2.0)
    tracked_frame = _tracked_frame()
    tracked_frame.decoded_frame.decoded_message.measurements[0] = Measurement(
        dt=np.nan,
        dtheta=np.array([0.1, 0.2, 0.3]),
        dv=np.array([1.0, 2.0, 3.0]),
        quat_raw=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    monkeypatch.setattr(
        gnss_replay_input,
        "track_raw_frame",
        lambda _raw_frame, _tracker: tracked_frame,
    )

    events = list(iter_gnss_imu_arrivals([raw_frame], []))

    assert len(events) == 1
    assert events[0] is tracked_frame


def test_arrival_merge_is_causal_and_releases_gnss_first_at_equal_arrival(
    monkeypatch,
):
    """The merge holds one raw lookahead and resolves equal arrivals GNSS first."""
    first_raw_frame = _raw_frame(7, 2.0)
    second_raw_frame = _raw_frame(8, 4.0)
    first_fix = _gnss_fix(1.0, 1)
    second_fix = _gnss_fix(3.0, 2)
    equal_arrival_fix = _gnss_fix(4.0, 3)
    trailing_fix = _gnss_fix(5.0, 4)
    tracked_raw_frames: list[RawFrame] = []
    tracker_ids: list[int] = []
    tracked_frames: list[TrackedFrame] = []

    def fake_track_raw_frame(raw_frame: RawFrame, tracker: object) -> TrackedFrame:
        tracked_raw_frames.append(raw_frame)
        tracker_ids.append(id(tracker))
        tracked_frame = _tracked_frame_for_raw_frame(raw_frame)
        tracked_frames.append(tracked_frame)
        return tracked_frame

    monkeypatch.setattr(gnss_replay_input, "track_raw_frame", fake_track_raw_frame)

    arrivals = iter_gnss_imu_arrivals(
        [first_raw_frame, second_raw_frame],
        [first_fix, second_fix, equal_arrival_fix, trailing_fix],
    )

    assert next(arrivals) is first_fix
    assert tracked_raw_frames == []

    first_tracked_frame = next(arrivals)
    assert first_tracked_frame is tracked_frames[0]
    assert first_tracked_frame.decoded_frame.raw_frame is first_raw_frame
    assert tracked_raw_frames == [first_raw_frame]

    assert next(arrivals) is second_fix
    assert next(arrivals) is equal_arrival_fix
    assert tracked_raw_frames == [first_raw_frame]

    second_tracked_frame = next(arrivals)
    assert second_tracked_frame is tracked_frames[1]
    assert second_tracked_frame.decoded_frame.raw_frame is second_raw_frame
    assert tracked_raw_frames == [first_raw_frame, second_raw_frame]
    assert tracker_ids[0] == tracker_ids[1]

    assert next(arrivals) is trailing_fix
    with pytest.raises(StopIteration):
        next(arrivals)


def test_arrival_merge_accepts_one_source_and_rejects_backward_gnss_order(
    monkeypatch,
):
    """One-source streams work, while GNSS host-time reversals remain strict."""
    raw_frame = _raw_frame(7, 2.0)
    tracked_raw_frames: list[RawFrame] = []

    def fake_track_raw_frame(observed_raw_frame: RawFrame, _tracker: object) -> TrackedFrame:
        tracked_raw_frames.append(observed_raw_frame)
        return _tracked_frame_for_raw_frame(observed_raw_frame)

    monkeypatch.setattr(gnss_replay_input, "track_raw_frame", fake_track_raw_frame)

    raw_only_events = list(iter_gnss_imu_arrivals([raw_frame], []))

    assert len(raw_only_events) == 1
    assert isinstance(raw_only_events[0], TrackedFrame)
    assert tracked_raw_frames == [raw_frame]

    earlier_fix = _gnss_fix(2.0, 1)
    later_fix = _gnss_fix(1.0, 2)
    gnss_only_events = iter_gnss_imu_arrivals([], [earlier_fix, later_fix])

    assert next(gnss_only_events) is earlier_fix
    with pytest.raises(ValueError, match="nondecreasing host-arrival order"):
        next(gnss_only_events)
