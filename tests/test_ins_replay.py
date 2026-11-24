"""Contract tests for causal raw INS replay over tracked BMU frames.

The runner is tested at the tracked-frame boundary, not with synthetic BMU
payload bytes. Wire decoding, frame provenance, and sequence tracking already
have their own focused tests; these cases pin only the replay policy that
combines their approved outputs.
"""
import numpy as np
import pytest

from bootins.replay import raw_ins as ins_replay
from bootins.bmu.decode import DecodedMessage, Measurement, MessageInfo
from bootins.bmu.provenance import DecodedFrame
from bootins.bmu.sequence import GapEvent
from bootins.bmu.stream import TrackedFrame
from bootins.bmu.transport import RawFrame
from bootins.estimation.mechanization import G_NED


G = np.linalg.norm(G_NED)


def _measurement(
    dt: float,
    *,
    dtheta: np.ndarray | None = None,
    dv: np.ndarray | None = None,
    quat_raw: np.ndarray | None = None,
) -> Measurement:
    """Return one decoded measurement with optional targeted defects."""
    return Measurement(
        dt=dt,
        dtheta=np.zeros(3) if dtheta is None else np.asarray(dtheta, dtype=float),
        dv=np.array([0.0, 0.0, G * dt]) if dv is None else np.asarray(dv, dtype=float),
        quat_raw=(
            np.array([1.0, 0.0, 0.0, 0.0])
            if quat_raw is None
            else np.asarray(quat_raw, dtype=float)
        ),
    )


def _tracked_frame(
    frame_index: int,
    sequence_num: int,
    measurements: list[Measurement],
    *,
    gap_before: GapEvent | None = None,
) -> TrackedFrame:
    """Return one tracked frame with explicit capture and decoder provenance."""
    raw_frame = RawFrame(
        frame_index=frame_index,
        arrival_monotonic=float(frame_index),
        payload=f"payload-{frame_index}".encode(),
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


def _mock_tracked_frames(
    monkeypatch: pytest.MonkeyPatch,
    tracked_frames: list[TrackedFrame],
) -> list[RawFrame]:
    """Make the runner receive the prescribed already-tracked frame sequence."""
    raw_frames = [frame.decoded_frame.raw_frame for frame in tracked_frames]

    def fake_iter_tracked_frames(observed_raw_frames):
        assert observed_raw_frames is raw_frames
        return iter(tracked_frames)

    monkeypatch.setattr(
        ins_replay,
        "iter_tracked_frames",
        fake_iter_tracked_frames,
    )
    return raw_frames


def test_raw_ins_replay_starts_post_initialization_and_stays_at_rest(monkeypatch):
    """Static increments cancel gravity and return N + 1 aligned states."""
    tracked_frames = [
        _tracked_frame(0, 10, [_measurement(0.1), _measurement(0.1)]),
        _tracked_frame(1, 11, [_measurement(0.1), _measurement(0.1)]),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    result = ins_replay.run_raw_ins_replay(
        raw_frames,
        initialization_duration=0.2,
    )

    assert result.initialization.increment_count == 2
    np.testing.assert_allclose(result.initialization.duration, 0.2, atol=1e-12)
    np.testing.assert_allclose(result.time, [0.0, 0.1, 0.2], atol=1e-12)
    assert result.positions_n.shape == (3, 3)
    assert result.velocities_n.shape == (3, 3)
    assert result.quaternions_nb.shape == (3, 4)
    np.testing.assert_allclose(result.positions_n, np.zeros((3, 3)), atol=1e-12)
    np.testing.assert_allclose(result.velocities_n, np.zeros((3, 3)), atol=1e-12)
    assert result.coast_plans == ()


def test_raw_ins_replay_keeps_the_boundary_increment_wholly_in_initialization(
    monkeypatch,
):
    """A preintegrated increment crossing the duration is never split."""
    tracked_frames = [
        _tracked_frame(0, 10, [_measurement(0.1), _measurement(0.3)]),
        _tracked_frame(1, 11, [_measurement(0.2)]),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    result = ins_replay.run_raw_ins_replay(
        raw_frames,
        initialization_duration=0.25,
    )

    assert result.initialization.increment_count == 2
    np.testing.assert_allclose(result.initialization.duration, 0.4, atol=1e-12)
    np.testing.assert_allclose(result.time, [0.0, 0.2], atol=1e-12)


def test_raw_ins_replay_refuses_a_gap_beyond_the_bounded_policy(monkeypatch):
    """A multi-notification gap cannot be silently coasted."""
    tracked_frames = [
        _tracked_frame(0, 10, [_measurement(0.1)]),
        _tracked_frame(
            4,
            14,
            [_measurement(0.1)],
            gap_before=GapEvent(
                previous_seq_num=10,
                current_seq_num=14,
                missing_message_count=3,
            ),
        ),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    with pytest.raises(
        ValueError,
        match=r"exceeds the bounded recovery policy: 3 missing notification\(s\), limit 1",
    ):
        ins_replay.run_raw_ins_replay(raw_frames, initialization_duration=0.1)


def test_raw_ins_replay_coasts_one_post_initialization_notification_gap(monkeypatch):
    """A permitted gap contributes one synthetic posterior before real data resumes."""
    tracked_frames = [
        _tracked_frame(0, 10, [_measurement(0.1)]),
        _tracked_frame(
            2,
            12,
            [_measurement(0.1)],
            gap_before=GapEvent(10, 12, 1),
        ),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    result = ins_replay.run_raw_ins_replay(
        raw_frames,
        initialization_duration=0.1,
    )

    assert len(result.coast_plans) == 1
    assert result.coast_plans[0].reference_frame_index == 0
    assert result.coast_plans[0].revealing_frame_index == 2
    np.testing.assert_allclose(result.time, [0.0, 0.1, 0.2], atol=1e-12)
    assert result.positions_n.shape == (3, 3)


def test_raw_ins_replay_refuses_a_gap_during_static_initialization(monkeypatch):
    """Synthetic coast data can never contribute to static initialization."""
    tracked_frames = [
        _tracked_frame(0, 10, [_measurement(0.1)]),
        _tracked_frame(
            2,
            12,
            [_measurement(0.1)],
            gap_before=GapEvent(10, 12, 1),
        ),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    with pytest.raises(
        ValueError,
        match="Static initialization cannot cross a BMU sequence gap",
    ):
        ins_replay.run_raw_ins_replay(raw_frames, initialization_duration=0.2)


def test_raw_ins_replay_rejects_quality_flags_with_source_context(monkeypatch):
    """A structural concern identifies both its frame and block position."""
    tracked_frames = [
        _tracked_frame(
            7,
            10,
            [
                _measurement(0.1),
                _measurement(0.1, dv=np.array([np.nan, 0.0, 0.0])),
            ],
        ),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    with pytest.raises(
        ValueError,
        match=r"frame 7, measurement 1: NONFINITE_DV",
    ):
        ins_replay.run_raw_ins_replay(raw_frames, initialization_duration=0.1)


@pytest.mark.parametrize("initialization_duration", [0.0, -0.1, np.nan, np.inf])
def test_raw_ins_replay_rejects_unusable_initialization_durations(
    initialization_duration,
):
    """A static interval must represent a finite, forward span of IMU time."""
    with pytest.raises(ValueError, match="finite and positive"):
        ins_replay.run_raw_ins_replay([], initialization_duration)


def test_raw_ins_replay_rejects_capture_shorter_than_initialization(monkeypatch):
    """The runner cannot silently initialize from less time than requested."""
    tracked_frames = [_tracked_frame(0, 10, [_measurement(0.1), _measurement(0.1)])]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    with pytest.raises(ValueError, match="Capture ends before the requested"):
        ins_replay.run_raw_ins_replay(raw_frames, initialization_duration=0.3)


def test_raw_ins_replay_requires_an_increment_after_initialization(monkeypatch):
    """Initialization-only captures are not misrepresented as trajectories."""
    tracked_frames = [_tracked_frame(0, 10, [_measurement(0.1), _measurement(0.1)])]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    with pytest.raises(ValueError, match="at least one increment after initialization"):
        ins_replay.run_raw_ins_replay(raw_frames, initialization_duration=0.2)


def test_raw_ins_replay_propagates_source_errors_unchanged(monkeypatch):
    """A framing, decoder, or sequence failure retains its original diagnosis."""
    source_error = ValueError("synthetic decoder failure")

    def failing_iter_tracked_frames(_raw_frames):
        raise source_error

    monkeypatch.setattr(
        ins_replay,
        "iter_tracked_frames",
        failing_iter_tracked_frames,
    )

    with pytest.raises(ValueError) as raised:
        ins_replay.run_raw_ins_replay([], initialization_duration=0.1)

    assert raised.value is source_error
