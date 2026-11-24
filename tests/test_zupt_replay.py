"""Contract tests for causal INS plus ZUPT replay.

These cases work at the tracked-frame boundary. They prove the new runner's
phase transition and EKF wiring while leaving wire decoding, frame provenance,
sequence tracking, and SHOE scoring to their own focused test modules.
"""
import numpy as np
import pytest

from bootins.replay import ins_zupt as zupt_replay
from bootins.bmu.decode import DecodedMessage, Measurement, MessageInfo
from bootins.bmu.provenance import DecodedFrame
from bootins.bmu.sequence import GapEvent
from bootins.bmu.stream import TrackedFrame
from bootins.bmu.transport import RawFrame
from bootins.estimation.esekf import (
    NOM_BG,
    NOM_BA,
    NOM_Q,
    NOM_V,
    esekf_predict,
    esekf_update,
)
from bootins.estimation.initialization import initialize_static_zupt
from bootins.estimation.mechanization import G_NED


G = np.linalg.norm(G_NED)
SIGMA_A = 0.5
SIGMA_G = 0.25


def _increment(
    dt: float,
    *,
    dtheta: np.ndarray | None = None,
    dv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one rest-like increment with optional body-frame overrides."""
    return (
        np.zeros(3) if dtheta is None else np.asarray(dtheta, dtype=float),
        np.array([0.0, 0.0, G * dt]) if dv is None else np.asarray(dv, dtype=float),
        float(dt),
    )


def _measurement_from_increment(
    increment: tuple[np.ndarray, np.ndarray, float],
) -> Measurement:
    """Represent one synthetic increment as an already-decoded BMU measurement."""
    dtheta, dv, dt = increment
    return Measurement(
        dt=dt,
        dtheta=dtheta,
        dv=dv,
        quat_raw=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def _tracked_frame(
    frame_index: int,
    sequence_num: int,
    increments: list[tuple[np.ndarray, np.ndarray, float]],
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
                measurements=[
                    _measurement_from_increment(increment)
                    for increment in increments
                ],
            ),
        ),
        gap_before=gap_before,
    )


def _mock_tracked_frames(
    monkeypatch: pytest.MonkeyPatch,
    tracked_frames: list[TrackedFrame],
) -> list[RawFrame]:
    """Make the runner receive prescribed tracked-frame results."""
    raw_frames = [frame.decoded_frame.raw_frame for frame in tracked_frames]

    def fake_iter_tracked_frames(observed_raw_frames):
        assert observed_raw_frames is raw_frames
        return iter(tracked_frames)

    monkeypatch.setattr(
        zupt_replay,
        "iter_tracked_frames",
        fake_iter_tracked_frames,
    )
    return raw_frames


def _P0() -> np.ndarray:
    """Return a nontrivial ESEKF covariance for update and alignment checks."""
    return np.diag([
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1e-4, 1e-4, 1e-4,
        1.0, 1.0, 1.0,
        1e-6, 1e-6, 1e-6,
    ])


def _run(raw_frames: list[RawFrame], **overrides) -> zupt_replay.ZuptReplayResult:
    """Run the canonical test configuration with named setting overrides."""
    settings = {
        "minimum_stance_duration": 0.1,
        "startup_search_limit": 1.0,
        "initial_accel_bias_b": np.zeros(3),
        "P0": _P0(),
        "Q": np.eye(15) * 1e-8,
        "Q_coast": np.diag([*(np.ones(9) * 1e-7), *np.zeros(6)]),
        "R_zupt": np.eye(3) * 1e-4,
        "window_size": 1,
        "threshold": 1e-6,
        "sigma_a": SIGMA_A,
        "sigma_g": SIGMA_G,
    }
    settings.update(overrides)
    return zupt_replay.run_zupt_replay(raw_frames, **settings)


def _vel_h(nominal: np.ndarray) -> np.ndarray:
    """Return the zero-velocity measurement model's nominal prediction."""
    return nominal[NOM_V]


_H_VEL = np.zeros((3, 15), dtype=float)
_H_VEL[:, 3:6] = np.eye(3)


def test_zupt_replay_returns_initialized_prior_and_aligned_posteriors(monkeypatch):
    """A qualified static startup creates the time-zero prior, then posteriors."""
    gyro_bias_b = np.array([0.02, -0.01, 0.03])
    startup = [
        _increment(0.1, dtheta=gyro_bias_b * 0.1),
        _increment(0.1, dtheta=gyro_bias_b * 0.1),
    ]
    post_initialization = [_increment(0.1), _increment(0.2)]
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [_tracked_frame(0, 10, startup + post_initialization)],
    )

    result = _run(raw_frames, minimum_stance_duration=0.2, threshold=1e3)

    assert result.initialization_interval.start_index == 0
    assert result.initialization_interval.replay_start_index == 2
    assert result.initialization.static_level.increment_count == 2
    np.testing.assert_allclose(result.nominals[0, NOM_Q], result.initialization.static_level.q_fb)
    np.testing.assert_allclose(result.nominals[0, NOM_BG], gyro_bias_b, atol=1e-12)
    np.testing.assert_allclose(result.time, [0.0, 0.1, 0.3], atol=1e-12)
    assert result.nominals.shape == (3, 16)
    assert result.covariances.shape == (3, 15, 15)
    assert result.stance_flags.shape == (2,)
    np.testing.assert_array_equal(result.stance_flags, [True, True])


def test_zupt_replay_starts_a_fresh_estimation_detector(monkeypatch):
    """Startup detector history cannot create a ZUPT at the first replay epoch."""
    startup = [_increment(0.1), _increment(0.1), _increment(0.1)]
    post_initialization = [_increment(0.1) for _ in range(4)]
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [_tracked_frame(0, 10, startup + post_initialization)],
    )

    result = _run(
        raw_frames,
        window_size=3,
        minimum_stance_duration=0.1,
    )

    np.testing.assert_array_equal(result.stance_flags, [False, False, True, True])


def test_zupt_replay_matches_manual_predict_then_zupt_loop(monkeypatch):
    """Every flagged post-init increment is predicted before its ZUPT update."""
    gyro_bias_b = np.array([0.01, 0.02, -0.01])
    startup = _increment(0.1, dtheta=gyro_bias_b * 0.1)
    post_initialization = [
        _increment(0.1, dtheta=gyro_bias_b * 0.1),
        _increment(0.2, dtheta=gyro_bias_b * 0.2),
    ]
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [_tracked_frame(0, 10, [startup, *post_initialization])],
    )
    P0 = _P0()
    Q = np.eye(15) * 1e-8
    R_zupt = np.eye(3) * 1e-4
    initial_accel_bias_b = np.array([0.25, -0.5, 0.75])

    result = _run(
        raw_frames,
        P0=P0,
        Q=Q,
        R_zupt=R_zupt,
        threshold=1e3,
        initial_accel_bias_b=initial_accel_bias_b,
    )

    initialization = initialize_static_zupt([startup])
    expected_nominal = np.zeros(16)
    expected_nominal[NOM_Q] = initialization.static_level.q_fb
    expected_nominal[NOM_BG] = initialization.gyro_bias_b
    expected_nominal[NOM_BA] = initial_accel_bias_b
    expected_covariance = P0.copy()
    expected_nominals = [expected_nominal]
    expected_covariances = [expected_covariance]
    for increment in post_initialization:
        dtheta, dv, dt = increment
        expected_nominal, expected_covariance = esekf_predict(
            expected_nominal,
            expected_covariance,
            dtheta,
            dv,
            dt,
            Q,
        )
        expected_nominal, expected_covariance = esekf_update(
            expected_nominal,
            expected_covariance,
            np.zeros(3),
            _vel_h,
            _H_VEL,
            R_zupt,
        )
        expected_nominals.append(expected_nominal)
        expected_covariances.append(expected_covariance)

    np.testing.assert_array_equal(result.stance_flags, [True, True])
    np.testing.assert_allclose(result.nominals, np.stack(expected_nominals), atol=1e-12)
    np.testing.assert_allclose(
        result.covariances,
        np.stack(expected_covariances),
        atol=1e-12,
    )
    initial_accel_bias_b[0] = -1.0
    np.testing.assert_allclose(result.nominals[0, NOM_BA], [0.25, -0.5, 0.75])


def test_zupt_replay_keeps_the_qualifying_increment_in_initialization(monkeypatch):
    """The increment reaching startup duration is never processed twice."""
    startup = [_increment(0.1), _increment(0.1), _increment(0.1)]
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [_tracked_frame(0, 10, startup + [_increment(0.2)])],
    )

    result = _run(raw_frames, minimum_stance_duration=0.25)

    assert result.initialization_interval.replay_start_index == 3
    np.testing.assert_allclose(result.initialization.static_level.duration, 0.3)
    np.testing.assert_allclose(result.time, [0.0, 0.2], atol=1e-12)


def test_zupt_replay_rejects_capture_without_a_qualified_startup_stance(monkeypatch):
    """An incomplete startup cannot be represented as an initialized replay."""
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [_tracked_frame(0, 10, [_increment(0.1), _increment(0.1)])],
    )

    with pytest.raises(ValueError, match="Could not initialize"):
        _run(raw_frames, minimum_stance_duration=0.3)


def test_zupt_replay_requires_an_increment_after_initialization(monkeypatch):
    """A final qualifying increment establishes no trajectory and is rejected."""
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [_tracked_frame(0, 10, [_increment(0.1), _increment(0.1)])],
    )

    with pytest.raises(ValueError, match="at least one increment after initialization"):
        _run(raw_frames, minimum_stance_duration=0.2)


def test_zupt_replay_rejects_a_confirmed_sequence_gap(monkeypatch):
    """A gap beyond the bounded recovery policy cannot be coasted."""
    raw_frames = _mock_tracked_frames(
        monkeypatch,
        [
            _tracked_frame(0, 10, [_increment(0.1)]),
            _tracked_frame(
                4,
                14,
                [_increment(0.1)],
                gap_before=GapEvent(
                    previous_seq_num=10,
                    current_seq_num=14,
                    missing_message_count=3,
                ),
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"exceeds the bounded recovery policy: 3 missing notification\(s\), limit 1",
    ):
        _run(raw_frames)


def test_zupt_replay_coasts_post_init_gap_without_a_zupt(monkeypatch):
    """A coast resets SHOE and emits synthetic non-ZUPT posterior rows."""
    tracked_frames = [
        _tracked_frame(0, 10, [_increment(0.1), _increment(0.1)]),
        _tracked_frame(1, 11, [_increment(0.1)]),
        _tracked_frame(
            3,
            13,
            [_increment(0.1)],
            gap_before=GapEvent(11, 13, 1),
        ),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    result = _run(raw_frames, window_size=2)

    assert len(result.coast_plans) == 1
    assert result.coast_plans[0].reference_frame_index == 1
    assert result.coast_plans[0].revealing_frame_index == 3
    np.testing.assert_allclose(result.time, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(result.stance_flags, [False, False, False])


def test_zupt_replay_adds_coast_covariance_only_to_synthetic_increments(
    monkeypatch,
):
    """Synthetic coast predicts use Q plus Q_coast; real predicts use Q."""
    tracked_frames = [
        _tracked_frame(0, 10, [_increment(0.1), _increment(0.1)]),
        _tracked_frame(1, 11, [_increment(0.1)]),
        _tracked_frame(
            3,
            13,
            [_increment(0.1)],
            gap_before=GapEvent(11, 13, 1),
        ),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)
    process_covariances: list[np.ndarray] = []
    original_predict = zupt_replay.esekf_predict

    def record_predict(
        nominal: np.ndarray,
        covariance: np.ndarray,
        dtheta: np.ndarray,
        dv: np.ndarray,
        dt: float,
        process_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        process_covariances.append(np.array(process_covariance, copy=True))
        return original_predict(nominal, covariance, dtheta, dv, dt, process_covariance)

    monkeypatch.setattr(zupt_replay, "esekf_predict", record_predict)
    Q = np.diag(np.linspace(1e-8, 15e-8, 15))
    Q_coast = np.diag([*np.linspace(1e-6, 9e-6, 9), *np.zeros(6)])

    _run(raw_frames, Q=Q, Q_coast=Q_coast, window_size=2)

    assert len(process_covariances) == 3
    np.testing.assert_allclose(process_covariances[0], Q, atol=0.0)
    np.testing.assert_allclose(process_covariances[1], Q + Q_coast, atol=0.0)
    np.testing.assert_allclose(process_covariances[2], Q, atol=0.0)


def test_zupt_replay_refuses_gap_during_startup(monkeypatch):
    """A synthetic coast cannot contribute to startup stance selection."""
    tracked_frames = [
        _tracked_frame(0, 10, [_increment(0.1)]),
        _tracked_frame(
            2,
            12,
            [_increment(0.1)],
            gap_before=GapEvent(10, 12, 1),
        ),
    ]
    raw_frames = _mock_tracked_frames(monkeypatch, tracked_frames)

    with pytest.raises(
        ValueError,
        match="Static initialization cannot cross a BMU sequence gap",
    ):
        _run(raw_frames, minimum_stance_duration=0.2)
