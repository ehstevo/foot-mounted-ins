"""Tests for replayable clock-aware IMU and optional-ZUPT transitions."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from bootins.replay import gnss_ins_zupt as gnss_zupt_replay
from bootins.estimation.clock_augmented_esekf import (
    ERR_DV,
    NOM_BA,
    NOM_BG,
    NOM_P,
    NOM_Q,
    NOM_TAU,
    NOM_V,
    esekf_predict_clock,
    esekf_update_clock,
)
from bootins.aiding.gnss_position_clock import gnss_position_update_ned_gated
from bootins.replay.clock_delayed_gnss import TimedGnssFixNed
from bootins.sensors.gnss_observations import GnssFixLLH
from bootins.replay.gnss_schedule import GnssScheduleSummary, ScheduledGnssFix
from bootins.config.gnss_ins_zupt import GnssZuptConfig
from bootins.replay.gnss_ins_zupt import (
    ClockAwareGnssIns,
    ClockImuEvent,
    ClockImuPosterior,
    DelayedGnssUpdate,
    DelayedGnssUpdateStatus,
    GnssFusionDiagnostic,
    GnssZuptReplayResult,
    _apply_clock_imu_event,
    _initial_clock_nominal,
    _select_gnss_aided_startup,
    _to_timed_gnss_fix_ned,
)
from bootins.estimation.initialization import (
    GnssAidedStartup,
    GnssCourseEstimate,
    GnssCourseInitialization,
    StaticLevelInitialization,
    StaticZuptInitialization,
    StanceInitializationInterval,
    estimate_static_gnss_origin,
)
from bootins.bmu.decode import DecodedMessage, Measurement, MessageInfo
from bootins.bmu.provenance import DecodedFrame
from bootins.bmu.sequence import GapEvent
from bootins.bmu.stream import TrackedFrame
from bootins.bmu.transport import RawFrame
from bootins.replay.imu_gap_recovery import (
    ImuGapRecoveryPolicy,
    ImuGapRecoveryProcessor,
)


def _nominal() -> np.ndarray:
    """Return one nonstationary clock-augmented nominal state."""
    return np.array([
        1.0, -2.0, 0.5,
        0.4, -0.3, 0.2,
        1.0, 0.0, 0.0, 0.0,
        0.01, -0.02, 0.03,
        0.001, -0.002, 0.003,
        -12.0,
    ])


def _covariance() -> np.ndarray:
    """Return one symmetric positive-definite clock error covariance."""
    diagonal = np.linspace(0.1, 1.6, 16)
    return np.diag(diagonal)


def _event(
    *,
    stance_flag: bool,
    additional_process_covariance: np.ndarray | None = None,
) -> ClockImuEvent:
    """Return one IMU event whose predicted velocity is nonzero."""
    return ClockImuEvent(
        increment=(
            np.array([0.003, -0.002, 0.001]),
            np.array([0.04, -0.03, -0.09]),
            0.01,
        ),
        stance_flag=stance_flag,
        additional_process_covariance=additional_process_covariance,
    )


def _velocity_h(nominal: np.ndarray) -> np.ndarray:
    """Return the manual clock-filter velocity measurement prediction."""
    return nominal[NOM_V]


_H_VEL_CLOCK = np.zeros((3, 16), dtype=float)
_H_VEL_CLOCK[:, ERR_DV] = np.eye(3)
_Z_ZUPT = np.zeros(3, dtype=float)
_Q = np.diag(np.linspace(1e-7, 2e-7, 16))
_R_ZUPT = np.diag([0.02, 0.03, 0.04])


def _startup_config() -> GnssZuptConfig:
    """Return minimal valid settings for startup-composition tests."""
    return GnssZuptConfig(
        initial_stance_duration=0.2,
        initial_stance_search_limit=2.0,
        terminal_stance_duration=0.2,
        terminal_stance_search_limit=2.0,
        course_minimum_fix_count=3,
        course_minimum_baseline=10.0,
        course_maximum_cross_track_rms=1.0,
        initial_accel_bias_b=np.zeros(3),
        P0=np.eye(16),
        Q=np.eye(16),
        Q_coast=np.diag([*(np.ones(9) * 1e-4), *np.zeros(7)]),
        R_zupt=np.eye(3),
        R_gnss=np.eye(3),
        nis_threshold=10.0,
        window_size=2,
        threshold=100.0,
        sigma_a=1.0,
        sigma_g=1.0,
    )


def _startup_increment(identifier: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one distinct but otherwise unremarkable IMU increment."""
    return (
        np.array([identifier, 0.0, 0.0]),
        np.array([0.0, identifier, 0.0]),
        0.01,
    )


def _startup_tracked_frame(
    frame_index: int,
    *increments: tuple[np.ndarray, np.ndarray, float],
    gap_before: GapEvent | None = None,
) -> TrackedFrame:
    """Return one factual decoded IMU frame for startup driver tests."""
    raw_frame = RawFrame(
        frame_index=frame_index,
        arrival_monotonic=float(frame_index),
        payload=b"test frame",
    )
    tracked_frame = TrackedFrame(
        decoded_frame=DecodedFrame(
            raw_frame=raw_frame,
            decoded_message=DecodedMessage(
                message_info=MessageInfo(
                    sequence_num=frame_index,
                    device_timestamp_raw=0,
                ),
                measurements=[
                    Measurement(
                        dt=dt,
                        dtheta=np.array(dtheta, dtype=float, copy=True),
                        dv=np.array(dv, dtype=float, copy=True),
                        quat_raw=np.array([1.0, 0.0, 0.0, 0.0]),
                    )
                    for dtheta, dv, dt in increments
                ],
            ),
        ),
        gap_before=gap_before,
    )
    return tracked_frame


def _startup_fix(native_record_index: int) -> GnssFixLLH:
    """Return one valid GNSS event distinguished by native record index."""
    return GnssFixLLH(
        gps_week=2300,
        gps_tow=1000.0 + native_record_index,
        host_arrival_monotonic=float(native_record_index),
        latitude=0.7,
        longitude=-1.1,
        height_ellipsoid=123.0,
        fix_valid=True,
        satellite_count=12,
        horizontal_accuracy=1.0,
        vertical_accuracy=1.5,
        native_record_index=native_record_index,
    )


def _completed_startup() -> GnssAidedStartup:
    """Return startup evidence with deliberately distinct initial/terminal facts."""
    seed_fix = _startup_fix(5)
    initial_static = StaticZuptInitialization(
        static_level=StaticLevelInitialization(
            q_fb=np.array([1.0, 0.0, 0.0, 0.0]),
            mean_specific_force_b=np.array([0.0, 0.0, -9.8]),
            duration=0.2,
            increment_count=20,
        ),
        gyro_bias_b=np.array([9.0, 8.0, 7.0]),
    )
    terminal_static = StaticZuptInitialization(
        static_level=StaticLevelInitialization(
            q_fb=np.array([1.0, 0.0, 0.0, 0.0]),
            mean_specific_force_b=np.array([0.0, 0.0, -9.8]),
            duration=0.2,
            increment_count=20,
        ),
        gyro_bias_b=np.array([0.01, -0.02, 0.03]),
    )
    course = GnssCourseEstimate(
        direction_ne=np.array([0.8, 0.6]),
        course=float(np.arctan2(0.6, 0.8)),
        baseline=25.0,
        cross_track_rms=0.5,
        fix_count=10,
    )
    terminal_initialization = GnssCourseInitialization(
        terminal_static=terminal_static,
        course=course,
        q_nb=np.array([0.5, 0.5, -0.5, 0.5]),
    )
    interval = StanceInitializationInterval(
        start_index=0,
        replay_start_index=20,
        duration=0.2,
        increment_count=20,
    )
    return GnssAidedStartup(
        initial_static=initial_static,
        initial_interval=interval,
        terminal_initialization=terminal_initialization,
        terminal_interval=interval,
        static_origin=estimate_static_gnss_origin((seed_fix,)),
        seed_fix=seed_fix,
    )


def test_startup_driver_preserves_post_seed_event_and_dispatches_causally(
    monkeypatch: pytest.MonkeyPatch,
):
    """Startup consumes each event once and leaves the following event available."""
    config = _startup_config()
    first_fix = _startup_fix(1)
    seed_fix = _startup_fix(2)
    startup_frame = _startup_tracked_frame(
        3,
        _startup_increment(10.0),
        _startup_increment(20.0),
    )
    following_frame = _startup_tracked_frame(4, _startup_increment(30.0))
    arrivals = iter([first_fix, startup_frame, seed_fix, following_frame])
    startup_result = object()
    selector_constructor_arguments: list[dict[str, object]] = []
    detector_constructor_arguments: list[dict[str, object]] = []
    observed_gnss_fixes: list[GnssFixLLH] = []
    observed_imu_updates: list[tuple[float, bool]] = []
    observed_detector_increments: list[float] = []

    class FakeStartupSelector:
        """Record the runner's selector interaction without initialization math."""

        def __init__(self, **kwargs: object) -> None:
            selector_constructor_arguments.append(kwargs)

        def update_gnss(self, fix: GnssFixLLH) -> object | None:
            observed_gnss_fixes.append(fix)
            return startup_result if fix is seed_fix else None

        def update_imu(
            self,
            increment: tuple[np.ndarray, np.ndarray, float],
            stance_flag: bool,
        ) -> None:
            observed_imu_updates.append((float(increment[0][0]), stance_flag))
            return None

        def finish(self) -> object:
            raise AssertionError("Startup must return at the valid seed event.")

    class FakeDetector:
        """Return prescribed causal stance decisions for the startup frame."""

        def __init__(self, **kwargs: object) -> None:
            detector_constructor_arguments.append(kwargs)
            self._flags = iter((False, True))

        def update(self, increment: tuple[np.ndarray, np.ndarray, float]) -> bool:
            observed_detector_increments.append(float(increment[0][0]))
            return next(self._flags)

    monkeypatch.setattr(gnss_zupt_replay, "GnssAidedStartupSelector", FakeStartupSelector)
    monkeypatch.setattr(gnss_zupt_replay, "ShoeStanceDetector", FakeDetector)

    result = _select_gnss_aided_startup(
        arrivals,
        config,
        ImuGapRecoveryProcessor(ImuGapRecoveryPolicy()),
    )

    assert result is startup_result
    assert observed_gnss_fixes == [first_fix, seed_fix]
    assert observed_detector_increments == [10.0, 20.0]
    assert observed_imu_updates == [(10.0, False), (20.0, True)]
    assert selector_constructor_arguments == [
        {
            "initial_stance_duration": config.initial_stance_duration,
            "initial_stance_search_limit": config.initial_stance_search_limit,
            "terminal_stance_duration": config.terminal_stance_duration,
            "terminal_stance_search_limit": config.terminal_stance_search_limit,
            "course_minimum_fix_count": config.course_minimum_fix_count,
            "course_minimum_baseline": config.course_minimum_baseline,
            "course_maximum_cross_track_rms": config.course_maximum_cross_track_rms,
        }
    ]
    assert detector_constructor_arguments == [
        {
            "window_size": config.window_size,
            "threshold": config.threshold,
            "sigma_a": config.sigma_a,
            "sigma_g": config.sigma_g,
        }
    ]
    assert next(arrivals) is following_frame


def test_startup_driver_propagates_the_selector_eof_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Exhaustion remains a phase-specific startup failure, not a null result."""
    first_fix = _startup_fix(1)
    failure = ValueError("No qualified GNSS course before input ended.")
    observed_gnss_fixes: list[GnssFixLLH] = []
    finish_call_count = 0

    class FakeStartupSelector:
        """Expose EOF behavior without repeating the selector state machine."""

        def __init__(self, **_kwargs: object) -> None:
            pass

        def update_gnss(self, fix: GnssFixLLH) -> None:
            observed_gnss_fixes.append(fix)
            return None

        def update_imu(
            self,
            _increment: tuple[np.ndarray, np.ndarray, float],
            _stance_flag: bool,
        ) -> None:
            return None

        def finish(self) -> object:
            nonlocal finish_call_count
            finish_call_count += 1
            raise failure

    class FakeDetector:
        """Satisfy the driver constructor contract without any IMU events."""

        def __init__(self, **_kwargs: object) -> None:
            pass

        def update(self, _increment: tuple[np.ndarray, np.ndarray, float]) -> bool:
            raise AssertionError("The test source contains no IMU event.")

    monkeypatch.setattr(gnss_zupt_replay, "GnssAidedStartupSelector", FakeStartupSelector)
    monkeypatch.setattr(gnss_zupt_replay, "ShoeStanceDetector", FakeDetector)

    with pytest.raises(ValueError) as raised:
        _select_gnss_aided_startup(
            iter([first_fix]),
            _startup_config(),
            ImuGapRecoveryProcessor(ImuGapRecoveryPolicy()),
        )

    assert raised.value is failure
    assert observed_gnss_fixes == [first_fix]
    assert finish_call_count == 1


def test_startup_driver_refuses_a_confirmed_imu_gap():
    """A permitted operational coast is never eligible for aided startup."""
    first_frame = _startup_tracked_frame(10, _startup_increment(1.0))
    revealing_frame = _startup_tracked_frame(
        12,
        _startup_increment(2.0),
        gap_before=GapEvent(10, 12, 1),
    )

    with pytest.raises(
        ValueError,
        match="GNSS-aided startup cannot cross a BMU sequence gap",
    ):
        _select_gnss_aided_startup(
            iter([first_frame, revealing_frame]),
            _startup_config(),
            ImuGapRecoveryProcessor(ImuGapRecoveryPolicy()),
        )


def test_initial_clock_nominal_uses_terminal_facts_and_supplied_bias():
    """Aided state preserves terminal facts and owns its supplied bias estimate."""
    startup = _completed_startup()
    initial_accel_bias_b = np.array([0.25, -0.5, 0.75])

    nominal = _initial_clock_nominal(startup, initial_accel_bias_b)

    assert nominal.shape == (NOM_TAU + 1,)
    np.testing.assert_array_equal(nominal[NOM_P], np.zeros(3))
    np.testing.assert_array_equal(nominal[NOM_V], np.zeros(3))
    np.testing.assert_allclose(
        nominal[NOM_Q],
        startup.terminal_initialization.q_nb,
    )
    np.testing.assert_array_equal(nominal[NOM_BA], initial_accel_bias_b)
    np.testing.assert_allclose(
        nominal[NOM_BG],
        startup.terminal_initialization.terminal_static.gyro_bias_b,
    )
    assert nominal[NOM_TAU] == 0.0
    assert not np.allclose(nominal[NOM_BG], startup.initial_static.gyro_bias_b)

    nominal[NOM_Q][0] = -1.0
    nominal[NOM_BA][0] = -1.0
    nominal[NOM_BG][0] = -1.0
    np.testing.assert_allclose(
        startup.terminal_initialization.q_nb,
        [0.5, 0.5, -0.5, 0.5],
    )
    np.testing.assert_allclose(
        startup.terminal_initialization.terminal_static.gyro_bias_b,
        [0.01, -0.02, 0.03],
    )
    np.testing.assert_allclose(initial_accel_bias_b, [0.25, -0.5, 0.75])


def test_scheduled_fix_conversion_splits_static_origin_and_seed_time_references(
    monkeypatch: pytest.MonkeyPatch,
):
    """Position uses the static origin while epoch rebasing uses the seed."""
    startup = _completed_startup()
    static_origin_fix = replace(
        startup.seed_fix,
        latitude=0.65,
        longitude=-1.15,
        height_ellipsoid=100.0,
    )
    startup = replace(
        startup,
        static_origin=estimate_static_gnss_origin((static_origin_fix,)),
    )
    source_fix = GnssFixLLH(
        gps_week=2300,
        gps_tow=1020.5,
        host_arrival_monotonic=9.0,
        latitude=0.72,
        longitude=-1.08,
        height_ellipsoid=130.0,
        fix_valid=False,
        satellite_count=0,
        horizontal_accuracy=50.0,
        vertical_accuracy=75.0,
        native_record_index=8,
    )
    scheduled_fix = ScheduledGnssFix(source_fix, t_arrival_i=1.25)
    expected_p_ned = np.array([11.0, -12.0, 13.0])
    observed_llh_arguments: list[tuple[float, ...]] = []

    def fake_llh_to_ned(*arguments: float) -> np.ndarray:
        observed_llh_arguments.append(arguments)
        return expected_p_ned

    monkeypatch.setattr(gnss_zupt_replay, "llh_to_ned", fake_llh_to_ned)

    timed_fix = _to_timed_gnss_fix_ned(scheduled_fix, startup)

    assert observed_llh_arguments == [
        (
            source_fix.latitude,
            source_fix.longitude,
            source_fix.height_ellipsoid,
            startup.static_origin.latitude,
            startup.static_origin.longitude,
            startup.static_origin.height_ellipsoid,
        )
    ]
    assert timed_fix.t_fix_g == pytest.approx(
        source_fix.t_fix_g - startup.seed_fix.t_fix_g
    )
    assert timed_fix.t_arrival_i == pytest.approx(scheduled_fix.t_arrival_i)
    np.testing.assert_array_equal(timed_fix.p_ned, expected_p_ned)


def test_replay_result_retains_aligned_complete_diagnostics():
    """The result exposes initialized/posterior history and scheduler evidence."""
    startup = _completed_startup()
    update = DelayedGnssUpdate(
        status=DelayedGnssUpdateStatus.ACCEPTED,
        t_fix_i_hat=0.01,
        t_applied_i=0.02,
        nis=1.5,
    )
    diagnostic = GnssFusionDiagnostic(
        fix=_startup_fix(1),
        t_fix_g=0.01,
        t_arrival_i=0.02,
        p_gnss_ned=np.array([1.0, -2.0, 0.5]),
        tau_hat=0.0,
        update=update,
        innovation_ned=np.array([0.1, -0.2, 0.05]),
        innovation_covariance_ned=np.eye(3),
        prior_position_covariance_ned=np.eye(3) * 2.0,
    )
    summary = GnssScheduleSummary(
        dropped_before_imu_count=2,
        dropped_after_imu_count=3,
    )
    result = GnssZuptReplayResult(
        startup=startup,
        time=np.array([0.0, 0.01]),
        nominals=np.zeros((2, NOM_TAU + 1)),
        covariances=np.zeros((2, 16, 16)),
        stance_flags=np.array([True]),
        gnss_updates=(update,),
        gnss_fusion_diagnostics=(diagnostic,),
        invalid_gnss_fix_count=4,
        summary=summary,
        coast_plans=(),
    )

    assert result.startup is startup
    np.testing.assert_array_equal(result.time, [0.0, 0.01])
    assert result.nominals.shape == (2, NOM_TAU + 1)
    assert result.covariances.shape == (2, 16, 16)
    np.testing.assert_array_equal(result.stance_flags, [True])
    assert result.gnss_updates == (update,)
    assert result.gnss_fusion_diagnostics == (diagnostic,)
    assert result.invalid_gnss_fix_count == 4
    assert result.summary is summary
    assert result.coast_plans == ()
    with pytest.raises(FrozenInstanceError):
        result.invalid_gnss_fix_count = 0


def test_runner_defers_one_frame_then_returns_corrected_history(
    monkeypatch: pytest.MonkeyPatch,
):
    """A one-frame future-mapped fix fuses at its next IMU-frame opportunity."""
    startup = _completed_startup()
    deferred_epoch = 0.03 + 0.5 * gnss_zupt_replay._TIMELINE_ATOL
    valid_fix = GnssFixLLH(
        gps_week=2300,
        gps_tow=1030.0,
        host_arrival_monotonic=10.0,
        latitude=0.71,
        longitude=-1.09,
        height_ellipsoid=124.0,
        fix_valid=True,
        satellite_count=12,
        horizontal_accuracy=1.0,
        vertical_accuracy=1.5,
        native_record_index=30,
    )
    invalid_fix = GnssFixLLH(
        gps_week=2300,
        gps_tow=1031.0,
        host_arrival_monotonic=11.0,
        latitude=0.72,
        longitude=-1.08,
        height_ellipsoid=125.0,
        fix_valid=False,
        satellite_count=0,
        horizontal_accuracy=50.0,
        vertical_accuracy=75.0,
        native_record_index=31,
    )
    trailing_fix = GnssFixLLH(
        gps_week=2300,
        gps_tow=1032.0,
        host_arrival_monotonic=13.0,
        latitude=0.73,
        longitude=-1.07,
        height_ellipsoid=126.0,
        fix_valid=True,
        satellite_count=12,
        horizontal_accuracy=1.0,
        vertical_accuracy=1.5,
        native_record_index=32,
    )
    first_frame = _startup_tracked_frame(
        10,
        _startup_increment(1.0),
        _startup_increment(2.0),
    )
    second_frame = _startup_tracked_frame(12, _startup_increment(3.0))
    arrivals = [valid_fix, first_frame, invalid_fix, second_frame, trailing_fix]
    detector_arguments: list[dict[str, object]] = []
    converted_fixes: list[ScheduledGnssFix] = []
    filters: list[object] = []

    class FakeDetector:
        """Return one fixed causal detector decision per post-startup increment."""

        def __init__(self, **kwargs: object) -> None:
            detector_arguments.append(kwargs)
            self._flags = iter((True, False, True))

        def update(self, _increment: object) -> bool:
            return next(self._flags)

    class FakeReplayFilter:
        """Expose runner calls while modeling a history rewritten by fusion."""

        def __init__(
            self,
            nominal0: np.ndarray,
            P0: np.ndarray,
            Q: np.ndarray,
            R_zupt: np.ndarray,
        ) -> None:
            del Q, R_zupt
            self._nominal = np.array(nominal0, dtype=float, copy=True)
            self.initial_nominal = self._nominal.copy()
            self._P = np.array(P0, dtype=float, copy=True)
            self._t_i = 0.0
            self._posteriors: list[ClockImuPosterior] = []
            self._corrected = False
            self.fused_fixes: list[TimedGnssFixNed] = []
            filters.append(self)

        @property
        def nominal(self) -> np.ndarray:
            return self._nominal.copy()

        @property
        def P(self) -> np.ndarray:
            return self._P.copy()

        @property
        def t_i(self) -> float:
            return self._t_i

        @property
        def posteriors(self) -> tuple[ClockImuPosterior, ...]:
            return tuple(self._posteriors)

        def advance(self, event: ClockImuEvent) -> None:
            self._t_i += event.increment[2]
            value = len(self._posteriors) + 1
            if self._corrected:
                value += 100
            nominal = np.full(NOM_TAU + 1, value, dtype=float)
            nominal[NOM_TAU] = 0.0
            covariance = np.eye(16) * value
            self._nominal = nominal
            self._P = covariance
            self._posteriors.append(
                ClockImuPosterior(self._t_i, nominal, covariance)
            )

        def fuse_delayed_gnss(
            self,
            fix: TimedGnssFixNed,
            R_pos: np.ndarray,
            nis_threshold: float,
        ) -> DelayedGnssUpdate:
            del R_pos, nis_threshold
            self.fused_fixes.append(fix)
            self._corrected = True
            self._posteriors = [
                ClockImuPosterior(
                    posterior.t_i,
                    np.full(NOM_TAU + 1, index + 101, dtype=float),
                    np.eye(16) * (index + 101),
                )
                for index, posterior in enumerate(self._posteriors)
            ]
            self._nominal = self._posteriors[-1].nominal.copy()
            self._P = self._posteriors[-1].P.copy()
            return DelayedGnssUpdate(
                status=DelayedGnssUpdateStatus.ACCEPTED,
                t_fix_i_hat=fix.t_fix_g,
                t_applied_i=fix.t_arrival_i,
                nis=1.25,
            )

        def _fuse_delayed_gnss_with_evaluation(
            self,
            fix: TimedGnssFixNed,
            R_pos: np.ndarray,
            nis_threshold: float,
        ) -> tuple[DelayedGnssUpdate, object]:
            update = self.fuse_delayed_gnss(fix, R_pos, nis_threshold)
            return (
                update,
                gnss_zupt_replay._GnssPreUpdateEvaluation(
                    tau_hat=0.0,
                    innovation_ned=np.array([0.1, -0.2, 0.3]),
                    innovation_covariance_ned=np.eye(3),
                    prior_position_covariance_ned=np.eye(3) * 2.0,
                ),
            )

    def fake_to_timed_fix(
        scheduled_fix: ScheduledGnssFix,
        observed_startup: GnssAidedStartup,
    ) -> TimedGnssFixNed:
        converted_fixes.append(scheduled_fix)
        assert observed_startup is startup
        return TimedGnssFixNed(
            t_fix_g=deferred_epoch,
            t_arrival_i=scheduled_fix.t_arrival_i,
            p_ned=np.array([1.0, 2.0, 3.0]),
        )

    monkeypatch.setattr(
        gnss_zupt_replay,
        "iter_gnss_imu_arrivals",
        lambda _raw_frames, _gnss_fixes: iter(arrivals),
    )
    monkeypatch.setattr(
        gnss_zupt_replay,
        "_select_gnss_aided_startup",
        lambda _arrivals, _config, _recovery_processor: startup,
    )
    monkeypatch.setattr(gnss_zupt_replay, "ShoeStanceDetector", FakeDetector)
    monkeypatch.setattr(gnss_zupt_replay, "ClockAwareGnssIns", FakeReplayFilter)
    monkeypatch.setattr(gnss_zupt_replay, "_to_timed_gnss_fix_ned", fake_to_timed_fix)

    config = replace(
        _startup_config(),
        initial_accel_bias_b=np.array([0.25, -0.5, 0.75]),
    )
    result = gnss_zupt_replay.run_gnss_zupt_replay([], [], config)

    assert detector_arguments == [
        {
            "window_size": 2,
            "threshold": 100.0,
            "sigma_a": 1.0,
            "sigma_g": 1.0,
        }
    ]
    assert len(filters) == 1
    np.testing.assert_array_equal(
        filters[0].initial_nominal[NOM_BA],
        config.initial_accel_bias_b,
    )
    assert len(filters[0].fused_fixes) == 1
    assert filters[0].fused_fixes[0].t_arrival_i == pytest.approx(0.02)
    assert result.gnss_updates[0].t_fix_i_hat == pytest.approx(deferred_epoch)
    assert len(result.gnss_fusion_diagnostics) == 1
    diagnostic = result.gnss_fusion_diagnostics[0]
    assert diagnostic.fix is valid_fix
    assert diagnostic.t_fix_g == pytest.approx(deferred_epoch)
    assert diagnostic.t_arrival_i == pytest.approx(0.02)
    np.testing.assert_allclose(diagnostic.p_gnss_ned, [1.0, 2.0, 3.0])
    assert diagnostic.update is result.gnss_updates[0]
    assert diagnostic.innovation_ned is not None
    assert converted_fixes == [ScheduledGnssFix(valid_fix, t_arrival_i=0.02)]
    np.testing.assert_allclose(result.time, [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_array_equal(result.stance_flags, [True, False, True])
    np.testing.assert_array_equal(result.nominals[1:, 0], [101.0, 102.0, 103.0])
    np.testing.assert_array_equal(result.covariances[1:, 0, 0], [101.0, 102.0, 103.0])
    assert [update.status for update in result.gnss_updates] == [
        DelayedGnssUpdateStatus.ACCEPTED
    ]
    assert result.invalid_gnss_fix_count == 1
    assert result.summary == GnssScheduleSummary(
        dropped_before_imu_count=0,
        dropped_after_imu_count=1,
    )


@pytest.mark.parametrize(
    ("include_later_frame", "t_fix_g", "expected_final_time"),
    (
        (False, 0.02, 0.01),
        (True, 0.03, 0.02),
    ),
    ids=("end-of-stream", "still-future-after-next-frame"),
)
def test_runner_drops_future_mapped_fix_without_reachable_opportunity(
    monkeypatch: pytest.MonkeyPatch,
    include_later_frame: bool,
    t_fix_g: float,
    expected_final_time: float,
):
    """A fix is dropped if no permitted frame reaches its mapped epoch."""
    startup = _completed_startup()
    valid_fix = _startup_fix(1)
    first_frame = _startup_tracked_frame(1, _startup_increment(0.001))
    arrivals: list[GnssFixLLH | TrackedFrame] = [valid_fix, first_frame]
    if include_later_frame:
        arrivals.append(_startup_tracked_frame(2, _startup_increment(0.002)))

    class FakeDetector:
        """Return unflagged decisions for every post-startup increment."""

        def __init__(self, **_kwargs: object) -> None:
            pass

        def update(self, _increment: object) -> bool:
            return False

    def fake_to_timed_fix(
        scheduled_fix: ScheduledGnssFix,
        observed_startup: GnssAidedStartup,
    ) -> TimedGnssFixNed:
        assert scheduled_fix.fix is valid_fix
        assert observed_startup is startup
        return TimedGnssFixNed(
            t_fix_g=t_fix_g,
            t_arrival_i=scheduled_fix.t_arrival_i,
            p_ned=np.zeros(3),
        )

    monkeypatch.setattr(
        gnss_zupt_replay,
        "iter_gnss_imu_arrivals",
        lambda _raw_frames, _gnss_fixes: iter(arrivals),
    )
    monkeypatch.setattr(
        gnss_zupt_replay,
        "_select_gnss_aided_startup",
        lambda _arrivals, _config, _recovery_processor: startup,
    )
    monkeypatch.setattr(gnss_zupt_replay, "ShoeStanceDetector", FakeDetector)
    monkeypatch.setattr(gnss_zupt_replay, "_to_timed_gnss_fix_ned", fake_to_timed_fix)

    result = gnss_zupt_replay.run_gnss_zupt_replay([], [], _startup_config())

    assert len(result.gnss_updates) == 1
    update = result.gnss_updates[0]
    assert update.status is DelayedGnssUpdateStatus.FUTURE_DROPPED
    assert update.t_fix_i_hat == pytest.approx(t_fix_g)
    assert update.t_applied_i is None
    assert update.nis is None
    assert len(result.gnss_fusion_diagnostics) == 1
    diagnostic = result.gnss_fusion_diagnostics[0]
    assert diagnostic.fix is valid_fix
    assert diagnostic.update is update
    assert diagnostic.innovation_ned is None
    assert diagnostic.innovation_covariance_ned is None
    assert diagnostic.prior_position_covariance_ned is None
    assert result.time[-1] == pytest.approx(expected_final_time)
    assert result.invalid_gnss_fix_count == 0
    assert result.summary == GnssScheduleSummary(
        dropped_before_imu_count=0,
        dropped_after_imu_count=0,
    )


def test_runner_coasts_one_post_startup_gap_without_synthetic_gnss_release(
    monkeypatch: pytest.MonkeyPatch,
):
    """One operational gap coasts causally before the revealing-frame release."""
    startup = _completed_startup()
    first_frame = _startup_tracked_frame(
        10,
        _startup_increment(0.001),
        _startup_increment(0.002),
    )
    revealing_frame = _startup_tracked_frame(
        12,
        _startup_increment(0.003),
        _startup_increment(0.004),
        gap_before=GapEvent(10, 12, 1),
    )
    scheduled_fix = replace(
        _startup_fix(1),
        host_arrival_monotonic=10.5,
    )
    arrivals = [first_frame, scheduled_fix, revealing_frame]
    detector_instances: list[object] = []
    converted_fixes: list[ScheduledGnssFix] = []
    advanced_events: list[ClockImuEvent] = []
    config = _startup_config()
    original_advance = ClockAwareGnssIns.advance

    class FakeDetector:
        """Record real detector calls and its coast-induced reset."""

        def __init__(self, **_kwargs: object) -> None:
            self.update_count = 0
            self.reset_count = 0
            detector_instances.append(self)

        def update(self, _increment: object) -> bool:
            self.update_count += 1
            return True

        def reset(self) -> None:
            self.reset_count += 1

    def fake_to_timed_fix(
        observed_scheduled_fix: ScheduledGnssFix,
        observed_startup: GnssAidedStartup,
    ) -> TimedGnssFixNed:
        assert observed_startup is startup
        converted_fixes.append(observed_scheduled_fix)
        return TimedGnssFixNed(
            t_fix_g=0.0,
            t_arrival_i=observed_scheduled_fix.t_arrival_i,
            p_ned=np.zeros(3),
        )

    def record_advance(
        replay_filter: ClockAwareGnssIns,
        event: ClockImuEvent,
    ) -> None:
        advanced_events.append(event)
        original_advance(replay_filter, event)

    monkeypatch.setattr(
        gnss_zupt_replay,
        "iter_gnss_imu_arrivals",
        lambda _raw_frames, _gnss_fixes: iter(arrivals),
    )
    monkeypatch.setattr(
        gnss_zupt_replay,
        "_select_gnss_aided_startup",
        lambda _arrivals, _config, _recovery_processor: startup,
    )
    monkeypatch.setattr(gnss_zupt_replay, "ShoeStanceDetector", FakeDetector)
    monkeypatch.setattr(gnss_zupt_replay, "_to_timed_gnss_fix_ned", fake_to_timed_fix)
    monkeypatch.setattr(ClockAwareGnssIns, "advance", record_advance)

    result = gnss_zupt_replay.run_gnss_zupt_replay([], [], config)

    assert len(detector_instances) == 1
    detector = detector_instances[0]
    assert detector.update_count == 4
    assert detector.reset_count == 1
    assert len(result.coast_plans) == 1
    assert result.coast_plans[0].gap_event == GapEvent(10, 12, 1)
    assert result.coast_plans[0].reference_frame_index == 10
    assert result.coast_plans[0].revealing_frame_index == 12
    np.testing.assert_allclose(
        result.time,
        [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    )
    np.testing.assert_array_equal(
        result.stance_flags,
        [True, True, False, False, True, True],
    )
    assert len(advanced_events) == 6
    for event in (*advanced_events[:2], *advanced_events[4:]):
        assert event.additional_process_covariance is None
    for event in advanced_events[2:4]:
        assert event.stance_flag is False
        assert event.additional_process_covariance is not None
        np.testing.assert_allclose(
            event.additional_process_covariance,
            config.Q_coast,
            atol=0.0,
        )
    assert len(converted_fixes) == 1
    assert converted_fixes[0].fix is scheduled_fix
    assert converted_fixes[0].t_arrival_i == pytest.approx(0.06)
    assert len(result.gnss_updates) == 1
    assert result.summary == GnssScheduleSummary(
        dropped_before_imu_count=0,
        dropped_after_imu_count=0,
    )


def test_runner_refuses_a_completed_startup_without_a_posterior(
    monkeypatch: pytest.MonkeyPatch,
):
    """A complete startup alone is insufficient to form an aided trajectory."""
    startup = _completed_startup()

    monkeypatch.setattr(
        gnss_zupt_replay,
        "iter_gnss_imu_arrivals",
        lambda _raw_frames, _gnss_fixes: iter(()),
    )
    monkeypatch.setattr(
        gnss_zupt_replay,
        "_select_gnss_aided_startup",
        lambda _arrivals, _config, _recovery_processor: startup,
    )

    with pytest.raises(
        ValueError,
        match="requires at least one post-startup IMU increment",
    ):
        gnss_zupt_replay.run_gnss_zupt_replay([], [], _startup_config())


def test_unflagged_event_matches_one_clock_predict_without_mutating_prior():
    """An unflagged event is exactly one pure clock-aware predict step."""
    nominal = _nominal()
    covariance = _covariance()
    nominal_before = nominal.copy()
    covariance_before = covariance.copy()
    event = _event(stance_flag=False)
    dtheta, dv, dt = event.increment

    expected_nominal, expected_covariance = esekf_predict_clock(
        nominal,
        covariance,
        dtheta,
        dv,
        dt,
        _Q,
    )
    actual_nominal, actual_covariance = _apply_clock_imu_event(
        nominal,
        covariance,
        event,
        _Q,
        _R_ZUPT,
    )

    np.testing.assert_allclose(actual_nominal, expected_nominal, atol=1e-12)
    np.testing.assert_allclose(actual_covariance, expected_covariance, atol=1e-12)
    np.testing.assert_array_equal(nominal, nominal_before)
    np.testing.assert_array_equal(covariance, covariance_before)


def test_event_extra_process_covariance_augments_one_clock_predict():
    """An event-specific covariance is added to the normal predict covariance."""
    nominal = _nominal()
    covariance = _covariance()
    additional_covariance = np.diag(np.linspace(1e-5, 16e-5, 16))
    event = _event(
        stance_flag=False,
        additional_process_covariance=additional_covariance,
    )
    dtheta, dv, dt = event.increment

    expected_nominal, expected_covariance = esekf_predict_clock(
        nominal,
        covariance,
        dtheta,
        dv,
        dt,
        _Q + additional_covariance,
    )
    actual_nominal, actual_covariance = _apply_clock_imu_event(
        nominal,
        covariance,
        event,
        _Q,
        _R_ZUPT,
    )

    np.testing.assert_allclose(actual_nominal, expected_nominal, atol=1e-12)
    np.testing.assert_allclose(actual_covariance, expected_covariance, atol=1e-12)


def test_flagged_event_matches_predict_then_clock_zupt():
    """A flagged event updates the predicted, not pre-predict, posterior."""
    nominal = _nominal()
    covariance = _covariance()
    event = _event(stance_flag=True)
    dtheta, dv, dt = event.increment

    expected_nominal, expected_covariance = esekf_predict_clock(
        nominal,
        covariance,
        dtheta,
        dv,
        dt,
        _Q,
    )
    expected_nominal, expected_covariance = esekf_update_clock(
        expected_nominal,
        expected_covariance,
        _Z_ZUPT,
        _velocity_h,
        _H_VEL_CLOCK,
        _R_ZUPT,
    )
    actual_nominal, actual_covariance = _apply_clock_imu_event(
        nominal,
        covariance,
        event,
        _Q,
        _R_ZUPT,
    )

    np.testing.assert_allclose(actual_nominal, expected_nominal, atol=1e-12)
    np.testing.assert_allclose(actual_covariance, expected_covariance, atol=1e-12)


def test_clock_imu_event_owns_immutable_increment_and_stance_decision():
    """Delayed replay retains the original increment and stance decision."""
    dtheta = np.array([0.003, -0.002, 0.001])
    dv = np.array([0.04, -0.03, -0.09])
    event = ClockImuEvent(
        increment=(dtheta, dv, 0.01),
        stance_flag=True,
    )

    dtheta[0] = 99.0
    dv[1] = 88.0

    with pytest.raises(FrozenInstanceError):
        event.stance_flag = False
    with pytest.raises(ValueError):
        event.increment[0][0] = 77.0
    with pytest.raises(ValueError):
        event.increment[1][1] = 66.0

    np.testing.assert_allclose(event.increment[0], [0.003, -0.002, 0.001])
    np.testing.assert_allclose(event.increment[1], [0.04, -0.03, -0.09])
    assert event.additional_process_covariance is None


def test_clock_imu_event_owns_symmetric_additional_process_covariance():
    """A covariance copy tolerates roundoff skew and cannot be mutated later."""
    additional_covariance = np.eye(16)
    additional_covariance[0, 1] = 1e-14
    event = _event(
        stance_flag=False,
        additional_process_covariance=additional_covariance,
    )

    additional_covariance[0, 0] = 99.0

    assert event.additional_process_covariance is not None
    np.testing.assert_allclose(
        event.additional_process_covariance,
        event.additional_process_covariance.T,
        atol=0.0,
    )
    assert event.additional_process_covariance[0, 0] == pytest.approx(1.0)
    assert event.additional_process_covariance[0, 1] == pytest.approx(0.5e-14)
    with pytest.raises(ValueError):
        event.additional_process_covariance[0, 0] = 0.0


@pytest.mark.parametrize(
    ("additional_covariance", "message"),
    (
        (np.zeros((15, 15)), "shape"),
        (np.full((16, 16), np.nan), "finite"),
        (np.eye(16) + np.eye(16, k=1), "symmetric"),
        (np.diag(np.concatenate(([-1.0], np.ones(15)))), "positive semidefinite"),
    ),
)
def test_clock_imu_event_rejects_invalid_additional_process_covariance(
    additional_covariance: np.ndarray,
    message: str,
):
    """Saved event covariance must remain a valid 16-state covariance."""
    with pytest.raises(ValueError, match=message):
        _event(
            stance_flag=False,
            additional_process_covariance=additional_covariance,
        )


def test_aider_owns_constructor_inputs_and_returns_state_copies():
    """Caller mutation cannot bypass the IMU-event history contract."""
    nominal0 = _nominal()
    P0 = _covariance()
    Q = _Q.copy()
    R_zupt = _R_ZUPT.copy()
    aider = ClockAwareGnssIns(nominal0, P0, Q, R_zupt)

    nominal0[0] = 99.0
    P0[0, 0] = 88.0
    Q[0, 0] = 77.0
    R_zupt[0, 0] = 66.0
    nominal_view = aider.nominal
    covariance_view = aider.P
    nominal_view[1] = 55.0
    covariance_view[1, 1] = 44.0

    event = _event(stance_flag=True)
    expected_nominal, expected_covariance = _apply_clock_imu_event(
        _nominal(),
        _covariance(),
        event,
        _Q,
        _R_ZUPT,
    )
    aider.advance(event)

    np.testing.assert_allclose(aider.nominal, expected_nominal, atol=1e-12)
    np.testing.assert_allclose(aider.P, expected_covariance, atol=1e-12)
    assert aider.t_i == pytest.approx(0.01)


def test_aider_advances_posterior_time_and_private_history_in_event_order():
    """Every stored entry is the posterior of its original IMU/ZUPT event."""
    nominal0 = _nominal()
    P0 = _covariance()
    aider = ClockAwareGnssIns(nominal0, P0, _Q, _R_ZUPT)
    first_event = _event(stance_flag=False)
    second_event = _event(stance_flag=True)

    first_nominal, first_covariance = _apply_clock_imu_event(
        nominal0,
        P0,
        first_event,
        _Q,
        _R_ZUPT,
    )
    second_nominal, second_covariance = _apply_clock_imu_event(
        first_nominal,
        first_covariance,
        second_event,
        _Q,
        _R_ZUPT,
    )

    aider.advance(first_event)
    aider.advance(second_event)

    assert aider.t_i == pytest.approx(0.02)
    np.testing.assert_allclose(aider.nominal, second_nominal, atol=1e-12)
    np.testing.assert_allclose(aider.P, second_covariance, atol=1e-12)
    assert len(aider._history) == 2
    assert aider._history[0].event is first_event
    assert aider._history[1].event is second_event
    assert aider._history[0].t_i == pytest.approx(0.01)
    assert aider._history[1].t_i == pytest.approx(0.02)
    np.testing.assert_allclose(aider._history[0].nominal, first_nominal, atol=1e-12)
    np.testing.assert_allclose(
        aider._history[0].P,
        first_covariance,
        atol=1e-12,
    )


def test_delayed_gnss_update_result_distinguishes_all_outcomes_immutably():
    """Terminal GNSS outcomes cannot be conflated or changed."""
    accepted = DelayedGnssUpdate(
        status=DelayedGnssUpdateStatus.ACCEPTED,
        t_fix_i_hat=0.01,
        t_applied_i=0.01,
        nis=2.0,
    )
    rejected = DelayedGnssUpdate(
        status=DelayedGnssUpdateStatus.REJECTED,
        t_fix_i_hat=0.02,
        t_applied_i=0.02,
        nis=12.0,
    )
    stale = DelayedGnssUpdate(
        status=DelayedGnssUpdateStatus.STALE,
        t_fix_i_hat=-0.01,
        t_applied_i=None,
        nis=None,
    )
    future_dropped = DelayedGnssUpdate(
        status=DelayedGnssUpdateStatus.FUTURE_DROPPED,
        t_fix_i_hat=0.04,
        t_applied_i=None,
        nis=None,
    )

    assert accepted.status is DelayedGnssUpdateStatus.ACCEPTED
    assert rejected.status is DelayedGnssUpdateStatus.REJECTED
    assert stale.status is DelayedGnssUpdateStatus.STALE
    assert stale.t_applied_i is None
    assert stale.nis is None
    assert future_dropped.status is DelayedGnssUpdateStatus.FUTURE_DROPPED
    assert future_dropped.t_applied_i is None
    assert future_dropped.nis is None
    with pytest.raises(FrozenInstanceError):
        accepted.status = DelayedGnssUpdateStatus.REJECTED


def test_gnss_fusion_diagnostic_owns_read_only_evaluation_arrays():
    """A diagnostic cannot be changed through caller-owned measurement arrays."""
    position = np.array([1.0, -2.0, 0.5])
    innovation = np.array([0.3, -0.4, 0.1])
    innovation_covariance = np.diag([4.0, 5.0, 6.0])
    prior_position_covariance = np.diag([1.0, 2.0, 3.0])
    update = DelayedGnssUpdate(
        status=DelayedGnssUpdateStatus.ACCEPTED,
        t_fix_i_hat=0.2,
        t_applied_i=0.21,
        nis=2.0,
    )

    diagnostic = GnssFusionDiagnostic(
        fix=_startup_fix(1),
        t_fix_g=0.2,
        t_arrival_i=0.3,
        p_gnss_ned=position,
        tau_hat=-0.01,
        update=update,
        innovation_ned=innovation,
        innovation_covariance_ned=innovation_covariance,
        prior_position_covariance_ned=prior_position_covariance,
    )

    position[0] = 99.0
    innovation[0] = 99.0
    innovation_covariance[0, 0] = 99.0
    prior_position_covariance[0, 0] = 99.0

    np.testing.assert_allclose(diagnostic.p_gnss_ned, [1.0, -2.0, 0.5])
    np.testing.assert_allclose(diagnostic.innovation_ned, [0.3, -0.4, 0.1])
    np.testing.assert_allclose(
        diagnostic.innovation_covariance_ned,
        np.diag([4.0, 5.0, 6.0]),
    )
    np.testing.assert_allclose(
        diagnostic.prior_position_covariance_ned,
        np.diag([1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError):
        diagnostic.innovation_ned[0] = 0.0


def test_public_diagnostic_snapshot_survives_later_delayed_replay():
    """Later history rewrites cannot change an earlier public diagnostic."""
    aider, _ = _aider_with_three_events()
    first_tau_hat = aider.nominal[NOM_TAU]
    first_fix = _timed_fix(
        t_fix_g=0.01 - first_tau_hat,
        t_arrival_i=aider.t_i,
        p_ned=np.array([1.5, -0.5, 0.25]),
    )
    first_update, first_evaluation = aider._fuse_delayed_gnss_with_evaluation(
        first_fix,
        np.eye(3) * 0.5,
        1e6,
    )
    assert first_evaluation is not None
    diagnostic = GnssFusionDiagnostic(
        fix=_startup_fix(1),
        t_fix_g=first_fix.t_fix_g,
        t_arrival_i=first_fix.t_arrival_i,
        p_gnss_ned=first_fix.p_ned,
        tau_hat=first_evaluation.tau_hat,
        update=first_update,
        innovation_ned=first_evaluation.innovation_ned,
        innovation_covariance_ned=first_evaluation.innovation_covariance_ned,
        prior_position_covariance_ned=first_evaluation.prior_position_covariance_ned,
    )
    expected_innovation = diagnostic.innovation_ned.copy()
    expected_innovation_covariance = diagnostic.innovation_covariance_ned.copy()
    expected_prior_position_covariance = diagnostic.prior_position_covariance_ned.copy()

    second_tau_hat = aider.nominal[NOM_TAU]
    second_update = aider.fuse_delayed_gnss(
        _timed_fix(
            t_fix_g=0.02 - second_tau_hat,
            t_arrival_i=aider.t_i,
            p_ned=np.array([0.25, 0.5, -0.1]),
        ),
        np.eye(3) * 0.5,
        1e6,
    )

    assert second_update.status is DelayedGnssUpdateStatus.ACCEPTED
    np.testing.assert_array_equal(diagnostic.innovation_ned, expected_innovation)
    np.testing.assert_array_equal(
        diagnostic.innovation_covariance_ned,
        expected_innovation_covariance,
    )
    np.testing.assert_array_equal(
        diagnostic.prior_position_covariance_ned,
        expected_prior_position_covariance,
    )


def test_gnss_fusion_diagnostic_requires_status_consistent_evidence():
    """Only evaluated updates carry innovation and covariance snapshots."""
    stale_update = DelayedGnssUpdate(
        status=DelayedGnssUpdateStatus.STALE,
        t_fix_i_hat=-0.1,
        t_applied_i=None,
        nis=None,
    )
    stale = GnssFusionDiagnostic(
        fix=_startup_fix(2),
        t_fix_g=0.0,
        t_arrival_i=0.1,
        p_gnss_ned=np.zeros(3),
        tau_hat=0.0,
        update=stale_update,
        innovation_ned=None,
        innovation_covariance_ned=None,
        prior_position_covariance_ned=None,
    )
    assert stale.innovation_ned is None

    accepted_update = DelayedGnssUpdate(
        status=DelayedGnssUpdateStatus.ACCEPTED,
        t_fix_i_hat=0.1,
        t_applied_i=0.1,
        nis=1.0,
    )
    with pytest.raises(ValueError, match="requires innovation"):
        GnssFusionDiagnostic(
            fix=_startup_fix(3),
            t_fix_g=0.1,
            t_arrival_i=0.2,
            p_gnss_ned=np.zeros(3),
            tau_hat=0.0,
            update=accepted_update,
            innovation_ned=np.zeros(3),
            innovation_covariance_ned=None,
            prior_position_covariance_ned=np.eye(3),
        )

    with pytest.raises(ValueError, match="p_gnss_ned"):
        GnssFusionDiagnostic(
            fix=_startup_fix(4),
            t_fix_g=0.1,
            t_arrival_i=0.2,
            p_gnss_ned=np.zeros(2),
            tau_hat=0.0,
            update=stale_update,
            innovation_ned=None,
            innovation_covariance_ned=None,
            prior_position_covariance_ned=None,
        )


def test_find_update_index_snaps_to_first_inclusive_history_posterior():
    """Mapped GNSS time never splits an increment or moves backward in history."""
    aider = ClockAwareGnssIns(_nominal(), _covariance(), _Q, _R_ZUPT)
    aider.advance(_event(stance_flag=False))
    aider.advance(_event(stance_flag=True))
    aider.advance(_event(stance_flag=False))

    assert aider._find_update_index(0.0) == 0
    assert aider._find_update_index(0.01) == 0
    assert aider._find_update_index(0.015) == 1
    assert aider._find_update_index(0.02) == 1
    assert aider._find_update_index(0.025) == 2
    assert aider._find_update_index(0.03) == 2
    assert aider._find_update_index(0.03 + 0.5 * gnss_zupt_replay._TIMELINE_ATOL) == 2
    assert aider._find_update_index(0.031) is None


def _aider_with_three_events(
    additional_process_covariance: np.ndarray | None = None,
) -> tuple[ClockAwareGnssIns, tuple[ClockImuEvent, ...]]:
    """Return one three-posterior history with a replay-visible ZUPT event."""
    aider = ClockAwareGnssIns(_nominal(), _covariance(), _Q, _R_ZUPT)
    events = (
        _event(stance_flag=False),
        _event(
            stance_flag=True,
            additional_process_covariance=additional_process_covariance,
        ),
        _event(stance_flag=False),
    )
    for event in events:
        aider.advance(event)
    return aider, events


def _timed_fix(*, t_fix_g: float, t_arrival_i: float, p_ned: np.ndarray) -> TimedGnssFixNed:
    """Return one already-local-NED GNSS fix on the test clock convention."""
    return TimedGnssFixNed(
        t_fix_g=t_fix_g,
        t_arrival_i=t_arrival_i,
        p_ned=p_ned,
    )


def test_accepted_delayed_gnss_rebuilds_later_zupt_events_and_current_state():
    """A delayed replay preserves later event-specific process covariance."""
    additional_covariance = np.diag(np.linspace(1e-5, 16e-5, 16))
    aider, events = _aider_with_three_events(additional_covariance)
    original_history = tuple(aider._history)
    R_pos = np.eye(3) * 0.5
    tau_hat = aider.nominal[-1]
    fix = _timed_fix(
        t_fix_g=0.01 - tau_hat,
        t_arrival_i=aider.t_i,
        p_ned=np.array([1.5, -0.5, 0.25]),
    )

    expected_nominal, expected_P, expected_nis, expected_accepted = (
        gnss_position_update_ned_gated(
            original_history[0].nominal,
            original_history[0].P,
            fix.p_ned,
            R_pos,
            1e6,
        )
    )
    assert expected_accepted
    for event in events[1:]:
        expected_nominal, expected_P = _apply_clock_imu_event(
            expected_nominal,
            expected_P,
            event,
            _Q,
            _R_ZUPT,
        )

    result = aider.fuse_delayed_gnss(fix, R_pos, 1e6)

    assert result.status is DelayedGnssUpdateStatus.ACCEPTED
    assert result.t_fix_i_hat == pytest.approx(0.01)
    assert result.t_applied_i == pytest.approx(0.01)
    assert result.nis == pytest.approx(expected_nis)
    assert aider.t_i == pytest.approx(0.03)
    np.testing.assert_allclose(aider.nominal, expected_nominal, atol=1e-12)
    np.testing.assert_allclose(aider.P, expected_P, atol=1e-12)
    assert aider._history[0].event is events[0]
    assert aider._history[1].event is events[1]
    assert aider._history[2].event is events[2]
    assert events[1].additional_process_covariance is not None
    np.testing.assert_allclose(
        events[1].additional_process_covariance,
        additional_covariance,
        atol=0.0,
    )
    assert [entry.t_i for entry in aider._history] == pytest.approx([0.01, 0.02, 0.03])


def test_internal_delayed_fusion_captures_only_true_pre_update_evaluations():
    """Snapshots describe the snapped pre-update state, never replayed history."""
    aider, _ = _aider_with_three_events()
    tau_hat = aider.nominal[NOM_TAU]
    expected_entry = aider._history[0]
    R_pos = np.diag([0.5, 0.75, 1.25])
    fix = _timed_fix(
        t_fix_g=0.01 - tau_hat,
        t_arrival_i=aider.t_i,
        p_ned=np.array([1.5, -0.5, 0.25]),
    )

    accepted, evaluation = aider._fuse_delayed_gnss_with_evaluation(
        fix,
        R_pos,
        1e6,
    )

    assert accepted.status is DelayedGnssUpdateStatus.ACCEPTED
    assert evaluation is not None
    np.testing.assert_allclose(evaluation.tau_hat, tau_hat, atol=0.0)
    np.testing.assert_allclose(
        evaluation.innovation_ned,
        fix.p_ned - expected_entry.nominal[NOM_P],
        atol=0.0,
    )
    H = np.zeros((3, 16))
    H[:, :3] = np.eye(3)
    H[:, -1] = expected_entry.nominal[NOM_V]
    np.testing.assert_allclose(
        evaluation.innovation_covariance_ned,
        H @ expected_entry.P @ H.T + R_pos,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evaluation.prior_position_covariance_ned,
        expected_entry.P[:3, :3],
        atol=0.0,
    )

    rejected_aider, _ = _aider_with_three_events()
    rejected_tau_hat = rejected_aider.nominal[NOM_TAU]
    rejected, rejected_evaluation = rejected_aider._fuse_delayed_gnss_with_evaluation(
        _timed_fix(
            t_fix_g=0.01 - rejected_tau_hat,
            t_arrival_i=rejected_aider.t_i,
            p_ned=np.array([1000.0, -1000.0, 500.0]),
        ),
        R_pos,
        1e-6,
    )
    assert rejected.status is DelayedGnssUpdateStatus.REJECTED
    assert rejected_evaluation is not None

    stale_aider, _ = _aider_with_three_events()
    stale_tau_hat = stale_aider.nominal[NOM_TAU]
    stale, stale_evaluation = stale_aider._fuse_delayed_gnss_with_evaluation(
        _timed_fix(
            t_fix_g=-stale_tau_hat,
            t_arrival_i=stale_aider.t_i,
            p_ned=np.zeros(3),
        ),
        R_pos,
        1e6,
    )
    assert stale.status is DelayedGnssUpdateStatus.STALE
    assert stale_evaluation is None


def test_delayed_gnss_accepts_fix_delivered_before_current_posterior():
    """A runner-deferred fix remains causal when it arrived at an earlier epoch."""
    aider, _ = _aider_with_three_events()
    tau_hat = aider.nominal[-1]

    update = aider.fuse_delayed_gnss(
        _timed_fix(
            t_fix_g=0.01 - tau_hat,
            t_arrival_i=0.02,
            p_ned=np.array([1.5, -0.5, 0.25]),
        ),
        np.eye(3) * 0.5,
        1e6,
    )

    assert update.status is DelayedGnssUpdateStatus.ACCEPTED
    assert update.t_fix_i_hat == pytest.approx(0.01)
    assert update.t_applied_i == pytest.approx(0.01)
    assert update.nis is not None
    assert aider.t_i == pytest.approx(0.03)


def test_rejected_and_stale_delayed_fixes_are_strict_history_noops():
    """Expected unusable GNSS evidence never mutates current state or history."""
    aider, _ = _aider_with_three_events()
    history_before = tuple(aider._history)
    nominal_before = aider.nominal
    P_before = aider.P
    tau_hat = aider.nominal[-1]

    rejected = aider.fuse_delayed_gnss(
        _timed_fix(
            t_fix_g=0.01 - tau_hat,
            t_arrival_i=aider.t_i,
            p_ned=np.array([1000.0, -1000.0, 500.0]),
        ),
        np.eye(3) * 0.01,
        1.0,
    )

    assert rejected.status is DelayedGnssUpdateStatus.REJECTED
    assert rejected.t_applied_i == pytest.approx(0.01)
    assert rejected.nis is not None
    assert tuple(aider._history) == history_before
    np.testing.assert_array_equal(aider.nominal, nominal_before)
    np.testing.assert_array_equal(aider.P, P_before)

    stale = aider.fuse_delayed_gnss(
        _timed_fix(
            t_fix_g=0.0 - tau_hat,
            t_arrival_i=aider.t_i,
            p_ned=np.zeros(3),
        ),
        np.eye(3),
        1e6,
    )

    assert stale.status is DelayedGnssUpdateStatus.STALE
    assert stale.t_applied_i is None
    assert stale.nis is None
    assert tuple(aider._history) == history_before
    np.testing.assert_array_equal(aider.nominal, nominal_before)
    np.testing.assert_array_equal(aider.P, P_before)


def test_delayed_gnss_rejects_empty_future_arrival_and_future_mapped_contracts():
    """Broken causal delivery contracts fail rather than silently misfusing."""
    empty_aider = ClockAwareGnssIns(_nominal(), _covariance(), _Q, _R_ZUPT)
    empty_fix = _timed_fix(t_fix_g=12.0, t_arrival_i=0.0, p_ned=np.zeros(3))

    with pytest.raises(RuntimeError, match="first IMU posterior"):
        empty_aider.fuse_delayed_gnss(empty_fix, np.eye(3), 1e6)

    aider, _ = _aider_with_three_events()
    tau_hat = aider.nominal[-1]
    with pytest.raises(RuntimeError, match="cannot lie after the current"):
        aider.fuse_delayed_gnss(
            _timed_fix(
                t_fix_g=0.01 - tau_hat,
                t_arrival_i=aider.t_i + 0.01,
                p_ned=np.zeros(3),
            ),
            np.eye(3),
            1e6,
        )
    with pytest.raises(ValueError, match="after the current"):
        aider.fuse_delayed_gnss(
            _timed_fix(
                t_fix_g=0.04 - tau_hat,
                t_arrival_i=aider.t_i,
                p_ned=np.zeros(3),
            ),
            np.eye(3),
            1e6,
        )


def test_posterior_snapshots_are_immutable_and_reflect_delayed_replay():
    """Public snapshots own corrected history without exposing filter arrays."""
    aider, _events = _aider_with_three_events()
    snapshots_before = aider.posteriors

    assert all(isinstance(snapshot, ClockImuPosterior) for snapshot in snapshots_before)
    assert [snapshot.t_i for snapshot in snapshots_before] == pytest.approx(
        [0.01, 0.02, 0.03]
    )
    np.testing.assert_allclose(snapshots_before[-1].nominal, aider.nominal)
    np.testing.assert_allclose(snapshots_before[-1].P, aider.P)
    with pytest.raises(ValueError):
        snapshots_before[0].nominal[0] = 99.0
    with pytest.raises(ValueError):
        snapshots_before[0].P[0, 0] = 99.0

    tau_hat = aider.nominal[-1]
    update = aider.fuse_delayed_gnss(
        _timed_fix(
            t_fix_g=0.01 - tau_hat,
            t_arrival_i=aider.t_i,
            p_ned=np.array([1.5, -0.5, 0.25]),
        ),
        np.eye(3) * 0.5,
        1e6,
    )
    snapshots_after = aider.posteriors

    assert update.status is DelayedGnssUpdateStatus.ACCEPTED
    assert not np.allclose(
        snapshots_after[-1].nominal,
        snapshots_before[-1].nominal,
    )
    np.testing.assert_allclose(snapshots_after[-1].nominal, aider.nominal)
    np.testing.assert_allclose(snapshots_after[-1].P, aider.P)
