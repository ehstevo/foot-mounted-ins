"""Causally replay INS propagation with zero-velocity pseudo-measurements.

The startup phase selects the first qualifying contiguous real stance, then
uses all of its increments for static leveling and gyro-bias initialization.
The next real increment begins estimation at local replay time zero with a
fresh SHOE detector. Each subsequent real increment follows the fixed sequence
``predict -> causal stance decision -> optional ZUPT correction``.

Positions and velocities represented by the nominal state use a run-local
Forward-Right-Down (FRD) frame, not geographic North-East-Down (NED). A
permitted post-initialization notification gap is represented by explicit
synthetic held-increment propagation: it receives added coast covariance, no
stance decision, and no ZUPT; the detector is reset before real data resumes.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from bootins.bmu.stream import iter_tracked_frames
from bootins.bmu.transport import RawFrame
from bootins.estimation.esekf import (
    NOM_BG,
    NOM_BA,
    NOM_Q,
    NOM_V,
    esekf_predict,
    esekf_update,
)
from bootins.estimation.initialization import (
    StaticZuptInitialization,
    StanceInitializationInterval,
    StanceInitializationSelector,
    initialize_static_zupt,
)
from bootins.replay.imu_gap_recovery import (
    ImuCoastPlan,
    ImuFrameEvent,
    ImuGapRecoveryPolicy,
    ImuGapRecoveryProcessor,
)
from bootins.estimation.mechanization import Increment
from bootins.aiding.stance_glrt import ShoeStanceDetector


class _ReplayState(Enum):
    """Represent the forward-only startup and estimation phases."""

    INITIALIZING = auto()
    ESTIMATING = auto()


def _vel_h(nominal: np.ndarray) -> np.ndarray:
    """Return the velocity predicted by the zero-velocity measurement model."""
    return nominal[NOM_V]


_H_VEL = np.zeros((3, 15), dtype=float)
_H_VEL[:, 3:6] = np.eye(3)
_Z_ZUPT = np.zeros(3, dtype=float)


@dataclass(frozen=True)
class ZuptReplayResult:
    """Store one causally processed INS-plus-ZUPT replay.

    Attributes:
        initialization: Static leveling and gyro-bias result derived from the
            selected real startup stance.
        initialization_interval: Provenance for that startup stance within the
            recorded increment stream.
        time: ``(N + 1,)`` local replay time in s. Row zero is the
            initialized prior immediately before the first estimation increment.
        nominals: ``(N + 1, 16)`` nominal-state history in the run-local FRD
            frame. Each row is the posterior at the corresponding ``time``.
        covariances: ``(N + 1, 15, 15)`` error-state posterior covariance
            history aligned one-to-one with ``nominals``.
        stance_flags: ``(N,)`` causal SHOE decisions. Flag ``k`` belongs to the
            increment producing posterior row ``k + 1``.
        coast_plans: Immutable provenance for synthetic post-startup coast
            intervals; empty when all operational IMU frames were contiguous.

    Notes:
        A synthetic coast always has a ``False`` stance flag, because it is
        reconstructed propagation rather than an observed IMU increment and
        cannot create a ZUPT correction.
    """

    initialization: StaticZuptInitialization
    initialization_interval: StanceInitializationInterval
    time: np.ndarray
    nominals: np.ndarray
    covariances: np.ndarray
    stance_flags: np.ndarray
    coast_plans: tuple[ImuCoastPlan, ...] = ()


def _new_detector(
    window_size: int,
    threshold: float,
    sigma_a: float,
    sigma_g: float,
) -> ShoeStanceDetector:
    """Construct an independent causal SHOE detector.

    Args:
        window_size: Number of trailing increments in each GLRT window.
        threshold: GLRT stance-decision threshold.
        sigma_a: Accelerometer-rate noise scale in m/s².
        sigma_g: Gyroscope-rate noise scale in rad/s.

    Returns:
        A detector with no retained increment history.
    """
    return ShoeStanceDetector(
        window_size=window_size,
        threshold=threshold,
        sigma_a=sigma_a,
        sigma_g=sigma_g,
    )


def run_zupt_replay(
    raw_frames: Iterable[RawFrame],
    *,
    minimum_stance_duration: float,
    startup_search_limit: float,
    initial_accel_bias_b: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
    Q_coast: np.ndarray,
    R_zupt: np.ndarray,
    window_size: int,
    threshold: float,
    sigma_a: float,
    sigma_g: float,
) -> ZuptReplayResult:
    """Replay one IMU capture using causal ZUPT-aided INS propagation.

    Args:
        raw_frames: Complete recorded BMU frames in source-arrival order.
        minimum_stance_duration: Required selected-startup stance duration in s.
        startup_search_limit: Maximum startup search duration in s.
        initial_accel_bias_b: ``(3,)`` initial body-frame accelerometer bias in
            m/s² for the nominal state.
        P0: ``(15, 15)`` error-state covariance at local replay time zero.
        Q: ``(15, 15)`` process covariance added for every real increment.
        Q_coast: ``(15, 15)`` additional process covariance for every synthetic
            coast increment.
        R_zupt: ``(3, 3)`` covariance of the zero-velocity pseudo-measurement
            in (m/s)².
        window_size: Number of trailing increments used by the SHOE detector.
        threshold: SHOE GLRT stance-decision threshold.
        sigma_a: Accelerometer-rate noise scale in m/s².
        sigma_g: Gyroscope-rate noise scale in rad/s.

    Returns:
        Initialized, row-aligned nominal-state and covariance histories in the
        run-local FRD frame, with causal stance and coast provenance.

    Raises:
        ValueError: A source, structural-quality, or unrecoverable-gap failure
            occurs; startup crosses a gap or does not qualify; or no increment
            follows initialization.
        RuntimeError: The IMU recovery processor emits an unsupported event.

    Notes:
        The whole duration-crossing startup increment remains initialization
        data. The following increment begins estimation with a new detector, so
        startup history cannot create an immediate ZUPT. Synthetic coasts use
        ``Q + Q_coast`` and a forced false stance flag; real increments use
        ``Q`` and can receive the usual ZUPT correction.
    """
    initialization_detector = _new_detector(
        window_size,
        threshold,
        sigma_a,
        sigma_g,
    )
    selector = StanceInitializationSelector(
        minimum_stance_duration=minimum_stance_duration,
        startup_search_limit=startup_search_limit,
    )

    state = _ReplayState.INITIALIZING
    startup_increments: list[Increment] = []
    initialization: StaticZuptInitialization | None = None
    initialization_interval: StanceInitializationInterval | None = None
    estimation_detector: ShoeStanceDetector | None = None
    nominal: np.ndarray | None = None
    covariance: np.ndarray | None = None
    time_history: list[float] = []
    nominals: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    stance_flags: list[bool] = []
    recovery_processor = ImuGapRecoveryProcessor(ImuGapRecoveryPolicy())
    coast_plans: list[ImuCoastPlan] = []

    for tracked_frame in iter_tracked_frames(raw_frames):
        for event in recovery_processor.update(tracked_frame):
            if isinstance(event, ImuCoastPlan):
                if state is _ReplayState.INITIALIZING:
                    raise ValueError(
                        "Static initialization cannot cross a BMU sequence gap."
                    )
                assert nominal is not None
                assert covariance is not None
                assert estimation_detector is not None

                estimation_detector.reset()
                coast_plans.append(event)
                for increment in event.increments:
                    dtheta, dv, dt = increment
                    nominal, covariance = esekf_predict(
                        nominal,
                        covariance,
                        dtheta,
                        dv,
                        dt,
                        Q + Q_coast,
                    )
                    time_history.append(time_history[-1] + dt)
                    nominals.append(nominal)
                    covariances.append(covariance)
                    stance_flags.append(False)
                continue

            if not isinstance(event, ImuFrameEvent):
                raise RuntimeError("IMU recovery processor emitted an unknown event.")

            for increment in event.increments:
                if state is _ReplayState.INITIALIZING:
                    startup_increments.append(increment)
                    stance_flag = initialization_detector.update(increment)
                    selected_interval = selector.update(increment, stance_flag)
                    if selected_interval is None:
                        continue

                    initialization_interval = selected_interval
                    initialization_increments = startup_increments[
                        initialization_interval.start_index :
                        initialization_interval.replay_start_index
                    ]
                    initialization = initialize_static_zupt(initialization_increments)
                    nominal = np.zeros(16, dtype=float)
                    nominal[NOM_Q] = initialization.static_level.q_fb
                    nominal[NOM_BG] = initialization.gyro_bias_b
                    nominal[NOM_BA] = initial_accel_bias_b
                    covariance = np.array(P0, dtype=float, copy=True)
                    time_history.append(0.0)
                    nominals.append(nominal)
                    covariances.append(covariance)
                    estimation_detector = _new_detector(
                        window_size,
                        threshold,
                        sigma_a,
                        sigma_g,
                    )
                    state = _ReplayState.ESTIMATING
                    continue

                assert nominal is not None
                assert covariance is not None
                assert estimation_detector is not None

                dtheta, dv, dt = increment
                nominal, covariance = esekf_predict(
                    nominal,
                    covariance,
                    dtheta,
                    dv,
                    dt,
                    Q,
                )
                stance_flag = estimation_detector.update(increment)
                if stance_flag:
                    nominal, covariance = esekf_update(
                        nominal,
                        covariance,
                        _Z_ZUPT,
                        _vel_h,
                        _H_VEL,
                        R_zupt,
                    )

                time_history.append(time_history[-1] + dt)
                nominals.append(nominal)
                covariances.append(covariance)
                stance_flags.append(stance_flag)

    if initialization is None or initialization_interval is None:
        raise ValueError("Could not initialize from a qualifying startup stance.")
    if not stance_flags:
        raise ValueError(
            "ZUPT replay requires at least one increment after initialization."
        )

    return ZuptReplayResult(
        initialization=initialization,
        initialization_interval=initialization_interval,
        time=np.asarray(time_history, dtype=float),
        nominals=np.stack(nominals),
        covariances=np.stack(covariances),
        stance_flags=np.asarray(stance_flags, dtype=bool),
        coast_plans=tuple(coast_plans),
    )
