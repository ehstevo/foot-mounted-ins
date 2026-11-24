"""Causally replay a raw, unaided inertial-navigation trajectory.

The replay consumes recorded :class:`~bootins.bmu.transport.RawFrame` evidence
in source order, initializes a run-local Forward-Right-Down (FRD) navigation
frame from a caller-declared real static interval, and mechanizes every later
increment. This local FRD frame is not a geographic North-East-Down (NED)
frame: its Forward axis is set by the initial body orientation rather than by
geographic North.

Sequence tracking and structural-quality assessment occur at the IMU input
boundary. After initialization, the bounded recovery policy may synthesize one
missing notification from previously observed rates; that reconstructed
interval remains explicit in the result. This module deliberately provides no
ZUPT, GNSS aiding, bias estimation, covariance propagation, or uncertainty
inflation. Those responsibilities belong to the corresponding aided replays.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from bootins.bmu.stream import iter_tracked_frames
from bootins.bmu.transport import RawFrame
from bootins.replay.imu_gap_recovery import (
    ImuCoastPlan,
    ImuFrameEvent,
    ImuGapRecoveryPolicy,
    ImuGapRecoveryProcessor,
)
from bootins.estimation.initialization import StaticLevelInitialization, initialize_static_level
from bootins.estimation.mechanization import Increment, State, mechanize_step


class _ReplayState(Enum):
    """Represent the forward-only initialization and mechanization phases."""

    INITIALIZING = auto()
    ESTIMATING = auto()


@dataclass(frozen=True)
class RawINSReplayResult:
    """Store one initialized raw-INS trajectory and its recovery provenance.

    Attributes:
        initialization: Static leveling result derived only from the real
            initialization increments.
        time: ``(N + 1,)`` replay time in s, with zero at the initialized
            state immediately after the initialization-boundary increment.
        positions_n: ``(N + 1, 3)`` positions in m in the run-local FRD frame.
        velocities_n: ``(N + 1, 3)`` velocities in m/s in the run-local FRD
            frame.
        quaternions_nb: ``(N + 1, 4)`` scalar-first body-to-run-local-FRD
            quaternions ``[w, x, y, z]``.
        coast_plans: Immutable provenance records for bounded synthetic
            propagation intervals. An empty tuple means no such interval was
            used after initialization.

    Notes:
        Row zero is the initialized zero-position, zero-velocity state. Row
        ``k + 1`` is the posterior after post-initialization increment ``k``;
        therefore all trajectory arrays share the same row alignment.
    """

    initialization: StaticLevelInitialization
    time: np.ndarray
    positions_n: np.ndarray
    velocities_n: np.ndarray
    quaternions_nb: np.ndarray
    coast_plans: tuple[ImuCoastPlan, ...] = ()


def _validated_initialization_duration(initialization_duration: float) -> float:
    """Validate a caller-declared static-initialization duration.

    Args:
        initialization_duration: Requested real static duration (s).

    Returns:
        The finite, strictly positive duration as ``float``.

    Raises:
        ValueError: The value cannot be converted to a finite positive number.
    """
    try:
        duration = float(initialization_duration)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Initialization duration must be finite and positive."
        ) from error

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("Initialization duration must be finite and positive.")

    return duration


def run_raw_ins_replay(
    raw_frames: Iterable[RawFrame],
    initialization_duration: float,
) -> RawINSReplayResult:
    """Replay captured IMU evidence as an unaided inertial trajectory.

    Args:
        raw_frames: Complete recorded BMU frames in source-arrival order.
        initialization_duration: Required duration (s) of real contiguous
            static increments used to level the run-local FRD frame.

    Returns:
        A row-aligned raw-INS trajectory whose time origin is the initialized
        state immediately after the whole increment that crosses the requested
        initialization duration.

    Raises:
        ValueError: An input, sequence, or structural-quality failure occurs;
            initialization crosses a gap; the capture ends before initialization
            completes; or no increment remains for estimation.
        RuntimeError: The IMU recovery processor emits an unsupported event.

    Notes:
        Initialization consumes only actual recorded increments. The increment
        that first reaches the requested duration belongs wholly to
        initialization; the following real increment begins estimation. After
        initialization only, a qualifying missing notification is propagated
        from held prior evidence before its revealing real frame. This is a
        bounded reconstructed interval, not a recovered measurement.
    """
    initialization_duration = _validated_initialization_duration(initialization_duration)
    recovery_processor = ImuGapRecoveryProcessor(ImuGapRecoveryPolicy())
    initialization_increments: list[Increment] = []
    initialization_elapsed = 0.0
    initialization: StaticLevelInitialization | None = None
    replay_state = _ReplayState.INITIALIZING
    current_state: State | None = None
    state_history: list[State] = []
    time_history: list[float] = []
    replay_time = 0.0
    coast_plans: list[ImuCoastPlan] = []

    for tracked_frame in iter_tracked_frames(raw_frames):
        for event in recovery_processor.update(tracked_frame):
            if isinstance(event, ImuCoastPlan):
                if replay_state is _ReplayState.INITIALIZING:
                    raise ValueError(
                        "Static initialization cannot cross a BMU sequence gap."
                    )
                assert current_state is not None
                coast_plans.append(event)
                for increment in event.increments:
                    current_state = mechanize_step(current_state, *increment)
                    replay_time += increment[2]
                    time_history.append(replay_time)
                    state_history.append(current_state)
            elif isinstance(event, ImuFrameEvent):
                for increment in event.increments:
                    if replay_state is _ReplayState.INITIALIZING:
                        initialization_increments.append(increment)
                        initialization_elapsed += increment[2]
                        if initialization_elapsed >= initialization_duration:
                            initialization = initialize_static_level(
                                initialization_increments
                            )
                            replay_state = _ReplayState.ESTIMATING
                            time_history.append(0.0)
                            current_state = (np.zeros(3), np.zeros(3), initialization.q_fb)
                            state_history.append(current_state)
                        continue

                    assert current_state is not None
                    current_state = mechanize_step(current_state, *increment)
                    replay_time += increment[2]
                    time_history.append(replay_time)
                    state_history.append(current_state)
            else:
                raise RuntimeError("IMU recovery processor emitted an unknown event.")

    if initialization is None:
        raise ValueError(
            "Capture ends before the requested initialization duration: "
            f"got {initialization_elapsed:.6g} s, need {initialization_duration:.6g} s."
        )
    if len(state_history) == 1:
        raise ValueError(
            "Raw INS replay requires at least one increment after initialization."
        )

    return RawINSReplayResult(
        initialization=initialization,
        time=np.asarray(time_history, dtype=float),
        positions_n=np.stack([state[0] for state in state_history]),
        velocities_n=np.stack([state[1] for state in state_history]),
        quaternions_nb=np.stack([state[2] for state in state_history]),
        coast_plans=tuple(coast_plans),
    )
