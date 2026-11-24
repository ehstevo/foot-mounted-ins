"""Orchestrate causal GNSS, INS, and ZUPT replay from recorded evidence.

The runner merges host-arrival-ordered GNSS and IMU evidence, selects a
GNSS-aided startup, initializes a clock-augmented INS, and advances each
post-startup IMU increment with causal stance detection. It releases GNSS only
at real processed-IMU frame epochs, converts receiver LLH to a terminal-static
local NED frame, and delegates delayed measurement mapping, update, and
historical reconstruction to :mod:`bootins.replay.gnss_fusion_engine`.

Terminal static evidence supplies the authoritative attitude, gyro bias, and
zero velocity. Receiver fixes collected during that stance together with the
seed receiver fix define the ECEF-averaged geographic origin; the seed also
independently defines GNSS-relative time zero. Host arrival establishes
availability only; it is not GNSS measurement time. Bounded IMU coasts are
explicit synthetic propagation and never trigger GNSS scheduling. This module
assembles the corrected history and diagnostic evidence; it does not implement
the ESEKF or GNSS measurement model.
"""
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from bootins.aiding.gnss_position_clock import gnss_time_to_imu_time
from bootins.aiding.stance_glrt import ShoeStanceDetector
from bootins.bmu.stream import TrackedFrame
from bootins.bmu.transport import RawFrame
from bootins.config.gnss_ins_zupt import GnssZuptConfig
from bootins.estimation.clock_augmented_esekf import NOM_BA, NOM_BG, NOM_Q, NOM_TAU
from bootins.estimation.initialization import GnssAidedStartup, GnssAidedStartupSelector
from bootins.frames.geodesy import llh_to_ned
from bootins.replay.clock_delayed_gnss import TimedGnssFixNed
from bootins.replay.gnss_arrivals import iter_gnss_imu_arrivals
from bootins.replay.gnss_fusion_engine import (
    ClockAwareGnssIns,
    _GnssPreUpdateEvaluation,
    _apply_clock_imu_event,
)
from bootins.replay.gnss_fusion_models import (
    ClockImuEvent,
    ClockImuPosterior,
    DelayedGnssUpdate,
    DelayedGnssUpdateStatus,
    GnssFusionDiagnostic,
    _TIMELINE_ATOL,
)
from bootins.replay.gnss_schedule import (
    GnssArrivalScheduler,
    GnssScheduleSummary,
    ImuArrivalEpoch,
    ScheduledGnssFix,
)
from bootins.replay.imu_gap_recovery import (
    ImuCoastPlan,
    ImuFrameEvent,
    ImuGapRecoveryPolicy,
    ImuGapRecoveryProcessor,
)
from bootins.sensors.gnss_observations import GnssFixLLH

def _select_gnss_aided_startup(
    gnss_imu_arrivals: Iterator[GnssFixLLH | TrackedFrame],
    config: GnssZuptConfig,
    recovery_processor: ImuGapRecoveryProcessor,
) -> GnssAidedStartup:
    """Select GNSS-aided startup from causal arrivals without over-consuming.

    Args:
        gnss_imu_arrivals: Shared iterator of host-arrival-ordered receiver fixes
            and tracked IMU frames.
        config: Aided-startup and SHOE detector settings.
        recovery_processor: Stateful factual-frame validator and bounded coast
            planner shared with subsequent operational replay.

    Returns:
        Startup evidence containing the initial and terminal static intervals,
        course alignment, terminal-static geographic origin, and seed fix.

    Raises:
        ValueError: Structural IMU evidence, a startup sequence gap, or the
            startup selector prevents a qualifying initialization.

    Notes:
        Events are consumed strictly in recorded host-arrival order. All startup
        stance, course, terminal-static, origin, and seed evidence must be real
        and contiguous; a generated coast plan therefore rejects startup. The
        function returns immediately when the terminal seed completes startup,
        leaving the iterator at the first post-seed event for operational replay.
    """
    startup_selector = GnssAidedStartupSelector(
        initial_stance_duration=config.initial_stance_duration,
        initial_stance_search_limit=config.initial_stance_search_limit,
        terminal_stance_duration=config.terminal_stance_duration,
        terminal_stance_search_limit=config.terminal_stance_search_limit,
        course_minimum_fix_count=config.course_minimum_fix_count,
        course_minimum_baseline=config.course_minimum_baseline,
        course_maximum_cross_track_rms=config.course_maximum_cross_track_rms,
    )

    stance_detector = ShoeStanceDetector(
        window_size=config.window_size,
        threshold=config.threshold,
        sigma_a=config.sigma_a,
        sigma_g=config.sigma_g,
    )

    for event in gnss_imu_arrivals:
        if isinstance(event, GnssFixLLH):
            startup = startup_selector.update_gnss(event)
            if startup is not None:
                return startup
        elif isinstance(event, TrackedFrame):
            imu_events = recovery_processor.update(event)
            for imu_event in imu_events:
                if isinstance(imu_event, ImuCoastPlan):
                    raise ValueError(
                        "GNSS-aided startup cannot cross a BMU sequence gap."
                    )
                elif isinstance(imu_event, ImuFrameEvent):
                    for increment in imu_event.increments:
                        stance_flag = stance_detector.update(increment)
                        startup = startup_selector.update_imu(increment, stance_flag)
                        if startup is not None:
                            return startup
    return startup_selector.finish()


def _initial_clock_nominal(
    startup: GnssAidedStartup,
    initial_accel_bias_b: np.ndarray,
) -> np.ndarray:
    """Build the clock-augmented nominal state at the initialization epoch.

    Args:
        startup: Qualified GNSS-aided startup evidence.
        initial_accel_bias_b: ``(3,)`` initial body-frame accelerometer bias in
            m/s².

    Returns:
        A newly allocated 17-state nominal with zero local-NED position,
        terminal-static zero velocity, terminal attitude and gyro bias, the
        supplied accelerometer bias, and zero clock offset.

    Notes:
        ``startup.static_origin`` defines the geographic meaning of local-NED
        position zero, while ``startup.seed_fix`` defines GNSS-relative time
        zero. Terminal-static attitude and gyro bias are authoritative because
        that stance immediately precedes the first operational IMU increment.
    """
    nominal = np.zeros(NOM_TAU + 1, dtype=float)
    nominal[NOM_Q] = startup.terminal_initialization.q_nb
    nominal[NOM_BG] = startup.terminal_initialization.terminal_static.gyro_bias_b
    nominal[NOM_BA] = initial_accel_bias_b
    return nominal


def _to_timed_gnss_fix_ned(
    scheduled_fix: ScheduledGnssFix,
    startup: GnssAidedStartup,
) -> TimedGnssFixNed:
    """Convert a causally delivered receiver fix to replay fusion coordinates.

    Args:
        scheduled_fix: Original receiver evidence and its scheduler-assigned
            local IMU delivery time.
        startup: Qualified terminal-static origin and seed-time reference.

    Returns:
        A local-NED position fix with GNSS time relative to the seed fix and
        unchanged scheduler delivery time.

    Notes:
        The ECEF-averaged origin, formed from terminal-stance fixes and the seed
        fix, fixes the local geographic NED frame. The seed independently also
        establishes GNSS-relative time zero. This structural conversion does
        not gate receiver validity or alter availability evidence.
    """
    fix = scheduled_fix.fix
    static_origin = startup.static_origin
    seed_time_reference = startup.seed_fix.t_fix_g
    p_ned = llh_to_ned(
        fix.latitude,
        fix.longitude,
        fix.height_ellipsoid,
        static_origin.latitude,
        static_origin.longitude,
        static_origin.height_ellipsoid,
    )

    return TimedGnssFixNed(
        t_fix_g=fix.t_fix_g - seed_time_reference,
        t_arrival_i=scheduled_fix.t_arrival_i,
        p_ned=p_ned,
    )


def _build_gnss_fusion_diagnostic(
    fix_llh: GnssFixLLH,
    fix_ned: TimedGnssFixNed,
    tau_hat: float,
    update: DelayedGnssUpdate,
    update_evaluation: _GnssPreUpdateEvaluation | None,
) -> GnssFusionDiagnostic:
    """Combine a terminal fusion outcome with receiver and timing provenance.

    Args:
        fix_llh: Original receiver fix retained as immutable evidence.
        fix_ned: Seed-relative time and local-NED conversion for that fix.
        tau_hat: Pre-outcome clock estimate in s when no evaluation snapshot is
            available.
        update: Terminal delayed-GNSS timing and gating outcome.
        update_evaluation: Optional pre-update filter evidence from a snapped
            historical posterior.

    Returns:
        A diagnostic record with owned public array snapshots when evaluation
        occurred, or absent snapshots for stale and future-dropped fixes.

    Notes:
        An accepted or NIS-rejected outcome uses the evaluation's pre-update
        clock estimate and evidence. A stale or future-dropped outcome has no
        measurement evaluation and therefore no innovation or covariance data.
    """
    if update_evaluation is None:
        innovation_ned = None
        innovation_covariance_ned = None
        prior_position_covariance_ned = None
    else:
        tau_hat = update_evaluation.tau_hat
        innovation_ned = update_evaluation.innovation_ned
        innovation_covariance_ned = update_evaluation.innovation_covariance_ned
        prior_position_covariance_ned = (
            update_evaluation.prior_position_covariance_ned
        )

    return GnssFusionDiagnostic(
        fix=fix_llh,
        t_fix_g=fix_ned.t_fix_g,
        t_arrival_i=fix_ned.t_arrival_i,
        p_gnss_ned=fix_ned.p_ned,
        tau_hat=tau_hat,
        update=update,
        innovation_ned=innovation_ned,
        innovation_covariance_ned=innovation_covariance_ned,
        prior_position_covariance_ned=prior_position_covariance_ned,
    )


@dataclass(frozen=True)
class _DeferredGnssFix:
    """Retain one valid fix awaiting its one later real-frame opportunity.

    Attributes:
        fix: Original receiver evidence.
        timed_fix: Corresponding seed-relative local-NED observation already
            assigned its original scheduler delivery epoch.

    Notes:
        This private record exists only when the mapped measurement epoch is
        later than the current posterior. The next nonempty real IMU frame
        either permits fusion or produces a future-dropped outcome.
    """

    fix: GnssFixLLH
    timed_fix: TimedGnssFixNed


@dataclass(frozen=True)
class GnssZuptReplayResult:
    """Store complete evidence from one clock-aware GNSS/INS/ZUPT replay.

    Attributes:
        startup: Qualified GNSS-aided initialization evidence, including the
            terminal-static origin and seed receiver fix.
        time: ``(N + 1,)`` local replay time in s, with row zero at the
            initialized seed-time epoch and one later row per post-startup IMU
            increment, real or synthetic.
        nominals: ``(N + 1, 17)`` clock-augmented nominal-state posteriors in
            static-origin local NED coordinates, aligned with ``time``.
        covariances: ``(N + 1, 16, 16)`` clock-augmented error-state posterior
            covariances aligned one-to-one with ``nominals``.
        stance_flags: ``(N,)`` causal SHOE decisions. Flag ``k`` belongs to the
            increment producing posterior row ``k + 1``; synthetic coasts are
            always ``False``.
        gnss_updates: Terminal outcomes for all receiver-valid scheduled fixes,
            in causal terminal-outcome order.
        gnss_fusion_diagnostics: Same-order detailed receiver, timing, and
            optional pre-update evidence corresponding one-to-one with
            ``gnss_updates``.
        invalid_gnss_fix_count: Number of scheduled receiver-invalid fixes that
            were counted but never passed to the filter.
        coast_plans: Immutable observed-gap provenance and synthetic-motion plans
            used after startup; empty if operational IMU evidence was contiguous.
        summary: Counts of GNSS fixes unavailable before the first or after the
            final usable IMU posterior.

    Notes:
        The initialized row contains the terminal-static state before the first
        operational increment. Delayed accepted GNSS corrections are reflected
        retrospectively in their corrected historical row and every later
        reconstructed posterior. ``time`` is a replay product; receiver
        measurement epochs remain in the GNSS diagnostic evidence.
    """

    startup: GnssAidedStartup
    time: np.ndarray
    nominals: np.ndarray
    covariances: np.ndarray
    stance_flags: np.ndarray
    gnss_updates: tuple[DelayedGnssUpdate, ...]
    gnss_fusion_diagnostics: tuple[GnssFusionDiagnostic, ...]
    invalid_gnss_fix_count: int
    coast_plans: tuple[ImuCoastPlan, ...]
    summary: GnssScheduleSummary


def run_gnss_zupt_replay(
    raw_frames: Iterable[RawFrame],
    gnss_fixes: Iterable[GnssFixLLH],
    config: GnssZuptConfig,
    gnss_outage: float | None = None,
) -> GnssZuptReplayResult:
    """Replay causal GNSS, INS, and ZUPT evidence from one recorded run.

    Args:
        raw_frames: Recorded BMU frames in source host-arrival order.
        gnss_fixes: Receiver fixes in nondecreasing host-arrival order. Each
            record retains separate receiver measurement and host-arrival times.
        config: Validated startup, detector, ESEKF, ZUPT, GNSS, and coast-noise
            settings.
        gnss_outage: Optional local replay time in s at which subsequently
            arriving GNSS evidence is excluded. ``None`` retains all fixes.

    Returns:
        Row-aligned initialized and corrected posterior histories, causal stance
        flags, bounded-coast provenance, scheduler summary, and complete
        receiver-valid GNSS terminal-outcome diagnostics.

    Raises:
        ValueError: Source validation or startup fails, startup crosses a gap,
            an IMU gap exceeds bounded recovery, or no post-startup IMU increment
            produces a posterior.
        RuntimeError: A scheduler or filter causal-lifecycle contract is
            violated.

    Notes:
        Startup consumes initial stance, course, terminal stance, origin fixes,
        and a seed fix. Terminal-stance fixes together with the seed establish
        the local NED origin; the seed also sets GNSS-relative time zero and is
        neither scheduled nor fused. During operational replay, a permitted
        missing notification
        creates held-increment synthetic propagation with a detector reset,
        false stance flags, coast covariance, and no GNSS scheduling. GNSS is
        released only at a nonempty real-frame posterior. A valid future-mapped
        fix receives one later nonempty real-frame opportunity before becoming
        future-dropped; receiver-invalid scheduled fixes are counted without a
        filter call. After source exhaustion, the result is reconstructed from
        the corrected filter history so accepted delayed updates appear in their
        historical and all later replayed rows.
    """
    recovery_processor = ImuGapRecoveryProcessor(ImuGapRecoveryPolicy())
    arrival_events = iter_gnss_imu_arrivals(raw_frames, gnss_fixes)
    startup = _select_gnss_aided_startup(arrival_events, config, recovery_processor)
    nominal0 = _initial_clock_nominal(startup, config.initial_accel_bias_b)
    replay_filter = ClockAwareGnssIns(
        nominal0=nominal0,
        P0=config.P0,
        Q=config.Q,
        R_zupt=config.R_zupt,
    )

    initial_nominal = replay_filter.nominal
    initial_covariance = replay_filter.P

    stance_detector = ShoeStanceDetector(
        window_size=config.window_size,
        threshold=config.threshold,
        sigma_a=config.sigma_a,
        sigma_g=config.sigma_g,
    )

    gnss_scheduler = GnssArrivalScheduler()

    stance_flags: list[bool] = []
    gnss_updates: list[DelayedGnssUpdate] = []
    gnss_fusion_diagnostics: list[GnssFusionDiagnostic] = []
    deferred_gnss_fixes: deque[_DeferredGnssFix] = deque()
    coast_plans: list[ImuCoastPlan] = []
    invalid_gnss_fix_count = 0

    for arrival_event in arrival_events:
        if isinstance(arrival_event, GnssFixLLH):
            if gnss_outage is not None and replay_filter.t_i >= gnss_outage:
                continue
            gnss_scheduler.submit_fix(arrival_event)
            continue
        if isinstance(arrival_event, TrackedFrame):
            recovery_events = recovery_processor.update(arrival_event)
            for recovery_event in recovery_events:
                if isinstance(recovery_event, ImuCoastPlan):
                    stance_detector.reset()
                    coast_plans.append(recovery_event)
                    for increment in recovery_event.increments:
                        clock_event = ClockImuEvent(
                            increment=increment,
                            stance_flag=False,
                            additional_process_covariance=config.Q_coast,
                        )
                        replay_filter.advance(clock_event)
                        stance_flags.append(False)
                elif isinstance(recovery_event, ImuFrameEvent):
                    if not recovery_event.increments:
                        continue
                    for increment in recovery_event.increments:
                        stance_flag = stance_detector.update(increment)
                        clock_event = ClockImuEvent(increment, stance_flag)
                        replay_filter.advance(clock_event)
                        stance_flags.append(stance_flag)

                    # Each deferred fix gets exactly its next nonempty IMU frame.
                    while deferred_gnss_fixes:
                        deferred_fix = deferred_gnss_fixes.popleft()
                        timed_fix = deferred_fix.timed_fix
                        tau_hat = replay_filter.nominal[NOM_TAU]
                        mapped_epoch = gnss_time_to_imu_time(timed_fix.t_fix_g, tau_hat)
                        if mapped_epoch <= replay_filter.t_i + _TIMELINE_ATOL:
                            update, evaluation = replay_filter._fuse_delayed_gnss_with_evaluation(
                                fix=timed_fix,
                                R_pos=config.R_gnss,
                                nis_threshold=config.nis_threshold,
                            )
                        else:
                            update = DelayedGnssUpdate(
                                status=DelayedGnssUpdateStatus.FUTURE_DROPPED,
                                t_fix_i_hat=mapped_epoch,
                                t_applied_i=None,
                                nis=None,
                            )
                            evaluation = None
                        gnss_updates.append(update)
                        gnss_fusion_diagnostics.append(
                            _build_gnss_fusion_diagnostic(
                                deferred_fix.fix,
                                timed_fix,
                                tau_hat,
                                update,
                                evaluation,
                            )
                        )

                    imu_epoch = ImuArrivalEpoch(
                        t_i=replay_filter.t_i,
                        host_arrival_monotonic=(
                            arrival_event.decoded_frame.raw_frame.arrival_monotonic
                        ),
                    )
                    released_fixes = gnss_scheduler.release_at_imu_epoch(imu_epoch)
                    for scheduled_fix in released_fixes:
                        if not scheduled_fix.fix.fix_valid:
                            invalid_gnss_fix_count += 1
                        else:
                            timed_fix = _to_timed_gnss_fix_ned(scheduled_fix, startup)
                            tau_hat = replay_filter.nominal[NOM_TAU]
                            mapped_epoch = gnss_time_to_imu_time(timed_fix.t_fix_g, tau_hat)
                            if mapped_epoch <= replay_filter.t_i + _TIMELINE_ATOL:
                                update, evaluation = replay_filter._fuse_delayed_gnss_with_evaluation(
                                    timed_fix,
                                    config.R_gnss,
                                    config.nis_threshold,
                                )
                                gnss_updates.append(update)
                                gnss_fusion_diagnostics.append(
                                    _build_gnss_fusion_diagnostic(
                                        scheduled_fix.fix,
                                        timed_fix,
                                        tau_hat,
                                        update,
                                        evaluation,
                                    )
                                )
                            else:
                                deferred_gnss_fixes.append(
                                    _DeferredGnssFix(scheduled_fix.fix, timed_fix)
                                )

    posteriors = replay_filter.posteriors
    if not posteriors:
        raise ValueError(
            "GNSS + ZUPT replay requires at least one post-startup IMU increment."
        )

    # No later nonempty IMU frame remains to resolve these deferred fixes.
    for deferred_fix in deferred_gnss_fixes:
        timed_fix = deferred_fix.timed_fix
        tau_hat = replay_filter.nominal[NOM_TAU]
        mapped_epoch = gnss_time_to_imu_time(timed_fix.t_fix_g, tau_hat)
        update = DelayedGnssUpdate(
            status=DelayedGnssUpdateStatus.FUTURE_DROPPED,
            t_fix_i_hat=mapped_epoch,
            t_applied_i=None,
            nis=None,
        )
        gnss_updates.append(update)
        gnss_fusion_diagnostics.append(
            _build_gnss_fusion_diagnostic(
                deferred_fix.fix,
                timed_fix,
                tau_hat,
                update,
                None,
            )
        )

    if len(gnss_updates) != len(gnss_fusion_diagnostics) or any(
        update is not diagnostic.update
        for update, diagnostic in zip(gnss_updates, gnss_fusion_diagnostics)
    ):
        raise RuntimeError("GNSS outcomes and diagnostics must remain aligned.")

    summary = gnss_scheduler.finish()
    time = np.asarray(
        [0.0, *(posterior.t_i for posterior in posteriors)],
        dtype=float,
    )
    nominals = np.stack(
        (initial_nominal, *(posterior.nominal for posterior in posteriors))
    )
    covariances = np.stack(
        (initial_covariance, *(posterior.P for posterior in posteriors))
    )

    return GnssZuptReplayResult(
        startup=startup,
        time=time,
        nominals=nominals,
        covariances=covariances,
        stance_flags=np.asarray(stance_flags, dtype=bool),
        gnss_updates=tuple(gnss_updates),
        gnss_fusion_diagnostics=tuple(gnss_fusion_diagnostics),
        invalid_gnss_fix_count=invalid_gnss_fix_count,
        coast_plans=tuple(coast_plans),
        summary=summary,
    )
