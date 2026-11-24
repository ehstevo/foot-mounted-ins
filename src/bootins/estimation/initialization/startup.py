"""Causal GNSS-aided navigation startup coordination.

Startup requires an initial static interval, a qualified GNSS course, a terminal
static interval, and a receiver-valid seed fix. The initial static interval is
retained as diagnostic evidence. The terminal static interval is authoritative
for initial attitude and gyro bias after its run-local FRD attitude is aligned
with GNSS course. Receiver-valid fixes accumulated during the terminal stance,
plus the seed fix, establish a separate ECEF-averaged geographic origin.

The seed fix preserves the GNSS-time and provenance reference for the run. It
does not replace the static origin, and the geographic origin is not a
multipath correction. This selector coordinates causal evidence only: it does
not run stance detection, transform GNSS time to IMU time, or perform filtering.
"""
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from bootins.estimation.initialization.gnss_course import (
    GnssCourseEstimate,
    GnssCourseInitialization,
    GnssCourseSelector,
    GnssStaticOrigin,
    estimate_static_gnss_origin,
    initialize_static_gnss,
)
from bootins.estimation.initialization.static import (
    StanceInitializationInterval,
    StanceInitializationSelector,
    StaticZuptInitialization,
    initialize_static_zupt,
)
from bootins.estimation.mechanization import Increment
from bootins.sensors.gnss_observations import GnssFixLLH

@dataclass(frozen=True)
class GnssAidedStartup:
    """Immutable evidence that authorizes the start of aided navigation.

    Attributes:
        initial_static: Initial static leveling and gyro-bias evidence retained
            for diagnostics, not used as the authoritative navigation state.
        initial_interval: Increment boundaries of the initial diagnostic stance.
        terminal_initialization: Authoritative terminal static leveling,
            gyro-bias, and GNSS-course alignment.
        terminal_interval: Increment boundaries of the authoritative terminal
            stance. ``replay_start_index`` identifies the first subsequent IMU
            increment in the terminal sequence.
        static_origin: ECEF-averaged WGS-84 origin from terminal-stance
            receiver evidence and the seed fix.
        seed_fix: Receiver-valid GNSS observation that defines the seed-relative
            GNSS-time and provenance reference.

    Notes:
        The terminal initialization establishes initial attitude, zero velocity,
        and gyro bias. The initial static result remains diagnostic evidence.
        The frozen dataclass does not make contained NumPy arrays in nested
        results immutable.
    """

    initial_static: StaticZuptInitialization
    initial_interval: StanceInitializationInterval
    terminal_initialization: GnssCourseInitialization
    terminal_interval: StanceInitializationInterval
    static_origin: GnssStaticOrigin
    seed_fix: GnssFixLLH


class _GnssAidedStartupPhase(Enum):
    """Internal phases of one source-neutral GNSS-aided startup protocol."""

    INITIAL_STANCE = auto()
    COURSE = auto()
    TERMINAL_STANCE = auto()
    WAITING_FOR_SEED = auto()
    READY = auto()
    FAILED = auto()


class GnssAidedStartupSelector:
    """Causally assemble one complete GNSS-aided initialization evidence set.

    The caller supplies already-decided stance flags and GNSS fixes in causal
    arrival order. This selector does not own detector evaluation, raw capture
    framing, GNSS-to-IMU clock mapping, or filter updates.

    The protocol progresses through initial stance, course collection, terminal
    stance, and a seed-fix wait. A receiver-valid seed must arrive before the
    qualified terminal stance ends. The terminal selector retains its
    duration-crossing increment; once ``update_gnss`` emits the startup result,
    the external event loop retains subsequent IMU increments for navigation.

    Attributes:
        _phase: Current causal startup phase.
        _initial_stance_selector: Selector for diagnostic initial static evidence.
        _course_selector: Selector for qualified GNSS course-over-ground evidence.
        _terminal_stance_selector: Selector for authoritative terminal stance.
        _terminal_static_fixes: Valid receiver fixes accumulated for the current
            contiguous terminal-stance candidate.
        _result: Completed startup evidence after a valid seed is accepted.
    """

    def __init__(
        self,
        initial_stance_duration: float,
        initial_stance_search_limit: float,
        terminal_stance_duration: float,
        terminal_stance_search_limit: float,
        course_minimum_fix_count: int,
        course_minimum_baseline: float,
        course_maximum_cross_track_rms: float,
    ) -> None:
        """Initialize one GNSS-aided startup protocol.

        Args:
            initial_stance_duration: Required initial contiguous stance duration
                in s.
            initial_stance_search_limit: Maximum initial-stance search duration
                in s.
            terminal_stance_duration: Required terminal contiguous stance
                duration in s.
            terminal_stance_search_limit: Maximum terminal-stance search duration
                in s.
            course_minimum_fix_count: Minimum receiver-valid fixes needed to
                evaluate a GNSS course candidate.
            course_minimum_baseline: Minimum projected course baseline in m.
            course_maximum_cross_track_rms: Maximum straight-line-fit RMS in m.

        Raises:
            ValueError: If either stance selector rejects its timing settings.

        Notes:
            GNSS-course parameter validation is owned by the aided-profile
            boundary rather than this selector.
        """
        self._phase = _GnssAidedStartupPhase.INITIAL_STANCE
        self._initial_stance_selector = StanceInitializationSelector(
            minimum_stance_duration=initial_stance_duration,
            startup_search_limit=initial_stance_search_limit,
        )
        self._terminal_stance_selector = StanceInitializationSelector(
            minimum_stance_duration=terminal_stance_duration,
            startup_search_limit=terminal_stance_search_limit,
        )

        self._course_selector = GnssCourseSelector(
            minimum_fix_count=course_minimum_fix_count,
            minimum_baseline=course_minimum_baseline,
            maximum_cross_track_rms=course_maximum_cross_track_rms,
        )

        self._initial_increments: list[Increment] = []
        self._terminal_increments: list[Increment] = []

        self._initial_static: StaticZuptInitialization | None = None
        self._initial_interval: StanceInitializationInterval | None = None
        self._course: GnssCourseEstimate | None = None
        self._terminal_initialization: GnssCourseInitialization | None = None
        self._terminal_interval: StanceInitializationInterval | None = None
        self._terminal_static_fixes: list[GnssFixLLH] = []
        self._terminal_stance_candidate_active: bool = False
        self._result: GnssAidedStartup | None = None

    def update_imu(
        self,
        increment: Increment,
        stance_flag: bool,
    ) -> GnssAidedStartup | None:
        """Consume one IMU increment and its causal stance decision.

        Args:
            increment: Body-frame ``(dtheta, dv, dt)`` IMU increment.
            stance_flag: Scalar Python or NumPy Boolean causal stance decision.

        Returns:
            ``None``. Completed startup evidence is emitted only by
            ``update_gnss`` when a receiver-valid seed arrives.

        Raises:
            RuntimeError: If startup already completed.
            ValueError: If a stance selector rejects its input, or if a
                non-Boolean stance flag is supplied while waiting for the seed.

        Notes:
            The initial stance becomes diagnostic static evidence. During the
            terminal candidate, receiver-valid GNSS fixes are retained only
            while the current IMU stance evidence remains contiguous. The
            terminal stance establishes authoritative static initialization.
        """
        if self._phase is _GnssAidedStartupPhase.READY:
            raise RuntimeError("GNSS-aided startup is already complete.")

        if self._phase is _GnssAidedStartupPhase.INITIAL_STANCE:
            self._initial_increments.append(increment)
            interval = self._initial_stance_selector.update(increment, stance_flag)
            if interval is None:
                return None
            self._initial_interval = interval
            self._initial_static = initialize_static_zupt(
                self._initial_increments[
                    interval.start_index : interval.replay_start_index
                ]
            )
            self._phase = _GnssAidedStartupPhase.COURSE
            return None

        if self._phase is _GnssAidedStartupPhase.TERMINAL_STANCE:
            if not stance_flag:
                self._terminal_static_fixes = []
                self._terminal_stance_candidate_active = False
            else:
                self._terminal_stance_candidate_active = True
            self._terminal_increments.append(increment)
            interval = self._terminal_stance_selector.update(increment, stance_flag)
            if interval is None:
                return None
            self._terminal_interval = interval
            assert self._course is not None
            self._terminal_initialization = initialize_static_gnss(
                self._terminal_increments[
                    interval.start_index : interval.replay_start_index
                ],
                self._course,
            )
            self._phase = _GnssAidedStartupPhase.WAITING_FOR_SEED
            return None

        if self._phase is _GnssAidedStartupPhase.WAITING_FOR_SEED:
            if not isinstance(stance_flag, (bool, np.bool_)):
                raise ValueError("Stance flag must be a Boolean scalar.")
            if not stance_flag:
                self._phase = _GnssAidedStartupPhase.FAILED
            return None

        return None

    def update_gnss(self, fix: GnssFixLLH) -> GnssAidedStartup | None:
        """Consume one causally arriving GNSS fix and complete startup if ready.

        Args:
            fix: GNSS observation containing receiver validity, WGS-84 LLH,
                GNSS-time, and provenance evidence.

        Returns:
            Completed ``GnssAidedStartup`` only when a receiver-valid seed fix
            arrives during the qualified terminal stance; otherwise ``None``.

        Raises:
            RuntimeError: If startup already completed.

        Notes:
            In the course phase, valid fixes feed the course selector. During a
            terminal stance, valid fixes become geographic-origin evidence.
            In the seed wait, the first valid fix both completes the static
            origin average and establishes seed-relative GNSS time. Invalid
            fixes never complete startup.
        """
        if self._phase is _GnssAidedStartupPhase.READY:
            raise RuntimeError("GNSS-aided startup is already complete.")

        if self._phase is _GnssAidedStartupPhase.COURSE:
            course_estimate = self._course_selector.update(fix)
            if course_estimate is None:
                return None
            self._course = course_estimate
            self._phase = _GnssAidedStartupPhase.TERMINAL_STANCE
            return None

        if self._phase is _GnssAidedStartupPhase.TERMINAL_STANCE:
            if fix.fix_valid and self._terminal_stance_candidate_active:
                self._terminal_static_fixes.append(fix)
            return None

        if self._phase is _GnssAidedStartupPhase.WAITING_FOR_SEED:
            if not fix.fix_valid:
                return None
            assert self._initial_static is not None
            assert self._initial_interval is not None
            assert self._terminal_initialization is not None
            assert self._terminal_interval is not None

            static_origin = estimate_static_gnss_origin(
                (*self._terminal_static_fixes, fix)
            )

            result = GnssAidedStartup(
                initial_static=self._initial_static,
                initial_interval=self._initial_interval,
                terminal_initialization=self._terminal_initialization,
                terminal_interval=self._terminal_interval,
                static_origin=static_origin,
                seed_fix=fix,
            )
            self._result = result
            self._phase = _GnssAidedStartupPhase.READY
            return result

        return None

    def finish(self) -> GnssAidedStartup:
        """Return completed startup evidence or report the terminal failure.

        Returns:
            The completed ``GnssAidedStartup`` evidence record.

        Raises:
            ValueError: If input ended before qualified initial stance, GNSS
                course, terminal stance, or seed evidence was obtained, or if
                the qualified terminal stance ended before a valid seed.
            AssertionError: If the internal phase is unknown.
        """
        if self._phase is _GnssAidedStartupPhase.READY:
            assert self._result is not None
            return self._result
        if self._phase is _GnssAidedStartupPhase.INITIAL_STANCE:
            raise ValueError("No qualifying initial stance before input ended.")
        if self._phase is _GnssAidedStartupPhase.COURSE:
            raise ValueError("No qualified GNSS course before input ended.")
        if self._phase is _GnssAidedStartupPhase.TERMINAL_STANCE:
            raise ValueError("No qualifying terminal stance before input ended.")
        if self._phase is _GnssAidedStartupPhase.WAITING_FOR_SEED:
            raise ValueError(
                "No receiver-valid seed arrived during the qualified terminal stance."
            )
        if self._phase is _GnssAidedStartupPhase.FAILED:
            raise ValueError(
                "Qualified terminal stance ended before a receiver-valid seed arrived."
            )
        raise AssertionError("Unknown GNSS-aided startup phase.")
