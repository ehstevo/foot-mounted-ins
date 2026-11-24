"""Physical and causal initialization checks for replay navigation."""

import numpy as np
import pytest

from bootins.estimation.initialization import (
    GnssAidedStartupSelector,
    GnssCourseEstimate,
    GnssCourseSelector,
    estimate_static_gnss_origin,
    initialize_static_gnss,
    initialize_static_level,
    initialize_static_zupt,
    select_first_qualified_stance,
)
from bootins.estimation.initialization import gnss_course
from bootins.estimation.mechanization import G_NED
from bootins.frames.geodesy import ecef_to_llh, llh_to_ecef
from bootins.frames.quaternion import quat_to_dcm
from bootins.sensors.gnss_observations import GnssFixLLH


GRAVITY = np.linalg.norm(G_NED)


def _increment(gyro_rate: object = (0.0, 0.0, 0.0), dt: float = 0.01) -> tuple[np.ndarray, np.ndarray, float]:
    return np.asarray(gyro_rate, dtype=float) * dt, np.array([0.0, 0.0, GRAVITY * dt]), dt


def _fix(index: int, *, valid: bool = True) -> GnssFixLLH:
    return GnssFixLLH(
        gps_week=2300, gps_tow=1000.0 + index, host_arrival_monotonic=float(index),
        latitude=0.1 + index * 1e-6, longitude=-1.1, height_ellipsoid=123.0,
        fix_valid=valid, satellite_count=12, horizontal_accuracy=1.0,
        vertical_accuracy=1.5, native_record_index=index,
    )


def test_static_leveling_defines_the_run_local_forward_right_down_frame():
    initialization = initialize_static_level([_increment(dt=0.01), _increment(dt=0.02)])
    body_to_run = quat_to_dcm(initialization.q_fb)

    assert initialization.duration == pytest.approx(0.03)
    np.testing.assert_allclose(body_to_run @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(body_to_run @ [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], atol=1e-12)


def test_static_zupt_initialization_time_weights_gyro_bias_and_preserves_leveling():
    increments = [_increment([0.1, -0.2, 0.3], 0.01), _increment([-0.1, 0.4, 0.2], 0.03)]
    initialization = initialize_static_zupt(increments)

    np.testing.assert_allclose(initialization.gyro_bias_b, [-0.05, 0.25, 0.225])
    np.testing.assert_allclose(
        quat_to_dcm(initialization.static_level.q_fb),
        quat_to_dcm(initialize_static_level(increments).q_fb),
        atol=1e-12,
    )


def test_static_initialization_refuses_undefined_force_or_timing_evidence():
    with pytest.raises(ValueError, match="at least one"):
        initialize_static_level([])
    with pytest.raises(ValueError, match="positive"):
        initialize_static_level([(np.zeros(3), np.ones(3), 0.0)])


def test_stance_selection_uses_the_first_contiguous_duration_crossing_interval():
    increments = [_increment(dt=0.1)] * 7
    selected = select_first_qualified_stance(increments, np.array([False, True, True, True, False, True, True]), 0.3, 0.7)

    assert selected.start_index == 1
    assert selected.replay_start_index == 4
    assert selected.increment_count == 3
    assert selected.duration == pytest.approx(0.3)


def test_static_gnss_origin_is_the_ecef_mean_of_valid_receiver_evidence():
    first = _fix(0)
    second = GnssFixLLH(
        gps_week=first.gps_week, gps_tow=first.gps_tow + 1.0, host_arrival_monotonic=1.0,
        latitude=0.1001, longitude=-1.1002, height_ellipsoid=129.0, fix_valid=True,
        satellite_count=12, horizontal_accuracy=1.0, vertical_accuracy=1.5, native_record_index=1,
    )

    origin = estimate_static_gnss_origin((first, second))
    expected = ecef_to_llh((llh_to_ecef(first.latitude, first.longitude, first.height_ellipsoid) + llh_to_ecef(second.latitude, second.longitude, second.height_ellipsoid)) / 2.0)

    np.testing.assert_allclose([origin.latitude, origin.longitude, origin.height_ellipsoid], expected, atol=1e-9)
    assert origin.contributing_fixes == (first, second)


def test_course_selector_qualifies_a_straight_track_then_locks_its_estimate(monkeypatch):
    points = {1: np.array([4.0, 3.0]), 2: np.array([8.0, 6.0])}
    monkeypatch.setattr(
        gnss_course,
        "llh_to_ned",
        lambda latitude, *_args: np.array([*points[round((latitude - 0.1) * 1e6)], 0.0]),
    )
    selector = GnssCourseSelector(3, 9.0, 0.01)

    assert selector.update(_fix(0)) is None
    assert selector.update(_fix(1)) is None
    estimate = selector.update(_fix(2))

    assert estimate is not None
    np.testing.assert_allclose(estimate.direction_ne, [0.8, 0.6])
    assert estimate.course == pytest.approx(np.arctan2(0.6, 0.8))
    assert selector.update(_fix(3, valid=False)) is estimate


def test_terminal_static_gnss_alignment_maps_boot_forward_into_the_qualified_course():
    course = GnssCourseEstimate(np.array([1.0, 0.0]), 0.0, 25.0, 0.0, 10)
    initialization = initialize_static_gnss([_increment([0.2, 0.0, 0.0])], course)

    np.testing.assert_allclose(quat_to_dcm(initialization.q_nb) @ [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(initialization.terminal_static.gyro_bias_b, [0.2, 0.0, 0.0])


def test_aided_startup_requires_initial_stance_course_terminal_stance_and_fresh_seed(monkeypatch):
    monkeypatch.setattr(gnss_course, "llh_to_ned", lambda latitude, *_args: np.array([2.0, 0.0, 0.0]))
    selector = GnssAidedStartupSelector(0.015, 0.1, 0.015, 0.1, 2, 1.0, 0.01)
    terminal_fix = _fix(2)
    seed_fix = _fix(3)

    assert selector.update_gnss(_fix(9)) is None
    assert selector.update_imu(_increment([0.1, 0.0, 0.0]), True) is None
    assert selector.update_imu(_increment([0.1, 0.0, 0.0]), True) is None
    assert selector.update_gnss(_fix(0)) is None
    assert selector.update_gnss(_fix(1)) is None
    assert selector.update_imu(_increment([0.4, 0.0, 0.0]), True) is None
    assert selector.update_gnss(terminal_fix) is None
    assert selector.update_imu(_increment([0.4, 0.0, 0.0]), True) is None
    startup = selector.update_gnss(seed_fix)

    assert startup is not None
    assert startup.seed_fix is seed_fix
    assert startup.static_origin.contributing_fixes == (terminal_fix, seed_fix)
    np.testing.assert_allclose(startup.initial_static.gyro_bias_b, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(startup.terminal_initialization.terminal_static.gyro_bias_b, [0.4, 0.0, 0.0])
