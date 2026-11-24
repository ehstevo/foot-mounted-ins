"""Causal GNSS-arrival scheduling checks independent of fusion."""

import pytest

from bootins.replay.gnss_schedule import GnssArrivalScheduler, ImuArrivalEpoch
from bootins.sensors.gnss_observations import GnssFixLLH


def _fix(arrival: float, index: int = 0) -> GnssFixLLH:
    return GnssFixLLH(
        gps_week=2300, gps_tow=1000.0 + index, host_arrival_monotonic=arrival,
        latitude=0.7, longitude=-1.1, height_ellipsoid=123.0, fix_valid=True,
        satellite_count=14, horizontal_accuracy=0.8, vertical_accuracy=1.2,
        native_record_index=index,
    )


def _epoch(t_i: float, arrival: float) -> ImuArrivalEpoch:
    return ImuArrivalEpoch(t_i=t_i, host_arrival_monotonic=arrival)


def test_first_epoch_drops_prestart_fixes_releases_equal_and_defers_future():
    scheduler = GnssArrivalScheduler()
    scheduler.submit_fix(_fix(9.0, 1))
    equal_fix = _fix(10.0, 2)
    scheduler.submit_fix(equal_fix)
    scheduler.submit_fix(_fix(12.0, 3))

    released = scheduler.release_at_imu_epoch(_epoch(1.5, 10.0))
    summary = scheduler.finish()

    assert released[0].fix is equal_fix
    assert released[0].t_arrival_i == pytest.approx(1.5)
    assert (summary.dropped_before_imu_count, summary.dropped_after_imu_count) == (1, 1)


def test_release_uses_first_eligible_epoch_and_submission_order_for_ties():
    scheduler = GnssArrivalScheduler()
    scheduler.submit_fix(_fix(3.0, 1))
    scheduler.submit_fix(_fix(3.0, 2))
    assert scheduler.release_at_imu_epoch(_epoch(1.0, 2.0)) == ()

    released = scheduler.release_at_imu_epoch(_epoch(2.0, 3.0))

    assert [item.fix.native_record_index for item in released] == [1, 2]
    assert [item.t_arrival_i for item in released] == [2.0, 2.0]


def test_late_fix_releases_at_the_next_available_epoch_without_backdating():
    scheduler = GnssArrivalScheduler()
    scheduler.release_at_imu_epoch(_epoch(1.0, 10.0))
    late_fix = _fix(9.0)
    scheduler.submit_fix(late_fix)

    assert scheduler.release_at_imu_epoch(_epoch(2.0, 11.0))[0].fix is late_fix


def test_scheduler_rejects_backward_source_or_epoch_time():
    scheduler = GnssArrivalScheduler()
    scheduler.submit_fix(_fix(10.0))
    with pytest.raises(ValueError, match="nondecreasing"):
        scheduler.submit_fix(_fix(9.0))

    scheduler.release_at_imu_epoch(_epoch(1.0, 10.0))
    with pytest.raises(ValueError, match="t_i"):
        scheduler.release_at_imu_epoch(_epoch(1.0, 11.0))


def test_finish_closes_a_valid_empty_timeline_after_the_first_epoch():
    scheduler = GnssArrivalScheduler()
    assert scheduler.release_at_imu_epoch(_epoch(0.0, 10.0)) == ()
    assert scheduler.finish().dropped_after_imu_count == 0
    with pytest.raises(RuntimeError, match="already finished"):
        scheduler.finish()
