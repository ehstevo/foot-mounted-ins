"""Merge factual GNSS and IMU source arrivals for causal aided replay.

GNSS and BMU evidence are separate recordings whose host-monotonic arrival
timestamps determine when each observation becomes available to a replay. This
module emits their causal merged order without conflating host availability
with the GNSS receiver measurement epoch or an IMU posterior time. It decodes
and sequence-tracks an IMU frame only when that frame becomes the next event,
so later IMU evidence cannot influence earlier processing.

The merge preserves each :class:`~bootins.bmu.stream.TrackedFrame` and
:class:`~bootins.sensors.gnss_observations.GnssFixLLH` as observed. Structural
quality assessment, gap recovery, initialization, GNSS scheduling, coordinate
conversion, and filtering belong to later pipeline stages.
"""
from collections.abc import Iterable, Iterator

from bootins.bmu.sequence import SequenceTracker
from bootins.bmu.stream import TrackedFrame, track_raw_frame
from bootins.bmu.transport import RawFrame
from bootins.sensors.gnss_observations import GnssFixLLH


def iter_gnss_imu_arrivals(
    raw_frames: Iterable[RawFrame],
    gnss_fixes: Iterable[GnssFixLLH],
) -> Iterator[TrackedFrame | GnssFixLLH]:
    """Yield GNSS fixes and tracked IMU frames in causal host-arrival order.

    Args:
        raw_frames: Recorded BMU frames whose ``arrival_monotonic`` values
            identify IMU observation availability.
        gnss_fixes: Receiver records in nondecreasing
            ``host_arrival_monotonic`` order. Their ``t_fix_g`` values remain
            receiver measurement epochs and are not used for merge ordering.

    Yields:
        The next factual :class:`GnssFixLLH` or :class:`TrackedFrame` available
        on the shared host timeline. For equal host-arrival timestamps, the
        GNSS fix is yielded before the IMU frame.

    Raises:
        ValueError: GNSS host-arrival timestamps decrease within
            ``gnss_fixes``, or IMU decoding or sequence tracking rejects a raw
            frame as it becomes available.

    Notes:
        Only one lookahead item from each source is retained. Trailing GNSS
        fixes remain factual events after raw-frame EOF. This layer does not
        assess structural quality, repair a gap, initialize navigation, schedule
        delivery to IMU posteriors, or apply a filter update.
    """
    raw_frame_iterator = iter(raw_frames)
    gnss_iterator = iter(gnss_fixes)

    next_raw_frame: RawFrame | None = next(raw_frame_iterator, None)
    next_gnss_fix: GnssFixLLH | None = next(gnss_iterator, None)

    sequence_tracker = SequenceTracker()

    last_gnss_host_arrival = None

    if next_gnss_fix is not None:
        last_gnss_host_arrival = next_gnss_fix.host_arrival_monotonic

    while next_raw_frame is not None and next_gnss_fix is not None:
        if next_gnss_fix.host_arrival_monotonic <= next_raw_frame.arrival_monotonic:
            yield next_gnss_fix
            next_gnss_fix = next(gnss_iterator, None)
            if next_gnss_fix is not None:
                assert last_gnss_host_arrival is not None
                if next_gnss_fix.host_arrival_monotonic < last_gnss_host_arrival:
                    raise ValueError(
                        "GNSS fixes must be in nondecreasing host-arrival order."
                    )
                last_gnss_host_arrival = next_gnss_fix.host_arrival_monotonic
        else:
            tracked_frame = track_raw_frame(next_raw_frame, sequence_tracker)
            yield tracked_frame
            next_raw_frame = next(raw_frame_iterator, None)

    while next_raw_frame is None and next_gnss_fix is not None:
        yield next_gnss_fix
        next_gnss_fix = next(gnss_iterator, None)
        if next_gnss_fix is not None:
            assert last_gnss_host_arrival is not None
            if next_gnss_fix.host_arrival_monotonic < last_gnss_host_arrival:
                raise ValueError(
                    "GNSS fixes must be in nondecreasing host-arrival order."
                )
            last_gnss_host_arrival = next_gnss_fix.host_arrival_monotonic

    while next_gnss_fix is None and next_raw_frame is not None:
        tracked_frame = track_raw_frame(next_raw_frame, sequence_tracker)
        yield tracked_frame
        next_raw_frame = next(raw_frame_iterator, None)
