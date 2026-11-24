"""Project selected archived UBX navigation evidence into normalized fixes.

This source-specific adapter pairs NAV-PVT position evidence with NAV-TIMEGPS
clock evidence bearing the same native integer time-of-week tag. A valid pair
produces one receiver-independent :class:`GnssFixLLH` record whose absolute GPS
epoch, host-arrival provenance, WGS-84 position, and receiver-quality evidence
remain traceable to the PVT archive record.

The adapter supports only these selected navigation message families. It does
not read serial hardware, repair truncated archives, map GNSS time to the IMU
timeline, choose filter covariance, assess receiver accuracy, or apply fusion.
Unknown valid UBX frames remain native-archive evidence but are irrelevant to
this projection.
"""
from collections.abc import Iterable
from dataclasses import dataclass

from bootins.sensors.gnss_observations import GnssFixLLH
from bootins.sensors.ubx import (
    UbxNavPvt,
    UbxNavTimeGps,
    decode_nav_pvt,
    decode_nav_timegps,
    decode_ubx_frame,
)
from bootins.sensors.ubx_archive import UbxCaptureRecord


_GPS_WEEK_DURATION = 604800.0
_NAV_MESSAGE_CLASS = 0x01
_NAV_PVT_MESSAGE_ID = 0x07
_NAV_TIMEGPS_MESSAGE_ID = 0x20


@dataclass(frozen=True)
class UbxFixProjection:
    """Store normalized fixes and complete NAV-PVT timing-association counts.

    Attributes:
        fixes: Normalized receiver observations ordered by PVT archive index and
            validated to have nondecreasing host-arrival timestamps. Receiver
            invalidity remains in each record's ``fix_valid`` field.
        nav_pvt_count: Count of encountered NAV-PVT frames.
        nav_timegps_count: Count of encountered NAV-TIMEGPS frames, including
            records whose time-of-week or week validity was false.
        unpaired_pvt_count: Count of PVT observations with no matching
            NAV-TIMEGPS native time-of-week tag.
        invalid_time_pvt_count: Count of pending PVT observations for which
            matching timing evidence was encountered but did not establish both
            valid time-of-week and GPS-week facts.

    Notes:
        A receiver-invalid PVT is included as normalized evidence rather than
        dropped. The counts describe native timing association only, not
        geodetic accuracy, host/IMU time alignment, or filter acceptance.
    """

    fixes: tuple[GnssFixLLH, ...]
    nav_pvt_count: int
    nav_timegps_count: int
    unpaired_pvt_count: int
    invalid_time_pvt_count: int


@dataclass(frozen=True)
class _PendingPvt:
    """Retain one PVT observation until matching valid clock evidence arrives.

    Attributes:
        observation: Decoded NAV-PVT receiver position and quality evidence.
        record: Native archive record supplying host-arrival time and record
            index for the eventual normalized observation.
    """

    observation: UbxNavPvt
    record: UbxCaptureRecord


def _fix_from_pair(
    pvt: UbxNavPvt,
    timing: UbxNavTimeGps,
    record: UbxCaptureRecord,
) -> GnssFixLLH:
    """Build one normalized fix from same-time-of-week PVT and timing evidence.

    Args:
        pvt: Selected receiver position and quality evidence from NAV-PVT.
        timing: Same-time-of-week valid GPS clock evidence from NAV-TIMEGPS.
        record: PVT native archive record supplying host-arrival provenance.

    Returns:
        One normalized WGS-84 receiver position observation.

    Raises:
        ValueError: The combined evidence violates the normalized-fix contract.

    Notes:
        The native millisecond time-of-week and signed nanosecond correction
        form GPS time of week. A correction crossing a week boundary adjusts the
        GPS week before construction. PVT archive provenance supplies host
        arrival and native-record identity.
    """
    gps_week = timing.gps_week
    gps_tow = pvt.tow * 1e-3 + timing.fractional_tow * 1e-9
    if gps_tow < 0.0:
        gps_week -= 1
        gps_tow += _GPS_WEEK_DURATION
    elif gps_tow >= _GPS_WEEK_DURATION:
        gps_week += 1
        gps_tow -= _GPS_WEEK_DURATION

    try:
        return GnssFixLLH(
            gps_week=gps_week,
            gps_tow=gps_tow,
            host_arrival_monotonic=record.host_arrival_monotonic,
            latitude=pvt.latitude,
            longitude=pvt.longitude,
            height_ellipsoid=pvt.height_ellipsoid,
            fix_valid=pvt.fix_valid,
            satellite_count=pvt.satellite_count,
            horizontal_accuracy=pvt.horizontal_accuracy,
            vertical_accuracy=pvt.vertical_accuracy,
            native_record_index=record.record_index,
        )
    except ValueError as error:
        raise ValueError(
            f"UBX archive record {record.record_index}: "
            "cannot form a normalized GNSS fix: "
            f"{error}"
        ) from error


def _sort_and_validate_host_order(
    fixes: list[GnssFixLLH],
) -> tuple[GnssFixLLH, ...]:
    """Restore PVT archive order and validate causal host-arrival evidence.

    Args:
        fixes: Newly projected fixes in timing-pair completion order.

    Returns:
        The same fixes ordered by their PVT native-record indices.

    Raises:
        ValueError: Host-arrival timestamps decrease in restored PVT order.
    """
    fixes.sort(key=lambda fix: fix.native_record_index)
    for previous, current in zip(fixes, fixes[1:]):
        if current.host_arrival_monotonic < previous.host_arrival_monotonic:
            raise ValueError(
                f"UBX archive record {current.native_record_index}: "
                "host_arrival_monotonic must be nondecreasing across PVT fixes."
            )
    return tuple(fixes)


def project_ubx_records(records: Iterable[UbxCaptureRecord]) -> UbxFixProjection:
    """Pair selected archived UBX evidence into normalized GNSS observations.

    Args:
        records: Native UBX archive records in deterministic archive order.

    Returns:
        Normalized receiver fixes and counts of selected message and timing
        association outcomes.

    Raises:
        ValueError: An archive frame is not a valid UBX envelope, selected NAV
            payload decoding fails, host-arrival ordering decreases across PVT
            fixes, or a valid pair violates the normalized-fix contract.

    Notes:
        A pair requires equal native integer time-of-week tags and valid
        NAV-TIMEGPS time-of-week and GPS-week evidence. PVT records preceding
        their timing record remain pending. PVT records with unresolved or
        invalid timing are counted rather than synthesized; receiver-invalid
        paired fixes remain in the output. Unknown valid UBX messages are
        ignored by the projection but remain retained in the native archive.
    """
    valid_timing_by_tow: dict[int, UbxNavTimeGps] = {}
    pending_pvts_by_tow: dict[int, list[_PendingPvt]] = {}
    invalid_timing_tows: set[int] = set()
    fixes: list[GnssFixLLH] = []
    nav_pvt_count = 0
    nav_timegps_count = 0

    for record in records:
        try:
            frame = decode_ubx_frame(record.raw_frame)
        except ValueError as error:
            raise ValueError(
                f"UBX archive record {record.record_index}: {error}"
            ) from error

        if frame.message_class != _NAV_MESSAGE_CLASS:
            continue
        if frame.message_id == _NAV_PVT_MESSAGE_ID:
            nav_pvt_count += 1
            pvt = decode_nav_pvt(frame)
            timing = valid_timing_by_tow.get(pvt.tow)
            if timing is None:
                pending_pvts_by_tow.setdefault(pvt.tow, []).append(
                    _PendingPvt(observation=pvt, record=record)
                )
            else:
                fixes.append(_fix_from_pair(pvt, timing, record))
        elif frame.message_id == _NAV_TIMEGPS_MESSAGE_ID:
            nav_timegps_count += 1
            timing = decode_nav_timegps(frame)
            if not timing.tow_valid or not timing.week_valid:
                invalid_timing_tows.add(timing.tow)
                continue

            valid_timing_by_tow[timing.tow] = timing
            for pending in pending_pvts_by_tow.pop(timing.tow, []):
                fixes.append(
                    _fix_from_pair(
                        pending.observation,
                        timing,
                        pending.record,
                    )
                )

    unpaired_pvt_count = 0
    invalid_time_pvt_count = 0
    for tow, pending_pvts in pending_pvts_by_tow.items():
        if tow in invalid_timing_tows:
            invalid_time_pvt_count += len(pending_pvts)
        else:
            unpaired_pvt_count += len(pending_pvts)

    return UbxFixProjection(
        fixes=_sort_and_validate_host_order(fixes),
        nav_pvt_count=nav_pvt_count,
        nav_timegps_count=nav_timegps_count,
        unpaired_pvt_count=unpaired_pvt_count,
        invalid_time_pvt_count=invalid_time_pvt_count,
    )
