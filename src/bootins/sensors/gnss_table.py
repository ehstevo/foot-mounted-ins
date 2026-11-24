"""Read and write the strict normalized GNSS-fix CSV representation.

The table is a receiver-independent projection of a native GNSS archive. Its
exact required columns preserve GPS week/time-of-week, host availability, WGS-84
coordinates, receiver-validity and accuracy facts, and native-record linkage.
Latitude and longitude cross from degree-valued CSV fields to radian-valued
:class:`~bootins.sensors.gnss_observations.GnssFixLLH` fields only at this
boundary.

This module preserves invalid receiver fixes as evidence and requires
nondecreasing host-arrival order for causal replay. It does not parse native
bytes, assess scientific accuracy, select filter uncertainty, map GNSS time to
the IMU timeline, or apply a navigation update.
"""
import csv
from collections.abc import Iterable
from math import degrees, radians
from pathlib import Path

from bootins.sensors.gnss_observations import GnssFixLLH


_FIX_TABLE_COLUMNS = (
    "gps_week",
    "gps_tow",
    "host_arrival_monotonic",
    "latitude_deg",
    "longitude_deg",
    "height_ellipsoid",
    "fix_valid",
    "satellite_count",
    "horizontal_accuracy",
    "vertical_accuracy",
    "native_record_index",
)


def _parse_fix_valid(value: str) -> bool:
    """Return one canonical CSV Boolean without accepting ambiguous spellings."""
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("fix_valid must be 'true' or 'false'.")


def _parse_row(
    row: dict[str | None, str | list[str] | None],
    row_number: int,
) -> GnssFixLLH:
    """Convert one exact-schema CSV row into a normalized GNSS observation.

    Degree-valued geographic fields become radians before constructing
    ``GnssFixLLH``. The row's receiver validity is retained rather than used as
    an input-row drop policy.
    """
    if None in row:
        raise ValueError(
            f"GNSS fix table row {row_number}: contains more values than the header."
        )

    try:
        return GnssFixLLH(
            gps_week=int(row["gps_week"]),
            gps_tow=float(row["gps_tow"]),
            host_arrival_monotonic=float(row["host_arrival_monotonic"]),
            latitude=radians(float(row["latitude_deg"])),
            longitude=radians(float(row["longitude_deg"])),
            height_ellipsoid=float(row["height_ellipsoid"]),
            fix_valid=_parse_fix_valid(row["fix_valid"]),
            satellite_count=int(row["satellite_count"]),
            horizontal_accuracy=float(row["horizontal_accuracy"]),
            vertical_accuracy=float(row["vertical_accuracy"]),
            native_record_index=int(row["native_record_index"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"GNSS fix table row {row_number}: {error}") from error


def load_gnss_fix_table(path: Path) -> tuple[GnssFixLLH, ...]:
    """Load ordered normalized fixes from one exact-schema CSV table.

    Args:
        path: Filesystem path to a UTF-8 CSV with the required header:
            GPS week/time-of-week, host arrival, degree-valued WGS-84 LLH,
            receiver validity and accuracy evidence, and native record index.

    Returns:
        Immutable tuple of ``GnssFixLLH`` records in nondecreasing host-arrival
        order. A header-only table produces an empty tuple.

    Raises:
        OSError: The table cannot be opened or read.
        ValueError: The header differs from the exact schema, a row is malformed,
            or host-arrival order decreases.

    Notes:
        CSV latitude and longitude are degrees; returned records use radians.
        Receiver-invalid rows remain in the result with ``fix_valid=False`` for
        later replay policy to account for them explicitly.
    """
    table_path = Path(path)
    with table_path.open(encoding="utf-8", newline="") as table_file:
        reader = csv.DictReader(table_file)
        if tuple(reader.fieldnames or ()) != _FIX_TABLE_COLUMNS:
            expected_columns = ", ".join(_FIX_TABLE_COLUMNS)
            raise ValueError(
                "GNSS fix table header must exactly match: "
                f"{expected_columns}."
            )

        fixes: list[GnssFixLLH] = []
        for row_number, row in enumerate(reader, start=2):
            fix = _parse_row(row, row_number)
            if (
                fixes
                and fix.host_arrival_monotonic
                < fixes[-1].host_arrival_monotonic
            ):
                raise ValueError(
                    f"GNSS fix table row {row_number}: host_arrival_monotonic "
                    "must be nondecreasing."
                )
            fixes.append(fix)

    return tuple(fixes)


def write_gnss_fix_table(path: Path, fixes: Iterable[GnssFixLLH]) -> None:
    """Write normalized GNSS evidence using the exact table schema.

    Args:
        path: Destination CSV path.
        fixes: Normalized receiver records in nondecreasing host-arrival order.

    Raises:
        OSError: The destination cannot be opened or written.
        ValueError: Caller-provided host-arrival timestamps decrease.

    Notes:
        The writer preserves every supplied record, including receiver-invalid
        evidence. In-memory WGS-84 latitude and longitude are radians and are
        converted to the schema's degree-valued columns; no navigation or
        receiver-quality decision occurs here.
    """
    table_path = Path(path)
    previous_arrival: float | None = None

    with table_path.open("w", encoding="utf-8", newline="") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=_FIX_TABLE_COLUMNS)
        writer.writeheader()

        for row_number, fix in enumerate(fixes, start=2):
            if (
                previous_arrival is not None
                and fix.host_arrival_monotonic < previous_arrival
            ):
                raise ValueError(
                    f"GNSS fix table row {row_number}: host_arrival_monotonic "
                    "must be nondecreasing."
                )
            previous_arrival = fix.host_arrival_monotonic

            writer.writerow(
                {
                    "gps_week": fix.gps_week,
                    "gps_tow": repr(fix.gps_tow),
                    "host_arrival_monotonic": repr(fix.host_arrival_monotonic),
                    "latitude_deg": repr(degrees(fix.latitude)),
                    "longitude_deg": repr(degrees(fix.longitude)),
                    "height_ellipsoid": repr(fix.height_ellipsoid),
                    "fix_valid": "true" if fix.fix_valid else "false",
                    "satellite_count": fix.satellite_count,
                    "horizontal_accuracy": repr(fix.horizontal_accuracy),
                    "vertical_accuracy": repr(fix.vertical_accuracy),
                    "native_record_index": fix.native_record_index,
                }
            )
