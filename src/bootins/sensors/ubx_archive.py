"""Store timestamped native UBX frames and export validated byte streams.

Each archive record retains one complete received native UBX frame with the
host-monotonic timestamp at which it became available. Archive framing is
deliberately separate from UBX decoding: record I/O does not interpret message
family, payload semantics, GNSS measurement time, or navigation eligibility.

The standard-stream exporter validates each retained native frame before
writing its untouched bytes without archive headers. This preserves receiver
evidence for external tools while keeping host-arrival provenance in the native
archive container.
"""
import os
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from bootins.sensors.ubx import decode_ubx_frame


_RECORD_HEADER_FORMAT = "<dI"
_RECORD_HEADER_SIZE = struct.calcsize(_RECORD_HEADER_FORMAT)


@dataclass(frozen=True)
class UbxCaptureRecord:
    """Represent one timestamped native UBX archive record.

    Attributes:
        record_index: Zero-based deterministic archive order.
        host_arrival_monotonic: Host-monotonic availability timestamp in s. It
            is diagnostic host evidence, not a GNSS measurement epoch.
        raw_frame: Complete native UBX bytes retained exactly as archived for
            later validation, decoding, reparse, or standard-stream export.

    Notes:
        This frozen record preserves archive provenance but does not validate the
        native frame itself. UBX envelope validation occurs at decode or export.
    """

    record_index: int
    host_arrival_monotonic: float
    raw_frame: bytes


def write_ubx_record(
    capture_file: BinaryIO,
    host_arrival_monotonic: float,
    raw_frame: bytes,
) -> None:
    """Append one timestamped native UBX frame to an open archive file.

    Args:
        capture_file: Open binary destination owned by the caller.
        host_arrival_monotonic: Host-monotonic availability timestamp in s.
        raw_frame: Complete native UBX bytes to retain without interpretation.

    Raises:
        ValueError: ``raw_frame`` is not bytes or exceeds the archive length
            field capacity.

    Notes:
        This writer records framing and host provenance only. It does not
        validate UBX envelope structure or synchronize the caller-owned file.
    """
    if not isinstance(raw_frame, bytes):
        raise ValueError("raw_frame must be bytes.")
    if len(raw_frame) > 0xFFFFFFFF:
        raise ValueError("raw_frame cannot exceed the u32 archive length field.")

    header = struct.pack(
        _RECORD_HEADER_FORMAT,
        host_arrival_monotonic,
        len(raw_frame),
    )
    capture_file.write(header)
    capture_file.write(raw_frame)


def iter_ubx_records(path: str | Path) -> Iterator[UbxCaptureRecord]:
    """Yield timestamped native archive records in deterministic file order.

    Args:
        path: Filesystem path to the binary UBX archive.

    Yields:
        Framed archive records with host-arrival provenance and untouched native
        frame bytes.

    Raises:
        OSError: The archive cannot be opened or read.
        ValueError: EOF occurs within an archive header or its declared frame.

    Notes:
        Clean EOF before a new record begins ends iteration normally. A truncated
        tail is refused so incomplete native evidence cannot appear to be a
        shorter clean archive. This reader does not validate UBX frames.
    """
    archive_path = Path(path)
    record_index = 0

    with archive_path.open("rb") as archive_file:
        while True:
            header = archive_file.read(_RECORD_HEADER_SIZE)
            if not header:
                return
            if len(header) != _RECORD_HEADER_SIZE:
                raise ValueError(
                    f"Truncated UBX record header at index {record_index}: "
                    f"got {len(header)} B, expected {_RECORD_HEADER_SIZE} B."
                )

            host_arrival_monotonic, raw_frame_length = struct.unpack(
                _RECORD_HEADER_FORMAT,
                header,
            )
            raw_frame = archive_file.read(raw_frame_length)
            if len(raw_frame) != raw_frame_length:
                raise ValueError(
                    f"Truncated UBX raw frame at index {record_index}: "
                    f"got {len(raw_frame)} B, expected {raw_frame_length} B."
                )

            yield UbxCaptureRecord(
                record_index=record_index,
                host_arrival_monotonic=host_arrival_monotonic,
                raw_frame=raw_frame,
            )
            record_index += 1


def write_ubx_stream(
    records: Iterable[UbxCaptureRecord],
    output_path: str | Path,
) -> int:
    """Export validated native archive frames as one concatenated UBX stream.

    Args:
        records: Archive records in caller-supplied order.
        output_path: Destination path for the header-free native UBX stream.

    Returns:
        Number of exported complete validated frames.

    Raises:
        ValueError: An archived raw frame is not a complete valid UBX frame.
        OSError: The output stream cannot be opened, written, or synchronized.

    Notes:
        The output omits archive host-arrival headers so native UBX consumers,
        including PPK tools, receive only concatenated frame bytes. Every frame
        is validated but otherwise retained byte-for-byte in supplied order.
    """
    stream_path = Path(output_path)
    frame_count = 0

    with stream_path.open("wb") as stream_file:
        for record in records:
            try:
                decode_ubx_frame(record.raw_frame)
            except ValueError as error:
                raise ValueError(
                    f"UBX archive record {record.record_index}: "
                    f"cannot export standard UBX stream: {error}"
                ) from error

            stream_file.write(record.raw_frame)
            frame_count += 1

        stream_file.flush()
        os.fsync(stream_file.fileno())

    return frame_count
