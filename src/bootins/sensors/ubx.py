"""Validate UBX envelopes and decode the selected GNSS evidence families.

The module validates complete native UBX frames, reassembles contiguous byte
chunks into validated frames, and decodes selected fields from NAV-PVT and
NAV-TIMEGPS. It does not claim general UBX protocol coverage. The decoded
records retain receiver position, quality, and timing evidence needed by the
normalized GNSS-fix projection.

Serial reading, host-arrival timestamps, native-archive storage, PVT/TIMEGPS
association, local-coordinate conversion, and navigation decisions belong to
other modules. The documentation deliberately describes semantic fields rather
than private or vendor-restricted binary-layout definitions.
"""
import struct
from dataclasses import dataclass
from math import radians


_NAV_MESSAGE_CLASS = 0x01
_NAV_PVT_MESSAGE_ID = 0x07
_NAV_TIMEGPS_MESSAGE_ID = 0x20
_NAV_PVT_PAYLOAD_LENGTH = 92
_NAV_TIMEGPS_PAYLOAD_LENGTH = 16
_SYNC_BYTES = b"\xb5\x62"


@dataclass(frozen=True)
class UbxFrame:
    """Represent one checksum-validated native UBX message.

    Attributes:
        message_class: Native unsigned-byte message family identifier.
        message_id: Native unsigned-byte message identifier within the family.
        payload: Untouched message payload bytes excluding envelope and checksum.

    Notes:
        ``decode_ubx_frame`` creates this frozen record after envelope and
        checksum validation. Direct construction does not perform validation.
        This general frame record is distinct from the selected NAV observation
        records decoded below.
    """

    message_class: int
    message_id: int
    payload: bytes


@dataclass(frozen=True)
class UbxNavPvt:
    """Store selected receiver position and quality evidence from NAV-PVT.

    Attributes:
        tow: Native integer time-of-week association tag in ms. A projection
            pairs it with NAV-TIMEGPS evidence; it is not yet a full GPS epoch.
        fix_type: Receiver-reported native fix-type code retained without a
            navigation-policy interpretation.
        fix_valid: Receiver-reported GNSS-fix-valid Boolean evidence.
        satellite_count: Receiver-reported satellite count.
        longitude: Geodetic WGS-84 longitude in rad.
        latitude: Geodetic WGS-84 latitude in rad.
        height_ellipsoid: WGS-84 ellipsoidal height in m.
        horizontal_accuracy: Receiver-reported horizontal accuracy in m.
        vertical_accuracy: Receiver-reported vertical accuracy in m.

    Notes:
        Latitude and longitude cross to radians in this decoder. Receiver
        validity and accuracy remain evidence only; neither establishes true
        accuracy nor determines a navigation filter's later update decision.
    """

    tow: int
    fix_type: int
    fix_valid: bool
    satellite_count: int
    longitude: float
    latitude: float
    height_ellipsoid: float
    horizontal_accuracy: float
    vertical_accuracy: float


@dataclass(frozen=True)
class UbxNavTimeGps:
    """Store receiver GPS-clock evidence decoded from NAV-TIMEGPS.

    Attributes:
        tow: Native integer time-of-week association tag in ms.
        fractional_tow: Signed native time-of-week correction in ns.
        gps_week: Receiver-reported GPS week number.
        tow_valid: Receiver-reported validity of time-of-week evidence.
        week_valid: Receiver-reported validity of GPS-week evidence.
        leap_seconds_valid: Receiver-reported validity of leap-second evidence.

    Notes:
        A later projection requires valid time-of-week and GPS-week evidence to
        form a normalized GNSS epoch. These flags do not by themselves decide
        receiver position accuracy or navigation fusion eligibility.
    """

    tow: int
    fractional_tow: int
    gps_week: int
    tow_valid: bool
    week_valid: bool
    leap_seconds_valid: bool


def _require_nav_payload(
    frame: UbxFrame,
    *,
    message_id: int,
    payload_length: int,
    message_name: str,
) -> bytes:
    """Return one expected NAV payload or reject a mismatched native frame."""
    if (
        frame.message_class != _NAV_MESSAGE_CLASS
        or frame.message_id != message_id
    ):
        raise ValueError(f"expected a {message_name} frame.")
    if len(frame.payload) != payload_length:
        raise ValueError(
            f"{message_name} payload must contain exactly {payload_length} bytes."
        )
    return frame.payload


def decode_nav_pvt(frame: UbxFrame) -> UbxNavPvt:
    """Decode selected NAV-PVT position and receiver-quality evidence.

    Args:
        frame: Checksum-validated UBX frame expected to be NAV-PVT with its
            supported fixed payload length.

    Returns:
        Selected receiver position, validity, satellite-count, and accuracy
        evidence with geodetic angles converted to rad and metric fields in m.

    Raises:
        ValueError: The frame is not NAV-PVT or does not have the supported
            payload length.
    """
    payload = _require_nav_payload(
        frame,
        message_id=_NAV_PVT_MESSAGE_ID,
        payload_length=_NAV_PVT_PAYLOAD_LENGTH,
        message_name="UBX-NAV-PVT",
    )
    tow = struct.unpack_from("<I", payload, 0)[0]
    fix_type = payload[20]
    fix_valid = bool(payload[21] & 0x01)
    satellite_count = payload[23]
    raw_longitude, raw_latitude, raw_height = struct.unpack_from("<iii", payload, 24)
    raw_horizontal_accuracy, raw_vertical_accuracy = struct.unpack_from(
        "<II",
        payload,
        40,
    )

    return UbxNavPvt(
        tow=tow,
        fix_type=fix_type,
        fix_valid=fix_valid,
        satellite_count=satellite_count,
        longitude=radians(raw_longitude * 1e-7),
        latitude=radians(raw_latitude * 1e-7),
        height_ellipsoid=raw_height * 1e-3,
        horizontal_accuracy=raw_horizontal_accuracy * 1e-3,
        vertical_accuracy=raw_vertical_accuracy * 1e-3,
    )


def decode_nav_timegps(frame: UbxFrame) -> UbxNavTimeGps:
    """Decode NAV-TIMEGPS epoch tags and timing-validity evidence.

    Args:
        frame: Checksum-validated UBX frame expected to be NAV-TIMEGPS with its
            supported fixed payload length.

    Returns:
        Native time-of-week tag, fractional correction, GPS week, and receiver
        timing-validity evidence.

    Raises:
        ValueError: The frame is not NAV-TIMEGPS or does not have the supported
            payload length.
    """
    payload = _require_nav_payload(
        frame,
        message_id=_NAV_TIMEGPS_MESSAGE_ID,
        payload_length=_NAV_TIMEGPS_PAYLOAD_LENGTH,
        message_name="UBX-NAV-TIMEGPS",
    )
    tow, fractional_tow, gps_week = struct.unpack_from("<Iih", payload, 0)
    validity = payload[11]

    return UbxNavTimeGps(
        tow=tow,
        fractional_tow=fractional_tow,
        gps_week=gps_week,
        tow_valid=bool(validity & 0x01),
        week_valid=bool(validity & 0x02),
        leap_seconds_valid=bool(validity & 0x04),
    )


def ubx_checksum(data: bytes) -> tuple[int, int]:
    """Compute the two UBX checksum bytes for protected frame data.

    Args:
        data: Protected native-frame bytes excluding synchronization and checksum.

    Returns:
        ``(CK_A, CK_B)`` as unsigned-byte integers.

    Notes:
        The protected region includes message class, message ID, payload length,
        and payload. Synchronization and checksum bytes are excluded.
    """
    ck_a = 0
    ck_b = 0

    for byte in data:
        ck_a = (ck_a + byte) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF

    return ck_a, ck_b


def decode_ubx_frame(raw: bytes) -> UbxFrame:
    """Validate one complete native UBX envelope and expose its payload.

    Args:
        raw: Exactly one complete native frame including synchronization,
            envelope, payload, and checksum.

    Returns:
        Checksum-validated message class, identifier, and untouched payload.

    Raises:
        ValueError: The frame is too short, has invalid synchronization, its
            declared payload length disagrees with supplied bytes, or checksum
            validation fails.

    Notes:
        This validates envelope structure for any native UBX message. It does
        not decode message-specific payloads beyond the supported NAV decoders.
    """
    if len(raw) < 8:
        raise ValueError("UBX frame is shorter than its fixed eight-byte envelope.")

    if raw[:2] != _SYNC_BYTES:
        raise ValueError("UBX frame has invalid synchronization bytes.")

    payload_length = int.from_bytes(raw[4:6], "little")
    if len(raw) != payload_length + 8:
        raise ValueError("UBX frame length disagrees with its declared payload length.")

    ck_a, ck_b = ubx_checksum(raw[2:-2])

    if (raw[-2], raw[-1]) != (ck_a, ck_b):
        raise ValueError("UBX frame checksum is invalid.")

    return UbxFrame(
        message_class=raw[2],
        message_id=raw[3],
        payload=raw[6:-2],
    )


def encode_ubx_frame(frame: UbxFrame) -> bytes:
    """Encode one UBX frame record as canonical complete native bytes.

    Args:
        frame: Message class, identifier, and payload fields to encode.

    Returns:
        Complete native UBX bytes with deterministic synchronization, payload
        length, and checksum.

    Raises:
        ValueError: A class or identifier is not an unsigned byte, the payload
            is not bytes, or it exceeds the native length field capacity.

    Notes:
        For a frame obtained unchanged from ``decode_ubx_frame``, the encoding
        reproduces its complete native bytes. The function does not validate a
        message-specific payload definition.
    """
    if (
        isinstance(frame.message_class, bool)
        or not isinstance(frame.message_class, int)
        or not 0 <= frame.message_class <= 0xFF
    ):
        raise ValueError("UBX message_class must be an unsigned byte.")
    if (
        isinstance(frame.message_id, bool)
        or not isinstance(frame.message_id, int)
        or not 0 <= frame.message_id <= 0xFF
    ):
        raise ValueError("UBX message_id must be an unsigned byte.")
    if not isinstance(frame.payload, bytes):
        raise ValueError("UBX payload must be bytes.")
    if len(frame.payload) > 0xFFFF:
        raise ValueError("UBX payload cannot exceed 65535 bytes.")

    protected = (
        bytes((frame.message_class, frame.message_id))
        + len(frame.payload).to_bytes(2, "little")
        + frame.payload
    )
    checksum = ubx_checksum(protected)
    return _SYNC_BYTES + protected + bytes(checksum)


class UbxStreamFramer:
    """Reassemble validated UBX frames from contiguous arbitrary-size chunks.

    Attributes:
        _buffer: Unconsumed stream bytes, retained only while they may form a
            complete next native frame.

    Notes:
        Incomplete prefixes remain buffered for ``feed``. Invalid
        synchronization or a complete invalid frame raises without scanning
        ahead or discarding bytes; a higher layer owns recovery and reporting of
        raw receiver evidence.
    """

    def __init__(self) -> None:
        """Initialize a framer with no pending receiver-stream bytes."""
        self._buffer = bytearray()

    @property
    def pending_byte_count(self) -> int:
        """Return the count of incomplete bytes retained for later ``feed``."""
        return len(self._buffer)

    def feed(self, chunk: bytes) -> tuple[UbxFrame, ...]:
        """Consume one contiguous byte chunk and emit complete UBX frames.

        Args:
            chunk: Next contiguous native receiver-stream bytes.

        Returns:
            Every newly complete checksum-validated frame, in stream order.

        Raises:
            ValueError: The accumulated stream has invalid synchronization or a
                complete frame fails envelope or checksum validation.

        Notes:
            A final incomplete prefix remains buffered for a later call. The
            method does not scan forward for a later synchronization sequence.
        """
        self._buffer.extend(chunk)
        frames: list[UbxFrame] = []

        while self._buffer:
            if self._buffer[0] != 0xB5:
                raise ValueError("UBX stream has invalid synchronization bytes.")
            if len(self._buffer) == 1:
                break
            if self._buffer[1] != 0x62:
                raise ValueError("UBX stream has invalid synchronization bytes.")
            if len(self._buffer) < 6:
                break

            payload_length = int.from_bytes(self._buffer[4:6], "little")
            frame_length = payload_length + 8
            if len(self._buffer) < frame_length:
                break

            frame = decode_ubx_frame(bytes(self._buffer[:frame_length]))
            del self._buffer[:frame_length]
            frames.append(frame)

        return tuple(frames)

    def finish(self) -> None:
        """Confirm that the stream ended between complete frames.

        Raises:
            ValueError: The stream ended with an incomplete native frame.
        """
        if self._buffer:
            raise ValueError("UBX stream ended with an incomplete frame.")
