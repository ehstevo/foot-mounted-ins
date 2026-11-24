"""Decode private BMU wire data into public semantic IMU evidence.

The locally supplied device interface owns the nonredistributable binary
definition and is imported only when decoding is requested. This public module
exposes the approved semantic result: a device sequence number, a raw device
timestamp, and physical IMU increments. It intentionally does not publish
binary layouts, offsets, identifiers, or a full wire-header representation.

The device provides pre-integrated coning/sculling-corrected increments, so
mechanization consumes ``(dtheta, dv, dt)`` directly. Angular increments cross
from device degrees to project radians exactly once here; velocity increments
and durations enter in their supplied SI units. Raw quaternion values remain
available for evidence assessment, but their device component ordering is not
yet resolved and they are not used to establish navigation attitude.
"""
import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class MessageInfo:
    """Approved packet-level facts extracted from a decoded BMU message.

    This is a narrow semantic projection, not a public representation of the
    proprietary wire header.

    Attributes:
        sequence_num: Device-assigned notification sequence number. Sequence
            tracking interprets its continuity; this record preserves the
            decoder-extracted value.
        device_timestamp_raw: Uninterpreted device-clock value retained for
            diagnostics and provenance. INS integration uses per-measurement
            ``dt``, not this device time.

    Notes:
        This mutable dataclass is a semantic projection of one decoded message,
        not a public representation of the private wire header.
    """

    sequence_num: int
    device_timestamp_raw: int


@dataclass
class Measurement:
    """Represent one pre-integrated physical IMU sample from a BMU message.

    Attributes:
        dt: Integration duration in s for this sample.
        dtheta: Newly allocated ``float64`` body-frame angular increment with
            shape ``(3,)``, in rad. The decoder converts device degrees here
            exactly once when the local interface declares that representation.
        dv: Newly allocated ``float64`` body-frame velocity increment with shape
            ``(3,)``, in m/s.
        quat_raw: Newly allocated ``float64`` raw device quaternion values with
            shape ``(4,)``. Their component ordering remains unresolved.

    Notes:
        This mutable record does not independently validate structural quality,
        physical plausibility, or quaternion convention. Those are separate
        integrity and frame-interpretation concerns.
    """

    dt: float                # Sample integration duration in s.
    dtheta: np.ndarray       # Body-frame angular increment in rad, shape (3,).
    dv: np.ndarray           # Body-frame velocity increment in m/s, shape (3,).
    quat_raw: np.ndarray     # Raw device quaternion, ordering unresolved, shape (4,).


@dataclass
class DecodedMessage:
    """Combine packet-level semantic facts with decoded IMU increments.

    Attributes:
        message_info: Approved sequence and raw device-time evidence for the
            enclosing notification.
        measurements: Mutable decoded IMU increments in their original
            within-message order.

    Notes:
        A decoded message has no raw-frame or host-arrival provenance. The
        provenance layer associates it with the original recorded notification.
    """

    message_info: MessageInfo
    measurements: list[Measurement]


def decode_message(raw: bytes) -> DecodedMessage:
    """Decode one complete BMU payload into approved semantic evidence.

    Args:
        raw: Raw notification payload bytes. The caller supplies complete
            transport framing; this function interprets the device payload.

    Returns:
        Packet-level semantic facts and ordered physical IMU increments.

    Raises:
        RuntimeError: The locally supplied private device interface is absent.
        ValueError: The payload is shorter than the private header, has an
            unexpected message type, or disagrees with its declared measurement
            count.

    Notes:
        The local interface performs the private layout interpretation. This
        function projects only approved semantic fields, converts angular
        increments to radians once where required, and leaves raw quaternion
        component order unresolved.
    """
    try:
        from importlib import import_module

        iface = import_module("bootins.bmu.device_interface_local")
    except ModuleNotFoundError as error:
        if error.name == "bootins.bmu.device_interface_local":
            raise RuntimeError(
                "Decoding BMU messages requires local device-interface definitions."
            ) from error
        raise

    if len(raw) < iface.HEADER_SIZE:
        raise ValueError(
            f"Buffer too short for a header: got {len(raw)} B, "
            f"need >= {iface.HEADER_SIZE} B."
        )

    # The private interface owns the header layout; only approved semantic facts
    # are extracted into the public result below.
    header = struct.unpack(iface.HEADER_FORMAT, raw[:iface.HEADER_SIZE])
    msg_id = header[iface.H_MSG_ID]
    num_meas = header[iface.H_NUM_MEAS]
    seq_num = header[iface.H_SEQ_NUM]
    timestamp = header[iface.H_TIMESTAMP]

    if msg_id != iface.MSG_ID_QUATERNION:
        raise ValueError(
            "Unexpected BMU message type. Check device configuration and "
            "protocol interpretation."
        )

    expected_len = iface.HEADER_SIZE + num_meas * iface.BLOCK_SIZE
    if len(raw) != expected_len:
        raise ValueError(
            f"Message length disagrees with NumMeas={num_meas}: "
            f"expected {expected_len} B, got {len(raw)} B."
        )

    measurements: list[Measurement] = []
    for i in range(num_meas):
        offset = iface.HEADER_SIZE + i * iface.BLOCK_SIZE
        block = struct.unpack(iface.BLOCK_FORMAT, raw[offset:offset + iface.BLOCK_SIZE])

        dtheta = np.array(block[iface.B_DTHETA], dtype=np.float64)
        if iface.DTHETA_IN_DEGREES:
            dtheta = np.deg2rad(dtheta)  # Device degrees -> project radians once.

        measurements.append(Measurement(
            dt=float(block[iface.B_DT]),
            dtheta=dtheta,
            dv=np.array(block[iface.B_DV], dtype=np.float64),
            quat_raw=np.array(block[iface.B_QUAT], dtype=np.float64),  # order unresolved
        ))
    return DecodedMessage(
        message_info=MessageInfo(
            sequence_num=seq_num,
            device_timestamp_raw=timestamp,
        ),
        measurements=measurements,
    )
