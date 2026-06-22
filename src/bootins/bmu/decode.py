"""
decode.py -- turn one raw BMU (Quaternion Data) message into physical
measurements.

The wire layout (offsets/formats) is ICD-derived and lives in the gitignored
device_interface_local.py; this module holds only the generic decode logic, so
it is safe as a public artifact.

A Quaternion Data message = an 18-byte header + NumMeas x 46-byte blocks. Each block
carries the VN-100's coning/sculling-integrated increments: a device attitude
quaternion, a sample interval dt, an angular increment dtheta, and a velocity
increment dv. We consume these triples directly -- no re-integration, no
re-adding coning/sculling.

Units boundary: dtheta is reported in DEGREES and is converted to radians HERE,
exactly once. dv (m/s) and dt (s) pass through untouched.

Quaternion order is deliberately NOT assumed -- VectorNav output is commonly
scalar-LAST [x,y,z,w], while our project convention is scalar-FIRST [w,x,y,z].
We return the raw 4 components and resolve the order empirically downstream (at
level rest the scalar component is ~1, the vector part ~0).
"""
import struct
from dataclasses import dataclass

import numpy as np

from bootins.bmu import device_interface_local as iface


@dataclass
class Measurement:
    """One IMU measurement extracted from a Quaternion Data block. Vectors are float64."""
    dt: float                # sample interval, seconds
    dtheta: np.ndarray       # angular increment, RADIANS, shape (3,)
    dv: np.ndarray           # velocity increment, m/s, shape (3,)
    quat_raw: np.ndarray     # 4 components, ORDER UNRESOLVED, shape (4,)


def decode_message(raw: bytes) -> list[Measurement]:
    """Decode one raw message into its NumMeas Measurements.

    The message is validated before it is trusted: the buffer must hold at least a
    full header, the header must carry MsgID (parsing any other message as
    if it were ours would shift every field), and the buffer length must match the
    NumMeas the header claims. Each block is then unpacked, dtheta is converted
    deg -> rad, and every vector is returned as a float64 array.

    Raises:
        ValueError: buffer shorter than the header, wrong MsgID, or a length that
            disagrees with NumMeas -- all signal a framing bug upstream, so we
            fail loud rather than guess.
    """
    if len(raw) < iface.HEADER_SIZE:
        raise ValueError(
            f"Buffer too short for a header: got {len(raw)} B, "
            f"need >= {iface.HEADER_SIZE} B."
        )

    header = struct.unpack(iface.HEADER_FORMAT, raw[:iface.HEADER_SIZE])
    msg_id = header[iface.H_MSG_ID]
    num_meas = header[iface.H_NUM_MEAS]

    if msg_id != iface.MSG_ID_QUATERNION:
        raise ValueError(
            f"Unexpected MsgID 0x{msg_id:04x} (decoder handles only "
            f"0x{iface.MSG_ID_QUATERNION:04x}). Check the message type/endianness."
        )

    expected_len = iface.HEADER_SIZE + num_meas * iface.BLOCK_SIZE
    if len(raw) != expected_len:
        raise ValueError(
            f"Message length disagrees with NumMeas={num_meas}: "
            f"expected {expected_len} B, got {len(raw)} B."
        )

    measurements = []
    for i in range(num_meas):
        offset = iface.HEADER_SIZE + i * iface.BLOCK_SIZE
        block = struct.unpack(iface.BLOCK_FORMAT, raw[offset:offset + iface.BLOCK_SIZE])

        dtheta = np.array(block[iface.B_DTHETA], dtype=np.float64)
        if iface.DTHETA_IN_DEGREES:
            dtheta = np.deg2rad(dtheta)  # the units boundary -- crossed exactly once

        measurements.append(Measurement(
            dt=float(block[iface.B_DT]),
            dtheta=dtheta,
            dv=np.array(block[iface.B_DV], dtype=np.float64),
            quat_raw=np.array(block[iface.B_QUAT], dtype=np.float64),  # order unresolved
        ))

    return measurements

