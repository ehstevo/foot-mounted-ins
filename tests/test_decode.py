"""
test_decode.py -- known-truth tests for the Quaternion Data message decoder.

Strategy: build a synthetic Quaternion 3 message from KNOWN values (the _pack helper
below is the inverse of decode_message), decode it, and assert we recover exactly
what we put in -- with dtheta coming back in RADIANS even though we packed it in
degrees.

Scope note: this proves the decode *logic* (field slicing, deg->rad, quaternion
left un-reordered). It does NOT prove the layout CONSTANTS in
device_interface_local.py are right -- the packer and decoder share them, so a
wrong format string would round-trip happily. The layout itself is validated
against the real device (202 B messages, |q| ~ 1, dt ~ 0.01 at rest).
"""
import struct

import numpy as np
import pytest

from bootins.bmu import decode
from bootins.bmu import device_interface_local as iface


def _pack_0x0003(blocks: list[dict]) -> bytes:
    """Build a synthetic Quaternion Data message from a list of block dicts, each with
    keys: quat (4,), dt (float), dtheta_deg (3,), dv (3,). dtheta is packed in
    DEGREES, exactly as the device sends it."""
    num = len(blocks)
    paylen = 14 + num * iface.BLOCK_SIZE  # 14-byte header tail + N blocks
    header = struct.pack(
        iface.HEADER_FORMAT,
        iface.MSG_ID_QUATERNION,  # MsgID
        paylen,                   # PayLen
        0, 0, 0, 0,               # Status, BattSOC, Voltage, Reserved
        0,                        # Timestamp
        0,                        # SeqNum
        num,                      # NumMeas
    )
    body = b""
    for b in blocks:
        body += struct.pack(
            iface.BLOCK_FORMAT,
            *b["quat"],           # qtn[4]
            0,                    # VPE
            b["dt"],              # dt
            *b["dtheta_deg"],     # dtheta[3] in DEGREES
            *b["dv"],             # dv[3]
        )
    return header + body


def test_single_block_roundtrip():
    """A single block round-trips: every field comes back as packed, and dtheta
    is returned in RADIANS even though it was packed in degrees (the units
    boundary). The quaternion must come back in its original order, not reordered.
    """
    block = {
        "quat": [1,0,0,0],
        "dt": 0.01,
        "dtheta_deg": [1,2,3],
        "dv": [0.1, 0.2, 9.8]
    }
    msg = _pack_0x0003([block])
    out = decode.decode_message(msg)
    assert len(out) == 1
    assert out[0].dt == pytest.approx(0.01)
    np.testing.assert_allclose(out[0].dv, [0.1, 0.2, 9.8], rtol=1e-6)
    np.testing.assert_allclose(out[0].quat_raw, [1,0,0,0], rtol=1e-6)
    np.testing.assert_allclose(out[0].dtheta, np.deg2rad([1,2,3]), rtol=1e-6)



def test_multiple_blocks():
    """NumMeas=4 yields 4 Measurements in the order they were packed. Each block
    carries distinct quat/dtheta/dv values so a swapped or dropped block fails.
    """
    block1 = {
        "quat": [1,0,0,0],
        "dt": 0.01,
        "dtheta_deg": [1,2,3],
        "dv": [0.1, 0.2, 9.8]
    }
    block2 = {
        "quat": [0,1,0,0],
        "dt": 0.01,
        "dtheta_deg": [4,5,6],
        "dv": [0.3, 0.4, 9.8]
    }
    block3 = {
        "quat": [0,0,1,0],
        "dt": 0.01,
        "dtheta_deg": [7,8,9],
        "dv": [0.5, 0.6, 9.8]
    }
    block4 = {
        "quat": [0,0,0,1],
        "dt": 0.01,
        "dtheta_deg": [10,11,12],
        "dv": [0.7, 0.8, 9.8]
    }
    blocks = [block1, block2, block3, block4]
    msg = _pack_0x0003(blocks)
    out = decode.decode_message(msg)
    assert len(out) == 4
    assert out[0].dt == pytest.approx(0.01)
    np.testing.assert_allclose(out[0].dv, [0.1, 0.2, 9.8], rtol=1e-6)
    np.testing.assert_allclose(out[0].quat_raw, [1,0,0,0], rtol=1e-6)
    np.testing.assert_allclose(out[0].dtheta, np.deg2rad([1,2,3]), rtol=1e-6)
    assert out[1].dt == pytest.approx(0.01)
    np.testing.assert_allclose(out[1].dv, [0.3, 0.4, 9.8], rtol=1e-6)
    np.testing.assert_allclose(out[1].quat_raw, [0,1,0,0], rtol=1e-6)
    np.testing.assert_allclose(out[1].dtheta, np.deg2rad([4,5,6]), rtol=1e-6)
    assert out[2].dt == pytest.approx(0.01)
    np.testing.assert_allclose(out[2].dv, [0.5, 0.6, 9.8], rtol=1e-6)
    np.testing.assert_allclose(out[2].quat_raw, [0,0,1,0], rtol=1e-6)
    np.testing.assert_allclose(out[2].dtheta, np.deg2rad([7,8,9]), rtol=1e-6)
    assert out[3].dt == pytest.approx(0.01)
    np.testing.assert_allclose(out[3].dv, [0.7, 0.8, 9.8], rtol=1e-6)
    np.testing.assert_allclose(out[3].quat_raw, [0,0,0,1], rtol=1e-6)
    np.testing.assert_allclose(out[3].dtheta, np.deg2rad([10,11,12]), rtol=1e-6)


def test_rejects_wrong_msgid():
    """A message whose MsgID is not Quaternion Data must raise, not silently mis-parse a
    different layout as if it were ours.

    We build a valid Quaternion Data message and overwrite only the MsgID field (offset 0),
    so the buffer is well-formed in every other respect -- the decoder rejects it
    purely on the message type.
    """
    block = {"quat": [1, 0, 0, 0], "dt": 0.01, "dtheta_deg": [1, 2, 3], "dv": [0.1, 0.2, 9.8]}
    msg = bytearray(_pack_0x0003([block]))
    struct.pack_into(iface.ENDIAN + "H", msg, 0, 0x0001)
    with pytest.raises(ValueError):
        decode.decode_message(bytes(msg))
    
