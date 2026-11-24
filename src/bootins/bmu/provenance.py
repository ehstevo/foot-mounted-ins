"""Associate decoded BMU semantics with original raw-frame provenance.

This bridge preserves the audit path from decoded measurements to the recorded
notification payload and host-arrival evidence. It delegates payload decoding
to :mod:`bootins.bmu.decode` and leaves sequence-loss interpretation,
structural-quality assessment, and navigation policy to their respective
layers. It neither copies payload bytes nor changes the decoder's exception
behavior.
"""
from dataclasses import dataclass

from bootins.bmu.decode import DecodedMessage, decode_message
from bootins.bmu.transport import RawFrame


@dataclass(frozen=True)
class DecodedFrame:
    """One recorded BLE notification paired with its decoded message result.

    Attributes:
        raw_frame: Original transport record, retaining file order,
            host-monotonic arrival evidence, and untouched payload bytes.
        decoded_message: Semantic device facts and physical IMU increments
            decoded from ``raw_frame.payload``.

    Notes:
        The frozen wrapper retains object references rather than recreating raw
        transport evidence. It adds provenance only; it makes no navigation or
        packet-loss decision.
    """

    raw_frame: RawFrame
    decoded_message: DecodedMessage


def decode_raw_frame(raw_frame: RawFrame) -> DecodedFrame:
    """Decode one raw frame while retaining the exact source record.

    Args:
        raw_frame: Complete captured transport record with untouched payload and
            host-arrival provenance.

    Returns:
        A wrapper associating the same raw-frame object with its decoded message.

    Raises:
        RuntimeError: Private BMU decoder definitions are unavailable.
        ValueError: The payload cannot satisfy the decoder's framing or message
            contract.

    Notes:
        Decoder exceptions intentionally propagate unchanged. This layer adds
        provenance rather than a competing payload-validation policy.
    """
    decoded_message = decode_message(raw_frame.payload)

    return DecodedFrame(
        raw_frame=raw_frame,
        decoded_message=decoded_message,
    )
