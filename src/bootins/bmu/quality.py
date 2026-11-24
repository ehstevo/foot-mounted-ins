"""Assess structural usability of decoded IMU measurements without filtering.

The assessment reports nonfinite fields and unusable integration durations on
an already decoded :class:`~bootins.bmu.decode.Measurement`. It does not parse
transport data, modify measurements, decide whether a sample is physically
plausible, or choose a navigation response. Replay input policy separately
uses these structural facts to accept, reject, split, or recover a stream.
"""
from enum import IntFlag, auto

import numpy as np

from bootins.bmu.decode import Measurement


class QualityFlags(IntFlag):
    """Encode independent structural concerns on one decoded measurement.

    ``OK`` is zero, so callers can combine the remaining flags with bitwise-or
    and test an unflagged measurement directly against ``QualityFlags.OK``.

    ``NONFINITE_DT`` denotes a nonfinite duration, while ``NONPOSITIVE_DT``
    denotes a finite duration that cannot advance integration.
    ``NONFINITE_DTHETA``, ``NONFINITE_DV``, and ``NONFINITE_QUAT_RAW`` identify
    nonfinite components in their corresponding decoded fields. These flags
    establish structural validity only, not physical accuracy or a filter
    decision.
    """

    OK = 0
    NONFINITE_DT = auto()
    NONPOSITIVE_DT = auto()
    NONFINITE_DTHETA = auto()
    NONFINITE_DV = auto()
    NONFINITE_QUAT_RAW = auto()


def assess(measurement: Measurement) -> QualityFlags:
    """Assess one decoded measurement for threshold-free structural concerns.

    Args:
        measurement: Decoded IMU sample containing duration, body-frame
            increments, and raw quaternion evidence.

    Returns:
        Bitwise combination of every applicable :class:`QualityFlags` value.

    Notes:
        A nonfinite ``dt`` is distinct from a finite but nonpositive duration,
        so the latter is checked only when ``dt`` is finite. No gait-tuned
        magnitude, quaternion-norm, or scientific-accuracy threshold is used,
        and ``measurement`` is never mutated.
    """
    flags = QualityFlags.OK

    if not np.isfinite(measurement.dt):
        flags |= QualityFlags.NONFINITE_DT
    elif measurement.dt <= 0:
        flags |= QualityFlags.NONPOSITIVE_DT
    if not np.isfinite(measurement.dtheta).all():
        flags |= QualityFlags.NONFINITE_DTHETA
    if not np.isfinite(measurement.dv).all():
        flags |= QualityFlags.NONFINITE_DV
    if not np.isfinite(measurement.quat_raw).all():
        flags |= QualityFlags.NONFINITE_QUAT_RAW

    return flags
