"""Focused contracts for bounded, causal synthetic IMU coasting."""

import numpy as np
import pytest

from bootins.bmu.decode import DecodedMessage, Measurement, MessageInfo
from bootins.bmu.provenance import DecodedFrame
from bootins.bmu.sequence import GapEvent
from bootins.bmu.stream import TrackedFrame
from bootins.bmu.transport import RawFrame
from bootins.replay.imu_gap_recovery import ImuFrameEvent, ImuGapRecoveryPolicy, ImuGapRecoveryProcessor, plan_short_imu_coast


def _increment(value: float) -> tuple[np.ndarray, np.ndarray, float]:
    return np.array([value, value + 1.0, value + 2.0]), np.array([value + 3.0, value + 4.0, value + 5.0]), 0.01


def _frame(index: int, sequence: int, gap: GapEvent | None = None) -> TrackedFrame:
    measurements = [Measurement(dt=0.01, dtheta=_increment(value)[0], dv=_increment(value)[1], quat_raw=np.ones(4)) for value in range(4)]
    return TrackedFrame(
        DecodedFrame(RawFrame(index, float(index), b""), DecodedMessage(MessageInfo(sequence, 0), measurements)),
        gap,
    )


def test_coast_holds_the_last_prior_increment_with_owned_synthetic_vectors():
    previous = tuple(_increment(float(value)) for value in range(4))
    plan = plan_short_imu_coast(GapEvent(100, 102, 1), previous, 4, 17, 18, ImuGapRecoveryPolicy())
    previous[-1][0][:] = -1.0

    assert (plan.reference_frame_index, plan.revealing_frame_index, len(plan.increments)) == (17, 18, 4)
    assert plan.duration == pytest.approx(0.04)
    np.testing.assert_allclose(plan.increments[0][0], [3.0, 4.0, 5.0])
    assert not plan.increments[0][0].flags.writeable
    assert not plan.increments[0][1].flags.writeable


def test_coast_refuses_gaps_beyond_policy_or_mismatched_packet_shapes():
    previous = tuple(_increment(float(value)) for value in range(4))
    with pytest.raises(ValueError, match="exceeds"):
        plan_short_imu_coast(GapEvent(100, 103, 2), previous, 4, 17, 18, ImuGapRecoveryPolicy())
    with pytest.raises(ValueError, match="mismatched"):
        plan_short_imu_coast(GapEvent(100, 102, 1), previous, 3, 17, 18, ImuGapRecoveryPolicy())
    with pytest.raises(ValueError, match="greater"):
        plan_short_imu_coast(GapEvent(100, 102, 1), previous, 4, 18, 17, ImuGapRecoveryPolicy())


def test_processor_emits_contiguous_real_frame_and_retains_it_as_the_reference():
    processor = ImuGapRecoveryProcessor(ImuGapRecoveryPolicy())
    frame = _frame(17, 100)

    events = processor.update(frame)

    assert len(events) == 1
    assert isinstance(events[0], ImuFrameEvent)
    assert events[0].tracked_frame is frame
    frame.decoded_frame.decoded_message.measurements[0].dtheta[0] = 99.0
    np.testing.assert_allclose(events[0].increments[0][0], [0.0, 1.0, 2.0])
    assert not events[0].increments[0][0].flags.writeable


def test_processor_emits_coast_before_the_real_frame_that_reveals_a_gap():
    processor = ImuGapRecoveryProcessor(ImuGapRecoveryPolicy())
    processor.update(_frame(17, 100))
    revealing = _frame(18, 102, GapEvent(100, 102, 1))

    coast, real = processor.update(revealing)

    assert coast.reference_frame_index == 17
    assert coast.revealing_frame_index == 18
    assert real.tracked_frame is revealing
