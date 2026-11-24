"""Known-contract tests for structural per-measurement quality flags."""
import numpy as np
import pytest

from bootins.bmu.decode import Measurement
from bootins.bmu.quality import QualityFlags, assess


def _measurement(
    *,
    dt: float = 0.01,
    dtheta: np.ndarray | None = None,
    dv: np.ndarray | None = None,
    quat_raw: np.ndarray | None = None,
) -> Measurement:
    """Build one otherwise-valid measurement with an optional targeted defect."""
    return Measurement(
        dt=dt,
        dtheta=np.array([0.1, 0.2, 0.3]) if dtheta is None else dtheta,
        dv=np.array([0.4, 0.5, 0.6]) if dv is None else dv,
        quat_raw=np.array([1.0, 0.0, 0.0, 0.0]) if quat_raw is None else quat_raw,
    )


def test_assess_returns_ok_for_structurally_usable_measurement():
    """Finite physical values with positive dt receive no quality flags."""
    assert assess(_measurement()) == QualityFlags.OK


def test_assess_flags_nonfinite_dt_without_nonpositive_dt():
    """NaN dt is non-finite, not a meaningful nonpositive duration."""
    flags = assess(_measurement(dt=np.nan))

    assert flags == QualityFlags.NONFINITE_DT


@pytest.mark.parametrize("dt", [0.0, -0.01])
def test_assess_flags_finite_nonpositive_dt(dt):
    """Zero and negative finite intervals are invalid propagation durations."""
    assert assess(_measurement(dt=dt)) == QualityFlags.NONPOSITIVE_DT


@pytest.mark.parametrize(
    ("field", "flag"),
    [
        ("dtheta", QualityFlags.NONFINITE_DTHETA),
        ("dv", QualityFlags.NONFINITE_DV),
        ("quat_raw", QualityFlags.NONFINITE_QUAT_RAW),
    ],
)
def test_assess_flags_nonfinite_vector_field(field, flag):
    """Each physical vector receives its own independent non-finite flag."""
    values = {
        "dtheta": np.array([0.1, np.inf, 0.3]),
        "dv": np.array([0.4, np.nan, 0.6]),
        "quat_raw": np.array([1.0, 0.0, -np.inf, 0.0]),
    }

    assert assess(_measurement(**{field: values[field]})) == flag


def test_assess_combines_independent_quality_flags():
    """Several structural defects are retained rather than stopping at one."""
    flags = assess(_measurement(
        dt=np.nan,
        dtheta=np.array([np.nan, 0.2, 0.3]),
        dv=np.array([0.4, np.inf, 0.6]),
        quat_raw=np.array([1.0, 0.0, 0.0, np.nan]),
    ))

    assert flags == (
        QualityFlags.NONFINITE_DT
        | QualityFlags.NONFINITE_DTHETA
        | QualityFlags.NONFINITE_DV
        | QualityFlags.NONFINITE_QUAT_RAW
    )


def test_assess_does_not_mutate_measurement():
    """Quality assessment is observation-only and leaves its input unchanged."""
    measurement = _measurement()
    dtheta_before = measurement.dtheta.copy()
    dv_before = measurement.dv.copy()
    quat_raw_before = measurement.quat_raw.copy()

    assess(measurement)

    assert measurement.dt == 0.01
    np.testing.assert_array_equal(measurement.dtheta, dtheta_before)
    np.testing.assert_array_equal(measurement.dv, dv_before)
    np.testing.assert_array_equal(measurement.quat_raw, quat_raw_before)
