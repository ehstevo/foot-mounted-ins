"""Focused numerical checks for loose-coupled GNSS position aiding."""

import numpy as np

from bootins.aiding.gnss_position import (
    _H_POS,
    _pos_h,
    gnss_position_nis_ned,
    gnss_position_update_llh_gated,
    gnss_position_update_ned,
    gnss_position_update_ned_gated,
)
from bootins.estimation.esekf import ERR_DP, NOM_P, NOM_V, esekf_update


def _nominal() -> np.ndarray:
    return np.hstack((np.zeros(3), np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(6)))


def _model_jacobian() -> np.ndarray:
    jacobian = np.zeros((3, 15))
    jacobian[:, ERR_DP] = np.eye(3)
    return jacobian


def test_position_measurement_model_reads_only_nominal_position():
    nominal = _nominal()
    nominal[:6] = [1.0, -2.0, 3.0, 4.0, 5.0, 6.0]

    np.testing.assert_allclose(_pos_h(nominal), [1.0, -2.0, 3.0])
    np.testing.assert_allclose(_H_POS, _model_jacobian())


def test_ned_wrapper_matches_the_generic_esekf_update():
    nominal = _nominal()
    covariance = np.diag(np.arange(1.0, 16.0))
    measurement = np.array([3.0, -4.0, 2.0])
    measurement_covariance = np.diag([0.5, 0.75, 1.25])

    expected_nominal, expected_covariance = esekf_update(
        nominal, covariance, measurement, lambda state: state[NOM_P], _model_jacobian(), measurement_covariance
    )
    actual_nominal, actual_covariance = gnss_position_update_ned(
        nominal, covariance, measurement, measurement_covariance
    )

    np.testing.assert_allclose(actual_nominal, expected_nominal)
    np.testing.assert_allclose(actual_covariance, expected_covariance)


def test_position_fix_corrects_velocity_only_through_cross_covariance():
    nominal = _nominal()
    covariance = np.eye(15) * 1e-6
    covariance[0, 0] = 4.0
    covariance[3, 3] = covariance[0, 3] = covariance[3, 0] = 1.0

    corrected, _ = gnss_position_update_ned(nominal, covariance, [10.0, 0.0, 0.0], np.eye(3) * 0.5)

    assert corrected[NOM_V][0] > 0.0


def test_nis_matches_manual_innovation_covariance_formula():
    nominal = _nominal()
    nominal[NOM_P] = [1.0, -2.0, 0.5]
    covariance = np.diag(np.arange(1.0, 16.0))
    measurement = np.array([4.0, 1.0, -1.0])
    measurement_covariance = np.diag([0.5, 0.75, 1.25])
    innovation = measurement - nominal[NOM_P]
    innovation_covariance = _model_jacobian() @ covariance @ _model_jacobian().T + measurement_covariance

    expected_nis = innovation @ np.linalg.inv(innovation_covariance) @ innovation

    np.testing.assert_allclose(
        gnss_position_nis_ned(nominal, covariance, measurement, measurement_covariance),
        expected_nis,
    )


def test_gate_accepts_consistent_fixes_and_rejects_outliers_as_identity_noops():
    nominal = _nominal()
    covariance = np.eye(15)
    measurement_covariance = np.eye(3) * 0.25
    expected_nominal, expected_covariance = gnss_position_update_ned(
        nominal, covariance, [0.1, -0.1, 0.0], measurement_covariance
    )
    accepted_nominal, accepted_covariance, _, accepted = gnss_position_update_ned_gated(
        nominal, covariance, [0.1, -0.1, 0.0], measurement_covariance, 10.0
    )
    rejected_nominal, rejected_covariance, _, rejected = gnss_position_update_ned_gated(
        nominal, covariance, [100.0, 0.0, 0.0], measurement_covariance, 10.0
    )

    assert accepted and not rejected
    np.testing.assert_allclose(accepted_nominal, expected_nominal)
    np.testing.assert_allclose(accepted_covariance, expected_covariance)
    assert rejected_nominal is nominal
    assert rejected_covariance is covariance


def test_llh_gated_wrapper_reduces_to_the_local_ned_origin_case():
    nominal = _nominal()
    covariance = np.eye(15)

    updated, _, _, accepted = gnss_position_update_llh_gated(
        nominal, covariance, 0.7, -1.1, 125.0, 0.7, -1.1, 125.0, np.eye(3), 1.0
    )

    assert accepted
    np.testing.assert_allclose(updated[NOM_P], np.zeros(3))
