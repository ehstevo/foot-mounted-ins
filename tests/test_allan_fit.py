"""Analytical checks for caller-declared Allan power-law fits."""

import numpy as np
import pytest

from bootins.analysis.allan_fit import fit_allan_power_law


def test_fit_recovers_white_noise_slope_and_reference_deviation():
    tau = np.geomspace(0.01, 10.0, 21)
    deviation = 4.5e-5 * tau**-0.5

    result = fit_allan_power_law(tau, deviation, tau_min=0.01, tau_max=1.0, reference_tau=1.0)

    assert result.slope == pytest.approx(-0.5)
    assert result.reference_deviation == pytest.approx(4.5e-5)
    assert result.residual_rms == pytest.approx(0.0, abs=1e-14)
    assert result.sample_count == np.count_nonzero(tau <= 1.0)


def test_fit_honors_the_declared_interval_and_reference_time():
    tau = np.array([0.01, 0.02, 0.1, 0.5, 1.0, 2.0])
    deviation = 3.0 * tau**-0.5
    deviation[[0, -1]] *= [100.0, 0.01]

    result = fit_allan_power_law(tau, deviation, tau_min=0.02, tau_max=1.0, reference_tau=1.0)

    assert result.slope == pytest.approx(-0.5)
    assert result.reference_deviation == pytest.approx(3.0)
    assert result.residual_rms == pytest.approx(0.0, abs=1e-14)
    assert result.sample_count == 4
