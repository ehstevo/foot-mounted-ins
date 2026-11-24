"""Analytical checks for overlapping multi-axis Allan deviation."""

import numpy as np

from bootins.analysis.allan import overlapping_allan_deviation


def test_constant_offsets_cancel_for_every_requested_axis_and_cluster_size():
    result = overlapping_allan_deviation(
        np.tile(np.array([2.0, -3.0, 9.8]), (32, 1)),
        sample_period=0.01,
        cluster_sizes=(1, 2, 4, 8, 16),
    )

    np.testing.assert_array_equal(result.cluster_sizes, [1, 2, 4, 8, 16])
    np.testing.assert_allclose(result.tau, [0.01, 0.02, 0.04, 0.08, 0.16])
    np.testing.assert_allclose(result.deviation, 0.0, atol=1e-12)


def test_overlapping_statistic_matches_hand_calculated_multiaxis_values():
    result = overlapping_allan_deviation(
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]),
        sample_period=0.25,
        cluster_sizes=(1, 2),
    )

    np.testing.assert_allclose(result.tau, [0.25, 0.50])
    np.testing.assert_allclose(
        result.deviation,
        [[np.sqrt(0.5), np.sqrt(50.0)], [np.sqrt(2.0), np.sqrt(200.0)]],
    )
