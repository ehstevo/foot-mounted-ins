"""Focused tests for declared, reproducible GNSS tuning candidates."""

from types import SimpleNamespace

import numpy as np

from bootins.analysis.gnss_tuning import (
    GnssSweepCandidate,
    generate_gnss_sweep_candidates,
    parse_gnss_sweep_plan,
    resolve_gnss_sweep_candidate,
    summarize_gnss_sweep_candidate,
)
from bootins.estimation.clock_augmented_esekf import NOM_TAU
from bootins.replay.gnss_fusion_models import DelayedGnssUpdate, DelayedGnssUpdateStatus


def _plan() -> dict[str, object]:
    return {
        "mode": "one-at-a-time",
        "parameters": [{
            "name": "position-q", "path": "filter.process_covariance_diagonal.position",
            "values": [[0.001, 0.001, 0.001], [0.01, 0.01, 0.01]],
        }],
    }


def test_plan_generates_a_baseline_and_isolated_parameter_candidates():
    candidates = generate_gnss_sweep_candidates(parse_gnss_sweep_plan(_plan()), max_candidates=3)

    assert [candidate.name for candidate in candidates] == ["baseline", "position-q-1", "position-q-2"]
    assert candidates[0].overrides == ()


def test_resolution_changes_only_the_candidate_path_and_preserves_initial_calibration():
    base = {
        "filter": {
            "initial_accel_bias_b": [0.5, -0.25, 0.125],
            "process_covariance_diagonal": {"position": [1.0, 2.0, 3.0]},
        }
    }
    candidate = generate_gnss_sweep_candidates(parse_gnss_sweep_plan(_plan()), max_candidates=3)[2]

    resolved = resolve_gnss_sweep_candidate(base, candidate)

    assert base["filter"]["initial_accel_bias_b"] == [0.5, -0.25, 0.125]
    assert resolved["filter"]["initial_accel_bias_b"] == [0.5, -0.25, 0.125]
    assert resolved["filter"]["process_covariance_diagonal"]["position"] == [0.01, 0.01, 0.01]


def test_summary_keeps_accepted_and_rejected_nis_evidence_separate():
    updates = (
        DelayedGnssUpdate(DelayedGnssUpdateStatus.ACCEPTED, 0.2, 0.2, 1.0),
        DelayedGnssUpdate(DelayedGnssUpdateStatus.REJECTED, 0.2, 0.2, 12.0),
    )
    nominals = np.zeros((2, NOM_TAU + 1))
    result = SimpleNamespace(
        time=np.array([0.0, 2.5]), nominals=nominals, gnss_updates=updates,
        gnss_fusion_diagnostics=tuple(SimpleNamespace(update=update) for update in updates),
        invalid_gnss_fix_count=0,
        summary=SimpleNamespace(dropped_before_imu_count=0, dropped_after_imu_count=0),
    )

    summary = summarize_gnss_sweep_candidate(GnssSweepCandidate("baseline", ()), result)

    assert (summary.accepted_count, summary.rejected_count) == (1, 1)
    assert summary.accepted_nis.median == 1.0
    assert summary.rejected_nis.median == 12.0
    assert isinstance(summary.final_position_ned, np.ndarray)
    assert isinstance(summary.final_velocity_ned, np.ndarray)
    assert not summary.final_position_ned.flags.writeable
    assert not summary.final_velocity_ned.flags.writeable

    nominals[-1, 0] = 42.0

    assert summary.final_position_ned[0] == 0.0
