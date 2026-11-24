"""Focused parsing checks for the INS plus ZUPT navigation profile."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from bootins.config.ins_zupt import load_zupt_ins_config


def _document(initial_accel_bias_b: list[float] | None = None) -> dict[str, object]:
    return {
        "startup": {"minimum_stance_duration": 0.5, "startup_search_limit": 5.0},
        "detector": {"window_size": 7, "threshold": 5e5, "sigma_a": 0.25, "sigma_g": 0.1},
        "filter": {
            "initial_accel_bias_b": [-1.25, 0.0, 3.75] if initial_accel_bias_b is None else initial_accel_bias_b,
            "initial_covariance_diagonal": {
                "position": [1.0, 2.0, 3.0], "velocity": [4.0, 5.0, 6.0], "attitude": [7.0, 8.0, 9.0],
                "accel_bias": [10.0, 11.0, 12.0], "gyro_bias": [13.0, 14.0, 15.0],
            },
            "process_covariance_diagonal": {
                "position": [16.0, 17.0, 18.0], "velocity": [19.0, 20.0, 21.0], "attitude": [22.0, 23.0, 24.0],
                "accel_bias": [25.0, 26.0, 27.0], "gyro_bias": [28.0, 29.0, 30.0],
            },
            "coast_process_covariance_diagonal": {
                "position": [34.0, 35.0, 36.0], "velocity": [37.0, 38.0, 39.0], "attitude": [40.0, 41.0, 42.0],
                "accel_bias": [0.0, 0.0, 0.0], "gyro_bias": [0.0, 0.0, 0.0],
            },
            "zupt_measurement_covariance_diagonal": [31.0, 32.0, 33.0],
        },
    }


def _write(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_loader_builds_named_covariances_and_nonzero_initial_accel_bias(tmp_path):
    config = load_zupt_ins_config(_write(tmp_path, _document()))

    np.testing.assert_allclose(config.initial_accel_bias_b, [-1.25, 0.0, 3.75])
    np.testing.assert_allclose(np.diag(config.P0), np.arange(1.0, 16.0))
    np.testing.assert_allclose(np.diag(config.Q), np.arange(16.0, 31.0))
    np.testing.assert_allclose(np.diag(config.Q_coast), [*np.arange(34.0, 43.0), *np.zeros(6)])
    np.testing.assert_allclose(np.diag(config.R_zupt), [31.0, 32.0, 33.0])


def test_loader_preserves_an_explicit_zero_initial_accel_bias(tmp_path):
    config = load_zupt_ins_config(_write(tmp_path, _document([0.0, 0.0, 0.0])))

    np.testing.assert_array_equal(config.initial_accel_bias_b, np.zeros(3))


def test_loader_rejects_missing_or_nonfinite_initial_accel_bias(tmp_path):
    missing = _document()
    missing["filter"].pop("initial_accel_bias_b")
    with pytest.raises(ValueError, match="initial_accel_bias_b"):
        load_zupt_ins_config(_write(tmp_path, missing))

    with pytest.raises(ValueError, match="initial_accel_bias_b"):
        load_zupt_ins_config(_write(tmp_path, _document([1.0, float("nan"), 3.0])))


def test_loader_rejects_unfilled_tracked_public_template():
    template = Path(__file__).resolve().parents[1] / "config" / "ins_zupt" / "example.yaml"

    with pytest.raises(ValueError, match="minimum_stance_duration"):
        load_zupt_ins_config(template)


def test_loader_rejects_a_physically_invalid_coast_covariance(tmp_path):
    document = _document()
    document["filter"]["coast_process_covariance_diagonal"] = {
        "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0], "attitude": [0.0, 0.0, 0.0],
        "accel_bias": [0.0, 0.0, 0.0], "gyro_bias": [0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError, match="at least one positive"):
        load_zupt_ins_config(_write(tmp_path, document))
