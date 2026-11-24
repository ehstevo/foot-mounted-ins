"""Focused parsing checks for the GNSS, INS, and ZUPT navigation profile."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from bootins.config.gnss_ins_zupt import (
    load_gnss_zupt_config,
    load_gnss_zupt_config_document,
    parse_gnss_zupt_config,
)


def _document(initial_accel_bias_b: list[float] | None = None) -> dict[str, object]:
    return {
        "startup": {
            "initial_stance_duration": 0.5, "initial_stance_search_limit": 5.0,
            "terminal_stance_duration": 0.75, "terminal_stance_search_limit": 6.0,
            "course_minimum_fix_count": 10, "course_minimum_baseline": 25.0,
            "course_maximum_cross_track_rms": 2.5,
        },
        "detector": {"window_size": 7, "threshold": 5e5, "sigma_a": 0.25, "sigma_g": 0.1},
        "filter": {
            "initial_accel_bias_b": [-2.5, 0.0, 4.25] if initial_accel_bias_b is None else initial_accel_bias_b,
            "initial_covariance_diagonal": {
                "position": [1.0, 2.0, 3.0], "velocity": [4.0, 5.0, 6.0], "attitude": [7.0, 8.0, 9.0],
                "accel_bias": [10.0, 11.0, 12.0], "gyro_bias": [13.0, 14.0, 15.0], "clock_offset": 16.0,
            },
            "process_covariance_diagonal": {
                "position": [17.0, 18.0, 19.0], "velocity": [20.0, 21.0, 22.0], "attitude": [23.0, 24.0, 25.0],
                "accel_bias": [26.0, 27.0, 28.0], "gyro_bias": [29.0, 30.0, 31.0], "clock_offset": 32.0,
            },
            "coast_process_covariance_diagonal": {
                "position": [39.0, 40.0, 41.0], "velocity": [42.0, 43.0, 44.0], "attitude": [45.0, 46.0, 47.0],
                "accel_bias": [0.0, 0.0, 0.0], "gyro_bias": [0.0, 0.0, 0.0], "clock_offset": 0.0,
            },
            "zupt_measurement_covariance_diagonal": [33.0, 34.0, 35.0],
            "gnss_measurement_covariance_diagonal": [36.0, 37.0, 38.0], "nis_threshold": 39.0,
        },
    }


def _write(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_loader_builds_clock_aware_covariances_and_nonzero_initial_accel_bias(tmp_path):
    config = load_gnss_zupt_config(_write(tmp_path, _document()))

    np.testing.assert_allclose(config.initial_accel_bias_b, [-2.5, 0.0, 4.25])
    np.testing.assert_allclose(np.diag(config.P0), np.arange(1.0, 17.0))
    np.testing.assert_allclose(np.diag(config.Q), np.arange(17.0, 33.0))
    np.testing.assert_allclose(np.diag(config.R_zupt), [33.0, 34.0, 35.0])
    np.testing.assert_allclose(np.diag(config.R_gnss), [36.0, 37.0, 38.0])


def test_loader_preserves_an_explicit_zero_initial_accel_bias(tmp_path):
    config = load_gnss_zupt_config(_write(tmp_path, _document([0.0, 0.0, 0.0])))

    np.testing.assert_array_equal(config.initial_accel_bias_b, np.zeros(3))


def test_document_loader_and_parser_support_derived_profile_resolution(tmp_path):
    path = _write(tmp_path, _document())
    document = load_gnss_zupt_config_document(path)
    document["filter"]["initial_accel_bias_b"] = [0.5, -0.25, 0.125]

    config = parse_gnss_zupt_config(document)

    np.testing.assert_allclose(config.initial_accel_bias_b, [0.5, -0.25, 0.125])


def test_loader_rejects_missing_or_nonfinite_initial_accel_bias(tmp_path):
    missing = _document()
    missing["filter"].pop("initial_accel_bias_b")
    with pytest.raises(ValueError, match="initial_accel_bias_b"):
        load_gnss_zupt_config(_write(tmp_path, missing))

    with pytest.raises(ValueError, match="initial_accel_bias_b"):
        load_gnss_zupt_config(_write(tmp_path, _document([1.0, float("inf"), 3.0])))


def test_loader_rejects_unfilled_tracked_public_template():
    template = Path(__file__).resolve().parents[1] / "config" / "gnss_ins_zupt" / "example.yaml"

    with pytest.raises(ValueError, match="initial_stance_duration"):
        load_gnss_zupt_config(template)


def test_loader_rejects_invalid_gnss_measurement_covariance(tmp_path):
    document = _document()
    document["filter"]["gnss_measurement_covariance_diagonal"] = [1.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="gnss_measurement_covariance_diagonal"):
        load_gnss_zupt_config(_write(tmp_path, document))
