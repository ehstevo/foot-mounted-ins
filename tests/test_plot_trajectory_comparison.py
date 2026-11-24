"""Focused contracts for the exported-artifact trajectory comparison CLI."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from bootins.reporting.navigation_export import (
    NavigationHistoryLayout,
    write_esekf_history_csv,
    write_geographic_ned_origin_sidecar,
    write_raw_ins_trajectory_csv,
)
from bootins.sensors.gnss_observations import GnssFixLLH
from bootins.sensors.gnss_table import write_gnss_fix_table


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "plot_trajectory_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("plot_trajectory_comparison", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
plot_trajectory_comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_trajectory_comparison)


def _write_ppk_solution(path: Path) -> None:
    """Write two compact, valid RTKLIB geodetic solution rows."""
    path.write_text(
        "\n".join(
            (
                "% program : RTKLIB ver.2.4.3",
                "%  GPST latitude(deg) longitude(deg) height(m) Q ns "
                "sdn(m) sde(m) sdu(m) sdne(m) sdeu(m) sdun(m) age(s) ratio",
                "2026/11/04 15:53:22.000 39.478000000 -76.137000000 -21.8 "
                "1 12 0.02 0.03 0.08 0.00 0.00 0.00 0.0 3.0",
                "2026/11/04 15:53:23.000 39.478010000 -76.137010000 -21.7 "
                "2 11 0.03 0.04 0.09 0.00 0.00 0.00 1.0 2.0",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_aided_state(path: Path) -> None:
    """Write a minimal geographic-NED state history plus its exact origin."""
    nominals = np.zeros((2, 17))
    nominals[1, :3] = [1.0, 2.0, 3.0]
    covariances = np.repeat(np.eye(16)[None, :, :], 2, axis=0)
    write_esekf_history_csv(
        path,
        np.array([0.0, 0.1]),
        nominals,
        covariances,
        layout=NavigationHistoryLayout.GEOGRAPHIC_NED_CLOCK,
    )
    write_geographic_ned_origin_sidecar(
        path,
        np.radians(39.478),
        np.radians(-76.137),
        -21.8,
    )


def _write_raw_state(path: Path) -> None:
    """Write a minimal local-FRD raw trajectory artifact."""
    write_raw_ins_trajectory_csv(
        path,
        np.array([0.0, 0.1]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.zeros((2, 3)),
        np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )


def _write_gnss_fixes(path: Path, *, fix_valid: bool = True) -> None:
    """Write one validly structured GNSS table row for anchor selection."""
    write_gnss_fix_table(
        path,
        (
            GnssFixLLH(
                gps_week=2443,
                gps_tow=316402.0,
                host_arrival_monotonic=12.0,
                latitude=np.radians(39.478),
                longitude=np.radians(-76.137),
                height_ellipsoid=-21.8,
                fix_valid=fix_valid,
                satellite_count=12,
                horizontal_accuracy=0.5,
                vertical_accuracy=0.8,
                native_record_index=7,
            ),
        ),
    )


def test_load_rtklib_pos_preserves_gpst_and_quality_evidence(tmp_path):
    """The adapter keeps fixed and float evidence rather than collapsing it."""
    solution_path = tmp_path / "ppk.pos"
    _write_ppk_solution(solution_path)

    points = plot_trajectory_comparison.load_rtklib_pos(solution_path)

    assert len(points) == 2
    assert (points[0].gps_week, points[0].gps_tow) == pytest.approx((2443, 316402.0))
    assert points[0].quality == 1
    assert points[1].quality == 2
    assert points[0].latitude == pytest.approx(np.radians(39.478))
    assert points[1].std_east == pytest.approx(0.04)


def test_manifest_builds_aligned_aided_local_and_ppk_layers(tmp_path):
    """Each frame conversion is explicit before the generic map renderer sees it."""
    solution_path = tmp_path / "ppk.pos"
    aided_path = tmp_path / "aided.csv"
    raw_path = tmp_path / "raw.csv"
    manifest_path = tmp_path / "comparison.yaml"
    _write_ppk_solution(solution_path)
    _write_aided_state(aided_path)
    _write_raw_state(raw_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "title": "Fixture comparison",
                "basemap": "light",
                "ppk_solution": "ppk.pos",
                "traces": [
                    {
                        "kind": "state_csv",
                        "path": "aided.csv",
                        "label": "Aided",
                        "color": "#0072B2",
                    },
                    {
                        "kind": "state_csv",
                        "path": "raw.csv",
                        "label": "Raw",
                        "color": "#D55E00",
                        "registration": {
                            "anchor_ppk_gpst": "2026-11-04T15:53:22.000",
                            "heading_deg_clockwise_from_north": 0.0,
                        },
                    },
                    {
                        "kind": "ppk",
                        "quality": "all",
                        "label": "PPK solution",
                        "color": "#009E73",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    title, basemap, traces = plot_trajectory_comparison._build_map_traces(manifest_path)

    assert title == "Fixture comparison"
    assert basemap.value == "light"
    assert [trace.label for trace in traces] == ["Aided", "Raw", "PPK solution"]
    np.testing.assert_allclose(
        traces[0].positions_llh[0],
        [np.radians(39.478), np.radians(-76.137), -21.8],
    )
    np.testing.assert_allclose(
        traces[1].positions_llh[0],
        [np.radians(39.478), np.radians(-76.137), -21.8],
    )
    assert traces[2].positions_llh.shape == (2, 3)


def test_manifest_builds_local_frd_trace_from_gnss_anchor_without_ppk(tmp_path):
    """A receiver-table anchor does not require an unrelated PPK solution."""
    raw_path = tmp_path / "raw.csv"
    fixes_path = tmp_path / "fixes.csv"
    manifest_path = tmp_path / "comparison.yaml"
    _write_raw_state(raw_path)
    _write_gnss_fixes(fixes_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "title": "GNSS-anchored local trace",
                "basemap": "light",
                "traces": [
                    {
                        "kind": "state_csv",
                        "path": "raw.csv",
                        "label": "INS + ZUPT",
                        "color": "#56B4E9",
                        "registration": {
                            "anchor_gnss_fix": {
                                "path": "fixes.csv",
                                "gps_week": 2443,
                                "gps_tow": 316402.0,
                            },
                            "heading_deg_clockwise_from_north": 0.0,
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _title, _basemap, traces = plot_trajectory_comparison._build_map_traces(manifest_path)

    np.testing.assert_allclose(
        traces[0].positions_llh[0],
        [np.radians(39.478), np.radians(-76.137), -21.8],
    )


@pytest.mark.parametrize(
    ("registration", "message"),
    [
        (
            {"heading_deg_clockwise_from_north": 0.0},
            "exactly one",
        ),
        (
            {
                "anchor_ppk_gpst": "2026-11-04T15:53:22.000",
                "anchor_gnss_fix": {
                    "path": "fixes.csv",
                    "gps_week": 2443,
                    "gps_tow": 316402.0,
                },
                "heading_deg_clockwise_from_north": 0.0,
            },
            "exactly one",
        ),
    ],
)
def test_manifest_requires_exactly_one_local_frd_anchor(tmp_path, registration, message):
    """A local trace cannot silently choose or blend its geographic origin."""
    raw_path = tmp_path / "raw.csv"
    manifest_path = tmp_path / "comparison.yaml"
    _write_raw_state(raw_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "title": "Invalid registration",
                "basemap": "light",
                "traces": [
                    {
                        "kind": "state_csv",
                        "path": "raw.csv",
                        "label": "Raw",
                        "color": "#D55E00",
                        "registration": registration,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        plot_trajectory_comparison._build_map_traces(manifest_path)


def test_manifest_rejects_invalid_gnss_anchor_row(tmp_path):
    """An invalid receiver fix cannot silently define a map registration."""
    raw_path = tmp_path / "raw.csv"
    fixes_path = tmp_path / "fixes.csv"
    manifest_path = tmp_path / "comparison.yaml"
    _write_raw_state(raw_path)
    _write_gnss_fixes(fixes_path, fix_valid=False)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "title": "Invalid GNSS anchor",
                "basemap": "light",
                "traces": [
                    {
                        "kind": "state_csv",
                        "path": "raw.csv",
                        "label": "Raw",
                        "color": "#D55E00",
                        "registration": {
                            "anchor_gnss_fix": {
                                "path": "fixes.csv",
                                "gps_week": 2443,
                                "gps_tow": 316402.0,
                            },
                            "heading_deg_clockwise_from_north": 0.0,
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="receiver-valid"):
        plot_trajectory_comparison._build_map_traces(manifest_path)


def test_manifest_requires_an_exact_declared_ppk_anchor(tmp_path):
    """Local-FRD alignment cannot silently snap an origin to a nearby epoch."""
    solution_path = tmp_path / "ppk.pos"
    raw_path = tmp_path / "raw.csv"
    manifest_path = tmp_path / "comparison.yaml"
    _write_ppk_solution(solution_path)
    _write_raw_state(raw_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "title": "Invalid anchor",
                "basemap": "light",
                "ppk_solution": "ppk.pos",
                "traces": [
                    {
                        "kind": "state_csv",
                        "path": "raw.csv",
                        "label": "Raw",
                        "color": "#D55E00",
                        "registration": {
                            "anchor_ppk_gpst": "2026-11-04T15:53:22.500",
                            "heading_deg_clockwise_from_north": 0.0,
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact PPK epoch"):
        plot_trajectory_comparison._build_map_traces(manifest_path)
