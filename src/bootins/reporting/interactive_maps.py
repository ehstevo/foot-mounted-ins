"""Write interactive Folium maps from caller-prepared geographic trajectories.

This module accepts only WGS-84 LLH histories already prepared by a caller. It
has no knowledge of IMU captures, filters, local navigation frames, or geodesy
origins. Generated HTML embeds the selected displayed evidence and layer
definitions, but browsers normally retrieve basemap tiles from external
providers when they display it. A map's visual agreement is not an accuracy
measurement or a replacement for the original navigation evidence.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import html
from pathlib import Path
import re

import numpy as np


class FoliumBasemapStyle(Enum):
    """Select one curated external tiled background for an interactive map."""

    STREETS = "streets"
    SATELLITE = "satellite"
    LIGHT = "light"
    TOPOGRAPHIC = "topographic"


class GnssMapStatus(Enum):
    """Classify one receiver-valid GNSS point for presentation only.

    ``ACCEPTED`` and ``REJECTED`` were evaluated by GNSS fusion, with the latter
    NIS-rejected. ``STALE`` was too old for usable replay history;
    ``FUTURE_DROPPED`` mapped after the evaluable IMU timeline; and
    ``NOT_EVALUATED`` reached no fusion evaluation. These statuses are evidence
    of filter handling, not a claim that a receiver position is correct or wrong.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    FUTURE_DROPPED = "future_dropped"
    NOT_EVALUATED = "not_evaluated"


_GNSS_STATUS_STYLE = {
    GnssMapStatus.ACCEPTED: ("#2ca02c", "Accepted GNSS"),
    GnssMapStatus.REJECTED: ("#d62728", "NIS-rejected GNSS"),
    GnssMapStatus.STALE: ("#7f7f7f", "Stale GNSS"),
    GnssMapStatus.FUTURE_DROPPED: ("#9467bd", "Future-dropped GNSS"),
    GnssMapStatus.NOT_EVALUATED: ("#ff7f0e", "GNSS not evaluated"),
}
_DEFAULT_MAX_TRAJECTORY_POINTS = 10_000


@dataclass(frozen=True)
class LlhMapTrace:
    """Represent one named LLH trajectory layer for a comparison map.

    Attributes:
        label: Display-layer name supplied by the caller.
        positions_llh: ``(N, 3)`` ``[latitude, longitude, ellipsoidal height]``
            history in rad, rad, and m.
        color: CSS-compatible line color.
        visible: Initial Folium layer visibility.
        include_in_bounds: Whether this trace contributes to initial map bounds.

    Excluding a known divergent trace from bounds leaves its evidence available
    without forcing the initial view away from useful comparison geometry.
    """

    label: str
    positions_llh: object
    color: str
    visible: bool = True
    include_in_bounds: bool = True


def _validated_llh_positions(value: object, field: str) -> np.ndarray:
    """Return finite `[latitude, longitude, height]` positions in radians/meters."""
    try:
        positions = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be convertible to a float array.") from error
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError(f"{field} must have shape (sample_count, 3).")
    if positions.shape[0] == 0:
        raise ValueError(f"{field} must contain at least one position.")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{field} must contain only finite values.")
    if np.any(np.abs(positions[:, 0]) > np.pi / 2.0):
        raise ValueError(f"{field} latitude must be expressed in radians.")
    if np.any(np.abs(positions[:, 1]) > np.pi):
        raise ValueError(f"{field} longitude must be expressed in radians.")
    return positions


def _validated_gnss_presentation(
    positions: np.ndarray | None,
    statuses: Sequence[GnssMapStatus] | None,
    tooltips: Sequence[str] | None,
) -> tuple[np.ndarray | None, tuple[GnssMapStatus, ...] | None, tuple[str, ...] | None]:
    """Validate optional GNSS positions and their aligned presentation metadata."""
    if positions is None:
        if statuses is not None or tooltips is not None:
            raise ValueError("GNSS statuses and tooltips require GNSS positions.")
        return None, None, None

    validated_positions = _validated_llh_positions(positions, "gnss_positions_llh")
    if statuses is None:
        validated_statuses = (GnssMapStatus.NOT_EVALUATED,) * validated_positions.shape[0]
    else:
        validated_statuses = tuple(statuses)
        if len(validated_statuses) != validated_positions.shape[0]:
            raise ValueError("GNSS statuses must align 1:1 with GNSS positions.")
        if not all(isinstance(status, GnssMapStatus) for status in validated_statuses):
            raise ValueError("GNSS statuses must contain only GnssMapStatus values.")

    if tooltips is None:
        validated_tooltips = ("GNSS receiver fix",) * validated_positions.shape[0]
    else:
        validated_tooltips = tuple(tooltips)
        if len(validated_tooltips) != validated_positions.shape[0]:
            raise ValueError("GNSS tooltips must align 1:1 with GNSS positions.")
        if not all(isinstance(tooltip, str) for tooltip in validated_tooltips):
            raise ValueError("GNSS tooltips must contain only strings.")

    return validated_positions, validated_statuses, validated_tooltips


def _add_basemap(map_object: object, basemap: FoliumBasemapStyle) -> None:
    """Attach one attributed Folium tile layer without fetching any tiles yet."""
    import folium

    if basemap is FoliumBasemapStyle.STREETS:
        folium.TileLayer("OpenStreetMap", name="Streets", control=False).add_to(map_object)
    elif basemap is FoliumBasemapStyle.LIGHT:
        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Tiles © Esri",
            name="Light",
            control=False,
        ).add_to(map_object)
    elif basemap is FoliumBasemapStyle.TOPOGRAPHIC:
        folium.TileLayer("OpenTopoMap", name="Topographic", control=False).add_to(map_object)
    elif basemap is FoliumBasemapStyle.SATELLITE:
        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Tiles © Esri",
            name="Satellite",
            control=False,
        ).add_to(map_object)
    else:
        raise ValueError("basemap must be a FoliumBasemapStyle value.")


def _degree_locations(positions_llh: np.ndarray) -> list[list[float]]:
    """Convert radians/meters LLH rows into Folium latitude/longitude locations."""
    degrees = np.degrees(positions_llh[:, :2])
    return degrees.tolist()


def _display_positions(positions_llh: np.ndarray, maximum_count: int) -> np.ndarray:
    """Uniformly decimate a visual path while retaining its exact endpoints."""
    if not isinstance(maximum_count, int) or isinstance(maximum_count, bool):
        raise ValueError("max_trajectory_points must be an integer.")
    if maximum_count < 2:
        raise ValueError("max_trajectory_points must be at least two.")
    if positions_llh.shape[0] <= maximum_count:
        return positions_llh
    indices = np.linspace(0, positions_llh.shape[0] - 1, maximum_count, dtype=int)
    return positions_llh[indices]


def _validated_map_traces(
    traces: Sequence[LlhMapTrace],
) -> tuple[tuple[LlhMapTrace, np.ndarray], ...]:
    """Validate named LLH map layers without changing caller-owned positions."""
    if isinstance(traces, (str, bytes)) or not isinstance(traces, Sequence):
        raise ValueError("traces must be a sequence of LlhMapTrace values.")
    if not traces:
        raise ValueError("traces must contain at least one trajectory.")

    validated_traces: list[tuple[LlhMapTrace, np.ndarray]] = []
    for index, trace in enumerate(traces):
        if not isinstance(trace, LlhMapTrace):
            raise ValueError("traces must contain only LlhMapTrace values.")
        if not isinstance(trace.label, str) or not trace.label.strip():
            raise ValueError(f"trace {index} label must be a nonempty string.")
        if not isinstance(trace.color, str) or re.fullmatch(
            r"#[0-9A-Fa-f]{6}", trace.color
        ) is None:
            raise ValueError(f"trace {index} color must be a #RRGGBB string.")
        if not isinstance(trace.visible, bool):
            raise ValueError(f"trace {index} visible must be Boolean.")
        if not isinstance(trace.include_in_bounds, bool):
            raise ValueError(f"trace {index} include_in_bounds must be Boolean.")
        validated_traces.append(
            (
                trace,
                _validated_llh_positions(
                    trace.positions_llh,
                    f"trace {index} positions_llh",
                ),
            )
        )

    if not any(trace.include_in_bounds for trace, _positions in validated_traces):
        raise ValueError("At least one trace must be included in the initial map bounds.")
    return tuple(validated_traces)


def _terminal_lockout_index(statuses: Sequence[GnssMapStatus]) -> int | None:
    """Return the first rejected point after the final accepted GNSS outcome."""
    accepted_indices = [
        index for index, status in enumerate(statuses) if status is GnssMapStatus.ACCEPTED
    ]
    if not accepted_indices:
        return None
    for index in range(accepted_indices[-1] + 1, len(statuses)):
        if statuses[index] is GnssMapStatus.REJECTED:
            return index
    return None


def write_llh_trajectory_map(
    output_path: str | Path,
    trajectory_positions_llh: object,
    *,
    gnss_positions_llh: object | None = None,
    gnss_statuses: Sequence[GnssMapStatus] | None = None,
    gnss_tooltips: Sequence[str] | None = None,
    basemap: FoliumBasemapStyle = FoliumBasemapStyle.LIGHT,
    title: str = "GNSS + INS + ZUPT replay",
    max_trajectory_points: int = _DEFAULT_MAX_TRAJECTORY_POINTS,
) -> None:
    """Write an interactive estimated-trajectory map with optional GNSS layers.

    Args:
        output_path: Destination HTML path, overwritten by the map artifact.
        trajectory_positions_llh: Estimated ``(N, 3)`` LLH states in rad, rad,
            and m.
        gnss_positions_llh: Optional aligned receiver-fix LLH history in the
            same units.
        gnss_statuses: Optional per-fix fusion-handling status.
        gnss_tooltips: Optional per-fix explanatory presentation text.
        basemap: Curated external tile-layer style.
        title: Nonempty HTML map title.
        max_trajectory_points: Maximum displayed estimated states before uniform
            presentation-only decimation.

    Raises:
        ValueError: Coordinates, GNSS metadata alignment, style, title, or point
            limit are invalid.
        RuntimeError: The optional Folium dependency is unavailable.
        OSError: The HTML artifact cannot be written.

    Notes:
        The map contains the estimated path, endpoint markers, and, when given,
        the receiver-valid path split into status layers. GNSS status is filter
        handling evidence, not a correctness label. Estimated states are uniformly
        decimated only for HTML display (exact endpoints retained); no CSV or
        navigation evidence changes. Tile retrieval occurs later in the browser
        and may fail independently of locally written trajectory layers.
    """
    if not isinstance(basemap, FoliumBasemapStyle):
        raise ValueError("basemap must be a FoliumBasemapStyle value.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a nonempty string.")

    trajectory = _validated_llh_positions(
        trajectory_positions_llh,
        "trajectory_positions_llh",
    )
    gnss_positions, statuses, tooltips = _validated_gnss_presentation(
        gnss_positions_llh,
        gnss_statuses,
        gnss_tooltips,
    )
    displayed_trajectory = _display_positions(trajectory, max_trajectory_points)

    try:
        import folium
    except ImportError as error:
        raise RuntimeError(
            "Interactive LLH maps require the 'folium' map dependency."
        ) from error

    trajectory_locations = _degree_locations(displayed_trajectory)
    coordinate_histories = [trajectory[:, :2]]
    if gnss_positions is not None:
        coordinate_histories.append(gnss_positions[:, :2])
    all_coordinates = np.degrees(np.vstack(coordinate_histories))
    center = np.mean(all_coordinates, axis=0).tolist()

    map_object = folium.Map(location=center, tiles=None, control_scale=True)
    _add_basemap(map_object, basemap)
    map_object.get_root().html.add_child(
        folium.Element(
            "<h3 style=\"position: fixed; top: 8px; left: 50px; z-index: 1000; "
            "background: rgba(255, 255, 255, 0.85); padding: 6px 10px; "
            "border-radius: 4px;\">"
            f"{html.escape(title)}"
            "</h3>"
        )
    )

    trajectory_layer = folium.FeatureGroup(name="Estimated trajectory", show=True)
    folium.PolyLine(
        trajectory_locations,
        color="#0072B2",
        weight=4,
        opacity=0.9,
        tooltip=(
            "GNSS + INS + ZUPT estimated trajectory "
            f"({displayed_trajectory.shape[0]} displayed of {trajectory.shape[0]} states)"
        ),
    ).add_to(trajectory_layer)
    folium.Marker(
        trajectory_locations[0],
        tooltip="Estimated trajectory start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(trajectory_layer)
    folium.Marker(
        trajectory_locations[-1],
        tooltip="Estimated trajectory end",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(trajectory_layer)
    trajectory_layer.add_to(map_object)

    if gnss_positions is not None:
        assert statuses is not None
        assert tooltips is not None
        gnss_locations = _degree_locations(gnss_positions)
        receiver_path_layer = folium.FeatureGroup(name="GNSS receiver path", show=True)
        folium.PolyLine(
            gnss_locations,
            color="#666666",
            weight=2,
            opacity=0.6,
            tooltip="Receiver-valid GNSS fixes in source time order",
        ).add_to(receiver_path_layer)
        receiver_path_layer.add_to(map_object)

        for status, (color, layer_name) in _GNSS_STATUS_STYLE.items():
            layer = folium.FeatureGroup(name=layer_name, show=True)
            for location, point_status, tooltip in zip(
                gnss_locations,
                statuses,
                tooltips,
                strict=True,
            ):
                if point_status is not status:
                    continue
                folium.CircleMarker(
                    location,
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    tooltip=folium.Tooltip(
                        html.escape(tooltip).replace("\n", "<br>"),
                        sticky=True,
                    ),
                ).add_to(layer)
            layer.add_to(map_object)

        lockout_index = _terminal_lockout_index(statuses)
        if lockout_index is not None:
            lockout_layer = folium.FeatureGroup(name="GNSS integrity marker", show=True)
            folium.Marker(
                gnss_locations[lockout_index],
                tooltip="Terminal NIS lockout begins",
                icon=folium.Icon(color="red", icon="warning-sign"),
            ).add_to(lockout_layer)
            lockout_layer.add_to(map_object)

    map_object.fit_bounds(
        [
            [
                float(np.min(all_coordinates[:, 0])),
                float(np.min(all_coordinates[:, 1])),
            ],
            [
                float(np.max(all_coordinates[:, 0])),
                float(np.max(all_coordinates[:, 1])),
            ],
        ]
    )
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.save(str(Path(output_path)))


def write_llh_trajectory_comparison_map(
    output_path: str | Path,
    traces: Sequence[LlhMapTrace],
    *,
    basemap: FoliumBasemapStyle = FoliumBasemapStyle.LIGHT,
    title: str = "Trajectory comparison",
) -> None:
    """Write a layered interactive map from caller-prepared LLH trajectories.

    Args:
        output_path: Destination HTML path, overwritten by the map artifact.
        traces: Nonempty named LLH layers with caller-declared provenance.
        basemap: Curated external tile-layer style.
        title: Nonempty HTML map title.

    Raises:
        ValueError: Traces, LLH coordinates, style, title, or bounds selection is
            invalid.
        RuntimeError: The optional Folium dependency is unavailable.
        OSError: The HTML artifact cannot be written.

    Notes:
        This renderer accepts no filter, capture, GNSS, PPK, or registration
        details. Callers supply already-aligned traces and truthful labels. Every
        trace remains one full-resolution polyline; map layers visualize evidence
        but do not establish positional accuracy.
    """
    if not isinstance(basemap, FoliumBasemapStyle):
        raise ValueError("basemap must be a FoliumBasemapStyle value.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a nonempty string.")

    validated_traces = _validated_map_traces(traces)
    bound_positions = [
        positions[:, :2]
        for trace, positions in validated_traces
        if trace.include_in_bounds
    ]
    all_bound_coordinates = np.degrees(np.vstack(bound_positions))
    center = np.mean(all_bound_coordinates, axis=0).tolist()

    try:
        import folium
    except ImportError as error:
        raise RuntimeError(
            "Interactive LLH maps require the 'folium' map dependency."
        ) from error

    map_object = folium.Map(
        location=center,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    _add_basemap(map_object, basemap)
    map_object.get_root().html.add_child(
        folium.Element(
            "<h3 style=\"position: fixed; top: 8px; left: 50px; z-index: 1000; "
            "background: rgba(255, 255, 255, 0.85); padding: 6px 10px; "
            "border-radius: 4px;\">"
            f"{html.escape(title)}"
            "</h3>"
        )
    )

    for trace, positions in validated_traces:
        locations = _degree_locations(positions)
        layer = folium.FeatureGroup(name=trace.label, show=trace.visible)
        folium.PolyLine(
            locations,
            color=trace.color,
            weight=2,
            opacity=0.9,
            tooltip=f"{trace.label} ({positions.shape[0]} positions)",
        ).add_to(layer)
        layer.add_to(map_object)

    map_object.fit_bounds(
        [
            [
                float(np.min(all_bound_coordinates[:, 0])),
                float(np.min(all_bound_coordinates[:, 1])),
            ],
            [
                float(np.max(all_bound_coordinates[:, 0])),
                float(np.max(all_bound_coordinates[:, 1])),
            ],
        ]
    )
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.save(str(Path(output_path)))
