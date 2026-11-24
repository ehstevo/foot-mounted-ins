"""Plot already-estimated local and geographic trajectory histories.

These helpers consume coordinate histories rather than raw BMU captures. Local
horizontal plots require an explicit metre-based frame and report horizontal
endpoint displacement in that same declared frame. Geographic LLH plots project
for tile alignment in Web Mercator while presenting longitude and latitude on
the visible axes. Projection and basemap operations are display-only; they do
not alter navigation estimates, produce a geodetic accuracy measure, or supply
a reference trajectory.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import numpy as np


_WEB_MERCATOR_RADIUS = 6378137.0
_WEB_MERCATOR_MAX_LATITUDE = float(np.arctan(np.sinh(np.pi)))
_BASEMAP_STYLE_ATTRIBUTE = "_bootins_basemap_style"
_MAP_MINIMUM_SPAN_M = 25.0
_MAP_MARGIN_FRACTION = 0.1


class HorizontalFrame(Enum):
    """Select the declared three-component frame of a local position history.

    ``LOCAL_FRD`` uses ``[Forward, Right, Down]``; ``GEOGRAPHIC_NED`` uses
    ``[North, East, Down]``. Both display the second horizontal component on x
    and first on y to preserve conventional rightward/eastward horizontal axes.
    """

    LOCAL_FRD = "local_frd"
    GEOGRAPHIC_NED = "geographic_ned"


class BasemapStyle(Enum):
    """Select the required tiled background for a geographic trajectory plot."""

    STREETS = "streets"
    SATELLITE = "satellite"
    LIGHT = "light"
    TOPOGRAPHIC = "topographic"


@dataclass(frozen=True)
class HorizontalClosure:
    """Represent displayed horizontal endpoint displacement in one frame.

    Attributes:
        frame: Source position convention used for this closure.
        x: Final-minus-initial x displacement in m: Right for local FRD or
            East for geographic NED.
        y: Final-minus-initial y displacement in m: Forward for local FRD or
            North for geographic NED.
        distance: Euclidean horizontal endpoint separation in m.

    Down is excluded from all closure fields.
    """

    frame: HorizontalFrame
    x: float
    y: float
    distance: float


@dataclass(frozen=True)
class HorizontalTrajectoryPlot:
    """Return a horizontal plot axes and its frame-qualified closure.

    Attributes:
        axes: Matplotlib axes receiving the trajectory, start, and end markers.
        closure: Horizontal endpoint displacement derived from displayed data.
    """

    axes: Axes
    closure: HorizontalClosure


@dataclass(frozen=True)
class LlhTrajectoryPlot:
    """Return an LLH map plot axes and its selected basemap style.

    Attributes:
        axes: Matplotlib axes containing display-projected paths and map tiles.
        basemap: Curated tile source selected for display.
    """

    axes: Axes
    basemap: BasemapStyle


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    """Return finite position vectors with one three-vector per epoch."""
    try:
        validated_positions = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("positions must be convertible to a float array.") from error

    if validated_positions.ndim != 2 or validated_positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (sample_count, 3).")
    if validated_positions.shape[0] == 0:
        raise ValueError("positions must contain at least one position.")
    if not np.all(np.isfinite(validated_positions)):
        raise ValueError("positions must contain only finite values.")
    return validated_positions


def _llh_to_web_mercator(positions_llh: np.ndarray) -> np.ndarray:
    """Project finite radian LLH positions to spherical Web Mercator meters."""
    latitude = positions_llh[:, 0]
    if np.any(np.abs(latitude) > _WEB_MERCATOR_MAX_LATITUDE):
        raise ValueError(
            "LLH latitude exceeds the Web Mercator coverage limit of "
            "approximately 85.051 degrees."
        )
    longitude = positions_llh[:, 1]
    return np.column_stack(
        (
            _WEB_MERCATOR_RADIUS * longitude,
            _WEB_MERCATOR_RADIUS
            * np.log(np.tan(np.pi / 4.0 + latitude / 2.0)),
        )
    )


def _add_basemap(axes: Axes, basemap: BasemapStyle) -> None:
    """Fetch and draw one attributed tiled basemap beneath established axes."""
    try:
        import contextily
    except ImportError as error:
        raise RuntimeError(
            "LLH trajectory plots require the 'contextily' map dependency."
        ) from error

    if basemap is BasemapStyle.STREETS:
        source = contextily.providers.Esri.WorldStreetMap
        zoom = "auto"
    elif basemap is BasemapStyle.SATELLITE:
        source = contextily.providers.Esri.WorldImagery
        zoom = "auto"
    elif basemap is BasemapStyle.LIGHT:
        source = contextily.providers.Esri.WorldGrayCanvas
        zoom = 16
    elif basemap is BasemapStyle.TOPOGRAPHIC:
        source = contextily.providers.Esri.WorldTopoMap
        zoom = "auto"
    else:
        raise ValueError("basemap must be a BasemapStyle value.")

    try:
        contextily.add_basemap(axes, source=source, zoom=zoom)
    except Exception as error:
        raise RuntimeError(
            f"Could not retrieve {basemap.value} basemap tiles."
        ) from error


def _ensure_basemap(axes: Axes, basemap: BasemapStyle) -> None:
    """Draw a basemap once or refuse a conflicting shared-axes style."""
    existing_basemap = getattr(axes, _BASEMAP_STYLE_ATTRIBUTE, None)
    if existing_basemap is None:
        _add_basemap(axes, basemap)
        setattr(axes, _BASEMAP_STYLE_ATTRIBUTE, basemap)
    elif existing_basemap is not basemap:
        raise ValueError("Shared LLH plot axes must use one basemap style.")


def _set_map_extent(axes: Axes, projected_paths: Sequence[np.ndarray]) -> None:
    """Set a padded equal-scale map extent that contains every projected path."""
    all_positions = np.vstack(projected_paths)
    east_min, north_min = np.min(all_positions, axis=0)
    east_max, north_max = np.max(all_positions, axis=0)
    span = max(
        float(east_max - east_min),
        float(north_max - north_min),
        _MAP_MINIMUM_SPAN_M,
    )
    padding = _MAP_MARGIN_FRACTION * span
    half_span = span / 2.0 + padding
    east_center = (east_min + east_max) / 2.0
    north_center = (north_min + north_max) / 2.0
    axes.set_xlim(east_center - half_span, east_center + half_span)
    axes.set_ylim(north_center - half_span, north_center + half_span)


def _add_trajectory(
    axes: Axes,
    positions_mercator: np.ndarray,
    trajectory_label: str,
) -> None:
    """Draw one projected path and unlabeled endpoint markers on map axes."""
    east = positions_mercator[:, 0]
    north = positions_mercator[:, 1]
    axes.plot(east, north, label=trajectory_label)
    axes.scatter(
        east[0],
        north[0],
        color="tab:green",
        marker="o",
        label="_nolegend_",
        zorder=3,
    )
    axes.scatter(
        east[-1],
        north[-1],
        color="tab:red",
        marker="X",
        label="_nolegend_",
        zorder=3,
    )


def _format_longitude(easting: float, _tick_index: int) -> str:
    """Format a Web-Mercator easting as its geographic longitude in degrees."""
    longitude = np.degrees(easting / _WEB_MERCATOR_RADIUS)
    return f"{longitude:.3f}°"


def _format_latitude(northing: float, _tick_index: int) -> str:
    """Format a Web-Mercator northing as its geographic latitude in degrees."""
    latitude = np.degrees(2.0 * np.arctan(np.exp(northing / _WEB_MERCATOR_RADIUS)) - np.pi / 2.0)
    return f"{latitude:.3f}°"


def _style_trajectory_axes(axes: Axes, title: str | None) -> None:
    """Apply the compact shared presentation contract for trajectory axes."""
    if title is not None:
        axes.set_title(title)
    axes.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize="small",
        framealpha=0.9,
    )


def _style_llh_axes(axes: Axes, title: str | None) -> None:
    """Present a Web-Mercator map with geographic coordinate labels."""
    axes.set_xlabel("Longitude (°)")
    axes.set_ylabel("Latitude (°)")
    axes.xaxis.set_major_formatter(FuncFormatter(_format_longitude))
    axes.yaxis.set_major_formatter(FuncFormatter(_format_latitude))
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)
    _style_trajectory_axes(axes, title)


def _axis_labels(frame: HorizontalFrame) -> tuple[str, str]:
    """Return displayed horizontal axis labels for one supported frame."""
    if frame is HorizontalFrame.LOCAL_FRD:
        return "Right displacement (m)", "Forward displacement (m)"
    if frame is HorizontalFrame.GEOGRAPHIC_NED:
        return "East displacement (m)", "North displacement (m)"
    raise ValueError("frame must be a HorizontalFrame value.")


def plot_horizontal_trajectory(
    positions: np.ndarray,
    *,
    frame: HorizontalFrame = HorizontalFrame.LOCAL_FRD,
    axes: Axes | None = None,
    trajectory_label: str = "Trajectory",
    title: str | None = None,
) -> HorizontalTrajectoryPlot:
    """Plot one declared-frame horizontal path and report its endpoint closure.

    Args:
        positions: Finite ``(N, 3)`` position history in m: ``[F, R, D]`` for
            local FRD or ``[N, E, D]`` for geographic NED.
        frame: Declared interpretation of ``positions``.
        axes: Existing axes for a caller-managed overlay, or ``None`` to create
            a figure and axes.
        trajectory_label: Legend label for the plotted path.
        title: Optional axes title.

    Returns:
        Populated axes and final-minus-initial horizontal closure in the declared
        frame.

    Raises:
        ValueError: The frame or finite three-component position history is
            invalid.

    Notes:
        The second horizontal component is displayed on x and the first on y;
        Down is not plotted or included in closure. Overlaid paths must already
        share a frame—the function does not register or transform estimates.
    """
    if not isinstance(frame, HorizontalFrame):
        raise ValueError("frame must be a HorizontalFrame value.")
    validated_positions = _validate_positions(positions)
    x_label, y_label = _axis_labels(frame)

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots()

    y_position = validated_positions[:, 0]
    x_position = validated_positions[:, 1]
    closure_x = float(x_position[-1] - x_position[0])
    closure_y = float(y_position[-1] - y_position[0])
    closure_distance = float(np.hypot(closure_x, closure_y))

    axes.plot(x_position, y_position, label=trajectory_label)
    axes.scatter(
        x_position[0],
        y_position[0],
        color="tab:green",
        marker="o",
        label="_nolegend_",
        zorder=3,
    )
    axes.scatter(
        x_position[-1],
        y_position[-1],
        color="tab:red",
        marker="X",
        label="_nolegend_",
        zorder=3,
    )
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)
    _style_trajectory_axes(axes, title)

    return HorizontalTrajectoryPlot(
        axes=axes,
        closure=HorizontalClosure(
            frame=frame,
            x=closure_x,
            y=closure_y,
            distance=closure_distance,
        ),
    )


def plot_llh_trajectory(
    positions: np.ndarray,
    *,
    basemap: BasemapStyle = BasemapStyle.SATELLITE,
    axes: Axes | None = None,
    trajectory_label: str = "Trajectory",
    title: str | None = None,
) -> LlhTrajectoryPlot:
    """Plot one WGS-84 LLH trajectory over a required tiled basemap.

    Args:
        positions: Finite ``(N, 3)`` ``[latitude, longitude, ellipsoidal height]``
            history in rad, rad, and m.
        basemap: Curated tiled background used only for visualization.
        axes: Existing axes, or ``None`` to create a figure and axes.
        trajectory_label: Legend label for the trajectory.
        title: Optional axes title.

    Returns:
        Axes with a Web-Mercator-projected path and geographic visible labels.

    Raises:
        ValueError: Inputs are invalid or outside Web Mercator latitude coverage.
        RuntimeError: The optional map dependency or requested tiles are unavailable.

    Notes:
        Web Mercator exists only to align with tiles; the input navigation state
        remains WGS-84 LLH. Use :func:`plot_llh_trajectories` for overlays so the
        combined paths set the initial tile extent. The map never establishes
        positional accuracy or replaces a scientific reference.
    """
    if not isinstance(basemap, BasemapStyle):
        raise ValueError("basemap must be a BasemapStyle value.")
    validated_positions = _validate_positions(positions)
    positions_mercator = _llh_to_web_mercator(validated_positions)

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots()

    _add_trajectory(axes, positions_mercator, trajectory_label)
    _set_map_extent(axes, (positions_mercator,))
    _ensure_basemap(axes, basemap)
    _style_llh_axes(axes, title)

    return LlhTrajectoryPlot(axes=axes, basemap=basemap)


def plot_llh_trajectories(
    trajectories: Sequence[tuple[np.ndarray, str]],
    *,
    basemap: BasemapStyle = BasemapStyle.SATELLITE,
    axes: Axes | None = None,
    title: str | None = None,
) -> LlhTrajectoryPlot:
    """Plot multiple WGS-84 LLH paths over one basemap covering every path.

    Args:
        trajectories: Nonempty ``(positions, label)`` pairs, with each position
            history ordered ``[latitude, longitude, ellipsoidal height]`` in
            rad, rad, and m.
        basemap: Curated tiled background used only for visualization.
        axes: Existing axes, or ``None`` to create a figure and axes.
        title: Optional axes title.

    Returns:
        Axes with all paths and the selected tiled basemap.

    Raises:
        ValueError: A trajectory, label, frame value, or coordinate is invalid.
        RuntimeError: The map dependency or requested tiles are unavailable.

    Notes:
        All paths are validated and display-projected before map retrieval so
        their union determines the extent. Callers retain responsibility for
        geodetic alignment and provenance; the renderer performs no registration.
    """
    if not isinstance(basemap, BasemapStyle):
        raise ValueError("basemap must be a BasemapStyle value.")
    if not trajectories:
        raise ValueError("trajectories must contain at least one LLH path.")

    projected_trajectories: list[tuple[np.ndarray, str]] = []
    for positions, trajectory_label in trajectories:
        if not isinstance(trajectory_label, str) or not trajectory_label:
            raise ValueError("Each trajectory label must be a nonempty string.")
        projected_trajectories.append(
            (_llh_to_web_mercator(_validate_positions(positions)), trajectory_label)
        )

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots()

    projected_paths = tuple(
        positions_mercator for positions_mercator, _label in projected_trajectories
    )
    _set_map_extent(axes, projected_paths)
    for positions_mercator, trajectory_label in projected_trajectories:
        _add_trajectory(axes, positions_mercator, trajectory_label)
    _ensure_basemap(axes, basemap)
    _style_llh_axes(axes, title)

    return LlhTrajectoryPlot(axes=axes, basemap=basemap)
