from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from formation import (
    GlobalPlanner,
    LocalPathWindow,
    MapBuilder,
    NarrowEntranceConfig,
    NarrowingCorridorConfig,
    ObstacleClusterConfig,
    PathManager,
    PreviewCurveConfig,
    PreviewCurvePlanner,
    RightAngleCorridorConfig,
    SCurveCorridorConfig,
)

SUPPORTED_MAP_TYPES = [
    "right_angle_corridor",
    "s_curve_corridor",
    "obstacle_cluster",
    "narrow_entrance",
    "narrowing_corridor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize occupancy, inflated map, distance field, and global A* path."
    )
    parser.add_argument(
        "--map-type",
        default="s_curve_corridor",
        choices=SUPPORTED_MAP_TYPES,
        help="Deterministic map type to visualize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the figure is shown interactively.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Overlay a single local Bezier preview curve near the start pose.",
    )
    parser.add_argument(
        "--preview-distance",
        type=float,
        default=3.5,
        help="Maximum forward observation distance used to cut the local preview window.",
    )
    parser.add_argument(
        "--hide-raw-path",
        action="store_true",
        help="Hide the raw A* grid path polyline.",
    )
    parser.add_argument(
        "--hide-waypoints",
        action="store_true",
        help="Hide simplified global waypoints and the polyline connecting them.",
    )
    parser.add_argument(
        "--show-local-window",
        action="store_true",
        help="Show the truncated local observation window used by preview planning.",
    )
    parser.add_argument(
        "--show-local-subgoal",
        action="store_true",
        help="Show the local subgoal at the preview-distance cutoff.",
    )
    parser.add_argument(
        "--overlay-preview-along-path",
        action="store_true",
        help="Overlay multiple local preview curves sampled continuously along the full global path.",
    )
    parser.add_argument(
        "--overlay-sample-spacing",
        type=float,
        default=0.75,
        help="Sample overlay preview reference points every N meters along the simplified global path.",
    )
    parser.add_argument(
        "--show-distance-field",
        action="store_true",
        help="Show the distance field panel. Hidden by default.",
    )
    return parser.parse_args()


def build_config(map_type: str):
    if map_type == "right_angle_corridor":
        return RightAngleCorridorConfig()
    if map_type == "s_curve_corridor":
        return SCurveCorridorConfig()
    if map_type == "obstacle_cluster":
        return ObstacleClusterConfig()
    if map_type == "narrow_entrance":
        return NarrowEntranceConfig()
    if map_type == "narrowing_corridor":
        return NarrowingCorridorConfig()
    raise ValueError(f"Unsupported map type: {map_type}")


def map_extent(map_data) -> list[float]:
    origin_x, origin_y = map_data.origin_xy
    return [origin_x, origin_x + map_data.width_m, origin_y, origin_y + map_data.height_m]


def plot_path(
    ax,
    path_xy,
    *,
    color: str,
    linewidth: float,
    label: str,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    if not path_xy:
        return
    xs = [point[0] for point in path_xy]
    ys = [point[1] for point in path_xy]
    ax.plot(xs, ys, color=color, linewidth=linewidth, label=label, linestyle=linestyle, alpha=alpha)


def sample_polyline(points_xy: list[tuple[float, float]], spacing_m: float) -> list[tuple[float, float]]:
    if len(points_xy) <= 1:
        return list(points_xy)

    arrays = np.asarray(points_xy, dtype=float)
    segment_vectors = arrays[1:] - arrays[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length < 1e-9:
        return [points_xy[0]]

    spacing = max(spacing_m, 1e-3)
    sample_positions = np.arange(0.0, total_length, spacing)
    if sample_positions.size == 0 or sample_positions[-1] < total_length:
        sample_positions = np.append(sample_positions, total_length)

    sampled: list[tuple[float, float]] = []
    segment_index = 0
    for arc_s in sample_positions:
        while segment_index < len(segment_lengths) - 1 and cumulative[segment_index + 1] < arc_s:
            segment_index += 1
        local_length = segment_lengths[segment_index]
        if local_length < 1e-9:
            point = arrays[segment_index]
        else:
            ratio = (arc_s - cumulative[segment_index]) / local_length
            point = arrays[segment_index] + ratio * segment_vectors[segment_index]
        sampled.append((float(point[0]), float(point[1])))
    return sampled


def collect_overlay_previews(
    path,
    map_data,
    preview_distance_m: float,
    sample_spacing_m: float,
):
    if len(path.waypoints_xy) < 2:
        return [], []

    reference_points = sample_polyline(path.waypoints_xy, sample_spacing_m)
    config = PreviewCurveConfig(preview_distance_m=preview_distance_m)
    path_manager = PathManager(path)
    planner = PreviewCurvePlanner(config)

    local_windows = []
    preview_curves = []
    for ref_xy in reference_points:
        local_window = path_manager.get_local_path_window_from_projection(
            ref_xy,
            preview_distance_m=preview_distance_m,
            min_points=config.min_window_points,
        )
        preview_curve = planner.plan(map_data, ref_xy, local_window)
        local_windows.append(local_window)
        preview_curves.append(preview_curve)
    return local_windows, preview_curves


def build_preview_at_start(path, map_data, preview_distance_m: float):
    config = PreviewCurveConfig(preview_distance_m=preview_distance_m)
    path_manager = PathManager(path)
    local_window = path_manager.get_local_path_window_from_projection(
        map_data.start_xy,
        preview_distance_m=preview_distance_m,
        min_points=config.min_window_points,
    )
    preview_curve = PreviewCurvePlanner(config).plan(map_data, map_data.start_xy, local_window)
    return local_window, preview_curve


def draw_world_panel(
    ax,
    map_data,
    path,
    *,
    preview_curve=None,
    local_window: LocalPathWindow | None = None,
    overlay_preview_curves=None,
    overlay_local_windows=None,
    show_raw_path: bool,
    show_waypoints: bool,
    show_local_window: bool,
    show_local_subgoal: bool,
) -> None:
    extent = map_extent(map_data)
    ax.imshow(
        map_data.occupancy.astype(float),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
        alpha=0.95,
    )
    if show_raw_path:
        plot_path(ax, path.raw_waypoints_xy, color="tab:blue", linewidth=1.2, label="A* raw path")
    if show_waypoints:
        plot_path(ax, path.waypoints_xy, color="tab:red", linewidth=2.0, label="Simplified waypoints")
        if path.waypoints_xy:
            xs = [point[0] for point in path.waypoints_xy]
            ys = [point[1] for point in path.waypoints_xy]
            ax.scatter(xs, ys, color="tab:red", s=16, zorder=3)
    if show_local_window and overlay_local_windows:
        for index, overlay_window in enumerate(overlay_local_windows):
            plot_path(
                ax,
                overlay_window.points_xy,
                color="tab:cyan",
                linewidth=1.0,
                label="Overlay local windows" if index == 0 else "_nolegend_",
                linestyle="--",
                alpha=0.30,
            )
    if show_local_window and local_window is not None:
        plot_path(
            ax,
            local_window.points_xy,
            color="tab:cyan",
            linewidth=1.5,
            label="Local observation window",
            linestyle="--",
        )
    if overlay_preview_curves:
        for index, overlay_curve in enumerate(overlay_preview_curves):
            if not overlay_curve.points_xy:
                continue
            color = "tab:orange" if overlay_curve.is_safe else "tab:pink"
            plot_path(
                ax,
                overlay_curve.points_xy,
                color=color,
                linewidth=1.5,
                label="Overlay preview curves" if index == 0 else "_nolegend_",
                alpha=0.60,
            )
    if preview_curve is not None and preview_curve.points_xy:
        plot_path(
            ax,
            preview_curve.points_xy,
            color="tab:orange",
            linewidth=2.2,
            label=f"Preview ({preview_curve.source_mode})",
        )
    if show_local_subgoal and overlay_preview_curves:
        overlay_subgoals = [curve.local_subgoal_xy for curve in overlay_preview_curves if curve.local_subgoal_xy is not None]
        if overlay_subgoals:
            ax.scatter(
                [point[0] for point in overlay_subgoals],
                [point[1] for point in overlay_subgoals],
                color="tab:brown",
                s=18,
                marker="x",
                label="Overlay local subgoals",
                zorder=4,
                alpha=0.70,
            )
    if show_local_subgoal and preview_curve is not None and preview_curve.local_subgoal_xy is not None:
        ax.scatter(
            *preview_curve.local_subgoal_xy,
            color="tab:brown",
            s=42,
            marker="x",
            label="Local subgoal",
            zorder=5,
        )
    if overlay_preview_curves:
        overlay_ref_points = [curve.points_xy[0] for curve in overlay_preview_curves if curve.points_xy]
        if overlay_ref_points:
            ax.scatter(
                [point[0] for point in overlay_ref_points],
                [point[1] for point in overlay_ref_points],
                color="tab:green",
                s=12,
                marker="o",
                label="Overlay ref points",
                zorder=4,
                alpha=0.60,
            )
        safe_count = sum(1 for curve in overlay_preview_curves if curve.is_safe)
        total_count = len(overlay_preview_curves)
        ax.text(
            0.02,
            0.02,
            f"overlay previews: {safe_count}/{total_count} safe",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    ax.scatter(*map_data.start_xy, color="tab:green", s=50, marker="o", label="Start", zorder=4)
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=50, marker="*", label="Goal", zorder=4)
    ax.set_title(f"{map_data.name}: occupancy + path")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)


def draw_grid_panel(ax, grid: np.ndarray, title: str, extent: list[float], cmap: str) -> None:
    ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")


def draw_distance_field_panel(ax, map_data, extent: list[float]) -> None:
    im = ax.imshow(
        map_data.distance_field,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_title("distance field [m]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_figure_axes(show_distance_field: bool):
    panel_count = 3 if show_distance_field else 2
    fig, axes = plt.subplots(1, panel_count, figsize=(6 * panel_count, 5), constrained_layout=True)
    if panel_count == 1:
        axes = [axes]
    return fig, axes


def build_figure_title(path, map_data, preview_curve, overlay_preview_curves) -> str:
    title = f"{map_data.name} | raw path points={len(path.grid_path_rc)} | waypoints={len(path.waypoints_xy)}"
    if preview_curve is not None:
        title += (
            f" | preview={preview_curve.source_mode}"
            f" | obs_dist={preview_curve.observation_distance_m:.2f}m"
            f" | end_dist={preview_curve.curve_end_distance_m:.2f}m"
            f" | safe={preview_curve.is_safe}"
        )
    if overlay_preview_curves:
        safe_count = sum(1 for curve in overlay_preview_curves if curve.is_safe)
        title += f" | overlay_safe={safe_count}/{len(overlay_preview_curves)}"
    return title


def save_or_show_figure(fig, output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {output}")
    else:
        plt.show()


def draw_visualization(
    map_data,
    path,
    *,
    preview_curve=None,
    local_window: LocalPathWindow | None = None,
    overlay_preview_curves=None,
    overlay_local_windows=None,
    show_raw_path: bool,
    show_waypoints: bool,
    show_local_window: bool,
    show_local_subgoal: bool,
    show_distance_field: bool,
    output: Path | None,
) -> None:
    extent = map_extent(map_data)
    fig, axes = create_figure_axes(show_distance_field)
    draw_world_panel(
        axes[0],
        map_data,
        path,
        preview_curve=preview_curve,
        local_window=local_window,
        overlay_preview_curves=overlay_preview_curves,
        overlay_local_windows=overlay_local_windows,
        show_raw_path=show_raw_path,
        show_waypoints=show_waypoints,
        show_local_window=show_local_window,
        show_local_subgoal=show_local_subgoal,
    )
    draw_grid_panel(
        axes[1],
        map_data.inflated_occupancy.astype(float),
        f"inflated occupancy (r={map_data.inflation_radius:.2f} m)",
        extent,
        "gray_r",
    )
    if show_distance_field:
        draw_distance_field_panel(axes[2], map_data, extent)
    fig.suptitle(build_figure_title(path, map_data, preview_curve, overlay_preview_curves))
    save_or_show_figure(fig, output)


def render_overlay_preview_along_path(
    map_data,
    path,
    *,
    preview_distance_m: float,
    sample_spacing_m: float,
    show_raw_path: bool,
    show_waypoints: bool,
    show_local_window: bool,
    show_local_subgoal: bool,
    show_distance_field: bool,
    output: Path | None,
) -> None:
    overlay_local_windows, overlay_preview_curves = collect_overlay_previews(
        path,
        map_data,
        preview_distance_m=preview_distance_m,
        sample_spacing_m=sample_spacing_m,
    )
    draw_visualization(
        map_data,
        path,
        overlay_preview_curves=overlay_preview_curves,
        overlay_local_windows=overlay_local_windows,
        show_raw_path=show_raw_path,
        show_waypoints=show_waypoints,
        show_local_window=show_local_window,
        show_local_subgoal=show_local_subgoal,
        show_distance_field=show_distance_field,
        output=output,
    )


def render_single_preview(
    map_data,
    path,
    *,
    preview_distance_m: float,
    show_raw_path: bool,
    show_waypoints: bool,
    show_local_window: bool,
    show_local_subgoal: bool,
    show_distance_field: bool,
    output: Path | None,
) -> None:
    local_window, preview_curve = build_preview_at_start(path, map_data, preview_distance_m)
    draw_visualization(
        map_data,
        path,
        preview_curve=preview_curve,
        local_window=local_window,
        show_raw_path=show_raw_path,
        show_waypoints=show_waypoints,
        show_local_window=show_local_window,
        show_local_subgoal=show_local_subgoal,
        show_distance_field=show_distance_field,
        output=output,
    )


def render_base_map(
    map_data,
    path,
    *,
    show_raw_path: bool,
    show_waypoints: bool,
    show_distance_field: bool,
    output: Path | None,
) -> None:
    draw_visualization(
        map_data,
        path,
        show_raw_path=show_raw_path,
        show_waypoints=show_waypoints,
        show_local_window=False,
        show_local_subgoal=False,
        show_distance_field=show_distance_field,
        output=output,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args.map_type)

    builder = MapBuilder()
    planner = GlobalPlanner()
    map_data = builder.build(args.map_type, config)
    path = planner.plan(map_data)

    if args.overlay_preview_along_path:
        render_overlay_preview_along_path(
            map_data,
            path,
            preview_distance_m=args.preview_distance,
            sample_spacing_m=args.overlay_sample_spacing,
            show_raw_path=not args.hide_raw_path,
            show_waypoints=not args.hide_waypoints,
            show_local_window=args.show_local_window,
            show_local_subgoal=args.show_local_subgoal,
            show_distance_field=args.show_distance_field,
            output=args.output,
        )
        return

    if args.show_preview or args.show_local_window or args.show_local_subgoal:
        render_single_preview(
            map_data,
            path,
            preview_distance_m=args.preview_distance,
            show_raw_path=not args.hide_raw_path,
            show_waypoints=not args.hide_waypoints,
            show_local_window=args.show_local_window,
            show_local_subgoal=args.show_local_subgoal,
            show_distance_field=args.show_distance_field,
            output=args.output,
        )
        return

    render_base_map(
        map_data,
        path,
        show_raw_path=not args.hide_raw_path,
        show_waypoints=not args.hide_waypoints,
        show_distance_field=args.show_distance_field,
        output=args.output,
    )


if __name__ == "__main__":
    main()
