from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os as _os
_ENV_LIB = _os.path.join(sys.prefix, "lib")
if _os.path.isdir(_ENV_LIB) and "CASADIPATH" not in _os.environ:
    _os.environ["CASADIPATH"] = _ENV_LIB

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from formation import (
    ControllerReferenceBuilder,
    DistributedFormationMPC,
    FormationLibrary,
    FormationSelector,
    GlobalPlanner,
    MapBuilder,
    MPCConfig,
    MultiRobotSimulator,
    NarrowEntranceConfig,
    NarrowingCorridorConfig,
    ObstacleClusterConfig,
    PathManager,
    PreviewCurvePlanner,
    RightAngleCorridorConfig,
    SCurveCorridorConfig,
    trace_summary,
)
from formation.mpc_controller import query_distance_field


SUPPORTED_MAP_TYPES = ["right_angle_corridor", "narrowing_corridor", "s_curve_corridor", "obstacle_cluster", "narrow_entrance"]
ROBOT_COLORS = ["tab:blue", "tab:green", "tab:brown", "tab:pink"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the guide-to-MPC rollout pipeline and visualize trajectories.")
    parser.add_argument("--map-type", choices=SUPPORTED_MAP_TYPES, default="right_angle_corridor")
    parser.add_argument("--preview-distance", type=float, default=3.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show-distance-field", action="store_true")
    parser.add_argument("--full-path", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=None)
    parser.add_argument("--max-replans", type=int, default=None)
    parser.add_argument("--animation-output", type=Path, default=None)
    parser.add_argument("--animation-fps", type=int, default=12)
    parser.add_argument("--animation-step", type=int, default=1)
    parser.add_argument("--replan-interval", type=int, default=10,
                        help="MPC steps between replanning (default: 10, ~2s at dt=0.2)")
    parser.add_argument("--preview-output", type=Path, default=None)
    parser.add_argument("--corridor-output", type=Path, default=None)
    return parser.parse_args()


def build_map_config(map_type: str, robot_radius: float):
    if map_type == "right_angle_corridor":
        return RightAngleCorridorConfig(robot_radius=robot_radius)
    if map_type == "narrowing_corridor":
        return NarrowingCorridorConfig(robot_radius=robot_radius)
    if map_type == "s_curve_corridor":
        return SCurveCorridorConfig(robot_radius=robot_radius)
    if map_type == "obstacle_cluster":
        return ObstacleClusterConfig(robot_radius=robot_radius)
    if map_type == "narrow_entrance":
        return NarrowEntranceConfig(robot_radius=robot_radius)
    raise ValueError(f"Unsupported map type: {map_type}")


def build_pipeline(
    map_type: str,
    preview_distance: float,
    *,
    full_path: bool,
    parallel_workers: int | None,
    goal_tolerance: float | None,
    max_replans: int | None,
    replan_interval: int,
):
    robot_radius = 0.09
    safety_margin = 0.06
    inter_robot_margin = 0.10
    config = build_map_config(map_type, robot_radius)
    builder = MapBuilder()
    map_data = builder.build(map_type, config)
    global_path = GlobalPlanner().plan(map_data)
    path_manager = PathManager(global_path)
    preview_planner = PreviewCurvePlanner()
    selector = FormationSelector()
    library = FormationLibrary.build_default(robot_radius=robot_radius, inter_robot_margin=inter_robot_margin)
    formations = library.list()

    mpc_config = MPCConfig(
        dt=0.2,
        horizon_steps=10,
        v_max=0.8,
        omega_max=1.2,
        robot_radius=robot_radius,
        safety_margin=safety_margin,
        inter_robot_margin=inter_robot_margin,
    )
    reference_builder = ControllerReferenceBuilder(mpc_config)
    controller = DistributedFormationMPC(mpc_config)
    simulator = MultiRobotSimulator(controller)

    if map_type == "narrowing_corridor":
        ref_x = 2.8
        ref_xy = (ref_x, config.centerline_slope * ref_x + config.centerline_intercept)
        window = path_manager.get_local_path_window_from_projection(ref_xy, preview_distance_m=preview_distance)
    else:
        ref_xy = map_data.start_xy
        window = path_manager.get_local_path_window(ref_xy, preview_distance_m=preview_distance)

    preview = preview_planner.plan(map_data, ref_xy, window)
    selection = selector.select_target_formation(
        map_data,
        preview,
        formations,
        robot_radius,
        safety_margin,
    )
    # Always start from square formation for consistent initial state
    square = library.get("square")
    square_eval = next((ev for ev in selection.evaluations if ev.formation_name == "square"), None)
    if square_eval is None:
        square_eval = selector.evaluate_candidate_formation(
            map_data, preview, selection.curve_band, square, robot_radius, safety_margin,
        )
        selection.evaluations.append(square_eval)
    square_guide = selector.guide_generator.build(square_eval, square)
    controller_reference = reference_builder.build(square_guide)
    initial_states = simulator.initial_states_from_reference(controller_reference)
    selection.selected_formation = square
    selection.selected_evaluation = square_eval
    selection.guide = square_guide

    if full_path:
        trace = simulator.simulate_full_path(
            initial_states,
            map_data,
            global_path,
            formations,
            preview_planner,
            selector,
            reference_builder,
            robot_radius,
            safety_margin,
            preview_distance,
            current_formation=selection.selected_formation,
            goal_tolerance=goal_tolerance,
            max_replans=max_replans,
            replan_interval=replan_interval,
        )
    else:
        trace = simulator.simulate(initial_states, controller_reference, map_data)

    return {
        "map_data": map_data,
        "global_path": global_path,
        "preview": preview,
        "selection": selection,
        "controller_reference": controller_reference,
        "trace": trace,
        "mpc_config": mpc_config,
        "full_path": full_path,
        "formations": formations,
    }


def map_extent(map_data) -> list[float]:
    origin_x, origin_y = map_data.origin_xy
    return [origin_x, origin_x + map_data.width_m, origin_y, origin_y + map_data.height_m]


def draw_rollout(context: dict, *, show_distance_field: bool, output: Path | None) -> None:
    map_data = context["map_data"]
    trace = context["trace"]
    mpc_config = context["mpc_config"]
    full_path = context["full_path"]
    summary = trace_summary(trace)

    panel_count = 2 if show_distance_field else 1
    fig, axes = plt.subplots(1, panel_count, figsize=(7 * panel_count, 6), constrained_layout=True)
    if panel_count == 1:
        axes = [axes]

    extent = map_extent(map_data)
    ax = axes[0]
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent, cmap="gray_r", interpolation="nearest", alpha=0.95)

    initial_states = trace.state_history[0] if trace.state_history else []
    for robot_index in range(len(initial_states)):
        color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
        rollout_points = [(state[robot_index].x, state[robot_index].y) for state in trace.state_history]
        xs = [point[0] for point in rollout_points]
        ys = [point[1] for point in rollout_points]
        ax.plot(xs, ys, color=color, linewidth=2.2, label=f"robot {robot_index} trajectory")
        ax.scatter(initial_states[robot_index].x, initial_states[robot_index].y, color=color, s=28, marker="o", label=f"robot {robot_index} start", zorder=4)

    ax.scatter(*map_data.goal_xy, color="tab:purple", s=70, marker="*", label="goal", zorder=4)
    title = (
        f"{map_data.name}\ntrack_max={trace.max_tracking_error:.3f}m | "
        f"clear_min={trace.min_obstacle_clearance:.3f}m | "
        f"pair_min={trace.min_pairwise_distance:.3f}m"
    )
    if full_path:
        title += f"\ngoal={summary['goal_distance']:.3f}m | cycles={summary['replanning_cycles']} | stop={summary['stop_reason']}"
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    if show_distance_field:
        im = axes[1].imshow(
            map_data.distance_field,
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        axes[1].set_title("distance field [m]")
        axes[1].set_xlabel("x [m]")
        axes[1].set_ylabel("y [m]")
        axes[1].set_aspect("equal")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    mode = "full path" if full_path else "single segment"
    fig.suptitle(f"MPC rollout ({mode}) | dt={mpc_config.dt:.2f}s | horizon={mpc_config.horizon_steps} | steps={trace.step_count}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved rollout figure to {output}")
        plt.close(fig)
    else:
        plt.show()


def save_rollout_animation(context: dict, output: Path, *, fps: int, frame_step: int) -> None:
    map_data = context["map_data"]
    trace = context["trace"]
    summary = trace_summary(trace)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    extent = map_extent(map_data)
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent, cmap="gray_r", interpolation="nearest", alpha=0.95)
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=70, marker="*", label="goal", zorder=4)

    initial_states = trace.state_history[0] if trace.state_history else []
    trajectory_lines = []
    current_markers = []
    for robot_index in range(len(initial_states)):
        color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
        ax.scatter(initial_states[robot_index].x, initial_states[robot_index].y, color=color, s=28, marker="o", label=f"robot {robot_index} start", zorder=4)
        trajectory_line, = ax.plot([], [], color=color, linewidth=2.2, label=f"robot {robot_index} trajectory")
        current_marker, = ax.plot([], [], color=color, marker="o", markersize=6)
        trajectory_lines.append(trajectory_line)
        current_markers.append(current_marker)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    frames = list(range(0, len(trace.state_history), max(1, frame_step)))
    if not frames or frames[-1] != len(trace.state_history) - 1:
        frames.append(len(trace.state_history) - 1)

    def update(frame_index: int):
        state_index = frames[frame_index]
        for robot_index in range(len(initial_states)):
            rollout_points = [(state[robot_index].x, state[robot_index].y) for state in trace.state_history[: state_index + 1]]
            trajectory_lines[robot_index].set_data(
                [point[0] for point in rollout_points],
                [point[1] for point in rollout_points],
            )
            current_point = rollout_points[-1]
            current_markers[robot_index].set_data([current_point[0]], [current_point[1]])
        ax.set_title(
            f"step={state_index}/{len(trace.state_history) - 1} | "
            f"goal={summary['goal_distance']:.3f}m | "
            f"clear_min={trace.min_obstacle_clearance:.3f}m | "
            f"pair_min={trace.min_pairwise_distance:.3f}m"
        )
        return [*trajectory_lines, *current_markers]

    animation = FuncAnimation(fig, update, frames=len(frames), interval=max(1, int(1000 / max(1, fps))), blit=False)
    suffix = output.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    elif suffix in (".mp4", ".mov", ".avi", ".mkv"):
        writer = FFMpegWriter(fps=fps)
    else:
        output = output.with_suffix(".mp4")
        writer = FFMpegWriter(fps=fps)
    animation.save(output, writer=writer, dpi=140)
    plt.close(fig)
    print(f"Saved rollout animation to {output}")


def print_metrics(context: dict) -> None:
    map_data = context["map_data"]
    selection = context["selection"]
    trace = context["trace"]
    summary = trace_summary(trace)
    final_states = trace.final_states
    final_clearances = [query_distance_field(map_data, (state.x, state.y)) for state in final_states]

    formations_by_width = sorted(context["formations"], key=lambda f: f.lateral_half_width, reverse=True)
    print("=== Formation Selection ===")
    print(f"preference order (wider > narrower): {[f'{f.name}({f.lateral_half_width:.2f}m)' for f in formations_by_width]}")
    print(f"evaluated formations: {[ev.formation_name for ev in selection.evaluations]}")
    for ev in selection.evaluations:
        status = "SAFE" if ev.is_safe else ("BAND_OK" if ev.band_feasible else "INFEA")
        print(f"  {ev.formation_name:20s} {status:8s}  margin={ev.score_breakdown.min_corridor_margin_m:+.4f}m  "
              f"embed_cost={ev.score_breakdown.embedding_cost:.4f}  "
              f"clearance={ev.min_slot_clearance_m:.4f}m")
    print(f"=> selected: {selection.selected_formation.name}")

    print(f"\n=== Tracking ===")
    print(f"guide samples: {len(selection.guide.guide_samples)}")
    print(f"max tracking error: {trace.max_tracking_error:.4f} m")
    print(f"min obstacle clearance: {trace.min_obstacle_clearance:.4f} m")
    print(f"min pairwise distance: {trace.min_pairwise_distance:.4f} m")
    print(f"final clearances: {[round(value, 4) for value in final_clearances]}")
    if context["full_path"]:
        print(f"\n=== Full Path ===")
        print(f"reached goal: {summary['reached_goal']}")
        print(f"goal distance: {summary['goal_distance']:.4f} m")
        print(f"replanning cycles: {summary['replanning_cycles']}")
        print(f"stop reason: {summary['stop_reason']}")
        print(f"formation sequence: {summary['selected_formations']}")
        print(f"plan wall time: {summary['plan_wall_time_s']:.4f} s")
        print(f"solve wall time: {summary['solve_wall_time_s']:.4f} s")


def draw_preview_curve(context: dict, output: Path) -> None:
    """Overlay all replan cycles' preview curves on a single map."""
    map_data = context["map_data"]
    global_path = context["global_path"]
    trace = context["trace"]
    meta = trace.metadata
    preview_points_list = meta.get("per_cycle_preview_points", [])
    ref_history = meta.get("preview_ref_history", [])
    selected = meta.get("selected_formations", [])

    if not preview_points_list:
        preview_points_list = [context["preview"].points_xy]
        ref_history = [map_data.start_xy]
        selected = [context["selection"].selected_formation.name]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    extent = map_extent(map_data)
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent, cmap="gray_r", interpolation="nearest", alpha=0.95)
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=80, marker="*", label="goal", zorder=5)
    ax.scatter(*map_data.start_xy, color="tab:green", s=60, marker="o", label="start", zorder=5)

    waypoints = global_path.waypoints_xy
    ax.plot([p[0] for p in waypoints], [p[1] for p in waypoints], "k--", linewidth=1.0, alpha=0.4, label="global path")

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(preview_points_list)))
    for i, (pts, ref_xy, sel) in enumerate(zip(preview_points_list, ref_history, selected)):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors[i], linewidth=2.0,
                label=f"cycle {i+1} [{sel}]")
        ax.scatter(*ref_xy, color=colors[i], s=40, marker="s", zorder=4)

    ax.set_title(f"Preview Curves ({len(preview_points_list)} replan cycles)\n{map_data.name}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview curves to {output}")


def draw_corridor(context: dict, output: Path) -> None:
    """Overlay all replan cycles' chord + strip cells on a single map."""
    map_data = context["map_data"]
    trace = context["trace"]
    meta = trace.metadata
    strip_cells_list = meta.get("per_cycle_strip_cells", [])
    chord_centers_list = meta.get("per_cycle_chord_centers", [])
    chord_endpoints_list = meta.get("per_cycle_chord_endpoints", [])
    selected = meta.get("selected_formations", [])

    if not strip_cells_list:
        cb = context["selection"].curve_band
        strip_cells_list = [[cell.vertices_xy for cell in cb.strip_cells]]
        chord_centers_list = [[s.center_xy for s in cb.samples]]
        chord_endpoints_list = [[(s.left_xy, s.right_xy) for s in cb.samples]]
        selected = [context["selection"].selected_formation.name]

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    extent = map_extent(map_data)
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent, cmap="gray_r", interpolation="nearest", alpha=0.95)
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=80, marker="*", label="goal", zorder=5)

    from matplotlib.patches import Polygon
    colors = plt.cm.tab10(np.linspace(0, 1, len(strip_cells_list)))
    for i, (cells, centers, ends, sel) in enumerate(zip(
        strip_cells_list, chord_centers_list, chord_endpoints_list, selected)):
        color = colors[i]
        # Strip cells
        for cell_verts in cells:
            ax.add_patch(Polygon(list(cell_verts), closed=True, facecolor=color,
                                 edgecolor=color, alpha=0.12, linewidth=0.5))
        # Chord lines
        for j, (left, right) in enumerate(ends):
            if j % 3 == 0:
                ax.plot([left[0], right[0]], [left[1], right[1]], color=color, linewidth=1.2, alpha=0.6)
        # Centerline
        cx = [c[0] for c in centers]; cy = [c[1] for c in centers]
        ax.plot(cx, cy, color=color, linewidth=2.5, alpha=0.9, label=f"cycle {i+1} [{sel}]")

    ax.set_title(f"Safe Corridors: Chords + Strip Cells ({len(strip_cells_list)} cycles)\n{map_data.name}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved corridors to {output}")


def main() -> None:
    args = parse_args()
    context = build_pipeline(
        args.map_type,
        args.preview_distance,
        full_path=args.full_path,
        parallel_workers=args.parallel_workers,
        goal_tolerance=args.goal_tolerance,
        max_replans=args.max_replans,
        replan_interval=args.replan_interval,
    )
    print_metrics(context)
    if args.output is not None or (args.animation_output is None and args.preview_output is None and args.corridor_output is None):
        draw_rollout(context, show_distance_field=args.show_distance_field, output=args.output)
    if args.preview_output is not None:
        draw_preview_curve(context, args.preview_output)
    if args.corridor_output is not None:
        draw_corridor(context, args.corridor_output)
    if args.animation_output is not None:
        save_rollout_animation(
            context,
            args.animation_output,
            fps=max(1, args.animation_fps),
            frame_step=max(1, args.animation_step),
        )


if __name__ == "__main__":
    main()
