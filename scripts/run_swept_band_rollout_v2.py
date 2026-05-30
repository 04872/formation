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
from matplotlib.colors import ListedColormap

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
from formation.swept_band import SweptBandBuilder
from formation.types import LocalPreviewPath

SUPPORTED_MAP_TYPES = [
    "right_angle_corridor", "narrowing_corridor", "s_curve_corridor",
    "obstacle_cluster", "narrow_entrance",
]
ROBOT_COLORS = ["tab:blue", "tab:green", "tab:brown", "tab:pink"]
BAND_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#8B5A00",  # brown
    "#332288",  # indigo
]
BAND_LINESTYLES = ["-", "--", "-.", ":"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Formation rollout with SweptBand visualisation")
    p.add_argument("map", nargs="?", choices=SUPPORTED_MAP_TYPES, default="right_angle_corridor",
                   help="Map type (positional, e.g. right_angle_corridor)")
    p.add_argument("--map-type", choices=SUPPORTED_MAP_TYPES, default=None,
                   help="Map type (flag alternative)")
    p.add_argument("--preview-distance", type=float, default=3.5)
    p.add_argument("--goal-tolerance", type=float, default=None)
    p.add_argument("--max-replans", type=int, default=None)
    p.add_argument("--replan-interval", type=int, default=10)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("results"),
                   help="Output directory (default: results/)")
    return p.parse_args()


def _map_type(args) -> str:
    return args.map_type or args.map


def _out(args, stem: str, ext: str) -> Path:
    mt = _map_type(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args.out_dir / f"{stem}_{mt}.{ext}"


# ── pipeline (shared with old script) ─────────────────────────────

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
    map_type: str, preview_distance: float, *,
    goal_tolerance: float | None, max_replans: int | None, replan_interval: int,
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
        dt=0.2, horizon_steps=10, v_max=0.8, omega_max=1.2,
        robot_radius=robot_radius, safety_margin=safety_margin,
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
    selection = selector.select_target_formation(map_data, preview, formations, robot_radius, safety_margin)

    # Start from square for consistent initial state
    square = library.get("square")
    square_eval = next((ev for ev in selection.evaluations if ev.formation_name == "square"), None)
    if square_eval is None:
        square_eval = selector.evaluate_candidate_formation(
            map_data, preview, selection.curve_band, square, robot_radius, safety_margin)
        selection.evaluations.append(square_eval)
    square_guide = selector.guide_generator.build(square_eval, square)
    controller_reference = reference_builder.build(square_guide)
    initial_states = simulator.initial_states_from_reference(controller_reference)
    selection.selected_formation = square
    selection.selected_evaluation = square_eval
    selection.guide = square_guide

    # Inject swept‑band feasibility
    from formation.formation_feasibility_v2 import FormationFeasibilityV2
    from formation.formation_feasibility import FeasibilityConfig
    simulator._feasibility = FormationFeasibilityV2(FeasibilityConfig(mode="swept_band_v2"))
    trace = simulator.simulate_full_path(
        initial_states, map_data, global_path, formations,
        preview_planner, selector, reference_builder,
        robot_radius, safety_margin, preview_distance,
        current_formation=selection.selected_formation,
        goal_tolerance=goal_tolerance, max_replans=max_replans,
        replan_interval=replan_interval,
    )
    return {
        "map_data": map_data, "global_path": global_path,
        "preview": preview, "selection": selection,
        "trace": trace, "mpc_config": mpc_config,
        "formations": formations,
    }


def map_extent(map_data) -> list[float]:
    ox, oy = map_data.origin_xy
    return [ox, ox + map_data.width_m, oy, oy + map_data.height_m]


# ── trajectory plot (unchanged) ───────────────────────────────────

def draw_trajectory(context: dict, output: Path):
    map_data = context["map_data"]
    trace = context["trace"]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    extent = map_extent(map_data)
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent,
              cmap="gray_r", interpolation="nearest", alpha=0.95)
    initial_states = trace.state_history[0] if trace.state_history else []
    for ri in range(len(initial_states)):
        color = ROBOT_COLORS[ri % len(ROBOT_COLORS)]
        pts = [(s[ri].x, s[ri].y) for s in trace.state_history]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2.2, label=f"robot {ri}")
        ax.scatter(initial_states[ri].x, initial_states[ri].y, color=color, s=28, marker="o", zorder=4)
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=70, marker="*", label="goal", zorder=4)
    s = trace_summary(trace)
    ax.set_title(f"Trajectories — {map_data.name}\ngoal={s['goal_distance']:.2f}m cycles={s['replanning_cycles']}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  traj → {output}")


# ── preview curves (unchanged) ────────────────────────────────────

def draw_preview(context: dict, output: Path):
    map_data = context["map_data"]
    global_path = context["global_path"]
    trace = context["trace"]
    meta = trace.metadata
    pts_list = meta.get("per_cycle_preview_points", [])
    refs = meta.get("preview_ref_history", [])
    sels = meta.get("selected_formations", [])
    if not pts_list:
        pts_list = [context["preview"].points_xy]
        refs = [map_data.start_xy]
        sels = [context["selection"].selected_formation.name]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    extent = map_extent(map_data)
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent,
              cmap="gray_r", interpolation="nearest", alpha=0.95)
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=80, marker="*", label="goal", zorder=5)
    ax.scatter(*map_data.start_xy, color="tab:green", s=60, marker="o", label="start", zorder=5)
    wps = global_path.waypoints_xy
    ax.plot([p[0] for p in wps], [p[1] for p in wps], "k--", linewidth=1.0, alpha=0.4, label="global path")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(pts_list)))
    for i, (pts, ref, sel) in enumerate(zip(pts_list, refs, sels)):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors[i], linewidth=2.0,
                label=f"cycle {i+1} [{sel}]")
        ax.scatter(*ref, color=colors[i], s=40, marker="s", zorder=4)
    # Draw centreline (swept‑band optimised or controller reference)
    cl_list = trace.metadata.get("per_cycle_centerline", [])
    if cl_list and any(cl_list):
        for i, ct_pts in enumerate(cl_list):
            if ct_pts:
                ax.plot([p[0] for p in ct_pts], [p[1] for p in ct_pts],
                        linestyle="--", color=colors[i], linewidth=1.5, alpha=0.8)
    else:
        refs_list = [ref.center_points_xy for ref in trace.reference_history]
        for i, ct_pts in enumerate(refs_list):
            if ct_pts:
                ax.plot([p[0] for p in ct_pts], [p[1] for p in ct_pts],
                        linestyle=":", color=colors[i], linewidth=1.0, alpha=0.7)

    ax.set_title(f"Preview Curves + Centreline ({len(pts_list)} cycles) — {map_data.name}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  preview → {output}")


# ── feasibility centerline plot ──────────────────────────────────

def draw_feasibility_centerline(context: dict, output: Path):
    """Draw the per-cycle centerline used when judging formation feasibility."""
    map_data = context["map_data"]
    trace = context["trace"]
    meta = trace.metadata
    centerline_list = meta.get("per_cycle_centerline", [])
    preview_list = meta.get("per_cycle_preview_points", [])
    selected = meta.get("selected_formations", [])

    if not centerline_list:
        centerline_list = [context["preview"].points_xy]
        preview_list = [context["preview"].points_xy]
        selected = [context["selection"].selected_formation.name]

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    extent = map_extent(map_data)
    ax.imshow(
        map_data.occupancy.astype(float),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
        alpha=0.95,
    )
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=80, marker="*", label="goal", zorder=5)
    ax.scatter(*map_data.start_xy, color="tab:green", s=60, marker="o", label="start", zorder=5)

    colors = plt.cm.tab20(np.linspace(0.05, 0.95, max(1, len(centerline_list))))
    for i, cl_pts in enumerate(centerline_list):
        if not cl_pts:
            continue
        color = colors[i % len(colors)]
        ax.plot(
            [p[0] for p in cl_pts],
            [p[1] for p in cl_pts],
            color=color,
            linewidth=2.6,
            linestyle="-",
            label=f"cycle {i+1} centerline [{selected[i] if i < len(selected) else ''}]",
            zorder=4,
        )
        # lightly show the preview curve that fed the feasibility check
        if i < len(preview_list) and preview_list[i]:
            ax.plot(
                [p[0] for p in preview_list[i]],
                [p[1] for p in preview_list[i]],
                color=color,
                linewidth=1.0,
                linestyle=":",
                alpha=0.55,
                zorder=3,
            )
            ax.scatter(
                preview_list[i][0][0],
                preview_list[i][0][1],
                color=color,
                s=18,
                marker="s",
                alpha=0.8,
                zorder=4,
            )

    ax.set_title(f"Feasibility Centerline per Cycle — {map_data.name}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  centerline → {output}")


# ── SweptBand visualisation (NEW) ─────────────────────────────────

def _make_temp_preview_path(pts_xy, map_data) -> LocalPreviewPath:
    """Build a minimal LocalPreviewPath from a list of points."""
    arc = [0.0]
    for k in range(1, len(pts_xy)):
        arc.append(arc[-1] + np.hypot(pts_xy[k][0] - pts_xy[k-1][0],
                                       pts_xy[k][1] - pts_xy[k-1][1]))
    tangents = []
    for k in range(len(pts_xy)):
        if k == 0:
            d = (pts_xy[1][0] - pts_xy[0][0], pts_xy[1][1] - pts_xy[0][1])
        elif k == len(pts_xy) - 1:
            d = (pts_xy[-1][0] - pts_xy[-2][0], pts_xy[-1][1] - pts_xy[-2][1])
        else:
            d = (pts_xy[k+1][0] - pts_xy[k-1][0], pts_xy[k+1][1] - pts_xy[k-1][1])
        dn = np.hypot(*d) or 1e-9
        tangents.append((d[0]/dn, d[1]/dn))
    norms = [(-t[1], t[0]) for t in tangents]
    cl = [query_distance_field(map_data, p) for p in pts_xy]
    return LocalPreviewPath(
        points_xy=list(pts_xy), arc_lengths=arc,
        tangents_xy=tangents, normals_xy=norms,
        curvatures=[0.0] * len(pts_xy),
        source_mode="keypoint_opt", is_safe=True,
        min_clearance=min(cl) if cl else 0.0,
        local_subgoal_xy=pts_xy[-1] if pts_xy else (0.0, 0.0),
    )


def draw_swept_band(context: dict, output: Path):
    map_data = context["map_data"]
    trace = context["trace"]
    meta = trace.metadata
    pts_list = meta.get("per_cycle_preview_points", [])
    sels = meta.get("selected_formations", [])
    if not pts_list:
        pts_list = [context["preview"].points_xy]
        sels = [context["selection"].selected_formation.name]

    band_builder = SweptBandBuilder()
    clearance = 0.15

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    extent = map_extent(map_data)
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent,
              cmap="gray_r", interpolation="nearest", alpha=0.95)
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=80, marker="*", label="goal", zorder=5)

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, min(len(pts_list), 10))))
    ox, oy = map_data.origin_xy
    res = map_data.resolution
    for i, (pts, sel) in enumerate(zip(pts_list, sels)):
        color = colors[i % len(colors)]
        tmp = _make_temp_preview_path(pts, map_data)
        # build band for every cycle and draw with contrasting color/linestyle
        band_idx = i % len(BAND_COLORS)
        band = band_builder.build(map_data, tmp, clearance)
        band_mask = np.where(band._grid > 0.0, 1.0, np.nan)
        band_cmap = ListedColormap([BAND_COLORS[band_idx]])
        ax.imshow(band_mask, origin="lower",
                  extent=[ox, ox + band._w * res, oy, oy + band._h * res],
                  cmap=band_cmap, alpha=0.18, vmin=0, vmax=1,
                  interpolation="nearest", zorder=1)
        xs = np.linspace(ox + 0.5 * res, ox + (band._w - 0.5) * res, band._w)
        ys = np.linspace(oy + 0.5 * res, oy + (band._h - 0.5) * res, band._h)
        ax.contour(
            xs,
            ys,
            band._grid,
            levels=[0.0],
            colors=[BAND_COLORS[band_idx]],
            linewidths=1.6,
            linestyles=[BAND_LINESTYLES[band_idx % len(BAND_LINESTYLES)]],
            zorder=2,
        )
        # analytic offset polylines (L/R + caps) using band color, slightly darker
        pts_arr = np.asarray(tmp.points_xy)
        norms_arr = np.asarray(tmp.normals_xy)
        from formation.swept_band import SweptBandBuilder as _SB
        _ct = 0.09 + 0.06
        _step = 0.05; _ms = 60
        _rm = np.zeros(len(pts_arr)); _rp = np.zeros(len(pts_arr))
        _sb = _SB()
        for _j in range(len(pts_arr)):
            _rm[_j] = _sb._raycast(map_data, pts_arr[_j], -norms_arr[_j], _ct, _step, _ms)
            _rp[_j] = _sb._raycast(map_data, pts_arr[_j], norms_arr[_j], _ct, _step, _ms)
        _L = np.column_stack([pts_arr[:,0] - _rm * norms_arr[:,0],
                              pts_arr[:,1] - _rm * norms_arr[:,1]])
        _R = np.column_stack([pts_arr[:,0] + _rp * norms_arr[:,0],
                              pts_arr[:,1] + _rp * norms_arr[:,1]])
        ax.plot(_L[:,0], _L[:,1], color=BAND_COLORS[band_idx], linewidth=1.4, zorder=3, alpha=0.95)
        ax.plot(_R[:,0], _R[:,1], color=BAND_COLORS[band_idx], linewidth=1.4, zorder=3, alpha=0.95)
        ax.plot([_L[0,0],_R[0,0]], [_L[0,1],_R[0,1]], color=BAND_COLORS[band_idx], linestyle=':', linewidth=1.0, zorder=3, alpha=0.7)
        ax.plot([_L[-1,0],_R[-1,0]], [_L[-1,1],_R[-1,1]], color=BAND_COLORS[band_idx], linestyle=':', linewidth=1.0, zorder=3, alpha=0.7)
        # Centreline
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=color, linewidth=2.2, alpha=0.95,
                linestyle=BAND_LINESTYLES[i % len(BAND_LINESTYLES)],
                label=f"cycle {i+1} [{sel}]")

    ax.set_title(f"Swept Band — cycles ({1}-{len(pts_list)}) — {map_data.name}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  band → {output}")


# ── animation (unchanged) ─────────────────────────────────────────

def save_animation(context: dict, output: Path, *, fps: int):
    map_data = context["map_data"]
    trace = context["trace"]
    summary = trace_summary(trace)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    extent = map_extent(map_data)
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent,
              cmap="gray_r", interpolation="nearest", alpha=0.95)
    ax.scatter(*map_data.goal_xy, color="tab:purple", s=70, marker="*", label="goal", zorder=4)
    initial_states = trace.state_history[0] if trace.state_history else []
    tlines, cmarkers = [], []
    for ri in range(len(initial_states)):
        c = ROBOT_COLORS[ri % len(ROBOT_COLORS)]
        ax.scatter(initial_states[ri].x, initial_states[ri].y, color=c, s=28, marker="o", label=f"robot {ri}", zorder=4)
        tl, = ax.plot([], [], color=c, linewidth=2.2)
        cm, = ax.plot([], [], color=c, marker="o", markersize=6)
        tlines.append(tl); cmarkers.append(cm)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    frames = list(range(0, len(trace.state_history), max(1, 2)))
    if not frames or frames[-1] != len(trace.state_history) - 1:
        frames.append(len(trace.state_history) - 1)

    def update(fi):
        si = frames[fi]
        for ri in range(len(initial_states)):
            pts = [(s[ri].x, s[ri].y) for s in trace.state_history[:si + 1]]
            tlines[ri].set_data([p[0] for p in pts], [p[1] for p in pts])
            cmarkers[ri].set_data([pts[-1][0]], [pts[-1][1]])
        ax.set_title(f"step {si}/{len(trace.state_history)-1} | goal={summary['goal_distance']:.2f}m")
        return [*tlines, *cmarkers]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=max(1, int(1000 / max(1, fps))), blit=False)
    suffix = output.suffix.lower()
    if suffix == ".gif":
        w = PillowWriter(fps=fps)
    else:
        output = output.with_suffix(".mp4")
        w = FFMpegWriter(fps=fps)
    anim.save(output, writer=w, dpi=140)
    plt.close(fig)
    print(f"  mp4 → {output}")


# ── main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    mt = _map_type(args)
    print(f"=== SweptBand rollout: {mt} ===")
    ctx = build_pipeline(
        mt, args.preview_distance,
        goal_tolerance=args.goal_tolerance,
        max_replans=args.max_replans,
        replan_interval=args.replan_interval,
    )
    s = trace_summary(ctx["trace"])
    print(f"goal={s['goal_distance']:.2f}m cycles={s['replanning_cycles']} "
          f"form={s['selected_formations']}")

    draw_trajectory(ctx, _out(args, "traj", "png"))
    draw_preview(ctx, _out(args, "preview", "png"))
    draw_feasibility_centerline(ctx, _out(args, "centerline", "png"))
    draw_swept_band(ctx, _out(args, "band", "png"))
    if not args.no_mp4:
        save_animation(ctx, _out(args, "rollout", "mp4"), fps=max(1, args.fps))


if __name__ == "__main__":
    main()
