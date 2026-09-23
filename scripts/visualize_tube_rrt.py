from __future__ import annotations

import time

_MODULE_START = time.perf_counter()

import argparse
import csv
import io
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from formation import (
    FormationLibrary,
    MapBuilder,
    Pose2D,
    PostFenceConfig,
    RandomCirclesConfig,
    SinglePostConfig,
    TubeRRTConfig,
    TubeRRTPlanner,
    TubeRRTResult,
    project_robot_paths,
    transform_slots,
)
from formation.tube_rrt import interpolate_pose
from formation.types import wrap_to_pi

TREE_CMAP = colormaps["viridis"]
RHO_CMAP = colormaps["plasma"]
PATH_COLOR = "#d62728"
STRADDLE_COLOR = "#e7298a"
ROBOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
MAP_NAMES = ("post_fence", "random_circles", "single_post")
TURTLEBOT3_BURGER_RADIUS = 0.113  # circumscribed circle of the 138 mm x 178 mm TurtleBot3 Burger footprint
DEFAULT_SAFETY_MARGIN = 0.06
FIGURE_FILES = {
    "overview": ("overview.png", "总览：搜索树、joint tube、编队投影、tube 宽度四宫格"),
    "tree": ("1_tree.png", "最终搜索树 (x, y) 投影，颜色 = 节点插入顺序，短线 = yaw；叠加被拒绝的 steer 点、"
                           "rewire 移除的边，粉色圈 = 障碍夹在机器人之间的节点"),
    "growth": ("2_growth.png", "由 trace 回放的树生长快照：sample、nearest、q_new 及其安全球 xy 截面"),
    "tube": ("3_tube.png", "joint tube：安全球 xy 截面、SE(2) 双锥、rho 沿路径曲线、相邻球严格重叠检查"),
    "formation": ("4_formation.png", "沿路径等间距采样的编队姿态：每个姿态一种颜色并编号，轮廓线连接各机器人"
                                     "（粉色填充 = 障碍夹在机器人之间）；机器人轨迹、瓶颈处 r + margin + rho 圆，"
                                     "以及 theta 曲线"),
    "frames": ("5_formation_frames.png", "关键帧放大：障碍夹在机器人之间（或瓶颈）附近的连续 tube 节点；每个子图以"
                                         "“上一节点 + 当前节点”为中心、比例尺相同，虚线为上一节点姿态，黑箭头为中心位移"),
    "convergence": ("6_convergence.png", "最优路径代价随迭代下降曲线，以及节点 / 拒绝 / rewire 累计数"),
}


def convex_hull(points: np.ndarray, keep_collinear: bool = False) -> np.ndarray:
    pts = sorted(map(tuple, np.round(points, 12)))
    if len(pts) < 3:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def turns_wrong(o, a, b):
        value = cross(o, a, b)
        return value < -1e-12 if keep_collinear else value <= 0

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and turns_wrong(lower[-2], lower[-1], p):
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and turns_wrong(upper[-2], upper[-1], p):
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def gate_half_widths(slots: np.ndarray) -> list[float]:
    """Half distances between robots that are neighbours on the formation boundary (the gaps an obstacle enters by)."""
    centered = slots - slots.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered)
    if singular[1] <= 1e-9 * max(singular[0], 1.0):
        ring = slots[np.argsort(centered @ vt[0])]
        pairs = zip(ring, ring[1:])
    else:
        ring = convex_hull(slots, keep_collinear=True)
        pairs = zip(ring, np.roll(ring, -1, axis=0))
    return [float(np.linalg.norm(b - a)) / 2.0 for a, b in pairs]


def pass_through_radius(slots: np.ndarray, robot_radius: float, safety_margin: float) -> float:
    """Largest circular obstacle radius that fits between two neighbouring robots with positive clearance."""
    return max(gate_half_widths(slots)) - robot_radius - safety_margin


def parse_obstacle_radius(text: str) -> str:
    if text == "auto":
        return text
    try:
        values = [float(v) for v in text.split(":")]
    except ValueError:
        raise argparse.ArgumentTypeError("expected 'auto', a radius R or a range MIN:MAX") from None
    if len(values) not in (1, 2) or min(values) <= 0.0 or values[0] > values[-1]:
        raise argparse.ArgumentTypeError("radius must be positive and MIN <= MAX")
    return text


def build_map_config(args, bound: float):
    """Map config from CLI; 'auto' obstacle size is chosen relative to the pass-through bound of the formation."""
    base = {"robot_radius": args.robot_radius, "safety_margin": args.safety_margin}
    spec = args.obstacle_radius
    if spec == "auto":
        if bound <= 0.0:
            raise SystemExit(f"formation gaps are too narrow for any obstacle to pass between robots "
                             f"(bound {bound:.3f} m); increase --slot-scale or reduce --robot-radius/--safety-margin")
        radii = (0.3 * bound, 0.8 * bound) if args.map == "random_circles" else (0.5 * bound, 0.5 * bound)
    elif spec is not None:
        values = [float(v) for v in spec.split(":")]
        radii = (values[0], values[-1])
    else:
        radii = None
    if args.map == "random_circles":
        config = RandomCirclesConfig(seed=args.seed, **base)
        if radii is not None:
            config.radius_min, config.radius_max = radii
        if args.obstacle_count is not None:
            config.obstacle_count = args.obstacle_count
        return config
    if args.obstacle_count is not None:
        raise SystemExit("--obstacle-count only applies to random_circles")
    if radii is not None and radii[0] != radii[1]:
        raise SystemExit(f"{args.map} uses a single obstacle radius; pass R instead of MIN:MAX")
    config = PostFenceConfig(**base) if args.map == "post_fence" else SinglePostConfig(**base)
    if radii is not None:
        config.post_radius = radii[0]
    return config


def straddled_obstacles(pose: Pose2D, slots: np.ndarray, map_data) -> list[int]:
    """Indices of obstacles whose center lies strictly inside the convex hull of the robot centers."""
    hull = convex_hull(transform_slots(pose, slots))
    if len(hull) < 3:
        return []
    inside = []
    for index, primitive in enumerate(map_data.obstacle_primitives):
        c = np.asarray(primitive["center_xy"], dtype=float)
        edges = np.roll(hull, -1, axis=0) - hull
        rel = c - hull
        if np.all(edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0] > 1e-12):
            inside.append(index)
    return inside


class Tee(io.TextIOBase):
    def __init__(self, stream) -> None:
        self.stream = stream
        self.buffer_text = io.StringIO()

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.buffer_text.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()

    def getvalue(self) -> str:
        return self.buffer_text.getvalue()


def draw_map(axis, map_data, *, show_goal_region: float | None = None, labels: bool = True) -> None:
    ox, oy = map_data.origin_xy
    axis.set_xlim(ox, ox + map_data.width_m)
    axis.set_ylim(oy, oy + map_data.height_m)
    axis.set_aspect("equal")
    axis.set_facecolor("#fafafa")
    axis.grid(True, color="0.9", linewidth=0.6, zorder=0)
    for primitive in map_data.obstacle_primitives:
        axis.add_patch(Circle(primitive["center_xy"], primitive["radius"], facecolor="0.25", edgecolor="0.1", zorder=2))
    if show_goal_region is not None:
        axis.add_patch(Circle(map_data.goal_xy, show_goal_region, fill=False, linestyle=":", edgecolor="tab:orange",
                              linewidth=1.2, zorder=3))
    axis.scatter(*map_data.start_xy, marker="s", s=70, color="tab:green", edgecolor="k", zorder=9,
                 label="start" if labels else None)
    axis.scatter(*map_data.goal_xy, marker="*", s=180, color="tab:orange", edgecolor="k", zorder=9,
                 label="goal" if labels else None)


def draw_path(axis, poses: list[Pose2D], *, label: str | None = "joint path (center)", nodes: bool = True) -> None:
    if not poses:
        return
    xs, ys = [p.x for p in poses], [p.y for p in poses]
    axis.plot(xs, ys, color="white", linewidth=5.0, solid_capstyle="round", zorder=6)
    axis.plot(xs, ys, color=PATH_COLOR, linewidth=2.4, solid_capstyle="round", zorder=7, label=label)
    if nodes:
        axis.scatter(xs, ys, s=16, color=PATH_COLOR, edgecolor="white", linewidth=0.6, zorder=8)


def heading_segments(poses: list[Pose2D], length: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [((p.x, p.y), (p.x + length * math.cos(p.yaw), p.y + length * math.sin(p.yaw))) for p in poses]


def tree_style(node_count: int) -> dict[str, float]:
    if node_count <= 150:
        return {"node": 22, "edge": 1.4, "tick": 0.18, "tick_alpha": 1.0, "reject_alpha": 0.5}
    if node_count <= 600:
        return {"node": 9, "edge": 1.0, "tick": 0.12, "tick_alpha": 0.6, "reject_alpha": 0.35}
    return {"node": 5, "edge": 0.8, "tick": 0.09, "tick_alpha": 0.45, "reject_alpha": 0.25}


def draw_tree(axis, result: TubeRRTResult, map_data, goal_connect: float, slots: np.ndarray) -> None:
    """Final search tree in the (x, y) projection, colored by node insertion order."""
    draw_map(axis, map_data, show_goal_region=goal_connect)
    nodes = result.tree_nodes
    norm = Normalize(0, max(1, len(nodes) - 1))
    style = tree_style(len(nodes))

    collisions = [e.steered for e in result.trace if e.status == "collision"]
    no_overlap = [e.steered for e in result.trace if e.status == "no_overlap"]
    if collisions:
        axis.scatter([p.x for p in collisions], [p.y for p in collisions], marker="x", s=12, color="#e41a1c",
                     alpha=style["reject_alpha"], linewidth=0.7, zorder=3, label=f"rejected: collision ({len(collisions)})")
    if no_overlap:
        axis.scatter([p.x for p in no_overlap], [p.y for p in no_overlap], marker="o", s=10, facecolor="none",
                     edgecolor="#ff7f00", alpha=style["reject_alpha"] + 0.1, linewidth=0.7, zorder=3,
                     label=f"rejected: no tube overlap ({len(no_overlap)})")

    rewired = [(nodes[old].pose, nodes[child].pose) for e in result.trace for child, old, _ in e.rewires]
    if rewired and len(rewired) <= 300:
        axis.add_collection(LineCollection([((a.x, a.y), (b.x, b.y)) for a, b in rewired], colors="0.6",
                                           linestyles="--", linewidths=0.6, alpha=0.6, zorder=3))
        axis.plot([], [], color="0.6", linestyle="--", linewidth=0.8, label=f"edge removed by rewire ({len(rewired)})")
    elif rewired:
        axis.plot([], [], color="none", label=f"rewires: {len(rewired)} (not drawn)")

    segments, colors = [], []
    for index, node in enumerate(nodes):
        if node.parent is None:
            continue
        parent = nodes[node.parent]
        segments.append(((parent.pose.x, parent.pose.y), (node.pose.x, node.pose.y)))
        colors.append(TREE_CMAP(norm(index)))
    axis.add_collection(LineCollection(segments, colors=colors, linewidths=style["edge"], zorder=4))
    show_ticks = len(nodes) <= 600
    if show_ticks:
        axis.add_collection(LineCollection(heading_segments([n.pose for n in nodes], style["tick"]), colors="0.15",
                                           linewidths=0.7, alpha=style["tick_alpha"], zorder=5))
    scatter = axis.scatter([n.pose.x for n in nodes], [n.pose.y for n in nodes], c=np.arange(len(nodes)),
                           cmap=TREE_CMAP, norm=norm, s=style["node"], edgecolor="k", linewidth=0.3, zorder=5)
    axis.plot([], [], color=TREE_CMAP(0.6), linewidth=1.4, marker="o", markersize=4, markeredgecolor="k",
              label=f"tree ({len(nodes)} nodes{', tick = yaw' if show_ticks else ''})")
    straddling = [n.pose for n in nodes if straddled_obstacles(n.pose, slots, map_data)]
    if straddling:
        axis.scatter([p.x for p in straddling], [p.y for p in straddling], s=60, facecolor="none",
                     edgecolor=STRADDLE_COLOR, linewidth=1.3, zorder=6,
                     label=f"obstacle between robots ({len(straddling)} nodes)")
    draw_path(axis, result.path_poses, nodes=False)
    colorbar = axis.figure.colorbar(scatter, ax=axis, fraction=0.03, pad=0.01)
    colorbar.set_label("node insertion order")


def replay_parents(result: TubeRRTResult, upto_event: int) -> dict[int, int | None]:
    parents: dict[int, int | None] = {0: None}
    for event in result.trace[:upto_event + 1]:
        if event.node_index is not None:
            parents[event.node_index] = event.parent
        for child, _, new_parent in event.rewires:
            parents[child] = new_parent
    return parents


def replay_path(result: TubeRRTResult, parents: dict[int, int | None], goal_index: int) -> list[Pose2D]:
    indices, index = [], goal_index
    while index is not None:
        indices.append(index)
        index = parents[index]
    return [result.tree_nodes[i].pose for i in reversed(indices)]


def draw_growth_snapshot(axis, result: TubeRRTResult, map_data, event_index: int, goal_connect: float,
                         final: bool) -> None:
    draw_map(axis, map_data, show_goal_region=goal_connect, labels=False)
    nodes = result.tree_nodes
    parents = replay_parents(result, event_index)
    norm = Normalize(0, max(1, len(nodes) - 1))
    style = tree_style(len(parents))
    rejected = [e.steered for e in result.trace[:event_index + 1] if e.status in ("collision", "no_overlap")]
    if rejected:
        axis.scatter([p.x for p in rejected], [p.y for p in rejected], marker="x", s=8, color="#e41a1c",
                     alpha=style["reject_alpha"] * 0.7, linewidth=0.6, zorder=3)
    edges = [(c, p) for c, p in parents.items() if p is not None]
    axis.add_collection(LineCollection([((nodes[p].pose.x, nodes[p].pose.y), (nodes[c].pose.x, nodes[c].pose.y))
                                        for c, p in edges],
                                       colors=[TREE_CMAP(norm(c)) for c, _ in edges],
                                       linewidths=style["edge"] * 0.9, zorder=4))
    indices = sorted(parents)
    axis.scatter([nodes[i].pose.x for i in indices], [nodes[i].pose.y for i in indices], c=indices, cmap=TREE_CMAP,
                 norm=norm, s=style["node"] * 0.7, edgecolor="k", linewidth=0.25, zorder=5)

    event = result.trace[event_index]
    if final:
        draw_path(axis, result.path_poses, label=None, nodes=False)
    elif event.status == "goal":
        draw_path(axis, replay_path(result, parents, event.node_index), label=None, nodes=False)
    near = nodes[event.nearest].pose
    axis.scatter(event.sample.x, event.sample.y, marker="P", s=70, color="tab:red", edgecolor="k", zorder=8)
    axis.plot([near.x, event.sample.x], [near.y, event.sample.y], color="tab:red", linestyle=":", linewidth=1.0, zorder=7)
    axis.scatter(near.x, near.y, s=70, facecolor="none", edgecolor="tab:red", linewidth=1.5, zorder=8)
    if event.radius > 0.0:
        axis.add_patch(Circle((event.steered.x, event.steered.y), event.radius, facecolor="tab:cyan", alpha=0.25,
                              edgecolor="tab:blue", linewidth=1.0, zorder=6))
    axis.scatter(event.steered.x, event.steered.y, s=30, color="tab:blue", edgecolor="k", zorder=9)
    tag = "first goal" if event.iteration == result.first_goal_iteration and event.status == "goal" else event.status
    if final:
        tag = f"final, best path cost {result.path_cost:.2f}"
    extra = f", rewires={len(event.rewires)}" if event.rewires else ""
    axis.set_title(f"iteration {event.iteration}: {len(indices)} nodes, {tag}{extra}", fontsize=10)


def growth_picks(result: TubeRRTResult, count: int) -> list[int]:
    grow_events = [i for i, e in enumerate(result.trace) if e.status in ("added", "goal")]
    first_goal = next((i for i, e in enumerate(result.trace) if e.status == "goal"), None)
    if first_goal is None or first_goal == grow_events[-1]:
        fractions = np.linspace(0.05, 1.0, count)
        return sorted({grow_events[round(f * (len(grow_events) - 1))] for f in fractions})
    before = [i for i in grow_events if i < first_goal]
    after = [i for i in grow_events if i > first_goal]
    picks = {before[round(f * (len(before) - 1))] for f in np.linspace(0.1, 0.8, count - 3)} if before else set()
    picks |= {first_goal, after[len(after) // 3], grow_events[-1]}
    return sorted(picks)


def draw_growth(figure, result: TubeRRTResult, map_data, goal_connect: float) -> None:
    axes = figure.subplots(2, 3, sharex=True, sharey=True)
    if not any(e.status in ("added", "goal") for e in result.trace):
        for axis in axes.flat:
            draw_map(axis, map_data, labels=False)
        return
    picks = growth_picks(result, axes.size)
    for axis, event_index in zip(axes.flat, picks):
        draw_growth_snapshot(axis, result, map_data, event_index, goal_connect,
                             final=result.success and event_index == picks[-1])
    for axis in list(axes.flat)[len(picks):]:
        axis.set_visible(False)
    handles = [
        Line2D([], [], marker="P", color="tab:red", markeredgecolor="k", linestyle="", markersize=9, label="sample q_rand"),
        Line2D([], [], marker="o", color="none", markeredgecolor="tab:red", linestyle="", markersize=9, label="nearest node"),
        Line2D([], [], marker="o", color="tab:blue", markeredgecolor="k", linestyle="", markersize=6, label="steered q_new"),
        Line2D([], [], marker="o", color="tab:cyan", alpha=0.5, linestyle="", markersize=12,
               label="xy-slice of new safety ball (radius rho)"),
        Line2D([], [], marker="x", color="#e41a1c", linestyle="", markersize=6, label="rejected so far"),
        Line2D([], [], color=PATH_COLOR, linewidth=2.4, label="current best path"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)


def draw_tube_xy(axis, result: TubeRRTResult, map_data, rho_norm: Normalize) -> None:
    draw_map(axis, map_data)
    for pose, radius in zip(result.path_poses, result.path_radii):
        axis.add_patch(Circle((pose.x, pose.y), radius, facecolor=RHO_CMAP(rho_norm(radius)), alpha=0.35,
                              edgecolor=RHO_CMAP(rho_norm(radius)), linewidth=1.2, zorder=4))
    draw_path(axis, result.path_poses)
    if result.path_radii:
        k = int(np.argmin(result.path_radii))
        pose = result.path_poses[k]
        axis.annotate(f"bottleneck rho={result.path_radii[k]:.3f}", (pose.x, pose.y), xytext=(0, 55),
                      textcoords="offset points", ha="center", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6"),
                      arrowprops=dict(arrowstyle="->", color="k"), zorder=10)
    axis.set_title("joint tube, xy slice: disk = {c : d_G((c, theta_k), q_k) < rho_k}", fontsize=10)


def draw_tube_3d(axis, result: TubeRRTResult, formation_radius: float, rho_norm: Normalize) -> None:
    if not result.path_poses:
        return
    yaws = np.unwrap([p.yaw for p in result.path_poses])
    phi = np.linspace(0.0, 2.0 * math.pi, 28)
    for pose, yaw, radius in zip(result.path_poses, yaws, result.path_radii):
        height = radius / formation_radius
        dtheta = np.linspace(-height, height, 15)
        ring = radius - formation_radius * np.abs(dtheta)
        xs = pose.x + np.outer(ring, np.cos(phi))
        ys = pose.y + np.outer(ring, np.sin(phi))
        zs = np.degrees(yaw + np.repeat(dtheta[:, None], phi.size, axis=1))
        axis.plot_surface(xs, ys, zs, color=RHO_CMAP(rho_norm(radius)), alpha=0.2, linewidth=0, shade=True)
    tree = result.tree_nodes
    axis.scatter([n.pose.x for n in tree], [n.pose.y for n in tree], np.degrees([n.pose.yaw for n in tree]),
                 s=3, color="0.5", alpha=0.35, depthshade=False)
    axis.plot([p.x for p in result.path_poses], [p.y for p in result.path_poses], np.degrees(yaws),
              color=PATH_COLOR, linewidth=2.0, marker="o", markersize=3)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("theta [deg]")
    axis.set_box_aspect((3, 2, 1.8), zoom=1.15)
    axis.view_init(elev=22, azim=-58)
    axis.set_title("safety balls in SE(2): bicones\n||dc|| + R_F |dtheta| < rho", fontsize=10)


def path_arclength(poses: list[Pose2D], metric) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum([metric(a, b) for a, b in zip(poses, poses[1:])])])


def draw_rho_profile(axis, result: TubeRRTResult, metric, rho_norm: Normalize) -> None:
    if not result.path_poses:
        return
    s = path_arclength(result.path_poses, metric)
    radii = np.asarray(result.path_radii)
    axis.fill_between(s, 0.0, radii, color=RHO_CMAP(0.55), alpha=0.25)
    axis.plot(s, radii, color="k", linewidth=1.0)
    axis.scatter(s, radii, c=radii, cmap=RHO_CMAP, norm=rho_norm, s=30, edgecolor="k", zorder=5)
    axis.axhline(result.bottleneck, color=PATH_COLOR, linestyle="--", linewidth=1.2,
                 label=f"bottleneck = {result.bottleneck:.3f}")
    axis.set_xlabel("joint-path length along d_G")
    axis.set_ylabel("rho_k (safety radius)")
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="0.9")
    axis.legend(fontsize=8, loc="upper right")
    axis.set_title("tube width along the path", fontsize=10)


def draw_overlap_check(axis, result: TubeRRTResult, metric) -> None:
    poses, radii = result.path_poses, result.path_radii
    if len(poses) < 2:
        return
    edges = np.arange(len(poses) - 1)
    distance = np.asarray([metric(a, b) for a, b in zip(poses, poses[1:])])
    allowed = np.asarray(radii[:-1]) + np.asarray(radii[1:])
    axis.bar(edges, allowed, color="tab:cyan", alpha=0.45, label="rho_k + rho_{k+1}")
    axis.bar(edges, distance, width=0.45, color="tab:blue", label="d_G(q_k, q_{k+1})")
    axis.set_xlabel("path edge k")
    axis.set_ylabel("metric distance")
    axis.set_xticks(edges)
    axis.tick_params(axis="x", labelsize=7)
    axis.grid(True, axis="y", color="0.9")
    axis.legend(fontsize=8, loc="upper right")
    axis.set_title(f"strict ball overlap d_G < rho_k + rho_(k+1): min margin {np.min(allowed - distance):.3f}",
                   fontsize=10)


def dense_path(poses: list[Pose2D], per_edge: int = 12) -> list[Pose2D]:
    if len(poses) < 2:
        return list(poses)
    dense = [poses[0]]
    for a, b in zip(poses, poses[1:]):
        dense += [interpolate_pose(a, b, t) for t in np.linspace(0.0, 1.0, per_edge + 1)[1:]]
    return dense


def sample_poses_evenly(poses: list[Pose2D], count: int, formation_radius: float) -> list[Pose2D]:
    """Poses at equal spacing of translation + R_F * rotation along the geodesic joint path."""
    if len(poses) < 2:
        return list(poses)
    lengths = [math.hypot(b.x - a.x, b.y - a.y) + formation_radius * abs(wrap_to_pi(b.yaw - a.yaw))
               for a, b in zip(poses, poses[1:])]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = []
    for target in np.linspace(0.0, cumulative[-1], count):
        k = min(int(np.searchsorted(cumulative, target, side="right")) - 1, len(lengths) - 1)
        alpha = 0.0 if lengths[k] == 0.0 else (target - cumulative[k]) / lengths[k]
        samples.append(interpolate_pose(poses[k], poses[k + 1], alpha))
    return samples


def draw_robot_trajectories(axis, poses: list[Pose2D], slots: np.ndarray, *, alpha: float = 0.9,
                            labels: bool = True) -> None:
    for index, robot_path in enumerate(project_robot_paths(dense_path(poses), slots)):
        axis.plot([p[0] for p in robot_path], [p[1] for p in robot_path], color=ROBOT_COLORS[index % len(ROBOT_COLORS)],
                  linewidth=1.3, alpha=alpha, zorder=4, label=f"robot {index}" if labels else None)


def draw_formation_pose(axis, pose: Pose2D, slots: np.ndarray, map_data, robot_radius: float, color, *,
                        label: str | None = None, ghost: bool = False, zorder: int = 6) -> None:
    """One formation pose: outline through the robots, robot disks, heading arrow and an optional number."""
    points = transform_slots(pose, slots)
    hull = convex_hull(points)
    straddle = bool(straddled_obstacles(pose, slots, map_data))
    if ghost:
        if len(hull) >= 3:
            axis.add_patch(Polygon(hull, closed=True, fill=False, edgecolor="0.45", linestyle="--", linewidth=1.2,
                                   zorder=zorder))
        for x, y in points:
            axis.add_patch(Circle((x, y), robot_radius, facecolor="0.8", edgecolor="0.45", linewidth=0.6, zorder=zorder))
        return
    if len(hull) >= 3:
        axis.add_patch(Polygon(hull, closed=True, facecolor=STRADDLE_COLOR if straddle else color,
                               alpha=0.28 if straddle else 0.10, edgecolor="none", zorder=zorder))
        axis.add_patch(Polygon(hull, closed=True, fill=False, edgecolor=color, linewidth=2.0, zorder=zorder + 1))
    elif len(points) >= 2:
        axis.plot(points[:, 0], points[:, 1], color=color, linewidth=2.0, zorder=zorder + 1)
    for index, (x, y) in enumerate(points):
        axis.add_patch(Circle((x, y), robot_radius, facecolor=ROBOT_COLORS[index % len(ROBOT_COLORS)], edgecolor=color,
                              linewidth=1.2, zorder=zorder + 2))
    heading = 0.6 * max(float(np.max(np.linalg.norm(slots, axis=1))), 0.2)
    axis.annotate("", xy=(pose.x + heading * math.cos(pose.yaw), pose.y + heading * math.sin(pose.yaw)),
                  xytext=(pose.x, pose.y), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4), zorder=zorder + 3)
    if label is not None:
        axis.text(pose.x, pose.y, label, fontsize=8, fontweight="bold", ha="center", va="center", color="k",
                  bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor=color, linewidth=1.2),
                  zorder=zorder + 4)


def path_view_limits(poses: list[Pose2D], slots: np.ndarray, map_data, pad: float) -> tuple[tuple, tuple]:
    points = np.vstack([transform_slots(p, slots) for p in dense_path(poses, 4)])
    ox, oy = map_data.origin_xy
    low = np.maximum(points.min(axis=0) - pad, (ox, oy))
    high = np.minimum(points.max(axis=0) + pad, (ox + map_data.width_m, oy + map_data.height_m))
    return (low[0], high[0]), (low[1], high[1])


def draw_formation(axis, result: TubeRRTResult, map_data, slots: np.ndarray, planner: TubeRRTPlanner,
                   count: int = 10) -> None:
    draw_map(axis, map_data)
    poses = result.path_poses
    if not poses:
        return
    draw_robot_trajectories(axis, poses, slots)
    axis.plot([p.x for p in poses], [p.y for p in poses], color="k", linestyle=":", linewidth=1.0, zorder=4,
              label="center (tube nodes)")
    axis.scatter([p.x for p in poses], [p.y for p in poses], s=8, color="k", zorder=4)
    samples = sample_poses_evenly(poses, count, planner.formation_radius)
    cmap = colormaps["turbo"]
    for k, pose in enumerate(samples):
        draw_formation_pose(axis, pose, slots, map_data, planner.robot_radius, cmap(0.08 + 0.84 * k / max(1, count - 1)),
                            label=str(k + 1), zorder=6 + k * 6)
    axis.fill([], [], color=STRADDLE_COLOR, alpha=0.3, label="obstacle between robots")
    k = int(np.argmin(result.path_radii))
    reach = planner.robot_radius + planner.safety_margin + result.path_radii[k]
    for x, y in transform_slots(poses[k], slots):
        axis.add_patch(Circle((x, y), reach, fill=False, edgecolor=PATH_COLOR, linestyle="--", linewidth=1.0, zorder=200))
    axis.plot([], [], color=PATH_COLOR, linestyle="--", label="bottleneck: r + margin + rho")
    xlim, ylim = path_view_limits(poses, slots, map_data, pad=0.8)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_title(f"{count} formation poses evenly spaced along the joint path (numbered in order, outline = formation)",
                   fontsize=10)


def key_frame_indices(result: TubeRRTResult, slots: np.ndarray, map_data, count: int) -> list[int]:
    poses = result.path_poses
    straddle = [k for k, p in enumerate(poses) if straddled_obstacles(p, slots, map_data)]
    if straddle:
        low, high = min(straddle) - 1, max(straddle) + 1
    else:
        center = int(np.argmin(result.path_radii))
        low, high = center - count // 2, center + count // 2
    while high - low + 1 < count and (low > 0 or high < len(poses) - 1):
        low, high = low - 1, high + 1
    low, high = max(0, low), min(len(poses) - 1, high)
    return sorted({int(round(v)) for v in np.linspace(low, high, min(count, high - low + 1))})


def draw_formation_frames(figure, result: TubeRRTResult, map_data, slots: np.ndarray, planner: TubeRRTPlanner,
                          count: int = 6) -> None:
    poses = result.path_poses
    axes = figure.subplots(2, 3).flat
    if not poses:
        return
    frames = key_frame_indices(result, slots, map_data, count)
    boxes = []
    for k in frames:
        points = np.vstack([transform_slots(poses[j], slots) for j in (max(0, k - 1), k)])
        boxes.append((points.min(axis=0), points.max(axis=0)))
    half = max(float(np.max(high - low)) / 2.0 for low, high in boxes) + planner.robot_radius + 0.35
    for axis, k, (low, high) in zip(axes, frames, boxes):
        center = (low + high) / 2.0
        xlim, ylim = (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)
        draw_map(axis, map_data, labels=False)
        for index in straddled_obstacles(poses[k], slots, map_data):
            primitive = map_data.obstacle_primitives[index]
            axis.add_patch(Circle(primitive["center_xy"], primitive["radius"], facecolor="0.25",
                                  edgecolor=STRADDLE_COLOR, linewidth=2.5, zorder=3))
        draw_robot_trajectories(axis, poses, slots, alpha=0.35, labels=False)
        if k > 0:
            draw_formation_pose(axis, poses[k - 1], slots, map_data, planner.robot_radius, None, ghost=True, zorder=5)
            axis.annotate("", xy=(poses[k].x, poses[k].y), xytext=(poses[k - 1].x, poses[k - 1].y),
                          arrowprops=dict(arrowstyle="->", color="k", lw=1.2), zorder=20)
        draw_formation_pose(axis, poses[k], slots, map_data, planner.robot_radius, "tab:blue", zorder=10)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        move = (f", step d_G={planner.metric(poses[k - 1], poses[k]):.2f}" if k > 0 else "")
        axis.set_title(f"node {k}/{len(poses) - 1}: theta={math.degrees(poses[k].yaw):.1f} deg, "
                       f"rho={result.path_radii[k]:.3f}{move}", fontsize=9)
    for axis in list(axes)[len(frames):]:
        axis.set_visible(False)
    handles = [
        Line2D([], [], color="tab:blue", linewidth=2.0, label="current pose (outline through robots)"),
        Line2D([], [], color="0.45", linestyle="--", linewidth=1.2, label="previous tube node"),
        Line2D([], [], marker="o", color="0.25", markeredgecolor=STRADDLE_COLOR, markeredgewidth=2.5, linestyle="",
               markersize=10, label="obstacle between robots"),
        Line2D([], [], color="0.5", alpha=0.6, linewidth=1.3, label="robot trajectories"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)


def draw_yaw_profile(axis, result: TubeRRTResult) -> None:
    poses = result.path_poses
    if not poses:
        return
    s = np.concatenate([[0.0], np.cumsum([math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(poses, poses[1:])])])
    axis.plot(s, np.degrees(np.unwrap([p.yaw for p in poses])), color="k", marker="o", markersize=3)
    axis.set_xlabel("center arc length [m]")
    axis.set_ylabel("theta [deg]")
    axis.grid(True, color="0.9")
    axis.set_title("formation heading along the joint path", fontsize=10)


def search_counters(result: TubeRRTResult) -> dict[str, np.ndarray]:
    iterations = np.arange(1, result.iterations + 1)
    counts = {key: np.zeros(result.iterations, dtype=int) for key in ("added", "collision", "no_overlap", "rewires")}
    for event in result.trace:
        if event.status == "goal":
            continue
        counts[event.status][event.iteration - 1] += 1
        counts["rewires"][event.iteration - 1] += len(event.rewires)
    return {"iteration": iterations, **{key: np.cumsum(value) for key, value in counts.items()}}


def draw_convergence(figure, result: TubeRRTResult) -> None:
    cost_axis, count_axis = figure.subplots(1, 2)
    if result.cost_history:
        steps = [(it, cost) for it, cost in result.cost_history] + [(result.iterations, result.cost_history[-1][1])]
        cost_axis.step([s[0] for s in steps], [s[1] for s in steps], where="post", color=PATH_COLOR, linewidth=1.8)
        cost_axis.scatter([s[0] for s in result.cost_history], [s[1] for s in result.cost_history], s=14,
                          color=PATH_COLOR, zorder=5)
        first_iteration, first_cost = result.cost_history[0]
        cost_axis.axvline(first_iteration, color="0.4", linestyle=":", linewidth=1.0)
        cost_axis.annotate(f"first goal\nit {first_iteration}, cost {first_cost:.2f}", (first_iteration, first_cost),
                           xytext=(10, -5), textcoords="offset points", fontsize=9, va="top")
        cost_axis.annotate(f"final cost {result.path_cost:.2f}", (result.iterations, result.path_cost),
                           xytext=(-5, 10), textcoords="offset points", fontsize=9, ha="right")
        cost_axis.set_xlim(0, max(result.iterations, 1))
    else:
        cost_axis.text(0.5, 0.5, "no goal connection", transform=cost_axis.transAxes, ha="center")
    cost_axis.set_xlabel("iteration")
    cost_axis.set_ylabel("best goal cost (d_G path length)")
    cost_axis.grid(True, color="0.9")
    cost_axis.set_title("anytime improvement of the joint path", fontsize=10)

    counters = search_counters(result)
    for key, color, label in (("added", "tab:green", "accepted nodes"), ("collision", "#e41a1c", "rejected: collision"),
                              ("no_overlap", "#ff7f00", "rejected: no tube overlap"), ("rewires", "0.4", "rewires")):
        count_axis.plot(counters["iteration"], counters[key], color=color, linewidth=1.5, label=label)
    if result.first_goal_iteration is not None:
        count_axis.axvline(result.first_goal_iteration, color="0.4", linestyle=":", linewidth=1.0)
    count_axis.set_xlabel("iteration")
    count_axis.set_ylabel("cumulative count")
    count_axis.grid(True, color="0.9")
    count_axis.legend(fontsize=8, loc="upper left")
    count_axis.set_title("search statistics", fontsize=10)


def summary_line(result: TubeRRTResult) -> str:
    status = "success" if result.success else result.failure_reason
    cost = f", path cost {result.path_cost:.2f}" if result.success else ""
    return (f"Joint Tube-RRT ({status}): {result.iterations} iterations, {len(result.tree_nodes)} nodes, "
            f"{len(result.path_poses)} path nodes{cost}, bottleneck rho = {result.bottleneck:.3f}")


def build_figures(result: TubeRRTResult, map_data, slots: np.ndarray, planner: TubeRRTPlanner) -> dict[str, plt.Figure]:
    goal_connect = planner.config.goal_connect_distance
    rho_norm = Normalize(0.0, max(result.path_radii or [1.0]))
    summary = summary_line(result)
    figures: dict[str, plt.Figure] = {}

    overview = plt.figure(figsize=(16, 10.5))
    grid = overview.add_gridspec(2, 2)
    axis = overview.add_subplot(grid[0, 0])
    draw_tree(axis, result, map_data, goal_connect, slots)
    axis.set_title("1. search tree (color = insertion order)", fontsize=11)
    axis = overview.add_subplot(grid[0, 1])
    draw_tube_xy(axis, result, map_data, rho_norm)
    axis.set_title("2. joint tube (xy slice of each safety ball)", fontsize=11)
    axis = overview.add_subplot(grid[1, 0])
    draw_formation(axis, result, map_data, slots, planner)
    axis.set_title("3. per-robot projection of the joint path", fontsize=11)
    axis = overview.add_subplot(grid[1, 1])
    draw_rho_profile(axis, result, planner.metric, rho_norm)
    axis.set_title("4. tube width rho_k along the path", fontsize=11)
    for axis in overview.axes:
        if axis.get_label() != "<colorbar>" and axis.get_legend_handles_labels()[0] and axis.get_legend() is None:
            axis.legend(fontsize=7, loc="upper left", framealpha=0.85)
    overview.suptitle(summary, fontsize=12)
    figures["overview"] = overview

    tree = plt.figure(figsize=(13, 8.2))
    axis = tree.add_subplot()
    draw_tree(axis, result, map_data, goal_connect, slots)
    axis.legend(fontsize=8, loc="upper left", framealpha=0.9)
    axis.set_title(summary, fontsize=11)
    figures["tree"] = tree

    growth = plt.figure(figsize=(16, 8.6))
    draw_growth(growth, result, map_data, goal_connect)
    growth.suptitle("tree growth: sample -> nearest -> steer -> safety ball -> overlap check -> choose parent / rewire",
                    fontsize=12)
    figures["growth"] = growth

    tube = plt.figure(figsize=(16, 10))
    grid = tube.add_gridspec(2, 2, height_ratios=(1.5, 1), width_ratios=(1.1, 1))
    axis = tube.add_subplot(grid[0, 0])
    draw_tube_xy(axis, result, map_data, rho_norm)
    axis.legend(fontsize=8, loc="upper left")
    draw_tube_3d(tube.add_subplot(grid[0, 1], projection="3d"), result, planner.formation_radius, rho_norm)
    draw_rho_profile(tube.add_subplot(grid[1, 0]), result, planner.metric, rho_norm)
    draw_overlap_check(tube.add_subplot(grid[1, 1]), result, planner.metric)
    tube.suptitle(f"joint tube U_0 -> ... -> U_M  (R_F = {planner.formation_radius:.3f} m)", fontsize=12)
    figures["tube"] = tube

    main_height = 8.0
    if result.path_poses:
        xlim, ylim = path_view_limits(result.path_poses, slots, map_data, pad=0.8)
        main_height = float(np.clip(13.0 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]), 3.0, 9.0))
    formation_figure = plt.figure(figsize=(14, main_height + 3.4))
    grid = formation_figure.add_gridspec(2, 1, height_ratios=(main_height, 2.6))
    axis = formation_figure.add_subplot(grid[0])
    draw_formation(axis, result, map_data, slots, planner)
    axis.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    draw_yaw_profile(formation_figure.add_subplot(grid[1]), result)
    formation_figure.suptitle("rigid formation along the joint path", fontsize=12)
    figures["formation"] = formation_figure

    frames = plt.figure(figsize=(16, 10))
    draw_formation_frames(frames, result, map_data, slots, planner)
    frames.suptitle("key frames of the rigid formation (each panel centered on the node pair, same scale)", fontsize=12)
    figures["frames"] = frames

    convergence = plt.figure(figsize=(14, 5))
    draw_convergence(convergence, result)
    convergence.suptitle(summary, fontsize=11)
    figures["convergence"] = convergence

    for name, figure in figures.items():
        if name in ("growth", "frames"):
            figure.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=2.0)
        else:
            figure.tight_layout()
    return figures


def number_tag(value: float) -> str:
    return f"{value:g}"


def variant_name(args) -> str:
    parts = [args.formation]
    if args.slot_scale != 1.0:
        parts.append(f"x{number_tag(args.slot_scale)}")
    if args.robot_radius != TURTLEBOT3_BURGER_RADIUS:
        parts.append(f"r{number_tag(args.robot_radius)}")
    if args.safety_margin != DEFAULT_SAFETY_MARGIN:
        parts.append(f"sm{number_tag(args.safety_margin)}")
    if args.obstacle_radius is not None:
        parts.append(f"obs{args.obstacle_radius.replace(':', '-')}")
    if args.obstacle_count is not None:
        parts.append(f"n{args.obstacle_count}")
    parts.append("first_goal" if args.first_goal else f"anytime_it{args.iterations}")
    if args.margin_weight > 0.0:
        parts.append(f"wm{number_tag(args.margin_weight)}")
    if args.no_step_backoff:
        parts.append("fixedstep")
    return "_".join(parts)


def result_stats(result: TubeRRTResult, map_data, slots: np.ndarray) -> dict[str, int]:
    return {
        "tree_nodes": len(result.tree_nodes),
        "goal_nodes": sum(e.status == "goal" for e in result.trace),
        "rejected_collision": sum(e.status == "collision" for e in result.trace),
        "rejected_no_overlap": sum(e.status == "no_overlap" for e in result.trace),
        "rewires": sum(len(e.rewires) for e in result.trace),
        "backoff_nodes": sum(e.status == "added" and e.attempts > 1 for e in result.trace),
        "straddle_tree_nodes": sum(bool(straddled_obstacles(n.pose, slots, map_data)) for n in result.tree_nodes),
        "straddle_path_nodes": sum(bool(straddled_obstacles(p, slots, map_data)) for p in result.path_poses),
    }


def git_revision() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}{' (有未提交改动)' if dirty else ''}"


def write_path_csv(path: Path, result: TubeRRTResult, slots: np.ndarray) -> None:
    robots = project_robot_paths(result.path_poses, slots)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "cx", "cy", "theta_rad", "rho"] + [f"robot{i}_{axis}" for i in range(len(robots)) for axis in "xy"])
        for k, (pose, radius) in enumerate(zip(result.path_poses, result.path_radii)):
            row = [k, f"{pose.x:.6f}", f"{pose.y:.6f}", f"{pose.yaw:.6f}", f"{radius:.6f}"]
            row += [f"{value:.6f}" for robot in robots for value in robot[k]]
            writer.writerow(row)


def write_report(run_dir: Path, summary: dict, console: str) -> None:
    config, stats = summary["config"], summary["stats"]
    first = summary["first_goal_iteration"]
    lines = [
        f"# Tube-RRT 运行记录：{summary['map']} / seed {summary['seed']} / {summary['variant']}",
        "",
        f"- 运行时间：{summary['timestamp']}",
        f"- 代码版本：`{summary['git']}`",
        f"- 结果目录：`{summary['run_dir']}`",
        "",
        "复现命令（在仓库根目录执行）：",
        "",
        "```bash",
        summary["command"],
        "```",
        "",
        "## 结果",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 是否成功 | {'是' if summary['success'] else '否：' + summary['failure_reason']} |",
        f"| 执行迭代数 | {summary['iterations']} / {config['max_iterations']} |",
        f"| 首次连到目标的迭代 | {first if first is not None else '-'} |",
        f"| 树节点数（含目标节点） | {stats['tree_nodes']}（目标节点 {stats['goal_nodes']}） |",
        f"| 被拒绝：碰撞 / 安全球不重叠 | {stats['rejected_collision']} / {stats['rejected_no_overlap']} |",
        f"| rewire 次数 | {stats['rewires']} |",
        f"| 步长回退后才接受的节点数 | {stats['backoff_nodes']} |",
        f"| 障碍夹在机器人之间的节点：树 / 路径 | {stats['straddle_tree_nodes']} / "
        f"{stats['straddle_path_nodes']}（路径共 {summary['path_nodes']} 个节点） |",
        f"| 路径代价（含 J_margin）：首次 → 最终 | {summary['first_goal_cost']} → {summary['path_cost']} |",
        f"| 路径 d_G 长度 | {summary['path_length']} |",
        f"| tube 瓶颈 rho_min | {summary['bottleneck']:.3f} |",
        "",
        "## 配置",
        "",
        "| 参数 | 值 |",
        "| --- | --- |",
        f"| 地图 | {summary['map']}, seed {summary['seed']} |",
        f"| 编队 | {summary['formation']} × {number_tag(summary['slot_scale'])}"
        f"（R_F = {summary['formation_radius']:.3f} m，相邻机器人最小间距 {summary['min_slot_distance']:.3f} m） |",
        f"| 机器人半径 / 安全余量 | {summary['robot_radius']:.3f} m"
        f"{'（TurtleBot3 Burger 外接圆）' if summary['robot_radius'] == TURTLEBOT3_BURGER_RADIUS else ''}"
        f" / {summary['safety_margin']:.3f} m |",
        f"| 障碍半径 | {summary['obstacle_radius_range'][0]:.3f}–{summary['obstacle_radius_range'][1]:.3f} m"
        f"（设置：{summary['obstacle_radius_spec']}，共 {summary['obstacle_count']} 个） |",
        f"| 能从相邻机器人之间穿过的障碍半径上限 | {summary['pass_through_bound']:.3f} m"
        f"（= 最宽相邻间距/2 − 机器人半径 − 安全余量；小于它的障碍 {summary['passable_obstacles']}"
        f"/{summary['obstacle_count']} 个） |",
        *[f"| {key} | {value} |" for key, value in config.items()],
        "",
        "## 耗时",
        "",
        "| 阶段 | 秒 |",
        "| --- | --- |",
        *[f"| {key} | {value:.3f} |" for key, value in summary["timing"].items()],
        "",
        "## 图",
        "",
    ]
    for name, (filename, description) in FIGURE_FILES.items():
        lines += [f"### {filename}", "", description, "", f"![{name}]({filename})", ""]
    lines += [
        "## 其他文件",
        "",
        "- `path.csv`：联合路径节点（cx, cy, theta, rho）及各机器人投影坐标。",
        "- `summary.json`：本次运行的机器可读摘要，`results/tube_rrt/README.md` 索引由它生成。",
        "",
        "## 控制台输出",
        "",
        "```text",
        console.rstrip(),
        "```",
        "",
    ]
    (run_dir / "run.md").write_text("\n".join(lines), encoding="utf-8")


def write_index(out_dir: Path) -> Path:
    runs = []
    for summary_path in sorted(out_dir.glob("*/*/summary.json")):
        runs.append(json.loads(summary_path.read_text(encoding="utf-8")))
    lines = [
        "# Tube-RRT 可视化结果索引",
        "",
        "本文件由 `scripts/visualize_tube_rrt.py` 在每次运行后自动重新生成，请勿手动编辑。",
        "",
        "目录结构：`results/tube_rrt/<地图>_seed<seed>/<变体>/`，变体名依次由以下部分组成：编队名；`x<k>`（槽位整体放大 k 倍，"
        "k=1 时省略）；`first_goal`（首次连到目标即停止）或 `anytime_it<N>`（跑满 N 次迭代，保留代价最小的目标节点）；"
        "`wm<w>`（J_margin 权重）；`fixedstep`（关闭步长回退）。每个运行目录下的 `run.md` 是可直接阅读的运行报告。",
        "",
        "“夹障碍节点”指障碍中心落在机器人凸包内的路径节点数，用来判断规划是否利用了“障碍从机器人之间穿过”。",
        "",
        "“可穿过障碍”指半径小于“相邻机器人间隙允许的障碍半径上限”的障碍个数；为 0 时规划器只能绕行。",
        "",
        "| 运行 | 成功 | 节点 | 首次到达迭代 | 代价 首次→最终 | d_G 长度 | 瓶颈 rho | 可穿过障碍 / 上限 m | "
        "夹障碍节点 / 路径节点 | 规划耗时 s | 运行时间 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        relative = Path(os.path.relpath(REPO_ROOT / run["run_dir"], out_dir)).as_posix()
        stats = run["stats"]
        lines.append(
            f"| [{relative}]({relative}/run.md) | {'是' if run['success'] else '否'} | {stats['tree_nodes']} | "
            f"{run['first_goal_iteration'] if run['first_goal_iteration'] is not None else '-'} | "
            f"{run['first_goal_cost']} → {run['path_cost']} | {run.get('path_length', '-')} | {run['bottleneck']:.3f} | "
            f"{run.get('passable_obstacles', '-')}/{run.get('obstacle_count', '-')} / "
            f"{run.get('pass_through_bound', float('nan')):.3f} | "
            f"{stats.get('straddle_path_nodes', '-')} / {run['path_nodes']} | {run['timing']['tube_rrt']:.2f} | "
            f"{run['timestamp']} |"
        )
    lines.append("")
    index = out_dir / "README.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return index


def reproduce_command(argv: list[str]) -> str:
    python = os.path.relpath(sys.executable, REPO_ROOT) if Path(sys.executable).is_absolute() else sys.executable
    script = os.path.relpath(Path(__file__).resolve(), REPO_ROOT)
    return "MPLBACKEND=Agg " + shlex.join([python, script, *argv])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Joint Tube-RRT plan.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "tube_rrt",
                        help="results root; each run is saved to <out-dir>/<map>_seed<seed>/<variant>/")
    parser.add_argument("--map", default="random_circles", choices=MAP_NAMES)
    parser.add_argument("--formation", default="square", choices=["square", "column", "horizontal_line", "t_shape"])
    parser.add_argument("--slot-scale", type=float, default=1.0,
                        help="scale all formation slots (e.g. 2 widens the inter-robot gaps so obstacles can pass between)")
    parser.add_argument("--robot-radius", type=float, default=TURTLEBOT3_BURGER_RADIUS,
                        help="robot disk radius in m (default: TurtleBot3 Burger, 138x178 mm footprint -> 0.113)")
    parser.add_argument("--safety-margin", type=float, default=DEFAULT_SAFETY_MARGIN,
                        help="extra clearance per robot in m")
    parser.add_argument("--obstacle-radius", type=parse_obstacle_radius, default=None,
                        help="obstacle size: R, MIN:MAX (random_circles) or 'auto' (sized from the largest obstacle "
                             "that fits between neighbouring robots); default keeps the map's own sizes")
    parser.add_argument("--obstacle-count", type=int, default=None, help="number of random circles")
    parser.add_argument("--seed", type=int, default=7, help="planner seed; also the map seed for random_circles")
    parser.add_argument("--iterations", type=int, default=2500, help="Tube-RRT iteration budget (max_iterations)")
    parser.add_argument("--first-goal", action="store_true",
                        help="stop at the first goal connection instead of running the full budget")
    parser.add_argument("--margin-weight", type=float, default=0.0,
                        help="J_margin weight w: edge cost = d_G * (1 + w / min(rho_a, rho_b))")
    parser.add_argument("--no-step-backoff", action="store_true",
                        help="disable step backoff (always steer by the fixed metric_step)")
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--show", action="store_true", help="also open the figures interactively after saving")
    args = parser.parse_args()
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.slot_scale <= 0.0:
        parser.error("--slot-scale must be positive")
    if args.margin_weight < 0.0:
        parser.error("--margin-weight must be non-negative")
    if args.robot_radius <= 0.0 or args.safety_margin < 0.0:
        parser.error("--robot-radius must be positive and --safety-margin non-negative")
    if args.obstacle_count is not None and args.obstacle_count < 0:
        parser.error("--obstacle-count must be non-negative")

    tee = Tee(sys.stdout)
    sys.stdout = tee
    timing: dict[str, float] = {"startup_imports": time.perf_counter() - _MODULE_START}
    print(f"timing startup_imports={timing['startup_imports']:.3f}s", flush=True)
    slots = FormationLibrary.build_default(args.robot_radius).get(args.formation).slots * args.slot_scale
    bound = pass_through_radius(slots, args.robot_radius, args.safety_margin)
    map_start = time.perf_counter()
    map_data = MapBuilder().build(args.map, build_map_config(args, bound))
    timing["map_build"] = time.perf_counter() - map_start
    print(f"timing map_build={timing['map_build']:.3f}s", flush=True)
    obstacle_radii = [float(p["radius"]) for p in map_data.obstacle_primitives]
    passable = sum(r < bound for r in obstacle_radii)
    print(f"geometry robot_radius={args.robot_radius:.3f} safety_margin={args.safety_margin:.3f} "
          f"pass_through_bound={bound:.3f} obstacle_radius={min(obstacle_radii, default=0):.3f}-"
          f"{max(obstacle_radii, default=0):.3f} passable_obstacles={passable}/{len(obstacle_radii)}", flush=True)
    if obstacle_radii and passable == 0:
        print("note: every obstacle is larger than the pass-through bound, so none can pass between robots "
              "(try --obstacle-radius auto or a larger --slot-scale)", flush=True)
    config = TubeRRTConfig(seed=args.seed, max_iterations=args.iterations, progress_interval=args.progress_interval,
                           record_trace=True, stop_on_first_goal=args.first_goal, margin_weight=args.margin_weight,
                           step_backoff=not args.no_step_backoff)
    planner = TubeRRTPlanner(map_data, slots, Pose2D(*map_data.start_xy, 0.0), config=config)
    print("planning...", flush=True)
    planning_start = time.perf_counter()
    result = planner.plan()
    timing["tube_rrt"] = time.perf_counter() - planning_start
    print(f"timing tube_rrt={timing['tube_rrt']:.3f}s", flush=True)
    print(
        f"planning result: success={result.success} iterations={result.iterations} nodes={len(result.tree_nodes)} "
        f"path_nodes={len(result.path_poses)} path_cost={result.path_cost:.3f} tube_bottleneck={result.bottleneck:.3f}",
        flush=True,
    )

    plot_start = time.perf_counter()
    figures = build_figures(result, map_data, slots, planner)
    timing["plot_build"] = time.perf_counter() - plot_start
    print(f"timing plot_build={timing['plot_build']:.3f}s", flush=True)

    run_dir = args.out_dir / f"{args.map}_seed{args.seed}" / variant_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_start = time.perf_counter()
    for name, figure in figures.items():
        figure.savefig(run_dir / FIGURE_FILES[name][0], dpi=150)
    write_path_csv(run_dir / "path.csv", result, slots)
    timing["save"] = time.perf_counter() - save_start
    timing["total"] = time.perf_counter() - _MODULE_START
    print(f"timing save={timing['save']:.3f}s total={timing['total']:.3f}s", flush=True)

    display_dir = run_dir.resolve()
    display_dir = display_dir.relative_to(REPO_ROOT) if display_dir.is_relative_to(REPO_ROOT) else display_dir
    print(f"saved {display_dir}/", flush=True)
    sys.stdout = tee.stream
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_revision(),
        "command": reproduce_command(sys.argv[1:]),
        "run_dir": display_dir.as_posix(),
        "map": args.map,
        "variant": variant_name(args),
        "formation": args.formation,
        "slot_scale": args.slot_scale,
        "formation_radius": planner.formation_radius,
        "min_slot_distance": min(float(np.linalg.norm(a - b)) for i, a in enumerate(slots) for b in slots[i + 1:]),
        "robot_radius": args.robot_radius,
        "safety_margin": args.safety_margin,
        "obstacle_radius_spec": args.obstacle_radius or "地图默认",
        "obstacle_radius_range": [min(obstacle_radii, default=0.0), max(obstacle_radii, default=0.0)],
        "pass_through_bound": bound,
        "passable_obstacles": passable,
        "obstacle_count": len(obstacle_radii),
        "seed": args.seed,
        "mode": "first_goal" if args.first_goal else f"anytime_it{args.iterations}",
        "success": result.success,
        "failure_reason": result.failure_reason,
        "iterations": result.iterations,
        "first_goal_iteration": result.first_goal_iteration,
        "first_goal_cost": f"{result.cost_history[0][1]:.3f}" if result.cost_history else "-",
        "path_cost": f"{result.path_cost:.3f}" if result.success else "-",
        "path_length": (f"{sum(planner.metric(a, b) for a, b in zip(result.path_poses, result.path_poses[1:])):.3f}"
                        if result.success else "-"),
        "path_nodes": len(result.path_poses),
        "bottleneck": result.bottleneck,
        "stats": result_stats(result, map_data, slots),
        "config": {key: value for key, value in asdict(config).items() if key not in ("record_trace",)},
        "timing": timing,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(run_dir, summary, tee.getvalue())
    out_root = args.out_dir.resolve()
    index = write_index(out_root)
    print(f"report {display_dir}/run.md", flush=True)
    print(f"index  {index.relative_to(REPO_ROOT) if index.is_relative_to(REPO_ROOT) else index}", flush=True)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
