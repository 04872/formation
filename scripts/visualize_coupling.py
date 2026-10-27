#!/usr/bin/env python3
"""Visualize single-entry RRT* (A) and anchor-cost RRT* + post-hoc chain optimisation (B) on one scenario / seed.

Figures follow scripts/visualize_tube_rrt.py: search tree, growth, joint tube (path regions), rigid formation along the
optimised route, key frames, convergence, the coupling-specific edge view and the path regions as (x, y, theta)
solids.  Both variants share the region expansion core and the random stream, so they build the same regions; the
comparison figure puts their region chains and routes side by side.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import shlex
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from compare_coupling import A, B, DESCRIPTIONS, LEGACY, ROBOT_RADIUS, SAFETY_MARGIN, SCENARIOS, route_profile, scenario
from formation.polyhedral_cell import PolyhedralCellModel
from formation.region_coupling import PLANNERS, AnchorRRTStar, CouplingConfig, SingleEntryRRTStar
from formation.tube_rrt import project_robot_paths, transform_slots
from formation.types import Pose2D, wrap_to_pi
from visualize_tube_rrt import (PATH_COLOR, RHO_CMAP, ROBOT_COLORS, STRADDLE_COLOR, TREE_CMAP, draw_formation_pose,
                                draw_map, draw_path, draw_poly_cell_solid, draw_robot_trajectories, git_revision,
                                heading_segments, path_view_limits, poly_slice, sample_poses_evenly,
                                straddled_obstacles, tree_style)

LABELS = {A: "A: single-entry RRT*", B: "B: anchor-cost RRT* + post-hoc chain",
          "proxy_rrtstar": "B0: seed-distance proxy RRT*", "portal_rrtstar": "B1: fixed-portal RRT*"}
VARIANT_COLORS = {A: "tab:red", B: "tab:blue", "proxy_rrtstar": "tab:green", "portal_rrtstar": "tab:purple"}
FIGURE_FILES = {
    "overview": ("overview.png", "总览：区域树、路径区域链与认证路线、编队投影、沿路线的稠密 guarded clearance 四宫格"),
    "tree": ("1_tree.png", "最终区域树 (x, y) 投影：节点为区域 seed，颜色 = 插入顺序，短线 = yaw；A 的边画在 entry 之间"
                           "（e_parent -> e_child，正是 J 累加的折线），B 的边画成 seed -> (overlap witness) -> seed 的"
                           "认证折线；红叉 = steer 失败的采样，粉色圈 = 障碍夹在机器人之间的 seed"),
    "growth": ("2_growth.png", "由 checkpoint 快照回放的生长过程：父子 seed 连线、当轮采样 q_rand、被 region-gap 选中的区域 "
                               "C_i（蓝框）、新区域 C_new 的 dtheta = 0 截面（青色），以及当时的最优路线"),
    "tube": ("3_tube.png", "joint tube：路径区域链 C_0 -> ... -> C_m 的 dtheta = 0 截面与链优化得到的过渡点 z_k、"
                           "(x, y, theta) 中的区域实体、沿路线的稠密 clearance、公共部分最窄（最大公共余量最小）"
                           "的相邻区域对在过渡点 yaw 下的截面"
                           "（实线 = 障碍支撑平面，虚线 = validity guard）"),
    "formation": ("4_formation.png", "沿优化路线等间距采样的编队姿态（编号、轮廓线连接各机器人，粉色填充 = 障碍夹在机器人"
                                     "之间）、各机器人轨迹、最小 clearance 处的机器人安全圈；下方为 yaw 剖面"),
    "frames": ("5_formation_frames.png", "关键帧放大：障碍夹在机器人之间（或 clearance 最小处）附近的连续姿态，比例尺相同，"
                                         "虚线为上一帧姿态，黑箭头为中心位移"),
    "convergence": ("6_convergence.png", "最优路径长度随迭代下降：实线 = 共享链优化后的长度，虚线 = 在线代价（A 为 J，"
                                         "B 为锚点代价 J~）；右图为区域数、steer 失败、rewire 等累计数"),
    "coupling": ("7_coupling.png", "耦合方式：全部区域截面与最终树边，按边的来历着色（A：发现父区域 / ChooseParent-Rewire "
                                   "换到的父区域；B：seed 线段直连 / 经 overlap witness 绕行）；ChooseParent / Rewire "
                                   "判定计数；在线时间分解"),
    "tube3d": ("8_tube_3d.png", "路径区域链在 (x, y, theta) 中的三维实体，两个视角；底面为障碍投影，红线为优化路线"),
}


@dataclass
class Run:
    scenario: str
    variant: str
    seed: int
    planner: object
    map_data: object
    slots: np.ndarray
    result: dict
    steer_log: list
    snapshots: list
    wall_time: float
    route: list = field(default_factory=list)
    sequence: list = field(default_factory=list)
    arc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dense: list = field(default_factory=list)
    clearance: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def final(self) -> dict:
        return self.result["checkpoints"][-1]

    @property
    def cells(self) -> list:
        return [self.planner.cells[i] for i in self.sequence]


def execute(name: str, variant: str, seed: int, iterations: int, every: int) -> Run:
    """Plan once with dense checkpoints; the steer outcome of every iteration and a tree snapshot at every checkpoint
    are recorded by wrapping the planner (online timing excludes checkpoints, as in compare_coupling)."""
    map_data, slots, start = scenario(name)
    checkpoints = tuple(range(every, iterations, every)) + (iterations,)
    config = CouplingConfig(seed=seed, max_iterations=iterations, checkpoints=checkpoints)
    planner = PLANNERS[variant](map_data, slots, start, config)
    steer_log, snapshots = [], []
    steer, checkpoint = planner._steer, planner._checkpoint

    def logged_steer(parent, sample):
        out = steer(parent, sample)
        status = "added" if out[0] is not None else ("covered" if out[2] == 0 else "failed")
        steer_log.append((sample, parent, status, len(planner.cells) if out[0] is not None else None))
        return out

    def logged_checkpoint(iteration, online):
        record = checkpoint(iteration, online)
        snapshots.append(dict(record, stats=dict(planner.stats), parent=list(planner.parent)))
        return record

    planner._steer, planner._checkpoint = logged_steer, logged_checkpoint
    started = time.perf_counter()
    result = planner.plan()
    run = Run(name, variant, seed, planner, map_data, slots, result, steer_log, snapshots,
              time.perf_counter() - started)
    if run.final["success"]:
        run.route = [Pose2D(*p) for p in run.final["route"]]
        run.sequence = list(run.final["sequence"])
        arc, poses, clearance = route_profile(map_data, slots, run.final["route"])
        run.arc, run.clearance = arc, clearance
        run.dense = [Pose2D(*p) for p in poses]
    if isinstance(planner, SingleEntryRRTStar):
        for i in range(len(planner.cells)):
            planner._refresh(i)
    return run


# -- geometry helpers ----------------------------------------------------------------------------------------------
def slices(run: Run) -> list:
    cache = getattr(run, "_slices", None)
    if cache is None:
        cache = [poly_slice(cell, 0.0) for cell in run.planner.cells]
        run._slices = cache
    return cache


def tree_edge(run: Run, i: int) -> tuple[list[Pose2D], str]:
    """The polyline the online cost of edge parent(i) -> i measures, and its kind."""
    planner = run.planner
    p = planner.parent[i]
    if isinstance(planner, SingleEntryRRTStar):
        kind = "discovery" if p == planner.discovery[i] else "improved"
        return [planner.entry[p], planner.entry[i]], kind
    a, b = planner.cells[p].pose, planner.cells[i].pose
    if isinstance(planner, AnchorRRTStar):
        v = np.array((b.x - a.x, b.y - a.y, planner.rho * wrap_to_pi(b.yaw - a.yaw)))
        d = float(np.linalg.norm(v))
        if d > 1e-12:
            lo_i, hi_i = planner._interval(planner.cells[p], v / d)
            lo_j, hi_j = planner._interval(planner.cells[i], -v / d)
            if not max(lo_i, d - hi_j, 0.0) <= min(hi_i, d - lo_j, d):
                witness = planner._overlap(planner.cells[p], planner.cells[i])
                if witness is not None:
                    return [a, witness, b], "detour"
        return [a, b], "direct"
    return [a, b], "discovery" if p == planner.discovery[i] else "improved"


EDGE_STYLES = {
    "discovery": ("tab:blue", "parent = discovery region"),
    "improved": ("tab:purple", "parent changed by ChooseParent / Rewire"),
    "direct": ("tab:green", "seed segment inside C_i ∪ C_j (direct)"),
    "detour": ("tab:orange", "detour through overlap witness w_ij"),
}


def overlap_margins(run: Run) -> list[float]:
    """``max_z min(slack_k(z), slack_k+1(z))`` per consecutive chain pair: how wide the overlap the route passes is."""
    geo, cells = run.planner.geo, run.cells
    return [geo.max_margin(a, b)[0] for a, b in zip(cells, cells[1:])]


def view_limits(run: Run, pad: float = 0.8):
    if run.route:
        return path_view_limits(run.route, run.slots, run.map_data, pad)
    ox, oy = run.map_data.origin_xy
    return (ox, ox + run.map_data.width_m), (oy, oy + run.map_data.height_m)


def rho_norm(run: Run) -> Normalize:
    return Normalize(0.0, max([cell.radius for cell in run.cells] or [1.0]))


# -- drawing -------------------------------------------------------------------------------------------------------
def draw_tree(axis, run: Run) -> None:
    planner, map_data = run.planner, run.map_data
    draw_map(axis, map_data)
    cells = planner.cells
    n = len(cells)
    norm = Normalize(0, max(1, n - 1))
    style = tree_style(n)
    failed = [s for s, _, status, _ in run.steer_log if status == "failed"]
    if failed:
        axis.scatter([p.x for p in failed], [p.y for p in failed], marker="x", s=12, color="#e41a1c",
                     alpha=style["reject_alpha"], linewidth=0.7, zorder=3, label=f"steer failed ({len(failed)} samples)")
    segments, colors = [], []
    for i in range(1, n):
        poses, _ = tree_edge(run, i)
        segments.append([(p.x, p.y) for p in poses])
        colors.append(TREE_CMAP(norm(i)))
    axis.add_collection(LineCollection(segments, colors=colors, linewidths=style["edge"], zorder=4))
    seeds = [c.pose for c in cells]
    if isinstance(planner, SingleEntryRRTStar):
        axis.scatter([e.x for e in planner.entry], [e.y for e in planner.entry], s=style["node"] * 0.6, marker="D",
                     color="0.2", linewidth=0, zorder=5, label="entry e_i")
    show_ticks = n <= 600
    if show_ticks:
        axis.add_collection(LineCollection(heading_segments(seeds, style["tick"]), colors="0.15", linewidths=0.7,
                                           alpha=style["tick_alpha"], zorder=5))
    scatter = axis.scatter([p.x for p in seeds], [p.y for p in seeds], c=np.arange(n), cmap=TREE_CMAP, norm=norm,
                           s=style["node"], edgecolor="k", linewidth=0.3, zorder=5)
    axis.plot([], [], color=TREE_CMAP(0.6), linewidth=1.4, marker="o", markersize=4, markeredgecolor="k",
              label=f"region tree ({n} regions{', tick = yaw' if show_ticks else ''})")
    straddling = [p for p in seeds if straddled_obstacles(p, run.slots, map_data)]
    if straddling:
        axis.scatter([p.x for p in straddling], [p.y for p in straddling], s=60, facecolor="none",
                     edgecolor=STRADDLE_COLOR, linewidth=1.3, zorder=6,
                     label=f"obstacle between robots ({len(straddling)} seeds)")
    draw_path(axis, run.route, label="optimised route", nodes=False)
    colorbar = axis.figure.colorbar(scatter, ax=axis, fraction=0.03, pad=0.01)
    colorbar.set_label("region insertion order")


def growth_picks(run: Run, count: int = 6) -> list[int]:
    snaps = run.snapshots
    first = next((k for k, s in enumerate(snaps) if s["success"]), None)
    if first is None:
        return sorted({int(round(f * (len(snaps) - 1))) for f in np.linspace(0.1, 1.0, count)})
    picks = {first, len(snaps) - 1}
    before = list(range(first))
    for f in (0.3, 0.7):
        if before:
            picks.add(before[int(round(f * (len(before) - 1)))])
    first_iteration = snaps[first]["iteration"]
    for target in (2 * first_iteration, snaps[-1]["iteration"] / 3):
        picks.add(int(np.argmin([abs(s["iteration"] - target) for s in snaps])))
    k = 1
    while len(picks) < min(count, len(snaps)):
        picks.add(int(round(k / count * (len(snaps) - 1))))
        k += 1
    return sorted(picks)[-count:] if len(picks) > count else sorted(picks)


def draw_growth_snapshot(axis, run: Run, snap: dict) -> None:
    planner, map_data = run.planner, run.map_data
    draw_map(axis, map_data, labels=False)
    parent = snap["parent"]
    n = len(parent)
    total = len(planner.cells)
    norm = Normalize(0, max(1, total - 1))
    style = tree_style(n)
    polygons = [p for p in slices(run)[:n] if p is not None]
    axis.add_collection(PolyCollection(polygons, facecolors="tab:cyan", edgecolors="none", alpha=0.03, zorder=3))
    iteration = snap["iteration"]
    failed = [s for s, _, status, _ in run.steer_log[:iteration] if status == "failed"]
    if failed:
        axis.scatter([p.x for p in failed], [p.y for p in failed], marker="x", s=8, color="#e41a1c",
                     alpha=style["reject_alpha"] * 0.7, linewidth=0.6, zorder=3)
    seeds = [c.pose for c in planner.cells[:n]]
    axis.add_collection(LineCollection([((seeds[p].x, seeds[p].y), (seeds[i].x, seeds[i].y))
                                        for i, p in enumerate(parent) if p is not None],
                                       colors=[TREE_CMAP(norm(i)) for i, p in enumerate(parent) if p is not None],
                                       linewidths=style["edge"] * 0.9, zorder=4))
    axis.scatter([p.x for p in seeds], [p.y for p in seeds], c=np.arange(n), cmap=TREE_CMAP, norm=norm,
                 s=style["node"] * 0.7, edgecolor="k", linewidth=0.25, zorder=5)
    if snap.get("route"):
        draw_path(axis, [Pose2D(*p) for p in snap["route"]], label=None, nodes=False)
    sample, selected, status, index = run.steer_log[iteration - 1]
    polygon = slices(run)[selected]
    if polygon is not None:
        axis.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor="tab:blue", linewidth=1.4, zorder=7))
    near = planner.cells[selected].pose
    axis.scatter(sample.x, sample.y, marker="P", s=70, color="tab:red", edgecolor="k", zorder=8)
    axis.plot([near.x, sample.x], [near.y, sample.y], color="tab:red", linestyle=":", linewidth=1.0, zorder=7)
    axis.scatter(near.x, near.y, s=70, facecolor="none", edgecolor="tab:red", linewidth=1.5, zorder=8)
    if index is not None and slices(run)[index] is not None:
        axis.add_patch(Polygon(slices(run)[index], closed=True, facecolor="tab:cyan", alpha=0.35,
                               edgecolor="tab:blue", linewidth=1.0, zorder=6))
        new = planner.cells[index].pose
        axis.scatter(new.x, new.y, s=30, color="tab:blue", edgecolor="k", zorder=9)
    cost = f", route {snap['optimised_cost']:.2f}" if snap["success"] else ", no goal region yet"
    tag = " (first goal)" if run.result["first"] and iteration == run.result["first"]["iteration"] else ""
    axis.set_title(f"iteration {iteration}{tag}: {n} regions, steer {status}{cost}", fontsize=10)


def draw_growth(figure, run: Run) -> None:
    axes = figure.subplots(2, 3, sharex=True, sharey=True)
    picks = growth_picks(run, axes.size)
    for axis, k in zip(axes.flat, picks):
        draw_growth_snapshot(axis, run, run.snapshots[k])
    for axis in list(axes.flat)[len(picks):]:
        axis.set_visible(False)
    handles = [
        Line2D([], [], marker="P", color="tab:red", markeredgecolor="k", linestyle="", markersize=9, label="sample q_rand"),
        Line2D([], [], marker="o", color="none", markeredgecolor="tab:red", linestyle="", markersize=9,
               label="seed of the region-gap choice C_i"),
        Line2D([], [], color="tab:blue", linewidth=1.4, label="C_i (dtheta = 0 section)"),
        Line2D([], [], marker="s", color="tab:cyan", alpha=0.5, linestyle="", markersize=11, label="new region C_new"),
        Line2D([], [], marker="x", color="#e41a1c", linestyle="", markersize=6, label="steer failed so far"),
        Line2D([], [], color=PATH_COLOR, linewidth=2.4, label="current optimised route"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)


def draw_chain_xy(axis, run: Run) -> None:
    draw_map(axis, run.map_data)
    norm = rho_norm(run)
    for cell in run.cells:
        polygon = poly_slice(cell, 0.0)
        if polygon is not None:
            color = RHO_CMAP(norm(cell.radius))
            axis.add_patch(Polygon(polygon, closed=True, facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.0,
                                   zorder=4))
    portals = run.route[1:-1]
    if portals:
        axis.scatter([p.x for p in portals], [p.y for p in portals], marker="D", s=18, color="tab:cyan",
                     edgecolor="k", linewidth=0.5, zorder=9, label="z_k in C_k ∩ C_k+1 (chain optimum)")
    if run.cells:
        cell = min(run.cells, key=lambda c: c.radius)
        axis.annotate(f"smallest in-radius = {cell.radius:.3f}", (cell.pose.x, cell.pose.y), xytext=(0, 55),
                      textcoords="offset points", ha="center", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6"),
                      arrowprops=dict(arrowstyle="->", color="k"), zorder=10)
    draw_path(axis, run.route, label="optimised route start -> z_0 -> ... -> goal")
    axis.set_title(f"region chain ({len(run.cells)} regions): sections P_i(dtheta = 0) and transition points",
                   fontsize=10)


def draw_clearance(axis, run: Run) -> None:
    if not len(run.arc):
        return
    s, clearance = run.arc, run.clearance
    axis.fill_between(s, 0.0, np.maximum(clearance, 0.0), color=RHO_CMAP(0.55), alpha=0.25)
    axis.fill_between(s, 0.0, np.minimum(clearance, 0.0), color="#e41a1c", alpha=0.45)
    axis.plot(s, clearance, color="k", linewidth=1.0)
    waypoints = np.concatenate([[0.0], np.cumsum([run.planner.geo.distance(a, b)
                                                  for a, b in zip(run.route, run.route[1:])])])
    values = np.interp(waypoints, s, clearance)
    axis.scatter(waypoints, values, s=22, color="tab:cyan", edgecolor="k", zorder=5, label="route waypoints z_k")
    worst = float(clearance.min())
    axis.axhline(worst, color=PATH_COLOR, linestyle="--", linewidth=1.2, label=f"minimum guarded clearance = {worst:.3f}")
    axis.axhline(0.0, color="0.3", linewidth=0.8)
    axis.set_xlabel("route length d_W [m]")
    axis.set_ylabel("guarded clearance [m]")
    axis.grid(True, color="0.9")
    axis.legend(fontsize=8, loc="upper right")
    axis.set_title("clearance along the densely sampled optimised route (robot radius + safety margin removed)",
                   fontsize=10)


def draw_portal_slice(axis, run: Run) -> None:
    margins = overlap_margins(run)
    if not margins:
        axis.text(0.5, 0.5, "no transition point", transform=axis.transAxes, ha="center")
        axis.set_axis_off()
        return
    k = int(np.argmin(margins))
    portal = run.route[k + 1]
    draw_map(axis, run.map_data, labels=False)
    outlines = []
    for cell, color, name in ((run.cells[k], "tab:blue", "C_k"), (run.cells[k + 1], "tab:orange", "C_k+1")):
        polygon, labels = PolyhedralCellModel.slice_polygon(cell, wrap_to_pi(portal.yaw - cell.pose.yaw))
        if len(polygon):
            outlines.append(polygon)
            axis.add_patch(Polygon(polygon, closed=True, facecolor=color, alpha=0.15, edgecolor="none", zorder=4,
                                   label=f"{name}(theta_z)"))
            for j, label in enumerate(labels.tolist()):
                a, b = polygon[j], polygon[(j + 1) % len(polygon)]
                guard = label < 0
                axis.plot([a[0], b[0]], [a[1], b[1]], color=color, linestyle="--" if guard else "-",
                          linewidth=1.0 if guard else 2.0, zorder=5)
        axis.scatter(cell.pose.x, cell.pose.y, marker="o", s=30, color=color, edgecolor="k", zorder=7)
    draw_formation_pose(axis, portal, run.slots, run.map_data, ROBOT_RADIUS, None, ghost=True, zorder=3)
    axis.scatter(portal.x, portal.y, marker="D", s=40, color="tab:cyan", edgecolor="k", zorder=20, label="z_k")
    axis.plot([], [], color="0.3", linewidth=2.0, label="robot-obstacle support plane")
    axis.plot([], [], color="0.3", linewidth=1.0, linestyle="--", label="validity guard ||dc|| + rho|dtheta| <= r_g")
    points = np.vstack(outlines + [transform_slots(portal, run.slots)])
    low, high = points.min(axis=0) - 0.3, points.max(axis=0) + 0.3
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.legend(fontsize=7, loc="upper left", framealpha=0.85)
    axis.set_title(f"narrowest overlap C_{k} ∩ C_{k + 1} ({k + 1}/{len(margins)}): max common slack = "
                   f"{margins[k]:.3f}, z_{k} theta = {portal.yaw:.3f} rad", fontsize=10)


def draw_chain_3d(axis, run: Run, max_cells: int, title: str) -> None:
    cells = run.cells
    picks = np.unique(np.linspace(0, len(cells) - 1, min(max_cells, len(cells)), dtype=int)) if cells else []
    yaws = np.unwrap([c.pose.yaw for c in cells]) if cells else []
    norm = Normalize(0, max(1, len(picks) - 1))
    for order, i in enumerate(picks):
        draw_poly_cell_solid(axis, cells[i], yaws[i], TREE_CMAP(norm(order)))
    if run.route:
        route_yaws = np.unwrap([p.yaw for p in run.route])
        if cells:
            route_yaws += 2 * math.pi * round((yaws[0] - route_yaws[0]) / (2 * math.pi))
        axis.plot([p.x for p in run.route], [p.y for p in run.route], route_yaws, color=PATH_COLOR, linewidth=2.2,
                  marker="o", markersize=3, zorder=10)
    if len(picks):
        shown = [cells[i] for i in picks]
        low = min(yaws[i] - cells[i].yaw_limit for i in picks)
        xs = np.concatenate([[c.pose.x - c.guard, c.pose.x + c.guard] for c in shown])
        ys = np.concatenate([[c.pose.y - c.guard, c.pose.y + c.guard] for c in shown])
        ring = np.linspace(0.0, 2.0 * math.pi, 40)
        for primitive in run.map_data.obstacle_primitives:
            (cx, cy), radius = primitive["center_xy"], primitive["radius"]
            if xs.min() - radius <= cx <= xs.max() + radius and ys.min() - radius <= cy <= ys.max() + radius:
                axis.plot(cx + radius * np.cos(ring), cy + radius * np.sin(ring), np.full(len(ring), low),
                          color="0.25", linewidth=1.0)
        axis.set_xlim(xs.min(), xs.max())
        axis.set_ylim(ys.min(), ys.max())
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("theta [rad]")
    axis.set_box_aspect((3, 2, 1.8), zoom=1.15)
    axis.set_title(title, fontsize=10)


def draw_formation(axis, run: Run, count: int = 10) -> None:
    draw_map(axis, run.map_data)
    if not run.route:
        return
    draw_robot_trajectories(axis, run.route, run.slots)
    axis.plot([p.x for p in run.route], [p.y for p in run.route], color="k", linestyle=":", linewidth=1.0, zorder=4,
              label="center (route waypoints)")
    axis.scatter([p.x for p in run.route], [p.y for p in run.route], s=8, color="k", zorder=4)
    cmap = colormaps["turbo"]
    for k, pose in enumerate(sample_poses_evenly(run.route, count, run.planner.rho)):
        draw_formation_pose(axis, pose, run.slots, run.map_data, ROBOT_RADIUS,
                            cmap(0.08 + 0.84 * k / max(1, count - 1)), label=str(k + 1), zorder=6 + k * 6)
    axis.fill([], [], color=STRADDLE_COLOR, alpha=0.3, label="obstacle between robots")
    k = int(np.argmin(run.clearance))
    reach = ROBOT_RADIUS + SAFETY_MARGIN + float(run.clearance[k])
    for x, y in transform_slots(run.dense[k], run.slots):
        axis.add_patch(Circle((x, y), reach, fill=False, edgecolor=PATH_COLOR, linestyle="--", linewidth=1.0,
                              zorder=200))
    axis.plot([], [], color=PATH_COLOR, linestyle="--", label="minimum clearance + robot/safety radius")
    xlim, ylim = view_limits(run)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_title(f"{count} formation poses evenly spaced along the optimised route (numbered, outline = formation)",
                   fontsize=10)


def draw_yaw_profile(axis, run: Run) -> None:
    if not len(run.arc):
        return
    yaws = np.unwrap([p.yaw for p in run.dense])
    yaws -= 2 * math.pi * round(float(np.mean(yaws)) / (2 * math.pi))
    axis.plot(run.arc, yaws, color="k", linewidth=1.2)
    waypoints = np.concatenate([[0.0], np.cumsum([run.planner.geo.distance(a, b)
                                                  for a, b in zip(run.route, run.route[1:])])])
    axis.scatter(waypoints, np.interp(waypoints, run.arc, yaws), s=12, color="k", zorder=5)
    axis.set_xlabel("route length d_W [m]")
    axis.set_ylabel("theta [rad] (unwrapped)")
    low = math.floor(min(yaws.min(), -math.pi / 2) / (math.pi / 2))
    high = math.ceil(max(yaws.max(), math.pi / 2) / (math.pi / 2))
    names = {-4: r"$-2\pi$", -3: r"$-3\pi/2$", -2: r"$-\pi$", -1: r"$-\pi/2$", 0: "0", 1: r"$\pi/2$", 2: r"$\pi$",
             3: r"$3\pi/2$", 4: r"$2\pi$"}
    ticks = range(low, high + 1)
    axis.set_ylim(low * math.pi / 2, high * math.pi / 2)
    axis.set_yticks([k * math.pi / 2 for k in ticks], [names.get(k, f"{k}π/2") for k in ticks])
    axis.grid(True, color="0.9")
    axis.set_title("formation heading along the optimised route", fontsize=10)


def key_frames(run: Run, count: int) -> list[int]:
    """Entering / middle / leaving frames of the straddle events (obstacle inside the robot hull), spread over the
    route if there are more events than panels; otherwise frames around the minimum clearance."""
    poses = run.dense
    last = len(poses) - 1
    straddle = [k for k, p in enumerate(poses) if straddled_obstacles(p, run.slots, run.map_data)]
    if straddle:
        events = [[straddle[0], straddle[0]]]
        for k in straddle[1:]:
            if k - events[-1][1] > 1:
                events.append([k, k])
            events[-1][1] = k
        per = 3
        chosen = [events[int(round(v))] for v in np.linspace(0, len(events) - 1, max(1, min(len(events), count // per)))]
        frames = set()
        for low, high in chosen:
            pad = max(1, (high - low) // 2)
            frames |= {max(0, low - pad), (low + high) // 2, min(last, high + pad)}
        return sorted(frames)
    window = max(count, len(poses) // 12)
    center = int(np.argmin(run.clearance))
    low, high = max(0, center - window // 2), min(last, center + window // 2)
    return sorted({int(round(v)) for v in np.linspace(low, high, min(count, high - low + 1))})


def draw_frames(figure, run: Run, count: int = 6) -> None:
    axes = figure.subplots(2, 3).flat
    if not run.dense:
        return
    frames = key_frames(run, count)
    poses = run.dense
    previous = [frames[j - 1] if j and run.arc[k] - run.arc[frames[j - 1]] < 2.0 else None
                for j, k in enumerate(frames)]
    boxes = []
    for k, prev in zip(frames, previous):
        points = np.vstack([transform_slots(poses[i], run.slots) for i in ((k,) if prev is None else (prev, k))])
        boxes.append((points.min(axis=0), points.max(axis=0)))
    half = max(float(np.max(high - low)) / 2.0 for low, high in boxes) + ROBOT_RADIUS + 0.35
    for axis, k, j, (low, high) in zip(axes, frames, previous, boxes):
        center = (low + high) / 2.0
        draw_map(axis, run.map_data, labels=False)
        for index in straddled_obstacles(poses[k], run.slots, run.map_data):
            primitive = run.map_data.obstacle_primitives[index]
            axis.add_patch(Circle(primitive["center_xy"], primitive["radius"], facecolor="0.25",
                                  edgecolor=STRADDLE_COLOR, linewidth=2.5, zorder=3))
        draw_robot_trajectories(axis, run.route, run.slots, alpha=0.35, labels=False)
        move = ""
        if j is not None:
            prev = poses[j]
            draw_formation_pose(axis, prev, run.slots, run.map_data, ROBOT_RADIUS, None, ghost=True, zorder=5)
            axis.annotate("", xy=(poses[k].x, poses[k].y), xytext=(prev.x, prev.y),
                          arrowprops=dict(arrowstyle="->", color="k", lw=1.2), zorder=20)
            move = f", step d_W={run.planner.geo.distance(prev, poses[k]):.2f}"
        draw_formation_pose(axis, poses[k], run.slots, run.map_data, ROBOT_RADIUS, "tab:blue", zorder=10)
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.set_title(f"s = {run.arc[k]:.2f}/{run.arc[-1]:.2f}: theta={wrap_to_pi(poses[k].yaw):.3f} rad, "
                       f"clearance={run.clearance[k]:.3f}{move}", fontsize=9)
    for axis in list(axes)[len(frames):]:
        axis.set_visible(False)
    handles = [
        Line2D([], [], color="tab:blue", linewidth=2.0, label="current pose (outline through robots)"),
        Line2D([], [], color="0.45", linestyle="--", linewidth=1.2, label="previous frame"),
        Line2D([], [], marker="o", color="0.25", markeredgecolor=STRADDLE_COLOR, markeredgewidth=2.5, linestyle="",
               markersize=10, label="obstacle between robots"),
        Line2D([], [], color="0.5", alpha=0.6, linewidth=1.3, label="robot trajectories"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)


def draw_convergence(figure, run: Run) -> None:
    cost_axis, count_axis = figure.subplots(1, 2)
    snaps = sorted(run.snapshots, key=lambda s: s["iteration"])
    solved = [s for s in snaps if s["success"]]
    if solved:
        its = [s["iteration"] for s in solved] + [snaps[-1]["iteration"]]
        opt = [s["optimised_cost"] for s in solved]
        raw = [s["raw_cost"] for s in solved]
        cost_axis.step(its, opt + opt[-1:], where="post", color=PATH_COLOR, linewidth=1.8,
                       label="optimised chain length (shared optimiser)")
        cost_axis.step(its, raw + raw[-1:], where="post", color="0.35", linestyle="--", linewidth=1.3,
                       label="online cost " + ("J (entry chain)" if isinstance(run.planner, SingleEntryRRTStar)
                                               else "J~ (proxy)"))
        first = run.result["first"]
        cost_axis.axvline(first["iteration"], color="0.4", linestyle=":", linewidth=1.0)
        cost_axis.annotate(f"first goal\nit {first['iteration']}, {first['online_time_s']:.2f} s, "
                           f"opt {first['optimised_cost']:.2f}", (first["iteration"], first["optimised_cost"]),
                           xytext=(10, -5), textcoords="offset points", fontsize=9, va="top")
        cost_axis.annotate(f"final opt {opt[-1]:.3f} / online {raw[-1]:.3f}", (its[-1], opt[-1]), xytext=(-5, 10),
                           textcoords="offset points", fontsize=9, ha="right")
        cost_axis.set_xlim(0, its[-1])
        cost_axis.legend(fontsize=8, loc="upper right")
    else:
        cost_axis.text(0.5, 0.5, "no goal region", transform=cost_axis.transAxes, ha="center")
    cost_axis.set_xlabel("iteration")
    cost_axis.set_ylabel("path length d_W")
    cost_axis.grid(True, color="0.9")
    cost_axis.set_title("anytime improvement (checkpoints every "
                        f"{snaps[1]['iteration'] - snaps[0]['iteration'] if len(snaps) > 1 else '-'} iterations)",
                        fontsize=10)
    its = [s["iteration"] for s in snaps]
    series = [("regions", "tab:green", "regions", [s["regions"] for s in snaps]),
              ("steer_failed", "#e41a1c", "steer failed", None), ("covered_samples", "0.6", "sample already covered", None),
              ("rewires", "0.3", "rewires", None), ("choose_parent_improved", "tab:purple", "ChooseParent improved", None),
              ("repairs", "tab:brown", "lazy entry repairs (A)", None)]
    for key, color, label, values in series:
        if values is None:
            if key not in snaps[-1]["stats"]:
                continue
            values = [s["stats"][key] for s in snaps]
        count_axis.plot(its, values, color=color, linewidth=1.5, label=label)
    if run.result["first"]:
        count_axis.axvline(run.result["first"]["iteration"], color="0.4", linestyle=":", linewidth=1.0)
    count_axis.set_xlabel("iteration")
    count_axis.set_ylabel("cumulative count")
    count_axis.grid(True, color="0.9")
    count_axis.legend(fontsize=8, loc="upper left")
    count_axis.set_title("search statistics", fontsize=10)


def time_parts(run: Run) -> list[tuple[str, float]]:
    st = run.result["stats"]
    return [("sample + select + projection", st["select_time_s"]), ("steer + new region", st["steer_time_s"]),
            ("connect (coupling)", st["connect_time_s"]),
            ("path phase (extract + optimise)", run.final.get("extract_time_s", 0.0)
             + run.final.get("optimise_time_s", 0.0))]


def draw_coupling(figure, run: Run) -> None:
    planner = run.planner
    grid = figure.add_gridspec(2, 2, width_ratios=(1.7, 1))
    axis = figure.add_subplot(grid[:, 0])
    draw_map(axis, run.map_data)
    polygons = [p for p in slices(run) if p is not None]
    axis.add_collection(PolyCollection(polygons, facecolors="tab:cyan", edgecolors="none", alpha=0.025, zorder=3))
    groups: dict[str, list] = {}
    for i in range(1, len(planner.cells)):
        poses, kind = tree_edge(run, i)
        groups.setdefault(kind, []).append([(p.x, p.y) for p in poses])
    for kind, segments in groups.items():
        color, name = EDGE_STYLES[kind]
        axis.add_collection(LineCollection(segments, colors=color, linewidths=0.6 if kind in ("discovery", "direct")
                                           else 1.1, alpha=0.75, zorder=5, label=f"{name} ({len(segments)} edges)"))
    draw_path(axis, run.route, label="optimised route", nodes=False)
    axis.legend(fontsize=8, loc="upper left", framealpha=0.85)
    axis.set_title(f"{len(polygons)} regions (dtheta = 0 sections); final tree edges by origin", fontsize=10)

    st = run.result["stats"]
    count_axis = figure.add_subplot(grid[0, 1])
    keys = [("choose_parent_tests", "ChooseParent tests"), ("choose_parent_improved", "ChooseParent improved"),
            ("rewire_tests", "Rewire tests"), ("rewires", "rewires"), ("repairs", "lazy entry repairs"),
            ("attach_solves", "attach projections"), ("edges_direct", "direct seed edges"),
            ("edges_detour", "witness detours"), ("neighbor_candidates", "neighbour candidates"),
            ("neighbor_separated", "cheap separation rejects")]
    keys = [(k, label) for k, label in keys if k in st]
    count_axis.barh(np.arange(len(keys)), [max(st[k], 0.8) for k, _ in keys], color="tab:purple", height=0.6)
    for y, (k, _) in enumerate(keys):
        count_axis.text(max(st[k], 0.8), y, f" {st[k]}", va="center", fontsize=8)
    count_axis.set_yticks(np.arange(len(keys)), [label for _, label in keys], fontsize=8)
    count_axis.set_xscale("log")
    count_axis.invert_yaxis()
    count_axis.grid(True, axis="x", color="0.9")
    count_axis.set_title("coupling decisions (whole run)", fontsize=10)

    time_axis = figure.add_subplot(grid[1, 1])
    parts = time_parts(run)
    colors = ["tab:gray", "tab:olive", VARIANT_COLORS.get(run.variant, "tab:red"), "tab:cyan"]
    time_axis.barh(np.arange(len(parts)), [v for _, v in parts], color=colors, height=0.6)
    for y, (_, v) in enumerate(parts):
        time_axis.text(v, y, f" {v:.3f} s", va="center", fontsize=8)
    time_axis.set_yticks(np.arange(len(parts)), [k for k, _ in parts], fontsize=8)
    time_axis.invert_yaxis()
    time_axis.set_xlabel("seconds")
    time_axis.grid(True, axis="x", color="0.9")
    time_axis.set_title(f"online {run.result['online_time_s']:.2f} s (+ final path phase)", fontsize=10)


def summary_line(run: Run) -> str:
    final = run.final
    cost = (f"optimised length {final['optimised_cost']:.3f} (online {final['raw_cost']:.3f}), "
            f"{final['chain_regions']} chain regions, min clearance {run.clearance.min():.3f}"
            if final["success"] else "no goal region")
    return (f"{LABELS.get(run.variant, run.variant)} | {run.scenario} seed {run.seed}: {final['iteration']} iterations, "
            f"{run.result['regions']} regions, online {run.result['online_time_s']:.2f} s, {cost}")


def build_figures(run: Run) -> dict[str, plt.Figure]:
    summary = summary_line(run)
    figures: dict[str, plt.Figure] = {}

    overview = plt.figure(figsize=(16, 10.5))
    grid = overview.add_gridspec(2, 2)
    axis = overview.add_subplot(grid[0, 0])
    draw_tree(axis, run)
    axis.set_title("1. region tree (color = insertion order)", fontsize=11)
    axis = overview.add_subplot(grid[0, 1])
    draw_chain_xy(axis, run)
    axis.set_title("2. path region chain (dtheta = 0 sections) and transition points", fontsize=11)
    axis = overview.add_subplot(grid[1, 0])
    draw_formation(axis, run)
    axis.set_title("3. per-robot projection of the optimised route", fontsize=11)
    axis = overview.add_subplot(grid[1, 1])
    draw_clearance(axis, run)
    axis.set_title("4. guarded clearance along the optimised route", fontsize=11)
    for axis in overview.axes:
        if axis.get_label() != "<colorbar>" and axis.get_legend_handles_labels()[0] and axis.get_legend() is None:
            axis.legend(fontsize=7, loc="upper left", framealpha=0.85)
    overview.suptitle(summary, fontsize=12)
    figures["overview"] = overview

    tree = plt.figure(figsize=(13, 8.2))
    axis = tree.add_subplot()
    draw_tree(axis, run)
    axis.legend(fontsize=8, loc="upper left", framealpha=0.9)
    axis.set_title(summary, fontsize=10)
    figures["tree"] = tree

    growth = plt.figure(figsize=(16, 8.6))
    draw_growth(growth, run)
    growth.suptitle("region growth: q_rand -> region-gap choice C_i -> project onto C_i -> step towards q_rand -> "
                    "C_new overlapping C_i -> " + ("continuous ChooseParent / Rewire of entries"
                                                   if isinstance(run.planner, SingleEntryRRTStar)
                                                   else "anchor-cost ChooseParent / Rewire"), fontsize=12)
    figures["growth"] = growth

    tube = plt.figure(figsize=(16, 10))
    grid = tube.add_gridspec(2, 2, height_ratios=(1.5, 1), width_ratios=(1.1, 1))
    axis = tube.add_subplot(grid[0, 0])
    draw_chain_xy(axis, run)
    axis.legend(fontsize=8, loc="upper left")
    axis3d = tube.add_subplot(grid[0, 1], projection="3d")
    draw_chain_3d(axis3d, run, 12, "path regions as solids in (x, y, theta)")
    axis3d.view_init(elev=22, azim=-58)
    draw_clearance(tube.add_subplot(grid[1, 0]), run)
    draw_portal_slice(tube.add_subplot(grid[1, 1]), run)
    tube.suptitle(f"joint tube C_0 -> ... -> C_m  (rho = {run.planner.rho:.3f} m, delta = "
                  f"{run.planner.config.overlap_margin:g})", fontsize=12)
    figures["tube"] = tube

    main_height = 8.0
    if run.route:
        xlim, ylim = view_limits(run)
        main_height = float(np.clip(13.0 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]), 3.0, 9.0))
    formation = plt.figure(figsize=(14, main_height + 3.4))
    grid = formation.add_gridspec(2, 1, height_ratios=(main_height, 2.6))
    axis = formation.add_subplot(grid[0])
    draw_formation(axis, run)
    axis.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    draw_yaw_profile(formation.add_subplot(grid[1]), run)
    formation.suptitle("rigid formation along the optimised route", fontsize=12)
    figures["formation"] = formation

    frames = plt.figure(figsize=(16, 10))
    draw_frames(frames, run)
    frames.suptitle("key frames of the rigid formation (same scale; dashed = previous frame)", fontsize=12)
    figures["frames"] = frames

    convergence = plt.figure(figsize=(14, 5))
    draw_convergence(convergence, run)
    convergence.suptitle(summary, fontsize=11)
    figures["convergence"] = convergence

    coupling = plt.figure(figsize=(17, 8.5))
    draw_coupling(coupling, run)
    coupling.suptitle(f"{LABELS.get(run.variant, run.variant)}: tree edges, coupling decisions and online time",
                      fontsize=12)
    figures["coupling"] = coupling

    solids = plt.figure(figsize=(18, 8.5))
    shown = min(40, len(run.cells))
    for column, (elev, azim) in enumerate(((24, -60), (8, -95))):
        axis3d = solids.add_subplot(1, 2, column + 1, projection="3d")
        draw_chain_3d(axis3d, run, 40, f"{shown} path regions (elev {elev}, azim {azim}); floor = obstacle footprints")
        axis3d.view_init(elev=elev, azim=azim)
    solids.suptitle(f"region chain in (x, y, theta) (rho = {run.planner.rho:.3f} m)", fontsize=12)
    figures["tube3d"] = solids

    for name, figure in figures.items():
        if name in ("growth", "frames"):
            figure.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=2.0)
        elif name not in ("tube3d",):
            figure.tight_layout()
    return figures


def build_comparison(runs: list[Run]) -> plt.Figure:
    figure = plt.figure(figsize=(17, 11))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.5, 1))
    axis = figure.add_subplot(grid[0, 0])
    base = runs[0]
    draw_map(axis, base.map_data)
    chains = [set(r.sequence) for r in runs]
    for run in runs:
        color = VARIANT_COLORS.get(run.variant, "k")
        for cell in run.cells:
            polygon = poly_slice(cell, 0.0)
            if polygon is not None:
                axis.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor=color, alpha=0.35, linewidth=0.8,
                                       zorder=4))
        if run.route:
            axis.plot([p.x for p in run.route], [p.y for p in run.route], color="white", linewidth=4.5, zorder=6)
            axis.plot([p.x for p in run.route], [p.y for p in run.route], color=color, linewidth=2.0, zorder=7,
                      label=f"{LABELS.get(run.variant, run.variant)}: {run.final['optimised_cost']:.3f} "
                            f"({len(run.sequence)} regions)")
    shared = len(set.intersection(*chains)) if chains else 0
    xlim = [min(view_limits(r)[0][0] for r in runs), max(view_limits(r)[0][1] for r in runs)]
    ylim = [min(view_limits(r)[1][0] for r in runs), max(view_limits(r)[1][1] for r in runs)]
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.legend(fontsize=9, loc="upper left", framealpha=0.9)
    axis.set_title(f"final region chains (outlines) and optimised routes; {shared} regions shared by all chains",
                   fontsize=10)

    axis = figure.add_subplot(grid[0, 1])
    for run in runs:
        solved = sorted([s for s in run.snapshots if s["success"]], key=lambda s: s["iteration"])
        if solved:
            color = VARIANT_COLORS.get(run.variant, "k")
            axis.step([s["online_time_s"] for s in solved], [s["optimised_cost"] for s in solved], where="post",
                      color=color, linewidth=1.8, label=f"{LABELS.get(run.variant, run.variant)}")
            axis.scatter([solved[0]["online_time_s"]], [solved[0]["optimised_cost"]], color=color, s=30, zorder=5)
    axis.set_xscale("log")
    axis.set_xlabel("online time [s] (dot = first goal)")
    axis.set_ylabel("optimised path length d_W")
    axis.grid(True, color="0.9")
    axis.legend(fontsize=8)
    axis.set_title("quality vs online time (same regions, different coupling)", fontsize=10)

    axis = figure.add_subplot(grid[1, 0])
    for run in runs:
        if len(run.arc):
            axis.plot(run.arc, run.clearance, color=VARIANT_COLORS.get(run.variant, "k"), linewidth=1.1,
                      label=f"{LABELS.get(run.variant, run.variant)}: min {run.clearance.min():.3f}")
    axis.axhline(0.0, color="0.3", linewidth=0.8)
    axis.set_xlabel("route length d_W [m]")
    axis.set_ylabel("guarded clearance [m]")
    axis.grid(True, color="0.9")
    axis.legend(fontsize=8, loc="upper right")
    axis.set_title("clearance along the optimised routes", fontsize=10)

    axis = figure.add_subplot(grid[1, 1])
    names = [p for p, _ in time_parts(runs[0])]
    left = np.zeros(len(runs))
    for k, (name, color) in enumerate(zip(names, ["tab:gray", "tab:olive", "tab:red", "tab:cyan"])):
        values = np.array([time_parts(r)[k][1] for r in runs])
        axis.barh(np.arange(len(runs)), values, left=left, color=color, height=0.55, label=name)
        left += values
    for y, total in enumerate(left):
        axis.text(total, y, f" {total:.2f} s", va="center", fontsize=9)
    axis.set_yticks(np.arange(len(runs)), [LABELS.get(r.variant, r.variant) for r in runs], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("seconds")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(True, axis="x", color="0.9")
    axis.set_title("time breakdown (single process)", fontsize=10)
    figure.suptitle(f"{base.scenario} seed {base.seed}: {base.result['regions']} shared regions after "
                    f"{base.final['iteration']} iterations", fontsize=12)
    figure.tight_layout()
    return figure


# -- outputs -------------------------------------------------------------------------------------------------------
def write_path_csv(path: Path, run: Run) -> None:
    robots = project_robot_paths(run.route, run.slots)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "cx", "cy", "theta_rad", "guarded_clearance"]
                        + [f"robot{i}_{axis}" for i in range(len(robots)) for axis in "xy"])
        for k, pose in enumerate(run.route):
            clearance = float(route_profile(run.map_data, run.slots, [(pose.x, pose.y, pose.yaw)] * 2)[2][0])
            writer.writerow([k, f"{pose.x:.6f}", f"{pose.y:.6f}", f"{pose.yaw:.6f}", f"{clearance:.6f}"]
                            + [f"{value:.6f}" for robot in robots for value in robot[k]])


def run_summary(run: Run, command: str, run_dir: Path) -> dict:
    final, first = run.final, run.result["first"]
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "git": git_revision(), "command": command,
        "run_dir": run_dir.relative_to(ROOT).as_posix() if run_dir.is_relative_to(ROOT) else str(run_dir),
        "scenario": run.scenario, "seed": run.seed, "variant": run.variant,
        "iterations": final["iteration"], "regions": run.result["regions"], "goal_regions": run.result["goal_cells"],
        "first_goal_iteration": first["iteration"] if first else None,
        "first_goal_time_s": first["online_time_s"] if first else None,
        "first_goal_cost": first["optimised_cost"] if first else None,
        "online_time_s": run.result["online_time_s"], "wall_time_s": run.wall_time,
        "raw_cost": final.get("raw_cost"), "optimised_cost": final.get("optimised_cost"),
        "chain_regions": final.get("chain_regions"),
        "min_clearance": float(run.clearance.min()) if len(run.clearance) else None,
        "straddle_route_samples": sum(bool(straddled_obstacles(p, run.slots, run.map_data)) for p in run.dense),
        "time_parts": dict(time_parts(run)), "stats": run.result["stats"],
        "solver_failures": run.result["solver_failures"],
        "config": asdict(run.planner.config),
    }


def write_report(run_dir: Path, summary: dict) -> None:
    st = summary["stats"]
    fmt = lambda v, f="{:.3f}": "-" if v is None else f.format(v)
    lines = [
        f"# 区域耦合可视化：{summary['scenario']} / seed {summary['seed']} / {LABELS.get(summary['variant'])}",
        "",
        f"- 运行时间：{summary['timestamp']}",
        f"- 代码版本：`{summary['git']}`",
        f"- 结果目录：`{summary['run_dir']}`",
        f"- 场景：{DESCRIPTIONS[summary['scenario']]}",
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
        f"| 迭代数 / 区域数 / 可连目标的区域数 | {summary['iterations']} / {summary['regions']} / "
        f"{summary['goal_regions']} |",
        f"| 首次连到目标：迭代 / 在线时间 / 优化后长度 | {fmt(summary['first_goal_iteration'], '{}')} / "
        f"{fmt(summary['first_goal_time_s'])} s / {fmt(summary['first_goal_cost'])} |",
        f"| 最终：在线代价 / 优化后长度（d_W） | {fmt(summary['raw_cost'])} / {fmt(summary['optimised_cost'])} |",
        f"| 路径区域链长度 | {fmt(summary['chain_regions'], '{}')} |",
        f"| 优化路线上稠密采样的最小 guarded clearance | {fmt(summary['min_clearance'])} m |",
        f"| 路线稠密采样中障碍夹在机器人之间的样本数 | {summary['straddle_route_samples']} |",
        f"| 在线时间（不含 checkpoint） / 含 checkpoint 与记录的总耗时 | {summary['online_time_s']:.2f} s / "
        f"{summary['wall_time_s']:.2f} s |",
        *[f"| 时间：{k} | {v:.3f} s |" for k, v in summary["time_parts"].items()],
        f"| 求解器回退 / 失败 | {summary['solver_failures']} |",
        *[f"| stats.{k} | {v if isinstance(v, int) else f'{v:.3f}'} |" for k, v in st.items()],
        "",
        "## 配置",
        "",
        "| 参数 | 值 |",
        "| --- | --- |",
        f"| 机器人半径 / 安全余量 | {ROBOT_RADIUS:.3f} m / {SAFETY_MARGIN:.3f} m |",
        *[f"| {k} | {v} |" for k, v in summary["config"].items() if k != "checkpoints"],
        "",
        "## 图",
        "",
    ]
    for name, (filename, description) in FIGURE_FILES.items():
        lines += [f"### {filename}", "", description, "", f"![{name}]({filename})", ""]
    lines += ["## 其他文件", "", "- `path.csv`：优化路线的 waypoint（cx, cy, theta, guarded_clearance）及各机器人坐标。",
              "- `summary.json`：机器可读摘要，上级目录的 `README.md` 索引由它生成。", ""]
    (run_dir / "run.md").write_text("\n".join(lines), encoding="utf-8")


def write_index(out_dir: Path) -> Path:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*/*/summary.json"))]
    lines = [
        "# 区域耦合可视化索引",
        "",
        "本文件由 `scripts/visualize_coupling.py` 自动生成。目录结构：`<场景>_seed<seed>/<变体>/`，"
        "每个场景目录下的 `0_compare.png` 把同一 seed 的两种耦合方式放在一起（同一组区域）。",
        "",
        "| 运行 | 区域 | 首解迭代 / 时间 | 在线代价 | 优化后长度 | 链长 | 最小 clearance | 在线时间 s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        relative = Path(os.path.relpath(ROOT / run["run_dir"], out_dir)).as_posix()
        first = (f"{run['first_goal_iteration']} / {run['first_goal_time_s']:.2f} s"
                 if run["first_goal_iteration"] is not None else "-")
        lines.append(f"| [{relative}]({relative}/run.md) | {run['regions']} | {first} | "
                     f"{run['raw_cost'] or float('nan'):.3f} | {run['optimised_cost'] or float('nan'):.3f} | "
                     f"{run['chain_regions'] or '-'} | {run['min_clearance'] or float('nan'):.3f} | "
                     f"{run['online_time_s']:.2f} |")
    lines.append("")
    index = out_dir / "README.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS + LEGACY)
    parser.add_argument("--variants", nargs="+", default=[A, B], choices=tuple(PLANNERS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=CouplingConfig.max_iterations)
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="tree snapshot / chain optimisation interval (growth and convergence figures)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "coupling_viz")
    args = parser.parse_args()
    python = os.path.relpath(sys.executable, ROOT) if Path(sys.executable).is_absolute() else sys.executable
    command = "MPLBACKEND=Agg " + shlex.join([python, os.path.relpath(Path(__file__).resolve(), ROOT), *sys.argv[1:]])
    out_dir = args.out_dir.resolve()
    for name in args.scenarios:
        runs = []
        for variant in args.variants:
            started = time.perf_counter()
            run = execute(name, variant, args.seed, args.iterations, args.checkpoint_every)
            run_dir = out_dir / f"{name}_seed{args.seed}" / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            figures = build_figures(run)
            for key, figure in figures.items():
                figure.savefig(run_dir / FIGURE_FILES[key][0], dpi=150)
                plt.close(figure)
            if run.route:
                write_path_csv(run_dir / "path.csv", run)
            summary = run_summary(run, command, run_dir)
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            write_report(run_dir, summary)
            runs.append(run)
            print(f"{name:8s} {variant:22s} regions={run.result['regions']} online={run.result['online_time_s']:.2f}s "
                  f"opt={run.final.get('optimised_cost')} total={time.perf_counter() - started:.1f}s -> "
                  f"{summary['run_dir']}", flush=True)
        if len(runs) > 1 and all(r.route for r in runs):
            figure = build_comparison(runs)
            figure.savefig(out_dir / f"{name}_seed{args.seed}" / "0_compare.png", dpi=150)
            plt.close(figure)
    index = write_index(out_dir)
    print(f"index {index.relative_to(ROOT) if index.is_relative_to(ROOT) else index}")


if __name__ == "__main__":
    main()
