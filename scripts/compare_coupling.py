#!/usr/bin/env python3
"""Single-entry RRT* vs proxy-cost RRT* + post-hoc chain optimisation (method/对比.md, method/方案b.md).

Both planners share the region expansion core and the random stream, so per seed they build the same regions;
what differs is the online bookkeeping (continuous entries vs proxy costs in ChooseParent / Rewire) and which region chain
is handed to the shared chain optimiser.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from formation.formation_library import FormationLibrary
from formation.map_builder import MapBuilder
from formation.map_config import BaseMapConfig, PostFenceConfig, RandomCirclesConfig
from formation.region_coupling import PLANNERS, CouplingConfig
from formation.types import Pose2D

ROBOT_RADIUS = 0.113
SAFETY_MARGIN = 0.06
VARIANTS = tuple(PLANNERS)
STRADDLE = ("forest", "mixed", "fences")
SCENARIOS = ("zigzag", "dense", "trap", "slots") + STRADDLE
LEGACY = ("gate", "clutter", "fence")
DESCRIPTIONS = {
    "zigzag": "16x10 m，三道圆柱墙，门洞上下交替（净宽约 1.35 m），必须 S 形穿过三个门；正方形编队 0.5 m",
    "dense": "20x12 m，50 个随机圆障碍（r=0.2-0.5，障碍间净距 >= 1.0 m），大量同伦类、部分缝隙只容轴对齐通过；正方形编队 0.5 m",
    "trap": "16x10 m，起点在开口背向目标的 U 形陷阱内，目标前有一片障碍；正方形编队 0.5 m",
    "slots": "16x10 m，1.5 m 长的列队编队先横穿竖墙窄缝（yaw≈0），再在下方腔室内转 90° 竖穿横墙窄缝（yaw≈±90°）",
    "forest": "16x10 m，细柱林（抖动网格间距 1.4 m，r=0.08-0.16，柱间净空 < 编队整体通过所需的 1.35 m）；"
              "正方形编队 1.0 m，可穿过的障碍半径上限 0.327 m，只能让柱子从机器人之间穿过",
    "mixed": "16x10 m，两道大圆墙（r=0.6，不可穿过），每道墙一个立有细柱的门洞（必须跨柱，路线短）和一个靠边的"
             "宽口（无需跨柱，绕远）；正方形编队 1.0 m",
    "fences": "16x10 m，两道错开半个间距的细柱栅栏（间距 1.1 m，r=0.08）中间夹一个 r=1.2 的大障碍，"
              "必须两次跨柱并绕开大障碍；正方形编队 1.0 m",
    "gate": "一面圆柱墙，唯一门洞偏离起终点连线（y=1.6，净宽约 1.35 m）；正方形编队 0.5 m",
    "clutter": "16 个随机圆障碍（r=0.25-0.45，障碍间净距 >= 1.2 m），连线附近多条同伦路线可选；正方形编队 0.5 m",
    "fence": "贯穿全图的细柱栅栏（间距 1.1 m），编队放大到 1.0 m，只能让柱子从机器人之间穿过",
}


def wall(p0, p1, gaps=(), radius=0.2, spacing=0.35):
    """Circles along the segment p0 -> p1; ``gaps`` are (arc position, clear width) openings between circle edges."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    pieces, begin = [], 0.0
    for c, w in sorted(gaps):
        pieces.append((begin, c - w / 2 - radius))
        begin = c + w / 2 + radius
    pieces.append((begin, length))
    circles = []
    for a, b in pieces:
        if b < a:
            continue
        for s in np.linspace(a, b, int(math.ceil((b - a) / spacing)) + 1):
            circles.append((tuple((p0 + (p1 - p0) * s / length).tolist()), radius))
    return circles


def scenario(name: str):
    base = dict(robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN)
    library = FormationLibrary.build_default(ROBOT_RADIUS)
    square = library.get("square").slots
    builder = MapBuilder()
    start_yaw = 0.0
    if name == "zigzag":
        config = BaseMapConfig(width_m=16.0, height_m=10.0, start_xy=(-7.0, 0.0), goal_xy=(7.0, 0.0), **base)
        circles = []
        for x, gate in ((-3.5, 3.0), (0.0, -3.0), (3.5, 3.0)):
            circles += wall((x, -5.0), (x, 5.0), [(gate + 5.0, 1.35)])
        map_data = builder._build_circles("zigzag", config, circles)
        slots = square
    elif name == "dense":
        config = RandomCirclesConfig(width_m=20.0, height_m=12.0, start_xy=(-9.0, 0.0), goal_xy=(9.0, 0.0), seed=3,
                                     obstacle_count=50, radius_min=0.2, radius_max=0.5, obstacle_clearance=1.0, **base)
        map_data = builder.build("random_circles", config)
        slots = square
    elif name == "trap":
        config = BaseMapConfig(width_m=16.0, height_m=10.0, start_xy=(-3.0, 0.0), goal_xy=(6.5, 0.0), **base)
        circles = (wall((-1.5, -2.2), (-1.5, 2.2)) + wall((-1.5, 2.2), (-5.5, 2.2))
                   + wall((-1.5, -2.2), (-5.5, -2.2)))
        for center in ((3.0, 1.2), (3.2, -1.0), (4.6, 0.1), (4.4, 2.6), (4.5, -2.4), (2.2, 3.2), (2.4, -3.3)):
            circles.append((center, 0.4))
        map_data = builder._build_circles("trap", config, circles)
        slots = square
    elif name == "slots":
        config = BaseMapConfig(width_m=16.0, height_m=10.0, start_xy=(-6.0, -2.5), goal_xy=(5.0, 3.0), **base)
        circles = wall((-8.0, 0.0), (8.0, 0.0), [(11.0, 0.85)]) + wall((-2.0, -5.0), (-2.0, -0.35), [(2.5, 0.85)])
        map_data = builder._build_circles("slots", config, circles)
        slots = library.get("column").slots
    elif name in STRADDLE:
        config = BaseMapConfig(width_m=16.0, height_m=10.0, start_xy=(-7.0, 0.0), goal_xy=(7.0, 0.0), **base)
        circles = []
        if name == "forest":
            rng = np.random.default_rng(11)
            for x in np.arange(-5.6, 5.61, 1.4):
                for y in np.arange(-4.2, 4.21, 1.4):
                    center = (float(x + rng.uniform(-0.3, 0.3)), float(y + rng.uniform(-0.3, 0.3)))
                    circles.append((center, float(rng.uniform(0.08, 0.16))))
        elif name == "mixed":
            for x, post_y, open_y in ((-2.5, 0.0, 3.6), (2.5, 0.0, -3.6)):
                circles += wall((x, -5.0), (x, 5.0), [(post_y + 5.0, 1.8), (open_y + 5.0, 1.6)], radius=0.6, spacing=0.8)
                circles.append(((x, post_y), 0.1))
        else:
            for x, offset in ((-3.0, 0.0), (3.0, 0.55)):
                circles += [((x, float(y)), 0.08) for y in np.arange(-4.95 + offset, 4.95, 1.1)]
            circles.append(((0.0, 0.0), 1.2))
        map_data = builder._build_circles(name, config, circles)
        slots = square * 2.0
    elif name == "gate":
        ys = [y for y in np.arange(-3.8, 3.81, 0.35) if abs(y - 1.6) > 0.7]
        circles = [((0.0, float(y)), 0.2) for y in ys]
        map_data = builder._build_circles("gate", BaseMapConfig(**base), circles)
        slots = square
    elif name == "clutter":
        config = RandomCirclesConfig(seed=5, obstacle_count=16, radius_min=0.25, radius_max=0.45,
                                     obstacle_clearance=1.2, **base)
        map_data = builder.build("random_circles", config)
        slots = square
    elif name == "fence":
        map_data = builder.build("post_fence", PostFenceConfig(**base))
        slots = square * 2.0
    else:
        raise ValueError(name)
    return map_data, np.asarray(slots, dtype=float), Pose2D(*map_data.start_xy, start_yaw)


def route_profile(map_data, slots, route, step=0.01):
    """Dense samples of the piecewise-linear route in (x, y, theta): d_W arc length, poses (n, 3) and guarded
    clearance of the worst robot (robot radius and safety margin subtracted)."""
    rho = float(np.max(np.linalg.norm(slots, axis=1)))
    centers = np.array([p["center_xy"] for p in map_data.obstacle_primitives]).reshape((-1, 2))
    radii = np.array([p["radius"] for p in map_data.obstacle_primitives])
    ox, oy = map_data.origin_xy
    arcs, poses, clearances, offset = [], [], [], 0.0
    for a, b in zip(route, route[1:]):
        dyaw = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
        length = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (rho * dyaw) ** 2)
        s = np.linspace(0.0, 1.0, max(2, int(math.ceil(length / step)) + 1))
        x, y, t = a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * dyaw
        c, si = np.cos(t)[:, None], np.sin(t)[:, None]
        px = x[:, None] + c * slots[None, :, 0] - si * slots[None, :, 1]
        py = y[:, None] + si * slots[None, :, 0] + c * slots[None, :, 1]
        d = np.minimum.reduce((px - ox, ox + map_data.width_m - px, py - oy, oy + map_data.height_m - py))
        if len(radii):
            d = np.minimum(d, np.min(np.hypot(px[..., None] - centers[:, 0], py[..., None] - centers[:, 1]) - radii,
                                     axis=2))
        arcs.append(offset + s * length)
        poses.append(np.column_stack((x, y, t)))
        clearances.append(d.min(axis=1))
        offset += length
    if not arcs:
        return np.zeros(0), np.zeros((0, 3)), np.zeros(0)
    return (np.concatenate(arcs), np.concatenate(poses),
            np.concatenate(clearances) - ROBOT_RADIUS - SAFETY_MARGIN)


def route_clearance(map_data, slots, route, step=0.01):
    """Minimum guarded clearance along the piecewise-linear route in (x, y, theta), densely sampled."""
    clearances = route_profile(map_data, slots, route, step)[2]
    return float(clearances.min()) if len(clearances) else math.inf


def run_one(task):
    name, variant, seed, config = task
    map_data, slots, start = scenario(name)
    planner = PLANNERS[variant](map_data, slots, start, replace(config, seed=seed))
    result = planner.plan()
    for record in [result["first"]] + result["checkpoints"]:
        if record and record.get("route"):
            record["route_clearance"] = route_clearance(map_data, slots, record["route"])
    seeds = np.round([(c.pose.x, c.pose.y, c.pose.yaw) for c in planner.cells], 9)
    result.update(scenario=name, seed=seed, region_signature=hashlib.sha1(seeds.tobytes()).hexdigest())
    return result


def flatten(result):
    rows = []
    for record in ([result["first"]] if result["first"] else []) + result["checkpoints"]:
        row = dict(scenario=result["scenario"], variant=result["variant"], seed=result["seed"], kind=record["kind"],
                   iteration=record["iteration"], regions=record["regions"], success=record["success"],
                   online_time_s=record["online_time_s"])
        if record["success"]:
            row.update(raw_cost=record["raw_cost"], optimised_cost=record["optimised_cost"],
                       chain_regions=record["chain_regions"], extract_time_s=record["extract_time_s"],
                       optimise_time_s=record["optimise_time_s"], route_clearance=record.get("route_clearance"))
        rows.append(row)
    return rows


A, B = "single_entry_rrtstar", "anchor_rrtstar"


def _median(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(values)) if values else math.nan


def _at_time(result, limit):
    """Optimised cost of the latest checkpoint whose online time is ``<= limit`` (inf if none has a path)."""
    cost = math.inf
    for record in sorted([r for r in [result["first"]] + result["checkpoints"] if r],
                         key=lambda r: r["online_time_s"]):
        if record["online_time_s"] <= limit + 1e-12 and record["success"]:
            cost = record["optimised_cost"]
    return cost


def analyze(out_dir: Path):
    results = [json.loads(line) for line in (out_dir / "runs.jsonl").open()]
    by_key = {(r["scenario"], r["seed"], r["variant"]): r for r in results}
    scenarios = list(dict.fromkeys(r["scenario"] for r in results))
    seeds = sorted({r["seed"] for r in results})
    core_rows, cost_rows, pair_rows = [], [], []
    for name in scenarios:
        pairs = [(by_key[(name, s, A)], by_key[(name, s, B)]) for s in seeds
                 if (name, s, A) in by_key and (name, s, B) in by_key]
        identical = sum(a["regions"] == b["regions"] and a["region_signature"] == b["region_signature"]
                        and a["stats"]["steer_cells"] == b["stats"]["steer_cells"] for a, b in pairs)
        st = [b["stats"] for _, b in pairs]
        tries = [s["iterations"] - s["covered_samples"] for s in st]
        added = [s["iterations"] - s["covered_samples"] - s["steer_failed"] for s in st]
        core_rows.append(dict(
            scenario=name, seeds=len(pairs), identical_region_sets=identical,
            success=sum(b["success"] for _, b in pairs),
            regions=_median([b["regions"] for _, b in pairs]),
            first_iteration=_median([b["first"]["iteration"] for _, b in pairs if b["first"]]),
            covered_sample_pct=100 * _median([s["covered_samples"] / s["iterations"] for s in st]),
            first_try_pct=100 * _median([s["steer_first_try"] / t for s, t in zip(st, tries)]),
            steer_success_pct=100 * _median([n / t for n, t in zip(added, tries)]),
            backoffs_per_region=_median([s["backoffs"] / max(n, 1) for s, n in zip(st, added)]),
            cells_per_region=_median([s["steer_cells"] / max(n, 1) for s, n in zip(st, added)]),
            core_time_s=_median([s["select_time_s"] + s["steer_time_s"] for s in st])))
        for a, b in pairs:
            fa, fb = a["first"], b["first"]
            last_a, last_b = a["checkpoints"][-1], b["checkpoints"][-1]
            row = dict(scenario=name, seed=a["seed"],
                       online_a=a["online_time_s"], online_b=b["online_time_s"],
                       connect_a=a["stats"]["connect_time_s"], connect_b=b["stats"]["connect_time_s"],
                       connect_solves_a=a["stats"]["connect_solves"], connect_solves_b=b["stats"]["connect_solves"],
                       rewires_a=a["stats"]["rewires"], repairs_a=a["stats"]["repairs"], rewires_b=b["stats"]["rewires"],
                       first_time_a=fa["online_time_s"] if fa else math.nan,
                       first_time_b=fb["online_time_s"] if fb else math.nan)
            for tag, ra, rb in [("first", fa, fb)] + [(f"it{ca['iteration']}", ca, cb)
                                                      for ca, cb in zip(a["checkpoints"], b["checkpoints"])]:
                if not (ra and rb and ra["success"] and rb["success"]):
                    continue
                row[f"{tag}_opt_a"], row[f"{tag}_opt_b"] = ra["optimised_cost"], rb["optimised_cost"]
                row[f"{tag}_raw_a"], row[f"{tag}_raw_b"] = ra["raw_cost"], rb["raw_cost"]
                row[f"{tag}_same_chain"] = ra["sequence"] == rb["sequence"]
            # Equal online time: B's whole budget vs what A had found by the same online time.
            row["equal_time_s"] = b["online_time_s"]
            row["equal_time_opt_a"] = _at_time(a, b["online_time_s"])
            row["equal_time_opt_b"] = last_b.get("optimised_cost", math.inf)
            row["path_phase_a"] = last_a.get("extract_time_s", math.nan) + last_a.get("optimise_time_s", math.nan)
            row["path_phase_b"] = last_b.get("extract_time_s", math.nan) + last_b.get("optimise_time_s", math.nan)
            row["min_clearance"] = min([r.get("route_clearance", math.inf)
                                        for r in [fa, fb] + a["checkpoints"] + b["checkpoints"] if r])
            row["solver_failures"] = a["solver_failures"] + b["solver_failures"]
            pair_rows.append(row)
        rows = [r for r in pair_rows if r["scenario"] == name]
        tags = ["first"] + [f"it{c['iteration']}" for c in pairs[0][0]["checkpoints"]] + ["equal_time"]
        for tag in tags:
            both = [r for r in rows if math.isfinite(r.get(f"{tag}_opt_a", math.nan))
                    and math.isfinite(r.get(f"{tag}_opt_b", math.nan))]
            if not both:
                continue
            diff = [100 * (r[f"{tag}_opt_a"] / r[f"{tag}_opt_b"] - 1) for r in both]
            cost_rows.append(dict(
                scenario=name, checkpoint=tag, pairs=len(both),
                opt_a_median=_median([r[f"{tag}_opt_a"] for r in both]),
                opt_b_median=_median([r[f"{tag}_opt_b"] for r in both]),
                diff_pct_median=_median(diff), diff_pct_mean=float(np.mean(diff)),
                diff_pct_p10=float(np.percentile(diff, 10)), diff_pct_p90=float(np.percentile(diff, 90)),
                a_better=sum(d < -0.1 for d in diff), b_better=sum(d > 0.1 for d in diff),
                tie=sum(abs(d) <= 0.1 for d in diff),
                same_chain=sum(bool(r.get(f"{tag}_same_chain")) for r in both) if tag != "equal_time" else None,
                raw_gap_a_pct=_median([100 * (r[f"{tag}_raw_a"] / r[f"{tag}_opt_a"] - 1) for r in both])
                if tag != "equal_time" else None,
                raw_gap_b_pct=_median([100 * (r[f"{tag}_raw_b"] / r[f"{tag}_opt_b"] - 1) for r in both])
                if tag != "equal_time" else None))
    for path, rows in (("core.csv", core_rows), ("cost.csv", cost_rows), ("pairs.csv", pair_rows)):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out_dir / path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    timing = []
    for name in scenarios:
        rows = [r for r in pair_rows if r["scenario"] == name]
        timing.append(dict(
            scenario=name,
            online_a=_median([r["online_a"] for r in rows]), online_b=_median([r["online_b"] for r in rows]),
            online_ratio=_median([r["online_a"] / r["online_b"] for r in rows]),
            connect_a=_median([r["connect_a"] for r in rows]), connect_b=_median([r["connect_b"] for r in rows]),
            connect_ratio=_median([r["connect_a"] / r["connect_b"] for r in rows]),
            connect_solves_a=_median([r["connect_solves_a"] for r in rows]),
            connect_solves_b=_median([r["connect_solves_b"] for r in rows]),
            first_time_a=_median([r["first_time_a"] for r in rows]),
            first_time_b=_median([r["first_time_b"] for r in rows]),
            first_ratio=_median([r["first_time_a"] / r["first_time_b"] for r in rows]),
            path_phase_a=_median([r["path_phase_a"] for r in rows]),
            path_phase_b=_median([r["path_phase_b"] for r in rows]),
            rewires_a=_median([r["rewires_a"] for r in rows]), repairs_a=_median([r["repairs_a"] for r in rows]),
            rewires_b=_median([r["rewires_b"] for r in rows]),
            min_clearance=min(r["min_clearance"] for r in rows),
            solver_failures=sum(r["solver_failures"] for r in rows)))
    fields = list(timing[0])
    with (out_dir / "timing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(timing)
    plot(out_dir, results, scenarios, pair_rows)
    return core_rows, cost_rows, timing


def plot(out_dir: Path, results, scenarios, pair_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors = {A: "C3", B: "C0"}
    labels = {A: "A: single-entry RRT*", B: "B: anchor-cost RRT* + post-hoc"}
    fig, axes = plt.subplots(2, len(scenarios), figsize=(6.2 * len(scenarios), 8.5), squeeze=False)
    for col, name in enumerate(scenarios):
        map_data, slots, _ = scenario(name)
        ax = axes[0, col]
        for p in map_data.obstacle_primitives:
            ax.add_patch(Circle(p["center_xy"], p["radius"], color=".35"))
        seed = 0
        for variant in (B, A):
            r = next(x for x in results if x["scenario"] == name and x["seed"] == seed and x["variant"] == variant)
            route = np.array(r["checkpoints"][-1].get("route") or [[np.nan] * 3])
            ax.plot(route[:, 0], route[:, 1], color=colors[variant], lw=2 if variant == B else 1.2,
                    ls="-" if variant == B else "--", label=f"{labels[variant]}  "
                    f"{r['checkpoints'][-1].get('optimised_cost', math.nan):.3f}")
        ax.scatter(*map_data.start_xy, c="green", zorder=5)
        ax.scatter(*map_data.goal_xy, c="orange", marker="*", s=120, zorder=5)
        ox, oy = map_data.origin_xy
        ax.set(xlim=(ox, ox + map_data.width_m), ylim=(oy, oy + map_data.height_m), aspect="equal",
               title=f"{name}: seed {seed}, final optimised routes")
        ax.legend(fontsize=7, loc="lower right")
        ax = axes[1, col]
        for variant in (A, B):
            runs = [x for x in results if x["scenario"] == name and x["variant"] == variant]
            iterations = [c["iteration"] for c in runs[0]["checkpoints"]]
            times = [_median([x["checkpoints"][k]["online_time_s"] for x in runs]) for k in range(len(iterations))]
            costs = [_median([x["checkpoints"][k].get("optimised_cost", math.nan) for x in runs])
                     for k in range(len(iterations))]
            ax.plot(times, costs, "o-", color=colors[variant], label=labels[variant])
            for t, c, it in zip(times, costs, iterations):
                if variant == B:
                    ax.annotate(str(it), (t, c), fontsize=6, xytext=(2, 4), textcoords="offset points")
        ax.set(xscale="log", xlabel="median online time [s] (labels: iteration)",
               ylabel="median optimised path length [d_W]", title=f"{name}: quality vs online time")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.2 * len(scenarios), 4), squeeze=False)
    for col, name in enumerate(scenarios):
        rows = [r for r in pair_rows if r["scenario"] == name]
        tags = ["first"] + [k[:-6] for k in rows[0] if k.startswith("it") and k.endswith("_opt_a")] + ["equal_time"]
        data = [[100 * (r[f"{t}_opt_a"] / r[f"{t}_opt_b"] - 1) for r in rows
                 if math.isfinite(r.get(f"{t}_opt_a", math.nan)) and math.isfinite(r.get(f"{t}_opt_b", math.nan))]
                for t in tags]
        ax = axes[0, col]
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(tags) + 1), tags, rotation=30, fontsize=8)
        ax.axhline(0, color="k", lw=.8)
        ax.set(ylabel="(A - B) / B  optimised length [%]", title=f"{name}: < 0 means A shorter")
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "paired_cost.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS + LEGACY)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=CouplingConfig.max_iterations)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "coupling")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args.out_dir)
        return
    checkpoints = tuple(c for c in CouplingConfig.checkpoints if c < args.iterations) + (args.iterations,)
    config = CouplingConfig(max_iterations=args.iterations, checkpoints=checkpoints)
    (args.out_dir / "experiment.json").write_text(json.dumps(
        dict(config=asdict(config), seeds=args.seeds, scenarios={s: DESCRIPTIONS[s] for s in args.scenarios},
             robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN), indent=2, ensure_ascii=False))
    tasks = [(s, v, seed, config) for s in args.scenarios for seed in range(args.seeds) for v in args.variants]
    results = []
    with ProcessPoolExecutor(args.workers) as pool, (args.out_dir / "runs.jsonl").open("w") as stream:
        for result in pool.map(run_one, tasks):
            results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            first = result["first"] or {}
            print(f"{result['scenario']:8s} seed={result['seed']:2d} {result['variant']:22s} "
                  f"regions={result['regions']:4d} online={result['online_time_s']:7.2f}s "
                  f"first_it={first.get('iteration')} final={result['checkpoints'][-1].get('optimised_cost')}",
                  flush=True)
    rows = [row for result in results for row in flatten(result)]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.out_dir / "checkpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    analyze(args.out_dir)


if __name__ == "__main__":
    main()
