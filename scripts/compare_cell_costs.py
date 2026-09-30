#!/usr/bin/env python3
"""Path quality of every cell model under one metric (orientation, first / second order, polyhedral, tube).

Each planner reports the cost in its own metric (the chart cells use ``max_i ||u + J a_i phi||``, which is
smaller than ``d_G``), so the final path of every run is re-measured the same way:

* ``L_G``: ``sum ||dc|| + rho |dtheta|`` over the certified route (every segment is linear in (x, y, theta));
* robot path length: every robot's travelled distance along the densely interpolated route (mean / max);
* gap: smallest robot-surface-to-obstacle-surface distance along the dense route (walls included).

Output: ``results/cell_costs/{runs.csv,summary.csv,README.md,comparison.png}``.

Example:
    ../env-rebuilt/bin/python scripts/compare_cell_costs.py --seeds 20 --workers 24
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPS = ("random_circles", "single_post", "post_fence")
MODELS = ("orientation", "first_order", "second_order", "polyhedral", "polyhedral_tube")
LABELS = {"orientation": "orientation", "first_order": "一阶", "second_order": "二阶", "polyhedral": "polyhedral\nv4 facet",
          "polyhedral_tube": "polyhedral\nTube-RRT*"}
ROBOT_RADIUS = 0.113
SAFETY_MARGIN = 0.06
DENSE_STEP = 0.01
FIELDS = ("map", "model", "seed", "success", "first_iteration", "reported_cost", "L_G", "robot_mean", "robot_max",
          "gap", "path_nodes", "plan_time_s")


def measure(poses, slots: np.ndarray, rho: float, map_data) -> dict:
    xyz = np.array([(p.x, p.y, p.yaw) for p in poses])
    delta = np.diff(xyz, axis=0)
    delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
    lengths = np.hypot(delta[:, 0], delta[:, 1]) + rho * np.abs(delta[:, 2])
    dense = [xyz[:1]]
    for start, step, length in zip(xyz[:-1], delta, lengths):
        count = max(int(math.ceil(length / DENSE_STEP)), 1)
        dense.append(start + np.linspace(0.0, 1.0, count + 1)[1:, None] * step)
    dense = np.concatenate(dense)
    c, s = np.cos(dense[:, 2]), np.sin(dense[:, 2])
    robots = np.stack((dense[:, None, 0] + c[:, None] * slots[:, 0] - s[:, None] * slots[:, 1],
                       dense[:, None, 1] + s[:, None] * slots[:, 0] + c[:, None] * slots[:, 1]), axis=-1)
    travelled = np.linalg.norm(np.diff(robots, axis=0), axis=-1).sum(axis=0)
    ox, oy = map_data.origin_xy
    gap = np.minimum.reduce([robots[..., 0] - ox, ox + map_data.width_m - robots[..., 0],
                             robots[..., 1] - oy, oy + map_data.height_m - robots[..., 1]])
    for primitive in map_data.obstacle_primitives:
        cx, cy = primitive["center_xy"]
        gap = np.minimum(gap, np.hypot(robots[..., 0] - cx, robots[..., 1] - cy) - primitive["radius"])
    return {"L_G": float(lengths.sum()), "robot_mean": float(travelled.mean()), "robot_max": float(travelled.max()),
            "gap": float(gap.min()) - ROBOT_RADIUS}


def worker(model: str, map_name: str, seeds: list[int], iterations: int) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from formation import (FormationLibrary, MapBuilder, Pose2D, PostFenceConfig, RandomCirclesConfig,
                           SinglePostConfig, TubeRRTConfig, make_tube_rrt_planner)

    configs = {"random_circles": RandomCirclesConfig(seed=7, robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN),
               "single_post": SinglePostConfig(robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN),
               "post_fence": PostFenceConfig(robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN)}
    map_data = MapBuilder().build(map_name, configs[map_name])
    slots = FormationLibrary.build_default(ROBOT_RADIUS).get("square").slots * 2.0
    rho = float(np.max(np.linalg.norm(slots, axis=1)))
    for seed in seeds:
        config = TubeRRTConfig(cell_model=model, seed=seed, max_iterations=iterations, stop_on_first_goal=False,
                               progress_interval=0)
        planner = make_tube_rrt_planner(map_data, slots, Pose2D(*map_data.start_xy, 0.0), config=config)
        started = time.perf_counter()
        result = planner.plan()
        row = {"map": map_name, "model": model, "seed": seed, "success": int(result.success),
               "plan_time_s": time.perf_counter() - started}
        if result.success:
            row.update(first_iteration=result.first_goal_iteration, reported_cost=result.path_cost,
                       path_nodes=len(result.path_nodes), **measure(result.path_poses, slots, rho, map_data))
        print(json.dumps(row), flush=True)


def run_task(task: tuple[str, str, list[int], int]) -> list[dict]:
    model, map_name, seeds, iterations = task
    payload = json.dumps({"model": model, "map": map_name, "seeds": seeds, "iterations": iterations})
    env = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", payload], env=env,
                          capture_output=True, text=True, cwd=REPO_ROOT)
    if done.returncode != 0:
        raise RuntimeError(f"{model} / {map_name} failed:\n{done.stderr[-3000:]}")
    return [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for map_name in MAPS:
        for model in MODELS:
            group = [r for r in rows if r["map"] == map_name and r["model"] == model]
            if not group:
                continue
            entry = {"map": map_name, "model": model, "runs": len(group),
                     "success": float(np.mean([r["success"] for r in group]))}
            for metric in FIELDS[4:]:
                values = np.asarray([r.get(metric, math.nan) for r in group], dtype=float)
                values = values[np.isfinite(values)]
                q1, med, q3 = np.percentile(values, (25, 50, 75)) if len(values) else (math.nan,) * 3
                entry.update({f"{metric}_q1": q1, f"{metric}_median": med, f"{metric}_q3": q3})
            summary.append(entry)
    return summary


PANELS = (
    ("L_G", "(a) 路径长度 L_G = Σ‖Δc‖ + ρ|Δθ| [m]（统一口径，越小越好）"),
    ("robot_mean", "(b) 机器人平均行驶距离 [m]（越小越好）"),
    ("robot_max", "(c) 机器人最长行驶距离 [m]（越小越好）"),
    ("gap", "(d) 沿路线最小真实间隙 [m]（机器人表面到障碍表面）"),
    ("reported_cost", "(e) 各规划器自报代价（各自度量，不可直接比）"),
    ("plan_time_s", "(f) 2500 次迭代耗时 [s]（24 进程并行）"),
)
MAP_STYLE = {"random_circles": ("tab:blue", "o"), "single_post": ("tab:orange", "s"), "post_fence": ("tab:green", "^")}


def plot(summary: list[dict], seeds: int, iterations: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    table = {(e["map"], e["model"]): e for e in summary}
    figure = plt.figure(figsize=(20, 10.5))
    grid = figure.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.1), wspace=0.3, hspace=0.35)
    x = np.arange(len(MODELS))
    for index, (metric, title) in enumerate(PANELS):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        for shift, (name, (color, marker)) in zip((-0.2, 0.0, 0.2), MAP_STYLE.items()):
            entries = [table[(name, m)] for m in MODELS]
            med = np.array([e[f"{metric}_median"] for e in entries])
            err = (med - [e[f"{metric}_q1"] for e in entries], [e[f"{metric}_q3"] for e in entries] - med)
            axis.errorbar(x + shift, med, yerr=err, fmt=marker, color=color, ms=6, capsize=2.5, linestyle="none",
                          label=name)
        axis.set_xticks(x, [LABELS[m] for m in MODELS], fontsize=8.5)
        for boundary in x[:-1] + 0.5:
            axis.axvline(boundary, color="0.85", linewidth=0.8)
        if metric == "gap":
            axis.axhline(SAFETY_MARGIN, color="0.4", linestyle=":", linewidth=1.0, label="safety_margin 0.06")
        if metric == "plan_time_s":
            axis.set_yscale("log")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3, axis="y")
        if index in (0, 3):
            axis.legend(fontsize=8)
    text = figure.add_subplot(grid[:, 3])
    text.axis("off")
    text.text(0.0, 1.0, TEXT.format(seeds=seeds, iterations=iterations, findings=findings(table)), va="top",
              ha="left", fontsize=10, linespacing=1.5, transform=text.transAxes,
              bbox=dict(boxstyle="round", facecolor="0.97", edgecolor="0.8"))
    figure.suptitle("路径质量对比：orientation / 一阶 / 二阶 / polyhedral v4 / polyhedral Tube-RRT*（1 m square 编队）",
                    fontsize=14)
    figure.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(figure)


TEXT = """设置
• 1 m square 编队（ρ = 0.707 m），{seeds} 个 planner seed ×
  3 张地图，{iterations} 次迭代 anytime，各 cell 用默认参数；
  点 = 中位数，误差棒 = 四分位。
• 各规划器的自报代价度量不同：orientation / polyhedral
  用 d_G = ‖Δc‖ + ρ|Δθ|，一阶 / 二阶用 chart 范数
  max_i ‖u + J a_i φ‖（≤ d_G）。所以 (a)–(d) 都在最终
  认证路线（节点 + portal 折线）上统一重算。

指标
• L_G：路线的 d_G 长度（平移 + ρ × 转角）。
• 机器人行驶距离：路线按 0.01 m 稠密插值后每个机器人
  的实际轨迹长度，平均 / 最大。
• 最小间隙：机器人表面到障碍 / 墙表面的最小距离，
  规划时要求 ≥ safety_margin（0.06，虚线）。

结论
{findings}"""


def findings(table: dict) -> str:
    lines = []
    for name in MAPS:
        values = " / ".join(f"{table[(name, m)]['L_G_median']:.2f}" for m in MODELS)
        lines.append(f"• {name} L_G：{values}")
    lines.append("  （顺序：orientation / 一阶 / 二阶 / v4 / tube）")
    return "\n".join(lines + FINDINGS)


FINDINGS = [
    "• L_G 最短是 v4（11.68 / 11.30 / 11.43）；tube 比 v4 长",
    "  0.30 / 0.24 / 0.08，比一阶 / 二阶短 0.5–0.9，比",
    "  orientation 短 0.5–1.55。一阶与二阶几乎相同。",
    "• 一阶 / 二阶自报代价比 L_G 低 0.1–0.2：chart 范数",
    "  max_i ‖u + J a_i φ‖ 不把平移和转动相加。",
    "• 机器人实际行驶距离：tube 11.02 / 10.67 / 10.67，",
    "  与 v4 持平或更短（v4 11.03 / 10.79 / 10.87）；orientation",
    "  路线把“原地转”和“平移”分开，行驶距离 = L_G，最长。",
    "• 首解迭代：tube 104 / 64 / 74，v4 52 / 37 / 57，",
    "  chart cell 与 orientation 130–250。",
    "• 最小间隙都 ≥ 0.1 m；post_fence 上 polyhedral 两种约",
    "  0.16，其余约 0.10。",
    "• 耗时（24 进程并行）：一阶 0.5–0.6 s 最快，tube",
    "  1.2–1.4 s，v4 2.6–3.1 s；低负载下 tube 0.74–0.86 s，一阶",
    "  0.32 s、二阶 0.49 s、orientation 0.6 s。",
]


def write_markdown(summary: list[dict], args, out_dir: Path, elapsed: float) -> None:
    lines = ["# 各 cell 模型的路径质量对比", "",
             f"`scripts/compare_cell_costs.py --seeds {args.seeds} --iterations {args.iterations}`"
             + (f"，{len(summary) * args.seeds} 次运行，约 {elapsed:.0f} s。" if math.isfinite(elapsed) else "。")
             + "每条最终路线在同一口径下重算；表中为中位数 [四分位]。", "", "![comparison](comparison.png)", "",
             "| 地图 | cell | 成功率 | 首解迭代 | 自报代价 | L_G [m] | 机器人平均 / 最长行驶 [m] | 最小间隙 [m] | 耗时 [s] |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in summary:
        def cell(metric: str, digits: int = 2) -> str:
            return f"{e[metric + '_median']:.{digits}f} [{e[metric + '_q1']:.{digits}f}, {e[metric + '_q3']:.{digits}f}]"
        lines.append(f"| `{e['map']}` | {e['model']} | {e['success']:.2f} | {e['first_iteration_median']:.0f} | "
                     f"{e['reported_cost_median']:.2f} | {cell('L_G')} | {e['robot_mean_median']:.2f} / "
                     f"{e['robot_max_median']:.2f} | {cell('gap', 3)} | {e['plan_time_s_median']:.2f} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def load_rows(path: Path) -> list[dict]:
    with open(path) as handle:
        return [{k: (v if k in ("map", "model") else float(v) if v != "" else math.nan) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "cell_costs")
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker)
        worker(job["model"], job["map"], job["seeds"], job["iterations"])
        return
    elapsed = math.nan
    if args.plot_only:
        rows = load_rows(args.out / "runs.csv")
    else:
        seeds = list(range(args.seeds))
        tasks = [(model, map_name, seeds[i:i + args.chunk], args.iterations)
                 for model in MODELS for map_name in MAPS for i in range(0, len(seeds), args.chunk)]
        tasks = [tasks[i] for i in np.random.default_rng(0).permutation(len(tasks))]
        started = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            rows = [row for chunk in pool.map(run_task, tasks) for row in chunk]
        elapsed = time.perf_counter() - started
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "runs.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, FIELDS, restval="")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (MAPS.index(r["map"]), MODELS.index(r["model"]), r["seed"])))
    summary = summarize(rows)
    with open(args.out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(summary, int(min(e["runs"] for e in summary)), args.iterations, args.out / "comparison.png")
    write_markdown(summary, args, args.out, elapsed)
    print(f"{len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
