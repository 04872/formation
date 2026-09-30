"""p_region ablation of the polyhedral region/uniform RRT* (1 m square formation).

For every map, sampling schedule and planner seed one anytime run of ``--iterations`` iterations is made and
T_first, C_first, C_final, N_query (pose proximity queries = cells built, overall and before the first
solution) and robot-obstacle distance pairs are recorded.  Writes ``runs.csv``, ``summary.csv``, ``README.md``
and ``ablation.png`` / ``convergence.png`` into the output directory.

    ../env-rebuilt/bin/python scripts/ablate_region_sampling.py --seeds 10 --workers 16
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from formation import (
    FormationLibrary,
    FrontierConfig,
    MapBuilder,
    Pose2D,
    PostFenceConfig,
    RandomCirclesConfig,
    SinglePostConfig,
    TubeRRTConfig,
    make_tube_rrt_planner,
)

ROBOT_RADIUS = 0.113
SAFETY_MARGIN = 0.06
MAPS = ("random_circles", "single_post", "post_fence")
CONSTANT_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
METRICS = ("success", "T_first_s", "C_first", "C_final", "N_query_first", "N_query", "pair_queries",
           "first_iteration", "nodes", "region_nodes", "plan_time_s")


def schedules() -> dict[str, FrontierConfig]:
    variants = {f"p={level:g}": FrontierConfig(region_schedule="constant", region_probability=level)
                for level in CONSTANT_LEVELS}
    variants["switch 0.5->0.2"] = FrontierConfig(region_schedule="switch", region_before=0.5, region_after=0.2)
    variants["exp 0.5->0.2"] = FrontierConfig(region_schedule="exp", region_max=0.5, region_min=0.2,
                                              region_decay=0.002)
    return variants


def build_map(name: str):
    configs = {"random_circles": RandomCirclesConfig(seed=7, robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN),
               "single_post": SinglePostConfig(robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN),
               "post_fence": PostFenceConfig(robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN)}
    return MapBuilder().build(name, configs[name])


def run_one(job: tuple[str, str, int, int]) -> dict:
    map_name, variant, seed, iterations = job
    map_data = build_map(map_name)
    slots = FormationLibrary.build_default(ROBOT_RADIUS).get("square").slots * 2.0
    config = TubeRRTConfig(cell_model="polyhedral", seed=seed, max_iterations=iterations, stop_on_first_goal=False,
                           progress_interval=0)
    planner = make_tube_rrt_planner(map_data, slots, Pose2D(*map_data.start_xy, 0.0), config=config,
                                    frontier_config=schedules()[variant])
    started = time.perf_counter()
    result = planner.plan()
    elapsed = time.perf_counter() - started
    stats = result.overlap_stats
    history = [(int(it), float(cost)) for it, cost in result.cost_history]
    return {
        "map": map_name, "variant": variant, "seed": seed, "success": int(result.success),
        "T_first_s": stats["first_goal_time_s"], "C_first": stats["first_goal_cost"],
        "C_final": result.path_cost if result.success else math.nan,
        "N_query_first": stats["first_goal_pose_queries"] if result.success else math.nan,
        "N_query": stats["pose_queries"], "pair_queries": stats["pair_queries"],
        "first_iteration": result.first_goal_iteration if result.success else math.nan,
        "nodes": len(result.tree_nodes), "region_nodes": stats["region_nodes"],
        "region_iterations": stats["region_iterations"], "region_fallback": stats["region_fallback"],
        "rejected_redundant": stats["rejected_redundant"], "rejected_no_progress": stats["rejected_no_progress"],
        "plan_time_s": elapsed, "history": ";".join(f"{it}:{cost:.4f}" for it, cost in history),
    }


def cost_curve(history: str, iterations: int, grid: np.ndarray) -> np.ndarray:
    curve = np.full(len(grid), math.nan)
    for item in filter(None, history.split(";")):
        it, cost = item.split(":")
        curve[grid >= int(it)] = float(cost)
    return curve


def summarize(rows: list[dict], variants: list[str]) -> list[dict]:
    summary = []
    for map_name in MAPS:
        for variant in variants:
            group = [row for row in rows if row["map"] == map_name and row["variant"] == variant]
            if not group:
                continue
            entry = {"map": map_name, "variant": variant, "runs": len(group)}
            for metric in METRICS:
                values = np.asarray([row[metric] for row in group], dtype=float)
                finite = values[np.isfinite(values)]
                entry[f"{metric}_mean"] = float(finite.mean()) if len(finite) else math.nan
                entry[f"{metric}_median"] = float(np.median(finite)) if len(finite) else math.nan
                entry[f"{metric}_std"] = float(finite.std()) if len(finite) else math.nan
            summary.append(entry)
    return summary


def plot(rows: list[dict], variants: list[str], iterations: int, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("T_first_s", "T_first [s]"), ("C_first", "C_first (d_G)"), ("C_final", f"C_{iterations} (d_G)"),
              ("N_query_first", "N_query before first solution"))
    figure, axes = plt.subplots(len(MAPS), len(panels), figsize=(20, 4.2 * len(MAPS)), squeeze=False)
    positions = np.arange(len(variants))
    for r, map_name in enumerate(MAPS):
        for c, (metric, label) in enumerate(panels):
            axis = axes[r, c]
            data = []
            for variant in variants:
                values = np.asarray([row[metric] for row in rows if row["map"] == map_name and row["variant"] == variant],
                                    dtype=float)
                data.append(values[np.isfinite(values)])
            axis.boxplot(data, positions=positions, widths=0.6, showfliers=True)
            for x, values in zip(positions, data):
                axis.scatter(np.full(len(values), x) + np.random.default_rng(0).uniform(-0.15, 0.15, len(values)),
                             values, s=8, alpha=0.5, color="tab:blue", zorder=3)
            axis.set_xticks(positions, variants, rotation=35, ha="right", fontsize=8)
            axis.set_ylabel(label)
            axis.grid(True, axis="y", color="0.9")
            if r == 0:
                axis.set_title(label, fontsize=11)
        axes[r, 0].annotate(map_name, (-0.32, 0.5), xycoords="axes fraction", rotation=90, va="center", fontsize=12)
    figure.suptitle("p_region ablation (1 m square, polyhedral region/uniform RRT*); boxes over planner seeds", fontsize=13)
    figure.tight_layout()
    figure.savefig(out_dir / "ablation.png", dpi=130)
    plt.close(figure)

    grid = np.arange(1, iterations + 1)
    figure, axes = plt.subplots(1, len(MAPS), figsize=(18, 5), squeeze=False)
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.95, len(CONSTANT_LEVELS)))
    styles = {f"p={level:g}": dict(color=color) for level, color in zip(CONSTANT_LEVELS, colors)}
    styles["switch 0.5->0.2"] = dict(color="tab:red", linestyle="--")
    styles["exp 0.5->0.2"] = dict(color="tab:orange", linestyle=":")
    for axis, map_name in zip(axes[0], MAPS):
        for variant in variants:
            curves = np.asarray([cost_curve(row["history"], iterations, grid) for row in rows
                                 if row["map"] == map_name and row["variant"] == variant])
            if not len(curves):
                continue
            median = np.median(np.where(np.isfinite(curves), curves, math.inf), axis=0)
            median[~np.isfinite(median)] = np.nan
            axis.plot(grid, median, label=variant, linewidth=1.4, **styles.get(variant, {}))
        axis.set_xscale("log")
        axis.set_xlabel("iteration")
        axis.set_ylabel("median best cost over seeds (unsolved = inf)")
        axis.set_title(map_name)
        axis.grid(True, color="0.9")
        axis.legend(fontsize=8)
    figure.suptitle("anytime convergence per p_region schedule", fontsize=13)
    figure.tight_layout()
    figure.savefig(out_dir / "convergence.png", dpi=130)
    plt.close(figure)


def write_markdown(summary: list[dict], args, out_dir: Path, elapsed: float) -> None:
    def cell(entry: dict, metric: str, digits: int) -> str:
        mean, std = entry[f"{metric}_mean"], entry[f"{metric}_std"]
        return "-" if not math.isfinite(mean) else f"{mean:.{digits}f} ± {std:.{digits}f}"

    lines = [
        "# p_region 消融：polyhedral region/uniform RRT*",
        "",
        f"- 编队：square，相邻机器人 1 m（rho = 0.707 m），机器人半径 {ROBOT_RADIUS} m，安全余量 {SAFETY_MARGIN} m",
        f"- 每个配置：{args.seeds} 个 planner seed（0..{args.seeds - 1}），anytime {args.iterations} 次迭代；"
        f"random_circles 地图 seed 7",
        "- p=<p>：常数 p_region；switch：首解前 0.5、之后 0.2；exp：0.2 + 0.3 e^{-0.002 t}",
        "- T_first：首解墙钟时间；C_first / C_final：首解 / 最终 d_G 路径代价；N_query：位姿 proximity 查询次数"
        "（每次查询返回所有 robot-obstacle 距离，= 构造的 cell 数），pair：robot-obstacle 距离对总数",
        f"- 总耗时 {elapsed:.0f} s；复现：`../env-rebuilt/bin/python scripts/ablate_region_sampling.py "
        f"--seeds {args.seeds} --iterations {args.iterations}`",
        "",
        "均值 ± 标准差（T_first / C_first / N_query_first 只统计成功的 seed）。",
        "",
    ]
    for map_name in MAPS:
        lines += [f"## {map_name}", "",
                  "| 调度 | 成功率 | T_first [s] | 首解迭代 | N_query 首解前 | C_first | "
                  f"C_{args.iterations} | N_query 总 | pair 查询 | region 节点 | 节点 | 时间 [s] |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for entry in (e for e in summary if e["map"] == map_name):
            lines.append(
                f"| {entry['variant']} | {entry['success_mean']:.2f} | {cell(entry, 'T_first_s', 3)} | "
                f"{cell(entry, 'first_iteration', 0)} | {cell(entry, 'N_query_first', 0)} | {cell(entry, 'C_first', 2)} | "
                f"{cell(entry, 'C_final', 2)} | {cell(entry, 'N_query', 0)} | {cell(entry, 'pair_queries', 0)} | "
                f"{cell(entry, 'region_nodes', 0)} | {cell(entry, 'nodes', 0)} | {cell(entry, 'plan_time_s', 2)} |")
        lines.append("")
    lines += ["## 图", "", "![ablation](ablation.png)", "", "![convergence](convergence.png)", ""]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--maps", nargs="+", default=list(MAPS), choices=MAPS)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "region_ablation")
    args = parser.parse_args()
    variants = list(schedules())
    jobs = [(map_name, variant, seed, args.iterations) for map_name in args.maps for variant in variants
            for seed in range(args.seeds)]
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, row in enumerate(pool.map(run_one, jobs, chunksize=1), start=1):
            rows.append(row)
            if count % 20 == 0 or count == len(jobs):
                print(f"{count}/{len(jobs)} runs, {time.perf_counter() - started:.0f} s", flush=True)
    elapsed = time.perf_counter() - started
    with (args.out / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, variants)
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(rows, variants, args.iterations, args.out)
    write_markdown(summary, args, args.out, elapsed)
    print(f"wrote {args.out}/README.md")


if __name__ == "__main__":
    main()
