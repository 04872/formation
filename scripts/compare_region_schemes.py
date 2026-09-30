#!/usr/bin/env python3
"""Compare the region-guided sampling schemes of the polyhedral planner (one per commit) on equal terms.

The ``tube`` schemes run ``RegionTubeRRTPlanner`` (``cell_model = "polyhedral_tube"``: region-gap nearest,
line TubeSteer, witness overlaps) with ``RegionTubeConfig`` overrides instead of ``FrontierConfig`` ones.

Every scheme runs in its own detached git worktree (``/tmp/formation_schemes/<commit>``) with the same
maps, 1 m square formation, planner seeds and iteration budget.  T_first and N_query are measured
outside the planner (wrapping ``cells.make_cell`` and ``_store``), so versions that did not record
them are measured the same way.  Output: ``results/region_schemes/{runs.csv,summary.csv,README.md,
comparison.png}``.

Example:
    ../env-rebuilt/bin/python scripts/compare_region_schemes.py --seeds 20 --workers 24
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
WORKTREES = Path("/tmp/formation_schemes")
MAPS = ("random_circles", "single_post", "post_fence")
ROBOT_RADIUS = 0.113
SAFETY_MARGIN = 0.06
REGION_P = 0.85
SCHEMES = {
    "uniform": ("HEAD", {"region_schedule": "constant", "region_probability": 0.0}),
    "v1 multi-seed": ("f36dc0c", {}),
    "v2 outward": ("bededcf", {}),
    "v3 source-split": ("ffe0d5b", {}),
    "v4 facet": ("HEAD", {}),
    "v2 outward p=0.85": ("bededcf", {"region_schedule": "constant", "region_probability": REGION_P}),
    "v3 source-split p=0.85": ("ffe0d5b", {"region_schedule": "constant", "region_probability": REGION_P}),
    "v4 facet p=0.85": ("HEAD", {"region_schedule": "constant", "region_probability": REGION_P}),
    "tube gap": ("HEAD", {"__model": "polyhedral_tube"}),
    "tube point": ("HEAD", {"__model": "polyhedral_tube", "nearest": "point"}),
    "tube no-exact": ("HEAD", {"__model": "polyhedral_tube", "exact_overlap": False}),
    "tube extend": ("HEAD", {"__model": "polyhedral_tube", "colliding": "extend"}),
}
FIELDS = ("map", "scheme", "seed", "success", "first_iteration", "T_first_s", "N_query_first", "C_first", "C_final",
          "N_query", "lp_calls", "nodes", "plan_time_s", "history")


def worker(root: str, overrides: dict, map_name: str, seeds: list[int], iterations: int) -> None:
    sys.path.insert(0, root)
    from dataclasses import fields

    from formation import (FormationLibrary, FrontierConfig, MapBuilder, Pose2D, PostFenceConfig, RandomCirclesConfig,
                           SinglePostConfig, TubeRRTConfig, make_tube_rrt_planner)

    configs = {"random_circles": RandomCirclesConfig(seed=7, robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN),
               "single_post": SinglePostConfig(robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN),
               "post_fence": PostFenceConfig(robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN)}
    map_data = MapBuilder().build(map_name, configs[map_name])
    slots = FormationLibrary.build_default(ROBOT_RADIUS).get("square").slots * 2.0
    overrides = dict(overrides)
    model = overrides.pop("__model", "polyhedral")
    if model == "polyhedral_tube":
        from formation import RegionTubeConfig
        options = {"tube_config": RegionTubeConfig(**overrides)}
    else:
        names = {f.name for f in fields(FrontierConfig)}
        unknown = set(overrides) - names
        if unknown:
            raise ValueError(f"{root}: FrontierConfig has no {sorted(unknown)}")
        options = {"frontier_config": FrontierConfig(**overrides)}
    for seed in seeds:
        config = TubeRRTConfig(cell_model=model, seed=seed, max_iterations=iterations,
                               stop_on_first_goal=False, progress_interval=0)
        planner = make_tube_rrt_planner(map_data, slots, Pose2D(*map_data.start_xy, 0.0), config=config, **options)
        counter = {"queries": 0, "first": None}
        make_cell, store = planner.cells.make_cell, planner._store
        goal = np.asarray(planner.goal_xy, dtype=float)

        def counted(*args, **kwargs):
            counter["queries"] += 1
            return make_cell(*args, **kwargs)

        def stored(node):
            index = store(node)
            if counter["first"] is None and np.allclose((node.pose.x, node.pose.y), goal):
                counter["first"] = (time.perf_counter() - started, counter["queries"])
            return index

        planner.cells.make_cell, planner._store = counted, stored
        started = time.perf_counter()
        result = planner.plan()
        elapsed = time.perf_counter() - started
        history = [(int(it), float(cost)) for it, cost in result.cost_history]
        first = counter["first"] or (math.nan, math.nan)
        print(json.dumps({
            "map": map_name, "seed": seed, "success": int(result.success),
            "first_iteration": result.first_goal_iteration if result.success else math.nan,
            "T_first_s": first[0], "N_query_first": first[1],
            "C_first": history[0][1] if history else math.nan,
            "C_final": result.path_cost if result.success else math.nan,
            "N_query": counter["queries"], "lp_calls": (result.overlap_stats or {}).get("lp_calls", math.nan),
            "nodes": len(result.tree_nodes), "plan_time_s": elapsed,
            "history": ";".join(f"{it}:{cost:.4f}" for it, cost in history),
        }), flush=True)


def worktree(commit: str) -> Path:
    if commit == "HEAD":
        return REPO_ROOT
    path = WORKTREES / commit
    if not path.exists():
        WORKTREES.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(path), commit], check=True,
                       capture_output=True)
    return path


def run_task(task: tuple[str, str, list[int], int]) -> list[dict]:
    scheme, map_name, seeds, iterations = task
    commit, overrides = SCHEMES[scheme]
    root = worktree(commit)
    payload = json.dumps({"root": str(root), "overrides": overrides, "map": map_name, "seeds": seeds,
                          "iterations": iterations})
    env = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONPATH": str(root)}
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", payload], env=env,
                          capture_output=True, text=True, cwd=root)
    if done.returncode != 0:
        raise RuntimeError(f"{scheme} / {map_name} failed:\n{done.stderr[-3000:]}")
    return [{**json.loads(line), "scheme": scheme} for line in done.stdout.splitlines() if line.startswith("{")]


def median_iqr(values) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return tuple(np.percentile(finite, (25, 50, 75))) if len(finite) else (math.nan,) * 3


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for map_name in MAPS:
        for scheme in SCHEMES:
            group = [r for r in rows if r["map"] == map_name and r["scheme"] == scheme]
            if not group:
                continue
            entry = {"map": map_name, "scheme": scheme, "runs": len(group),
                     "success": float(np.mean([r["success"] for r in group]))}
            for metric in ("first_iteration", "T_first_s", "N_query_first", "C_first", "C_final", "N_query",
                           "lp_calls", "plan_time_s"):
                q1, med, q3 = median_iqr([r[metric] for r in group])
                entry.update({f"{metric}_q1": q1, f"{metric}_median": med, f"{metric}_q3": q3})
            summary.append(entry)
    return summary


DEFAULTS = ("uniform", "v1 multi-seed", "v2 outward", "v3 source-split", "v4 facet", "tube gap", "tube point",
            "tube no-exact", "tube extend")
TICKS = ("uniform", "v1\n多seed", "v2\n外法向", "v3\n分来源", "v4\nfacet", "tube\ngap", "tube\npoint",
         "tube\n无LP", "tube\nextend")
WITH_P85 = ("v2 outward", "v3 source-split", "v4 facet")
MAP_STYLE = {"random_circles": ("tab:blue", "o"), "single_post": ("tab:orange", "s"), "post_fence": ("tab:green", "^")}
PANELS = (
    ("first_iteration", "(a) 首解迭代数", True),
    ("N_query_first", "(b) 首解前 N_query（建 cell 次数）", True),
    ("T_first_s", "(c) T_first：首解墙钟时间 [s]", True),
    ("C_first", "(d) C_first：首解代价 [d_G]", False),
    ("C_final", "(e) C_2500：最终代价 [d_G]", False),
    ("plan_time_s", "(f) 2500 次迭代总耗时 [s]", False),
)
TEXT = """方案（同一 polyhedral cell、同一 RRT* 主体，只换 region 采样）
• uniform：不用 region，q_rand ~ U(SE(2))（goal bias 0.12），
  最近节点 steer。基线。
• v1 多 seed 预建（f36dc0c）：固定 85% 迭代取 exposed
  frontier 候选（5 个 yaw 截面的边界点），沿方向放 3 个
  步长的 seed 各建 cell，按 J = ρ_new + φ(ρ_overlap) + Δgoal
  选最好的插入 → 每次迭代约 3 次 proximity query。
• v2 统一外法向（bededcf）：p_region 混合、每次迭代 1 个
  cell；16 方向扇形求边界点，所有 exposed 边界一律沿
  外法向，从源 cell 内 steer 到边界外。
• v3 按边界来源（ffe0d5b）：区分 obstacle-limited（障碍行
  起作用，不向外扩）与 expandable（guard / yaw cap），
  seed 放在证书外 δ 处但不越过已知障碍行。
• v4 facet 几何（当前）：guard 弧沿外法向；障碍 facet 沿
  切向绕行；Δθ ∈ {{+,0,−}} 取旋转裕度最大；每个 cell 常数
  个方向（约 7 个）。
• tube（当前，另一种思路）：不做 region 采样，Tube-RRT*
  流程 q_rand → C_rand → region-gap nearest（argmin
  D − ℓ_i(u) − ℓ_rand(−u)）→ 沿直线 TubeSteer → 直线公共
  区间即 overlap 证书（不解 LP）；NearConnect / rewire 先用
  直线 / 内切圆心 witness，乐观代价可能改进时才解 LP。
  point = 点最近；无LP = 从不解 LP；extend = 碰撞 q_rand
  也朝它扩展（不丢弃）。

设置
• 实心：各方案默认（v2–v4 为 switch：首解前 p_region
  0.5、之后 0.2；v1 固定 0.85）。空心：v2–v4 统一 p=0.85，
  与 v1 的 region 比例相同。
• 1 m square 编队（ρ = 0.707 m），{seeds} 个 seed × 3 张地图，
  {iterations} 次迭代 anytime；点 = 中位数，误差棒 = 四分位。
• (g)–(i) 曲线在过半 seed 找到解后才出现。

指标
• N_query：位姿 proximity query 次数（= 建 cell 次数），
  代表碰撞检测开销；v1 每次迭代约 3 次，v2–v4 ≤ 1 次，
  tube 每次迭代 1 次（C_rand，碰撞的也算）+ steer 重试。
• d_G = Σ(‖Δc‖ + ρ|Δθ|)：路径代价，越小越好。
• 耗时为 {workers} 进程并行、任务随机交错下的测量，
  有 ±20% 负载噪声；低负载耗时见 README。uniform / v4 /
  tube 在当前提交运行（含解析内切圆、CSC 构造加速），
  v1–v3 在旧提交运行。

结论
{findings}"""


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            rows.append({k: (v if k in ("map", "scheme", "history") else float(v)) for k, v in row.items()})
    return rows


def cost_curves(rows: list[dict], grid: np.ndarray) -> np.ndarray:
    curves = np.full((len(rows), len(grid)), math.inf)
    for k, row in enumerate(rows):
        for item in filter(None, str(row["history"]).split(";")):
            it, cost = item.split(":")
            curves[k, grid >= int(it)] = float(cost)
    return curves


def findings(summary: list[dict]) -> str:
    table = {(e["map"], e["scheme"]): e for e in summary}
    if any((name, s) not in table for name in MAPS for s in ("uniform", "v1 multi-seed", "v4 facet")):
        return "（部分方案未运行）"
    lines = []
    for name in MAPS:
        u, v1, v4 = (table[(name, s)] for s in ("uniform", "v1 multi-seed", "v4 facet"))
        lines.append(f"• {name}：首解前 N_query uniform {u['N_query_first_median']:.0f} / v1 "
                     f"{v1['N_query_first_median']:.0f} / v4 {v4['N_query_first_median']:.0f}；"
                     f"\n  C_2500 {u['C_final_median']:.2f} / {v1['C_final_median']:.2f} / {v4['C_final_median']:.2f}")
    lines += [
        "• v1 首解迭代最少，但每次迭代建约 3 个 cell：首解前查询数",
        "  比 v4 多 1.5–2.5 倍，2500 次迭代总耗时约为 uniform 的 2 倍。",
        "• v2 / v3 在 post_fence 上比 uniform 更慢，p=0.85 时更明显：",
        "  栅栏边界全是障碍面，向外扩展要么被拒、要么绕远。",
        "• v4 是唯一在三张地图、两种 p 下首解都快于 uniform 的单 cell",
        "  方案，且 p 越大越快（post_fence p=0.85 首解前 44 次查询）。",
        "• 最终代价：random_circles 上所有 region 方案都比 uniform 好",
        "  0.5–0.9；single_post 上差别在 ±0.35 内；post_fence 上",
        "  uniform 最低（11.15），v4 高 0.1–0.3，v2 / v3 高 0.7–0.9。",
        "• tube gap：首解迭代与 N_query 多于 v4（碰撞 q_rand 被丢弃",
        "  但仍花一次查询），T_first 与 v4 相当；LP 调用少 4–6 倍，",
        "  低负载 2500 次迭代 0.9–1.1 s（v4 2.3–2.8 s）；C_2500 比",
        "  v4 高 0.07–0.3。不解 LP 再快 ~10%，代价差 ≤ 0.17。",
        "• region-gap nearest 优于点最近（首解迭代更少、random_",
        "  circles 代价低 0.35）；extend 首解最快但 LP 翻倍、更慢。",
    ]
    return "\n".join(lines)


def plot(rows: list[dict], iterations: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    summary = summarize(rows)
    table = {(e["map"], e["scheme"]): e for e in summary}
    figure = plt.figure(figsize=(26, 15))
    grid = figure.add_gridspec(3, 4, width_ratios=(1, 1, 1, 1.25), height_ratios=(1, 1, 0.9), wspace=0.28, hspace=0.42)
    x = np.arange(len(DEFAULTS))
    offsets = dict(zip(MAPS, (-0.24, 0.0, 0.24)))
    for index, (metric, title, log) in enumerate(PANELS):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        for name, (color, marker) in MAP_STYLE.items():
            for suffix, filled, shift in (("", True, -0.05), (" p=0.85", False, 0.05)):
                xs, med, low, high = [], [], [], []
                for k, scheme in enumerate(DEFAULTS):
                    paired = scheme in WITH_P85
                    if suffix and not paired:
                        continue
                    entry = table.get((name, scheme + suffix))
                    if entry is None:
                        continue
                    xs.append(k + offsets[name] + (shift if paired else 0.0))
                    med.append(entry[f"{metric}_median"])
                    low.append(entry[f"{metric}_median"] - entry[f"{metric}_q1"])
                    high.append(entry[f"{metric}_q3"] - entry[f"{metric}_median"])
                axis.errorbar(xs, med, yerr=(low, high), fmt=marker, color=color, ms=6, capsize=2.5, mew=1.3,
                              mfc=color if filled else "white", linestyle="none",
                              label=f"{name}{'' if filled else ' (p=0.85)'}")
        axis.set_xticks(x, TICKS, fontsize=8)
        axis.axvspan(len(DEFAULTS) - 4.5, len(DEFAULTS) - 0.5, color="tab:cyan", alpha=0.07)
        for boundary in x[:-1] + 0.5:
            axis.axvline(boundary, color="0.85", linewidth=0.8)
        if log:
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.tick_params(axis="y", which="minor", labelsize=7)
        axis.set_title(title + ("（越小越好）" if metric != "plan_time_s" else ""), fontsize=10.5)
        axis.grid(alpha=0.3, axis="y")
        if index == 0:
            axis.legend(fontsize=7.5, ncol=2, loc="upper right")
    steps = np.arange(1, iterations + 1)
    colors = {"uniform": "0.35", "v1 multi-seed": "tab:purple", "v2 outward": "tab:brown", "v3 source-split": "tab:olive",
              "v4 facet": "tab:red", "tube gap": "tab:cyan", "tube no-exact": "tab:blue"}
    for column, name in enumerate(MAPS):
        axis = figure.add_subplot(grid[2, column])
        for scheme in colors:
            group = [r for r in rows if r["map"] == name and r["scheme"] == scheme]
            if not group:
                continue
            curve = np.median(cost_curves(group, steps), axis=0)
            axis.plot(steps, np.where(np.isfinite(curve), curve, np.nan), color=colors[scheme],
                      linewidth=2.0 if scheme in ("v4 facet", "tube gap") else 1.3, label=scheme)
        axis.set_xscale("log")
        axis.set_xlabel("迭代次数（对数）", fontsize=9)
        axis.set_ylabel("中位数路径代价 [d_G]", fontsize=9)
        axis.set_title(f"({'ghi'[column]}) {name}：中位数代价收敛（默认设置）", fontsize=10)
        axis.grid(alpha=0.3)
        if column == 0:
            axis.legend(fontsize=8)
    text = figure.add_subplot(grid[:, 3])
    text.axis("off")
    seeds = min(e["runs"] for e in summary)
    text.text(0.0, 1.0, TEXT.format(seeds=seeds, iterations=iterations, workers=24, findings=findings(summary)),
              va="top", ha="left", fontsize=10, linespacing=1.5, transform=text.transAxes,
              bbox=dict(boxstyle="round", facecolor="0.97", edgecolor="0.8"))
    figure.suptitle("region 采样方案 与 Tube-RRT* 对比（polyhedral cell，1 m square 编队，三张地图）", fontsize=15)
    figure.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(figure)


def write_markdown(summary: list[dict], args, out_dir: Path, elapsed: float) -> None:
    lines = ["# region 采样方案对比", "",
             f"`scripts/compare_region_schemes.py --seeds {args.seeds} --iterations {args.iterations}`"
             + (f"，{len(summary) * args.seeds} 次运行，约 {elapsed:.0f} s。" if math.isfinite(elapsed) else "。"),
             "每个方案在对应提交的 git worktree 中运行（同地图、同 seed、同迭代数），T_first / N_query 在规划器外统一测量。"
             "表中为中位数 [四分位]。", "", "![comparison](comparison.png)", "",
             "| 地图 | 方案 | 成功率 | 首解迭代 | 首解前 N_query | T_first [s] | C_first | C_2500 | N_query | LP 调用 | 耗时 [s] |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in summary:
        def cell(metric: str, digits: int) -> str:
            return f"{e[metric + '_median']:.{digits}f} [{e[metric + '_q1']:.{digits}f}, {e[metric + '_q3']:.{digits}f}]"
        lines.append(f"| `{e['map']}` | {e['scheme']} | {e['success']:.2f} | {cell('first_iteration', 0)} | "
                     f"{cell('N_query_first', 0)} | {cell('T_first_s', 3)} | {cell('C_first', 2)} | {cell('C_final', 2)} | "
                     f"{e['N_query_median']:.0f} | {e['lp_calls_median']:.0f} | {e['plan_time_s_median']:.2f} |")
    timing = out_dir / "timing_low_load.csv"
    if timing.exists():
        lines += ["", "## 低负载耗时", "",
                  "3 进程并行、每方案 6 个 seed（`--seeds 6 --workers 3 --chunk 2 --schemes ...`），比上表的 24 进程测量"
                  "更接近单独运行的耗时。", "",
                  "| 地图 | 方案 | T_first [s] | 2500 次迭代耗时 [s] | LP 调用 | C_2500 |", "| --- | --- | --- | --- | --- | --- |"]
        with open(timing) as handle:
            for e in csv.DictReader(handle):
                lines.append(f"| `{e['map']}` | {e['scheme']} | {float(e['T_first_s_median']):.3f} | "
                             f"{float(e['plan_time_s_median']):.2f} [{float(e['plan_time_s_q1']):.2f}, "
                             f"{float(e['plan_time_s_q3']):.2f}] | {float(e['lp_calls_median']):.0f} | "
                             f"{float(e['C_final_median']):.2f} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--chunk", type=int, default=4, help="planner seeds per worker process")
    parser.add_argument("--schemes", nargs="+", choices=list(SCHEMES), default=list(SCHEMES),
                        help="run only these schemes (e.g. a low-load timing subset)")
    parser.add_argument("--plot-only", action="store_true", help="redraw from an existing runs.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "region_schemes")
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker)
        worker(job["root"], job["overrides"], job["map"], job["seeds"], job["iterations"])
        return
    if args.plot_only:
        rows = load_rows(args.out / "runs.csv")
        plot(rows, args.iterations, args.out / "comparison.png")
        write_markdown(summarize(rows), args, args.out, math.nan)
        print(f"redrew {args.out / 'comparison.png'}")
        return
    for commit in {SCHEMES[s][0] for s in args.schemes}:
        worktree(commit)
    seeds = list(range(args.seeds))
    tasks = [(scheme, map_name, seeds[i:i + args.chunk], args.iterations)
             for scheme in args.schemes for map_name in MAPS for i in range(0, len(seeds), args.chunk)]
    order = np.random.default_rng(0).permutation(len(tasks))
    tasks = [tasks[i] for i in order]
    started = time.perf_counter()
    rows: list[dict] = []
    with ThreadPoolExecutor(args.workers) as pool:
        for chunk in pool.map(run_task, tasks):
            rows.extend(chunk)
    elapsed = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "runs.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (MAPS.index(r["map"]), list(SCHEMES).index(r["scheme"]), r["seed"])))
    summary = summarize(rows)
    with open(args.out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(rows, args.iterations, args.out / "comparison.png")
    write_markdown(summary, args, args.out, elapsed)
    print(f"{len(rows)} runs in {elapsed:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
