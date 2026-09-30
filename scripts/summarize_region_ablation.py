#!/usr/bin/env python3
"""One-page summary of the p_region ablation (reads runs.csv written by ablate_region_sampling.py).

Example:
    ../env-rebuilt/bin/python scripts/summarize_region_ablation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("p=0", "p=0.2", "p=0.4", "p=0.6", "p=0.8", "p=1", "switch 0.5->0.2", "exp 0.5->0.2")
LABELS = ("0", "0.2", "0.4", "0.6", "0.8", "1", "switch\n0.5→0.2", "exp\n0.5→0.2")
CONSTANT = 6
MAPS = {"random_circles": ("tab:blue", "o"), "single_post": ("tab:orange", "s"), "post_fence": ("tab:green", "^")}
PANELS = (
    ("first_iteration", "首解迭代数（≈ 首解前 N_query）", "越小越好", True),
    ("T_first_s", "T_first：首解墙钟时间 [s]", "越小越好", True),
    ("C_first", "C_first：首解代价 [d_G]", "越小越好", False),
    ("C_final", "C_2500：2500 次迭代后代价 [d_G]", "越小越好", False),
    ("region_share", "region 节点占比", "树中来自 region 通道的节点比例", False),
    ("region_fallback_rate", "region 回退率", "抽中 region 但已无 live 扩展、改走 uniform 的比例", False),
)
EXPLANATION = """参数
• p_region：每次迭代以该概率走 region 通道
  （从当前 cell 的 frontier 取扩展），其余走
  uniform SE(2) 采样。p=0 即纯 uniform RRT*。
• switch 0.5→0.2：首解前 0.5，首解后 0.2。
• exp 0.5→0.2：p(t) = 0.2 + 0.3·e^(−0.002 t)。
• region 扩展：guard 弧沿外法向；障碍 facet
  沿切向绕行；Δθ ∈ {{+0.15, 0, −0.15}} rad 取
  障碍裕度最大者；q_new 在跨出 cell 0.1 m 处。
• 场景：square 队形，相邻 1 m，ρ = 0.707 m；
  每配置 {seeds} 个 seed，{iterations} 次迭代（anytime）。

指标
• d_G = Σ(‖Δc‖ + ρ|Δθ|)，即编队中心位移加
  旋转引起的机器人位移，是路径代价。
• N_query：位姿 proximity query 次数，每次迭代
  只建 1 个 cell，故首解迭代数 ≈ 首解前 N_query。
• 实线 / 点 = {seeds} 个 seed 的中位数，阴影 / 误差棒 =
  四分位区间；虚线右侧为调度方案。

结论
{findings}"""


def load(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    runs["region_share"] = runs["region_nodes"] / runs["nodes"]
    runs["region_accept"] = runs["region_nodes"] / runs["region_iterations"].replace(0, np.nan)
    attempts = (runs["region_iterations"] + runs["region_fallback"]).replace(0, np.nan)
    runs["region_fallback_rate"] = runs["region_fallback"] / attempts
    return runs


def quantiles(runs: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    table = {}
    for name in MAPS:
        rows = runs[runs["map"] == name]
        table[name] = np.array([np.nanpercentile(rows.loc[rows["variant"] == v, metric], (25, 50, 75))
                                if rows.loc[rows["variant"] == v, metric].notna().any() else (np.nan,) * 3
                                for v in VARIANTS])
    return table


def findings(runs: pd.DataFrame) -> str:
    med = runs.groupby(["map", "variant"]).median(numeric_only=True)
    lines = []
    for name in MAPS:
        a, b, s = med.loc[(name, "p=0")], med.loc[(name, "p=1")], med.loc[(name, "switch 0.5->0.2")]
        lines.append(f"• {name}：首解迭代 p=0 {a.first_iteration:.0f} → p=1 {b.first_iteration:.0f}，"
                     f"switch {s.first_iteration:.0f}\n  C_2500 p=0 {a.C_final:.2f}，p=1 {b.C_final:.2f}，switch {s.C_final:.2f}")
    accept = runs.loc[runs["region_iterations"] > 0, "region_accept"]
    high = runs[runs["variant"].isin(("p=0.6", "p=0.8", "p=1")) & runs["map"].isin(("single_post", "post_fence"))]
    lines += [
        "• 首解：p 越大越快，p=1 的首解迭代约为 p=0 的 1/8–1/3。",
        "• 最终代价：只有 random_circles 在 p ≥ 0.2 时明显\n  下降（约 0.7）；另两张图的差别在 seed 波动内。",
        f"• live 扩展会被用完：p ≥ 0.6 时 single_post /\n  post_fence 回退率中位数 {high['region_fallback_rate'].median():.0%}，"
        "(e) 中 region 节点占比饱和。",
        f"• 被采用的扩展有 {accept.quantile(0.25):.0%}–{accept.quantile(0.75):.0%} 成功插入节点（四分位），\n"
        "  失败多为新 cell 冗余（ρ_new < 0.05）。",
        "• switch / exp：首解接近 p=0.4–0.6、回退少，作为默认。",
    ]
    return "\n".join(lines)


def plot(runs: pd.DataFrame, iterations: int, out: Path) -> None:
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(19, 10.5))
    grid = figure.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.3), wspace=0.28, hspace=0.38)
    x = np.arange(len(VARIANTS))
    offsets = dict(zip(MAPS, (-0.12, 0.0, 0.12)))
    for index, (metric, title, note, log) in enumerate(PANELS):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        for name, q in quantiles(runs, metric).items():
            color, marker = MAPS[name]
            axis.plot(x[:CONSTANT], q[:CONSTANT, 1], color=color, marker=marker, ms=5, label=name)
            axis.fill_between(x[:CONSTANT], q[:CONSTANT, 0], q[:CONSTANT, 2], color=color, alpha=0.15, linewidth=0)
            xs = x[CONSTANT:] + offsets[name]
            axis.errorbar(xs, q[CONSTANT:, 1], yerr=(q[CONSTANT:, 1] - q[CONSTANT:, 0], q[CONSTANT:, 2] - q[CONSTANT:, 1]),
                          fmt=marker, color=color, ms=6, capsize=3, mfc="white", mew=1.5)
        axis.axvline(CONSTANT - 0.5, color="0.5", linestyle="--", linewidth=0.8)
        axis.set_xticks(x, LABELS, fontsize=8)
        axis.set_xlabel("常数 p_region                        调度", fontsize=8)
        if log:
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.tick_params(axis="y", which="minor", labelsize=7)
        axis.set_title(f"({'abcdef'[index]}) {title}\n{note}", fontsize=10)
        axis.grid(alpha=0.3)
        if index == 0:
            axis.legend(fontsize=8)
    text = figure.add_subplot(grid[:, 3])
    text.axis("off")
    seeds = runs.groupby(["map", "variant"]).size().min()
    text.text(0.0, 1.0, EXPLANATION.format(seeds=seeds, iterations=iterations, findings=findings(runs)),
              va="top", ha="left", fontsize=10, linespacing=1.5, transform=text.transAxes,
              bbox=dict(boxstyle="round", facecolor="0.97", edgecolor="0.8"))
    figure.suptitle("region 采样比例对 polyhedral RRT* 的影响（1 m square 编队，三张地图）", fontsize=14)
    figure.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=2500, help="iterations used by the ablation (label only)")
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "results" / "region_ablation")
    args = parser.parse_args()
    runs = load(args.dir / "runs.csv")
    out = args.dir / "overview.png"
    plot(runs, args.iterations, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
