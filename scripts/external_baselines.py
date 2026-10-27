#!/usr/bin/env python3
"""External baselines (method/对比实验.md): RRT* and BIT* on the same rigid-formation problem as the region planner.

Stages:
  pilot    parameter sweep (RRT* eta, BIT* batch size) on held-out seeds; picks one setting per scenario and planner
  main     the picked settings on the evaluation seeds (same seeds as results/ablation)
  analyze  tables and anytime curves against the full region planner (A rows of results/ablation/runs.jsonl)

Baselines run on a wall-clock budget; checkpoints (path extraction and shortcutting) are excluded from online time,
as the chain optimisation is for the region planner.  ``L`` of a baseline is the shortcut length, the raw tree path
length is reported next to it.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ablation_coupling import cost_at, records
from compare_coupling import DESCRIPTIONS, SCENARIOS, STRADDLE, route_clearance, scenario
from formation.baselines import BASELINES, BaselineConfig

OURS = "single_entry_rrtstar"
SWEEP = {"rrtstar": ("step", (0.3, 0.6, 1.0)), "bitstar": ("batch_size", (50, 100, 200))}
PILOT_SEEDS = tuple(range(100, 110))
SELECTION_TIMES = (1.0, 2.0, 5.0, 10.0)
LABELS = {OURS: "ours (single-entry region RRT*)", "rrtstar": "RRT*", "bitstar": "BIT*"}
COLORS = {OURS: "C3", "rrtstar": "C2", "bitstar": "C0"}
BENCHMARKS = {"common": tuple(s for s in SCENARIOS if s not in STRADDLE), "gap": STRADDLE}


def run_task(task):
    name, planner, value, seed, config = task
    key = SWEEP[planner][0]
    config = replace(config, seed=seed, **{key: type(getattr(config, key))(value)})
    map_data, slots, start = scenario(name)
    result = BASELINES[planner](map_data, slots, start, config).plan()
    final = result["checkpoints"][-1]
    if final.get("route"):
        final["route_clearance"] = route_clearance(map_data, slots, final["route"])
    result.update(scenario=name, seed=seed, variant=planner, parameter={key: value})
    return result


def execute(tasks, path: Path, workers: int):
    random.Random(0).shuffle(tasks)
    with ProcessPoolExecutor(workers) as pool, path.open("w") as stream:
        for k, result in enumerate(pool.map(run_task, tasks), 1):
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if k % 25 == 0:
                print(f"{k}/{len(tasks)} done", flush=True)


def _median(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.median(values)) if values else math.nan


def pick(out_dir: Path) -> dict:
    """Per scenario and planner, the setting with the lowest mean (over SELECTION_TIMES) of the median best length."""
    runs = [json.loads(line) for line in (out_dir / "pilot.jsonl").open()]
    choice, rows = {}, []
    for name in dict.fromkeys(r["scenario"] for r in runs):
        for planner, (key, values) in SWEEP.items():
            scores = {}
            for value in values:
                group = [r for r in runs if r["scenario"] == name and r["variant"] == planner
                         and r["parameter"][key] == value]
                medians = [float(np.median([best_at(r, t) for r in group])) for t in SELECTION_TIMES]
                finite = [m if math.isfinite(m) else 1e3 for m in medians]
                scores[value] = float(np.mean(finite))
                rows.append(dict(scenario=name, planner=planner, **{key: value},
                                 **{f"median_best_{t:g}s": m for t, m in zip(SELECTION_TIMES, medians)}))
            choice.setdefault(name, {})[planner] = min(scores, key=scores.get)
    (out_dir / "pilot_choice.json").write_text(json.dumps(choice, indent=2))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out_dir / "pilot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return choice


def best_at(result, limit):
    """Best (post-processed) length found within ``limit`` online seconds, inf if none."""
    values = [r["optimised_cost"] for r in records(result) if r["success"] and r["online_time_s"] <= limit + 1e-12]
    return min(values) if values else math.inf


def analyze(out_dir: Path, ours_path: Path) -> None:
    base = [json.loads(line) for line in (out_dir / "runs.jsonl").open()]
    ours = [r for r in map(json.loads, ours_path.open()) if r["variant"] == OURS]
    results = ours + base
    by = {(r["scenario"], r["variant"], r["seed"]): r for r in results}
    scenarios = [s for s in SCENARIOS if any(r["scenario"] == s for r in base)]
    seeds = sorted({r["seed"] for r in base})
    choice = json.loads((out_dir / "pilot_choice.json").read_text())
    rows = []
    for name in scenarios:
        for variant in (OURS, *SWEEP):
            runs = [by[(name, variant, s)] for s in seeds if (name, variant, s) in by]
            final = [r["checkpoints"][-1] for r in runs]
            row = dict(benchmark=next(b for b, ss in BENCHMARKS.items() if name in ss), scenario=name,
                       planner=variant, parameter=json.dumps(choice[name].get(variant)) if variant != OURS else "",
                       runs=len(runs), success=sum(f["success"] for f in final),
                       t_first=_median([r["first"]["online_time_s"] if r["first"] else math.inf for r in runs]),
                       t_total=_median([r["online_time_s"] for r in runs]),
                       L_final=_median([best_at(r, math.inf) for r in runs]),
                       raw_final=_median([f.get("raw_cost", math.inf) for f in final]),
                       min_clearance=min(f.get("route_clearance", math.inf) for f in final),
                       states_checked=_median([r["stats"].get("states_checked", math.nan) for r in runs]),
                       vertices=_median([r["regions"] for r in runs]))
            if variant != OURS:
                pairs = [(by[(name, OURS, s)], by[(name, variant, s)]) for s in seeds
                         if (name, OURS, s) in by and (name, variant, s) in by]
                limit = [a["online_time_s"] for a, _ in pairs]
                at_ours = [best_at(b, t) for (_, b), t in zip(pairs, limit)]
                mine = [best_at(a, math.inf) for a, _ in pairs]
                diff = [100 * (b / a - 1) if math.isfinite(b) else math.inf for a, b in zip(mine, at_ours)]
                diff_full = [100 * (best_at(b, math.inf) / a - 1) if math.isfinite(best_at(b, math.inf)) else math.inf
                             for a, (_, b) in zip(mine, pairs)]
                row.update(success_at_ours_time=sum(math.isfinite(v) for v in at_ours),
                           L_at_ours_time=_median(at_ours), diff_at_ours_time=_median(diff),
                           worse_at_ours_time=sum(d > 0.1 for d in diff), better_at_ours_time=sum(d < -0.1 for d in diff),
                           diff_at_budget=_median(diff_full), worse_at_budget=sum(d > 0.1 for d in diff_full),
                           better_at_budget=sum(d < -0.1 for d in diff_full),
                           first_time_ratio=_median([b["first"]["online_time_s"] / a["first"]["online_time_s"]
                                                     if b["first"] else math.inf for a, b in pairs if a["first"]]))
            rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot(out_dir, results, scenarios, seeds)


def plot(out_dir: Path, results, scenarios, seeds) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = 4
    lines = math.ceil(len(scenarios) / columns)
    fig, axes = plt.subplots(lines, columns, figsize=(5.2 * columns, 4.2 * lines), squeeze=False)
    for axis, name in zip(axes.flat, scenarios):
        top = []
        for variant in (OURS, *SWEEP):
            runs = [r for r in results if r["scenario"] == name and r["variant"] == variant and r["seed"] in seeds]
            limit = max(r["online_time_s"] for r in runs)
            grid = np.geomspace(0.01, limit, 80)
            curve = np.array([np.median([best_at(r, t) for r in runs]) for t in grid])
            axis.plot(grid, curve, color=COLORS[variant], linewidth=1.8, label=LABELS[variant])
            if variant != OURS:
                raw = np.array([np.median([min([c["raw_cost"] for c in records(r) if c["success"]
                                                and c["online_time_s"] <= t] or [math.inf]) for r in runs])
                                for t in grid])
                axis.plot(grid, raw, color=COLORS[variant], linewidth=1.0, linestyle="--", alpha=0.6)
            top += [v for v in curve if math.isfinite(v)]
        if top:
            low, high = min(top), np.percentile(top, 85)
            axis.set_ylim(low - 0.03 * (high - low), high + 0.03 * (high - low))
        axis.set(xscale="log", xlabel="online time [s]", ylabel="median best length d_W",
                 title=f"{name} ({'gap' if name in STRADDLE else 'common'})")
        axis.grid(alpha=0.3)
    for axis in list(axes.flat)[len(scenarios):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("best path length vs online time, median over seeds (baselines: solid = shortcut, dashed = raw tree "
                 "path; ours: chain-optimised)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "anytime.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "main", "analyze"))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external")
    parser.add_argument("--ours", type=Path, default=ROOT / "results" / "ablation" / "runs.jsonl")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = tuple(float(t) for t in np.round(np.geomspace(0.02, args.budget, 40), 4))
    config = BaselineConfig(time_budget=args.budget, checkpoints=checkpoints)
    if args.stage == "pilot":
        tasks = [(s, p, v, seed, config) for s in args.scenarios for p, (_, values) in SWEEP.items()
                 for v in values for seed in PILOT_SEEDS]
        execute(tasks, args.out_dir / "pilot.jsonl", args.workers)
        print(json.dumps(pick(args.out_dir), indent=1))
    elif args.stage == "main":
        choice = json.loads((args.out_dir / "pilot_choice.json").read_text())
        (args.out_dir / "experiment.json").write_text(json.dumps(
            dict(config=asdict(config), seeds=args.seeds, choice=choice, pilot_seeds=PILOT_SEEDS,
                 selection_times=SELECTION_TIMES, scenarios={s: DESCRIPTIONS[s] for s in args.scenarios}),
            indent=2, ensure_ascii=False))
        tasks = [(s, p, choice[s][p], seed, config) for s in args.scenarios for p in SWEEP for seed in range(args.seeds)]
        execute(tasks, args.out_dir / "runs.jsonl", args.workers)
        analyze(args.out_dir, args.ours)
    else:
        analyze(args.out_dir, args.ours)


if __name__ == "__main__":
    main()
