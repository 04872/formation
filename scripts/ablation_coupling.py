#!/usr/bin/env python3
"""Ablations of the single-entry region RRT* (method/对比实验.md): q_i-launch, region proxy + post-hoc chain (B) and
no ChooseParent / Rewire, against the full method on the same scenarios and seeds.

Every variant hands its region chain to the same chain optimiser; ``L_final`` is the optimised length, the online
cost of the tree is reported next to it.  Online time excludes checkpoints (chain optimisation is timed apart).
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
from compare_coupling import DESCRIPTIONS, SCENARIOS, route_clearance, scenario
from formation.region_coupling import PLANNERS, CouplingConfig

FULL = "single_entry_rrtstar"
RAY = "single_entry_ray_launch"
VARIANTS = (FULL, f"{RAY}@0.3", f"{RAY}@0.6", f"{RAY}@1.0", "single_entry_no_rewire", "anchor_rrtstar")
"""``planner[@eta]``: the planner variant and its step ``eta`` (default ``CouplingConfig.step``)."""
LABELS = {FULL: "full (A)", f"{RAY}@0.3": "q_i ray-exit launch, eta 0.3", f"{RAY}@0.6": "q_i ray-exit launch, eta 0.6",
          f"{RAY}@1.0": "q_i ray-exit launch, eta 1.0", "single_entry_seed_launch": "q_i-launch (fixed eta from q_i)",
          "single_entry_seed_launch_covered": "q_i-launch, covered samples skipped",
          "single_entry_no_rewire": "no ChooseParent/Rewire", "anchor_rrtstar": "region proxy + post-hoc (B)",
          "single_entry_heading_hold": "heading-held expansion",
          "single_entry_lex_turn": "(J, Theta) lexicographic, eps_J 1%",
          "single_entry_lex_turn5": "(J, Theta) lexicographic, eps_J 5%",
          "single_entry_ancestor": "A + ancestor connection", "single_entry_entry_ray": "A + entry-ray launch",
          "single_entry_ancestor_entry_ray": "A + both",
          "single_entry_consistent_rewire": "A + consistent rewire",
          "single_entry_reference_parent": "A + reference-state ChooseParent",
          "single_entry_consistent": "A + consistent rewire + reference ChooseParent",
          f"{FULL}@0.3": "A, eta 0.3", f"{FULL}@1.0": "A, eta 1.0", f"{FULL}@1.5": "A, eta 1.5",
          "single_entry_adaptive_step05": "adaptive step, kappa 0.5", "single_entry_adaptive_step": "adaptive step, kappa 1.0",
          "single_entry_adaptive_step15": "adaptive step, kappa 1.5"}
COLORS = {FULL: "C3", f"{RAY}@0.3": "C9", f"{RAY}@0.6": "C2", f"{RAY}@1.0": "C5", "single_entry_seed_launch": "C7",
          "single_entry_seed_launch_covered": "C8", "single_entry_no_rewire": "C1", "anchor_rrtstar": "C0",
          "single_entry_heading_hold": "C4", "single_entry_lex_turn": "C6", "single_entry_lex_turn5": "C8",
          "single_entry_ancestor": "C0", "single_entry_entry_ray": "C2", "single_entry_ancestor_entry_ray": "C4",
          "single_entry_consistent_rewire": "C0", "single_entry_reference_parent": "C2", "single_entry_consistent": "C4",
          f"{FULL}@0.3": "C9", f"{FULL}@1.0": "C5", f"{FULL}@1.5": "C1",
          "single_entry_adaptive_step05": "C0", "single_entry_adaptive_step": "C2", "single_entry_adaptive_step15": "C4"}


def run_task(task):
    name, spec, seed, config = task
    planner_name, _, eta = spec.partition("@")
    config = replace(config, seed=seed, **({"step": float(eta)} if eta else {}))
    map_data, slots, start = scenario(name)
    result = PLANNERS[planner_name](map_data, slots, start, config).plan()
    keep = [result["first"], result["checkpoints"][-1]]
    for record in [result["first"]] + result["checkpoints"]:
        if not record:
            continue
        if any(record is k for k in keep):
            if record.get("route"):
                record["route_clearance"] = route_clearance(map_data, slots, record["route"])
        else:
            record.pop("route", None)
            record.pop("sequence", None)
    result.update(scenario=name, seed=seed, variant=spec)
    return result


def _median(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.median(values)) if values else math.nan


def records(result):
    return sorted([r for r in [result["first"]] + result["checkpoints"] if r], key=lambda r: r["online_time_s"])


def cost_at(result, limit):
    """Optimised length of the latest path found within ``limit`` online seconds (inf if none)."""
    cost = math.inf
    for record in records(result):
        if record["online_time_s"] <= limit + 1e-12 and record["success"]:
            cost = record["optimised_cost"]
    return cost


def analyze(out_dir: Path) -> None:
    results = [json.loads(line) for line in (out_dir / "runs.jsonl").open()]
    by = {(r["scenario"], r["variant"], r["seed"]): r for r in results}
    scenarios = list(dict.fromkeys(r["scenario"] for r in results))
    variants = [v for v in LABELS if any(r["variant"] == v for r in results)]
    seeds = sorted({r["seed"] for r in results})
    rows = []
    for name in scenarios:
        for variant in variants:
            runs = [by[(name, variant, s)] for s in seeds if (name, variant, s) in by]
            final = [r["checkpoints"][-1] for r in runs]
            row = dict(scenario=name, variant=variant, runs=len(runs), success=sum(f["success"] for f in final),
                       t_first=_median([r["first"]["online_time_s"] if r["first"] else math.inf for r in runs]),
                       it_first=_median([r["first"]["iteration"] if r["first"] else math.inf for r in runs]),
                       t_total=_median([r["online_time_s"] for r in runs]),
                       L_final=_median([f.get("optimised_cost", math.inf) for f in final]),
                       online_cost=_median([f.get("raw_cost", math.inf) for f in final]),
                       regions=_median([r["regions"] for r in runs]),
                       chain_regions=_median([f.get("chain_regions", math.nan) for f in final]),
                       connect_s=_median([r["stats"]["connect_time_s"] for r in runs]),
                       core_s=_median([r["stats"]["select_time_s"] + r["stats"]["steer_time_s"] for r in runs]),
                       rewires=_median([r["stats"].get("rewires", 0) for r in runs]),
                       min_clearance=min(f.get("route_clearance", math.inf) for f in final),
                       solver_failures=sum(r["solver_failures"] for r in runs))
            if variant != FULL:
                pairs = [(by[(name, FULL, s)], by[(name, variant, s)]) for s in seeds
                         if (name, FULL, s) in by and (name, variant, s) in by]
                same_it = [100 * (b["checkpoints"][-1]["optimised_cost"] / a["checkpoints"][-1]["optimised_cost"] - 1)
                           for a, b in pairs if a["checkpoints"][-1]["success"] and b["checkpoints"][-1]["success"]]
                same_t = []
                for a, b in pairs:
                    limit = min(a["online_time_s"], b["online_time_s"])
                    ca, cb = cost_at(a, limit), cost_at(b, limit)
                    same_t.append(100 * (cb / ca - 1) if math.isfinite(ca) and math.isfinite(cb)
                                  else (math.inf if math.isfinite(ca) else -math.inf if math.isfinite(cb) else math.nan))
                row.update(diff_same_it=_median(same_it), worse_same_it=sum(d > 0.1 for d in same_it),
                           better_same_it=sum(d < -0.1 for d in same_it),
                           diff_same_time=_median(same_t), worse_same_time=sum(d > 0.1 for d in same_t),
                           better_same_time=sum(d < -0.1 for d in same_t),
                           time_ratio=_median([b["online_time_s"] / a["online_time_s"] for a, b in pairs]),
                           first_time_ratio=_median([b["first"]["online_time_s"] / a["first"]["online_time_s"]
                                                     for a, b in pairs if a["first"] and b["first"]]))
            rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot(out_dir, results, scenarios, variants, seeds)


def plot(out_dir: Path, results, scenarios, variants, seeds) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = 4
    lines = math.ceil(len(scenarios) / columns)
    fig, axes = plt.subplots(lines, columns, figsize=(5.2 * columns, 4.2 * lines), squeeze=False)
    for axis, name in zip(axes.flat, scenarios):
        for variant in variants:
            runs = [r for r in results if r["scenario"] == name and r["variant"] == variant]
            limit = max(r["online_time_s"] for r in runs)
            grid = np.geomspace(0.02, limit, 60)
            # best path length found so far vs online time, median over seeds (inf until half the seeds have one)
            curve = [np.median([min([c["optimised_cost"] for c in records(r) if c["success"]
                                     and c["online_time_s"] <= t] or [math.inf]) for r in runs]) for t in grid]
            axis.plot(grid, curve, color=COLORS[variant], label=LABELS[variant], linewidth=1.6)
        axis.set(xscale="log", xlabel="online time [s]", ylabel="median best length d_W", title=name)
        finite = [l.get_ydata() for l in axis.get_lines()]
        values = np.concatenate([y[np.isfinite(y)] for y in finite])
        if len(values):
            low, high = values.min(), np.percentile(values, 90)
            axis.set_ylim(low - 0.02 * (high - low), high)
        axis.grid(alpha=0.3)
    for axis in list(axes.flat)[len(scenarios):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("anytime: best (chain-optimised) path length vs online time, median over seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "anytime.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=tuple(LABELS))
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=CouplingConfig.max_iterations)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "ablation")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        checkpoints = tuple(range(args.checkpoint_every, args.iterations, args.checkpoint_every)) + (args.iterations,)
        config = CouplingConfig(max_iterations=args.iterations, checkpoints=checkpoints)
        (args.out_dir / "experiment.json").write_text(json.dumps(
            dict(config=asdict(config), seeds=args.seeds, variants={v: LABELS[v] for v in args.variants},
                 scenarios={s: DESCRIPTIONS[s] for s in args.scenarios}), indent=2, ensure_ascii=False))
        tasks = [(s, v, seed, config) for s in args.scenarios for seed in range(args.seeds) for v in args.variants]
        random.Random(0).shuffle(tasks)
        with ProcessPoolExecutor(args.workers) as pool, (args.out_dir / "runs.jsonl").open("w") as stream:
            for k, result in enumerate(pool.map(run_task, tasks), 1):
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                if k % 50 == 0:
                    print(f"{k}/{len(tasks)} done", flush=True)
    analyze(args.out_dir)


if __name__ == "__main__":
    main()
