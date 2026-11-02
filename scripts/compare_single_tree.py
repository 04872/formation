#!/usr/bin/env python3
"""Serial, paired actual-wall-clock comparison against the current A@0.3.

Checkpoints and chain optimisation consume the budget. Only completed outputs
whose completion time is <= the requested budget count as available solutions.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/formation-matplotlib")

import argparse
import csv
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_coupling import route_clearance, scenario
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar
from formation.single_tree_rrtstar import SingleTreeConfig, SingleTreeRRTStar


def initialize_a(p):
    cell = p.model.make_cell(p.start)
    if not cell.valid:
        raise ValueError("start is in collision")
    p._add_cell(cell, None)
    p._init_root()
    if p.geo.goal_yaw(cell, p.goal_xy, p.start.yaw, p.config.overlap_margin) is not None:
        p.goal_cells.append(0)


def step_a(p):
    sample = p._sample()
    started = time.perf_counter()
    parent = p._select(sample)
    p.stats["select_time_s"] += time.perf_counter() - started
    cell, portal, _ = p._steer(parent, sample)
    if cell is not None:
        index = p._add_cell(cell, parent)
        started, solves = time.perf_counter(), p.geo.solves
        p._connect(index, parent, portal)
        p.stats["connect_time_s"] += time.perf_counter() - started
        p.stats["connect_solves"] += p.geo.solves - solves
        if p.geo.goal_yaw(cell, p.goal_xy, cell.pose.yaw, p.config.overlap_margin) is not None:
            p.goal_cells.append(index)


def checkpoint(p, is_single, iteration, start):
    record = p._checkpoint(iteration, time.perf_counter() - start)
    if is_single and record["success"]:
        began = time.perf_counter()
        # A and the new tree receive the identical post-hoc chain optimiser.
        solved = p.geo.chain(p.start, p.goal_xy, [p.cells[i] for i in record["owners"]],
                             p.config.overlap_margin * (1.0 - 1e-6))
        record["optimise_time_s"] = time.perf_counter() - began
        if solved is None:
            record.update(success=False, chain_solver_failed=True)
        else:
            record["tree_route"] = record["route"]
            record["optimised_cost"] = solved[0]
            record["route"] = [(q.x, q.y, q.yaw) for q in solved[1]]
    if record["success"]:
        # Validation is charged to wall-clock too. No uncharged final solve.
        record["route_clearance"] = route_clearance(p.map_data, p.slots, record["route"])
        if record.get("tree_route"):
            record["tree_route_clearance"] = route_clearance(p.map_data, p.slots, record["tree_route"])
        record["success"] = (math.isfinite(record["optimised_cost"])
                             and record["route_clearance"] >= p.config.safety_distance - 1e-6
                             and record.get("tree_route_clearance", math.inf) >= p.config.safety_distance - 1e-6)
        if not record["success"]:
            record["validation_failed"] = True
    record.update(wall_time_s=time.perf_counter() - start, nodes=len(p.parent),
                  solver_failures=p.geo.failures, stats=dict(p.stats))
    return record


def run_one(name, variant, seed, maximum, interval, radius):
    is_single = variant == "single_tree"
    config = (SingleTreeConfig(seed=seed, step=0.3, neighbor_radius=radius, goal_margin=0.02)
              if is_single else CouplingConfig(seed=seed, step=0.3))
    map_data, slots, start_pose = scenario(name)
    start = time.perf_counter()
    p = (SingleTreeRRTStar if is_single else SingleEntryRRTStar)(map_data, slots, start_pose, config)
    if is_single:
        p.initialize()
    else:
        initialize_a(p)
    iteration, records, first_seen, next_at = 0, [], None, interval
    while time.perf_counter() - start < maximum:
        (p.step() if is_single else step_a(p))
        iteration += 1
        p.stats["iterations"] = iteration
        elapsed = time.perf_counter() - start
        reachable_goal = bool(p.goal_nodes if is_single else p.goal_cells)
        if reachable_goal and first_seen is None:
            first_seen = elapsed
            rec = checkpoint(p, is_single, iteration, start)
            rec["kind"] = "first"
            records.append(rec)
        if elapsed >= next_at:
            rec = checkpoint(p, is_single, iteration, start)
            rec["kind"] = "checkpoint"
            records.append(rec)
            next_at = (math.floor((time.perf_counter() - start) / interval) + 1) * interval
    return dict(scenario=name, variant=variant, seed=seed, config=asdict(config), records=records,
                first_seen_s=first_seen, actual_wall_s=time.perf_counter() - start,
                iterations=iteration, nodes=len(p.parent), certificates=len(p.cells),
                solver_failures=p.geo.failures, stats=dict(p.stats))


def med(values):
    return float(np.median(values)) if values else None


def available(run, budget):
    return [r for r in run.get("records", []) if r["wall_time_s"] <= budget]


def summarize(runs, budgets, out):
    rows = []
    for name in sorted({r["scenario"] for r in runs}):
        for budget in budgets:
            for variant in ("A", "single_tree"):
                group = [r for r in runs if r["scenario"] == name and r["variant"] == variant]
                stats, successes = [], []
                for run in group:
                    recs = available(run, budget)
                    valid = [r for r in recs if r["success"]]
                    last = recs[-1] if recs else {}
                    if valid:
                        first = valid[0]
                        best = min(valid, key=lambda r: r["optimised_cost"])
                        current = valid[-1]
                        successes.append((run, first, best, current))
                    stats.append(last)
                row = dict(scenario=name, budget_s=budget, variant=variant, runs=len(group),
                           success=len(successes), errors=sum("error" in r for r in group),
                           first_seen_s=med([r["first_seen_s"] for r in group
                                             if r.get("first_seen_s") is not None and r["first_seen_s"] <= budget]),
                           first_s=med([x[1]["wall_time_s"] for x in successes]),
                           first_cost=med([x[1]["optimised_cost"] for x in successes]),
                           first_raw_cost=med([x[1]["raw_cost"] for x in successes]),
                           best_raw_cost=med([min(r["raw_cost"] for r in available(x[0], budget) if r["success"])
                                              for x in successes]),
                           best_cost=med([x[2]["optimised_cost"] for x in successes]),
                           current_cost=med([x[3]["optimised_cost"] for x in successes]),
                           current_raw_cost=med([x[3]["raw_cost"] for x in successes]),
                           improvement_pct=med([100*(1-x[2]["optimised_cost"]/x[1]["optimised_cost"]) for x in successes]),
                           raw_improvement_pct=med([100*(1-min(r["raw_cost"] for r in available(x[0], budget)
                                                              if r["success"])/x[1]["raw_cost"]) for x in successes]),
                           nodes=med([r["nodes"] for r in stats if "nodes" in r]),
                           certificates=med([r["regions"] for r in stats if "regions" in r]),
                           observed_wall_s=med([r["wall_time_s"] for r in stats if "wall_time_s" in r]),
                           actual_run_wall_s=med([r["actual_wall_s"] for r in group if "actual_wall_s" in r]),
                           checkpoint_total_s=med([sum(r["wall_time_s"]-r["online_time_s"]
                                                        for r in available(run, budget)) for run in group]),
                           chain_total_s=med([sum(r.get("optimise_time_s", 0.) for r in available(run, budget))
                                             for run in group]),
                           min_clearance=min([x[2]["route_clearance"] for x in successes], default=None),
                           solver_failures=sum(r.get("solver_failures", 0) for r in group))
                for key in ("select_time_s", "coverage_time_s", "steer_time_s", "connect_time_s", "propagation_time_s"):
                    row[key] = med([r["stats"][key] for r in stats if key in r.get("stats", {})])
                rows.append(row)
    paired, pairs = [], []
    lookup = {(r["scenario"], r["variant"], r["seed"]): r for r in runs}
    for name in sorted({r["scenario"] for r in runs}):
        for budget in budgets:
            changes, raw_changes = [], []
            for seed in sorted({r["seed"] for r in runs}):
                groups = [[q for q in available(lookup.get((name, v, seed), {}), budget) if q["success"]]
                          for v in ("A", "single_tree")]
                if all(groups):
                    costs = [min(r["optimised_cost"] for r in g) for g in groups]
                    changes.append(100*(costs[1]/costs[0]-1))
                    raw = [min(r["raw_cost"] for r in g) for g in groups]
                    raw_changes.append(100*(raw[1]/raw[0]-1))
                    pairs.append(dict(scenario=name, budget_s=budget, seed=seed,
                                      A_raw=raw[0], single_tree_raw=raw[1], raw_change_pct=raw_changes[-1],
                                      A_optimised=costs[0], single_tree_optimised=costs[1],
                                      optimised_change_pct=changes[-1]))
            paired.append(dict(scenario=name, budget_s=budget, common_success=len(changes),
                               single_tree_vs_A_pct=med(changes), better=sum(c < -0.1 for c in changes),
                               worse=sum(c > 0.1 for c in changes),
                               raw_single_tree_vs_A_pct=med(raw_changes),
                               raw_better=sum(c < -0.1 for c in raw_changes),
                               raw_worse=sum(c > 0.1 for c in raw_changes)))
    for filename, data in (("summary.csv", rows), ("paired.csv", paired), ("paired_runs.csv", pairs)):
        if not data:
            continue
        with (out / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out / "summary.json").write_text(json.dumps(dict(summary=rows, paired=paired), indent=2))
    if pairs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = sorted({p["scenario"] for p in pairs})
        fig, axes = plt.subplots(1, len(names), figsize=(6*len(names), 3.7), squeeze=False)
        for ax, name in zip(axes[0], names):
            for budget, marker in zip(budgets, ("o", "s", "^", "D")):
                group = [p for p in pairs if p["scenario"] == name and p["budget_s"] == budget]
                ax.plot([p["seed"] for p in group], [p["raw_change_pct"] for p in group],
                        marker=marker, linestyle="--", label=f"{budget:g}s raw")
                ax.plot([p["seed"] for p in group], [p["optimised_change_pct"] for p in group],
                        marker=marker, label=f"{budget:g}s + chain")
            ax.axhline(0, color="black", linewidth=.8)
            ax.set(title=name, xlabel="Matched seed", ylabel="Single tree relative to A (%)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "paired_cost_changes.png", dpi=160)
        plt.close(fig)
    print(json.dumps(dict(summary=rows, paired=paired), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=["forest", "slots"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budgets", type=float, nargs="+", default=[2., 8.])
    parser.add_argument("--checkpoint-interval", type=float, default=0.25)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "single_tree_reference")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        manifest = dict(command=sys.argv, python=sys.executable, platform=platform.platform(),
                        scenarios=args.scenarios, seeds=list(range(args.seeds)), budgets=args.budgets,
                        radius=args.radius, checkpoint_interval=args.checkpoint_interval,
                        timing="actual wall-clock including initialization, checkpoints, chain optimization and validation",
                        baseline="SingleEntryRRTStar / original PolyhedralCellModel / step .3 / goal margin .02",
                        new="fixed radius all candidates / dynamic cover / goal margin .02 for matched comparison")
        (args.out_dir / "experiment.json").write_text(json.dumps(manifest, indent=2))
        with (args.out_dir / "runs.jsonl").open("w") as stream:
            for name in args.scenarios:
                for seed in range(args.seeds):
                    order = ("A", "single_tree") if seed % 2 == 0 else ("single_tree", "A")
                    for variant in order:
                        try:
                            run = run_one(name, variant, seed, max(args.budgets), args.checkpoint_interval, args.radius)
                        except Exception:
                            run = dict(scenario=name, variant=variant, seed=seed, error=traceback.format_exc(), records=[])
                        stream.write(json.dumps(run) + "\n")
                        stream.flush()
                        print(name, seed, variant, run.get("nodes"), run.get("certificates"), run.get("error", "ok"), flush=True)
    runs = [json.loads(line) for line in (args.out_dir / "runs.jsonl").read_text().splitlines()]
    summarize(runs, args.budgets, args.out_dir)


if __name__ == "__main__":
    main()
