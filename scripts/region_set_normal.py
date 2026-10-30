#!/usr/bin/env python3
"""Counterfactual: A's seeds of iteration 6000, cells rebuilt with the normal-aware rotation bound, then the same
offline region-sequence search and joint chain optimiser as ``region_set_search`` (planner unchanged).

The old-cell numbers come from ``results/region_set_search/runs.jsonl`` (same A runs, deterministic); ``--check``
re-runs the old search on the first seed of each scenario to confirm that.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_coupling import route_clearance, scenario
from formation.region_coupling import CouplingConfig, NormalTurnRRTStar, SingleEntryRRTStar
from region_set_search import ITERATIONS, search


def rebuilt(planner, map_data, slots, start, seed):
    """A ``NormalTurnRRTStar`` holding the normal-aware cells of ``planner``'s seeds (not planned)."""
    other = NormalTurnRRTStar(map_data, slots, start, CouplingConfig(seed=seed))
    margin = other.config.overlap_margin
    for cell in planner.cells:
        index = other._add_cell(other.model.make_cell(cell.pose), None)
        if other.geo.goal_yaw(other.cells[index], other.goal_xy, cell.pose.yaw, margin) is not None:
            other.goal_cells.append(index)
    return other


def run(task):
    name, seed, check = task
    map_data, slots, start = scenario(name)
    planner = SingleEntryRRTStar(map_data, slots, start, CouplingConfig(seed=seed, checkpoints=(ITERATIONS,)))
    result = planner.plan()
    row = dict(scenario=name, seed=seed, A=result["checkpoints"][-1]["optimised_cost"], regions=len(planner.cells))
    if check:
        row["old_check"] = search(planner)["iterate"]
    other = rebuilt(planner, map_data, slots, start, seed)
    row["valid_new"] = sum(c.valid for c in other.cells)
    row["goal_cells_old"], row["goal_cells_new"] = len(planner.goal_cells), len(other.goal_cells)
    out = search(other)
    route = out.pop("route")
    row.update({f"new_{k}": v for k, v in out.items() if k != "history"})
    row["new_clearance"] = route_clearance(map_data, slots, route) if route else None
    return row


def main():
    out = ROOT / "results" / "region_set_normal"
    out.mkdir(parents=True, exist_ok=True)
    scenarios = sys.argv[1].split(",") if len(sys.argv) > 1 else ["forest", "slots"]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    old = {(r["scenario"], r["seed"]): r for r in map(json.loads, (ROOT / "results" / "region_set_search" / "runs.jsonl").open())}
    tasks = [(s, seed, seed == 0) for s in scenarios for seed in range(seeds)]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(run, tasks))
    with (out / "runs.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for name in scenarios:
        g = [r for r in rows if r["scenario"] == name]
        for r in g:
            o = old[(name, r["seed"])]
            r.update(bit=o["bit"], old_iterate=o["iterate"], old_A=o["A"])
        checks = [(r["old_check"], r["old_iterate"]) for r in g if "old_check" in r]
        med = lambda k: float(np.median([r[k] for r in g]))
        rel = lambda a, b: float(np.median([100 * (r[a] / r[b] - 1) for r in g]))
        print(f"{name}: A identical {all(r['A'] == r['old_A'] for r in g)}; old search reproduced "
              f"{all(abs(a - b) < 1e-9 for a, b in checks)}; goal regions {med('goal_cells_old'):.0f} -> {med('goal_cells_new'):.0f}; "
              f"overlap pairs new {med('new_pairs'):.0f}; min clearance {min(r['new_clearance'] for r in g):.3f}; "
              f"search time {med('new_search_time'):.1f}s")
        print(f"   A {med('A'):.3f}  old cells {med('old_iterate'):.3f}  new cells {med('new_iterate'):.3f}  BIT* {med('bit'):.3f}")
        print(f"   new vs old cells {rel('new_iterate', 'old_iterate'):+.2f}% (better {sum(r['new_iterate'] < r['old_iterate'] * 0.999 for r in g)}, "
              f"worse {sum(r['new_iterate'] > r['old_iterate'] * 1.001 for r in g)});  vs BIT*: old {rel('old_iterate', 'bit'):+.2f}%  "
              f"new {rel('new_iterate', 'bit'):+.2f}%  (A {rel('A', 'bit'):+.2f}%)")


if __name__ == "__main__":
    main()
