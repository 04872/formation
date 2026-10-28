#!/usr/bin/env python3
"""Does the cell form itself rule out the BIT* corridor?

1. Cells seeded on the BIT* (shortcut) path every ``spacing`` in d_W: all valid?  consecutive delta-overlap?  is the
   path covered?  and the chain optimiser's length on that region sequence vs the BIT* length.
2. The full method's tree (same seed): coverage of the BIT* path by the tree cells, and by the same cells rebuilt with
   the seed position kept and the seed yaw replaced by the yaw of the nearest BIT* path point.
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
from compare_coupling import SCENARIOS, route_profile, scenario
from formation.region_coupling import CouplingConfig, PLANNERS
from formation.types import Pose2D

SPACINGS = (0.1, 0.25, 0.5)
NEAR = 2.0
"""Only tree cells whose seed is within this xy distance of the BIT* path are rebuilt / tested for coverage."""


def bit_route(name, seed):
    for line in (ROOT / "results" / "external" / "runs.jsonl").open():
        r = json.loads(line)
        if r["scenario"] == name and r["seed"] == seed and r["variant"] == "bitstar":
            final = r["checkpoints"][-1]
            return final["route"], final["optimised_cost"]
    raise KeyError((name, seed))


def covered(model, cells, points):
    hit = np.zeros(len(points), dtype=bool)
    for cell in cells:
        if cell.valid:
            hit |= model.slack_many(cell, points) >= 0.0
    return hit


def run(task):
    name, seed = task
    map_data, slots, start = scenario(name)
    route, bit_length = bit_route(name, seed)
    arc, poses, _ = route_profile(map_data, slots, route, step=0.02)
    planner = PLANNERS["single_entry_rrtstar"](map_data, slots, start, CouplingConfig(seed=seed, checkpoints=(6000,)))
    model, geo, margin = planner.model, planner.geo, planner.config.overlap_margin
    row = dict(scenario=name, seed=seed, bit_length=bit_length)
    for spacing in SPACINGS:
        marks = np.unique(np.concatenate((np.arange(0.0, arc[-1], spacing), [arc[-1]])))
        idx = np.minimum(np.searchsorted(arc, marks), len(poses) - 1)
        cells = [model.make_cell(Pose2D(*poses[i])) for i in idx]
        valid = sum(c.valid for c in cells)
        overlaps = [geo.max_margin(a, b)[0] for a, b in zip(cells, cells[1:]) if a.valid and b.valid]
        broken = sum(t < margin for t in overlaps)
        result = dict(cells=len(cells), valid=valid, broken_overlaps=broken,
                      min_overlap=min(overlaps) if overlaps else math.nan,
                      coverage=float(covered(model, cells, poses).mean()))
        if valid == len(cells) and broken == 0:
            solved = geo.chain(start, map_data.goal_xy, cells, margin * (1.0 - 1e-6))
            result["chain_length"] = solved[0] if solved else math.nan
        row[f"bit_cells@{spacing:g}"] = result
    planner.plan()
    near = [c for c in planner.cells
            if np.min(np.hypot(poses[:, 0] - c.pose.x, poses[:, 1] - c.pose.y)) < NEAR]
    rebuilt = []
    for c in near:
        k = int(np.argmin(np.hypot(poses[:, 0] - c.pose.x, poses[:, 1] - c.pose.y)))
        rebuilt.append(model.make_cell(Pose2D(c.pose.x, c.pose.y, float(poses[k, 2]))))
    tree_hit, yaw_hit = covered(model, near, poses), covered(model, rebuilt, poses)
    row.update(tree_cells_near=len(near), tree_coverage=float(tree_hit.mean()),
               bit_yaw_valid=sum(c.valid for c in rebuilt), bit_yaw_coverage=float(yaw_hit.mean()),
               union_coverage=float((tree_hit | yaw_hit).mean()))
    return row


def main():
    out = ROOT / "results" / "bit_cell_coverage"
    out.mkdir(parents=True, exist_ok=True)
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tasks = [(s, seed) for s in SCENARIOS for seed in range(seeds)]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(run, tasks))
    with (out / "runs.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for name in SCENARIOS:
        g = [r for r in rows if r["scenario"] == name]
        line = f"{name:7s} BIT* L {np.median([r['bit_length'] for r in g]):.3f}"
        for spacing in SPACINGS:
            k = f"bit_cells@{spacing:g}"
            ok = [r for r in g if r[k]["valid"] == r[k]["cells"] and r[k]["broken_overlaps"] == 0]
            chain = [100 * (r[k]["chain_length"] / r["bit_length"] - 1) for r in ok if not math.isnan(r[k].get("chain_length", math.nan))]
            line += (f" | @{spacing:g}: all-valid&overlap {len(ok)}/{len(g)} cover {np.median([r[k]['coverage'] for r in g]):.3f}"
                     f" chain-vs-BIT* {np.median(chain) if chain else math.nan:+.2f}%")
        print(line)
        print(f"        tree cover {np.median([r['tree_coverage'] for r in g]):.3f}  same positions + BIT* yaw cover "
              f"{np.median([r['bit_yaw_coverage'] for r in g]):.3f} (valid {np.median([r['bit_yaw_valid'] / max(1, r['tree_cells_near']) for r in g]):.2f})"
              f"  union {np.median([r['union_coverage'] for r in g]):.3f}")


if __name__ == "__main__":
    main()
