#!/usr/bin/env python3
"""Two-stage path phase on the full method's final region chain: stage 1 = chain optimiser (J_W*), stage 2 = min total
turning on the same chain with J_W <= (1 + eps) J_W* (eps = inf: the minimum turning the chain admits), or with the
stage-1 translations held."""
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
from compare_coupling import SCENARIOS, route_clearance, scenario
from formation.region_coupling import CouplingConfig, PLANNERS

SLACKS = (0.0, 0.002, 0.01, 0.05, 1e3)


def turning(route) -> float:
    q = np.array(route)
    return float(np.degrees(np.abs(np.remainder(np.diff(q[:, 2]) + math.pi, 2 * math.pi) - math.pi).sum()))


def run(task):
    name, seed = task
    map_data, slots, start = scenario(name)
    config = CouplingConfig(seed=seed, checkpoints=(6000,), turn_slacks=SLACKS, turn_fixed_xy=True)
    final = PLANNERS["single_entry_rrtstar"](map_data, slots, start, config).plan()["checkpoints"][-1]
    row = dict(scenario=name, seed=seed, J=final["optimised_cost"], turn=turning(final["route"]))
    for key, post in final["post"].items():
        row[key] = dict(J=post["cost"], turn=turning(post["route"]) if post["route"] else math.nan,
                        clearance=route_clearance(map_data, slots, post["route"]) if post["route"] else math.nan,
                        time_s=post["time_s"])
    return row


def main():
    out = ROOT / "results" / "turn_post"
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(s, seed) for s in SCENARIOS for seed in range(50)]
    with ProcessPoolExecutor(8) as pool, (out / "runs.jsonl").open("w") as stream:
        rows = list(pool.map(run, tasks))
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    keys = [f"eps={e:g}" for e in SLACKS] + ["fixed_xy"]
    for name in SCENARIOS:
        group = [r for r in rows if r["scenario"] == name]
        line = f"{name:7s} stage1 J {np.median([r['J'] for r in group]):.3f} turn {np.median([r['turn'] for r in group]):5.0f}"
        for k in keys:
            line += (f" | {k}: dJ {np.median([100 * (r[k]['J'] / r['J'] - 1) for r in group]):+.2f}% "
                     f"turn {np.median([r[k]['turn'] for r in group]):4.0f} "
                     f"clr {min(r[k]['clearance'] for r in group):.3f}")
        print(line)


if __name__ == "__main__":
    main()
