#!/usr/bin/env python3
"""Does the full method (A) lock in its first goal branch?

Per run (A, default config):
1. goal branch at the first solution and at iterations 2000 / 4000 / 6000: goal region, region sequence, chain length,
   share of the final sequence shared with each earlier sequence, share of final-sequence regions born after the
   first solution;
2. goal regions added after the first solution, and whether the final sequence ends in one of them;
3. re-search on the fixed region set of iteration 6000: overlap graph of all regions (delta-overlap witnesses as
   nodes, straight d_W edges between witnesses of a common region, start and goal attached), Dijkstra, then the same
   chain optimiser on the found region sequence.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import heapq
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_coupling import route_clearance, scenario
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar
from formation.types import Pose2D

CHECKPOINTS = (2000, 4000, 6000)


def overlap_graph(planner):
    """Witness nodes per overlapping pair and the region -> witnesses map."""
    cells, geo, rho = planner.cells, planner.geo, planner.rho
    n = len(cells)
    x, y, t = planner._x[:n], planner._y[:n], planner._t[:n]
    g, lim = planner._g[:n], planner._lim[:n]
    witnesses, members = [], defaultdict(list)
    for i in range(n):
        dyaw = rho * np.abs(np.remainder(t[:i] - t[i] + math.pi, 2 * math.pi) - math.pi)
        mask = (np.hypot(x[:i] - x[i], y[:i] - y[i]) < g[:i] + g[i]) & (dyaw < lim[:i] + lim[i])
        for j in np.flatnonzero(mask).tolist():
            if planner._separated(cells[i], cells[j]):
                continue
            portal = planner._overlap(cells[i], cells[j])
            if portal is not None:
                k = len(witnesses)
                witnesses.append((portal, i, j))
                members[i].append(k)
                members[j].append(k)
    return witnesses, members


def research(planner):
    cells, geo, margin = planner.cells, planner.geo, planner.config.overlap_margin
    witnesses, members = overlap_graph(planner)
    poses = [w[0] for w in witnesses]
    start, goal = len(poses), len(poses) + 1
    adjacency = defaultdict(list)
    for region, ks in members.items():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                d = geo.distance(poses[ks[a]], poses[ks[b]])
                adjacency[ks[a]].append((ks[b], d, region))
                adjacency[ks[b]].append((ks[a], d, region))
    for region in range(len(cells)):
        if geo.slack(cells[region], planner.start) >= 0.0:
            for k in members[region]:
                adjacency[start].append((k, geo.distance(planner.start, poses[k]), region))
    goal_regions = set()
    for region in planner.goal_cells:
        for k in members[region]:
            yaw = geo.goal_yaw(cells[region], planner.goal_xy, poses[k].yaw, planner.config.overlap_margin)
            target = Pose2D(planner.goal_xy[0], planner.goal_xy[1], yaw)
            adjacency[k].append((goal, geo.distance(poses[k], target), region))
            goal_regions.add(region)
    dist, back = {start: 0.0}, {}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w, region in adjacency[u]:
            if d + w < dist.get(v, math.inf) - 1e-12:
                dist[v], back[v] = d + w, (u, region)
                heapq.heappush(heap, (d + w, v))
    if goal not in dist:
        return dict(graph_nodes=len(poses), graph_found=False)
    sequence, node = [], goal
    while node != start:
        node, region = back[node]
        if not sequence or sequence[-1] != region:
            sequence.append(region)
    sequence = sequence[::-1]
    solved = geo.chain(planner.start, planner.goal_xy, [cells[i] for i in sequence], margin * (1.0 - 1e-6))
    return dict(graph_nodes=len(poses), graph_found=True, graph_cost=dist[goal], graph_regions=len(sequence),
                research_length=solved[0] if solved else math.nan,
                research_route=[(p.x, p.y, p.yaw) for p in solved[1]] if solved else [])


def run(task):
    name, seed = task
    map_data, slots, start = scenario(name)
    planner = SingleEntryRRTStar(map_data, slots, start, CouplingConfig(seed=seed, checkpoints=CHECKPOINTS))
    result = planner.plan()
    first = result["first"]
    records = [("first", first)] + [(str(c["iteration"]), c) for c in result["checkpoints"]]
    born_first = first["regions"] if first else None
    final_seq = result["checkpoints"][-1]["sequence"]
    row = dict(scenario=name, seed=seed, regions=len(planner.cells), first_iteration=first and first["iteration"],
               first_regions=born_first, goal_cells=len(planner.goal_cells),
               goal_cells_after_first=sum(g >= born_first for g in planner.goal_cells) if first else None,
               final_goal_born_after_first=final_seq[-1] >= born_first if first else None,
               final_born_after_first=float(np.mean([i >= born_first for i in final_seq])) if first else None)
    for key, record in records:
        if not record or not record["success"]:
            row[f"len@{key}"] = row[f"goal@{key}"] = row[f"shared_with_final@{key}"] = None
            continue
        seq = record["sequence"]
        row[f"len@{key}"] = record["optimised_cost"]
        row[f"goal@{key}"] = seq[-1]
        row[f"shared_with_final@{key}"] = len(set(seq) & set(final_seq)) / len(final_seq)
    row.update(research(planner))
    if row.get("research_route"):
        row["research_clearance"] = route_clearance(map_data, slots, row.pop("research_route"))
    else:
        row.pop("research_route", None)
    return row


def main():
    out = ROOT / "results" / "lock_in"
    out.mkdir(parents=True, exist_ok=True)
    scenarios = sys.argv[1].split(",") if len(sys.argv) > 1 else ["forest", "dense", "slots"]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    tasks = [(s, seed) for s in scenarios for seed in range(seeds)]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(run, tasks))
    with (out / "runs.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for name in scenarios:
        g = [r for r in rows if r["scenario"] == name]
        med = lambda k: float(np.median([r[k] for r in g if r.get(k) is not None]))
        print(f"{name}: regions {med('regions'):.0f}, first at it {med('first_iteration'):.0f} with {med('first_regions'):.0f} regions")
        for key in ("first", "2000", "4000", "6000"):
            same_goal = np.mean([r[f"goal@{key}"] == r["goal@6000"] for r in g if r[f"goal@{key}"] is not None])
            print(f"   @{key:5s} len {med(f'len@{key}'):.3f}  goal == final goal {same_goal:.2f}  "
                  f"shared with final seq {med(f'shared_with_final@{key}'):.2f}")
        print(f"   final seq: share born after first {med('final_born_after_first'):.2f}; final goal born after first "
              f"{np.mean([r['final_goal_born_after_first'] for r in g]):.2f}; goal regions {med('goal_cells'):.0f} "
              f"(after first {med('goal_cells_after_first'):.0f})")
        found = [r for r in g if r.get("graph_found")]
        gain = [100 * (r["research_length"] / r["len@6000"] - 1) for r in found if not math.isnan(r["research_length"])]
        print(f"   re-search on fixed regions: found {len(found)}/{len(g)}, length {np.median([r['research_length'] for r in found]):.3f} "
              f"vs A {med('len@6000'):.3f}: median {np.median(gain):+.2f}% (better {sum(x < -0.1 for x in gain)}, worse "
              f"{sum(x > 0.1 for x in gain)}), min clearance {min(r.get('research_clearance', math.inf) for r in found):.3f}")


if __name__ == "__main__":
    main()
