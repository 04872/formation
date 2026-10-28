#!/usr/bin/env python3
"""Offline diagnostic for the "sample already covered -> skip" rule of the full method (A).

During a normal run of A every skipped sample ``q_rand`` (inside the selected region) is recorded.  After the run,
``C_r = C(q_rand)`` is built and tested against the final regions for single-region containment
``C_r^delta ⊆ C_j^delta`` (``C^delta`` = points with slack >= delta):

* not contained (proved): a test point of ``C_r^delta`` lies outside ``C_j^delta`` for every candidate ``j``;
* possibly contained: some ``C_j^delta`` holds all test points (test points = seed + quasi-random points of
  ``C_r^delta``; this side over-counts redundancy, so "new" counts are lower bounds);
* empty: ``C_r^delta`` is empty (``C_r`` cannot be connected at the overlap margin anyway).

Coverage of the BIT* path (same seed, results/external) by the tree regions, and with the proved non-contained
``C_r`` added.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bit_cell_coverage import bit_route, covered
from compare_coupling import SCENARIOS, route_profile, scenario
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar
from formation.types import Pose2D

TEST_POINTS = 512


class Recording(SingleEntryRRTStar):
    def _steer(self, parent, sample):
        before = self.stats["covered_samples"]
        result = super()._steer(parent, sample)
        if self.stats["covered_samples"] > before:
            self.skipped.append(sample)
        return result


def test_points(model, cell, delta, rng):
    g, limit = cell.guard, cell.yaw_limit
    u = rng.random((TEST_POINTS, 3)) * 2.0 - 1.0
    dx, dy, dphi = u[:, 0] * g, u[:, 1] * g, u[:, 2] * limit
    inside = model._local_slack(cell, dx, dy, dphi) >= delta
    pts = np.column_stack((cell.pose.x + dx[inside], cell.pose.y + dy[inside], cell.pose.yaw + dphi[inside]))
    return np.vstack(((cell.pose.x, cell.pose.y, cell.pose.yaw), pts))


def run(task):
    name, seed = task
    map_data, slots, start = scenario(name)
    planner = Recording(map_data, slots, start, CouplingConfig(seed=seed, checkpoints=(6000,)))
    planner.skipped = []
    planner.plan()
    model, delta = planner.model, planner.config.overlap_margin
    cells = planner.cells
    seeds = np.array([(c.pose.x, c.pose.y, c.pose.yaw) for c in cells])
    guards = np.array([c.guard for c in cells])
    rng = np.random.default_rng(seed)
    counts = dict(skipped=len(planner.skipped), invalid=0, empty=0, contained=0, new=0)
    new_cells = []
    for q in planner.skipped:
        cell = model.make_cell(q)
        if not cell.valid:
            counts["invalid"] += 1
            continue
        if model.slack(cell, q) < delta:
            counts["empty"] += 1
            continue
        pts = test_points(model, cell, delta, rng)
        near = np.flatnonzero(np.hypot(seeds[:, 0] - q.x, seeds[:, 1] - q.y) < guards + cell.guard)
        contained = False
        for j in near:
            if model.slack(cells[j], q) >= delta and np.all(model.slack_many(cells[j], pts) >= delta):
                contained = True
                break
        if contained:
            counts["contained"] += 1
        else:
            counts["new"] += 1
            new_cells.append(cell)
    row = dict(scenario=name, seed=seed, regions=len(cells), **counts)
    route, _ = bit_route(name, seed)
    _, poses, _ = route_profile(map_data, slots, route, step=0.02)
    near = lambda cs: [c for c in cs if np.min(np.hypot(poses[:, 0] - c.pose.x, poses[:, 1] - c.pose.y)) < 2.0]
    tree_hit = covered(model, near(cells), poses)
    new_hit = covered(model, near(new_cells), poses)
    row.update(tree_coverage=float(tree_hit.mean()), with_new_coverage=float((tree_hit | new_hit).mean()))
    return row


def main():
    out = ROOT / "results" / "covered_samples"
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
        med = lambda k: float(np.median([r[k] for r in g]))
        share = lambda k: float(np.median([r[k] / max(1, r["skipped"]) for r in g]))
        print(f"{name:7s} regions {med('regions'):6.0f} skipped {med('skipped'):6.0f} | contained {share('contained'):.2f} "
              f"new {share('new'):.2f} empty {share('empty'):.2f} invalid {share('invalid'):.2f} | BIT* path coverage "
              f"tree {med('tree_coverage'):.3f} -> +new {med('with_new_coverage'):.3f} "
              f"(min {min(r['tree_coverage'] for r in g):.3f} -> {min(r['with_new_coverage'] for r in g):.3f})")


if __name__ == "__main__":
    main()
