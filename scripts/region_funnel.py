#!/usr/bin/env python3
"""Where does A lose regions on the low-cost corridor?  Funnel along the BIT* path (diagnostic only).

The BIT* shortcut path of the same seed (densified every 0.02 in d_W) is the reference corridor; it does not enter
sampling or expansion.  A runs unchanged (step ``STEP``) and every iteration is logged:

  sample     q_rand within d_W ``NEAR`` of the path (and within ``NEAR`` in xy only)
  steer      q_rand inside the selected region (skipped), x* = Pi_{C_i}(q_rand) and the first candidate q_new near
             the path
  region     candidate cells: valid?  covering path points not yet covered by the tree ("useful")?
  overlap    useful valid candidates rejected because they miss the delta-overlap with the parent; whether they
             delta-overlap some other existing region
  access     accepted regions adding coverage; at the end, coverage of the path and links between consecutive path
             points (one region covers both, or covering regions delta-overlap)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bit_cell_coverage import bit_route
from compare_coupling import route_profile, scenario
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar

STEP = 0.3
NEAR = 0.3
ITERATIONS = 6000
SNAPSHOTS = (500, 1000, 2000, 4000, 6000)


class FunnelRRTStar(SingleEntryRRTStar):
    def setup(self, poses):
        self.path = poses
        self.hit = np.zeros(len(poses), bool)
        self.count = Counter()
        self.coverage_at = {}
        self.calls = 0

    def _distances(self, pose):
        p = self.path
        xy = np.hypot(p[:, 0] - pose.x, p[:, 1] - pose.y)
        d = np.sqrt(xy ** 2 + (self.rho * (np.remainder(p[:, 2] - pose.yaw + math.pi, 2 * math.pi) - math.pi)) ** 2)
        return d, xy

    def _near(self, pose):
        d, xy = self._distances(pose)
        return float(d.min()), float(xy.min())

    def _gap(self, pose):
        """d_W to the nearest path point not yet covered by the tree."""
        d, _ = self._distances(pose)
        return float(d[~self.hit].min()) if not self.hit.all() else math.inf

    def _covers(self, cell):
        return self.model.slack_many(cell, self.path) >= 0.0

    def _elsewhere(self, new, parent):
        """Does ``new`` delta-overlap an existing region other than ``parent``?"""
        n = len(self.cells)
        dyaw = self.rho * np.abs(np.remainder(self._t[:n] - new.pose.yaw + math.pi, 2 * math.pi) - math.pi)
        d = np.hypot(self._x[:n] - new.pose.x, self._y[:n] - new.pose.y)
        mask = (d < self._g[:n] + new.guard) & (dyaw < self._lim[:n] + new.rho * new.yaw_limit)
        mask[parent] = False
        idx = np.flatnonzero(mask)
        for j in idx[np.argsort(d[idx] + dyaw[idx])][:self.config.max_neighbors].tolist():
            if not self._separated(self.cells[j], new) and self._overlap(self.cells[j], new) is not None:
                return True
        return False

    def _steer(self, parent, sample):
        self.calls += 1
        c = self.count
        d_rand, xy_rand = self._near(sample)
        near = d_rand <= NEAR
        gap = self._gap(sample) <= NEAR
        tag = "gap" if gap else "near" if near else "far"
        c["samples"] += 1
        c["samples_gap"] += gap
        c["samples_near"] += near
        c["samples_near_xy"] += xy_rand <= NEAR
        cell = self.cells[parent]
        x_star = self.geo.project(cell, sample)
        anchor = cell.pose.yaw
        z_star, z_rand = self.geo.lift(x_star, anchor), self.geo.lift(sample, anchor)
        direction = z_rand - z_star
        length = float(np.linalg.norm(direction))
        result = (None, None, self.config.steer_attempts)
        if length < 1e-9:
            c[f"{tag}_skipped"] += 1
            self.stats["covered_samples"] += 1
        else:
            c[f"{tag}_xstar_near"] += self._near(x_star)[0] <= NEAR
            if gap:
                c["gap_xstar_dist"] += self._gap(x_star)
                c["gap_rand_to_xstar"] += length
            direction /= length
            step = self.config.step
            for attempt in range(1, self.config.steer_attempts + 1):
                pose = self.geo.pose(z_star + min(step, length) * direction)
                cand_near = self._near(pose)[0] <= NEAR
                if attempt == 1:
                    c[f"{tag}_qnew_near"] += cand_near
                    c[f"{tag}_qnew_gap"] += self._gap(pose) <= NEAR
                new = self.model.make_cell(pose)
                c["cand"] += 1
                c["cand_near"] += cand_near
                useful = 0
                if new.valid:
                    c["cand_valid"] += 1
                    c["cand_near_valid"] += cand_near
                    useful = int(np.count_nonzero(self._covers(new) & ~self.hit))
                    c["cand_useful"] += useful > 0
                ok = new.valid and not self._separated(cell, new)
                portal = self._overlap(cell, new) if ok else None
                c[f"{tag}_cand"] += 1
                c[f"{tag}_cand_invalid"] += not new.valid
                c[f"{tag}_cand_no_overlap"] += new.valid and portal is None
                c[f"{tag}_cand_useful"] += useful > 0
                if portal is not None:
                    result = (new, portal, attempt)
                    if useful:
                        c["accepted_useful"] += 1
                        c[f"accepted_useful_from_{tag}"] += 1
                        c["useful_points_added"] += useful
                        self.hit |= self._covers(new)
                    c["accepted"] += 1
                    c[f"accepted_from_{tag}"] += 1
                    break
                if useful:
                    c["useful_rejected_overlap"] += 1
                    c["useful_rejected_points"] += useful
                    c["useful_rejected_elsewhere"] += self._elsewhere(new, parent)
                step *= 0.5
            if result[0] is None:
                c[f"{tag}_failed"] += 1
                self.stats["steer_failed"] += 1
        if self.calls in SNAPSHOTS:
            self.coverage_at[self.calls] = float(self.hit.mean())
        return result


def links(planner):
    """Consecutive path points linked by a common region or by delta-overlapping covering regions."""
    path = planner.path
    near = [i for i, cell in enumerate(planner.cells)
            if np.min(np.hypot(path[:, 0] - cell.pose.x, path[:, 1] - cell.pose.y)) < cell.guard + 1e-9]
    cover = np.array([planner._covers(planner.cells[i]) for i in near]).reshape((len(near), len(path)))
    cache = {}

    def overlap(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            first, second = planner.cells[key[0]], planner.cells[key[1]]
            cache[key] = not planner._separated(first, second) and planner._overlap(first, second) is not None
        return cache[key]

    linked, breaks, uncovered_runs = 0, 0, 0
    prev_uncovered = False
    for k in range(len(path) - 1):
        a, b = np.flatnonzero(cover[:, k]), np.flatnonzero(cover[:, k + 1])
        if not len(b):
            uncovered_runs += not prev_uncovered
            prev_uncovered = True
        else:
            prev_uncovered = False
        if not len(a) or not len(b):
            continue
        if set(a.tolist()) & set(b.tolist()) or any(overlap(near[i], near[j]) for i in a for j in b):
            linked += 1
        else:
            breaks += 1
    return dict(covered_transitions=linked + breaks, linked=linked, overlap_breaks=breaks,
                uncovered_runs=uncovered_runs)


def run(task):
    name, seed = task
    map_data, slots, start = scenario(name)
    route, bit_length = bit_route(name, seed)
    _, poses, _ = route_profile(map_data, slots, route, step=0.02)
    planner = FunnelRRTStar(map_data, slots, start, CouplingConfig(seed=seed, step=STEP, checkpoints=(ITERATIONS,)))
    planner.setup(poses)
    result = planner.plan()
    row = dict(scenario=name, seed=seed, bit=bit_length, A=result["checkpoints"][-1]["optimised_cost"],
               regions=len(planner.cells), path_points=len(poses), coverage=float(planner.hit.mean()),
               coverage_at=planner.coverage_at, **planner.count)
    row.update(links(planner))
    return row


def main():
    out = ROOT / "results" / "region_funnel"
    out.mkdir(parents=True, exist_ok=True)
    scenarios = sys.argv[1].split(",") if len(sys.argv) > 1 else ["forest", "slots"]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(run, [(s, seed) for s in scenarios for seed in range(seeds)]))
    with (out / "runs.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for name in scenarios:
        g = [r for r in rows if r["scenario"] == name]
        tot = lambda k: sum(r.get(k, 0) for r in g)
        med = lambda k: float(np.median([r[k] for r in g]))
        n, near = tot("samples"), tot("samples_near")
        print(f"{name}: A {med('A'):.3f}  BIT {med('bit'):.3f}  regions {med('regions'):.0f}  coverage {med('coverage'):.3f}  "
              f"over iterations {[round(float(np.median([r['coverage_at'][str(s)] if str(s) in r['coverage_at'] else r['coverage_at'][s] for r in g])), 3) for s in SNAPSHOTS]}")
        gap = tot("samples_gap")
        print(f"   sample   near path {near / n:.3%} (xy only {tot('samples_near_xy') / n:.3%}); near an uncovered "
              f"path point {gap / n:.3%} ({gap / len(g):.1f} per run)")
        steered = gap - tot("gap_skipped")
        print(f"   gap samples: d_W(q_rand, x*) {tot('gap_rand_to_xstar') / max(1, steered):.3f}, "
              f"x* to nearest uncovered point {tot('gap_xstar_dist') / max(1, steered):.3f}")
        for tag in ("gap", "near", "far"):
            base = max(1, tot(f"{tag}_skipped") + tot(f"{tag}_failed") + tot(f"accepted_from_{tag}"))
            cand = max(1, tot(f"{tag}_cand"))
            print(f"   [{tag}] n={base}: skipped {tot(f'{tag}_skipped') / base:.3f}  x* near {tot(f'{tag}_xstar_near') / base:.3f}  "
                  f"q_new near {tot(f'{tag}_qnew_near') / base:.3f}  q_new near gap {tot(f'{tag}_qnew_gap') / base:.3f}  "
                  f"failed {tot(f'{tag}_failed') / base:.3f}  accepted useful {tot(f'accepted_useful_from_{tag}') / base:.3f}"
                  f" | candidates {cand}: invalid {tot(f'{tag}_cand_invalid') / cand:.3f}  no overlap "
                  f"{tot(f'{tag}_cand_no_overlap') / cand:.3f}  useful {tot(f'{tag}_cand_useful') / cand:.3f}")
        print(f"   region   candidates {tot('cand')}, near path {tot('cand_near') / tot('cand'):.3f}; valid {tot('cand_valid') / tot('cand'):.3f}"
              f" (near path {tot('cand_near_valid') / max(1, tot('cand_near')):.3f}); useful {tot('cand_useful') / tot('cand'):.4f}")
        print(f"   overlap  useful valid candidates {tot('cand_useful')}: accepted {tot('accepted_useful')}, rejected by parent overlap "
              f"{tot('useful_rejected_overlap')} (of which overlap another region {tot('useful_rejected_elsewhere')}); "
              f"new points accepted {tot('useful_points_added')} vs rejected {tot('useful_rejected_points')}")
        print(f"   access   covered transitions linked {tot('linked') / max(1, tot('covered_transitions')):.4f} "
              f"(overlap breaks per run {tot('overlap_breaks') / len(g):.2f}), uncovered runs per run {tot('uncovered_runs') / len(g):.2f}")


if __name__ == "__main__":
    main()
