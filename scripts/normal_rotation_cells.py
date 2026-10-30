#!/usr/bin/env python3
"""Normal-aware rotation bound on the same seeds (static diagnostic, planner unchanged).

Old row (robot r, obstacle j):  n^T dc - rho |dth| + e >= 0.
New row:                         n^T dc - a_+ [dth]_+ - a_- [-dth]_+ + e >= 0,
with f(t) = n^T (R(th_0 + t) - R(th_0)) s_r,  f'(t) = ||s_r|| cos(t + phi),  phi = angle(R(th_0) J s_r) - angle(n),
a_+ = max(0, -min_{[0, L]} f'),  a_- = max(0, max_{[-L, 0]} f').  Guard ``||dc|| + rho |dth| <= r_g`` and yaw range
``|dth| <= L`` unchanged, so C_old ⊆ C_new ⊆ free.

A runs unchanged (step 0.3, 6000 iterations); its seeds are rebuilt both ways and compared on
  safety      sampled points of new cells: true clearance >= d_s;  old ⊆ new on the same samples
  size        volume ratio (quasi-random box samples), share of cells with obstacle rows
  path        coverage of the BIT* path, uncovered runs, links between consecutive path points
  overlap     delta-overlapping pairs (guard / yaw prefilter, max common margin by SOCP)
  binding     for path points no old cell covers: the term limiting the best old cell (guard, yaw, row)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import clarabel
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bit_cell_coverage import bit_route
from compare_coupling import route_profile, scenario
from formation.polyhedral_cell import halton, wrap_array
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar

STEP = 0.3
ITERATIONS = 6000
SAFETY_CELLS = 200
SAMPLES = halton(256)


@dataclass
class Cell:
    x: float
    y: float
    t: float
    normals: np.ndarray
    offsets: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    guard: float
    limit: float
    rho: float

    @property
    def valid(self):
        radius = min(float(self.offsets.min()), self.guard) if len(self.offsets) else self.guard
        return radius > 0.0 and self.limit > 0.0


def _cos_min(a, b):
    k = math.ceil((a - math.pi) / (2 * math.pi))
    return -1.0 if math.pi + 2 * math.pi * k <= b else min(math.cos(a), math.cos(b))


def _cos_max(a, b):
    k = math.ceil(a / (2 * math.pi))
    return 1.0 if 2 * math.pi * k <= b else max(math.cos(a), math.cos(b))


def build(model, pose, normal_aware):
    """All broadphase rows with ``e < r_g`` (the lazy cell's pruning is exact, so old = A's cell)."""
    clearance, normals = model.proximity(model.robot_positions(pose))
    components = clearance.shape[1]
    flat = clearance.ravel()
    broadphase = flat < model.active_range
    guard = model.max_extent
    if broadphase.sum() < len(flat):
        guard = min(float(flat[~broadphase].min()) - model.safety_distance, guard)
    offsets = flat - model.safety_distance
    index = np.flatnonzero(broadphase & (offsets < guard))
    offsets, rows = offsets[index], normals.reshape((-1, 2))[index]
    radius = min(float(offsets.min()), guard) if len(index) else guard
    limit = min(model.yaw_chart, 0.5 * (radius + guard) / model.rho) if radius > 0.0 else 0.0
    plus, minus = np.full(len(index), model.rho), np.full(len(index), model.rho)
    if normal_aware:
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        for k, (flat_index, n) in enumerate(zip(index.tolist(), rows)):
            slot = model.slots[flat_index // components]
            jx, jy = -slot[1], slot[0]
            w = (c * jx - s * jy, s * jx + c * jy)
            phi = math.atan2(w[1], w[0]) - math.atan2(n[1], n[0])
            size = math.hypot(*slot)
            plus[k] = size * max(0.0, -_cos_min(phi, phi + limit))
            minus[k] = size * max(0.0, _cos_max(phi - limit, phi))
    return Cell(pose.x, pose.y, pose.yaw, rows, offsets, plus, minus, guard, limit, model.rho)


def slack(cell, points):
    dx, dy = points[:, 0] - cell.x, points[:, 1] - cell.y
    dth = wrap_array(points[:, 2] - cell.t)
    turn = cell.rho * np.abs(dth)
    terms = {"guard": cell.guard - np.hypot(dx, dy) - turn, "yaw": cell.rho * cell.limit - turn}
    if len(cell.offsets):
        rot = cell.plus[:, None] * np.maximum(dth, 0.0)[None, :] + cell.minus[:, None] * np.maximum(-dth, 0.0)[None, :]
        rows = cell.normals[:, :1] * dx[None, :] + cell.normals[:, 1:] * dy[None, :] + cell.offsets[:, None] - rot
        terms["row"] = rows.min(axis=0)
    return terms


def total(terms):
    return np.minimum.reduce(list(terms.values()))


def box_samples(cell):
    g, lim = cell.guard, cell.limit
    return np.column_stack((cell.x + (2 * SAMPLES[:, 0] - 1) * g, cell.y + (2 * SAMPLES[:, 1] - 1) * g,
                            cell.t + (2 * SAMPLES[:, 2] - 1) * lim))


SETTINGS = clarabel.DefaultSettings()
SETTINGS.verbose = False


def max_margin(first, second):
    """``max t`` with a common point at slack ``>= t`` in both cells (chart phi = rho * theta near ``first``)."""
    rho = first.rho
    a_rows, b_rows, cones = [], [], []
    for cell in (first, second):
        cphi = rho * (first.t + math.remainder(cell.t - first.t, 2 * math.pi))
        for n, e, ap, am in zip(cell.normals, cell.offsets, cell.plus, cell.minus):
            base = e - n @ (cell.x, cell.y)
            a_rows.append((-n[0], -n[1], ap / rho, 1.0)); b_rows.append(base + ap / rho * cphi)
            a_rows.append((-n[0], -n[1], -am / rho, 1.0)); b_rows.append(base - am / rho * cphi)
        a_rows.append((0.0, 0.0, 1.0, 1.0)); b_rows.append(rho * cell.limit + cphi)
        a_rows.append((0.0, 0.0, -1.0, 1.0)); b_rows.append(rho * cell.limit - cphi)
        for sigma in (1.0, -1.0):
            cones.append(((0.0, 0.0, sigma, 1.0), cell.guard + sigma * cphi, cell.x, cell.y))
    rows = len(a_rows) + 3 * len(cones)
    matrix = np.zeros((rows, 4))
    rhs = np.zeros(rows)
    matrix[:len(a_rows)] = a_rows
    rhs[:len(a_rows)] = b_rows
    k = len(a_rows)
    for head, h, cx, cy in cones:
        matrix[k] = head
        rhs[k] = h
        matrix[k + 1, 0] = matrix[k + 2, 1] = -1.0
        rhs[k + 1], rhs[k + 2] = -cx, -cy
        k += 3
    cone_types = [clarabel.NonnegativeConeT(len(a_rows))] + [clarabel.SecondOrderConeT(3)] * len(cones)
    q = np.array((0.0, 0.0, 0.0, -1.0))
    result = clarabel.DefaultSolver(sp.csc_matrix((4, 4)), q, sp.csc_matrix(matrix), rhs, cone_types, SETTINGS).solve()
    return float(result.x[3]) if str(result.status) in ("Solved", "AlmostSolved") else -math.inf


def path_stats(cells, poses, margin, overlaps):
    cover = np.array([total(slack(c, poses)) >= 0.0 for c in cells])
    hit = cover.any(axis=0)
    runs = int(np.sum(~hit[1:] & hit[:-1]) + (not hit[0]))
    linked = breaks = 0
    for k in range(len(poses) - 1):
        a, b = np.flatnonzero(cover[:, k]), np.flatnonzero(cover[:, k + 1])
        if not len(a) or not len(b):
            continue
        if set(a.tolist()) & set(b.tolist()) or any(overlaps.get((min(i, j), max(i, j)), False) for i in a for j in b):
            linked += 1
        else:
            breaks += 1
    return hit, dict(coverage=float(hit.mean()), uncovered_runs=runs, overlap_breaks=breaks, linked=linked)


def run(task):
    name, seed = task
    map_data, slots, start = scenario(name)
    route, bit_length = bit_route(name, seed)
    _, poses, _ = route_profile(map_data, slots, route, step=0.02)
    planner = SingleEntryRRTStar(map_data, slots, start, CouplingConfig(seed=seed, step=STEP, checkpoints=(ITERATIONS,)))
    planner.plan()
    model, margin = planner.model, planner.config.overlap_margin
    seeds = [c.pose for c in planner.cells]
    old = [build(model, p, False) for p in seeds]
    new = [build(model, p, True) for p in seeds]
    row = dict(scenario=name, seed=seed, regions=len(seeds))

    # A's own cells must equal the rebuilt old cells
    probe = np.vstack([box_samples(o) for o in old[:20]])
    row["old_matches_A"] = bool(all(np.allclose(total(slack(o, probe)) >= 0.0, model.slack_many(c, probe) >= 0.0)
                                    for o, c in zip(old[:20], planner.cells[:20])))

    # safety and containment on quasi-random box samples
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(seeds), size=min(SAFETY_CELLS, len(seeds)), replace=False)
    worst, not_contained, ratios = math.inf, 0, []
    for i in range(len(seeds)):
        pts = box_samples(old[i])
        inside_old, inside_new = total(slack(old[i], pts)) >= 0.0, total(slack(new[i], pts)) >= 0.0
        not_contained += int(np.sum(inside_old & ~inside_new))
        ratios.append(inside_new.sum() / max(1, inside_old.sum()))
        if i in pick and inside_new.any():
            q = pts[inside_new]
            c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
            px = q[:, 0, None] + c[:, None] * slots[None, :, 0] - s[:, None] * slots[None, :, 1]
            py = q[:, 1, None] + s[:, None] * slots[None, :, 0] + c[:, None] * slots[None, :, 1]
            clearance, _ = model.proximity(np.column_stack((px.ravel(), py.ravel())))
            worst = min(worst, float(clearance.min()) - model.safety_distance)
    row.update(worst_clearance_minus_ds=worst, not_contained=not_contained,
               volume_ratio_median=float(np.median(ratios)), volume_ratio_mean=float(np.mean(ratios)),
               share_with_rows=float(np.mean([len(o.offsets) > 0 for o in old])),
               alpha_over_rho=float(np.mean(np.concatenate([np.concatenate((c.plus, c.minus)) for c in new
                                                            if len(c.offsets)])) / model.rho))

    # delta-overlap pairs (old via A's own test, new by SOCP where old fails; C_old ⊆ C_new)
    n = len(seeds)
    x = np.array([c.x for c in old]); y = np.array([c.y for c in old]); t = np.array([c.t for c in old])
    g = np.array([c.guard for c in old]); lim = np.array([model.rho * c.limit for c in old])
    old_pairs, new_pairs = {}, {}
    for i in range(n):
        dyaw = model.rho * np.abs(wrap_array(t[:i] - t[i]))
        for j in np.flatnonzero((np.hypot(x[:i] - x[i], y[:i] - y[i]) < g[:i] + g[i]) & (dyaw < lim[:i] + lim[i])).tolist():
            a, b = planner.cells[j], planner.cells[i]
            ok = not planner._separated(a, b) and planner._overlap(a, b) is not None
            old_pairs[(j, i)] = ok
            new_pairs[(j, i)] = ok or max_margin(new[j], new[i]) >= margin
    row.update(candidate_pairs=len(old_pairs), old_overlaps=sum(old_pairs.values()), new_overlaps=sum(new_pairs.values()))

    hit_old, s_old = path_stats(old, poses, margin, old_pairs)
    hit_new, s_new = path_stats(new, poses, margin, new_pairs)
    row.update({f"old_{k}": v for k, v in s_old.items()})
    row.update({f"new_{k}": v for k, v in s_new.items()})
    row["new_only_points"] = int(np.sum(hit_new & ~hit_old))

    # binding term of the best old cell at points no old cell covers
    missing = poses[~hit_old]
    binding = {"guard": 0, "yaw": 0, "row": 0}
    if len(missing):
        best, which = np.full(len(missing), -np.inf), np.empty(len(missing), dtype=object)
        for c in old:
            terms = slack(c, missing)
            value = total(terms)
            better = value > best
            if better.any():
                names = list(terms)
                arg = np.argmin(np.vstack([terms[k] for k in names]), axis=0)
                best[better] = value[better]
                which[better] = np.array(names, dtype=object)[arg[better]]
        for k in binding:
            binding[k] = int(np.sum(which == k))
    row.update({f"missing_bound_{k}": v for k, v in binding.items()}, missing_points=len(missing))
    return row


def main():
    out = ROOT / "results" / "normal_rotation_cells"
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
        med = lambda k: float(np.median([r[k] for r in g]))
        tot = lambda k: sum(r[k] for r in g)
        print(f"{name}: old == A cells {all(r['old_matches_A'] for r in g)}; worst sampled clearance - d_s "
              f"{min(r['worst_clearance_minus_ds'] for r in g):+.2e}; old points outside new {tot('not_contained')}")
        print(f"   cells with obstacle rows {med('share_with_rows'):.2f}; mean alpha / rho {med('alpha_over_rho'):.3f}; "
              f"volume new/old median {med('volume_ratio_median'):.3f} mean {med('volume_ratio_mean'):.3f}")
        print(f"   overlapping pairs old {med('old_overlaps'):.0f} -> new {med('new_overlaps'):.0f} "
              f"({100 * (tot('new_overlaps') / tot('old_overlaps') - 1):+.1f}%) of {med('candidate_pairs'):.0f} candidates")
        print(f"   BIT* path coverage old {med('old_coverage'):.3f} -> new {med('new_coverage'):.3f}; uncovered runs "
              f"{med('old_uncovered_runs'):.1f} -> {med('new_uncovered_runs'):.1f}; overlap breaks {tot('old_overlap_breaks')} -> "
              f"{tot('new_overlap_breaks')}")
        m = max(1, tot("missing_points"))
        print(f"   points no old cell covers: best old cell bound by guard {tot('missing_bound_guard') / m:.2f}, "
              f"yaw {tot('missing_bound_yaw') / m:.2f}, obstacle row {tot('missing_bound_row') / m:.2f}; "
              f"newly covered {tot('new_only_points') / m:.3f}")


if __name__ == "__main__":
    main()
