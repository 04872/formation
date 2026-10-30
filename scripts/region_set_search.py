#!/usr/bin/env python3
"""How short a path does A's region set (iteration 6000) support?  Offline search, planner unchanged.

Graph: nodes are poses in the delta-overlap of region pairs, edges are straight d_W segments between nodes of a common
(convex) region, plus start (attached in regions containing it) and goal (attached in goal regions with the yaw closest
to the node).  Every candidate region sequence is scored by the same joint chain optimiser ``geo.chain``.

Levels (each includes the previous one):
  single   one overlap point per pair, Dijkstra sequence                       (= lock_in_diag re-search)
  multi    several overlap points per pair (projections of both seeds, start, goal, the overlap point), Dijkstra
  via      best path through every overlap pair (start and goal Dijkstra trees), top ``CANDIDATES`` distinct sequences
  iterate  chain transition points of every scored sequence added as nodes, ``via`` repeated until no gain
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_coupling import route_clearance, scenario
from external_baselines import best_at
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar
from formation.types import Pose2D, wrap_to_pi

ITERATIONS = 6000
CANDIDATES = 40
ROUNDS = 6


class RegionSet:
    def __init__(self, planner):
        self.p, self.geo, self.cells = planner, planner.geo, planner.cells
        self.margin = planner.config.overlap_margin
        self.start, self.goal_xy = planner.start, planner.goal_xy
        self.start_regions = [r for r in range(len(self.cells)) if self.geo.slack(self.cells[r], self.start) >= 0.0]
        self.goal_regions = sorted(planner.goal_cells)
        self.pairs = self._pairs()
        self.nodes: list[tuple[Pose2D, int, int]] = []
        self.scored: dict[tuple[int, ...], float] = {}
        self.routes: dict[tuple[int, ...], list[Pose2D]] = {}

    def _pairs(self):
        p, n = self.p, len(self.cells)
        x, y, t, g, lim = p._x[:n], p._y[:n], p._t[:n], p._g[:n], p._lim[:n]
        pairs = {}
        for i in range(n):
            dyaw = p.rho * np.abs(np.remainder(t[:i] - t[i] + math.pi, 2 * math.pi) - math.pi)
            mask = (np.hypot(x[:i] - x[i], y[:i] - y[i]) < g[:i] + g[i]) & (dyaw < lim[:i] + lim[i])
            for j in np.flatnonzero(mask).tolist():
                if p._separated(self.cells[i], self.cells[j]):
                    continue
                portal = p._overlap(self.cells[i], self.cells[j])
                if portal is not None:
                    pairs[(j, i)] = portal
        return pairs

    def _inside(self, pose, i, j):
        tol = self.margin * (1.0 - 1e-6)
        return self.geo.slack(self.cells[i], pose) >= tol and self.geo.slack(self.cells[j], pose) >= tol

    def add(self, pose, i, j):
        if self._inside(pose, i, j):
            self.nodes.append((pose, i, j))

    def seed_single(self):
        for (i, j), portal in self.pairs.items():
            self.add(portal, i, j)

    def seed_multi(self):
        margin = self.margin * (1.0 + 1e-6)
        for (i, j) in self.pairs:
            first, second = self.cells[i], self.cells[j]
            goal = Pose2D(self.goal_xy[0], self.goal_xy[1], first.pose.yaw)
            for anchor in (first.pose, second.pose, self.start, goal):
                z = self.geo.attach(anchor, first, second, margin)
                if z is not None:
                    self.add(z, i, j)

    # -- graph ---------------------------------------------------------------------------------------
    def graph(self):
        geo, rho = self.geo, self.geo.rho
        m = len(self.nodes)
        s, g = m, m + 1
        xyt = np.array([(q.x, q.y, q.yaw) for q, _, _ in self.nodes])
        members = defaultdict(list)
        for k, (_, i, j) in enumerate(self.nodes):
            members[i].append(k)
            members[j].append(k)
        rows, cols, weights = [], [], []
        for ks in members.values():
            if len(ks) < 2:
                continue
            ks = np.array(ks)
            q = xyt[ks]
            dyaw = np.remainder(q[:, None, 2] - q[None, :, 2] + math.pi, 2 * math.pi) - math.pi
            d = np.sqrt((q[:, None, 0] - q[None, :, 0]) ** 2 + (q[:, None, 1] - q[None, :, 1]) ** 2 + (rho * dyaw) ** 2)
            a, b = np.triu_indices(len(ks), 1)
            rows.append(ks[a]); cols.append(ks[b]); weights.append(d[a, b])
        extra_r, extra_c, extra_w = [], [], []
        for r in self.start_regions:
            for k in members.get(r, ()):
                extra_r.append(s); extra_c.append(k); extra_w.append(geo.distance(self.start, self.nodes[k][0]))
        self.goal_pose = {}
        for r in self.goal_regions:
            for k in members.get(r, ()):
                yaw = geo.goal_yaw(self.cells[r], self.goal_xy, self.nodes[k][0].yaw, self.margin)
                if yaw is None:
                    continue
                target = Pose2D(self.goal_xy[0], self.goal_xy[1], yaw)
                extra_r.append(k); extra_c.append(g); extra_w.append(geo.distance(self.nodes[k][0], target))
        rows, cols = rows + cols, cols + rows
        weights = weights + weights
        rows = np.concatenate(rows + [np.array(extra_r, int)])
        cols = np.concatenate(cols + [np.array(extra_c, int)])
        weights = np.concatenate(weights + [np.array(extra_w)])
        weights = np.maximum(weights, 1e-12)
        order = np.lexsort((weights, cols, rows))
        rows, cols, weights = rows[order], cols[order], weights[order]
        keep = np.ones(len(rows), bool)
        keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        self.members = members
        return csr_matrix((weights[keep], (rows[keep], cols[keep])), shape=(m + 2, m + 2))

    def _regions(self, path):
        """Region sequence of a node path start -> nodes -> goal, loops removed."""
        sets = [set(self.start_regions)] + [{self.nodes[k][1], self.nodes[k][2]} for k in path[1:-1]] \
            + [set(self.goal_regions)]
        seq = []
        for a, b in zip(sets, sets[1:]):
            common = a & b
            region = seq[-1] if seq and seq[-1] in common else min(common)
            if not seq or seq[-1] != region:
                seq.append(region)
        out = []
        for r in seq:
            if r in out:
                del out[out.index(r) + 1:]
            else:
                out.append(r)
        return tuple(out)

    @staticmethod
    def _walk(pred, source, target):
        path, node = [], target
        while node != source:
            if node < 0:
                return None
            path.append(node)
            node = pred[node]
        path.append(source)
        return path[::-1]

    def score(self, seq):
        if seq not in self.scored:
            solved = self.geo.chain(self.start, self.goal_xy, [self.cells[r] for r in seq], self.margin * (1.0 - 1e-6))
            self.scored[seq] = solved[0] if solved else math.inf
            if solved:
                self.routes[seq] = solved[1]
        return self.scored[seq]

    def dijkstra_best(self):
        graph = self.graph()
        m = len(self.nodes)
        dist, pred = dijkstra(graph, indices=m, return_predecessors=True)
        path = self._walk(pred, m, m + 1)
        if path is None:
            return None
        seq = self._regions(path)
        self.score(seq)
        return seq

    def via(self):
        graph = self.graph()
        m = len(self.nodes)
        ds, ps = dijkstra(graph, indices=m, return_predecessors=True)
        dg, pg = dijkstra(graph.T.tocsr(), indices=m + 1, return_predecessors=True)
        through = ds[:m] + dg[:m]
        best_pair = {}
        for k in np.argsort(through).tolist():
            if not math.isfinite(through[k]):
                break
            key = self.nodes[k][1:]
            if key not in best_pair:
                best_pair[key] = k
        seqs = []
        for k in sorted(best_pair.values(), key=lambda k: through[k]):
            head, tail = self._walk(ps, m, k), self._walk(pg, m + 1, k)
            seq = self._regions(head + tail[::-1][1:])
            if seq not in seqs:
                seqs.append(seq)
            if len(seqs) >= CANDIDATES:
                break
        for seq in seqs:
            self.score(seq)

    def add_transitions(self):
        for seq, route in self.routes.items():
            for k in range(len(seq) - 1):
                self.add(route[k + 1], seq[k], seq[k + 1])

    def best(self):
        seq = min(self.scored, key=self.scored.get) if self.scored else None
        return (self.scored[seq], seq) if seq else (math.inf, None)


def search(planner):
    out = {}
    single = RegionSet(planner)
    single.seed_single()
    single.dijkstra_best()
    out["single"] = single.best()[0]
    t0 = time.perf_counter()
    rs = RegionSet(planner)
    rs.seed_single()
    rs.seed_multi()
    out["nodes_multi"] = len(rs.nodes)
    rs.dijkstra_best()
    out["multi"] = rs.best()[0]
    rs.via()
    out["via"] = rs.best()[0]
    history = [out["via"]]
    for _ in range(ROUNDS):
        rs.add_transitions()
        rs.via()
        history.append(rs.best()[0])
        if history[-2] - history[-1] < 1e-6 * history[-2]:
            break
    length, seq = rs.best()
    out.update(iterate=length, rounds=len(history) - 1, history=history, sequences=len(rs.scored),
               nodes_final=len(rs.nodes), pairs=len(rs.pairs), search_time=time.perf_counter() - t0,
               best_sequence=list(seq) if seq else None)
    route = rs.routes.get(seq)
    out["route"] = [(q.x, q.y, q.yaw) for q in route] if route else []
    return out


def run(task):
    name, seed, bit = task
    map_data, slots, start = scenario(name)
    planner = SingleEntryRRTStar(map_data, slots, start, CouplingConfig(seed=seed, checkpoints=(ITERATIONS,)))
    result = planner.plan()
    final = result["checkpoints"][-1]
    first = result["first"]
    row = dict(scenario=name, seed=seed, bit=bit, regions=len(planner.cells), A=final["optimised_cost"],
               A_sequence=final["sequence"], first_regions=first["regions"] if first else None)
    row.update(search(planner))
    route = row.pop("route")
    row["clearance"] = route_clearance(map_data, slots, route) if route else None
    seq = row["best_sequence"]
    if seq:
        row["shared_with_A"] = len(set(seq) & set(row["A_sequence"])) / len(seq)
        row["born_after_first"] = float(np.mean([r >= row["first_regions"] for r in seq]))
        row["A_born_after_first"] = float(np.mean([r >= row["first_regions"] for r in row["A_sequence"]]))
    return row


def main():
    out = ROOT / "results" / "region_set_search"
    out.mkdir(parents=True, exist_ok=True)
    scenarios = sys.argv[1].split(",") if len(sys.argv) > 1 else ["forest", "dense", "slots"]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    bit = {(r["scenario"], r["seed"]): best_at(r, math.inf)
           for r in map(json.loads, (ROOT / "results" / "external" / "runs.jsonl").open()) if r["variant"] == "bitstar"}
    tasks = [(s, seed, bit.get((s, seed))) for s in scenarios for seed in range(seeds)]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(run, tasks))
    with (out / "runs.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for name in scenarios:
        g = [r for r in rows if r["scenario"] == name]
        med = lambda k: float(np.median([r[k] for r in g if r.get(k) is not None]))
        rel = lambda k, ref: float(np.median([100 * (r[k] / r[ref] - 1) for r in g if r.get(ref)]))
        print(f"{name}: A {med('A'):.3f}  BIT {med('bit'):.3f}  pairs {med('pairs'):.0f}  nodes {med('nodes_final'):.0f}  "
              f"seqs {med('sequences'):.0f}  rounds {med('rounds'):.0f}  time {med('search_time'):.1f}s  "
              f"min clearance {min(r['clearance'] for r in g if r['clearance'] is not None):.3f}")
        for level in ("single", "multi", "via", "iterate"):
            print(f"   {level:8s} {med(level):.3f}  vs A {rel(level, 'A'):+.2f}%  vs BIT {rel(level, 'bit'):+.2f}%")
        print(f"   A vs BIT {rel('A', 'bit'):+.2f}%;  best seq shared with A {med('shared_with_A'):.2f}, born after first "
              f"{med('born_after_first'):.2f} (A {med('A_born_after_first'):.2f})")


if __name__ == "__main__":
    main()
