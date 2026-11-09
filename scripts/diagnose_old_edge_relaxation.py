#!/usr/bin/env python3
"""Offline fixed-vertex short-edge relaxation for forest RRT*, seed 0."""
import heapq
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_coupling import scenario
from formation.baselines import BaselineConfig, RRTStar, distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formula-radius', action='store_true',
                        help='Use the unclipped final-n radius on the SAME original point set.')
    args = parser.parse_args()
    data, slots, start = scenario('forest')
    cfg = BaselineConfig(seed=0, clearance=.02, edge_resolution=.02, step=.3,
                         batch_size=200, goal_bias=.1, rewire_factor=1.1)
    planner = RRTStar(data, slots, start, cfg)
    assert planner.check.valid(planner.start)[0]
    t = time.perf_counter()
    for _ in range(6000):
        planner.step()
    replay_s = time.perf_counter() - t
    n = planner.n
    Q = planner.Q[:n].copy()
    old = planner.cost[:n].copy()
    parent = planner.parent
    old_goal = planner.best_path()[0]
    radius = planner.gamma * (math.log(n + 1) / (n + 1)) ** (1. / 3.) if args.formula_radius else .3
    t = time.perf_counter()
    # Three yaw charts make the Euclidean KD tree exact for wrapped angular distance.
    z = Q.copy()
    z[:, 2] = (z[:, 2] + math.pi) % (2 * math.pi) * planner.rho
    period = 2 * math.pi * planner.rho
    extended = np.concatenate([z + [0, 0, -period], z, z + [0, 0, period]])
    tree = cKDTree(extended)
    neighbors = tree.query_ball_point(z, radius + 1e-12)
    pairs = set()
    for u, row in enumerate(neighbors):
        for index in row:
            v = index % n
            if u != v:
                pairs.add((min(u, v), max(u, v)))
    pairs = sorted((u, v, distance(Q[u], Q[v], planner.rho)) for u, v in pairs)
    pairs = [(u, v, d) for u, v, d in pairs if d <= radius + 1e-12]
    adjacency = [[] for _ in range(n)]
    for u, v, d in pairs:
        adjacency[u].append((v, d))
        adjacency[v].append((u, d))
    # Current tree edges are retained even if the adaptive radius allowed a longer edge.
    tree_edges = {tuple(sorted((u, parent[u]))) for u in range(1, n)}
    pair_keys = {(u, v) for u, v, _ in pairs}
    for u, v in tree_edges - pair_keys:
        d = distance(Q[u], Q[v], planner.rho)
        adjacency[u].append((v, d))
        adjacency[v].append((u, d))
    graph_s = time.perf_counter() - t
    cache = {edge: True for edge in tree_edges}
    checks = 0
    check_s = 0.
    def feasible(u, v, d):
        nonlocal checks, check_s
        key = (min(u, v), max(u, v))
        if key not in cache:
            begin = time.perf_counter()
            cache[key] = planner.edge_valid(Q[u], Q[v], d)
            check_s += time.perf_counter() - begin
            checks += 1
        return cache[key]

    t = time.perf_counter()
    residual = []
    residual_candidates = 0
    for u, v, d in pairs:
        if abs(old[u] - old[v]) <= d + 1e-12:
            continue
        source, target = (u, v) if old[u] < old[v] else (v, u)
        residual_candidates += 1
        if feasible(source, target, d):
            residual.append((source, target, float(old[target] - old[source] - d)))
    residual_s = time.perf_counter() - t
    residual_checks, residual_check_s = checks, check_s
    t = time.perf_counter()
    dist = np.full(n, np.inf)
    dist[0] = 0.
    queue = [(0., 0)]
    settled = np.zeros(n, dtype=bool)
    while queue:
        g, u = heapq.heappop(queue)
        if settled[u]:
            continue
        settled[u] = True
        for v, d in adjacency[u]:
            if not settled[v] and g + d < dist[v] - 1e-12 and feasible(u, v, d):
                dist[v] = g + d
                heapq.heappush(queue, (dist[v], v))
    dijkstra_s = time.perf_counter() - t
    new_goal = float(np.min(dist[planner.goal_nodes] + np.asarray(planner.goal_legs)))
    result = dict(scenario='forest', seed=0, proposals=6000, vertices=n, radius=radius,
                  edge_resolution=.02, rho=planner.rho, known_goal_edges=len(planner.goal_nodes),
                  current_goal_cost=old_goal, fixed_graph_goal_cost=new_goal,
                  reduction_pct=100 * (old_goal - new_goal) / old_goal,
                  neighbor_pairs=len(pairs), tree_edges=len(tree_edges),
                  tree_edges_outside_radius=len(tree_edges - pair_keys),
                  residual_candidate_pairs=residual_candidates, residual_feasible_pairs=len(residual),
                  residual_max_gain=max((x[2] for x in residual), default=0.),
                  offline_checked_pairs=checks, offline_feasible_checked_pairs=sum(cache.values()) - len(tree_edges),
                  offline_blocked_checked_pairs=sum(not x for x in cache.values()),
                  residual_collision_checks=residual_checks, residual_collision_check_s=residual_check_s,
                  total_collision_check_s=check_s, replay_s=replay_s, graph_construction_s=graph_s,
                  residual_phase_s=residual_s, dijkstra_s=dijkstra_s,
                  settled_vertices=int(settled.sum()), improved_vertices=int(np.sum(dist < old - 1e-9)),
                  max_tree_cost_violation=float(np.max(dist - old)),
                  interpretation='Offline fixed points, existing tree edges and validated radius edges; no new goal edges or sampling. Not online performance.')
    suffix = '_formula_radius' if args.formula_radius else ''
    out = ROOT / f'results/exit_baselines/diagnostics/old_edge_relaxation{suffix}.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
