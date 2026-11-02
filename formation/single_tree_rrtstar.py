"""Fixed-anchor, shared-certificate single-tree RRT*.

The default neighbourhood is every state in a fixed radius, plus a reachable
owner fallback.  This is the reference implementation, without k-log truncation,
utility rejection or approximate state merging.  Dynamic certificate discovery
does not inherit the conditional fixed-cover theorem in method/.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from formation.region_coupling import CouplingConfig, _CorePlanner, _Program
from formation.types import Pose2D, wrap_to_pi


@dataclass
class SingleTreeConfig(CouplingConfig):
    neighbor_radius: float = 1.5
    goal_margin: float = 0.0


class SingleTreeRRTStar(_CorePlanner):
    variant = "single_tree_rrtstar"

    def initialize(self) -> None:
        cell = self.model.make_cell(self.start)
        if not cell.valid:
            raise ValueError("start is in collision")
        self._add_cell(cell, None)
        self.anchor = [self.start]
        self.owner = [0]
        self.parent = [None]
        self.children = [set()]
        self.cost = [0.0]
        self.edge_cost = [0.0]
        self.portal = [None]
        self.owner_nodes = [[0]]
        self.exact = {(0, self.start.x, self.start.y, self.start.yaw): 0}
        self.pair_witness = {}
        self.goal_nodes = []
        self._add_goal(0)
        self.stats.update(nodes=1, covered_insertions=0, exact_reconsider=0,
                          rewires=0, choose_parent_tests=0, rewire_tests=0,
                          endpoint_edges=0, same_owner_edges=0, via_edges=0,
                          witness_fallbacks=0, coverage_time_s=0.0,
                          propagation_time_s=0.0)

    def _point(self, pose, owner, reference):
        """Lift via its own certificate chart, then align the owner seed.

        A direct wrap relative to the other endpoint can select a different
        homotopy at the +/-pi seam and understate the certified segment length.
        """
        cell, ref = self.cells[owner], self.cells[reference]
        seed_yaw = ref.pose.yaw + wrap_to_pi(cell.pose.yaw - ref.pose.yaw)
        yaw = seed_yaw + wrap_to_pi(pose.yaw - cell.pose.yaw)
        return np.array((pose.x, pose.y, self.rho * yaw))

    def _segment(self, a, b, owner):
        return float(np.linalg.norm(self._point(a, owner, owner) - self._point(b, owner, owner)))

    def _portal_valid(self, portal, first, second, tolerance=1e-7):
        margin = self.config.overlap_margin
        return (min(self.geo.slack(self.cells[first], portal), self.geo.slack(self.cells[second], portal))
                >= margin - tolerance
                and abs(self._point(portal, first, first)[2] - self._point(portal, second, first)[2]) < 1e-7)

    def _edge(self, u, v, limit=math.inf):
        a, b = self.anchor[u], self.anchor[v]
        i, j = self.owner[u], self.owner[v]
        za, zb = self._point(a, i, i), self._point(b, j, i)
        lower = float(np.linalg.norm(za - zb))
        if lower >= limit - 1e-10:
            return None
        if i == j:
            self.stats["same_owner_edges"] += 1
            return lower, None
        # The endpoint test is essential: a persistent pair witness alone can
        # impose a non-vanishing detour even on arbitrarily close samples.
        for p in (a, b):
            if self._portal_valid(p, i, j, tolerance=0.0):
                self.stats["endpoint_edges"] += 1
                return lower, p
        key = tuple(sorted((i, j)))
        if key not in self.pair_witness:
            if self._separated(self.cells[i], self.cells[j]):
                self.pair_witness[key] = None
            else:
                self.pair_witness[key] = self._overlap(self.cells[i], self.cells[j])
        witness = self.pair_witness[key]
        if witness is None or not self._portal_valid(witness, i, j):
            return None
        program = _Program(5)
        program.cell(self.cells[i], self._point(self.cells[i].pose, i, i), 0, self.config.overlap_margin)
        program.cell(self.cells[j], self._point(self.cells[j].pose, j, i), 0, self.config.overlap_margin)
        program.distance(3, 0, za)
        program.distance(4, 0, zb)
        result = self.geo._run(program, np.array((0., 0., 0., 1., 1.)))
        portal = None if result is None else self.geo.pose(result[:3])
        if portal is None or not self._portal_valid(portal, i, j):
            portal = witness
            self.stats["witness_fallbacks"] += 1
        else:
            self.stats["via_edges"] += 1
        cost = self._segment(a, portal, i) + self._segment(portal, b, j)
        return (cost, portal) if cost < limit - 1e-10 else None

    def _covering(self, sample):
        """Query every known certificate, without a top-K or nearest-owner filter."""
        n = len(self.cells)
        dx, dy = sample.x - self._x[:n], sample.y - self._y[:n]
        turn = self.rho * np.abs((sample.yaw - self._t[:n] + math.pi) % (2 * math.pi) - math.pi)
        possible = np.flatnonzero((np.hypot(dx, dy) + turn <= self._g[:n]) & (turn <= self._lim[:n]))
        return [int(i) for i in possible if self.geo.slack(self.cells[int(i)], sample) >= 0.0]

    def _add_goal(self, node):
        p, i = self.anchor[node], self.owner[node]
        yaw = self.geo.goal_yaw(self.cells[i], self.goal_xy, p.yaw, self.config.goal_margin)
        if yaw is not None:
            goal = Pose2D(*self.goal_xy, yaw)
            self.goal_nodes.append((node, goal, self._segment(p, goal, i)))

    def _ancestors(self, node):
        ancestors = set()
        while node is not None:
            ancestors.add(node)
            node = self.parent[node]
        return ancestors

    def _reparent(self, node, parent, edge):
        old = self.parent[node]
        if old is not None:
            self.children[old].remove(node)
        self.parent[node] = parent
        self.children[parent].add(node)
        self.edge_cost[node], self.portal[node] = edge
        started = time.perf_counter()
        stack = [node]
        while stack:
            k = stack.pop()
            self.cost[k] = self.cost[self.parent[k]] + self.edge_cost[k]
            stack.extend(self.children[k])
        self.stats["propagation_time_s"] += time.perf_counter() - started

    def _connect_state(self, node, fallback):
        q = self.anchor[node]
        poses = np.array([(p.x, p.y, p.yaw) for p in self.anchor])
        delta = poses - (q.x, q.y, q.yaw)
        delta[:, 2] = self.rho * ((delta[:, 2] + math.pi) % (2 * math.pi) - math.pi)
        distance = np.linalg.norm(delta, axis=1)
        near = np.flatnonzero(distance <= self.config.neighbor_radius).tolist()
        candidates = set(near + [fallback])
        # Exact duplicates reconsider their fixed state, preserving their old
        # parent and excluding the complete descendant set from ChooseParent.
        forbidden, stack = set(), [node]
        while stack:
            k = stack.pop()
            forbidden.add(k)
            stack.extend(self.children[k])
        if self.parent[node] is not None:
            candidates.add(self.parent[node])
        best, best_edge, best_cost = self.parent[node], None, self.cost[node]
        for u in sorted(candidates - forbidden, key=lambda u: self.cost[u] + distance[u]):
            if self.cost[u] + distance[u] >= best_cost - 1e-10:
                continue
            self.stats["choose_parent_tests"] += 1
            edge = self._edge(u, node, best_cost - self.cost[u])
            if edge is not None and self.cost[u] + edge[0] < best_cost - 1e-10:
                best, best_edge, best_cost = u, edge, self.cost[u] + edge[0]
        if best_edge is not None:
            self._reparent(node, best, best_edge)
        assert math.isfinite(self.cost[node]), "reachable owner fallback must connect"
        ancestors = self._ancestors(node)
        for v in near:
            if v in ancestors or self.cost[node] + distance[v] >= self.cost[v] - 1e-10:
                continue
            self.stats["rewire_tests"] += 1
            edge = self._edge(node, v, self.cost[v] - self.cost[node])
            if edge is not None:
                self._reparent(v, node, edge)
                self.stats["rewires"] += 1

    def step(self):
        sample = self._sample()
        started = time.perf_counter()
        covered = self._covering(sample)
        self.stats["coverage_time_s"] += time.perf_counter() - started
        if covered:
            i = int(self.rng.choice(covered))  # fair owner, independent of g/utility
            fallback = self.owner_nodes[i][0]
            anchor = sample
            self.stats["covered_insertions"] += 1
        else:
            started = time.perf_counter()
            # Nearest acts on the very same fixed-anchor tree as rewiring.
            fallback = min(range(len(self.anchor)), key=lambda v: self.geo.distance(self.anchor[v], sample))
            owner = self.owner[fallback]
            self.stats["select_time_s"] += time.perf_counter() - started
            cell, anchor, _ = self._steer(owner, sample)
            if cell is None:
                return
            i = self._add_cell(cell, owner)
            self.owner_nodes.append([])
            self.pair_witness[tuple(sorted((owner, i)))] = anchor
        key = (i, anchor.x, anchor.y, anchor.yaw)
        if key in self.exact:
            node = self.exact[key]
            self.stats["exact_reconsider"] += 1
        else:
            node = len(self.anchor)
            self.exact[key] = node
            self.anchor.append(anchor)
            self.owner.append(i)
            self.parent.append(None)
            self.children.append(set())
            self.cost.append(math.inf)
            self.edge_cost.append(math.inf)
            self.portal.append(None)
            self.owner_nodes[i].append(node)
            self._add_goal(node)
        started, solves = time.perf_counter(), self.geo.solves
        self._connect_state(node, fallback)
        self.stats["connect_time_s"] += time.perf_counter() - started
        self.stats["connect_solves"] += self.geo.solves - solves
        self.stats["nodes"] = len(self.anchor)

    def current_path(self):
        if not self.goal_nodes:
            return None
        node, goal, terminal = min(self.goal_nodes, key=lambda g: self.cost[g[0]] + g[2])
        total = self.cost[node] + terminal
        sequence, cursor = [], node
        while cursor is not None:
            sequence.append(cursor)
            cursor = self.parent[cursor]
        sequence.reverse()
        route = [self.start]
        owners = [self.owner[sequence[0]]]
        for v in sequence[1:]:
            if self.portal[v] is not None:
                route.append(self.portal[v])
            route.append(self.anchor[v])
            if self.owner[v] != owners[-1]:
                owners.append(self.owner[v])
        route.append(goal)
        return total, route, owners, sequence

    def _checkpoint(self, iteration, online):
        current = self.current_path()
        record = dict(iteration=iteration, online_time_s=online, nodes=len(self.anchor),
                      regions=len(self.cells), success=current is not None)
        if current is not None:
            cost, route, owners, sequence = current
            record.update(raw_cost=cost, optimised_cost=cost, chain_regions=len(owners),
                          extract_time_s=0., optimise_time_s=0., sequence=sequence, owners=owners,
                          route=[(p.x, p.y, p.yaw) for p in route])
        return record

    def plan(self):
        self.initialize()
        first, checkpoints = None, []
        started = time.perf_counter()
        for iteration in range(1, self.config.max_iterations + 1):
            self.step()
            self.stats["iterations"] = iteration
            if first is None and self.goal_nodes:
                first = self._checkpoint(iteration, time.perf_counter() - started)
                first["kind"] = "first"
            if iteration in self.config.checkpoints:
                checkpoints.append(self._checkpoint(iteration, time.perf_counter() - started))
        return dict(variant=self.variant, success=bool(self.goal_nodes), first=first,
                    checkpoints=checkpoints, regions=len(self.cells), nodes=len(self.anchor),
                    online_time_s=time.perf_counter() - started, solver_failures=self.geo.failures,
                    stats=dict(self.stats))
