"""Tube-RRT* over polyhedral SE(2) cells with region-gap nearest and line-witness overlaps.

The search keeps Tube-RRT*'s flow ``q_rand -> C_rand -> nearest -> TubeSteer -> NearConnect -> rewire``;
the directional cells only change which node captures a sample and how far the tree advances:

1. ``q_rand ~ U(SE(2))`` (goal bias as in ``TubeRRTConfig``); a colliding sample is dropped, otherwise one
   proximity query gives ``C_rand``.
2. **Region-gap nearest**: for every node ``v_i = q_rand - q_i``, ``D_i = ||v_i||_G``, ``u_i = v_i / D_i`` and
   ``g_i = D_i - l_i(u_i) - l_rand(-u_i)`` with the analytic ray lengths
   ``l(u) = min(r_g, L / |u_theta|, min_k e_k / (rho |u_theta| - n_k^T u_c))``; the node with the smallest
   gap captures the sample (``nearest = "point"`` uses ``argmin D_i`` instead).
3. **Line TubeSteer** along ``q_i + s u_i``: a seed ``q(s)`` with cell ``C(s)`` is accepted when the two cells'
   intervals on the line share ``w = min(s, l_i) - max(0, s - l_s(-u)) >= min_witness``.  ``s = D`` reuses
   ``C_rand``; retries move to ``l_i + l_prev(-u) - min_witness`` and finally inside ``C_i``.
4. The shared interval is the edge certificate: its midpoint ``q_w`` is a portal strictly inside both
   convex cells, the route ``q_i -> q_w -> q(s)`` has length ``s`` and ``min`` of the two slacks at ``q_w``
   is a d_G ball radius inside both cells (``r_portal``, the edge width).  No LP / SOCP.
5. **NearConnect / rewire** over nodes whose guard disks and yaw ranges meet ``C_new``: line witnesses for
   all of them first (vectorised); an exact LP / SOCP overlap only for a pair without witness whose
   optimistic cost ``cost + D`` could still change the parent or a rewire.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from formation.polyhedral_cell import PolyhedralCell, wrap_array
from formation.tube_cell_first_order import PortalCertificate
from formation.tube_rrt import TubeRRTConfig, TubeRRTNode, TubeRRTResult, TubeRRTTraceEvent
from formation.tube_rrt_frontier import FrontierConfig, PolyhedralFrontierPlanner
from formation.types import FormationSpec, MapData, Pose2D, wrap_to_pi

_INSIDE_TOL = 1e-12
_CENTER_CANDIDATES = (0.5, 0.25, 0.75, 0.0, 1.0)


@dataclass
class RegionTubeConfig:
    safety_distance: float = 0.02
    active_range: float = 1.0
    parallel_tolerance: float = math.radians(5.0)
    max_extent: float = 1.5
    yaw_chart: float = math.pi / 2.0
    cell_samples: int = 128
    outer_reach: float = 0.5
    nearest: str = "gap"
    """``"gap"``: region-gap nearest ``argmin g_i``; ``"point"``: ``argmin D_i`` (point Voronoi bias)."""
    min_witness: float = 0.05
    """``w_min`` [m, d_G]: shared line interval required from TubeSteer (seeds inside the parent are exempt)."""
    steer_attempts: int = 3
    steer_fraction: float = 0.9
    """The last TubeSteer attempt places the seed at ``steer_fraction * l_i`` inside the parent cell."""
    max_step: float = 1.0
    max_neighbors: int = 12
    exact_overlap: bool = True
    """Solve LP / SOCP for NearConnect / rewire pairs without a line witness that could still improve a cost."""
    colliding: str = "drop"
    """A colliding ``q_rand``: ``"drop"`` (Tube-RRT*) or ``"extend"`` (capture with ``l_rand = 0`` and steer towards it)."""

    def __post_init__(self) -> None:
        if self.nearest not in ("gap", "point"):
            raise ValueError("nearest must be 'gap' or 'point'")
        if self.colliding not in ("drop", "extend"):
            raise ValueError("colliding must be 'drop' or 'extend'")
        if self.min_witness < 0.0 or self.steer_attempts < 1 or not 0.0 < self.steer_fraction < 1.0:
            raise ValueError("min_witness >= 0, steer_attempts >= 1 and steer_fraction in (0, 1) are required")
        if self.max_step <= 0.0 or self.max_neighbors < 0:
            raise ValueError("max_step > 0 and max_neighbors >= 0 are required")

    def cell_config(self) -> FrontierConfig:
        return FrontierConfig(safety_distance=self.safety_distance, active_range=self.active_range,
                              parallel_tolerance=self.parallel_tolerance, max_extent=self.max_extent,
                              yaw_chart=self.yaw_chart, cell_samples=self.cell_samples, outer_reach=self.outer_reach,
                              max_parent_candidates=self.max_neighbors)


class RegionTubeRRTPlanner(PolyhedralFrontierPlanner):
    """Tube-RRT* with region-gap nearest, line TubeSteer and lazy exact overlaps (``cell_model = "polyhedral_tube"``)."""

    def __init__(self, map_data: MapData, formation: FormationSpec | np.ndarray, start: Pose2D,
                 goal_xy: tuple[float, float] | None = None, config: TubeRRTConfig | None = None,
                 tube_config: RegionTubeConfig | None = None) -> None:
        self.tube_config = tube_config or RegionTubeConfig()
        super().__init__(map_data, formation, start, goal_xy, config or TubeRRTConfig(cell_model="polyhedral_tube"),
                         self.tube_config.cell_config())

    def edge_cost(self, first: TubeRRTNode, second: TubeRRTNode, certificate: PortalCertificate) -> float:
        """Route length, optionally penalised by ``margin_weight / r_portal`` (the certified edge width)."""
        length = certificate.route_length
        if self.config.margin_weight <= 0.0:
            return length
        return length * (1.0 + self.config.margin_weight / max(certificate.slack, 1e-9))

    # -- vectorised directional geometry -------------------------------------------------------------------
    def _register_rows(self, index: int, cell: PolyhedralCell) -> None:
        super()._register_rows(index, cell)
        if len(self._row_e) < len(self._row_offset):
            self._row_e = np.concatenate((self._row_e, np.zeros(len(self._row_offset) - len(self._row_e))))
        start = self._row_start[index]
        self._row_e[start:start + len(cell.offsets)] = cell.offsets

    def _directions(self, nodes: np.ndarray, pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
        """Unit chart directions ``u_i`` (M, 3) from each node to ``pose`` and ``D_i = ||v_i||_G``."""
        v = np.column_stack((pose.x - self._x[nodes], pose.y - self._y[nodes], wrap_array(pose.yaw - self._yaw[nodes])))
        length = np.hypot(v[:, 0], v[:, 1]) + self.formation_radius * np.abs(v[:, 2])
        return v / np.maximum(length, 1e-300)[:, None], length

    def _row_blocks(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rows of ``nodes``: row indices, owner position in ``nodes``, nodes with rows, reduceat bounds."""
        sizes = self._row_size[nodes]
        has = np.flatnonzero(sizes)
        sizes = sizes[has]
        bounds = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(int)
        rows = np.repeat(self._row_start[nodes][has] - bounds, sizes) + np.arange(int(sizes.sum()))
        return rows, np.repeat(has, sizes), has, bounds

    def _node_extents(self, nodes: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """``l_i(u_i) = max{t >= 0 : q_i + t u_i in C_i}`` for unit chart directions (one per node)."""
        turn = self.formation_radius * np.abs(directions[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            lengths = np.minimum(self._guard[nodes],
                                 np.where(turn > 0.0, self.formation_radius * self._yaw_limit[nodes] / turn, math.inf))
            rows, owner, has, bounds = self._row_blocks(nodes)
            if len(rows):
                rate = turn[owner] - np.einsum("ij,ij->i", self._row_normal[rows], directions[owner, :2])
                limit = np.where(rate > 1e-15, self._row_e[rows] / rate, math.inf)
                lengths[has] = np.minimum(lengths[has], np.minimum.reduceat(limit, bounds))
        return np.maximum(lengths, 0.0)

    def _node_slack(self, nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Slack [m] of ``C_i`` at ``points[i]`` (one point per node), as ``PolyhedralCellModel.slack``."""
        dx, dy = points[:, 0] - self._x[nodes], points[:, 1] - self._y[nodes]
        turn = self.formation_radius * np.abs(wrap_array(points[:, 2] - self._yaw[nodes]))
        slack = np.minimum(self._guard[nodes] - np.hypot(dx, dy) - turn,
                           self.formation_radius * self._yaw_limit[nodes] - turn)
        rows, owner, has, bounds = self._row_blocks(nodes)
        if len(rows):
            values = (self._row_normal[rows, 0] * dx[owner] + self._row_normal[rows, 1] * dy[owner]
                      + self._row_e[rows] - turn[owner])
            slack[has] = np.minimum(slack[has], np.minimum.reduceat(values, bounds))
        return slack

    def _cell_extents(self, cell: PolyhedralCell, directions: np.ndarray) -> np.ndarray:
        """``l(u) = max{t >= 0 : q_0 + t u in C}`` of one cell for many unit chart directions."""
        turn = cell.rho * np.abs(directions[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            lengths = np.minimum(cell.guard, np.where(turn > 0.0, cell.rho * cell.yaw_limit / turn, math.inf))
            if len(cell.offsets):
                rate = turn[None, :] - cell.normals @ directions[:, :2].T
                limit = np.where(rate > 1e-15, cell.offsets[:, None] / rate, math.inf)
                lengths = np.minimum(lengths, limit.min(axis=0))
        return np.maximum(lengths, 0.0)

    def _witnesses(self, nodes: np.ndarray, cell: PolyhedralCell) -> tuple[list[PortalCertificate | None], np.ndarray]:
        """Line witnesses between each node's cell and ``cell`` (``None`` where the seed segment has no common point)."""
        u, length = self._directions(nodes, cell.pose)
        forward = self._node_extents(nodes, u)
        backward = self._cell_extents(cell, -u)
        lo, hi = np.maximum(0.0, length - backward), np.minimum(length, forward)
        witnesses: list[PortalCertificate | None] = [None] * len(nodes)
        found = np.flatnonzero((hi > lo) & (length > 0.0))
        if not len(found):
            return witnesses, length
        s = 0.5 * (lo[found] + hi[found])
        points = np.column_stack((self._x[nodes[found]] + s * u[found, 0], self._y[nodes[found]] + s * u[found, 1],
                                  wrap_array(self._yaw[nodes[found]] + s * u[found, 2])))
        slack = np.minimum(self._node_slack(nodes[found], points), self.cells.slack_many(cell, points))
        for k, index in enumerate(found.tolist()):
            if slack[k] > _INSIDE_TOL:
                witnesses[index] = PortalCertificate(Pose2D(*points[k]), float(slack[k]), float(length[index]))
        return witnesses, length

    # -- Tube-RRT* steps ------------------------------------------------------------------------------------
    def _nearest(self, sample: Pose2D, sample_cell: PolyhedralCell) -> tuple[int, np.ndarray, float, float] | None:
        """``(i*, u, D, l_i(u))`` of the capturing node, or ``None`` when every node coincides with the sample."""
        count = len(self._nodes)
        nodes = np.arange(count)
        u, length = self._directions(nodes, sample)
        blocked = length < self.config.min_metric_step
        if self._goal_nodes:
            blocked[list(self._goal_nodes)] = True
        if np.all(blocked):
            return None
        point = int(np.argmin(np.where(blocked, math.inf, length)))
        if self.tube_config.nearest == "point":
            best = point
            forward = float(self._node_extents(np.array([best]), u[best:best + 1])[0])
        else:
            forward_all = self._node_extents(nodes, u)
            backward = self._cell_extents(sample_cell, -u) if sample_cell.valid else 0.0
            gap = length - forward_all - backward
            best = int(np.argmin(np.where(blocked, math.inf, gap)))
            forward = float(forward_all[best])
            self._stats["gap_negative"] += int(gap[best] <= 0.0)
        self._stats["nearest_differs"] += int(best != point)
        return best, u[best], float(length[best]), forward

    def _tube_steer(self, parent: int, u: np.ndarray, length: float, forward: float,
                    sample_cell: PolyhedralCell) -> tuple[Pose2D, PolyhedralCell, PortalCertificate | None, int, str]:
        """Farthest seed on ``q_i + s u`` whose cell shares at least ``min_witness`` of the line with ``C_i``."""
        t = self.tube_config
        origin = self._nodes[parent].pose
        s = min(length, t.max_step)
        cell, pose, status = None, origin, "no_progress"
        for attempt in range(1, t.steer_attempts + 1):
            if attempt == t.steer_attempts and attempt > 1:
                s = min(s, t.steer_fraction * forward)
            if s < self.config.min_metric_step:
                return pose, cell, None, attempt, "no_progress"
            pose = Pose2D(origin.x + s * u[0], origin.y + s * u[1], wrap_to_pi(origin.yaw + s * u[2]))
            if s == length:
                cell = sample_cell
            else:
                cell = self.cells.make_cell(pose)
                self._stats["steer_cells"] += 1
            if not cell.valid and s == length:
                status, s = "collision", max(forward - t.min_witness, 0.0)
                continue
            backward = float(self._cell_extents(cell, -u[None, :])[0]) if cell.valid else 0.0
            lo, hi = max(0.0, s - backward), min(s, forward)
            if cell.valid and hi > lo and (hi - lo >= t.min_witness or s <= forward):
                m = 0.5 * (lo + hi)
                portal = Pose2D(origin.x + m * u[0], origin.y + m * u[1], wrap_to_pi(origin.yaw + m * u[2]))
                slack = min(float(self._node_slack(np.array([parent]), np.array([[portal.x, portal.y, portal.yaw]]))[0]),
                            self.cells.slack(cell, portal))
                if slack > _INSIDE_TOL:
                    return pose, cell, PortalCertificate(portal, slack, s), attempt, "added"
            status = "collision" if not cell.valid else "no_overlap"
            s = max(forward + backward - t.min_witness if s > forward else 0.5 * s, 0.0)
        return pose, cell, None, t.steer_attempts, status

    def _store(self, node: TubeRRTNode) -> int:
        index = super()._store(node)
        self.edge_kind[index] = "goal" if node.parent is not None and node.cell is self._nodes[node.parent].cell else ""
        return index

    def _reset(self) -> None:
        super()._reset()
        self.edge_kind: dict[int, str] = {}
        self._row_e = np.zeros(len(self._row_offset))
        self._stats = {"samples_colliding": 0, "gap_negative": 0, "nearest_differs": 0, "steer_cells": 0,
                       "steer_first_try": 0, "rejected_no_progress": 0, "rejected_collision": 0,
                       "rejected_no_overlap": 0, "parent_witness": 0, "near_improved": 0, "rewires": 0,
                       "portal_radius_sum": 0.0}
        for kind in ("near", "rewire"):
            for key in ("witness", "center_witness", "exact_calls", "exact_accept", "skipped_bound"):
                self._stats[f"{kind}_{key}"] = 0

    def _center_witness(self, first: PolyhedralCell, second: PolyhedralCell) -> PortalCertificate | None:
        """Best of a few points on the segment between the two in-circle centres (still no optimisation)."""
        a = np.array((first.pose.x + first.center_offset[0], first.pose.y + first.center_offset[1], first.pose.yaw))
        b = np.array((second.pose.x + second.center_offset[0], second.pose.y + second.center_offset[1],
                      first.pose.yaw + wrap_to_pi(second.pose.yaw - first.pose.yaw)))
        ts = np.asarray(_CENTER_CANDIDATES)
        points = a[None, :] + ts[:, None] * (b - a)[None, :]
        slack = np.minimum(self.cells.slack_many(first, points), self.cells.slack_many(second, points))
        best = int(np.argmax(slack))
        if slack[best] <= _INSIDE_TOL:
            return None
        portal = Pose2D(float(points[best, 0]), float(points[best, 1]), wrap_to_pi(float(points[best, 2])))
        return self.cells._certificate(first, second, portal, float(slack[best]))

    def _connect(self, candidate: int, cell: PolyhedralCell, witness: PortalCertificate | None,
                 kind: str) -> tuple[PortalCertificate | None, str]:
        """Certificate of the pair and how it was found: ``line`` / ``center`` witness or ``exact`` LP / SOCP."""
        if witness is not None:
            self._stats[f"{kind}_witness"] += 1
            return witness, "line"
        first, second = (self._nodes[candidate].cell, cell) if kind == "near" else (cell, self._nodes[candidate].cell)
        witness = self._center_witness(first, second)
        if witness is not None:
            self._stats[f"{kind}_center_witness"] += 1
            return witness, "center"
        if not self.tube_config.exact_overlap:
            return None, ""
        self._stats[f"{kind}_exact_calls"] += 1
        certificate = self.cells.exact_overlap(first, second)
        self._stats[f"{kind}_exact_accept"] += int(certificate is not None)
        return certificate, "exact"

    def plan(self) -> TubeRRTResult:
        config = self.config
        started = time.perf_counter()
        self._reset()
        start_cell = self.cells.make_cell(self.start)
        start_node = TubeRRTNode(self.start, start_cell, None, 0.0)
        if not start_cell.valid:
            return TubeRRTResult(False, tree_nodes=[start_node], failure_reason="start is in collision",
                                 overlap_stats={**self.cells.stats, **self._stats})
        start_cell.mode = "start"
        self._store(start_node)
        goal_indices: list[int] = []
        best_goal_distance = float(self._goal_distance(self.start.x, self.start.y))
        trace: list[TubeRRTTraceEvent] = []
        best_cost: float | None = None
        first_goal_iteration: int | None = None
        first_goal = {"first_goal_time_s": math.nan, "first_goal_pose_queries": -1, "first_goal_cost": math.nan}
        cost_history: list[tuple[int, float]] = []
        iteration = 0
        for iteration in range(1, config.max_iterations + 1):
            report = config.progress_interval > 0 and iteration % config.progress_interval == 0
            sample = self._sample_pose()
            sample_cell = self.cells.make_cell(sample)
            usable = sample_cell.valid or self.tube_config.colliding == "extend"
            self._stats["samples_colliding"] += int(not sample_cell.valid)
            picked = self._nearest(sample, sample_cell) if usable else None
            if picked is None:
                status = "collision" if not usable else "no_progress"
                self._stats["rejected_no_progress"] += int(usable)
                if config.record_trace:
                    trace.append(TubeRRTTraceEvent(iteration, sample, -1, sample, sample_cell.radius, status,
                                                   cell=sample_cell, mode="tube"))
                if report:
                    self._print_progress(iteration, best_goal_distance, best_cost)
                continue
            parent, u, length, forward = picked
            nearest_index = parent
            q_new, cell, certificate, attempts, status = self._tube_steer(parent, u, length, forward, sample_cell)
            if certificate is None:
                self._stats[f"rejected_{status}"] += 1
                if config.record_trace:
                    trace.append(TubeRRTTraceEvent(iteration, sample, parent, q_new, cell.radius if cell else 0.0,
                                                   status, attempts=attempts, cell=cell, mode="tube"))
                if report:
                    self._print_progress(iteration, best_goal_distance, best_cost)
                continue
            self._stats["steer_first_try"] += int(attempts == 1)
            self._stats["parent_witness"] += 1
            cell.mode = "tube"
            new_node = TubeRRTNode(q_new, cell, None, 0.0)
            parent_certificate, parent_kind = certificate, "steer"
            parent_cost = self._nodes[parent].cost + self.edge_cost(self._nodes[parent], new_node, certificate)

            neighbors, _ = self._neighbors(cell)
            neighbors = neighbors[neighbors != parent]
            witnesses, lengths = self._witnesses(neighbors, cell) if len(neighbors) else ([], np.empty(0))
            bounds = self._cost[neighbors] + lengths
            order = np.argsort(bounds, kind="stable").tolist()
            for position, k in enumerate(order):
                if bounds[k] >= parent_cost:
                    self._stats["near_skipped_bound"] += len(order) - position
                    break
                candidate = int(neighbors[k])
                certificate, how = self._connect(candidate, cell, witnesses[k], "near")
                if certificate is None:
                    continue
                cost = self._nodes[candidate].cost + self.edge_cost(self._nodes[candidate], new_node, certificate)
                if cost < parent_cost:
                    parent, parent_cost, parent_certificate, parent_kind = candidate, cost, certificate, how
                    self._stats["near_improved"] += 1
            new_node.parent, new_node.cost, new_node.certificate = parent, parent_cost, parent_certificate
            new_index = self._store(new_node)
            self.edge_kind[new_index] = parent_kind
            self._stats["portal_radius_sum"] += parent_certificate.slack
            event = TubeRRTTraceEvent(iteration, sample, nearest_index, q_new, cell.radius, "added", new_index, parent,
                                      attempts=attempts, cell=cell, mode="tube")
            if config.record_trace:
                trace.append(event)
            self.expansion_log.append({"iteration": iteration, "node": new_index, "mode": "tube", "source": nearest_index,
                                       "parent": parent, "parent_kind": parent_kind, "attempts": attempts,
                                       "step": float(parent_certificate.route_length if parent_kind == "steer" else
                                                     self.metric(self._nodes[nearest_index].pose, q_new)),
                                       "portal_radius": parent_certificate.slack, "active_pairs": cell.active_count,
                                       "broadphase_pairs": cell.broadphase_pairs, "yaw_limit": cell.yaw_limit})

            for k in range(len(neighbors)):
                candidate = int(neighbors[k])
                if candidate == 0 or candidate == parent:
                    continue
                if parent_cost + lengths[k] >= self._nodes[candidate].cost:
                    self._stats["rewire_skipped_bound"] += 1
                    continue
                certificate, how = self._connect(candidate, cell, witnesses[k], "rewire")
                if certificate is None:
                    continue
                cost = parent_cost + self.edge_cost(new_node, self._nodes[candidate], certificate)
                if cost < self._nodes[candidate].cost:
                    old_parent = self._nodes[candidate].parent
                    event.rewires.append((candidate, old_parent, new_index))
                    self._children[old_parent].remove(candidate)
                    self._children[new_index].append(candidate)
                    self._nodes[candidate].parent, self._nodes[candidate].certificate = new_index, certificate
                    self._nodes[candidate].cost = self._cost[candidate] = cost
                    self.edge_kind[candidate] = how
                    self._update_descendant_costs(candidate)
                    self._stats["rewires"] += 1

            goal_distance = float(self._goal_distance(q_new.x, q_new.y))
            best_goal_distance = min(best_goal_distance, goal_distance)
            if goal_indices:
                best_cost = min(self._nodes[i].cost for i in goal_indices)
                if best_cost < cost_history[-1][1]:
                    cost_history.append((iteration, best_cost))
            goal_pose = Pose2D(self.goal_xy[0], self.goal_xy[1], q_new.yaw)
            goal_slack = self.cells.slack(cell, goal_pose)
            if goal_slack > 0.0 and (best_cost is None or parent_cost + goal_distance < best_cost):
                goal_certificate = PortalCertificate(goal_pose, goal_slack, goal_distance)
                goal_node = TubeRRTNode(goal_pose, cell, new_index, 0.0, goal_certificate)
                goal_node.cost = parent_cost + self.edge_cost(new_node, goal_node, goal_certificate)
                goal_index = self._store(goal_node)
                goal_indices.append(goal_index)
                self._goal_nodes.add(goal_index)
                if config.record_trace:
                    trace.append(TubeRRTTraceEvent(iteration, goal_pose, new_index, goal_pose, cell.radius, "goal",
                                                   goal_index, new_index, cell=cell, mode="tube"))
                best_cost = goal_node.cost
                cost_history.append((iteration, best_cost))
                if first_goal_iteration is None:
                    first_goal_iteration = iteration
                    first_goal = {"first_goal_time_s": time.perf_counter() - started,
                                  "first_goal_pose_queries": self.cells.stats["pose_queries"],
                                  "first_goal_cost": best_cost}
                    if config.progress_interval > 0:
                        print(f"goal connected iteration={iteration} accepted_nodes={len(self._nodes)}", flush=True)
                if config.stop_on_first_goal:
                    break
            if report:
                self._print_progress(iteration, best_goal_distance, best_cost)

        nodes = self._nodes
        edges = [(i, node.parent) for i, node in enumerate(nodes) if node.parent is not None]
        added = len(nodes) - 1 - len(goal_indices)
        stats = {**self.cells.stats, **self._stats, **first_goal, "tube_nodes": added,
                 "portal_radius_mean": self._stats["portal_radius_sum"] / max(added, 1),
                 "plan_time_s": time.perf_counter() - started}
        if not goal_indices:
            if config.progress_interval > 0:
                print(f"search failed iteration budget exhausted iteration={iteration}/{config.max_iterations} "
                      f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}", flush=True)
            return TubeRRTResult(False, tree_nodes=nodes, tree_edges=edges, failure_reason="iteration budget exhausted",
                                 trace=trace, iterations=iteration, overlap_stats=stats)
        best_goal = min(goal_indices, key=lambda i: nodes[i].cost)
        if nodes[best_goal].cost < cost_history[-1][1]:
            cost_history.append((iteration, nodes[best_goal].cost))
        route, indices = self._path(best_goal)
        radii = [float(v) for v in self.clearances([p.x for p in route], [p.y for p in route], [p.yaw for p in route])]
        if config.progress_interval > 0 and not config.stop_on_first_goal:
            print(f"search finished iteration={iteration} accepted_nodes={len(nodes)} goal_nodes={len(goal_indices)} "
                  f"best_cost={nodes[best_goal].cost:.3f}", flush=True)
        return TubeRRTResult(True, route, radii, nodes, edges, bottleneck=self.route_clearance(route), path_nodes=indices,
                             trace=trace, iterations=iteration, first_goal_iteration=first_goal_iteration,
                             path_cost=nodes[best_goal].cost, cost_history=cost_history, overlap_stats=stats)

    def _print_progress(self, iteration: int, best_goal_distance: float, best_cost: float | None) -> None:
        cost = "" if best_cost is None else f" best_cost={best_cost:.3f}"
        print(f"progress iteration={iteration}/{self.config.max_iterations} accepted_nodes={len(self._nodes)} "
              f"best_goal_distance={best_goal_distance:.3f}{cost}", flush=True)
