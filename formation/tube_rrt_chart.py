"""Joint-state RRT* over chart cells: first-order (``tube_cell_first_order``) or second-order
(``tube_cell_second_order``).  The orientation-sliced cell stays in ``tube_rrt.TubeRRTPlanner``;
``make_tube_rrt_planner`` picks the planner from ``TubeRRTConfig.cell_model``.
"""
from __future__ import annotations

import math

import numpy as np

from formation.tube_cell_first_order import FirstOrderCellModel, PortalCertificate, TubeCell, interpolate_pose
from formation.tube_cell_second_order import SecondOrderCellModel
from formation.tube_rrt import (
    TubeRRTConfig,
    TubeRRTNode,
    TubeRRTPlanner,
    TubeRRTResult,
    TubeRRTTraceEvent,
)
from formation.types import FormationSpec, MapData, Pose2D

CELL_MODELS = {model.name: model for model in (FirstOrderCellModel, SecondOrderCellModel)}


def make_tube_rrt_planner(map_data: MapData, formation: FormationSpec | np.ndarray, start: Pose2D,
                          goal_xy: tuple[float, float] | None = None,
                          config: TubeRRTConfig | None = None) -> TubeRRTPlanner | ChartCellTubeRRTPlanner:
    config = config or TubeRRTConfig()
    planner = TubeRRTPlanner if config.cell_model == "orientation" else ChartCellTubeRRTPlanner
    return planner(map_data, formation, start, goal_xy, config)


class ChartCellTubeRRTPlanner:
    """Joint-state RRT* over first- or second-order chart cells (``TubeRRTConfig.cell_model``)."""

    def __init__(self, map_data: MapData, formation: FormationSpec | np.ndarray, start: Pose2D,
                 goal_xy: tuple[float, float] | None = None,
                 config: TubeRRTConfig | None = None) -> None:
        self.map_data = map_data
        self.slots = np.asarray(formation.slots if isinstance(formation, FormationSpec) else formation, dtype=float)
        if self.slots.ndim != 2 or self.slots.shape[1] != 2 or len(self.slots) == 0:
            raise ValueError("formation slots must have shape (N, 2)")
        self.start = start
        self.goal_xy = map_data.goal_xy if goal_xy is None else tuple(goal_xy)
        self.config = config or TubeRRTConfig()
        if self.config.cell_model not in CELL_MODELS:
            raise ValueError(f"cell_model must be one of {sorted(CELL_MODELS)}")
        self.cells = CELL_MODELS[self.config.cell_model](self.slots, self.config.cell_eta)
        self.robot_radius = float(map_data.robot_radius)
        self.safety_margin = self.config.safety_margin + float(map_data.safety_margin)
        self.formation_radius = float(np.max(np.linalg.norm(self.slots, axis=1)))
        unsupported = [p.get("type") for p in map_data.obstacle_primitives if p.get("type") != "circle"]
        if unsupported:
            raise ValueError("TubeRRTPlanner supports only circle obstacle primitives")
        self._origin = np.asarray(map_data.origin_xy, dtype=float)
        self._upper = self._origin + (float(map_data.width_m), float(map_data.height_m))
        self._circle_centers = np.asarray([p["center_xy"] for p in map_data.obstacle_primitives],
                                          dtype=float).reshape((-1, 2))
        self._circle_radii = np.asarray([float(p["radius"]) for p in map_data.obstacle_primitives], dtype=float)
        self._clearance_margin = self.robot_radius + self.safety_margin
        self.rng = np.random.default_rng(self.config.seed)

    # -- geometry --------------------------------------------------------------------------------
    def metric(self, first: Pose2D, second: Pose2D) -> float:
        """``||second - first||_F`` in the chart of ``first`` (first-order max robot displacement)."""
        return self.cells.distance(first, second)

    def clearances(self, xs: np.ndarray, ys: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        """Guarded clearance ``d_obs`` (min over robots, minus robot radius and margin) for many poses."""
        xs, ys, yaws = (np.asarray(v, dtype=float).reshape(-1) for v in (xs, ys, yaws))
        c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        px = xs[:, None] + c * self.slots[None, :, 0] - s * self.slots[None, :, 1]
        py = ys[:, None] + s * self.slots[None, :, 0] + c * self.slots[None, :, 1]
        minimum = np.minimum.reduce((px - self._origin[0], self._upper[0] - px,
                                     py - self._origin[1], self._upper[1] - py))
        if len(self._circle_radii):
            dx = px[:, :, None] - self._circle_centers[None, None, :, 0]
            dy = py[:, :, None] - self._circle_centers[None, None, :, 1]
            minimum = np.minimum(minimum, np.min(np.hypot(dx, dy) - self._circle_radii, axis=2))
        return np.min(minimum, axis=1) - self._clearance_margin

    def clearance(self, pose: Pose2D) -> float:
        return float(self.clearances(pose.x, pose.y, pose.yaw)[0])

    def safety_cell(self, pose: Pose2D) -> TubeCell:
        return self.cells.make_cell(pose, self.clearance(pose))

    def cell_overlap(self, first: TubeRRTNode | TubeCell, second: TubeRRTNode | TubeCell) -> PortalCertificate | None:
        first = first.cell if isinstance(first, TubeRRTNode) else first
        second = second.cell if isinstance(second, TubeRRTNode) else second
        return self.cells.overlap(first, second)

    def densify(self, poses: list[Pose2D], step: float | None = None) -> list[Pose2D]:
        """Chart-linear interpolation of a route with at most ``step`` F-norm per sample."""
        step = self.config.route_check_step if step is None else step
        dense = poses[:1]
        for a, b in zip(poses, poses[1:]):
            count = max(1, int(math.ceil(self.metric(a, b) / step)))
            dense += [interpolate_pose(a, b, k / count) for k in range(1, count + 1)]
        return dense

    def route_clearance(self, poses: list[Pose2D]) -> float:
        dense = self.densify(poses)
        if not dense:
            return 0.0
        return float(np.min(self.clearances([p.x for p in dense], [p.y for p in dense], [p.yaw for p in dense])))

    # -- search ----------------------------------------------------------------------------------
    def _sample_pose(self) -> Pose2D:
        if self.rng.random() < self.config.goal_bias:
            return Pose2D(self.goal_xy[0], self.goal_xy[1], float(self.rng.uniform(-math.pi, math.pi)))
        ox, oy = self.map_data.origin_xy
        return Pose2D(float(self.rng.uniform(ox, ox + self.map_data.width_m)),
                      float(self.rng.uniform(oy, oy + self.map_data.height_m)),
                      float(self.rng.uniform(-math.pi, math.pi)))

    def _extend(self, near: TubeRRTNode, target: Pose2D) -> tuple[Pose2D, TubeCell, int, PortalCertificate | None]:
        distance = self.metric(near.pose, target)
        step = min(self.config.metric_step, distance)
        floor = 0.9 * self.cells.inner_step(near.cell, target)
        attempts = 0
        while True:
            attempts += 1
            pose = target if step >= distance else interpolate_pose(near.pose, target, step / distance)
            cell = self.safety_cell(pose)
            certificate = self.cells.overlap(near.cell, cell) if cell.radius > 0.0 else None
            if certificate is not None:
                return pose, cell, attempts, certificate
            next_step = max(0.5 * step, floor)
            if not self.config.step_backoff or next_step >= step or next_step < self.config.min_metric_step:
                return pose, cell, attempts, None
            step = next_step

    def edge_cost(self, first: TubeRRTNode, second: TubeRRTNode, certificate: PortalCertificate) -> float:
        length = certificate.route_length
        if self.config.margin_weight <= 0.0:
            return length
        return length * (1.0 + self.config.margin_weight / max(min(first.radius, second.radius), 1e-9))

    def _update_descendant_costs(self, nodes: list[TubeRRTNode], root_index: int,
                                 children: list[list[int]], node_cost: np.ndarray) -> None:
        stack = list(children[root_index])
        while stack:
            index = stack.pop()
            node = nodes[index]
            node.cost = nodes[node.parent].cost + self.edge_cost(nodes[node.parent], node, node.certificate)
            node_cost[index] = node.cost
            stack.extend(children[index])

    def _path(self, nodes: list[TubeRRTNode], index: int) -> tuple[list[Pose2D], list[int]]:
        indices: list[int] = []
        while index is not None:
            indices.append(index)
            index = nodes[index].parent
        indices.reverse()
        route = [nodes[indices[0]].pose]
        for child in indices[1:]:
            for pose in (nodes[child].certificate.portal, nodes[child].pose):
                if pose != route[-1]:
                    route.append(pose)
        return route, indices

    def _print_progress(self, iteration: int, nodes: list[TubeRRTNode], best_goal_distance: float,
                        best_cost: float | None) -> None:
        cost = "" if best_cost is None else f" best_cost={best_cost:.3f}"
        print(f"progress iteration={iteration}/{self.config.max_iterations} accepted_nodes={len(nodes)} "
              f"best_goal_distance={best_goal_distance:.3f}{cost}", flush=True)

    def plan(self) -> TubeRRTResult:
        config = self.config
        start_cell = self.safety_cell(self.start)
        nodes = [TubeRRTNode(self.start, start_cell, None, 0.0)]
        if start_cell.radius <= 0.0:
            return TubeRRTResult(False, tree_nodes=nodes, failure_reason="start is in collision")
        capacity = 2 * config.max_iterations + 3
        node_x, node_y, node_yaw, node_cos, node_sin, node_cost = (np.empty(capacity) for _ in range(6))

        def store(index: int, pose: Pose2D, cost: float) -> None:
            node_x[index], node_y[index], node_yaw[index] = pose.x, pose.y, pose.yaw
            node_cos[index], node_sin[index], node_cost[index] = math.cos(pose.yaw), math.sin(pose.yaw), cost

        def distances_to(pose: Pose2D) -> np.ndarray:
            count = len(nodes)
            return self.cells.distances_to(node_x[:count], node_y[:count], node_cos[:count], node_sin[:count],
                                           node_yaw[:count], pose)

        store(0, self.start, 0.0)
        children: list[list[int]] = [[]]
        best_goal_distance = math.hypot(self.start.x - self.goal_xy[0], self.start.y - self.goal_xy[1])
        trace: list[TubeRRTTraceEvent] = []
        goal_indices: list[int] = []
        best_cost: float | None = None
        first_goal_iteration: int | None = None
        cost_history: list[tuple[int, float]] = []
        iteration = 0
        for iteration in range(1, config.max_iterations + 1):
            report = config.progress_interval > 0 and iteration % config.progress_interval == 0
            sampled = self._sample_pose()
            nearest_index = int(np.argmin(distances_to(sampled)))
            pose, cell, attempts, nearest_certificate = self._extend(nodes[nearest_index], sampled)
            if nearest_certificate is None:
                if config.record_trace:
                    trace.append(TubeRRTTraceEvent(iteration, sampled, nearest_index, pose, cell.radius,
                                                   "collision" if cell.clearance <= 0.0 else "no_overlap",
                                                   attempts=attempts, cell=cell))
                if report:
                    self._print_progress(iteration, nodes, best_goal_distance, best_cost)
                continue

            new_node = TubeRRTNode(pose, cell, None, 0.0)
            distances = distances_to(pose)
            neighbors = np.flatnonzero(distances <= config.neighbor_radius)
            lower_bounds = node_cost[neighbors] + distances[neighbors]
            neighbors, lower_bounds = neighbors[np.argsort(lower_bounds, kind="stable")], np.sort(lower_bounds)
            parent, parent_certificate = nearest_index, nearest_certificate
            parent_cost = nodes[parent].cost + self.edge_cost(nodes[parent], new_node, parent_certificate)
            for candidate, bound in zip(neighbors.tolist(), lower_bounds.tolist()):
                if bound >= parent_cost:
                    break
                if candidate == nearest_index:
                    continue
                certificate = self.cells.overlap(nodes[candidate].cell, cell)
                if certificate is None:
                    continue
                candidate_cost = nodes[candidate].cost + self.edge_cost(nodes[candidate], new_node, certificate)
                if candidate_cost < parent_cost:
                    parent, parent_cost, parent_certificate = candidate, candidate_cost, certificate
            new_index = len(nodes)
            new_node.parent, new_node.cost, new_node.certificate = parent, parent_cost, parent_certificate
            nodes.append(new_node)
            store(new_index, pose, parent_cost)
            children.append([])
            children[parent].append(new_index)
            best_goal_distance = min(best_goal_distance, math.hypot(pose.x - self.goal_xy[0], pose.y - self.goal_xy[1]))
            event = TubeRRTTraceEvent(iteration, sampled, nearest_index, pose, cell.radius, "added", new_index,
                                      parent, attempts=attempts, cell=cell)
            if config.record_trace:
                trace.append(event)

            for candidate in neighbors.tolist():
                if candidate in (parent, 0) or parent_cost + distances[candidate] >= nodes[candidate].cost:
                    continue
                certificate = self.cells.overlap(cell, nodes[candidate].cell)
                if certificate is None:
                    continue
                new_cost = parent_cost + self.edge_cost(new_node, nodes[candidate], certificate)
                if new_cost < nodes[candidate].cost:
                    old_parent = nodes[candidate].parent
                    event.rewires.append((candidate, old_parent, new_index))
                    children[old_parent].remove(candidate)
                    children[new_index].append(candidate)
                    nodes[candidate].parent, nodes[candidate].certificate, nodes[candidate].cost = (
                        new_index, certificate, new_cost)
                    node_cost[candidate] = new_cost
                    self._update_descendant_costs(nodes, candidate, children, node_cost)

            if goal_indices:
                best_cost = min(nodes[i].cost for i in goal_indices)
                if best_cost < cost_history[-1][1]:
                    cost_history.append((iteration, best_cost))
            goal_distance = math.hypot(pose.x - self.goal_xy[0], pose.y - self.goal_xy[1])
            if goal_distance <= config.goal_connect_distance and (best_cost is None or parent_cost + goal_distance < best_cost):
                goal_pose = Pose2D(self.goal_xy[0], self.goal_xy[1], pose.yaw)
                goal_cell = self.safety_cell(goal_pose)
                goal_certificate = self.cells.overlap(cell, goal_cell)
                if goal_certificate is not None:
                    goal_node = TubeRRTNode(goal_pose, goal_cell, new_index, 0.0, goal_certificate)
                    goal_node.cost = parent_cost + self.edge_cost(new_node, goal_node, goal_certificate)
                    if best_cost is None or goal_node.cost < best_cost:
                        goal_index = len(nodes)
                        nodes.append(goal_node)
                        store(goal_index, goal_pose, goal_node.cost)
                        children.append([])
                        children[new_index].append(goal_index)
                        goal_indices.append(goal_index)
                        if config.record_trace:
                            trace.append(TubeRRTTraceEvent(iteration, goal_pose, new_index, goal_pose, goal_cell.radius,
                                                           "goal", goal_index, new_index, cell=goal_cell))
                        best_cost = goal_node.cost
                        cost_history.append((iteration, best_cost))
                        if first_goal_iteration is None:
                            first_goal_iteration = iteration
                            if config.progress_interval > 0:
                                print(f"goal connected iteration={iteration} accepted_nodes={len(nodes)}", flush=True)
                        if config.stop_on_first_goal:
                            break
            if report:
                self._print_progress(iteration, nodes, best_goal_distance, best_cost)

        edges = [(i, node.parent) for i, node in enumerate(nodes) if node.parent is not None]
        stats = dict(self.cells.stats)
        if not goal_indices:
            if config.progress_interval > 0:
                print(f"search failed iteration budget exhausted iteration={iteration}/{config.max_iterations} "
                      f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}", flush=True)
            return TubeRRTResult(False, tree_nodes=nodes, tree_edges=edges, failure_reason="iteration budget exhausted",
                                 trace=trace, iterations=iteration, overlap_stats=stats)
        best_goal = min(goal_indices, key=lambda i: nodes[i].cost)
        if nodes[best_goal].cost < cost_history[-1][1]:
            cost_history.append((iteration, nodes[best_goal].cost))
        route, indices = self._path(nodes, best_goal)
        radii = [float(v) for v in self.clearances([p.x for p in route], [p.y for p in route], [p.yaw for p in route])]
        if config.progress_interval > 0 and not config.stop_on_first_goal:
            print(f"search finished iteration={iteration} accepted_nodes={len(nodes)} goal_nodes={len(goal_indices)} "
                  f"best_cost={nodes[best_goal].cost:.3f}", flush=True)
        return TubeRRTResult(True, route, radii, nodes, edges, bottleneck=self.route_clearance(route), path_nodes=indices,
                             trace=trace, iterations=iteration, first_goal_iteration=first_goal_iteration,
                             path_cost=nodes[best_goal].cost, cost_history=cost_history, overlap_stats=stats)
