from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from formation.types import FormationSpec, MapData, Pose2D, wrap_to_pi


@dataclass
class TubeRRTConfig:
    max_iterations: int = 2500
    metric_step: float = 0.45
    neighbor_radius: float = 1.2
    goal_bias: float = 0.12
    goal_connect_distance: float = 0.8
    seed: int = 7
    safety_margin: float = 0.0
    progress_interval: int = 0
    record_trace: bool = False
    stop_on_first_goal: bool = True
    step_backoff: bool = True
    min_metric_step: float = 0.02
    margin_weight: float = 0.0


@dataclass
class TubeRRTNode:
    pose: Pose2D
    radius: float
    parent: int | None
    cost: float


@dataclass
class TubeRRTTraceEvent:
    """One search iteration; status is "collision", "no_overlap", "added" or "goal"."""

    iteration: int
    sample: Pose2D
    nearest: int
    steered: Pose2D
    radius: float
    status: str
    node_index: int | None = None
    parent: int | None = None
    rewires: list[tuple[int, int, int]] = field(default_factory=list)
    attempts: int = 1


@dataclass
class TubeRRTResult:
    success: bool
    path_poses: list[Pose2D] = field(default_factory=list)
    path_radii: list[float] = field(default_factory=list)
    tree_nodes: list[TubeRRTNode] = field(default_factory=list)
    tree_edges: list[tuple[int, int]] = field(default_factory=list)
    failure_reason: str = ""
    bottleneck: float = 0.0
    trace: list[TubeRRTTraceEvent] = field(default_factory=list)
    iterations: int = 0
    first_goal_iteration: int | None = None
    path_cost: float = 0.0
    cost_history: list[tuple[int, float]] = field(default_factory=list)



def transform_slots(pose: Pose2D, slots: np.ndarray) -> np.ndarray:
    """Return world slot centers for a rigid formation pose."""
    slots = np.asarray(slots, dtype=float)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rotation = np.asarray(((c, -s), (s, c)), dtype=float)
    return slots @ rotation.T + np.asarray((pose.x, pose.y))


def interpolate_pose(first: Pose2D, second: Pose2D, alpha: float) -> Pose2D:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return Pose2D(
        first.x + alpha * (second.x - first.x),
        first.y + alpha * (second.y - first.y),
        wrap_to_pi(first.yaw + alpha * wrap_to_pi(second.yaw - first.yaw)),
    )


class TubeRRTPlanner:
    """A compact joint-state RRT with safety-ball (tube) edge constraints."""

    def __init__(
        self,
        map_data: MapData,
        formation: FormationSpec | np.ndarray,
        start: Pose2D,
        goal_xy: tuple[float, float] | None = None,
        config: TubeRRTConfig | None = None,
    ) -> None:
        self.map_data = map_data
        self.slots = np.asarray(formation.slots if isinstance(formation, FormationSpec) else formation, dtype=float)
        if self.slots.ndim != 2 or self.slots.shape[1] != 2:
            raise ValueError("formation slots must have shape (N, 2)")
        self.start = start
        self.goal_xy = map_data.goal_xy if goal_xy is None else tuple(goal_xy)
        self.config = config or TubeRRTConfig()
        self.robot_radius = float(map_data.robot_radius)
        self.safety_margin = self.config.safety_margin + float(map_data.safety_margin)
        self.formation_radius = float(np.max(np.linalg.norm(self.slots, axis=1)))
        self.rng = np.random.default_rng(self.config.seed)
        unsupported = [p.get("type") for p in map_data.obstacle_primitives if p.get("type") != "circle"]
        if unsupported:
            raise ValueError("TubeRRTPlanner supports only circle obstacle primitives")

    def metric(self, first: Pose2D, second: Pose2D) -> float:
        dc = math.hypot(second.x - first.x, second.y - first.y)
        return dc + self.formation_radius * abs(wrap_to_pi(second.yaw - first.yaw))

    def clearance(self, pose: Pose2D) -> float:
        points = transform_slots(pose, self.slots)
        minimum = math.inf
        ox, oy = self.map_data.origin_xy
        for x, y in points:
            minimum = min(minimum, x - ox, ox + self.map_data.width_m - x,
                          y - oy, oy + self.map_data.height_m - y)
            for primitive in self.map_data.obstacle_primitives:
                cx, cy = primitive["center_xy"]
                minimum = min(minimum, math.hypot(x - float(cx), y - float(cy)) - float(primitive["radius"]))
        return float(minimum - self.robot_radius - self.safety_margin)

    def safety_radius(self, pose: Pose2D) -> float:
        return max(0.0, self.clearance(pose))

    def tube_overlap(self, first: TubeRRTNode | Pose2D, second: TubeRRTNode | Pose2D) -> bool:
        if isinstance(first, TubeRRTNode):
            first_pose, first_radius = first.pose, first.radius
        else:
            first_pose, first_radius = first, self.safety_radius(first)
        if isinstance(second, TubeRRTNode):
            second_pose, second_radius = second.pose, second.radius
        else:
            second_pose, second_radius = second, self.safety_radius(second)
        return self.metric(first_pose, second_pose) < first_radius + second_radius

    def _sample_pose(self) -> Pose2D:
        if self.rng.random() < self.config.goal_bias:
            return Pose2D(self.goal_xy[0], self.goal_xy[1], float(self.rng.uniform(-math.pi, math.pi)))
        ox, oy = self.map_data.origin_xy
        return Pose2D(float(self.rng.uniform(ox, ox + self.map_data.width_m)),
                      float(self.rng.uniform(oy, oy + self.map_data.height_m)),
                      float(self.rng.uniform(-math.pi, math.pi)))

    def _steer(self, source: Pose2D, target: Pose2D) -> Pose2D:
        distance = self.metric(source, target)
        if distance <= self.config.metric_step:
            return target
        return interpolate_pose(source, target, self.config.metric_step / distance)

    def _extend(self, near: TubeRRTNode, target: Pose2D) -> tuple[Pose2D, float, int, bool]:
        """Steer toward target, shrinking the step until the new ball overlaps the nearest ball.

        Clearance is 1-Lipschitz in d_G, so any step shorter than near.radius yields a valid, overlapping node;
        the backoff therefore ends at 0.9 * near.radius unless that is below min_metric_step.
        """
        distance = self.metric(near.pose, target)
        step = min(self.config.metric_step, distance)
        attempts = 0
        while True:
            attempts += 1
            pose = target if step >= distance else interpolate_pose(near.pose, target, step / distance)
            radius = self.safety_radius(pose)
            if radius > 0.0 and self.tube_overlap(near, TubeRRTNode(pose, radius, None, 0.0)):
                return pose, radius, attempts, True
            next_step = max(0.5 * step, 0.9 * near.radius)
            if not self.config.step_backoff or next_step >= step or next_step < self.config.min_metric_step:
                return pose, radius, attempts, False
            step = next_step

    def edge_cost(self, first: TubeRRTNode, second: TubeRRTNode) -> float:
        """d_G length, inflated by margin_weight / min(rho) so that low-clearance edges cost more (J_margin)."""
        length = self.metric(first.pose, second.pose)
        if self.config.margin_weight <= 0.0:
            return length
        return length * (1.0 + self.config.margin_weight / max(min(first.radius, second.radius), 1e-9))

    def _nearest(self, nodes: list[TubeRRTNode], pose: Pose2D) -> int:
        return min(range(len(nodes)), key=lambda i: self.metric(nodes[i].pose, pose))

    def _update_descendant_costs(self, nodes: list[TubeRRTNode], root_index: int) -> None:
        children: dict[int, list[int]] = {}
        for index, node in enumerate(nodes):
            if node.parent is not None:
                children.setdefault(node.parent, []).append(index)
        stack = list(children.get(root_index, []))
        while stack:
            index = stack.pop()
            parent = nodes[index].parent
            assert parent is not None
            nodes[index].cost = nodes[parent].cost + self.edge_cost(nodes[parent], nodes[index])
            stack.extend(children.get(index, []))

    def _path(self, nodes: list[TubeRRTNode], index: int) -> tuple[list[Pose2D], list[float]]:
        indices: list[int] = []
        while index is not None:
            indices.append(index)
            parent = nodes[index].parent
            if parent is None:
                break
            index = parent
        indices.reverse()
        return [nodes[i].pose for i in indices], [nodes[i].radius for i in indices]

    def _print_progress(self, iteration: int, nodes: list[TubeRRTNode], best_goal_distance: float,
                        best_cost: float | None) -> None:
        cost = "" if best_cost is None else f" best_cost={best_cost:.3f}"
        print(
            f"progress iteration={iteration}/{self.config.max_iterations} "
            f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}{cost}",
            flush=True,
        )

    def plan(self) -> TubeRRTResult:
        start_clearance = self.clearance(self.start)
        start_radius = max(0.0, start_clearance)
        nodes = [TubeRRTNode(self.start, start_radius, None, 0.0)]
        if start_clearance <= 0.0:
            return TubeRRTResult(False, tree_nodes=nodes, failure_reason="start is in collision", bottleneck=start_radius)
        best_bottleneck = start_radius
        best_goal_distance = math.hypot(self.start.x - self.goal_xy[0], self.start.y - self.goal_xy[1])
        progress_interval = self.config.progress_interval
        trace: list[TubeRRTTraceEvent] = []
        goal_indices: list[int] = []
        first_goal_iteration: int | None = None
        cost_history: list[tuple[int, float]] = []
        iteration = 0
        for iteration in range(1, self.config.max_iterations + 1):
            if goal_indices:
                best_cost = min(nodes[i].cost for i in goal_indices)
                if best_cost < cost_history[-1][1]:
                    cost_history.append((iteration - 1, best_cost))
            else:
                best_cost = None
            report = progress_interval > 0 and iteration % progress_interval == 0
            sampled = self._sample_pose()
            nearest_index = self._nearest(nodes, sampled)
            pose, radius, attempts, ok = self._extend(nodes[nearest_index], sampled)
            if not ok:
                if self.config.record_trace:
                    status = "collision" if radius <= 0.0 else "no_overlap"
                    trace.append(TubeRRTTraceEvent(iteration, sampled, nearest_index, pose, radius, status,
                                                   attempts=attempts))
                if report:
                    self._print_progress(iteration, nodes, best_goal_distance, best_cost)
                continue
            new_node = TubeRRTNode(pose, radius, None, 0.0)
            neighbor_indices = [i for i, node in enumerate(nodes) if self.metric(node.pose, pose) <= self.config.neighbor_radius]
            parent = nearest_index
            parent_cost = nodes[parent].cost + self.edge_cost(nodes[parent], new_node)
            for candidate in neighbor_indices:
                if self.tube_overlap(nodes[candidate], new_node):
                    candidate_cost = nodes[candidate].cost + self.edge_cost(nodes[candidate], new_node)
                    if candidate_cost < parent_cost:
                        parent, parent_cost = candidate, candidate_cost
            new_index = len(nodes)
            new_node.parent, new_node.cost = parent, parent_cost
            nodes.append(new_node)
            best_bottleneck = min(best_bottleneck, radius)
            best_goal_distance = min(best_goal_distance, math.hypot(pose.x - self.goal_xy[0], pose.y - self.goal_xy[1]))
            event = TubeRRTTraceEvent(iteration, sampled, nearest_index, pose, radius, "added", new_index, parent,
                                      attempts=attempts)
            if self.config.record_trace:
                trace.append(event)
            for candidate in neighbor_indices:
                if candidate == parent or candidate == 0:
                    continue
                new_cost = nodes[new_index].cost + self.edge_cost(new_node, nodes[candidate])
                if new_cost < nodes[candidate].cost and self.tube_overlap(nodes[new_index], nodes[candidate]):
                    event.rewires.append((candidate, nodes[candidate].parent, new_index))
                    nodes[candidate].parent = new_index
                    nodes[candidate].cost = new_cost
                    self._update_descendant_costs(nodes, candidate)
            goal_distance = math.hypot(pose.x - self.goal_xy[0], pose.y - self.goal_xy[1])
            improves = best_cost is None or nodes[new_index].cost + goal_distance < best_cost
            if goal_distance <= self.config.goal_connect_distance and improves:
                goal_pose = Pose2D(self.goal_xy[0], self.goal_xy[1], pose.yaw)
                goal_radius = self.safety_radius(goal_pose)
                goal_node = TubeRRTNode(goal_pose, goal_radius, new_index, 0.0)
                connected = goal_radius > 0.0 and self.tube_overlap(new_node, goal_node)
                if connected:
                    goal_node.cost = new_node.cost + self.edge_cost(new_node, goal_node)
                if connected and (best_cost is None or goal_node.cost < best_cost):
                    nodes.append(goal_node)
                    goal_indices.append(len(nodes) - 1)
                    if self.config.record_trace:
                        trace.append(TubeRRTTraceEvent(iteration, goal_pose, new_index, goal_pose, goal_radius,
                                                       "goal", len(nodes) - 1, new_index))
                    best_cost = goal_node.cost
                    cost_history.append((iteration, best_cost))
                    if first_goal_iteration is None:
                        first_goal_iteration = iteration
                        if progress_interval > 0:
                            if report:
                                self._print_progress(iteration, nodes, best_goal_distance, None)
                            print(f"goal connected iteration={iteration} accepted_nodes={len(nodes)}", flush=True)
                            report = False
                    if self.config.stop_on_first_goal:
                        break
            if report:
                self._print_progress(iteration, nodes, best_goal_distance, best_cost)
        edges = [(i, node.parent) for i, node in enumerate(nodes) if node.parent is not None]
        if goal_indices:
            best_goal = min(goal_indices, key=lambda i: nodes[i].cost)
            if nodes[best_goal].cost < cost_history[-1][1]:
                cost_history.append((iteration, nodes[best_goal].cost))
            path, radii = self._path(nodes, best_goal)
            if progress_interval > 0 and not self.config.stop_on_first_goal:
                print(f"search finished iteration={iteration} accepted_nodes={len(nodes)} goal_nodes={len(goal_indices)} "
                      f"best_cost={nodes[best_goal].cost:.3f}", flush=True)
            return TubeRRTResult(True, path, radii, nodes, edges, bottleneck=min(radii), trace=trace,
                                 iterations=iteration, first_goal_iteration=first_goal_iteration,
                                 path_cost=nodes[best_goal].cost, cost_history=cost_history)
        if progress_interval > 0:
            print(
                f"search failed iteration budget exhausted iteration={self.config.max_iterations}/{self.config.max_iterations} "
                f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}",
                flush=True,
            )
        return TubeRRTResult(False, tree_nodes=nodes, tree_edges=edges,
                             failure_reason="iteration budget exhausted", bottleneck=best_bottleneck, trace=trace,
                             iterations=iteration)


def project_robot_paths(path_poses: Iterable[Pose2D], slots: np.ndarray) -> list[list[tuple[float, float]]]:
    projected = [[] for _ in range(len(np.asarray(slots)))]
    for pose in path_poses:
        points = transform_slots(pose, slots)
        for index, point in enumerate(points):
            projected[index].append((float(point[0]), float(point[1])))
    return projected
