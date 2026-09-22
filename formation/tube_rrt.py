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


@dataclass
class TubeRRTNode:
    pose: Pose2D
    radius: float
    parent: int | None
    cost: float


@dataclass
class TubeRRTResult:
    success: bool
    path_poses: list[Pose2D] = field(default_factory=list)
    path_radii: list[float] = field(default_factory=list)
    tree_nodes: list[TubeRRTNode] = field(default_factory=list)
    tree_edges: list[tuple[int, int]] = field(default_factory=list)
    failure_reason: str = ""
    bottleneck: float = 0.0



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
            nodes[index].cost = nodes[parent].cost + self.metric(nodes[parent].pose, nodes[index].pose)
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

    def plan(self) -> TubeRRTResult:
        start_clearance = self.clearance(self.start)
        start_radius = max(0.0, start_clearance)
        nodes = [TubeRRTNode(self.start, start_radius, None, 0.0)]
        if start_clearance <= 0.0:
            return TubeRRTResult(False, tree_nodes=nodes, failure_reason="start is in collision", bottleneck=start_radius)
        best_bottleneck = start_radius
        best_goal_distance = math.hypot(self.start.x - self.goal_xy[0], self.start.y - self.goal_xy[1])
        progress_interval = self.config.progress_interval
        for iteration in range(1, self.config.max_iterations + 1):
            sampled = self._sample_pose()
            nearest_index = self._nearest(nodes, sampled)
            pose = self._steer(nodes[nearest_index].pose, sampled)
            radius = self.safety_radius(pose)
            if radius <= 0.0 or not self.tube_overlap(nodes[nearest_index], TubeRRTNode(pose, radius, None, 0.0)):
                if progress_interval > 0 and iteration % progress_interval == 0:
                    print(
                        f"progress iteration={iteration}/{self.config.max_iterations} "
                        f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}",
                        flush=True,
                    )
                continue
            neighbor_indices = [i for i, node in enumerate(nodes) if self.metric(node.pose, pose) <= self.config.neighbor_radius]
            parent = nearest_index
            parent_cost = nodes[parent].cost + self.metric(nodes[parent].pose, pose)
            for candidate in neighbor_indices:
                if self.tube_overlap(nodes[candidate], TubeRRTNode(pose, radius, None, 0.0)):
                    candidate_cost = nodes[candidate].cost + self.metric(nodes[candidate].pose, pose)
                    if candidate_cost < parent_cost:
                        parent, parent_cost = candidate, candidate_cost
            new_index = len(nodes)
            nodes.append(TubeRRTNode(pose, radius, parent, parent_cost))
            best_bottleneck = min(best_bottleneck, radius)
            best_goal_distance = min(best_goal_distance, math.hypot(pose.x - self.goal_xy[0], pose.y - self.goal_xy[1]))
            for candidate in neighbor_indices:
                if candidate == parent or candidate == 0:
                    continue
                new_cost = nodes[new_index].cost + self.metric(pose, nodes[candidate].pose)
                if new_cost < nodes[candidate].cost and self.tube_overlap(nodes[new_index], nodes[candidate]):
                    nodes[candidate].parent = new_index
                    nodes[candidate].cost = new_cost
                    self._update_descendant_costs(nodes, candidate)
            goal_distance = math.hypot(pose.x - self.goal_xy[0], pose.y - self.goal_xy[1])
            if goal_distance <= self.config.goal_connect_distance:
                goal_pose = Pose2D(self.goal_xy[0], self.goal_xy[1], pose.yaw)
                goal_radius = self.safety_radius(goal_pose)
                goal_node = TubeRRTNode(goal_pose, goal_radius, new_index, 0.0)
                if goal_radius > 0.0 and self.tube_overlap(nodes[new_index], goal_node):
                    goal_node.cost = nodes[new_index].cost + self.metric(pose, goal_pose)
                    nodes.append(goal_node)
                    path, radii = self._path(nodes, len(nodes) - 1)
                    edges = [(i, node.parent) for i, node in enumerate(nodes) if node.parent is not None]
                    if progress_interval > 0:
                        if iteration % progress_interval == 0:
                            print(
                                f"progress iteration={iteration}/{self.config.max_iterations} "
                                f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}",
                                flush=True,
                            )
                        print(
                            f"goal connected iteration={iteration} accepted_nodes={len(nodes)}",
                            flush=True,
                        )
                    return TubeRRTResult(True, path, radii, nodes, edges, bottleneck=min(radii))
            if progress_interval > 0 and iteration % progress_interval == 0:
                print(
                    f"progress iteration={iteration}/{self.config.max_iterations} "
                    f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}",
                    flush=True,
                )
        edges = [(i, node.parent) for i, node in enumerate(nodes) if node.parent is not None]
        if progress_interval > 0:
            print(
                f"search failed iteration budget exhausted iteration={self.config.max_iterations}/{self.config.max_iterations} "
                f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}",
                flush=True,
            )
        return TubeRRTResult(False, tree_nodes=nodes, tree_edges=edges,
                             failure_reason="iteration budget exhausted", bottleneck=best_bottleneck)


def project_robot_paths(path_poses: Iterable[Pose2D], slots: np.ndarray) -> list[list[tuple[float, float]]]:
    projected = [[] for _ in range(len(np.asarray(slots)))]
    for pose in path_poses:
        points = transform_slots(pose, slots)
        for index, point in enumerate(points):
            projected[index].append((float(point[0]), float(point[1])))
    return projected
