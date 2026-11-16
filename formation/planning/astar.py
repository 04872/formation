from __future__ import annotations

import heapq
import math

import numpy as np

from formation.types import GridIndex


class AStarPlanner:
    def __init__(self, connectivity: int = 8) -> None:
        if connectivity != 8:
            raise ValueError("Only 8-connected A* is supported in the first version.")
        self.connectivity = connectivity
        self._neighbor_offsets = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

    def plan(
        self,
        occupancy: np.ndarray,
        start_rc: GridIndex,
        goal_rc: GridIndex,
    ) -> list[GridIndex]:
        rows, cols = occupancy.shape
        self._validate_node(start_rc, occupancy, rows, cols, "start")
        self._validate_node(goal_rc, occupancy, rows, cols, "goal")

        open_heap: list[tuple[float, float, GridIndex]] = []
        heapq.heappush(open_heap, (self._heuristic(start_rc, goal_rc), 0.0, start_rc))
        came_from: dict[GridIndex, GridIndex] = {}
        g_score: dict[GridIndex, float] = {start_rc: 0.0}
        closed_set: set[GridIndex] = set()

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            if current == goal_rc:
                return self._reconstruct_path(came_from, current)
            closed_set.add(current)

            for neighbor, step_cost in self._neighbors(current, rows, cols):
                if occupancy[neighbor]:
                    continue
                tentative_cost = current_cost + step_cost
                if tentative_cost + 1e-12 >= g_score.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                priority = tentative_cost + self._heuristic(neighbor, goal_rc)
                heapq.heappush(open_heap, (priority, tentative_cost, neighbor))

        raise ValueError("A* could not find a path between start and goal.")

    def _validate_node(
        self,
        node: GridIndex,
        occupancy: np.ndarray,
        rows: int,
        cols: int,
        name: str,
    ) -> None:
        row, col = node
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"{name} node {node} lies outside occupancy grid bounds.")
        if occupancy[row, col]:
            raise ValueError(f"{name} node {node} lies inside occupied space.")

    def _neighbors(self, node: GridIndex, rows: int, cols: int) -> list[tuple[GridIndex, float]]:
        row, col = node
        neighbors: list[tuple[GridIndex, float]] = []
        for d_row, d_col, cost in self._neighbor_offsets:
            next_row = row + d_row
            next_col = col + d_col
            if 0 <= next_row < rows and 0 <= next_col < cols:
                neighbors.append(((next_row, next_col), cost))
        return neighbors

    def _heuristic(self, a: GridIndex, b: GridIndex) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _reconstruct_path(
        self,
        came_from: dict[GridIndex, GridIndex],
        current: GridIndex,
    ) -> list[GridIndex]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
