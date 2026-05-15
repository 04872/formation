from __future__ import annotations

from formation.astar import AStarPlanner
from formation.map_config import PlannerConfig
from formation.path_simplifier import grid_path_to_world, simplify_grid_path
from formation.types import GlobalPath, MapData


class GlobalPlanner:
    def __init__(self, planner_config: PlannerConfig | None = None) -> None:
        self.config = planner_config or PlannerConfig()
        self.astar = AStarPlanner(connectivity=self.config.connectivity)

    def plan(self, map_data: MapData) -> GlobalPath:
        start_rc = map_data.world_to_grid(map_data.start_xy)
        goal_rc = map_data.world_to_grid(map_data.goal_xy)
        grid_path = self.astar.plan(map_data.inflated_occupancy, start_rc, goal_rc)
        simplified_grid = simplify_grid_path(grid_path, map_data.inflated_occupancy)
        raw_waypoints_xy = grid_path_to_world(
            grid_path,
            map_data.origin_xy,
            map_data.resolution,
        )
        waypoints_xy = grid_path_to_world(
            simplified_grid,
            map_data.origin_xy,
            map_data.resolution,
        )
        return GlobalPath(
            grid_path_rc=grid_path,
            raw_waypoints_xy=raw_waypoints_xy,
            waypoints_xy=waypoints_xy,
            start_xy=map_data.start_xy,
            goal_xy=map_data.goal_xy,
            waypoint_grid_rc=simplified_grid,
        )
