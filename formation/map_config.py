from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Point2D = tuple[float, float]
MapType = Literal[
    "right_angle_corridor",
    "s_curve_corridor",
    "obstacle_cluster",
    "narrow_entrance",
    "narrowing_corridor",
]


@dataclass
class BaseMapConfig:
    resolution: float = 0.05
    robot_radius: float = 0.09
    safety_margin: float = 0.06
    global_planner_margin: float = 0.04
    width_m: float = 12.0
    height_m: float = 8.0
    start_xy: Point2D = (-5.0, 0.0)
    goal_xy: Point2D = (5.0, 0.0)


@dataclass
class RightAngleCorridorConfig(BaseMapConfig):
    width_m: float = 10.0
    height_m: float = 10.0
    start_xy: Point2D = (-4.2, -3.5)
    goal_xy: Point2D = (3.6, 3.8)
    corridor_width: float = 1.6
    horizontal_center_y: float = -3.5
    vertical_center_x: float = 3.6
    corner_x: float = 3.6
    corner_y: float = -3.5


@dataclass
class NarrowingCorridorConfig(BaseMapConfig):
    start_xy: Point2D = (-4.6, -0.08)
    goal_xy: Point2D = (4.6, 0.48)
    corridor_x_min: float = -4.8
    corridor_x_max: float = 5.4
    wide_width: float = 2.8
    narrow_width: float = 1.45
    centerline_slope: float = 0.06
    centerline_intercept: float = 0.2
    transition_start_x: float = -1.8
    transition_end_x: float = 3.4
    throat_padding_x: float = 0.8
    tail_width: float = 1.55
    tail_start_x: float = 3.4
    tail_end_x: float = 5.4


@dataclass
class SCurveCorridorConfig(BaseMapConfig):
    amplitude: float = 1.3
    corridor_width: float = 1.4
    period_m: float = 10.0


@dataclass
class ObstacleClusterConfig(BaseMapConfig):
    rectangular_obstacles: list[tuple[Point2D, float, float]] = field(
        default_factory=lambda: [
            ((-2.0, 1.5), 0.50, 0.50),
            ((-0.3, -1.6), 0.50, 0.50),
            ((1.5, 1.2), 0.50, 0.50),
            ((2.5, -0.7), 0.50, 0.50),
        ]
    )
    circular_obstacles: list[tuple[Point2D, float]] = field(
        default_factory=lambda: [
            ((-0.5, 0.3), 0.15),
            ((0.9, -0.2), 0.15),
        ]
    )


@dataclass
class NarrowEntranceConfig(BaseMapConfig):
    width_m: float = 16.0
    height_m: float = 12.0
    start_xy: Point2D = (-6.0, 3.0)
    goal_xy: Point2D = (6.0, -3.0)
    passage_width: float = 1.2
    passage_center_y: float = 0.0
    passage_start_x: float = -1.5
    passage_end_x: float = 1.5


@dataclass
class PlannerConfig:
    connectivity: int = 8
    heuristic: Literal["euclidean"] = "euclidean"
    simplify_remove_collinear: bool = True
    simplify_line_of_sight: bool = True


@dataclass
class ScenarioConfig:
    map_type: MapType
    map_config: BaseMapConfig
    planner_config: PlannerConfig = field(default_factory=PlannerConfig)
