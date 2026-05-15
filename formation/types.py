from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Point2D = tuple[float, float]
GridIndex = tuple[int, int]


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class GridCoord:
    row: int
    col: int


@dataclass
class MapData:
    name: str
    resolution: float
    width_m: float
    height_m: float
    origin_xy: Point2D
    occupancy: np.ndarray
    inflated_occupancy: np.ndarray
    distance_field: np.ndarray
    start_xy: Point2D
    goal_xy: Point2D
    obstacle_primitives: list[dict[str, Any]] = field(default_factory=list)
    inflation_radius: float = 0.0

    @property
    def rows(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def cols(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape

    def world_to_grid(self, xy: Point2D) -> GridIndex:
        x, y = xy
        origin_x, origin_y = self.origin_xy
        col = int(math.floor((x - origin_x) / self.resolution))
        row = int(math.floor((y - origin_y) / self.resolution))
        if not self.in_bounds((row, col)):
            raise ValueError(f"Point {xy} lies outside map bounds {self.name}.")
        return row, col

    def grid_to_world(self, rc: GridIndex) -> Point2D:
        row, col = rc
        if not self.in_bounds(rc):
            raise ValueError(f"Grid index {rc} lies outside map bounds {self.name}.")
        origin_x, origin_y = self.origin_xy
        x = origin_x + (col + 0.5) * self.resolution
        y = origin_y + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, rc: GridIndex) -> bool:
        row, col = rc
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_occupied(self, rc: GridIndex, inflated: bool = False) -> bool:
        row, col = rc
        grid = self.inflated_occupancy if inflated else self.occupancy
        return bool(grid[row, col])


@dataclass
class GlobalPath:
    grid_path_rc: list[GridIndex]
    raw_waypoints_xy: list[Point2D]
    waypoints_xy: list[Point2D]
    start_xy: Point2D
    goal_xy: Point2D
    waypoint_grid_rc: list[GridIndex] = field(default_factory=list)


@dataclass
class FormationSpec:
    name: str
    slots: np.ndarray
    lateral_half_width: float
    longitudinal_half_length: float
    bounding_radius: float
    min_pairwise_distance: float
    task_utility: float = 0.0


@dataclass
class PreviewCurveConfig:
    sample_spacing_m: float = 0.10
    bezier_tension: float = 0.35
    reduced_tension: float = 0.18
    min_point_spacing_m: float = 0.05
    preview_distance_m: float = 3.5
    min_window_points: int = 4
    clearance_threshold_m: float | None = None
    use_subgoal_truncation: bool = True
    include_ref_point: bool = True
    densify_spacing_m: float = 0.20
    bezier_segment_steps: int = 20
    short_preview_scale: float = 0.70
    fallback_min_distance_m: float = 1.2
    fallback_to_polyline: bool = True


@dataclass
class LocalPathWindow:
    points_xy: list[Point2D]
    local_subgoal_xy: Point2D
    accumulated_distance_m: float
    source_start_index: int
    source_end_index: int


@dataclass
class LocalPreviewPath:
    points_xy: list[Point2D]
    arc_lengths: list[float]
    tangents_xy: list[Point2D]
    normals_xy: list[Point2D]
    curvatures: list[float]
    source_mode: str
    is_safe: bool
    min_clearance: float
    local_subgoal_xy: Point2D | None = None
    truncated_window_xy: list[Point2D] = field(default_factory=list)
    observation_distance_m: float = 0.0
    curve_end_distance_m: float = 0.0
    waypoint_window_xy: list[Point2D] = field(default_factory=list)
    clearance_samples: list[float] = field(default_factory=list)
    failure_reason: str = ""
    used_tension: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.points_xy)

    @property
    def length_m(self) -> float:
        if not self.arc_lengths:
            return 0.0
        return float(self.arc_lengths[-1])

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.points_xy, dtype=float),
            np.asarray(self.arc_lengths, dtype=float),
            np.asarray(self.tangents_xy, dtype=float),
            np.asarray(self.normals_xy, dtype=float),
            np.asarray(self.curvatures, dtype=float),
        )
