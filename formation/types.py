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
    robot_radius: float = 0.09
    safety_margin: float = 0.06

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
    slots: np.ndarray            # original slots (centroid frame)
    lateral_half_width: float
    longitudinal_half_length: float
    bounding_radius: float
    min_pairwise_distance: float
    task_utility: float = 0.0


@dataclass
class CurveBandSample:
    arc_length_s: float
    center_xy: Point2D
    tangent_xy: Point2D
    normal_xy: Point2D
    left_xy: Point2D
    right_xy: Point2D
    half_width_m: float
    direction_offset_rad: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurveBandStripCell:
    start_index: int
    end_index: int
    left_start_xy: Point2D
    right_start_xy: Point2D
    right_end_xy: Point2D
    left_end_xy: Point2D
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vertices_xy(self) -> tuple[Point2D, Point2D, Point2D, Point2D]:
        return (
            self.left_start_xy,
            self.right_start_xy,
            self.right_end_xy,
            self.left_end_xy,
        )


@dataclass
class CurveBand:
    samples: list[CurveBandSample]
    strip_cells: list[CurveBandStripCell]
    source_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.samples)


@dataclass
class AssignmentResult:
    assignment: tuple[int, ...]
    total_cost: float
    max_cost: float
    per_robot_costs: list[float] = field(default_factory=list)


@dataclass
class FormationScoreBreakdown:
    min_corridor_margin_m: float = 0.0
    embedding_cost: float = 0.0
    corridor_violation_cost: float = 0.0
    switch_cost: float = 0.0
    task_utility: float = 0.0
    total_score: float = 0.0
    offset_cost: float = 0.0
    heading_cost: float = 0.0
    safety_margin_m: float = 0.0
    mean_clearance_m: float = 0.0
    min_slot_clearance_m: float = 0.0
    preview_alignment_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingQPResult:
    lateral_offsets_m: list[float] = field(default_factory=list)
    heading_offsets_rad: list[float] = field(default_factory=list)
    center_points_xy: list[Point2D] = field(default_factory=list)
    heading_rads: list[float] = field(default_factory=list)
    slot_points_by_step_xy: list[list[Point2D]] = field(default_factory=list)
    offset_cost: float = 0.0
    heading_cost: float = 0.0
    min_corridor_margin_m: float = 0.0
    corridor_violation_cost: float = 0.0
    is_feasible: bool = False
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FormationCandidateEvaluation:
    formation_name: str
    band_feasible: bool
    is_safe: bool
    score_breakdown: FormationScoreBreakdown
    center_points_xy: list[Point2D] = field(default_factory=list)
    heading_rads: list[float] = field(default_factory=list)
    slot_points_by_step_xy: list[list[Point2D]] = field(default_factory=list)
    assignment: AssignmentResult | None = None
    embedding_qp_result: EmbeddingQPResult | None = None
    lateral_offset_m: float = 0.0
    heading_offset_rad: float = 0.0
    min_slot_clearance_m: float = 0.0
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuideSample:
    center_xy: Point2D
    heading_rad: float
    robot_points_xy: list[Point2D]


@dataclass
class FormationGuide:
    formation_name: str
    guide_samples: list[GuideSample]
    assignment: AssignmentResult | None = None
    transition_alphas: list[float] = field(default_factory=list)
    switched: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    v: float = 0.0
    omega: float = 0.0

    @property
    def position_xy(self) -> Point2D:
        return (self.x, self.y)


@dataclass(frozen=True)
class ControlCommand:
    v: float
    omega: float


@dataclass
class MPCWeights:
    position: float = 25.0
    heading: float = 4.0
    input: float = 0.8
    input_smooth: float = 1.0
    initial_input_smooth: float = 1.0
    obstacle_slack: float = 800.0
    neighbor_slack: float = 1200.0
    relative_position: float = 2.0
    velocity_consensus: float = 0.5
    progress_sync: float = 0.5
    progress_rate_sync: float = 0.0
    terminal_position: float = 60.0


@dataclass
class MPCConfig:
    dt: float = 0.20
    horizon_steps: int = 10
    v_max: float = 0.80
    omega_max: float = 1.20
    robot_radius: float = 0.09
    safety_margin: float = 0.06
    inter_robot_margin: float = 0.10
    neighbor_prediction_mode: str = "previous_prediction"
    parallel_solve: bool = True
    parallel_workers: int | None = None
    weights: MPCWeights = field(default_factory=MPCWeights)


@dataclass(frozen=True)
class RobotReferenceSample:
    position_xy: Point2D
    yaw: float
    v_ref: float
    omega_ref: float
    alpha: float
    t: float


@dataclass
class RobotReferenceTrajectory:
    robot_index: int
    samples: list[RobotReferenceSample]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def window(self, start_index: int, horizon_steps: int) -> "RobotReferenceTrajectory":
        if not self.samples:
            return RobotReferenceTrajectory(robot_index=self.robot_index, samples=[])

        window_size = max(horizon_steps, 0) + 1
        clamped_start = min(max(start_index, 0), len(self.samples) - 1)
        window_samples = list(self.samples[clamped_start : clamped_start + window_size])
        while len(window_samples) < window_size:
            window_samples.append(window_samples[-1])
        return RobotReferenceTrajectory(robot_index=self.robot_index, samples=window_samples)


@dataclass
class FormationControllerReference:
    formation_name: str
    center_points_xy: list[Point2D]
    center_heading_rads: list[float]
    transition_alphas: list[float]
    robot_trajectories: list[RobotReferenceTrajectory]
    dt: float
    horizon_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.center_points_xy)

    @property
    def robot_count(self) -> int:
        return len(self.robot_trajectories)

    def window(self, start_index: int, horizon_steps: int | None = None) -> "FormationControllerReference":
        if self.sample_count <= 0:
            return FormationControllerReference(
                formation_name=self.formation_name,
                center_points_xy=[],
                center_heading_rads=[],
                transition_alphas=[],
                robot_trajectories=[
                    RobotReferenceTrajectory(robot_index=trajectory.robot_index, samples=[])
                    for trajectory in self.robot_trajectories
                ],
                dt=self.dt,
                horizon_steps=self.horizon_steps if horizon_steps is None else horizon_steps,
                metadata=dict(self.metadata),
            )

        window_horizon = self.horizon_steps if horizon_steps is None else horizon_steps
        window_size = max(window_horizon, 0) + 1
        clamped_start = min(max(start_index, 0), self.sample_count - 1)

        center_points_xy = list(self.center_points_xy[clamped_start : clamped_start + window_size])
        center_heading_rads = list(self.center_heading_rads[clamped_start : clamped_start + window_size])
        transition_alphas = list(self.transition_alphas[clamped_start : clamped_start + window_size])

        while len(center_points_xy) < window_size:
            center_points_xy.append(center_points_xy[-1])
            center_heading_rads.append(center_heading_rads[-1])
            transition_alphas.append(transition_alphas[-1])

        return FormationControllerReference(
            formation_name=self.formation_name,
            center_points_xy=center_points_xy,
            center_heading_rads=center_heading_rads,
            transition_alphas=transition_alphas,
            robot_trajectories=[
                trajectory.window(clamped_start, window_horizon) for trajectory in self.robot_trajectories
            ],
            dt=self.dt,
            horizon_steps=window_horizon,
            metadata=dict(self.metadata),
        )


@dataclass
class RobotPrediction:
    robot_index: int
    positions_xy: list[Point2D]
    yaw_rads: list[float]
    commands: list[ControlCommand] = field(default_factory=list)
    obstacle_slacks: list[float] = field(default_factory=list)
    neighbor_slacks: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.positions_xy)


@dataclass
class SimulationTrace:
    state_history: list[list[RobotState]]
    command_history: list[list[ControlCommand]]
    prediction_history: list[list[RobotPrediction]]
    tracking_errors: list[float]
    min_obstacle_clearances: list[float]
    min_pairwise_distances: list[float]
    reference_history: list[FormationControllerReference] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_states(self) -> list[RobotState]:
        return self.state_history[-1] if self.state_history else []

    @property
    def step_count(self) -> int:
        return len(self.command_history)

    @property
    def max_tracking_error(self) -> float:
        return max(self.tracking_errors, default=0.0)

    @property
    def min_obstacle_clearance(self) -> float:
        return min(self.min_obstacle_clearances, default=math.inf)

    @property
    def min_pairwise_distance(self) -> float:
        return min(self.min_pairwise_distances, default=math.inf)


@dataclass
class PreviewCurveConfig:
    sample_spacing_m: float = 0.05
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
    curve_refine_iterations: int = 5
    curve_refine_lambda_a: float = 1.0
    curve_refine_lambda_s: float = 12.0
    curve_refine_lambda_d: float = 0.75
    curve_refine_fixed_prefix_points: int = 1
    curve_refine_guide_weight: float = 0.08
    curve_refine_goal_weight: float = 0.7
    curve_refine_convergence_tol_m: float = 0.015
    curve_pref_clearance_m: float = 0.45


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
