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
    yaw_slices: int = 16
    cell_eta: float = 0.98
    yaw_slice_count: int | None = None
    cell_shrink: float | None = None

    def __post_init__(self) -> None:
        if self.yaw_slice_count is not None:
            self.yaw_slices = int(self.yaw_slice_count)
        if self.cell_shrink is not None:
            self.cell_eta = float(self.cell_shrink)


@dataclass
class OrientationSafeCell:
    """Conservative orientation-sliced free cell around a formation center.

    Guarded clearances already include robot radius and safety margin.  The
    analytic lower envelope is the certificate; finite yaw samples are only
    used to search for a common-yaw edge witness.
    """

    center: np.ndarray | tuple[float, float]
    anchor_yaw: float
    yaw_slices: np.ndarray
    guarded_clearances: np.ndarray
    formation_radius: float
    eta: float = 1.0
    component_lo: float | None = None
    component_hi: float | None = None
    full_circle: bool = False
    shrink: float | None = None
    _anchor_radius_value: float = field(init=False, repr=False)
    _radius_upper_bound: float = field(init=False, repr=False)
    _world_intervals_cache: tuple[tuple[float, float], ...] = field(init=False, repr=False)
    _yaw_slice_values: tuple[float, ...] = field(init=False, repr=False)
    _clearance_values: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(2)
        self.anchor_yaw = float(wrap_to_pi(self.anchor_yaw))
        self.yaw_slices = np.asarray(self.yaw_slices, dtype=float).reshape(-1)
        self.guarded_clearances = np.asarray(self.guarded_clearances, dtype=float).reshape(-1)
        if len(self.yaw_slices) != len(self.guarded_clearances) or len(self.yaw_slices) == 0:
            raise ValueError("yaw_slices and guarded_clearances must have the same non-zero length")
        # Keep the lift: wrapped differences are used in the formula, while
        # the equally spaced slice representatives retain their cyclic order.
        self.yaw_slices = np.asarray(self.yaw_slices, dtype=float)
        self._yaw_slice_values = tuple(float(value) for value in self.yaw_slices)
        self._clearance_values = tuple(float(value) for value in self.guarded_clearances)
        self.formation_radius = max(0.0, float(self.formation_radius))
        if self.shrink is not None:
            self.eta = float(self.shrink)
        self.eta = float(self.eta)
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("cell eta must lie in (0, 1]")
        if self.component_lo is None or self.component_hi is None:
            self.component_lo, self.component_hi, self.full_circle = self._anchor_component()
        else:
            self.component_lo = float(self.component_lo)
            self.component_hi = float(self.component_hi)
            self.full_circle = bool(self.full_circle)
        self._anchor_radius_value = self._radius_scalar(self.anchor_yaw)
        self._radius_upper_bound = self.eta * max(float(np.max(self.guarded_clearances)), 0.0)
        if self.full_circle:
            self._world_intervals_cache = (
                (self.anchor_yaw - math.pi + 1e-12, self.anchor_yaw + math.pi - 1e-12),
            )
        else:
            self._world_intervals_cache = tuple(
                (self.anchor_yaw + self.component_lo + 2.0 * math.pi * period,
                 self.anchor_yaw + self.component_hi + 2.0 * math.pi * period)
                for period in range(-2, 3)
            )

    def _anchor_component(self) -> tuple[float, float, bool]:
        """Compute the positive yaw component containing the anchor analytically."""
        d = self._clearance_values
        if self.formation_radius <= 1e-15:
            return (-math.pi, math.pi, True) if max(d) > 0.0 else (0.0, 0.0, False)
        intervals: list[tuple[float, float]] = []
        two_r = 2.0 * self.formation_radius
        for theta, clearance in zip(self._yaw_slice_values, d):
            if clearance <= 0.0:
                continue
            ratio = clearance / two_r
            if ratio > 1.0:
                return -math.pi, math.pi, True
            alpha = math.pi if ratio == 1.0 else 2.0 * math.asin(max(0.0, ratio))
            relative = wrap_to_pi(theta - self.anchor_yaw)
            # relative is in [-pi, pi].  Only this representative and its
            # two adjacent lifts can intersect the component containing zero;
            # farther periodic copies cannot change that component.
            for period in (-1, 0, 1):
                center = relative + 2.0 * math.pi * period
                intervals.append((center - alpha, center + alpha))
        if not intervals:
            return 0.0, 0.0, False
        intervals.sort()
        merged: list[list[float]] = []
        for lo, hi in intervals:
            # Equality is a zero-radius tangent and must not merge components.
            if not merged or lo >= merged[-1][1]:
                merged.append([lo, hi])
            else:
                merged[-1][1] = max(merged[-1][1], hi)
        component = next((part for part in merged if part[0] < 0.0 < part[1]), None)
        if component is None:
            return 0.0, 0.0, False
        lo, hi = component
        if hi - lo > 2.0 * math.pi - 1e-10:
            return -math.pi, math.pi, True
        return float(lo), float(hi), False

    @property
    def anchor_component(self) -> tuple[float, float]:
        return float(self.component_lo), float(self.component_hi)

    @property
    def component(self) -> tuple[float, float]:
        return self.anchor_component

    @property
    def is_full_circle(self) -> bool:
        return self.full_circle

    @property
    def R_F(self) -> float:
        return self.formation_radius

    @property
    def center_xy(self) -> tuple[float, float]:
        return float(self.center[0]), float(self.center[1])

    @property
    def yaw_clearances(self) -> np.ndarray:
        return self.guarded_clearances

    @property
    def direct_clearances(self) -> np.ndarray:
        return self.guarded_clearances

    @property
    def positive_component(self) -> tuple[float, float] | None:
        return None if self.full_circle else self.anchor_component

    def _component_offsets(self, theta: np.ndarray) -> np.ndarray:
        raw = np.arctan2(np.sin(theta - self.anchor_yaw), np.cos(theta - self.anchor_yaw))
        if self.full_circle:
            return raw
        valid = np.zeros(raw.shape, dtype=bool)
        for period in (-2, -1, 0, 1, 2):
            shifted = raw + 2.0 * math.pi * period
            valid |= (shifted >= float(self.component_lo)) & (shifted <= float(self.component_hi))
        return np.where(valid, raw, np.nan)

    def component_offset(self, theta: float) -> float | None:
        raw = wrap_to_pi(theta - self.anchor_yaw)
        if self.full_circle:
            return raw
        candidates = [raw + 2.0 * math.pi * period for period in (-2, -1, 0, 1, 2)]
        valid = [value for value in candidates
                 if float(self.component_lo) - 1e-12 <= value <= float(self.component_hi) + 1e-12]
        return min(valid, key=abs) if valid else None

    def _radius_scalar(self, theta: float) -> float:
        theta = float(theta)
        lower = -math.inf
        two_radius = 2.0 * self.formation_radius
        for yaw, clearance in zip(self._yaw_slice_values, self._clearance_values):
            delta = math.atan2(math.sin(theta - yaw), math.cos(theta - yaw))
            lower = max(lower, clearance - two_radius * math.sin(abs(delta) / 2.0))
        if self.full_circle:
            in_component = True
        else:
            raw = wrap_to_pi(theta - self.anchor_yaw)
            in_component = any(
                float(self.component_lo) - 1e-12 <= raw + 2.0 * math.pi * period <= float(self.component_hi) + 1e-12
                for period in (-2, -1, 0, 1, 2))
        return float(self.eta * lower) if in_component and lower > 0.0 else 0.0

    def _radius_unchecked(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the lower envelope for yaw(s) already in this cell's component.

        Callers must establish component membership separately.  Keeping this
        path free of component checks is important for the common-yaw batch in
        ``cell_overlap``; the caller still applies the strict positive-radius
        and portal checks before accepting a certificate.
        """
        if not isinstance(theta, np.ndarray) or theta.ndim == 0:
            theta = float(theta)
            lower = -math.inf
            two_radius = 2.0 * self.formation_radius
            for yaw, clearance in zip(self._yaw_slice_values, self._clearance_values):
                delta = math.atan2(math.sin(theta - yaw), math.cos(theta - yaw))
                lower = max(lower, clearance - two_radius * math.sin(abs(delta) / 2.0))
            return self.eta * lower
        values = np.asarray(theta, dtype=float)
        delta = np.arctan2(np.sin(values[..., None] - self.yaw_slices),
                           np.cos(values[..., None] - self.yaw_slices))
        lower = np.max(self.guarded_clearances - 2.0 * self.formation_radius *
                       np.sin(np.abs(delta) / 2.0), axis=-1)
        result = self.eta * lower
        return float(result) if values.ndim == 0 else result

    def radius(self, theta: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(theta, dtype=float)
        if values.ndim == 0:
            return self._radius_scalar(float(values))
        delta = np.arctan2(np.sin(values[..., None] - self.yaw_slices),
                           np.cos(values[..., None] - self.yaw_slices))
        lower = np.max(self.guarded_clearances - 2.0 * self.formation_radius *
                       np.sin(np.abs(delta) / 2.0), axis=-1)
        lower = np.maximum(lower, 0.0)
        offsets = self._component_offsets(values)
        result = self.eta * lower
        result = np.where(np.isfinite(offsets) & (lower > 0.0), result, 0.0)
        return result

    def anchor_radius(self) -> float:
        return self._anchor_radius_value

    @property
    def radius_upper_bound(self) -> float:
        return self._radius_upper_bound

    def valid_at(self, theta: float) -> bool:
        return self.radius(theta) > 0.0


@dataclass(frozen=True)
class EdgeCertificate:
    """Strict common-yaw portal certificate between two orientation cells."""

    witness_yaw: float
    portal_center: tuple[float, float]
    parent_radius: float
    child_radius: float
    parent_slack: float
    child_slack: float
    route_length: float

    @property
    def yaw(self) -> float:
        return self.witness_yaw

    @property
    def common_yaw(self) -> float:
        return self.witness_yaw

    @property
    def yaw_witness(self) -> float:
        return self.witness_yaw

    @property
    def portal_xy(self) -> tuple[float, float]:
        return self.portal_center

    @property
    def certified_route_length(self) -> float:
        return self.route_length


@dataclass
class TubeRRTNode:
    pose: Pose2D
    cell: OrientationSafeCell | float
    parent: int | None
    cost: float
    certificate: EdgeCertificate | None = None

    @property
    def edge_certificate(self) -> EdgeCertificate | None:
        return self.certificate

    @edge_certificate.setter
    def edge_certificate(self, value: EdgeCertificate | None) -> None:
        self.certificate = value

    @property
    def radius(self) -> float:
        # Backward-readable local value, intentionally not a scalar edge ball.
        return self.cell.anchor_radius() if isinstance(self.cell, OrientationSafeCell) else max(0.0, float(self.cell))

    @property
    def safe_cell(self) -> OrientationSafeCell | float:
        return self.cell


@dataclass
class TubeRRTTraceEvent:
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
    cell: OrientationSafeCell | None = None


@dataclass
class TubeRRTResult:
    success: bool
    path_poses: list[Pose2D] = field(default_factory=list)
    # Direct/local clearance samples retained for compatibility; never balls.
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
    slots = np.asarray(slots, dtype=float)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rotation = np.asarray(((c, -s), (s, c)), dtype=float)
    return slots @ rotation.T + np.asarray((pose.x, pose.y))


def interpolate_pose(first: Pose2D, second: Pose2D, alpha: float) -> Pose2D:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return Pose2D(first.x + alpha * (second.x - first.x),
                  first.y + alpha * (second.y - first.y),
                  wrap_to_pi(first.yaw + alpha * wrap_to_pi(second.yaw - first.yaw)))


class TubeRRTPlanner:
    """Joint-state RRT using orientation-sliced safe cells and portals."""

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
        self.robot_radius = float(map_data.robot_radius)
        self.safety_margin = self.config.safety_margin + float(map_data.safety_margin)
        self.formation_radius = float(np.max(np.linalg.norm(self.slots, axis=1)))
        self._slot_x = np.array(self.slots[:, 0], dtype=float, copy=True)
        self._slot_y = np.array(self.slots[:, 1], dtype=float, copy=True)
        self._slot_radii = np.linalg.norm(self.slots, axis=1)
        self._slot_x.setflags(write=False)
        self._slot_y.setflags(write=False)
        self._slot_radii.setflags(write=False)
        self._origin_x, self._origin_y = (float(map_data.origin_xy[0]),
                                          float(map_data.origin_xy[1]))
        self._max_x = self._origin_x + float(map_data.width_m)
        self._max_y = self._origin_y + float(map_data.height_m)
        circles = [primitive for primitive in map_data.obstacle_primitives
                   if primitive.get("type") == "circle"]
        self._circle_centers = np.asarray([primitive["center_xy"] for primitive in circles],
                                          dtype=float).reshape((-1, 2))
        self._circle_radii = np.asarray([float(primitive["radius"]) for primitive in circles],
                                        dtype=float)
        self._circle_guarded_radii = self._circle_radii + self.robot_radius + self.safety_margin
        self._circle_centers.setflags(write=False)
        self._circle_radii.setflags(write=False)
        self._circle_guarded_radii.setflags(write=False)
        self._clearance_margin = self.robot_radius + self.safety_margin
        self.rng = np.random.default_rng(self.config.seed)
        self._witness_candidates_cache: dict[
            tuple[int, int], tuple[OrientationSafeCell, OrientationSafeCell, tuple[float, ...]]
        ] = {}
        self._fast_witness_candidates_cache: dict[
            tuple[int, int], tuple[OrientationSafeCell, OrientationSafeCell, tuple[float, ...]]
        ] = {}
        self._fast_overlap_cache: dict[tuple[int, int], EdgeCertificate | None] = {}
        unsupported = [p.get("type") for p in map_data.obstacle_primitives if p.get("type") != "circle"]
        if unsupported:
            raise ValueError("TubeRRTPlanner supports only circle obstacle primitives")
        if self.config.yaw_slices < 3:
            raise ValueError("yaw_slices must be at least 3")
        offsets = 2.0 * math.pi * np.arange(int(self.config.yaw_slices), dtype=float) / int(self.config.yaw_slices)
        self._cell_offset_cos = np.cos(offsets)
        self._cell_offset_sin = np.sin(offsets)
        self._cell_offsets = offsets
        self._cell_offset_cos.setflags(write=False)
        self._cell_offset_sin.setflags(write=False)
        self._cell_offsets.setflags(write=False)

    def metric(self, first: Pose2D, second: Pose2D) -> float:
        dc = math.hypot(second.x - first.x, second.y - first.y)
        delta = second.yaw - first.yaw
        return dc + self.formation_radius * abs(math.atan2(math.sin(delta), math.cos(delta)))

    def _clearance_from_trig(self, cx: float, cy: float, cos_yaws: np.ndarray,
                             sin_yaws: np.ndarray) -> np.ndarray:
        slot_x = (float(cx) + cos_yaws[:, None] * self._slot_x[None, :]
                  - sin_yaws[:, None] * self._slot_y[None, :])
        slot_y = (float(cy) + sin_yaws[:, None] * self._slot_x[None, :]
                  + cos_yaws[:, None] * self._slot_y[None, :])
        minimum = np.minimum.reduce((
            slot_x - self._origin_x,
            self._max_x - slot_x,
            slot_y - self._origin_y,
            self._max_y - slot_y,
        ))
        if self._circle_centers.shape[0]:
            if self._circle_centers.shape[0] == 1:
                dx = slot_x - self._circle_centers[0, 0]
                dy = slot_y - self._circle_centers[0, 1]
                minimum = np.minimum(minimum, np.hypot(dx, dy) - self._circle_radii[0])
            else:
                dx = slot_x[:, :, None] - self._circle_centers[None, None, :, 0]
                dy = slot_y[:, :, None] - self._circle_centers[None, None, :, 1]
                obstacle_clearances = np.hypot(dx, dy) - self._circle_radii[None, None, :]
                minimum = np.minimum(minimum, np.min(obstacle_clearances, axis=2))
        return np.min(minimum, axis=1) - self._clearance_margin

    def _clearance_yaws(self, cx: float, cy: float, yaws: np.ndarray) -> np.ndarray:
        """Return guarded clearances for one center over a batch of yaws."""
        values = np.asarray(yaws, dtype=float).reshape(-1)
        return self._clearance_from_trig(cx, cy, np.cos(values), np.sin(values))

    def clearance(self, pose: Pose2D) -> float:
        return float(self._clearance_yaws(pose.x, pose.y, np.asarray((pose.yaw,)))[0])

    def safety_radius(self, pose: Pose2D) -> float:
        return max(0.0, self.clearance(pose))

    def safety_cell(self, pose: Pose2D) -> OrientationSafeCell:
        anchor = wrap_to_pi(pose.yaw)
        cos_anchor, sin_anchor = math.cos(anchor), math.sin(anchor)
        cos_yaws = cos_anchor * self._cell_offset_cos - sin_anchor * self._cell_offset_sin
        sin_yaws = sin_anchor * self._cell_offset_cos + cos_anchor * self._cell_offset_sin
        yaws = anchor + self._cell_offsets
        clearances = self._clearance_from_trig(pose.x, pose.y, cos_yaws, sin_yaws)
        return OrientationSafeCell((pose.x, pose.y), anchor, yaws, clearances,
                                   self.formation_radius, eta=self.config.cell_eta)

    orientation_cell = safety_cell
    make_cell = safety_cell

    def _cell_for(self, value: TubeRRTNode | Pose2D) -> OrientationSafeCell | None:
        if isinstance(value, TubeRRTNode):
            return value.cell if isinstance(value.cell, OrientationSafeCell) else None
        return self.safety_cell(value)

    def _cell_intervals_world(self, cell: OrientationSafeCell) -> tuple[tuple[float, float], ...]:
        return cell._world_intervals_cache

    def _witness_candidates(self, first: OrientationSafeCell, second: OrientationSafeCell,
                            *, full_search: bool = True) -> list[float]:
        cache = (self._witness_candidates_cache if full_search
                 else self._fast_witness_candidates_cache)
        key = (id(first), id(second))
        cached = cache.get(key)
        if cached is not None and cached[0] is first and cached[1] is second:
            return list(cached[2])

        intersections: list[tuple[float, float]] = []
        if full_search:
            for alo, ahi in self._cell_intervals_world(first):
                for blo, bhi in self._cell_intervals_world(second):
                    lo, hi = max(alo, blo), min(ahi, bhi)
                    if hi > lo:
                        intersections.append((lo, hi))
        else:
            # Align the second component's five possible 2*pi lifts against
            # the first component's anchor lift.  This is equivalent to the
            # full 25-pair intersection for a periodic component, but leaves
            # only a handful of intervals on the neighbor path.
            first_intervals = self._cell_intervals_world(first)
            second_intervals = self._cell_intervals_world(second)
            first_interval = first_intervals[0] if first.full_circle else first_intervals[len(first_intervals) // 2]
            second_interval = second_intervals[0] if second.full_circle else second_intervals[len(second_intervals) // 2]
            for period in range(-2, 3):
                blo = second_interval[0] + 2.0 * math.pi * period
                bhi = second_interval[1] + 2.0 * math.pi * period
                lo, hi = max(first_interval[0], blo), min(first_interval[1], bhi)
                if hi > lo:
                    intersections.append((lo, hi))
        intersections.sort()
        if not intersections and not (first.full_circle or second.full_circle):
            cache[key] = (first, second, ())
            return []

        if full_search:
            candidates: list[float] = [first.anchor_yaw, second.anchor_yaw]
            candidates.extend(float(v) for v in first.yaw_slices)
            candidates.extend(float(v) for v in second.yaw_slices)
            candidates.extend((lo + hi) / 2.0 for lo, hi in intersections)
            # Bounded adaptive refinement: samples approximate where to look, but
            # every accepted witness is checked by the analytic lower envelope.
            for lo, hi in intersections:
                step = (hi - lo) / 32.0
                candidates.extend(lo + step * index for index in range(1, 32))
        else:
            # Parent selection and rewiring only need a few deterministic
            # witnesses.  The anchors retain the historical tie-breaking order;
            # the common component midpoint and its endpoints cover the cases in
            # which neither anchor lies in the other cell's component.  This is
            # deliberately a witness search, not a safety shortcut: cell_overlap
            # still computes both lower-envelope radii and strict portal slack.
            candidates = [first.anchor_yaw, second.anchor_yaw]
            common_intervals = intersections
            if not common_intervals:
                common_intervals = (self._cell_intervals_world(second) if first.full_circle else
                                    self._cell_intervals_world(first) if second.full_circle else ())
            if len(common_intervals) > 1:
                # Multiple lifts can overlap the chosen first interval at a
                # periodic boundary.  They represent the same wrapped yaw;
                # retain the widest one for midpoint/endpoints while the
                # membership filter below still considers every lift.
                witness_intervals = (max(common_intervals, key=lambda part: part[1] - part[0]),)
            else:
                witness_intervals = common_intervals
            for lo, hi in witness_intervals:
                candidates.extend(((lo + hi) / 2.0, lo, hi))

        if not full_search:
            if first.full_circle and second.full_circle:
                intervals = ()
            else:
                intervals = intersections
                if not intervals:
                    intervals = (self._cell_intervals_world(second) if first.full_circle else
                                 self._cell_intervals_world(first) if second.full_circle else ())
            result: list[float] = []
            seen: set[int] = set()
            for value in candidates:
                if intervals and not any(lo <= value <= hi for lo, hi in intervals):
                    continue
                wrapped_value = math.atan2(math.sin(value), math.cos(value))
                key_value = int(round(wrapped_value * 1e12))
                if key_value in seen:
                    continue
                seen.add(key_value)
                result.append(wrapped_value)
            cache[key] = (first, second, tuple(result))
            return result

        # All useful witnesses must be in a common positive-radius component.
        # Filter before wrapping/deduplicating so invalid anchors and slices do
        # not pay for envelope evaluation.  Inclusive bounds retain the old
        # endpoint behavior; the strict radius check remains in cell_overlap.
        raw = np.asarray(candidates, dtype=float)
        if first.full_circle and second.full_circle:
            # A full-circle component has one canonical interval in the cache,
            # but is valid at every lift of yaw; no filtering is needed.
            in_common = np.ones(raw.shape, dtype=bool)
        else:
            intervals = intersections
            if not intervals:
                intervals = (self._cell_intervals_world(second) if first.full_circle else
                             self._cell_intervals_world(first) if second.full_circle else ())
            in_common = np.zeros(raw.shape, dtype=bool)
            for lo, hi in intervals:
                in_common |= (raw >= lo) & (raw <= hi)
        raw = raw[in_common]
        if raw.size == 0:
            cache[key] = (first, second, ())
            return []

        # Vectorized wrapping and first-occurrence selection preserve stable
        # candidate order while handling periodic component lifts.
        wrapped = np.arctan2(np.sin(raw), np.cos(raw))
        keys = np.rint(wrapped * 1e12).astype(np.int64)
        _, first_indices = np.unique(keys, return_index=True)
        first_indices.sort()
        result = wrapped[first_indices].tolist()
        cache[key] = (first, second, tuple(result))
        return result

    def cell_overlap(self, first: TubeRRTNode | Pose2D, second: TubeRRTNode | Pose2D,
                     distance: float | None = None, *, full_search: bool = True) -> EdgeCertificate | None:
        first_cell, second_cell = self._cell_for(first), self._cell_for(second)
        if first_cell is None or second_cell is None:
            return None
        fast_cache_key = ((id(first_cell), id(second_cell))
                          if (not full_search and isinstance(first, TubeRRTNode) and
                              isinstance(second, TubeRRTNode)) else None)
        if fast_cache_key is not None:
            cached = self._fast_overlap_cache.get(fast_cache_key, ...)
            if cached is not ...:
                return cached
            reverse_key = (fast_cache_key[1], fast_cache_key[0])
            reverse = self._fast_overlap_cache.get(reverse_key, ...)
            if reverse is not ...:
                if reverse is None:
                    self._fast_overlap_cache[fast_cache_key] = None
                    return None
                certificate = EdgeCertificate(
                    reverse.witness_yaw, reverse.portal_center,
                    reverse.child_radius, reverse.parent_radius,
                    reverse.child_slack, reverse.parent_slack, reverse.route_length)
                self._fast_overlap_cache[fast_cache_key] = certificate
                return certificate
        ca, cb = first_cell.center, second_cell.center
        dx, dy = float(cb[0] - ca[0]), float(cb[1] - ca[1])
        center_distance = math.hypot(dx, dy)
        if center_distance >= first_cell.radius_upper_bound + second_cell.radius_upper_bound:
            if fast_cache_key is not None:
                self._fast_overlap_cache[fast_cache_key] = None
            return None
        candidates = [first_cell.anchor_yaw, second_cell.anchor_yaw]
        if not candidates:
            if fast_cache_key is not None:
                self._fast_overlap_cache[fast_cache_key] = None
            return None

        # Both paths check the two historical anchor witnesses before doing
        # component-lift/refinement work.  The complete path still falls back
        # to every candidate when those anchors miss.
        candidate_index = 0
        anchor_only = True
        while True:
            while candidate_index < len(candidates):
                if anchor_only:
                    witness = candidates[candidate_index]
                    if candidate_index == 0:
                        ra = (first_cell.anchor_radius() if not full_search else
                              float(first_cell._radius_unchecked(witness)))
                        rb = float(second_cell._radius_unchecked(witness))
                    else:
                        ra = float(first_cell._radius_unchecked(witness))
                        rb = (second_cell.anchor_radius() if not full_search else
                              float(second_cell._radius_unchecked(witness)))
                    candidate_pairs = ((witness, ra, rb),)
                    candidate_index += 1
                elif not full_search:
                    witness = candidates[candidate_index]
                    candidate_pairs = ((witness,
                                        float(first_cell._radius_unchecked(witness)),
                                        float(second_cell._radius_unchecked(witness))),)
                    candidate_index += 1
                elif candidate_index == 0:
                    witness = candidates[0]
                    candidate_pairs = ((witness,
                                        float(first_cell._radius_unchecked(witness)),
                                        float(second_cell._radius_unchecked(witness))),)
                    candidate_index = 1
                else:
                    candidate_yaws = np.asarray(candidates[candidate_index:], dtype=float)
                    first_radii = np.asarray(first_cell._radius_unchecked(candidate_yaws), dtype=float)
                    second_radii = np.asarray(second_cell._radius_unchecked(candidate_yaws), dtype=float)
                    candidate_pairs = zip(candidates[candidate_index:], first_radii, second_radii)
                    candidate_index = len(candidates)

                for witness, ra, rb in candidate_pairs:
                    ra, rb = float(ra), float(rb)
                    if not (ra > 0.0 and rb > 0.0 and center_distance < ra + rb):
                        continue
                    if center_distance <= 1e-15:
                        portal_x, portal_y = float(ca[0]), float(ca[1])
                    else:
                        radius_sum = ra + rb
                        portal_x = (rb * float(ca[0]) + ra * float(cb[0])) / radius_sum
                        portal_y = (rb * float(ca[1]) + ra * float(cb[1])) / radius_sum
                    parent_distance = math.hypot(portal_x - float(ca[0]), portal_y - float(ca[1]))
                    child_distance = math.hypot(portal_x - float(cb[0]), portal_y - float(cb[1]))
                    parent_slack = ra - parent_distance
                    child_slack = rb - child_distance
                    if not (parent_slack > 0.0 and child_slack > 0.0):
                        continue
                    parent_offset, child_offset = first_cell.component_offset(witness), second_cell.component_offset(witness)
                    if parent_offset is None or child_offset is None:
                        continue
                    route_length = (self.formation_radius * abs(parent_offset) + parent_distance + child_distance +
                                    self.formation_radius * abs(child_offset))
                    certificate = EdgeCertificate(float(wrap_to_pi(witness)), (portal_x, portal_y),
                                                  ra, rb, parent_slack, child_slack, float(route_length))
                    if fast_cache_key is not None:
                        self._fast_overlap_cache[fast_cache_key] = certificate
                    return certificate
            if anchor_only:
                candidates = self._witness_candidates(first_cell, second_cell, full_search=full_search)
                candidate_index = 0
                anchor_only = False
                continue
            break
        if fast_cache_key is not None:
            self._fast_overlap_cache[fast_cache_key] = None
        return None

    def tube_overlap(self, first: TubeRRTNode | Pose2D, second: TubeRRTNode | Pose2D,
                     distance: float | None = None) -> bool:
        # Legacy predicate only; all planner decisions use cell_overlap itself.
        if isinstance(first, TubeRRTNode) and isinstance(second, TubeRRTNode):
            if not isinstance(first.cell, OrientationSafeCell) or not isinstance(second.cell, OrientationSafeCell):
                d = self.metric(first.pose, second.pose) if distance is None else distance
                return d < first.radius + second.radius
        return self.cell_overlap(first, second, distance) is not None

    def _sample_pose(self) -> Pose2D:
        if self.rng.random() < self.config.goal_bias:
            return Pose2D(self.goal_xy[0], self.goal_xy[1], float(self.rng.uniform(-math.pi, math.pi)))
        ox, oy = self.map_data.origin_xy
        return Pose2D(float(self.rng.uniform(ox, ox + self.map_data.width_m)),
                      float(self.rng.uniform(oy, oy + self.map_data.height_m)),
                      float(self.rng.uniform(-math.pi, math.pi)))

    def _steer(self, source: Pose2D, target: Pose2D) -> Pose2D:
        distance = self.metric(source, target)
        return target if distance <= self.config.metric_step else interpolate_pose(
            source, target, self.config.metric_step / distance)

    def _extend(self, near: TubeRRTNode, target: Pose2D) -> tuple[Pose2D, OrientationSafeCell, int, bool, EdgeCertificate | None]:
        distance = self.metric(near.pose, target)
        step = min(self.config.metric_step, distance)
        attempts = 0
        while True:
            attempts += 1
            pose = target if step >= distance else interpolate_pose(near.pose, target, step / distance)
            cell = self.safety_cell(pose)
            candidate = TubeRRTNode(pose, cell, None, 0.0)
            certificate = self.cell_overlap(near, candidate, full_search=True) if cell.anchor_radius() > 0.0 else None
            if certificate is not None:
                return pose, cell, attempts, True, certificate
            near_radius = near.cell.anchor_radius() if isinstance(near.cell, OrientationSafeCell) else near.radius
            next_step = max(0.5 * step, 0.9 * near_radius)
            if not self.config.step_backoff or next_step >= step or next_step < self.config.min_metric_step:
                return pose, cell, attempts, False, None
            step = next_step

    def _anchor_radius(self, node: TubeRRTNode) -> float:
        return node.cell.anchor_radius() if isinstance(node.cell, OrientationSafeCell) else node.radius

    def edge_cost(self, first: TubeRRTNode, second: TubeRRTNode, distance: float | None = None,
                  certificate: EdgeCertificate | None = None) -> float:
        cert = certificate if certificate is not None else second.certificate
        if cert is None:
            length = self.metric(first.pose, second.pose) if distance is None else distance
            clearance = min(self._anchor_radius(first), self._anchor_radius(second))
        else:
            length, clearance = cert.route_length, min(cert.parent_radius, cert.child_radius)
        return length if self.config.margin_weight <= 0.0 else length * (
            1.0 + self.config.margin_weight / max(clearance, 1e-9))

    def _batch_metric_distances(self, node_poses: np.ndarray | None, pose: Pose2D,
                                *, components: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if components is None:
            x_values, y_values, yaw_values = node_poses[:, 0], node_poses[:, 1], node_poses[:, 2]
        else:
            x_values, y_values, yaw_values = components
        dx, dy = pose.x - x_values, pose.y - y_values
        dyaw = pose.yaw - yaw_values
        yaw_distance = np.abs(np.remainder(dyaw + math.pi, 2.0 * math.pi) - math.pi)
        return np.sqrt(dx * dx + dy * dy) + self.formation_radius * yaw_distance

    def _update_descendant_costs(self, nodes: list[TubeRRTNode], root_index: int,
                                 children: list[list[int]]) -> None:
        stack = list(children[root_index])
        while stack:
            index = stack.pop()
            parent = nodes[index].parent
            assert parent is not None
            nodes[index].cost = nodes[parent].cost + self.edge_cost(nodes[parent], nodes[index])
            stack.extend(children[index])

    @staticmethod
    def _append_waypoint(path: list[Pose2D], pose: Pose2D) -> None:
        if not path or path[-1] != pose:
            path.append(pose)

    def _yaw_waypoints(self, cell: OrientationSafeCell, start: float, end: float) -> list[Pose2D]:
        first, last = cell.component_offset(start), cell.component_offset(end)
        if first is None or last is None:
            return [Pose2D(float(cell.center[0]), float(cell.center[1]), wrap_to_pi(end))]
        count = max(1, int(math.ceil(abs(last - first) / (math.pi / 12.0))))
        return [Pose2D(float(cell.center[0]), float(cell.center[1]),
                       wrap_to_pi(cell.anchor_yaw + float(offset)))
                for offset in np.linspace(first, last, count + 1)]

    def _certificate_route(self, parent: TubeRRTNode, child: TubeRRTNode,
                           certificate: EdgeCertificate) -> list[Pose2D]:
        witness = certificate.witness_yaw
        portal = Pose2D(*certificate.portal_center, witness)
        parent_center = Pose2D(float(parent.cell.center[0]), float(parent.cell.center[1]), witness)
        child_center = Pose2D(float(child.cell.center[0]), float(child.cell.center[1]), witness)
        route: list[Pose2D] = []
        for pose in self._yaw_waypoints(parent.cell, parent.pose.yaw, witness):
            self._append_waypoint(route, pose)
        self._append_waypoint(route, portal)
        self._append_waypoint(route, child_center)
        for pose in self._yaw_waypoints(child.cell, witness, child.pose.yaw):
            self._append_waypoint(route, pose)
        if route:
            route[0], route[-1] = parent.pose, child.pose
        return route

    def _path(self, nodes: list[TubeRRTNode], index: int) -> tuple[list[Pose2D], list[float]]:
        indices: list[int] = []
        while True:
            indices.append(index)
            parent = nodes[index].parent
            if parent is None:
                break
            index = parent
        indices.reverse()
        path: list[Pose2D] = []
        for parent_index, child_index in zip(indices, indices[1:]):
            parent, child = nodes[parent_index], nodes[child_index]
            segment = self._certificate_route(parent, child, child.certificate) if (
                child.certificate is not None and isinstance(parent.cell, OrientationSafeCell) and
                isinstance(child.cell, OrientationSafeCell)) else [parent.pose, child.pose]
            for pose in segment:
                self._append_waypoint(path, pose)
        if not path:
            path = [nodes[indices[0]].pose]
        return path, [max(0.0, self.clearance(pose)) for pose in path]

    def _print_progress(self, iteration: int, nodes: list[TubeRRTNode], best_goal_distance: float,
                        best_cost: float | None) -> None:
        cost = "" if best_cost is None else f" best_cost={best_cost:.3f}"
        print(f"progress iteration={iteration}/{self.config.max_iterations} accepted_nodes={len(nodes)} "
              f"best_goal_distance={best_goal_distance:.3f}{cost}", flush=True)

    def plan(self) -> TubeRRTResult:
        start_cell = self.safety_cell(self.start)
        start_clearance = float(start_cell.guarded_clearances[0])
        nodes = [TubeRRTNode(self.start, start_cell, None, 0.0)]
        node_x = np.empty(2 * self.config.max_iterations + 3, dtype=float)
        node_y = np.empty_like(node_x)
        node_yaw = np.empty_like(node_x)
        node_x[0], node_y[0], node_yaw[0] = self.start.x, self.start.y, self.start.yaw
        children: list[list[int]] = [[]]
        if start_clearance <= 0.0 or start_cell.anchor_radius() <= 0.0:
            return TubeRRTResult(False, tree_nodes=nodes, failure_reason="start is in collision", bottleneck=0.0)
        best_bottleneck = start_clearance
        best_goal_distance = math.hypot(self.start.x - self.goal_xy[0], self.start.y - self.goal_xy[1])
        trace: list[TubeRRTTraceEvent] = []
        goal_indices: list[int] = []
        first_goal_iteration: int | None = None
        cost_history: list[tuple[int, float]] = []
        iteration = 0
        for iteration in range(1, self.config.max_iterations + 1):
            best_cost = min((nodes[i].cost for i in goal_indices), default=None)
            if goal_indices and (not cost_history or best_cost < cost_history[-1][1]):
                cost_history.append((iteration - 1, best_cost))
            report = self.config.progress_interval > 0 and iteration % self.config.progress_interval == 0
            sampled = self._sample_pose()
            sample_distances = self._batch_metric_distances(
                None, sampled, components=(node_x[:len(nodes)], node_y[:len(nodes)], node_yaw[:len(nodes)]))
            nearest_index = int(np.flatnonzero(sample_distances <= sample_distances.min() + 1e-12)[0])
            pose, cell, attempts, ok, nearest_certificate = self._extend(nodes[nearest_index], sampled)
            local_clearance = max(0.0, float(cell.guarded_clearances[0]))
            if not ok:
                if self.config.record_trace:
                    trace.append(TubeRRTTraceEvent(iteration, sampled, nearest_index, pose, local_clearance,
                                                   "collision" if local_clearance <= 0.0 else "no_overlap",
                                                   attempts=attempts, cell=cell))
                if report:
                    self._print_progress(iteration, nodes, best_goal_distance, best_cost)
                continue
            new_node = TubeRRTNode(pose, cell, None, 0.0, nearest_certificate)
            pose_distances = self._batch_metric_distances(
                None, pose, components=(node_x[:len(nodes)], node_y[:len(nodes)], node_yaw[:len(nodes)]))
            neighbor_candidates = np.flatnonzero(pose_distances <= self.config.neighbor_radius + 1e-12)
            neighbors = [(int(i), float(pose_distances[i])) for i in neighbor_candidates]
            parent, parent_certificate = nearest_index, nearest_certificate
            parent_cost = (nodes[parent].cost + self.edge_cost(nodes[parent], new_node,
                                                               certificate=parent_certificate)
                           if parent_certificate is not None else math.inf)
            for candidate, distance in neighbors:
                if nodes[candidate].cost + distance >= parent_cost:
                    continue
                certificate = self.cell_overlap(nodes[candidate], new_node, full_search=False)
                if certificate is None:
                    continue
                candidate_cost = nodes[candidate].cost + self.edge_cost(nodes[candidate], new_node,
                                                                          distance, certificate)
                if candidate_cost < parent_cost:
                    parent, parent_cost, parent_certificate = candidate, candidate_cost, certificate
            if parent_certificate is None:
                continue
            new_index = len(nodes)
            new_node.parent, new_node.cost, new_node.certificate = parent, parent_cost, parent_certificate
            nodes.append(new_node)
            node_x[new_index], node_y[new_index], node_yaw[new_index] = pose.x, pose.y, pose.yaw
            children.append([])
            children[parent].append(new_index)
            best_bottleneck = min(best_bottleneck, local_clearance)
            best_goal_distance = min(best_goal_distance, math.hypot(pose.x - self.goal_xy[0], pose.y - self.goal_xy[1]))
            event = TubeRRTTraceEvent(iteration, sampled, nearest_index, pose, local_clearance, "added", new_index,
                                      parent, attempts=attempts, cell=cell)
            if self.config.record_trace:
                trace.append(event)
            for candidate, distance in neighbors:
                if candidate == parent or candidate == 0:
                    continue
                if new_node.cost + distance >= nodes[candidate].cost:
                    continue
                certificate = self.cell_overlap(new_node, nodes[candidate], full_search=False)
                if certificate is None:
                    continue
                new_cost = nodes[new_index].cost + self.edge_cost(new_node, nodes[candidate], certificate=certificate)
                if new_cost < nodes[candidate].cost:
                    old_parent = nodes[candidate].parent
                    assert old_parent is not None
                    event.rewires.append((candidate, old_parent, new_index))
                    children[old_parent].remove(candidate)
                    children[new_index].append(candidate)
                    nodes[candidate].parent, nodes[candidate].certificate, nodes[candidate].cost = (
                        new_index, certificate, new_cost)
                    self._update_descendant_costs(nodes, candidate, children)
            goal_distance = math.hypot(pose.x - self.goal_xy[0], pose.y - self.goal_xy[1])
            improves = best_cost is None or nodes[new_index].cost + goal_distance < best_cost
            if goal_distance <= self.config.goal_connect_distance and improves:
                goal_pose = Pose2D(self.goal_xy[0], self.goal_xy[1], pose.yaw)
                goal_cell = self.safety_cell(goal_pose)
                goal_node = TubeRRTNode(goal_pose, goal_cell, new_index, 0.0)
                goal_certificate = self.cell_overlap(new_node, goal_node)
                if goal_certificate is not None:
                    goal_node.certificate = goal_certificate
                    goal_node.cost = new_node.cost + self.edge_cost(new_node, goal_node,
                                                                     certificate=goal_certificate)
                    if best_cost is None or goal_node.cost < best_cost:
                        goal_index = len(nodes)
                        nodes.append(goal_node)
                        node_x[goal_index], node_y[goal_index], node_yaw[goal_index] = goal_pose.x, goal_pose.y, goal_pose.yaw
                        children.append([])
                        children[new_index].append(goal_index)
                        goal_indices.append(goal_index)
                        if self.config.record_trace:
                            trace.append(TubeRRTTraceEvent(iteration, goal_pose, new_index, goal_pose,
                                                           max(0.0, float(goal_cell.guarded_clearances[0])), "goal", goal_index,
                                                           new_index, cell=goal_cell))
                        best_cost = goal_node.cost
                        cost_history.append((iteration, best_cost))
                        if first_goal_iteration is None:
                            first_goal_iteration = iteration
                            if self.config.progress_interval > 0:
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
            if not cost_history or nodes[best_goal].cost < cost_history[-1][1]:
                cost_history.append((iteration, nodes[best_goal].cost))
            path, radii = self._path(nodes, best_goal)
            if self.config.progress_interval > 0 and not self.config.stop_on_first_goal:
                print(f"search finished iteration={iteration} accepted_nodes={len(nodes)} goal_nodes={len(goal_indices)} "
                      f"best_cost={nodes[best_goal].cost:.3f}", flush=True)
            return TubeRRTResult(True, path, radii, nodes, edges, bottleneck=min(radii, default=0.0), trace=trace,
                                 iterations=iteration, first_goal_iteration=first_goal_iteration,
                                 path_cost=nodes[best_goal].cost, cost_history=cost_history)
        if self.config.progress_interval > 0:
            print(f"search failed iteration budget exhausted iteration={self.config.max_iterations}/{self.config.max_iterations} "
                  f"accepted_nodes={len(nodes)} best_goal_distance={best_goal_distance:.3f}", flush=True)
        return TubeRRTResult(False, tree_nodes=nodes, tree_edges=edges,
                             failure_reason="iteration budget exhausted", bottleneck=best_bottleneck,
                             trace=trace, iterations=iteration)


def project_robot_paths(path_poses: Iterable[Pose2D], slots: np.ndarray) -> list[list[tuple[float, float]]]:
    projected = [[] for _ in range(len(np.asarray(slots)))]
    for pose in path_poses:
        points = transform_slots(pose, slots)
        for index, point in enumerate(points):
            projected[index].append((float(point[0]), float(point[1])))
    return projected
