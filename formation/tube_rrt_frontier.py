"""RRT* over polyhedral SE(2) cells with a region-guided sampling channel (``polyhedral_cell``).

The planner is a standard RRT*; the certified union ``U = cup_i C_i`` only supplies a bounded share
of biased samples.  Every iteration picks one sampling mode with ``xi ~ U(0, 1)``:

* ``xi >= p_region`` (uniform, RRT*): ``q_rand ~ U(SE(2))`` (with goal bias) -> nearest node by
  ``d_G`` -> steer by at most ``metric_step`` and at most ``steer_fraction`` of the way to the
  boundary of the nearest cell -> ``q_new``.  This channel provides global randomness, densification
  near the path, rewiring and anytime improvement.
* ``xi < p_region`` (region-guided): pick one precomputed expansion of an existing cell's exposed
  frontier and use its ``q_new``; the new cell is built there and must add at least ``min_new_ratio``
  of uncovered volume.

Expansions come straight from the frontier geometry of the section ``S_0`` of each new cell, a constant
number per frontier piece, without enumerating ``(u, dtheta)``:

* guard arc (only far, inactive pairs limit the certificate there; arcs are split into chunks of at most
  ``max_arc``): one direction ``normalize(w_t u_tan + w_g u_goal + w_n u_out)`` with the tangent sense
  towards the goal;
* obstacle facet (an active row ``n^T dc - rho |dtheta| >= -(d - d_s)`` binds): never along the violating
  outward normal ``-n``; two directions ``normalize(w_t (+-u_tan) + w_g u_goal_safe + w_n n)`` that slide
  along the facet, with ``u_goal_safe`` the goal direction without its component into the obstacle.

From the representative point ``b`` the direction is followed until it leaves ``S_0`` and then
``sample_offset`` further, so ``q_new`` is just outside the current certificate.  The yaw change is one of
``{-yaw_step, 0, +yaw_step}``, the one with the largest exact half-space margin
``min_k n_k^T (dc + (R(dtheta) - I) R_0 s_k) + e_k`` of all pairs near the cell (``outer_reach``).  An
expansion is kept only if that margin is positive (so ``q_new`` never enters a known restricted region
and its cell is valid), ``q_new`` stays within ``r_g + outer_reach`` (d_G) of the cell, in the map, and
uncovered by other cells.

Only the accepted ``q_new`` gets a cell, so one iteration builds at most one cell.  The new cell must
overlap the source cell or another neighbour; parents are re-chosen among overlapping neighbours and
neighbours are rewired.  A goal node is added when the goal pose lies inside the new cell (it shares
that cell).  Consecutive path nodes are joined by ``q_a -> q_p -> q_b`` with ``q_p`` strictly inside
both convex cells, so the route is certified collision free with clearance ``>= d_s``.

Expansions whose ``q_new`` gets covered by a later cell are dropped.  The score is
``S = alpha min(l / l_ref, 1) + beta U + gamma (G + 1) / 2`` with ``l`` the piece length, ``U`` the share of
probes beyond ``q_new`` outside the union and ``G`` the goal progress per unit length.  No distance query is
spent on it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from formation.polyhedral_cell import PolyhedralCell, PolyhedralCellModel, wrap_array
from formation.tube_cell_first_order import PortalCertificate, interpolate_pose
from formation.tube_rrt import TubeRRTConfig, TubeRRTNode, TubeRRTResult, TubeRRTTraceEvent
from formation.types import FormationSpec, MapData, Pose2D, wrap_to_pi

REGION_SCHEDULES = ("switch", "constant", "exp")


@dataclass
class FrontierConfig:
    safety_distance: float = 0.02
    """``d_s``: clearance kept inside every cell on top of robot radius and safety margin."""
    active_range: float = 1.0
    """Broadphase ``d_active``: pairs with guarded clearance below it may bound the cell."""
    parallel_tolerance: float = math.radians(5.0)
    """``eps``: of two rows with ``n_a^T n_b > cos eps`` only the tighter is kept."""
    max_extent: float = 1.5
    yaw_chart: float = math.pi / 2.0
    cell_samples: int = 128
    outer_reach: float = 0.5
    """Pairs with ``d - d_s < r_g + outer_reach`` choose the yaw change and validate seeds beyond the cell."""
    region_schedule: str = "switch"
    """``switch``: ``region_before`` until the first solution, then ``region_after``; ``constant``:
    ``region_probability``; ``exp``: ``region_min + (region_max - region_min) exp(-region_decay t)``."""
    region_probability: float = 0.4
    region_before: float = 0.5
    region_after: float = 0.2
    region_max: float = 0.5
    region_min: float = 0.2
    region_decay: float = 0.002
    max_arc: float = math.pi / 4.0
    """Guard arcs longer than this are split so each chunk has one outward direction."""
    guard_weights: tuple[float, float, float] = (0.3, 0.5, 1.0)
    """``(w_t, w_g, w_n)`` on guard arcs (``u_safe`` = outward normal)."""
    obstacle_weights: tuple[float, float, float] = (1.0, 0.3, 0.2)
    """``(w_t, w_g, w_n)`` on obstacle facets (``u_safe`` = facet normal pointing away from the obstacle)."""
    yaw_step: float = 0.15
    """``dtheta`` of an expansion is one of ``{-yaw_step, 0, +yaw_step}``."""
    sample_offset: float = 0.1
    """``delta`` [m]: how far beyond the exit point of the current section ``q_new`` is placed."""
    probe_step: float = 0.25
    probe_count: int = 3
    score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """``(alpha, beta, gamma)`` of ``S = alpha l + beta U + gamma G``."""
    score_temperature: float = 0.15
    length_ref: float = 1.0
    steer_fraction: float = 0.9
    min_new_ratio: float = 0.05
    overlap_band: tuple[float, float] = (0.1, 0.5)
    """Reference band of ``rho_overlap`` (reported, not enforced)."""
    max_parent_candidates: int = 12

    def __post_init__(self) -> None:
        if self.region_schedule not in REGION_SCHEDULES:
            raise ValueError(f"region_schedule must be one of {REGION_SCHEDULES}")
        for name in ("region_probability", "region_before", "region_after", "region_max", "region_min"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        low, high = self.overlap_band
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("overlap_band must satisfy 0 <= rho_min < rho_max <= 1")
        if self.score_temperature <= 0.0 or not 0.0 < self.steer_fraction < 1.0:
            raise ValueError("score_temperature > 0 and steer_fraction in (0, 1) are required")
        if self.sample_offset <= 0.0 or self.yaw_step < 0.0 or not 0.0 < self.max_arc <= math.pi:
            raise ValueError("sample_offset > 0, yaw_step >= 0 and max_arc in (0, pi] are required")
        if self.guard_weights[2] <= 0.0 or self.obstacle_weights[0] <= 0.0:
            raise ValueError("guard w_n and obstacle w_t must be positive")

    def region_probability_at(self, iteration: int, solved: bool) -> float:
        if self.region_schedule == "constant":
            return self.region_probability
        if self.region_schedule == "switch":
            return self.region_after if solved else self.region_before
        return self.region_min + (self.region_max - self.region_min) * math.exp(-self.region_decay * iteration)

    def schedule_label(self) -> str:
        if self.region_schedule == "constant":
            return f"p_region={self.region_probability:g}"
        if self.region_schedule == "switch":
            return f"p_region={self.region_before:g}->{self.region_after:g} at first solution"
        return (f"p_region={self.region_min:g}+({self.region_max:g}-{self.region_min:g})"
                f"exp(-{self.region_decay:g}t)")


FRONTIER_GUARD, FRONTIER_OBSTACLE = 1, 2


class _Frontier:
    """Precomputed expansions of all cells in growable arrays.

    ``anchors`` are the representative frontier points ``b``, ``points`` the seeds ``q_new``, ``directions``
    the planar unit directions ``u_expand`` and ``dtheta`` the chosen yaw change; ``kind`` is
    ``FRONTIER_GUARD`` or ``FRONTIER_OBSTACLE``.
    """

    _FIELDS = ("cell", "anchors", "points", "directions", "dtheta", "length", "progress", "kind", "probes",
               "probe_open", "alive", "score")

    def __init__(self, probe_count: int) -> None:
        self.size = 0
        self.probe_count = probe_count
        self._allocate(1024)

    def _allocate(self, capacity: int) -> None:
        old = {name: getattr(self, name) for name in self._FIELDS} if self.size else None
        self.cell = np.zeros(capacity, dtype=int)
        self.anchors = np.zeros((capacity, 3))
        self.points = np.zeros((capacity, 3))
        self.directions = np.zeros((capacity, 2))
        self.dtheta = np.zeros(capacity)
        self.length = np.zeros(capacity)
        self.progress = np.zeros(capacity)
        self.kind = np.zeros(capacity, dtype=int)
        self.probes = np.zeros((capacity, self.probe_count, 3))
        self.probe_open = np.zeros((capacity, self.probe_count), dtype=bool)
        self.alive = np.zeros(capacity, dtype=bool)
        self.score = np.zeros(capacity)
        if old is not None:
            for name in self._FIELDS:
                getattr(self, name)[:self.size] = old[name][:self.size]

    def add(self, count: int) -> slice:
        if self.size + count > len(self.cell):
            self._allocate(max(2 * len(self.cell), self.size + count))
        span = slice(self.size, self.size + count)
        self.size += count
        return span

    def live(self) -> np.ndarray:
        return np.flatnonzero(self.alive[:self.size])


class PolyhedralFrontierPlanner:
    """RRT* over polyhedral SE(2) cells plus a region-guided sampling channel (``cell_model = "polyhedral"``)."""

    def __init__(self, map_data: MapData, formation: FormationSpec | np.ndarray, start: Pose2D,
                 goal_xy: tuple[float, float] | None = None, config: TubeRRTConfig | None = None,
                 frontier_config: FrontierConfig | None = None) -> None:
        self.map_data = map_data
        self.slots = np.asarray(formation.slots if isinstance(formation, FormationSpec) else formation, dtype=float)
        if self.slots.ndim != 2 or self.slots.shape[1] != 2 or len(self.slots) == 0:
            raise ValueError("formation slots must have shape (N, 2)")
        self.start = start
        self.goal_xy = map_data.goal_xy if goal_xy is None else tuple(goal_xy)
        self.config = config or TubeRRTConfig(cell_model="polyhedral")
        self.frontier_config = frontier_config or FrontierConfig()
        self.robot_radius = float(map_data.robot_radius)
        self.safety_margin = self.config.safety_margin + float(map_data.safety_margin)
        self._clearance_margin = self.robot_radius + self.safety_margin
        f = self.frontier_config
        self.cells = PolyhedralCellModel(self.slots, map_data, self._clearance_margin, f.safety_distance,
                                         f.active_range, f.parallel_tolerance, f.max_extent, f.yaw_chart,
                                         f.cell_samples, f.outer_reach)
        self.formation_radius = self.cells.rho
        self._origin = np.asarray(map_data.origin_xy, dtype=float)
        self._upper = self._origin + (float(map_data.width_m), float(map_data.height_m))
        self._circle_centers = self.cells.circle_centers
        self._circle_radii = self.cells.circle_radii
        self.rng = np.random.default_rng(self.config.seed)
        self.frontier = _Frontier(f.probe_count)
        self.expansion_log: list[dict[str, float | int | str]] = []

    # -- geometry --------------------------------------------------------------------------------------
    def metric(self, first: Pose2D, second: Pose2D) -> float:
        """``d_G = ||dc|| + rho |dtheta|``: an upper bound on every robot's displacement."""
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

    def safety_cell(self, pose: Pose2D) -> PolyhedralCell:
        return self.cells.make_cell(pose)

    def cell_overlap(self, first: TubeRRTNode | PolyhedralCell,
                     second: TubeRRTNode | PolyhedralCell) -> PortalCertificate | None:
        first = first.cell if isinstance(first, TubeRRTNode) else first
        second = second.cell if isinstance(second, TubeRRTNode) else second
        return self.cells.overlap(first, second)

    def densify(self, poses: list[Pose2D], step: float | None = None) -> list[Pose2D]:
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

    def edge_cost(self, first: TubeRRTNode, second: TubeRRTNode, certificate: PortalCertificate) -> float:
        length = certificate.route_length
        if self.config.margin_weight <= 0.0:
            return length
        return length * (1.0 + self.config.margin_weight / max(min(first.radius, second.radius), 1e-9))

    # -- union of cells ----------------------------------------------------------------------------------
    def _nearby(self, x: float, y: float, yaw: float, reach: float, yaw_reach: float,
                exclude: int | None = None) -> np.ndarray:
        count = len(self._nodes)
        distance = np.hypot(self._x[:count] - x, self._y[:count] - y)
        dyaw = np.abs(wrap_array(self._yaw[:count] - yaw))
        mask = (distance < self._guard[:count] + reach) & (dyaw < self._yaw_limit[:count] + yaw_reach)
        if exclude is not None and exclude < count:
            mask[exclude] = False
        return np.flatnonzero(mask)

    def _register_rows(self, index: int, cell: PolyhedralCell) -> None:
        """Append the rows of ``cell`` in world form ``n^T c - rho |dtheta| + (e - n^T c_0) >= 0``."""
        count = len(cell.offsets)
        if self._row_count + count > len(self._row_offset):
            size = max(2 * len(self._row_offset), self._row_count + count)
            self._row_normal = np.vstack((self._row_normal, np.zeros((size - len(self._row_normal), 2))))
            self._row_offset = np.concatenate((self._row_offset, np.zeros(size - len(self._row_offset))))
        rows = slice(self._row_count, self._row_count + count)
        self._row_normal[rows] = cell.normals
        self._row_offset[rows] = cell.offsets - cell.normals @ (cell.pose.x, cell.pose.y)
        self._row_start[index], self._row_size[index] = self._row_count, count
        self._row_count += count

    def _covered(self, points: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Whether each world configuration lies strictly inside at least one of ``cells``."""
        if not len(cells) or not len(points):
            return np.zeros(len(points), dtype=bool)
        turn = self.formation_radius * np.abs(wrap_array(points[None, :, 2] - self._yaw[cells][:, None]))
        slack = np.minimum(self._guard[cells][:, None] - turn - np.hypot(points[None, :, 0] - self._x[cells][:, None],
                                                                        points[None, :, 1] - self._y[cells][:, None]),
                           self.formation_radius * self._yaw_limit[cells][:, None] - turn)
        sizes = self._row_size[cells]
        with_rows = np.flatnonzero(sizes)
        if len(with_rows):
            sizes, starts = sizes[with_rows], self._row_start[cells][with_rows]
            bounds = np.concatenate(([0], np.cumsum(sizes)[:-1]))
            rows = np.repeat(starts - bounds, sizes) + np.arange(int(sizes.sum()))
            values = (self._row_normal[rows] @ points[:, :2].T + self._row_offset[rows, None]
                      - np.repeat(turn[with_rows], sizes, axis=0))
            slack[with_rows] = np.minimum(slack[with_rows], np.minimum.reduceat(values, bounds, axis=0))
        return np.any(slack > 1e-12, axis=0)

    def _new_ratio(self, cell: PolyhedralCell) -> float:
        near = self._nearby(cell.pose.x, cell.pose.y, cell.pose.yaw, cell.guard, cell.yaw_limit)
        return float(np.mean(~self._covered(cell.samples, near)))

    def _overlap_ratio(self, parent: PolyhedralCell, cell: PolyhedralCell) -> float:
        in_parent = float(np.mean(self.cells.contains(parent, cell.samples)))
        in_cell = float(np.mean(self.cells.contains(cell, parent.samples)))
        intersection = 0.5 * (cell.volume * in_parent + parent.volume * in_cell)
        return float(np.clip(intersection / max(min(parent.volume, cell.volume), 1e-12), 0.0, 1.0))

    # -- frontier ---------------------------------------------------------------------------------------
    def _goal_distance(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        return np.hypot(np.asarray(x) - self.goal_xy[0], np.asarray(y) - self.goal_xy[1])

    def _score(self, indices: np.ndarray) -> None:
        f, store = self.frontier_config, self.frontier
        alpha, beta, gamma = f.score_weights
        uncovered = np.mean(store.probe_open[indices], axis=1)
        store.score[indices] = (alpha * np.minimum(store.length[indices] / f.length_ref, 1.0) + beta * uncovered
                                + gamma * 0.5 * (store.progress[indices] + 1.0))

    @staticmethod
    def _section_exit(cell: PolyhedralCell, anchors: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """``max{t >= 0 : anchor + t u in S_0}`` for chart anchors on the boundary of ``S_0`` and planar ``u``."""
        along = np.sum(anchors * directions, axis=1)
        exits = -along + np.sqrt(np.maximum(along ** 2 - np.sum(anchors ** 2, axis=1) + cell.guard ** 2, 0.0))
        if len(cell.offsets):
            rate = -(directions @ cell.normals.T)
            margin = anchors @ cell.normals.T + cell.offsets[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                limits = np.where(rate > 1e-12, np.maximum(margin, 0.0) / np.maximum(rate, 1e-300), math.inf)
            exits = np.minimum(exits, limits.min(axis=1))
        return np.maximum(exits, 0.0)

    def _add_frontier(self, index: int) -> None:
        """Expansions of cell ``index``: a constant number per frontier piece of its section ``S_0``.

        Guard arcs expand along ``normalize(w_t u_tan + w_g u_goal + w_n u_out)``; obstacle facets slide
        along both tangent senses with ``u_safe`` the facet normal away from the obstacle, never along the
        violating normal.  ``q_new`` is ``sample_offset`` beyond the exit from ``S_0`` with the yaw change in
        ``{0, +yaw_step, -yaw_step}`` of largest exact margin of the pairs near the cell.
        """
        f, cell = self.frontier_config, self._nodes[index].cell
        center = np.array((cell.pose.x, cell.pose.y))
        anchors, directions, lengths, kinds = [], [], [], []
        for piece in self.cells.frontier_pieces(cell, f.max_arc):
            anchor = piece["point"]
            goal = np.asarray(self.goal_xy) - center - anchor
            goal /= max(float(np.linalg.norm(goal)), 1e-12)
            tangent = piece["tangent"]
            if piece["kind"] == "guard":
                w_t, w_g, w_n = f.guard_weights
                tangent = tangent if tangent @ goal >= 0.0 else -tangent
                options = [(w_t * tangent + w_g * goal + w_n * piece["outward"], FRONTIER_GUARD)]
            else:
                w_t, w_g, w_n = f.obstacle_weights
                normal = -piece["outward"]
                safe_goal = goal - min(0.0, float(goal @ normal)) * normal
                options = [(w_t * sign * tangent + w_g * safe_goal + w_n * normal, FRONTIER_OBSTACLE)
                           for sign in (1.0, -1.0)]
            for u, kind in options:
                anchors.append(anchor)
                directions.append(u / max(float(np.linalg.norm(u)), 1e-12))
                lengths.append(piece["length"])
                kinds.append(kind)
        if not anchors:
            return
        anchors, directions = np.asarray(anchors), np.asarray(directions)
        lengths, kinds = np.asarray(lengths), np.asarray(kinds)
        seeds = center + anchors + (self._section_exit(cell, anchors, directions) + f.sample_offset)[:, None] * directions
        choices = np.array((0.0, f.yaw_step, -f.yaw_step))
        margins = np.stack([self.cells.rotation_margin(cell, np.column_stack((seeds, np.full(len(seeds), cell.pose.yaw + c))))
                            for c in choices])
        best = np.argmax(margins, axis=0)
        dthetas = choices[best]
        points = np.column_stack((seeds, wrap_array(cell.pose.yaw + dthetas)))
        reasons = np.full(len(points), "", dtype=object)
        reasons[margins[best, np.arange(len(points))] <= 0.0] = "obstacle"
        reach = np.hypot(seeds[:, 0] - center[0], seeds[:, 1] - center[1]) + cell.rho * np.abs(dthetas)
        reasons[(reasons == "") & (reach > cell.guard + self.cells.outer_reach)] = "far"
        reasons[(reasons == "") & (self.cells.slack_many(cell, points) >= 0.0)] = "inside"
        outside = np.any(seeds <= self._origin, axis=1) | np.any(seeds >= self._upper, axis=1)
        reasons[(reasons == "") & outside] = "bounds"
        pending = np.flatnonzero(reasons == "")
        if len(pending):
            others = self._nearby(center[0], center[1], cell.pose.yaw, cell.guard + self.cells.outer_reach,
                                  cell.yaw_limit + f.yaw_step, exclude=index)
            covered = self._covered(points[pending], others)
            reasons[pending[covered]] = "covered"
        for reason in ("obstacle", "far", "inside", "bounds", "covered"):
            self._stats[f"frontier_reject_{reason}"] += int(np.count_nonzero(reasons == reason))
        keep = reasons == ""
        for kind, name in ((FRONTIER_GUARD, "guard"), (FRONTIER_OBSTACLE, "obstacle")):
            self._stats[f"frontier_{name}_directions"] += int(np.count_nonzero(kinds == kind))
        if not np.any(keep):
            return
        anchors, directions, lengths, kinds = anchors[keep], directions[keep], lengths[keep], kinds[keep]
        points, dthetas = points[keep], dthetas[keep]
        steps = f.probe_step * np.arange(1, f.probe_count + 1)
        probes = np.repeat(points[:, None, :], f.probe_count, axis=1)
        probes[:, :, :2] += steps[None, :, None] * directions[:, None, :]
        near = self._nearby(center[0], center[1], cell.pose.yaw, cell.guard + self.cells.outer_reach + steps[-1],
                            cell.yaw_limit + f.yaw_step)
        open_space = ~self._covered(probes.reshape((-1, 3)), near)
        world_anchors = np.column_stack((center + anchors, np.full(len(anchors), cell.pose.yaw)))
        span = self.frontier.add(len(points))
        store = self.frontier
        store.cell[span], store.anchors[span], store.points[span] = index, world_anchors, points
        store.directions[span], store.dtheta[span], store.length[span], store.kind[span] = (directions, dthetas,
                                                                                           lengths, kinds)
        travel = np.hypot(points[:, 0] - world_anchors[:, 0], points[:, 1] - world_anchors[:, 1])
        store.progress[span] = np.clip((self._goal_distance(world_anchors[:, 0], world_anchors[:, 1])
                                        - self._goal_distance(points[:, 0], points[:, 1])) / np.maximum(travel, 1e-9),
                                       -1.0, 1.0)
        store.probes[span] = probes
        store.probe_open[span] = open_space.reshape((-1, f.probe_count))
        store.alive[span] = True
        self._score(np.arange(span.start, span.stop))
        self._stats["frontier_candidates"] += len(points)

    def _cover_frontier(self, index: int) -> None:
        """Drop candidates that the new cell ``index`` covers and close the probes it covers."""
        store, cell = self.frontier, self._nodes[index].cell
        live = store.live()
        if not len(live):
            return
        reach = cell.guard + self.frontier_config.probe_count * self.frontier_config.probe_step
        close = live[np.hypot(store.points[live, 0] - cell.pose.x, store.points[live, 1] - cell.pose.y) < reach]
        if not len(close):
            return
        covered = self.cells.contains(cell, store.points[close])
        store.alive[close[covered]] = False
        self._stats["frontier_covered"] += int(np.count_nonzero(covered))
        remaining = close[~covered]
        if len(remaining):
            inside = self.cells.contains(cell, store.probes[remaining].reshape((-1, 3))).reshape((len(remaining), -1))
            store.probe_open[remaining] &= ~inside
            self._score(remaining)

    def _pick_candidate(self) -> int | None:
        live = self.frontier.live()
        if not len(live):
            return None
        score = self.frontier.score[live]
        weights = np.exp((score - score.max()) / self.frontier_config.score_temperature)
        return int(live[self.rng.choice(len(live), p=weights / weights.sum())])

    # -- sampling and steering ----------------------------------------------------------------------------
    def _sample_pose(self) -> Pose2D:
        if self.rng.random() < self.config.goal_bias:
            return Pose2D(self.goal_xy[0], self.goal_xy[1], float(self.rng.uniform(-math.pi, math.pi)))
        ox, oy = self.map_data.origin_xy
        return Pose2D(float(self.rng.uniform(ox, ox + self.map_data.width_m)),
                      float(self.rng.uniform(oy, oy + self.map_data.height_m)),
                      float(self.rng.uniform(-math.pi, math.pi)))

    def _distances_to(self, pose: Pose2D) -> np.ndarray:
        count = len(self._nodes)
        return (np.hypot(self._x[:count] - pose.x, self._y[:count] - pose.y)
                + self.formation_radius * np.abs(wrap_array(self._yaw[:count] - pose.yaw)))

    def _steer(self, near: TubeRRTNode, target: Pose2D, max_length: float) -> Pose2D | None:
        """Chart-linear step towards ``target``: at most ``max_length`` (d_G) and strictly inside ``near.cell``."""
        dc = (target.x - near.pose.x, target.y - near.pose.y)
        dphi = wrap_to_pi(target.yaw - near.pose.yaw)
        length = math.hypot(*dc) + self.formation_radius * abs(dphi)
        if length <= 0.0:
            return None
        t = min(1.0, max_length / length, self.frontier_config.steer_fraction * self.cells.ray_extent(near.cell, dc, dphi))
        if t * length < self.config.min_metric_step:
            return None
        return Pose2D(near.pose.x + t * dc[0], near.pose.y + t * dc[1], wrap_to_pi(near.pose.yaw + t * dphi))

    def _region_sample(self) -> tuple[int, int, Pose2D] | None:
        """The ``q_new`` of a live expansion drawn by ``exp(S / tau)`` (``None`` when the frontier is empty)."""
        candidate = self._pick_candidate()
        if candidate is None:
            return None
        q = self.frontier.points[candidate]
        return candidate, int(self.frontier.cell[candidate]), Pose2D(float(q[0]), float(q[1]), float(q[2]))

    def _fail_candidate(self, candidate: int) -> None:
        """Expansions are deterministic, so a rejected one is dropped."""
        self.frontier.alive[candidate] = False
        self._stats["frontier_dropped"] += 1

    # -- tree -------------------------------------------------------------------------------------------
    def _store(self, node: TubeRRTNode) -> int:
        index = len(self._nodes)
        if index == len(self._x):
            for name in ("_x", "_y", "_yaw", "_guard", "_yaw_limit", "_cost", "_row_start", "_row_size"):
                old = getattr(self, name)
                setattr(self, name, np.concatenate((old, np.zeros(len(old), dtype=old.dtype))))
        self._nodes.append(node)
        self._children.append([])
        cell = node.cell
        self._register_rows(index, cell)
        self._x[index], self._y[index], self._yaw[index] = cell.pose.x, cell.pose.y, cell.pose.yaw
        self._guard[index], self._yaw_limit[index], self._cost[index] = cell.guard, cell.yaw_limit, node.cost
        if node.parent is not None:
            self._children[node.parent].append(index)
        return index

    def _update_descendant_costs(self, root: int) -> None:
        stack = list(self._children[root])
        while stack:
            index = stack.pop()
            node = self._nodes[index]
            node.cost = self._nodes[node.parent].cost + self.edge_cost(self._nodes[node.parent], node, node.certificate)
            self._cost[index] = node.cost
            stack.extend(self._children[index])

    def _neighbors(self, cell: PolyhedralCell) -> tuple[np.ndarray, np.ndarray]:
        """Non-goal nodes whose cells can overlap ``cell`` (guard disks and yaw ranges intersect), nearest first."""
        pose = cell.pose
        distances = self._distances_to(pose)
        neighbors = self._nearby(pose.x, pose.y, pose.yaw, cell.guard, cell.yaw_limit)
        if self._goal_nodes:
            neighbors = neighbors[~np.isin(neighbors, list(self._goal_nodes))]
        neighbors = neighbors[np.argsort(distances[neighbors], kind="stable")][:self.frontier_config.max_parent_candidates]
        return neighbors, distances

    def _path(self, index: int) -> tuple[list[Pose2D], list[int]]:
        indices: list[int] = []
        while index is not None:
            indices.append(index)
            index = self._nodes[index].parent
        indices.reverse()
        route = [self._nodes[indices[0]].pose]
        for child in indices[1:]:
            for pose in (self._nodes[child].certificate.portal, self._nodes[child].pose):
                if pose != route[-1]:
                    route.append(pose)
        return route, indices

    def frontier_snapshot(self) -> dict[str, np.ndarray]:
        """Live expansions: anchor ``b``, seed ``q_new``, planar direction, yaw change, kind, score, owner cell."""
        live = self.frontier.live()
        store = self.frontier
        return {name: getattr(store, name)[live].copy()
                for name in ("anchors", "points", "directions", "dtheta", "kind", "score", "cell", "length")}

    def _print_progress(self, iteration: int, best_goal_distance: float, best_cost: float | None) -> None:
        cost = "" if best_cost is None else f" best_cost={best_cost:.3f}"
        print(f"progress iteration={iteration}/{self.config.max_iterations} accepted_nodes={len(self._nodes)} "
              f"frontier={len(self.frontier.live())} best_goal_distance={best_goal_distance:.3f}{cost}", flush=True)

    def _reset(self) -> None:
        f = self.frontier_config
        self.rng = np.random.default_rng(self.config.seed)
        self.frontier = _Frontier(f.probe_count)
        self.expansion_log = []
        for key in self.cells.stats:
            self.cells.stats[key] = 0
        self._stats = {"region_iterations": 0, "uniform_iterations": 0, "region_nodes": 0, "uniform_nodes": 0,
                       "region_fallback": 0, "frontier_candidates": 0, "frontier_covered": 0, "frontier_dropped": 0,
                       "frontier_guard_directions": 0, "frontier_obstacle_directions": 0,
                       "frontier_reject_obstacle": 0, "frontier_reject_far": 0, "frontier_reject_inside": 0,
                       "frontier_reject_bounds": 0, "frontier_reject_covered": 0, "rejected_redundant": 0,
                       "rejected_no_progress": 0, "rejected_collision": 0, "rejected_no_overlap": 0}
        self._goal_nodes: set[int] = set()
        self._nodes: list[TubeRRTNode] = []
        self._children: list[list[int]] = []
        capacity = 1024
        self._x, self._y, self._yaw, self._guard, self._yaw_limit, self._cost = (np.zeros(capacity) for _ in range(6))
        self._row_start, self._row_size = np.zeros(capacity, dtype=int), np.zeros(capacity, dtype=int)
        self._row_normal, self._row_offset = np.zeros((8 * capacity, 2)), np.zeros(8 * capacity)
        self._row_count = 0

    def plan(self) -> TubeRRTResult:
        config, f = self.config, self.frontier_config
        started = time.perf_counter()
        self._reset()
        start_cell = self.cells.make_cell(self.start)
        start_node = TubeRRTNode(self.start, start_cell, None, 0.0)
        if not start_cell.valid:
            return TubeRRTResult(False, tree_nodes=[start_node], failure_reason="start is in collision",
                                 overlap_stats={**self.cells.stats, **self._stats})
        start_cell.mode = "start"
        self._store(start_node)
        self._add_frontier(0)
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
            region = self.rng.random() < f.region_probability_at(iteration, best_cost is not None)
            picked = self._region_sample() if region else None
            if region and picked is None:
                self._stats["region_fallback"] += 1
            if picked is not None:
                mode = "region"
                candidate, parent, sample = picked
                q_new, origin = sample, parent
            else:
                mode, candidate, origin = "uniform", None, None
                sample = self._sample_pose()
                distances = self._distances_to(sample)
                distances[list(self._goal_nodes)] = math.inf
                parent = int(np.argmin(distances))
                q_new = self._steer(self._nodes[parent], sample, config.metric_step)
            self._stats[f"{mode}_iterations"] += 1

            cell = self.cells.make_cell(q_new) if q_new is not None else None
            status = "no_progress" if cell is None else ("added" if cell.valid else "collision")
            new_ratio = self._new_ratio(cell) if status == "added" else 0.0
            if mode == "region" and status == "added" and new_ratio < f.min_new_ratio:
                status = "redundant"
            neighbors = distances = parent_certificate = None
            if status == "added":
                neighbors, distances = self._neighbors(cell)
                parent_certificate = self.cells.overlap(self._nodes[parent].cell, cell)
                if parent_certificate is None and mode == "region":
                    order = np.argsort(self._cost[neighbors] + distances[neighbors], kind="stable")
                    for neighbor in neighbors[order].tolist():
                        if neighbor != parent:
                            parent_certificate = self.cells.overlap(self._nodes[neighbor].cell, cell)
                            if parent_certificate is not None:
                                parent = neighbor
                                break
                if parent_certificate is None:
                    status = "no_overlap"
            if status != "added":
                self._stats[f"rejected_{status}"] += 1
                if candidate is not None:
                    self._fail_candidate(candidate)
                if config.record_trace:
                    trace.append(TubeRRTTraceEvent(iteration, sample, parent, q_new or sample,
                                                   cell.radius if cell is not None else 0.0, status,
                                                   cell=cell, mode=mode))
                if report:
                    self._print_progress(iteration, best_goal_distance, best_cost)
                continue

            overlap_ratio = self._overlap_ratio(self._nodes[parent].cell, cell)
            cell.new_ratio, cell.overlap_ratio, cell.mode = new_ratio, overlap_ratio, mode
            new_node = TubeRRTNode(q_new, cell, None, 0.0)
            source = parent
            parent_cost = self._nodes[parent].cost + self.edge_cost(self._nodes[parent], new_node, parent_certificate)
            order = np.argsort(self._cost[neighbors] + distances[neighbors], kind="stable")
            for neighbor in neighbors[order].tolist():
                if self._cost[neighbor] + distances[neighbor] >= parent_cost:
                    break
                if neighbor == parent:
                    continue
                certificate = self.cells.overlap(self._nodes[neighbor].cell, cell)
                if certificate is None:
                    continue
                cost = self._nodes[neighbor].cost + self.edge_cost(self._nodes[neighbor], new_node, certificate)
                if cost < parent_cost:
                    parent, parent_cost, parent_certificate = neighbor, cost, certificate
            new_node.parent, new_node.cost, new_node.certificate = parent, parent_cost, parent_certificate
            new_index = self._store(new_node)
            self._stats[f"{mode}_nodes"] += 1
            self.expansion_log.append({"iteration": iteration, "node": new_index, "mode": mode, "new_ratio": new_ratio,
                                       "overlap_ratio": overlap_ratio, "active_pairs": cell.active_count,
                                       "broadphase_pairs": cell.broadphase_pairs, "yaw_limit": cell.yaw_limit,
                                       "source": origin,
                                       "frontier_kind": int(self.frontier.kind[candidate]) if candidate is not None else 0,
                                       "frontier_point": (self.frontier.anchors[candidate].copy()
                                                          if candidate is not None else None)})
            event = TubeRRTTraceEvent(iteration, sample, source, q_new, cell.radius, "added", new_index, parent,
                                      cell=cell, mode=mode)
            if config.record_trace:
                trace.append(event)

            for neighbor in neighbors.tolist():
                if neighbor in (parent, 0) or parent_cost + distances[neighbor] >= self._nodes[neighbor].cost:
                    continue
                certificate = self.cells.overlap(cell, self._nodes[neighbor].cell)
                if certificate is None:
                    continue
                cost = parent_cost + self.edge_cost(new_node, self._nodes[neighbor], certificate)
                if cost < self._nodes[neighbor].cost:
                    old_parent = self._nodes[neighbor].parent
                    event.rewires.append((neighbor, old_parent, new_index))
                    self._children[old_parent].remove(neighbor)
                    self._children[new_index].append(neighbor)
                    self._nodes[neighbor].parent, self._nodes[neighbor].certificate = new_index, certificate
                    self._nodes[neighbor].cost = self._cost[neighbor] = cost
                    self._update_descendant_costs(neighbor)

            self._cover_frontier(new_index)
            self._add_frontier(new_index)
            goal_distance = float(self._goal_distance(q_new.x, q_new.y))
            best_goal_distance = min(best_goal_distance, goal_distance)

            if goal_indices:
                best_cost = min(self._nodes[i].cost for i in goal_indices)
                if best_cost < cost_history[-1][1]:
                    cost_history.append((iteration, best_cost))
            goal_pose = Pose2D(self.goal_xy[0], self.goal_xy[1], q_new.yaw)
            goal_slack = self.cells.slack(cell, goal_pose)
            if goal_slack > 0.0 and (best_cost is None or parent_cost + goal_distance < best_cost):
                certificate = PortalCertificate(goal_pose, goal_slack, goal_distance)
                goal_node = TubeRRTNode(goal_pose, cell, new_index, 0.0, certificate)
                goal_node.cost = parent_cost + self.edge_cost(new_node, goal_node, certificate)
                goal_index = self._store(goal_node)
                goal_indices.append(goal_index)
                self._goal_nodes.add(goal_index)
                if config.record_trace:
                    trace.append(TubeRRTTraceEvent(iteration, goal_pose, new_index, goal_pose, cell.radius, "goal",
                                                   goal_index, new_index, cell=cell, mode=mode))
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
        stats = {**self.cells.stats, **self._stats, **first_goal, "frontier_alive": int(len(self.frontier.live())),
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
