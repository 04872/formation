"""Region-guided SE(2) search over polyhedral cells (``polyhedral_cell``).

Each node stores ``(q_i, C_i)``.  Instead of sampling ``q_rand ~ SE(2)`` and extending the nearest
node, the search mostly expands the exposed frontier of the certified union ``U = cup_i C_i``:

1. every accepted cell contributes discrete candidates ``(i, u, dtheta)``: points on the edges of
   the translational polygons ``P_i(dtheta)`` of a few yaw slices, with extensibility
   ``l_i(u, dtheta)``, an uncovered-free-space estimate ``U_i`` (probes beyond the boundary that are
   collision free and outside ``U``) and the goal progress ``G_i``;
2. candidates covered by another cell are not on ``F_i = dC_i minus cup_{j != i} C_j`` and are
   dropped; the rest are sampled with weight ``exp(S / tau)``,
   ``S = alpha l / l_ref + beta U + gamma (G + 1) / 2``;
3. a candidate yields a few seeds ``q_i + (s l u, dtheta)``; each seed gets its cell, and the
   seed with the best ``J = w1 rho_new + w2 phi(rho_overlap) + w3 dd_goal`` among those whose cell
   overlaps the parent (and adds at least ``min_new_ratio`` new coverage) is inserted.

With probability ``uniform_probability`` an ordinary SE(2) sample (with goal bias) is steered from
its nearest node to the boundary of that node's cell instead.  Parents are then re-chosen among
overlapping neighbours and neighbours are rewired (RRT*), as in ``tube_rrt_chart``.  Consecutive
path nodes are joined by ``q_a -> q_p -> q_b`` with the portal ``q_p`` strictly inside both convex
cells, so the whole route is certified collision free with clearance ``>= d_s``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from formation.polyhedral_cell import PolyhedralCell, PolyhedralCellModel, wrap_array
from formation.tube_cell_first_order import PortalCertificate, interpolate_pose
from formation.tube_rrt import TubeRRTConfig, TubeRRTNode, TubeRRTResult, TubeRRTTraceEvent
from formation.types import FormationSpec, MapData, Pose2D, wrap_to_pi


@dataclass
class FrontierConfig:
    safety_distance: float = 0.02
    """``d_s``: clearance kept inside every cell on top of robot radius and safety margin."""
    active_range: float = 1.0
    max_active_per_robot: int = 3
    max_extent: float = 1.5
    yaw_limit: float = math.pi / 2.0
    guard_facets: int = 12
    cell_samples: int = 128
    yaw_slice_fractions: tuple[float, ...] = (0.0, -0.5, 0.5, -0.9, 0.9)
    frontier_spacing: float = 0.3
    probe_step: float = 0.25
    probe_count: int = 3
    uniform_probability: float = 0.15
    score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """``(alpha, beta, gamma)`` of ``S = alpha l + beta U + gamma G``."""
    score_temperature: float = 0.15
    length_ref: float = 1.0
    expand_weights: tuple[float, float, float] = (1.0, 0.5, 0.5)
    """``(w1, w2, w3)`` of ``J = w1 rho_new + w2 phi(rho_overlap) + w3 dd_goal``."""
    overlap_band: tuple[float, float] = (0.1, 0.5)
    min_new_ratio: float = 0.05
    uniform_min_new_ratio: float = 0.0
    """Uniform samples may add cells inside the union: they densify it so that rewiring can shorten the path."""
    step_scales: tuple[float, ...] = (0.8, 1.2, 1.6)
    max_candidate_failures: int = 3
    max_parent_candidates: int = 12

    def __post_init__(self) -> None:
        low, high = self.overlap_band
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("overlap_band must satisfy 0 <= rho_min < rho_max <= 1")
        if not 0.0 <= self.uniform_probability <= 1.0:
            raise ValueError("uniform_probability must lie in [0, 1]")
        if self.score_temperature <= 0.0 or not self.step_scales:
            raise ValueError("score_temperature must be positive and step_scales non-empty")


def overlap_preference(ratio: float, band: tuple[float, float]) -> float:
    """``phi``: 1 inside ``(rho_min, rho_max)``, falling linearly to 0 at ratio 0 and 1."""
    low, high = band
    if ratio < low:
        return ratio / low if low > 0.0 else 1.0
    if ratio > high:
        return (1.0 - ratio) / (1.0 - high) if high < 1.0 else 1.0
    return 1.0


class _Frontier:
    """Discrete candidates ``(i, u, dtheta)`` of all cells in growable arrays."""

    def __init__(self, probe_count: int) -> None:
        self.size = 0
        self.probe_count = probe_count
        self._allocate(1024)

    def _allocate(self, capacity: int) -> None:
        old = self.__dict__.copy() if self.size else None
        self.cell = np.zeros(capacity, dtype=int)
        self.points = np.zeros((capacity, 3))
        self.directions = np.zeros((capacity, 2))
        self.dtheta = np.zeros(capacity)
        self.length = np.zeros(capacity)
        self.progress = np.zeros(capacity)
        self.guard_edge = np.zeros(capacity, dtype=bool)
        self.probes = np.zeros((capacity, self.probe_count, 3))
        self.probe_free = np.zeros((capacity, self.probe_count), dtype=bool)
        self.probe_open = np.zeros((capacity, self.probe_count), dtype=bool)
        self.alive = np.zeros(capacity, dtype=bool)
        self.failures = np.zeros(capacity, dtype=int)
        self.score = np.zeros(capacity)
        if old is not None:
            for name in ("cell", "points", "directions", "dtheta", "length", "progress", "guard_edge", "probes",
                         "probe_free", "probe_open", "alive", "failures", "score"):
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
    """Frontier-guided RRT* over polyhedral SE(2) cells (``TubeRRTConfig.cell_model = "polyhedral"``)."""

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
                                         f.active_range, f.max_active_per_robot, f.max_extent, f.yaw_limit,
                                         f.guard_facets, f.cell_samples)
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
        """Append the half-spaces of ``cell`` in world form ``n^T c - kappa rho |dtheta| + (e - n^T c_0) >= 0``."""
        count = len(cell.offsets)
        if self._row_count + count > len(self._row_offset):
            size = max(2 * len(self._row_offset), self._row_count + count)
            self._row_normal = np.vstack((self._row_normal, np.zeros((size - len(self._row_normal), 2))))
            self._row_rate = np.concatenate((self._row_rate, np.zeros(size - len(self._row_rate))))
            self._row_offset = np.concatenate((self._row_offset, np.zeros(size - len(self._row_offset))))
        rows = slice(self._row_count, self._row_count + count)
        self._row_normal[rows] = cell.normals
        self._row_rate[rows] = cell.kappa * cell.rho
        self._row_offset[rows] = cell.offsets - cell.normals @ (cell.pose.x, cell.pose.y)
        self._row_start[index], self._row_size[index] = self._row_count, count
        self._row_count += count

    def _covered(self, points: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Whether each world configuration lies strictly inside at least one of ``cells``."""
        if not len(cells) or not len(points):
            return np.zeros(len(points), dtype=bool)
        sizes, starts = self._row_size[cells], self._row_start[cells]
        bounds = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        rows = np.repeat(starts - bounds, sizes) + np.arange(int(sizes.sum()))
        absphi = np.abs(wrap_array(points[None, :, 2] - self._yaw[cells][:, None]))
        values = (self._row_normal[rows] @ points[:, :2].T - self._row_rate[rows, None] * np.repeat(absphi, sizes, axis=0)
                  + self._row_offset[rows, None])
        slack = np.minimum(np.minimum.reduceat(values, bounds, axis=0),
                           self.formation_radius * (self._yaw_limit[cells][:, None] - absphi))
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
        uncovered = np.mean(store.probe_free[indices] & store.probe_open[indices], axis=1)
        score = (alpha * np.minimum(store.length[indices] / f.length_ref, 1.0) + beta * uncovered
                 + gamma * 0.5 * (store.progress[indices] + 1.0))
        store.score[indices] = score * 0.5 ** store.failures[indices]

    def _add_frontier(self, index: int) -> None:
        """Exposed frontier candidates of cell ``index`` on a few yaw slices."""
        f, cell = self.frontier_config, self._nodes[index].cell
        center = cell.pose
        points, directions, dthetas, lengths, guard_edges = [], [], [], [], []
        for fraction in f.yaw_slice_fractions:
            dtheta = fraction * 0.95 * cell.yaw_reach
            polygon, labels = self.cells.slice_polygon(cell, dtheta)
            if not len(polygon):
                continue
            local = polygon - (center.x, center.y)
            ends = np.roll(local, -1, axis=0)
            for start, end, label in zip(local, ends, labels.tolist()):
                edge = float(np.hypot(*(end - start)))
                pieces = max(1, int(round(edge / f.frontier_spacing)))
                for t in (np.arange(pieces) + 0.5) / pieces:
                    point = start + t * (end - start)
                    length = float(np.hypot(*point))
                    if length <= 1e-9:
                        continue
                    points.append(point)
                    directions.append(point / length)
                    dthetas.append(dtheta)
                    lengths.append(length)
                    guard_edges.append(label >= 0 and bool(cell.guard_rows[label]))
        if not points:
            return
        points, directions = np.asarray(points), np.asarray(directions)
        dthetas, lengths = np.asarray(dthetas), np.asarray(lengths)
        world = np.column_stack((center.x + points[:, 0], center.y + points[:, 1], wrap_array(center.yaw + dthetas)))
        reach = cell.guard + f.probe_count * f.probe_step
        others = self._nearby(center.x, center.y, center.yaw, reach, cell.yaw_limit, exclude=index)
        exposed = ~self._covered(world, others)
        if not np.any(exposed):
            return
        world, directions, dthetas, lengths = world[exposed], directions[exposed], dthetas[exposed], lengths[exposed]
        guard_edges = np.asarray(guard_edges)[exposed]
        steps = f.probe_step * np.arange(1, f.probe_count + 1)
        probes = np.repeat(world[:, None, :], f.probe_count, axis=1)
        probes[:, :, 0] += directions[:, 0:1] * steps[None, :]
        probes[:, :, 1] += directions[:, 1:2] * steps[None, :]
        flat = probes.reshape((-1, 3))
        free = (self.clearances(flat[:, 0], flat[:, 1], flat[:, 2]) > self.cells.safety_distance)
        near = self._nearby(center.x, center.y, center.yaw, reach, cell.yaw_limit)
        open_space = ~self._covered(flat, near)
        span = self.frontier.add(len(world))
        store = self.frontier
        store.cell[span], store.points[span], store.directions[span] = index, world, directions
        store.dtheta[span], store.length[span], store.guard_edge[span] = dthetas, lengths, guard_edges
        store.progress[span] = (self._goal_distance(center.x, center.y)
                                - self._goal_distance(world[:, 0], world[:, 1])) / lengths
        store.probes[span] = probes
        store.probe_free[span] = free.reshape((-1, f.probe_count))
        store.probe_open[span] = open_space.reshape((-1, f.probe_count))
        store.alive[span], store.failures[span] = True, 0
        self._score(np.arange(span.start, span.stop))
        self._stats["frontier_candidates"] += len(world)

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
            probes = store.probes[remaining].reshape((-1, 3))
            inside = self.cells.contains(cell, probes).reshape((len(remaining), -1))
            store.probe_open[remaining] &= ~inside
            self._score(remaining)

    def _pick_candidate(self) -> int | None:
        live = self.frontier.live()
        if not len(live):
            return None
        score = self.frontier.score[live]
        weights = np.exp((score - score.max()) / self.frontier_config.score_temperature)
        return int(live[self.rng.choice(len(live), p=weights / weights.sum())])

    # -- expansion --------------------------------------------------------------------------------------
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

    def _frontier_seeds(self, candidate: int) -> tuple[int, Pose2D, list[Pose2D]]:
        store = self.frontier
        parent = int(store.cell[candidate])
        center = self._nodes[parent].pose
        u, length = store.directions[candidate], store.length[candidate]
        yaw = wrap_to_pi(center.yaw + float(store.dtheta[candidate]))
        seeds = []
        for scale in self.frontier_config.step_scales:
            step = scale * float(self.rng.uniform(0.9, 1.1)) * length
            seeds.append(Pose2D(center.x + step * u[0], center.y + step * u[1], yaw))
        point = store.points[candidate]
        return parent, Pose2D(float(point[0]), float(point[1]), float(point[2])), seeds

    def _uniform_seeds(self) -> tuple[int, Pose2D, list[Pose2D]]:
        sample = self._sample_pose()
        parent = int(np.argmin(self._distances_to(sample)))
        near = self._nodes[parent]
        dc = (sample.x - near.pose.x, sample.y - near.pose.y)
        dphi = wrap_to_pi(sample.yaw - near.pose.yaw)
        reach = self.cells.ray_extent(near.cell, dc, dphi)
        seeds, seen = [], set()
        for scale in self.frontier_config.step_scales:
            t = min(1.0, scale * reach)
            if t <= 0.0 or t in seen:
                continue
            seen.add(t)
            seeds.append(Pose2D(near.pose.x + t * dc[0], near.pose.y + t * dc[1], wrap_to_pi(near.pose.yaw + t * dphi)))
        return parent, sample, seeds

    def _evaluate(self, parent: int, seeds: list[Pose2D],
                  min_new_ratio: float) -> tuple[dict | None, str, Pose2D, PolyhedralCell | None]:
        """Best seed by ``J_expand`` among those whose cell overlaps the parent and adds coverage."""
        f = self.frontier_config
        w1, w2, w3 = f.expand_weights
        parent_node = self._nodes[parent]
        parent_goal = float(self._goal_distance(parent_node.pose.x, parent_node.pose.y))
        best, status, last_pose, last_cell = None, "collision", seeds[0] if seeds else parent_node.pose, None
        rank = {"collision": 0, "no_overlap": 1, "redundant": 2}
        for pose in seeds:
            cell = self.cells.make_cell(pose)
            if not cell.valid:
                failure = "collision"
            else:
                certificate = self.cells.overlap(parent_node.cell, cell)
                if certificate is None:
                    failure = "no_overlap"
                else:
                    new_ratio = self._new_ratio(cell)
                    if new_ratio < min_new_ratio:
                        failure = "redundant"
                    else:
                        overlap_ratio = self._overlap_ratio(parent_node.cell, cell)
                        progress = (parent_goal - float(self._goal_distance(pose.x, pose.y))) / f.length_ref
                        score = (w1 * new_ratio + w2 * overlap_preference(overlap_ratio, f.overlap_band)
                                 + w3 * float(np.clip(progress, -1.0, 1.0)))
                        if best is None or score > best["score"]:
                            best = {"pose": pose, "cell": cell, "certificate": certificate, "score": score,
                                    "new_ratio": new_ratio, "overlap_ratio": overlap_ratio}
                        continue
            if rank[failure] >= rank[status]:
                status, last_pose, last_cell = failure, pose, cell
        return best, status, last_pose, last_cell

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
        self._x[index], self._y[index], self._yaw[index] = node.pose.x, node.pose.y, node.pose.yaw
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
        """Nodes whose cells can overlap ``cell`` (guard disks and yaw ranges intersect), nearest first."""
        pose = cell.pose
        distances = self._distances_to(pose)
        neighbors = self._nearby(pose.x, pose.y, pose.yaw, cell.guard, cell.yaw_limit)
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
        """Live exposed-frontier candidates (world points, owning cell, score, guard-edge flag)."""
        live = self.frontier.live()
        store = self.frontier
        return {"points": store.points[live].copy(), "cell": store.cell[live].copy(),
                "score": store.score[live].copy(), "guard_edge": store.guard_edge[live].copy(),
                "length": store.length[live].copy()}

    def _print_progress(self, iteration: int, best_goal_distance: float, best_cost: float | None) -> None:
        cost = "" if best_cost is None else f" best_cost={best_cost:.3f}"
        print(f"progress iteration={iteration}/{self.config.max_iterations} accepted_nodes={len(self._nodes)} "
              f"frontier={len(self.frontier.live())} best_goal_distance={best_goal_distance:.3f}{cost}", flush=True)

    def plan(self) -> TubeRRTResult:
        config, f = self.config, self.frontier_config
        self.rng = np.random.default_rng(config.seed)
        self.frontier = _Frontier(f.probe_count)
        self.expansion_log = []
        self._stats = {"frontier_iterations": 0, "uniform_iterations": 0, "frontier_nodes": 0, "uniform_nodes": 0,
                       "frontier_candidates": 0, "frontier_covered": 0, "frontier_dropped": 0,
                       "rejected_redundant": 0}
        self._nodes: list[TubeRRTNode] = []
        self._children: list[list[int]] = []
        capacity = 1024
        self._x, self._y, self._yaw, self._guard, self._yaw_limit, self._cost = (np.zeros(capacity) for _ in range(6))
        self._row_start, self._row_size = np.zeros(capacity, dtype=int), np.zeros(capacity, dtype=int)
        self._row_normal = np.zeros((16 * capacity, 2))
        self._row_rate, self._row_offset = np.zeros(16 * capacity), np.zeros(16 * capacity)
        self._row_count = 0
        start_cell = self.cells.make_cell(self.start)
        start_node = TubeRRTNode(self.start, start_cell, None, 0.0)
        if not start_cell.valid:
            return TubeRRTResult(False, tree_nodes=[start_node], failure_reason="start is in collision")
        start_cell.mode = "start"
        self._store(start_node)
        self._add_frontier(0)
        goal_indices: list[int] = []
        best_goal_distance = float(self._goal_distance(self.start.x, self.start.y))
        trace: list[TubeRRTTraceEvent] = []
        best_cost: float | None = None
        first_goal_iteration: int | None = None
        cost_history: list[tuple[int, float]] = []
        iteration = 0
        for iteration in range(1, config.max_iterations + 1):
            report = config.progress_interval > 0 and iteration % config.progress_interval == 0
            candidate = None if self.rng.random() < f.uniform_probability else self._pick_candidate()
            if candidate is None:
                mode = "uniform"
                parent, sample, seeds = self._uniform_seeds()
            else:
                mode = "frontier"
                parent, sample, seeds = self._frontier_seeds(candidate)
            self._stats[f"{mode}_iterations"] += 1
            min_new_ratio = f.min_new_ratio if mode == "frontier" else f.uniform_min_new_ratio
            best, status, last_pose, last_cell = self._evaluate(parent, seeds, min_new_ratio)
            if best is None:
                if status == "redundant":
                    self._stats["rejected_redundant"] += 1
                if candidate is not None:
                    self.frontier.failures[candidate] += 1
                    if self.frontier.failures[candidate] >= f.max_candidate_failures:
                        self.frontier.alive[candidate] = False
                        self._stats["frontier_dropped"] += 1
                    else:
                        self._score(np.array((candidate,)))
                if config.record_trace:
                    radius = last_cell.radius if last_cell is not None else 0.0
                    trace.append(TubeRRTTraceEvent(iteration, sample, parent, last_pose, radius, status,
                                                   attempts=len(seeds), cell=last_cell, mode=mode))
                if report:
                    self._print_progress(iteration, best_goal_distance, best_cost)
                continue

            pose, cell = best["pose"], best["cell"]
            cell.new_ratio, cell.overlap_ratio, cell.expand_score, cell.mode = (
                best["new_ratio"], best["overlap_ratio"], best["score"], mode)
            new_node = TubeRRTNode(pose, cell, None, 0.0)
            source = parent
            neighbors, distances = self._neighbors(cell)
            parent_certificate = best["certificate"]
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
            self.expansion_log.append({"iteration": iteration, "node": new_index, "mode": mode,
                                       "new_ratio": best["new_ratio"], "overlap_ratio": best["overlap_ratio"],
                                       "score": best["score"], "active_pairs": cell.active_count})
            event = TubeRRTTraceEvent(iteration, sample, source, pose, cell.radius, "added",
                                      new_index, parent, attempts=len(seeds), cell=cell, mode=mode)
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
            best_goal_distance = min(best_goal_distance, float(self._goal_distance(pose.x, pose.y)))

            if goal_indices:
                best_cost = min(self._nodes[i].cost for i in goal_indices)
                if best_cost < cost_history[-1][1]:
                    cost_history.append((iteration, best_cost))
            goal_distance = float(self._goal_distance(pose.x, pose.y))
            if goal_distance <= config.goal_connect_distance and (best_cost is None or parent_cost + goal_distance < best_cost):
                goal_pose = Pose2D(self.goal_xy[0], self.goal_xy[1], pose.yaw)
                goal_cell = self.cells.make_cell(goal_pose)
                goal_certificate = self.cells.overlap(cell, goal_cell)
                if goal_certificate is not None:
                    goal_node = TubeRRTNode(goal_pose, goal_cell, new_index, 0.0, goal_certificate)
                    goal_node.cost = parent_cost + self.edge_cost(new_node, goal_node, goal_certificate)
                    if best_cost is None or goal_node.cost < best_cost:
                        goal_cell.mode = "goal"
                        goal_index = self._store(goal_node)
                        goal_indices.append(goal_index)
                        self._cover_frontier(goal_index)
                        if config.record_trace:
                            trace.append(TubeRRTTraceEvent(iteration, goal_pose, new_index, goal_pose, goal_cell.radius,
                                                           "goal", goal_index, new_index, cell=goal_cell, mode=mode))
                        best_cost = goal_node.cost
                        cost_history.append((iteration, best_cost))
                        if first_goal_iteration is None:
                            first_goal_iteration = iteration
                            if config.progress_interval > 0:
                                print(f"goal connected iteration={iteration} accepted_nodes={len(self._nodes)}",
                                      flush=True)
                        if config.stop_on_first_goal:
                            break
            if report:
                self._print_progress(iteration, best_goal_distance, best_cost)

        nodes = self._nodes
        edges = [(i, node.parent) for i, node in enumerate(nodes) if node.parent is not None]
        stats = {**self.cells.stats, **self._stats, "frontier_alive": int(len(self.frontier.live()))}
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
