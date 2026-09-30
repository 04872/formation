"""Direction-aware SE(2) cells built from a few robot-obstacle proximity queries.

For a seed ``q_0 = (c_0, theta_0)`` robot ``i`` sits at ``p_i = c_0 + R(theta_0) s_i`` with
``||s_i|| = rho`` for every robot.  A proximity query of robot ``i`` against an obstacle component
``O_j`` (a circle or one of the four map walls) returns the guarded clearance ``d_ij`` (already
minus robot radius and safety margin) and the separating direction ``n_ij``.  Keeping robot ``i`` on
the free side of the supporting plane of ``O_j`` and using ``||(R(dtheta) - I) s_i|| =
2 rho sin(|dtheta| / 2) <= rho |dtheta|`` gives the directional constraint

    n_ij^T dc - rho |dtheta| >= -(d_ij - d_s).

Active pairs are chosen in two levels: a broadphase ``d_ij < d_active``, then only rows that are
real boundaries of the cell survive (of two rows with ``n_a^T n_b > cos eps`` only the tighter one is
kept, and rows that are redundant inside the guard are dropped).  All other pairs are covered by the
validity guard, kept as a cheap predicate rather than as extra half-spaces:

    g(q) = ||c - c_0|| + rho |theta - theta_0| - r_g <= 0,     r_g = d_inactive(q_0) - d_s,

since every robot moves by at most ``||dc|| + rho |dtheta|``.  The cell is

    C = {q : n_k^T dc - rho |dtheta| + e_k >= 0 for all k,  g(q) <= 0,  |dtheta| <= L},

and every configuration in it keeps clearance ``>= d_s``.  All rows share the unit coefficient of
``rho |dtheta|`` and the guard disk shrinks by the same amount, so the translational section is
``P(dtheta) = S_0 eroded by rho |dtheta|`` with ``S_0 = P(0)``: row redundancy decided at
``dtheta = 0`` holds for every yaw, and the yaw interval is exactly ``|dtheta| < R_in / rho``
(``R_in`` = in-radius of ``S_0``), capped by the chart range ``pi / 2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import clarabel
import numpy as np
import scipy.sparse as sp

from formation.tube_cell_first_order import PortalCertificate, interpolate_pose
from formation.types import MapData, Pose2D, wrap_to_pi

_SEGMENT_CANDIDATES = (1.0, 0.0, 0.5, 0.25, 0.75)
_INSIDE_TOL = 1e-12
_REDUNDANCY_VERTICES = 32


def halton(count: int, bases: tuple[int, ...] = (2, 3, 5)) -> np.ndarray:
    points = np.empty((count, len(bases)))
    for column, base in enumerate(bases):
        for index in range(count):
            fraction, value, n = 1.0, 0.0, index + 1
            while n > 0:
                fraction /= base
                value += fraction * (n % base)
                n //= base
            points[index, column] = value
    return points


def wrap_array(values: np.ndarray) -> np.ndarray:
    return np.remainder(values + math.pi, 2.0 * math.pi) - math.pi


def clip_polygon(vertices: list[tuple[float, float]], labels: list[int], normals: np.ndarray,
                 bounds: np.ndarray) -> tuple[list[tuple[float, float]], list[int]]:
    """Clip a convex polygon by ``n_k^T x + b_k >= 0``; edge ``j`` (vertex j -> j+1) keeps its row label."""
    for row, ((nx, ny), bound) in enumerate(zip(normals.tolist(), bounds.tolist())):
        values = [nx * x + ny * y + bound for x, y in vertices]
        if min(values) >= 0.0:
            continue
        clipped, clipped_labels = [], []
        count = len(vertices)
        for j in range(count):
            k = (j + 1) % count
            current, following = values[j], values[k]
            if current >= 0.0:
                clipped.append(vertices[j])
                clipped_labels.append(labels[j])
                if following < 0.0:
                    t = current / (current - following)
                    (x0, y0), (x1, y1) = vertices[j], vertices[k]
                    clipped.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
                    clipped_labels.append(row)
            elif following >= 0.0:
                t = current / (current - following)
                (x0, y0), (x1, y1) = vertices[j], vertices[k]
                clipped.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
                clipped_labels.append(labels[j])
        vertices, labels = clipped, clipped_labels
        if len(vertices) < 3:
            return [], []
    return vertices, labels


def regular_polygon(radius: float, count: int) -> list[tuple[float, float]]:
    angles = 2.0 * math.pi * np.arange(count) / count
    return [(radius * math.cos(a), radius * math.sin(a)) for a in angles]


@dataclass(eq=False)
class PolyhedralCell:
    pose: Pose2D
    clearance: float
    """Guarded clearance of the seed (minimum over robots and obstacle components)."""
    normals: np.ndarray
    offsets: np.ndarray
    """Rows ``n_k^T dc - rho |dtheta| + e_k >= 0`` of the directional cell ``C_dir``."""
    pairs: tuple[tuple[int, int], ...]
    """``(robot, component)`` of each row, in row order."""
    guard: float
    """``r_g`` of the validity predicate ``||dc|| + rho |dtheta| <= r_g``."""
    yaw_limit: float
    """``L = min(pi / 2, R_in / rho)``: the natural yaw half-width of the cell."""
    rho: float
    inradius: float = 0.0
    """``R_in``: in-radius of the section ``S_0 = P(0)`` (``C_dir`` and guard)."""
    center_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    """In-circle centre of ``S_0`` relative to ``c_0``; it lies in ``P(dtheta)`` for every valid yaw."""
    broadphase_pairs: int = 0
    outer_normals: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    outer_offsets: np.ndarray = field(default_factory=lambda: np.empty(0))
    outer_robots: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    """Unpruned rows of every pair with ``d - d_s < r_g + outer_reach`` (including inactive ones).

    Not part of the certificate: they tell which part of the guard boundary faces a known obstacle and keep
    seeds placed just beyond the cell out of known restricted regions."""
    samples: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    """Deterministic quasi-random world configurations ``(x, y, theta)`` inside the cell."""
    volume: float = 0.0
    """Volume in ``(x, y, rho * theta)`` [m^3], estimated from the quasi-random box samples."""
    new_ratio: float = math.nan
    overlap_ratio: float = math.nan
    mode: str = ""

    @property
    def radius(self) -> float:
        """Distance from the seed to the boundary of ``S_0``; positive iff the seed is strictly inside."""
        rows = float(np.min(self.offsets)) if len(self.offsets) else math.inf
        return min(rows, self.guard)

    @property
    def valid(self) -> bool:
        return self.radius > 0.0 and self.yaw_limit > 0.0

    @property
    def yaw_reach(self) -> float:
        """Largest ``|dtheta|`` whose translational section still contains ``dc = 0``."""
        return min(self.yaw_limit, self.radius / self.rho) if self.valid else 0.0

    @property
    def active_count(self) -> int:
        return len(self.offsets)


class PolyhedralCellModel:
    name = "polyhedral"

    def __init__(self, slots: np.ndarray, map_data: MapData, clearance_margin: float, safety_distance: float = 0.02,
                 active_range: float = 1.0, parallel_tolerance: float = math.radians(5.0), max_extent: float = 1.5,
                 yaw_chart: float = math.pi / 2.0, cell_samples: int = 128, outer_reach: float = 0.5) -> None:
        self.outer_reach = float(outer_reach)
        self.slots = np.asarray(slots, dtype=float)
        self.rho = float(np.max(np.linalg.norm(self.slots, axis=1)))
        if self.rho <= 0.0:
            raise ValueError("polyhedral cells need a formation with a positive rotation radius")
        if not 0.0 < yaw_chart <= math.pi / 2.0:
            raise ValueError("yaw_chart must lie in (0, pi/2]")
        if cell_samples < 1 or not 0.0 <= parallel_tolerance < math.pi / 2.0:
            raise ValueError("cell_samples >= 1 and parallel_tolerance in [0, pi/2) are required")
        self.safety_distance = float(safety_distance)
        self.active_range = float(active_range)
        self.parallel_cos = math.cos(parallel_tolerance)
        self.max_extent = float(max_extent)
        self.yaw_chart = float(yaw_chart)
        self.clearance_margin = float(clearance_margin)
        unsupported = [p.get("type") for p in map_data.obstacle_primitives if p.get("type") != "circle"]
        if unsupported:
            raise ValueError("polyhedral cells support only circle obstacle primitives")
        self.origin = np.asarray(map_data.origin_xy, dtype=float)
        self.upper = self.origin + (float(map_data.width_m), float(map_data.height_m))
        self.circle_centers = np.asarray([p["center_xy"] for p in map_data.obstacle_primitives],
                                         dtype=float).reshape((-1, 2))
        self.circle_radii = np.asarray([float(p["radius"]) for p in map_data.obstacle_primitives], dtype=float)
        self.wall_normals = np.array(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))
        self._unit_samples = halton(int(cell_samples))
        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False
        self.stats = {"cells_built": 0, "pose_queries": 0, "pair_queries": 0, "broadphase_pairs": 0,
                      "active_pairs": 0, "overlap_calls": 0, "quick_reject": 0, "quick_accept": 0, "lp_calls": 0,
                      "lp_reject": 0, "lp_accept": 0, "socp_calls": 0, "socp_accept": 0}

    @property
    def component_count(self) -> int:
        return 4 + len(self.circle_radii)

    def component_name(self, component: int) -> str:
        return ("wall x-", "wall x+", "wall y-", "wall y+")[component] if component < 4 else f"circle {component - 4}"

    # -- proximity queries ---------------------------------------------------------------------------
    def robot_positions(self, pose: Pose2D) -> np.ndarray:
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return np.column_stack((pose.x + c * self.slots[:, 0] - s * self.slots[:, 1],
                                pose.y + s * self.slots[:, 0] + c * self.slots[:, 1]))

    def proximity(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Guarded clearance ``d_ij`` (N, J) and unit separating directions ``n_ij`` (N, J, 2)."""
        points = np.asarray(points, dtype=float).reshape((-1, 2))
        count = len(points)
        walls = np.column_stack((points[:, 0] - self.origin[0], self.upper[0] - points[:, 0],
                                 points[:, 1] - self.origin[1], self.upper[1] - points[:, 1]))
        wall_normals = np.broadcast_to(self.wall_normals, (count, 4, 2))
        if not len(self.circle_radii):
            return walls - self.clearance_margin, np.array(wall_normals)
        offset = points[:, None, :] - self.circle_centers[None, :, :]
        distance = np.linalg.norm(offset, axis=2)
        normals = offset / np.maximum(distance, 1e-12)[:, :, None]
        clearance = np.hstack((walls, distance - self.circle_radii[None, :])) - self.clearance_margin
        return clearance, np.concatenate((wall_normals, normals), axis=1)

    def make_cell(self, pose: Pose2D) -> PolyhedralCell:
        self.stats["cells_built"] += 1
        self.stats["pose_queries"] += 1
        self.stats["pair_queries"] += len(self.slots) * self.component_count
        clearance, normals = self.proximity(self.robot_positions(pose))
        d_s = self.safety_distance
        broadphase = clearance < self.active_range
        inactive = clearance[~broadphase]
        guard = min(float(np.min(inactive)) - d_s if inactive.size else math.inf, self.max_extent)
        robots, components = np.nonzero(broadphase)
        rows, offsets = self._boundary_rows(normals[robots, components], clearance[robots, components] - d_s, guard)
        self.stats["broadphase_pairs"] += len(robots)
        self.stats["active_pairs"] += len(rows)
        outer = clearance - d_s < guard + self.outer_reach
        cell = PolyhedralCell(pose=pose, clearance=float(np.min(clearance)),
                              normals=normals[robots[rows], components[rows]].reshape((-1, 2)),
                              offsets=offsets, pairs=tuple(zip(robots[rows].tolist(), components[rows].tolist())),
                              guard=guard, yaw_limit=0.0, rho=self.rho, broadphase_pairs=len(robots),
                              outer_normals=normals[outer].reshape((-1, 2)), outer_offsets=clearance[outer] - d_s,
                              outer_robots=np.nonzero(outer)[0])
        if cell.radius > 0.0:
            cell.center_offset, cell.inradius = self._inscribed_circle(cell)
            cell.yaw_limit = min(self.yaw_chart, cell.inradius / self.rho)
            self._fill_samples(cell)
        return cell

    def _boundary_rows(self, normals: np.ndarray, offsets: np.ndarray, guard: float) -> tuple[np.ndarray, np.ndarray]:
        """Rows that bound ``S_0`` and their offsets.

        Of near-parallel rows only the tightest is kept; its offset is lowered to
        ``min(e_a, e_b - ||n_a - n_b|| r_g)`` so that inside the guard disk it still implies every dropped
        row (no change for exactly parallel rows).  Rows redundant inside the guard are then removed,
        which is exact for every yaw because each section is ``S_0`` eroded by ``rho |dtheta|``.
        """
        kept: list[int] = []
        adjusted: dict[int, float] = {}
        reach = guard if math.isfinite(guard) else 0.0
        for index in np.argsort(offsets, kind="stable").tolist():
            twin = next((other for other in kept if float(normals[index] @ normals[other]) > self.parallel_cos), None)
            if twin is None:
                kept.append(index)
                adjusted[index] = float(offsets[index])
            else:
                gap = float(np.linalg.norm(normals[index] - normals[twin])) * reach
                adjusted[twin] = min(adjusted[twin], float(offsets[index]) - gap)
        kept_offsets = np.asarray([adjusted[index] for index in kept])
        if not kept or guard <= 0.0 or kept_offsets.min() <= 0.0:
            return np.asarray(kept, dtype=int), kept_offsets
        # A circumscribed polygon of the guard disk only keeps rows cutting the thin gap: still exact.
        outer = regular_polygon(guard / math.cos(math.pi / _REDUNDANCY_VERTICES), _REDUNDANCY_VERTICES)
        _, labels = clip_polygon(outer, [-1] * len(outer), normals[kept], kept_offsets)
        boundary = sorted({label for label in labels if label >= 0})
        return np.asarray([kept[label] for label in boundary], dtype=int), kept_offsets[boundary]

    def _inscribed_circle(self, cell: PolyhedralCell) -> tuple[np.ndarray, float]:
        """``max r`` s.t. ``n_k^T x + e_k >= r`` and ``||x|| + r <= r_g`` (a 3-variable SOCP)."""
        if not len(cell.offsets):
            return np.zeros(2), cell.guard
        count = len(cell.offsets)
        matrix = np.zeros((count + 4, 3))
        matrix[:count, 0:2] = -cell.normals
        matrix[:count, 2] = 1.0
        matrix[count, 2] = -1.0
        matrix[count + 1, 2] = 1.0
        matrix[count + 2, 0] = -1.0
        matrix[count + 3, 1] = -1.0
        rhs = np.concatenate((cell.offsets, (0.0, cell.guard, 0.0, 0.0)))
        solver = clarabel.DefaultSolver(sp.csc_matrix((3, 3)), np.array((0.0, 0.0, -1.0)), sp.csc_matrix(matrix), rhs,
                                        [clarabel.NonnegativeConeT(count + 1), clarabel.SecondOrderConeT(3)],
                                        self._settings)
        solution = solver.solve()
        if str(solution.status) not in ("Solved", "AlmostSolved"):
            return np.zeros(2), cell.radius
        center = np.asarray(solution.x[:2], dtype=float)
        radius = min(float(np.min(cell.normals @ center + cell.offsets)), cell.guard - float(np.hypot(*center)))
        if radius < cell.radius:
            return np.zeros(2), cell.radius
        return center, radius

    def _fill_samples(self, cell: PolyhedralCell) -> None:
        g, limit = cell.guard, cell.yaw_limit
        unit = self._unit_samples
        dx, dy, dphi = (2.0 * unit[:, 0] - 1.0) * g, (2.0 * unit[:, 1] - 1.0) * g, (2.0 * unit[:, 2] - 1.0) * limit
        inside = self._local_slack(cell, dx, dy, dphi) > _INSIDE_TOL
        box = (2.0 * g) ** 2 * 2.0 * cell.rho * limit
        kept = int(np.count_nonzero(inside))
        cell.volume = box * max(kept, 0.5) / len(unit)
        if kept == 0:
            dx, dy, dphi, inside = np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1, dtype=bool)
        cell.samples = np.column_stack((cell.pose.x + dx[inside], cell.pose.y + dy[inside],
                                        wrap_array(cell.pose.yaw + dphi[inside])))

    # -- membership ------------------------------------------------------------------------------------
    @staticmethod
    def _local_slack(cell: PolyhedralCell, dx: np.ndarray, dy: np.ndarray, dphi: np.ndarray) -> np.ndarray:
        """Margin [m] of the directional rows, the guard predicate and the yaw range (positive = inside)."""
        turn = cell.rho * np.abs(dphi)
        slack = np.minimum(cell.guard - np.hypot(dx, dy) - turn, cell.rho * cell.yaw_limit - turn)
        if len(cell.offsets):
            rows = cell.normals[:, 0:1] * dx[None, :] + cell.normals[:, 1:2] * dy[None, :] + cell.offsets[:, None]
            slack = np.minimum(slack, rows.min(axis=0) - turn)
        return slack

    def slack_many(self, cell: PolyhedralCell, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        return self._local_slack(cell, points[:, 0] - cell.pose.x, points[:, 1] - cell.pose.y,
                                 wrap_array(points[:, 2] - cell.pose.yaw))

    def slack(self, cell: PolyhedralCell, pose: Pose2D) -> float:
        return float(self.slack_many(cell, np.array(((pose.x, pose.y, pose.yaw),)))[0])

    def contains(self, cell: PolyhedralCell, points: np.ndarray) -> np.ndarray:
        return self.slack_many(cell, points) > _INSIDE_TOL

    @staticmethod
    def directional_slack(cell: PolyhedralCell, points: np.ndarray, outer: bool = True) -> np.ndarray:
        """Margin [m] of obstacle rows alone (no guard / yaw range), ``inf`` without rows.

        ``outer=True`` uses the unpruned rows of all pairs near the guard (``outer_normals``), else the
        certificate rows of ``C_dir``.  Each row is a supporting half-space of a convex obstacle, so it bounds
        the clearance of its pair everywhere, not only inside the guard: a pose violating it heads into a
        known obstacle-limited region.
        """
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        normals, offsets = (cell.outer_normals, cell.outer_offsets) if outer else (cell.normals, cell.offsets)
        if not len(offsets):
            return np.full(len(points), math.inf)
        turn = cell.rho * np.abs(wrap_array(points[:, 2] - cell.pose.yaw))
        rows = normals @ (points[:, :2] - (cell.pose.x, cell.pose.y)).T + offsets[:, None]
        return rows.min(axis=0) - turn

    def rotation_margin(self, cell: PolyhedralCell, points: np.ndarray) -> np.ndarray:
        """Exact half-space margin ``min_k n_k^T (dc + (R(dtheta) - I) R_0 s_{i_k}) + e_k`` over the outer rows.

        Unlike the rows' ``-rho |dtheta|`` bound it tells which rotation sense relaxes a facet; it is still a
        global lower bound on ``clearance - d_s`` of each listed pair (``inf`` without outer rows).
        """
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        if not len(cell.outer_offsets):
            return np.full(len(points), math.inf)
        c0, s0 = math.cos(cell.pose.yaw), math.sin(cell.pose.yaw)
        arms = self.slots[cell.outer_robots] @ np.array(((c0, s0), (-s0, c0)))
        phi = wrap_array(points[:, 2] - cell.pose.yaw)
        cos, sin = np.cos(phi)[:, None], np.sin(phi)[:, None]
        shift_x = (cos - 1.0) * arms[None, :, 0] - sin * arms[None, :, 1]
        shift_y = sin * arms[None, :, 0] + (cos - 1.0) * arms[None, :, 1]
        dc = points[:, :2] - (cell.pose.x, cell.pose.y)
        values = (cell.outer_normals[None, :, 0] * (dc[:, 0:1] + shift_x)
                  + cell.outer_normals[None, :, 1] * (dc[:, 1:2] + shift_y) + cell.outer_offsets[None, :])
        return values.min(axis=1)

    @staticmethod
    def frontier_pieces(cell: PolyhedralCell, max_arc: float = math.pi / 4.0) -> list[dict]:
        """Frontier pieces of the section ``S_0``: guard arcs (split into chunks of at most ``max_arc``) and
        obstacle facets.  Each piece has a representative boundary point (chart ``dc``), the outward normal,
        a unit tangent, its length and, for facets, the row index.
        """
        polygon, labels = PolyhedralCellModel.slice_polygon(cell, 0.0)
        if not len(polygon):
            return []
        local = polygon - (cell.pose.x, cell.pose.y)
        pieces: list[dict] = []
        count = len(labels)
        for k in range(count):
            label = int(labels[k])
            if label < 0:
                continue
            a, b = local[k], local[(k + 1) % count]
            length = float(np.linalg.norm(b - a))
            if length <= 1e-9:
                continue
            normal = cell.normals[label]
            pieces.append({"kind": "obstacle", "point": 0.5 * (a + b), "outward": -normal,
                           "tangent": np.array((-normal[1], normal[0])), "length": length, "row": label})
        guard = labels < 0
        if not np.any(guard):
            return pieces
        start = int(np.argmin(guard)) if not np.all(guard) else 0
        runs, current = [], []
        for step in range(count):
            k = (start + step) % count
            if guard[k]:
                current.append(k)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        for run in runs:
            first, last = local[run[0]], local[(run[-1] + 1) % count]
            begin = math.atan2(first[1], first[0])
            sweep = (math.atan2(last[1], last[0]) - begin) % (2.0 * math.pi)
            if len(run) == count:
                sweep = 2.0 * math.pi
            chunks = max(1, int(math.ceil(sweep / max_arc - 1e-9)))
            for j in range(chunks):
                angle = begin + (j + 0.5) * sweep / chunks
                radial = np.array((math.cos(angle), math.sin(angle)))
                pieces.append({"kind": "guard", "point": cell.guard * radial, "outward": radial,
                               "tangent": np.array((-radial[1], radial[0])), "length": cell.guard * sweep / chunks,
                               "row": -1})
        return pieces

    # -- directional extensibility -------------------------------------------------------------------
    def translational_extent(self, cell: PolyhedralCell, directions: np.ndarray,
                             dtheta: float) -> tuple[np.ndarray, np.ndarray]:
        """``l(u, dtheta) = max{t >= 0 : (t u, dtheta) in C}`` for unit ``u`` (M, 2), and whether the guard binds."""
        directions = np.asarray(directions, dtype=float).reshape((-1, 2))
        turn = cell.rho * abs(dtheta)
        reach = cell.guard - turn
        if abs(dtheta) >= cell.yaw_limit or reach <= 0.0 or (len(cell.offsets) and np.min(cell.offsets) <= turn):
            return np.zeros(len(directions)), np.zeros(len(directions), dtype=bool)
        lengths = np.full(len(directions), reach)
        if len(cell.offsets):
            rate = -(cell.normals @ directions.T)
            with np.errstate(divide="ignore"):
                limits = np.where(rate > 1e-15, (cell.offsets - turn)[:, None] / np.maximum(rate, 1e-300), math.inf)
            lengths = np.minimum(lengths, limits.min(axis=0))
        return lengths, lengths >= reach - 1e-12

    def ray_extent(self, cell: PolyhedralCell, dc: tuple[float, float], dphi: float) -> float:
        """``max{t >= 0 : t (dc, dphi) in C}`` along a chart ray."""
        dc = np.asarray(dc, dtype=float)
        turn = cell.rho * abs(dphi)
        limits = [cell.guard / max(float(np.hypot(*dc)) + turn, 1e-300)]
        if abs(dphi) > 0.0:
            limits.append(cell.yaw_limit / abs(dphi))
        if len(cell.offsets):
            rate = turn - cell.normals @ dc
            limits.append(float(np.min(np.where(rate > 1e-15, cell.offsets / np.maximum(rate, 1e-300), math.inf))))
        return max(0.0, min(limits))

    @staticmethod
    def boundary_radii(cell: PolyhedralCell, angles: np.ndarray, dthetas: np.ndarray) -> np.ndarray:
        """Distance from the in-circle centre to the boundary of ``P(dtheta)`` along each angle, shape (T, A)."""
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        center = cell.center_offset
        turn = cell.rho * np.abs(np.asarray(dthetas, dtype=float))[:, None]
        along = directions @ center
        reach = np.maximum(cell.guard - turn, 0.0)
        radii = -along[None, :] + np.sqrt(np.maximum(along[None, :] ** 2 - center @ center + reach ** 2, 0.0))
        if len(cell.offsets):
            rate = -(cell.normals @ directions.T)
            margin = (cell.normals @ center + cell.offsets)[None, :, None] - turn[:, :, None]
            with np.errstate(divide="ignore"):
                limits = np.where(rate[None] > 1e-15, margin / np.maximum(rate[None], 1e-300), math.inf)
            radii = np.minimum(radii, limits.min(axis=1))
        return np.maximum(radii, 0.0)

    @staticmethod
    def slice_polygon(cell: PolyhedralCell, dtheta: float, guard_vertices: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Section ``P(dtheta)`` as world vertices (V, 2) and edge labels (row index, -1 on the guard arc).

        The guard arc is drawn by an inscribed ``guard_vertices``-gon; used for frontier display only.
        """
        turn = cell.rho * abs(dtheta)
        if abs(dtheta) >= cell.yaw_limit or cell.guard <= turn:
            return np.empty((0, 2)), np.empty(0, dtype=int)
        disk = regular_polygon(cell.guard - turn, guard_vertices)
        vertices, labels = clip_polygon(disk, [-1] * guard_vertices, cell.normals.reshape((-1, 2)),
                                        cell.offsets - turn)
        if not vertices:
            return np.empty((0, 2)), np.empty(0, dtype=int)
        return np.asarray(vertices) + (cell.pose.x, cell.pose.y), np.asarray(labels, dtype=int)

    # -- overlap ---------------------------------------------------------------------------------------
    def distance(self, first: Pose2D, second: Pose2D) -> float:
        return math.hypot(second.x - first.x, second.y - first.y) + self.rho * abs(wrap_to_pi(second.yaw - first.yaw))

    def _certificate(self, first: PolyhedralCell, second: PolyhedralCell, portal: Pose2D,
                     slack: float) -> PortalCertificate:
        route = self.distance(first.pose, portal) + self.distance(portal, second.pose)
        return PortalCertificate(portal, float(slack), float(route))

    def overlap(self, first: PolyhedralCell, second: PolyhedralCell) -> PortalCertificate | None:
        """A portal strictly inside both cells, or ``None``.  ``slack`` is the absolute margin [m].

        Order: guard / yaw quick reject, candidate points on the node segment, an LP over the
        directional rows only (guards enter as bounding boxes), the guard predicates at the LP point,
        and only then an SOCP with both guards.
        """
        self.stats["overlap_calls"] += 1
        if not first.valid or not second.valid:
            return None
        a, b = first.pose, second.pose
        dx, dy, delta = b.x - a.x, b.y - a.y, wrap_to_pi(b.yaw - a.yaw)
        if abs(delta) >= first.yaw_limit + second.yaw_limit or math.hypot(dx, dy) >= first.guard + second.guard:
            self.stats["quick_reject"] += 1
            return None
        ts = np.asarray(_SEGMENT_CANDIDATES)
        candidates = np.column_stack((a.x + ts * dx, a.y + ts * dy, a.yaw + ts * delta))
        slack = np.minimum(self.slack_many(first, candidates), self.slack_many(second, candidates))
        best = int(np.argmax(slack))
        if slack[best] > _INSIDE_TOL:
            self.stats["quick_accept"] += 1
            return self._certificate(first, second, interpolate_pose(a, b, float(ts[best])), float(slack[best]))
        return self._solve_overlap(first, second, dx, dy, delta)

    def _overlap_program(self, first: PolyhedralCell, second: PolyhedralCell, dx: float, dy: float,
                         delta: float, with_guards: bool) -> np.ndarray | None:
        """``max t`` over x = (u_x, u_y, phi, w_a, w_b, t) in the chart of ``first``; returns x or ``None``."""
        rows, rhs = [], []
        for cell, w_column, shift in ((first, 3, (0.0, 0.0)), (second, 4, (dx, dy))):
            block = np.zeros((len(cell.offsets), 6))
            block[:, 0:2] = -cell.normals
            block[:, w_column] = cell.rho
            block[:, 5] = 1.0
            rows.append(block)
            rhs.append(cell.offsets - cell.normals @ np.asarray(shift))
        extra = np.zeros((15, 6))
        extra[0, [2, 3]] = (1.0, -1.0)
        extra[1, [2, 3]] = (-1.0, -1.0)
        extra[2, [2, 4]] = (1.0, -1.0)
        extra[3, [2, 4]] = (-1.0, -1.0)
        extra[4, [3, 5]] = (first.rho, 1.0)
        extra[5, [4, 5]] = (second.rho, 1.0)
        extra[6, 5] = 1.0
        for k, (column, sign) in enumerate(((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0))):
            extra[7 + k, column] = sign
            extra[11 + k, column] = sign
        rows.append(extra)
        ga, gb = first.guard, second.guard
        rhs.append(np.array((0.0, 0.0, delta, -delta, first.rho * first.yaw_limit, second.rho * second.yaw_limit,
                             1.0, ga, ga, ga, ga, gb + dx, gb - dx, gb + dy, gb - dy)))
        cones = [clarabel.NonnegativeConeT(sum(len(r) for r in rows))]
        if with_guards:
            for w_column, gx, gy, g in ((3, 0.0, 0.0, ga), (4, dx, dy, gb)):
                cone = np.zeros((3, 6))
                cone[0, [w_column, 5]] = (first.rho, 1.0)
                cone[1, 0], cone[2, 1] = -1.0, -1.0
                rows.append(cone)
                rhs.append(np.array((g, -gx, -gy)))
                cones.append(clarabel.SecondOrderConeT(3))
        matrix = np.vstack(rows)
        solver = clarabel.DefaultSolver(sp.csc_matrix((6, 6)), np.array((0.0, 0.0, 0.0, 0.0, 0.0, -1.0)),
                                        sp.csc_matrix(matrix), np.concatenate(rhs), cones, self._settings)
        solution = solver.solve()
        if str(solution.status) not in ("Solved", "AlmostSolved") or -solution.obj_val <= 1e-9:
            return None
        return np.asarray(solution.x)

    def _solve_overlap(self, first: PolyhedralCell, second: PolyhedralCell, dx: float, dy: float,
                       delta: float) -> PortalCertificate | None:
        a = first.pose
        self.stats["lp_calls"] += 1
        x = self._overlap_program(first, second, dx, dy, delta, with_guards=False)
        if x is None:
            self.stats["lp_reject"] += 1
            return None
        portal = Pose2D(a.x + x[0], a.y + x[1], wrap_to_pi(a.yaw + x[2]))
        slack = min(self.slack(first, portal), self.slack(second, portal))
        if slack > _INSIDE_TOL:
            self.stats["lp_accept"] += 1
            return self._certificate(first, second, portal, slack)
        self.stats["socp_calls"] += 1
        x = self._overlap_program(first, second, dx, dy, delta, with_guards=True)
        if x is None:
            return None
        portal = Pose2D(a.x + x[0], a.y + x[1], wrap_to_pi(a.yaw + x[2]))
        slack = min(self.slack(first, portal), self.slack(second, portal))
        if slack <= _INSIDE_TOL:
            return None
        self.stats["socp_accept"] += 1
        return self._certificate(first, second, portal, slack)
