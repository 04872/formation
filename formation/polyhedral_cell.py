"""Direction-aware polyhedral SE(2) cells built from a few robot-obstacle proximity queries.

For a seed ``q_0 = (c_0, theta_0)`` robot ``i`` sits at ``p_i = c_0 + R(theta_0) s_i`` with
``||s_i|| = rho`` for every robot.  A proximity query of robot ``i`` against an obstacle component
``O_j`` (a circle or one of the four map walls) returns the guarded clearance ``d_ij`` (already
minus robot radius and safety margin) and the separating direction ``n_ij``.  Keeping robot ``i`` on
the free side of the supporting plane of ``O_j`` and using ``||(R(dtheta) - I) s_i|| =
2 rho sin(|dtheta| / 2) <= rho |dtheta|`` gives, for every active pair,

    n_ij^T dc - rho |dtheta| >= -(d_ij - d_s).

Pairs outside the active set are handled by the validity guard ``||dc|| + rho |dtheta| <= G`` with
``G = d_inactive(q_0) - d_s``: every robot moves by at most ``||dc|| + rho |dtheta|``, so the
clearance to any inactive obstacle stays ``>= d_s``.  The guard disk is replaced by its inscribed
regular M-gon, so the whole cell is a polyhedron in the chart ``(dc, dtheta)``:

    C(q_0) = {(dc, dtheta) : n_k^T dc - kappa_k rho |dtheta| + e_k >= 0 for all k,  |dtheta| <= L},

with ``kappa_k = 1, e_k = d_ij - d_s`` for obstacle rows and ``kappa_k = cos(pi / M),
e_k = G cos(pi / M)`` for guard facets.  ``|dtheta|`` is convex, so ``C`` is convex; ``L <= pi / 2``
keeps it inside one yaw chart.  Every configuration in ``C`` is collision free with clearance
``>= d_s``.  At fixed ``dtheta`` the translational section is the polygon ``P(dtheta)``.
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


@dataclass(eq=False)
class PolyhedralCell:
    pose: Pose2D
    clearance: float
    """Guarded clearance of the seed (minimum over robots and obstacle components)."""
    normals: np.ndarray
    kappa: np.ndarray
    offsets: np.ndarray
    guard_rows: np.ndarray
    """``True`` for validity-guard facets, ``False`` for active robot-obstacle planes."""
    pairs: tuple[tuple[int, int], ...]
    """``(robot, component)`` of each obstacle row, in row order."""
    guard: float
    yaw_limit: float
    rho: float
    samples: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    """Deterministic quasi-random world configurations ``(x, y, theta)`` inside the cell."""
    volume: float = 0.0
    """Volume in ``(x, y, rho * theta)`` [m^3], estimated from the quasi-random box samples."""
    new_ratio: float = math.nan
    overlap_ratio: float = math.nan
    expand_score: float = math.nan
    mode: str = ""

    @property
    def radius(self) -> float:
        """Translational in-radius at ``dtheta = 0``; positive iff the seed is strictly inside."""
        return float(np.min(self.offsets)) if len(self.offsets) else 0.0

    @property
    def valid(self) -> bool:
        return self.radius > 0.0 and self.yaw_limit > 0.0

    @property
    def yaw_reach(self) -> float:
        """Largest ``|dtheta|`` whose translational section still contains ``dc = 0``."""
        if not self.valid:
            return 0.0
        return min(self.yaw_limit, float(np.min(self.offsets / (self.kappa * self.rho))))

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(~self.guard_rows))


class PolyhedralCellModel:
    name = "polyhedral"

    def __init__(self, slots: np.ndarray, map_data: MapData, clearance_margin: float, safety_distance: float = 0.02,
                 active_range: float = 1.0, max_active_per_robot: int = 3, max_extent: float = 1.5,
                 yaw_limit: float = math.pi / 2.0, guard_facets: int = 12, cell_samples: int = 128) -> None:
        self.slots = np.asarray(slots, dtype=float)
        norms = np.linalg.norm(self.slots, axis=1)
        self.rho = float(np.max(norms))
        if self.rho <= 0.0:
            raise ValueError("polyhedral cells need a formation with a positive rotation radius")
        if not 0.0 < yaw_limit <= math.pi / 2.0:
            raise ValueError("yaw_limit must lie in (0, pi/2]")
        if guard_facets < 3 or max_active_per_robot < 1 or cell_samples < 1:
            raise ValueError("guard_facets >= 3, max_active_per_robot >= 1 and cell_samples >= 1 are required")
        self.safety_distance = float(safety_distance)
        self.active_range = float(active_range)
        self.max_active_per_robot = int(max_active_per_robot)
        self.max_extent = float(max_extent)
        self.yaw_limit = float(yaw_limit)
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
        angles = 2.0 * math.pi * (np.arange(guard_facets) + 0.5) / guard_facets
        self.guard_normals = -np.column_stack((np.cos(angles), np.sin(angles)))
        self.guard_kappa = math.cos(math.pi / guard_facets)
        self._unit_samples = halton(int(cell_samples))
        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False
        self._objective = np.zeros(6)
        self._objective[5] = -1.0
        self._hessian = sp.csc_matrix((6, 6))
        self.stats = {"cells_built": 0, "active_pairs": 0, "overlap_calls": 0, "quick_reject": 0,
                      "quick_accept": 0, "lp_calls": 0, "lp_accept": 0}

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
        clearance, normals = self.proximity(self.robot_positions(pose))
        d_s = self.safety_distance
        active = np.zeros(clearance.shape, dtype=bool)
        for robot, row in enumerate(clearance):
            order = np.argsort(row, kind="stable")[:self.max_active_per_robot]
            active[robot, order[row[order] <= self.active_range]] = True
        inactive = clearance[~active]
        guard = min(float(np.min(inactive)) - d_s if inactive.size else math.inf, self.max_extent)
        robots, components = np.nonzero(active)
        self.stats["active_pairs"] += len(robots)
        obstacle_normals = normals[robots, components]
        kappa_g = self.guard_kappa
        guard_count = len(self.guard_normals)
        cell = PolyhedralCell(
            pose=pose,
            clearance=float(np.min(clearance)),
            normals=np.vstack((obstacle_normals, self.guard_normals)),
            kappa=np.concatenate((np.ones(len(robots)), np.full(guard_count, kappa_g))),
            offsets=np.concatenate((clearance[robots, components] - d_s, np.full(guard_count, guard * kappa_g))),
            guard_rows=np.concatenate((np.zeros(len(robots), dtype=bool), np.ones(guard_count, dtype=bool))),
            pairs=tuple(zip(robots.tolist(), components.tolist())),
            guard=guard,
            yaw_limit=min(self.yaw_limit, guard / self.rho) if guard > 0.0 else 0.0,
            rho=self.rho,
        )
        if cell.valid:
            self._fill_samples(cell)
        return cell

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
        """``min_k (n_k^T dc - kappa_k rho |dphi| + e_k)`` together with ``rho (L - |dphi|)`` [m]."""
        absphi = np.abs(dphi)
        values = (cell.normals[:, 0:1] * dx[None, :] + cell.normals[:, 1:2] * dy[None, :]
                  - (cell.kappa * cell.rho)[:, None] * absphi[None, :] + cell.offsets[:, None])
        return np.minimum(values.min(axis=0), cell.rho * (cell.yaw_limit - absphi))

    def slack_many(self, cell: PolyhedralCell, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        return self._local_slack(cell, points[:, 0] - cell.pose.x, points[:, 1] - cell.pose.y,
                                 wrap_array(points[:, 2] - cell.pose.yaw))

    def slack(self, cell: PolyhedralCell, pose: Pose2D) -> float:
        return float(self.slack_many(cell, np.array(((pose.x, pose.y, pose.yaw),)))[0])

    def contains(self, cell: PolyhedralCell, points: np.ndarray) -> np.ndarray:
        return self.slack_many(cell, points) > _INSIDE_TOL

    # -- directional extensibility -------------------------------------------------------------------
    def translational_extent(self, cell: PolyhedralCell, directions: np.ndarray, dtheta: float) -> np.ndarray:
        """``l(u, dtheta) = max{t >= 0 : (t u, dtheta) in C}`` for unit directions ``u`` (M, 2)."""
        directions = np.asarray(directions, dtype=float).reshape((-1, 2))
        bound = cell.offsets - cell.kappa * cell.rho * abs(dtheta)
        if abs(dtheta) >= cell.yaw_limit or np.any(bound <= 0.0):
            return np.zeros(len(directions))
        rate = -(cell.normals @ directions.T)
        with np.errstate(divide="ignore"):
            limits = np.where(rate > 1e-15, bound[:, None] / np.maximum(rate, 1e-300), math.inf)
        return limits.min(axis=0)

    def ray_extent(self, cell: PolyhedralCell, dc: tuple[float, float], dphi: float) -> float:
        """``max{t >= 0 : t (dc, dphi) in C}`` along a chart ray."""
        rate = cell.kappa * cell.rho * abs(dphi) - cell.normals @ np.asarray(dc, dtype=float)
        limits = [float(np.min(np.where(rate > 1e-15, cell.offsets / np.maximum(rate, 1e-300), math.inf)))]
        if abs(dphi) > 0.0:
            limits.append(cell.yaw_limit / abs(dphi))
        return max(0.0, min(limits))

    @staticmethod
    def slice_polygon(cell: PolyhedralCell, dtheta: float) -> tuple[np.ndarray, np.ndarray]:
        """Translational section ``P(dtheta)`` as world vertices (V, 2) and the row index of each edge.

        Edge ``j`` runs from vertex ``j`` to ``j + 1`` and lies on row ``labels[j]``.  Empty when
        ``|dtheta| >= L`` or the half-planes do not intersect.
        """
        if abs(dtheta) >= cell.yaw_limit:
            return np.empty((0, 2)), np.empty(0, dtype=int)
        half = cell.guard + 1.0
        vertices = [(-half, -half), (half, -half), (half, half), (-half, half)]
        labels = [-1, -1, -1, -1]
        bounds = cell.offsets - cell.kappa * cell.rho * abs(dtheta)
        for row, ((nx, ny), bound) in enumerate(zip(cell.normals.tolist(), bounds.tolist())):
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
                return np.empty((0, 2)), np.empty(0, dtype=int)
        polygon = np.asarray(vertices) + (cell.pose.x, cell.pose.y)
        return polygon, np.asarray(labels, dtype=int)

    # -- overlap ---------------------------------------------------------------------------------------
    def distance(self, first: Pose2D, second: Pose2D) -> float:
        return math.hypot(second.x - first.x, second.y - first.y) + self.rho * abs(wrap_to_pi(second.yaw - first.yaw))

    def _certificate(self, first: PolyhedralCell, second: PolyhedralCell, portal: Pose2D,
                     slack: float) -> PortalCertificate:
        route = self.distance(first.pose, portal) + self.distance(portal, second.pose)
        return PortalCertificate(portal, float(slack), float(route))

    def overlap(self, first: PolyhedralCell, second: PolyhedralCell) -> PortalCertificate | None:
        """A portal strictly inside both cells, or ``None``.  ``slack`` is the absolute margin [m]."""
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

    def _solve_overlap(self, first: PolyhedralCell, second: PolyhedralCell, dx: float, dy: float,
                       delta: float) -> PortalCertificate | None:
        """``max t`` over x = (u_x, u_y, phi, w_a, w_b, t) in the chart of ``first`` (an LP)."""
        self.stats["lp_calls"] += 1
        rows, rhs = [], []
        for cell, w_column, shift in ((first, 3, (0.0, 0.0)), (second, 4, (dx, dy))):
            block = np.zeros((len(cell.offsets), 6))
            block[:, 0:2] = -cell.normals
            block[:, w_column] = cell.kappa * cell.rho
            block[:, 5] = 1.0
            rows.append(block)
            rhs.append(cell.offsets - cell.normals @ np.asarray(shift))
        extra = np.zeros((7, 6))
        extra[0, [2, 3]] = (1.0, -1.0)
        extra[1, [2, 3]] = (-1.0, -1.0)
        extra[2, [2, 4]] = (1.0, -1.0)
        extra[3, [2, 4]] = (-1.0, -1.0)
        extra[4, [3, 5]] = (first.rho, 1.0)
        extra[5, [4, 5]] = (second.rho, 1.0)
        extra[6, 5] = 1.0
        rows.append(extra)
        rhs.append(np.array((0.0, 0.0, delta, -delta, first.rho * first.yaw_limit, second.rho * second.yaw_limit,
                             1.0)))
        matrix = np.vstack(rows)
        solver = clarabel.DefaultSolver(self._hessian, self._objective, sp.csc_matrix(matrix),
                                        np.concatenate(rhs), [clarabel.NonnegativeConeT(len(matrix))], self._settings)
        solution = solver.solve()
        if str(solution.status) not in ("Solved", "AlmostSolved") or -solution.obj_val <= 1e-9:
            return None
        a = first.pose
        portal = Pose2D(a.x + solution.x[0], a.y + solution.x[1], wrap_to_pi(a.yaw + solution.x[2]))
        slack = min(self.slack(first, portal), self.slack(second, portal))
        if slack <= _INSIDE_TOL:
            return None
        self.stats["lp_accept"] += 1
        return self._certificate(first, second, portal, slack)
