"""Continuous region expansion under two coupling levels (method/对比.md, method/方案b.md).

Shared core, identical for both planners and driven by the same random stream::

    q_rand -> C_i (region-gap rule) -> x* = argmin_{x in C_i} d_W(x, q_rand)
           -> q_new = Steer(x*, q_rand; eta) -> C_new = C(q_new)

``C_new`` is accepted when ``C_i`` and ``C_new`` share a point with slack ``>= overlap_margin`` in both;
otherwise ``eta`` is halved.  Region selection, projection, steering and cell construction never read
path state, so with one seed both planners build exactly the same regions.  They differ only in what
they maintain on top:

* ``SingleEntryRRTStar``: nodes ``(C_i, e_i, J_i, p_i)``; continuous ChooseParent / Rewire on
  ``C_j ∩ C_new`` with lazy, version-triggered re-attachment of descendants.
* ``ProxyRRTStar``: nodes ``(C_i, q_i, J~_i, p_i)``; ChooseParent / Rewire on the proxy cost
  ``J~_j + d_W(q_j, q_new)`` over regions that overlap at the margin; no path state during search.

Both region chains (parent pointers from the best goal region) are scored by the same optimiser ``min sum ||p_{k+1} - p_k||`` over
``z_k in C_k ∩ C_{k+1}``.  Points live in the lifted chart ``z = (x, y, rho * theta)``, where ``d_W`` is
Euclidean and every cell is convex.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import clarabel
import numpy as np
import scipy.sparse as sp
from formation import region_qp
from formation.polyhedral_cell import NormalTurnCellModel, PolyhedralCell, PolyhedralCellModel, _csc
from formation.tube_rrt_region import _row_separates
from formation.types import MapData, Pose2D, wrap_to_pi


@dataclass
class CouplingConfig:
    max_iterations: int = 6000
    checkpoints: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 6000)
    seed: int = 0
    goal_bias: float = 0.1
    step: float = 0.6
    """Initial candidate step ``eta`` [m, d_W] from ``x*`` towards ``q_rand``."""
    steer_attempts: int = 4
    """``eta, eta/2, eta/4, ...``"""
    overlap_margin: float = 0.02
    """Required common slack [m] of two connected regions (same for steering, edges, entries and chains)."""
    max_neighbors: int = 16
    safety_distance: float = 0.02
    active_range: float = 1.0
    max_extent: float = 1.5
    turn_slacks: tuple[float, ...] = ()
    """Path phase, second stage per relative slack ``eps``: min total turning s.t. ``J_W <= (1 + eps) J_W*``."""
    turn_fixed_xy: bool = False
    """Path phase, second stage with the translations of the first stage held (diagnostic)."""


_ZERO_P: dict[int, sp.csc_matrix] = {}
_IDENTITY3 = sp.identity(3, format="csc")


def _rows(cell: PolyhedralCell) -> tuple[tuple[float, float, float], ...]:
    """Rows ``(n_x, n_y, e)`` as Python floats, cached on the cell (scalar tests beat NumPy for a few rows)."""
    rows = getattr(cell, "_py_rows", None)
    if rows is None:
        rows = cell._py_rows = tuple(zip(cell.normals[:, 0].tolist(), cell.normals[:, 1].tolist(),
                                         cell.offsets.tolist())) if len(cell.offsets) else ()
    return rows


def _turns(cell: PolyhedralCell) -> tuple[tuple[float, float], ...] | None:
    """Per-row ``(k_+, k_-)`` as Python floats, or ``None`` for plain cells (``k_+ = k_- = 1``)."""
    if cell.turn_plus is None:
        return None
    turns = getattr(cell, "_py_turns", None)
    if turns is None:
        turns = cell._py_turns = tuple(zip(cell.turn_plus.tolist(), cell.turn_minus.tolist()))
    return turns


def _turn_coefficients(cell: PolyhedralCell) -> tuple[np.ndarray, np.ndarray]:
    k = len(cell.offsets)
    if cell.turn_plus is None:
        return np.ones(k), np.ones(k)
    return cell.turn_plus, cell.turn_minus


def _turned_row_separates(cell: PolyhedralCell, other: PolyhedralCell, vx: float, vy: float, turn: float) -> bool:
    """``_row_separates`` for rows with turn coefficients ``k <= 1``: on ``other`` a row of ``cell`` is at most
    ``n^T v - k_min turn + e + h`` with ``h = min(r_g', min_j f_j + max(1 - k'_min,j, ||n + m_j||) r_g')``."""
    turns = _turns(cell) or ((1.0, 1.0),) * len(cell.offsets)
    other_turns = _turns(other) or ((1.0, 1.0),) * len(other.offsets)
    g = other.guard
    for (nx, ny, e), (kp, km) in zip(_rows(cell), turns):
        reach = g
        for (mx, my, f), (lp, lm) in zip(_rows(other), other_turns):
            reach = min(reach, f + max(1.0 - min(lp, lm), math.hypot(nx + mx, ny + my)) * g)
        if nx * vx + ny * vy - min(kp, km) * turn + e + reach < 0.0:
            return True
    return False


class _Program:
    """Clarabel program ``min q^T x`` s.t. ``b - A x`` in a product of nonnegative and second-order cones."""

    def __init__(self, nvar: int) -> None:
        self.nvar = nvar
        self.eq_a: list[np.ndarray] = []
        self.eq_b: list[np.ndarray] = []
        self.lin_a: list[np.ndarray] = []
        self.lin_b: list[np.ndarray] = []
        self.soc: list[tuple[np.ndarray, np.ndarray]] = []

    def fix(self, col: int, value: float) -> None:
        a = np.zeros((1, self.nvar))
        a[0, col] = 1.0
        self.eq_a.append(a)
        self.eq_b.append(np.array((value,)))

    def nonneg(self, a: np.ndarray, b: np.ndarray) -> None:
        self.lin_a.append(a)
        self.lin_b.append(b)

    def cone(self, a: np.ndarray, b: np.ndarray) -> None:
        self.soc.append((a, b))

    def cell(self, cell: PolyhedralCell, seed: np.ndarray, col: int, margin: float = 0.0,
             margin_col: int | None = None) -> None:
        """``z = x[col:col+3]`` in ``cell`` (lifted seed ``seed``) with slack ``>= margin`` (+ ``x[margin_col]``)."""
        n, e = cell.normals.reshape((-1, 2)), cell.offsets
        k = len(e)
        a = np.zeros((2 * k + 2, self.nvar))
        b = np.empty(2 * k + 2)
        for s, (sign, coef) in enumerate(zip((1.0, -1.0), _turn_coefficients(cell))):
            rows = slice(s * k, (s + 1) * k)
            a[rows, col] = -n[:, 0]
            a[rows, col + 1] = -n[:, 1]
            a[rows, col + 2] = sign * coef
            b[rows] = e - margin - n @ seed[:2] + sign * coef * seed[2]
            a[2 * k + s, col + 2] = sign
            b[2 * k + s] = cell.rho * cell.yaw_limit - margin + sign * seed[2]
        if margin_col is not None:
            a[:, margin_col] = 1.0
        self.nonneg(a, b)
        for sign in (1.0, -1.0):
            a = np.zeros((3, self.nvar))
            a[0, col + 2] = sign
            a[1, col], a[2, col + 1] = -1.0, -1.0
            if margin_col is not None:
                a[0, margin_col] = 1.0
            self.cone(a, np.array((cell.guard - margin + sign * seed[2], -seed[0], -seed[1])))

    def distance(self, t_col: int, first: int | np.ndarray, second: int | np.ndarray) -> None:
        """``||p - q|| <= x[t_col]``; ``p`` / ``q`` are the first column of a variable point or a constant point."""
        a = np.zeros((4, self.nvar))
        b = np.zeros(4)
        a[0, t_col] = -1.0
        for point, sign in ((first, 1.0), (second, -1.0)):
            if isinstance(point, (int, np.integer)):
                a[1:, point:point + 3] -= sign * np.eye(3)
            else:
                b[1:] += sign * np.asarray(point)
        self.cone(a, b)

    def solve(self, q: np.ndarray, settings) -> tuple[np.ndarray | None, str]:
        matrices = self.eq_a + self.lin_a + [a for a, _ in self.soc]
        rhs = self.eq_b + self.lin_b + [b for _, b in self.soc]
        cones = [clarabel.ZeroConeT(len(self.eq_b))] if self.eq_b else []
        cones += [clarabel.NonnegativeConeT(sum(len(b) for b in self.lin_b))]
        cones += [clarabel.SecondOrderConeT(len(b)) for _, b in self.soc]
        zero = _ZERO_P.get(self.nvar)
        if zero is None:
            zero = _ZERO_P[self.nvar] = sp.csc_matrix((self.nvar, self.nvar))
        solver = clarabel.DefaultSolver(zero, q, _csc(np.vstack(matrices)), np.concatenate(rhs), cones, settings)
        result = solver.solve()
        status = str(result.status)
        if status in ("Solved", "AlmostSolved"):
            return np.asarray(result.x), "ok"
        if status in ("PrimalInfeasible", "AlmostPrimalInfeasible"):
            return None, "infeasible"
        return None, "failed"


class RegionGeometry:
    """Convex queries on polyhedral cells in the lifted chart ``(x, y, rho * theta)``."""

    def __init__(self, model: PolyhedralCellModel) -> None:
        self.model, self.rho = model, model.rho
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.static_regularization_constant = 1e-10
        settings.tol_gap_abs = settings.tol_gap_rel = 1e-8
        settings.tol_feas = 1e-8
        settings.reduced_tol_feas = 1e-7
        settings.reduced_tol_gap_abs = settings.reduced_tol_gap_rel = 1e-7
        self.settings = settings
        self.solves = 0
        self.failures = 0
        self.fast = 0
        self.fallbacks = 0

    # -- charts --------------------------------------------------------------------------------------
    def lift(self, pose: Pose2D, anchor_yaw: float) -> np.ndarray:
        return np.array((pose.x, pose.y, self.rho * (anchor_yaw + wrap_to_pi(pose.yaw - anchor_yaw))))

    def pose(self, z: np.ndarray) -> Pose2D:
        return Pose2D(float(z[0]), float(z[1]), wrap_to_pi(float(z[2]) / self.rho))

    def distance(self, a: Pose2D, b: Pose2D) -> float:
        return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (self.rho * wrap_to_pi(b.yaw - a.yaw)) ** 2)

    def slack(self, cell: PolyhedralCell, pose: Pose2D) -> float:
        """``PolyhedralCellModel.slack`` for one pose, in scalar arithmetic."""
        dx, dy = pose.x - cell.pose.x, pose.y - cell.pose.y
        dpsi = cell.rho * wrap_to_pi(pose.yaw - cell.pose.yaw)
        turn = abs(dpsi)
        value = min(cell.guard - math.hypot(dx, dy), cell.rho * cell.yaw_limit) - turn
        turns = _turns(cell)
        if turns is None:
            for nx, ny, e in _rows(cell):
                value = min(value, nx * dx + ny * dy + e - turn)
        else:
            for (nx, ny, e), (kp, km) in zip(_rows(cell), turns):
                value = min(value, nx * dx + ny * dy + e - (kp if dpsi > 0.0 else km) * turn)
        return value

    def goal_yaw(self, cell: PolyhedralCell, goal_xy, toward_yaw: float, margin: float) -> float | None:
        """Yaw closest to ``toward_yaw`` with ``(goal_xy, yaw)`` in ``cell`` at slack ``>= margin``, or ``None``.
        At fixed ``dc`` the cell is ``rho |dtheta| <= min(n^T dc + e, r_g - ||dc||, rho L) - margin``."""
        dx, dy = goal_xy[0] - cell.pose.x, goal_xy[1] - cell.pose.y
        half = min(cell.guard - math.hypot(dx, dy), cell.rho * cell.yaw_limit) - margin
        if half < 0.0:
            return None
        up = down = half
        if len(cell.offsets):
            room = cell.normals @ (dx, dy) + cell.offsets - margin
            if float(room.min()) < 0.0:
                return None
            plus, minus = _turn_coefficients(cell)
            with np.errstate(divide="ignore"):
                up = min(half, float(np.min(np.where(plus > 0.0, room / plus, np.inf))))
                down = min(half, float(np.min(np.where(minus > 0.0, room / minus, np.inf))))
        offset = wrap_to_pi(toward_yaw - cell.pose.yaw)
        return wrap_to_pi(cell.pose.yaw + max(-down / cell.rho, min(up / cell.rho, offset)))

    def _constraints(self, cells, anchor: float, margin: float):
        """``A z <= b`` and guard cones ``||z_xy - c_xy|| <= g - sigma (z_theta - c_theta)`` of shrunk cells."""
        rows, rhs, cones = [], [], []
        for cell in cells:
            cached = getattr(cell, "_py_halfspaces", None)
            if cached is None:
                n, e = cell.normals.reshape((-1, 2)), cell.offsets
                plus, minus = _turn_coefficients(cell)
                signs = np.concatenate((plus, -minus, (1.0, -1.0)))
                matrix = np.zeros((len(signs), 3))
                matrix[:2 * len(e), :2] = np.vstack((-n, -n))
                matrix[:, 2] = signs
                base = np.concatenate((e - n @ (cell.pose.x, cell.pose.y), e - n @ (cell.pose.x, cell.pose.y),
                                       np.full(2, cell.rho * cell.yaw_limit)))
                cached = cell._py_halfspaces = (matrix, base, signs)
            matrix, base, signs = cached
            seed = self.lift(cell.pose, anchor)
            rows.append(matrix)
            rhs.append(base - margin + signs * seed[2])
            cones.append((seed, 1.0, cell.guard - margin))
            cones.append((seed, -1.0, cell.guard - margin))
        return np.vstack(rows), np.concatenate(rhs), cones

    @staticmethod
    def _feasible(z, a, b, cones, tol=1e-9) -> bool:
        if np.any(a @ z - b > tol):
            return False
        return all(math.hypot(z[0] - c[0], z[1] - c[1]) <= g - s * (z[2] - c[2]) + tol for c, s, g in cones)

    def _fast_projection(self, p: np.ndarray, a: np.ndarray, b: np.ndarray, cones) -> tuple[np.ndarray | None, bool]:
        """Exact projection of ``p`` when one halfspace, one guard cone or two halfspaces are active.

        A candidate is accepted only if it is feasible and its KKT multipliers are non-negative, which makes it the
        global minimiser.  Returns ``(z, True)`` on success and ``(None, False)`` when a solver is needed."""
        residual = a @ p - b
        bad_cones = [k for k, (c, s, g) in enumerate(cones) if math.hypot(p[0] - c[0], p[1] - c[1]) > g - s * (p[2] - c[2]) + 1e-12]
        violated = np.flatnonzero(residual > 1e-12)
        if not len(violated) and not bad_cones:
            return p, True
        norms = np.einsum("ij,ij->i", a, a)
        for k in violated[np.argsort(-residual[violated] / np.sqrt(norms[violated]))]:
            z = p - residual[k] / norms[k] * a[k]
            if self._feasible(z, a, b, cones):
                return z, True
        for k in bad_cones:
            c, s, g = cones[k]
            u = p[:2] - c[:2]
            t = g - s * (p[2] - c[2])
            length = math.hypot(u[0], u[1])
            if length <= -t:
                u, t = np.zeros(2), 0.0
            else:
                alpha = 0.5 * (length + t)
                u, t = alpha * u / max(length, 1e-300), alpha
            z = np.array((c[0] + u[0], c[1] + u[1], c[2] + s * (g - t)))
            if self._feasible(z, a, b, cones):
                return z, True
        candidates = violated[np.argsort(-residual[violated])][:4].tolist()
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                ki, kj = candidates[i], candidates[j]
                gii, gjj, gij = float(norms[ki]), float(norms[kj]), float(a[ki] @ a[kj])
                det = gii * gjj - gij * gij
                if det < 1e-12:
                    continue
                ri, rj = float(residual[ki]), float(residual[kj])
                li, lj = (gjj * ri - gij * rj) / det, (gii * rj - gij * ri) / det
                if li < -1e-12 or lj < -1e-12:
                    continue
                z = p - li * a[ki] - lj * a[kj]
                if self._feasible(z, a, b, cones):
                    return z, True
        return None, False

    @staticmethod
    def _cone_rows(cones) -> np.ndarray:
        return np.array([(c[0], c[1], c[2], s, g) for c, s, g in cones])

    def _solve_projection(self, p: np.ndarray, a: np.ndarray, b: np.ndarray, cones) -> np.ndarray | None:
        """``argmin 1/2 ||z - p||^2`` s.t. ``A z <= b`` and the guard cones; ``None`` if the set has no interior."""
        self.solves += 1
        status, z = region_qp.project(a, b, self._cone_rows(cones), p)
        if status >= 0:
            return z if status == 1 else None
        self.fallbacks += 1
        return self._clarabel_projection(p, a, b, cones)

    def _clarabel_projection(self, p: np.ndarray, a: np.ndarray, b: np.ndarray, cones) -> np.ndarray | None:
        rows = len(b) + 3 * len(cones)
        matrix = np.zeros((rows, 3))
        rhs = np.empty(rows)
        matrix[:len(b)] = a
        rhs[:len(b)] = b
        k = len(b)
        for c, s, g in cones:
            matrix[k, 2] = s
            matrix[k + 1, 0] = matrix[k + 2, 1] = -1.0
            rhs[k:k + 3] = (g + s * c[2], -c[0], -c[1])
            k += 3
        constraint = sp.csc_matrix((matrix.T.ravel(), np.tile(np.arange(rows), 3), np.arange(4) * rows),
                                   shape=(rows, 3))
        cone_types = [clarabel.NonnegativeConeT(len(b))] + [clarabel.SecondOrderConeT(3)] * len(cones)
        result = clarabel.DefaultSolver(_IDENTITY3, -p, constraint, rhs, cone_types, self.settings).solve()
        status = str(result.status)
        if status in ("Solved", "AlmostSolved"):
            return np.asarray(result.x)
        self.failures += int(status not in ("PrimalInfeasible", "AlmostPrimalInfeasible"))
        return None

    def _run(self, program: _Program, q: np.ndarray) -> np.ndarray | None:
        self.solves += 1
        x, status = program.solve(q, self.settings)
        self.failures += int(status == "failed")
        return x

    # -- queries ------------------------------------------------------------------------------------
    def project(self, cell: PolyhedralCell, target: Pose2D) -> Pose2D:
        """``argmin_{x in C} d_W(x, target)`` over the whole cell."""
        if self.slack(cell, target) >= 0.0:
            return target
        z = self._solve_projection(self.lift(target, cell.pose.yaw), *self._constraints((cell,), cell.pose.yaw, 0.0))
        if z is None:
            raise RuntimeError("projection onto a valid cell failed")
        return self.pose(z)

    def max_margin(self, first: PolyhedralCell, second: PolyhedralCell) -> tuple[float, Pose2D | None]:
        """``max t`` with a common point at slack ``>= t`` in both cells, and that point."""
        anchor = first.pose.yaw
        a, b, cones = self._constraints((first, second), anchor, 0.0)
        self.solves += 1
        status, t, z = region_qp.margin(a, b, self._cone_rows(cones))
        if status == 1:
            return t, self.pose(z)
        self.fallbacks += 1
        program = _Program(4)
        program.cell(first, self.lift(first.pose, anchor), 0, margin_col=3)
        program.cell(second, self.lift(second.pose, anchor), 0, margin_col=3)
        x = self._run(program, np.array((0.0, 0.0, 0.0, -1.0)))
        if x is None:
            return -math.inf, None
        return float(x[3]), self.pose(x[:3])

    def attach(self, entry: Pose2D, first: PolyhedralCell, second: PolyhedralCell, margin: float) -> Pose2D | None:
        """``argmin d_W(entry, z)`` over ``z in C_first ∩ C_second`` with slack ``>= margin``, ``None`` if empty."""
        if self.slack(first, entry) >= margin and self.slack(second, entry) >= margin:
            return entry
        anchor = first.pose.yaw
        z = self._solve_projection(self.lift(entry, anchor), *self._constraints((first, second), anchor, margin))
        return None if z is None else self.pose(z)

    def via(self, a: Pose2D, first: PolyhedralCell, second: PolyhedralCell, b: Pose2D, margin: float
            ) -> tuple[float, Pose2D] | None:
        """``min d_W(a, z) + d_W(z, b)`` over ``z in C_first ∩ C_second`` with slack ``>= margin``, and ``z``."""
        anchor = first.pose.yaw
        program = _Program(5)
        program.cell(first, self.lift(first.pose, anchor), 0, margin)
        program.cell(second, self.lift(second.pose, anchor), 0, margin)
        program.distance(3, 0, self.lift(a, anchor))
        program.distance(4, 0, self.lift(b, anchor))
        x = self._run(program, np.array((0.0, 0.0, 0.0, 1.0, 1.0)))
        if x is None:
            return None
        z = self.pose(x[:3])
        return self.distance(a, z) + self.distance(z, b), z

    def chain(self, start: Pose2D, goal_xy, cells: list[PolyhedralCell], margin: float
              ) -> tuple[float, list[Pose2D]] | None:
        """Shortest ``start -> z_0 -> ... -> z_{m-1} -> goal`` with ``z_k in C_k ∩ C_{k+1}`` (slack ``>= margin``)
        and ``(goal_xy, yaw) in C_m``; the goal yaw is free.  Returns the length and the poses."""
        program, points, t0 = self._chain_program(start, goal_xy, cells, margin, 0)
        q = np.zeros(program.nvar)
        q[t0:] = 1.0
        x = self._run(program, q)
        return None if x is None else self._chain_route(start, goal_xy, x, len(cells) - 1)

    def min_turn(self, start: Pose2D, goal_xy, cells: list[PolyhedralCell], margin: float,
                 budget: float | None = None, fixed: list[Pose2D] | None = None
                 ) -> tuple[float, list[Pose2D]] | None:
        """Second stage on the same region chain: ``min sum_k |theta_{k+1} - theta_k|`` with the chain constraints
        and either ``J_W <= budget`` or the translations of ``fixed`` (a route of the same chain) held."""
        m = len(cells) - 1
        program, points, t0 = self._chain_program(start, goal_xy, cells, margin, m + 1)
        u0 = t0 + m + 1
        if budget is not None:
            a = np.zeros((1, program.nvar))
            a[0, t0:u0] = 1.0
            program.nonneg(a, np.array((budget,)))
        if fixed is not None:
            for k in range(m):
                program.fix(3 * k, fixed[k + 1].x)
                program.fix(3 * k + 1, fixed[k + 1].y)
        a = np.zeros((2 * (m + 1), program.nvar))
        b = np.zeros(2 * (m + 1))
        for k in range(m + 1):
            d = np.zeros(program.nvar)
            c = 0.0
            for point, sign in ((points[k + 1], 1.0), (points[k], -1.0)):
                if isinstance(point, (int, np.integer)):
                    d[point + 2] += sign
                else:
                    c += sign * float(point[2])
            a[2 * k], b[2 * k] = d, -c
            a[2 * k, u0 + k] = -1.0
            a[2 * k + 1], b[2 * k + 1] = -d, c
            a[2 * k + 1, u0 + k] = -1.0
        program.nonneg(a, b)
        q = np.zeros(program.nvar)
        q[u0:] = 1.0
        x = self._run(program, q)
        return None if x is None else self._chain_route(start, goal_xy, x, m)

    def _chain_program(self, start: Pose2D, goal_xy, cells: list[PolyhedralCell], margin: float, extra: int):
        m = len(cells) - 1
        yaws = [cells[0].pose.yaw]
        for cell in cells[1:]:
            yaws.append(yaws[-1] + wrap_to_pi(cell.pose.yaw - yaws[-1]))
        seeds = [np.array((c.pose.x, c.pose.y, self.rho * y)) for c, y in zip(cells, yaws)]
        goal_col, t0 = 3 * m, 3 * m + 3
        program = _Program(t0 + m + 1 + extra)
        for k in range(m):
            program.cell(cells[k], seeds[k], 3 * k, margin)
            program.cell(cells[k + 1], seeds[k + 1], 3 * k, margin)
        program.cell(cells[m], seeds[m], goal_col, margin)
        program.fix(goal_col, goal_xy[0])
        program.fix(goal_col + 1, goal_xy[1])
        points: list[int | np.ndarray] = [self.lift(start, yaws[0])] + [3 * k for k in range(m)] + [goal_col]
        for k in range(m + 1):
            program.distance(t0 + k, points[k + 1], points[k])
        return program, points, t0

    def _chain_route(self, start: Pose2D, goal_xy, x: np.ndarray, m: int) -> tuple[float, list[Pose2D]]:
        goal_col = 3 * m
        route = [start] + [self.pose(x[3 * k:3 * k + 3]) for k in range(m)]
        route.append(Pose2D(goal_xy[0], goal_xy[1], wrap_to_pi(float(x[goal_col + 2]) / self.rho)))
        length = sum(self.distance(a, b) for a, b in zip(route, route[1:]))
        return length, route


class _CorePlanner:
    """Shared region expansion; subclasses add the coupling-specific bookkeeping in ``_connect``."""

    variant = "core"
    cell_model = PolyhedralCellModel

    def __init__(self, map_data: MapData, slots: np.ndarray, start: Pose2D, config: CouplingConfig) -> None:
        self.map_data, self.start, self.config = map_data, start, config
        self.goal_xy = tuple(map_data.goal_xy)
        self.slots = np.asarray(slots, dtype=float)
        margin = float(map_data.robot_radius) + float(map_data.safety_margin)
        self.model = self.cell_model(self.slots, map_data, margin, config.safety_distance, config.active_range,
                                     max_extent=config.max_extent, lazy_geometry=True)
        self._turned = self.cell_model is not PolyhedralCellModel
        self.geo = RegionGeometry(self.model)
        self.rho = self.model.rho
        self.rng = np.random.default_rng(config.seed)
        self.cells: list[PolyhedralCell] = []
        self.discovery: list[int | None] = []
        self.goal_cells: list[int] = []
        cap, rows = 256, 16
        self._x, self._y, self._t = np.zeros(cap), np.zeros(cap), np.zeros(cap)
        self._g, self._lim, self._a = np.zeros(cap), np.zeros(cap), np.zeros(cap)
        self._n = np.zeros((cap, rows, 2))
        self._e = np.full((cap, rows), np.inf)
        self._kp, self._km = np.ones((cap, rows)), np.ones((cap, rows))
        self.stats = {"iterations": 0, "covered_samples": 0, "steer_failed": 0, "steer_first_try": 0,
                      "steer_cells": 0, "backoffs": 0, "select_time_s": 0.0, "steer_time_s": 0.0,
                      "connect_time_s": 0.0, "connect_solves": 0, "core_solves": 0, "neighbor_candidates": 0,
                      "neighbor_separated": 0}

    # -- storage ------------------------------------------------------------------------------------
    def _add_cell(self, cell: PolyhedralCell, parent: int | None) -> int:
        i = len(self.cells)
        if i == len(self._x):
            grow = len(self._x)
            for name in ("_x", "_y", "_t", "_g", "_lim", "_a"):
                setattr(self, name, np.concatenate((getattr(self, name), np.zeros(grow))))
            self._n = np.concatenate((self._n, np.zeros((grow,) + self._n.shape[1:])))
            self._e = np.concatenate((self._e, np.full((grow, self._e.shape[1]), np.inf)))
            self._kp = np.concatenate((self._kp, np.ones((grow, self._kp.shape[1]))))
            self._km = np.concatenate((self._km, np.ones((grow, self._km.shape[1]))))
        k = len(cell.offsets)
        if k > self._e.shape[1]:
            extra = k - self._e.shape[1]
            self._n = np.concatenate((self._n, np.zeros((len(self._n), extra, 2))), axis=1)
            self._e = np.concatenate((self._e, np.full((len(self._e), extra), np.inf)), axis=1)
            self._kp = np.concatenate((self._kp, np.ones((len(self._kp), extra))), axis=1)
            self._km = np.concatenate((self._km, np.ones((len(self._km), extra))), axis=1)
        if cell.turn_plus is not None:
            self._kp[i, :k], self._km[i, :k] = cell.turn_plus, cell.turn_minus
        self._x[i], self._y[i], self._t[i] = cell.pose.x, cell.pose.y, cell.pose.yaw
        self._g[i], self._lim[i] = cell.guard, cell.rho * cell.yaw_limit
        self._a[i] = max(0.0, self.geo.slack(cell, cell.pose)) / math.sqrt(2.0)
        self._n[i, :k], self._e[i, :k] = cell.normals.reshape((-1, 2)), cell.offsets
        self.cells.append(cell)
        self.discovery.append(parent)
        return i

    # -- shared core --------------------------------------------------------------------------------
    def _sample(self) -> Pose2D:
        if self.rng.random() < self.config.goal_bias:
            return Pose2D(self.goal_xy[0], self.goal_xy[1], float(self.rng.uniform(-math.pi, math.pi)))
        ox, oy = self.map_data.origin_xy
        return Pose2D(float(self.rng.uniform(ox, ox + self.map_data.width_m)),
                      float(self.rng.uniform(oy, oy + self.map_data.height_m)),
                      float(self.rng.uniform(-math.pi, math.pi)))

    def _select(self, sample: Pose2D) -> int:
        """Region-gap rule ``argmin_i D_i - l_i(u_i)``: ``l_i`` is the reach of ``C_i`` from its seed along the
        seed-to-sample ray, so the gap is the d_W distance left after leaving ``C_i`` along that ray."""
        idx, gap = self._gap_candidates(sample, 1)
        return int(idx[np.argmin(gap)])

    def _gap_candidates(self, sample: Pose2D, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Regions that can be among the ``k`` smallest gaps ``D_i - l_i(u_i)``, and their gaps."""
        count = len(self.cells)
        v = np.column_stack((sample.x - self._x[:count], sample.y - self._y[:count],
                             self.rho * (np.remainder(sample.yaw - self._t[:count] + math.pi, 2 * math.pi) - math.pi)))
        length = np.sqrt(np.einsum("ij,ij->i", v, v))
        # a_i <= l_i <= r_g,i: only regions whose lower bound reaches the k-th best upper bound can be in the top k.
        upper = length - self._a[:count]
        bound = np.min(upper) if k == 1 or k >= count else np.partition(upper, k - 1)[k - 1]
        idx = np.flatnonzero(length - self._g[:count] <= bound)
        v, length = v[idx], length[idx]
        u = v / np.maximum(length, 1e-300)[:, None]
        turn = np.abs(u[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            reach = np.minimum(self._g[idx] / (np.hypot(u[:, 0], u[:, 1]) + turn),
                               np.where(turn > 0.0, self._lim[idx] / turn, np.inf))
            if self._turned:
                k = np.where(u[:, 2:3] > 0.0, self._kp[idx], self._km[idx])
                rate = k * turn[:, None] - np.einsum("ikj,ij->ik", self._n[idx], u[:, :2])
            else:
                rate = turn[:, None] - np.einsum("ikj,ij->ik", self._n[idx], u[:, :2])
            rows = np.where(rate > 1e-15, self._e[idx] / rate, np.inf)
        reach = np.minimum(reach, rows.min(axis=1))
        return idx, length - reach

    def _steer(self, parent: int, sample: Pose2D) -> tuple[PolyhedralCell | None, Pose2D | None, int]:
        """``x* = Pi_{C_i}(q_rand)``, then ``eta, eta/2, ...`` until ``C_new`` overlaps ``C_i`` by the margin."""
        cell = self.cells[parent]
        before = self.geo.solves
        started = time.perf_counter()
        x_star = self._launch(cell, sample)
        self.stats["select_time_s"] += time.perf_counter() - started
        started = time.perf_counter()
        anchor = cell.pose.yaw
        z_star, z_rand = self.geo.lift(x_star, anchor), self.geo.lift(sample, anchor)
        direction = z_rand - z_star
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            self.stats["covered_samples"] += 1
            self.stats["core_solves"] += self.geo.solves - before
            self.stats["steer_time_s"] += time.perf_counter() - started
            return None, None, 0
        direction /= length
        step = self._initial_step(cell, z_star, direction, anchor)
        result: tuple[PolyhedralCell | None, Pose2D | None, int] = (None, None, self.config.steer_attempts)
        for attempt in range(1, self.config.steer_attempts + 1):
            pose = self.geo.pose(z_star + min(step, length) * direction)
            new = self.model.make_cell(pose)
            self.stats["steer_cells"] += 1
            if new.valid and not self._separated(cell, new):
                portal = self._overlap(cell, new)
                if portal is not None:
                    result = (new, portal, attempt)
                    break
            step *= 0.5
        self.stats["core_solves"] += self.geo.solves - before
        self.stats["steer_time_s"] += time.perf_counter() - started
        new, _, attempt = result
        if new is None:
            self.stats["steer_failed"] += 1
        else:
            self.stats["steer_first_try"] += int(attempt == 1)
            self.stats["backoffs"] += attempt - 1
        return result

    def _launch(self, cell: PolyhedralCell, sample: Pose2D) -> Pose2D:
        """Where the step towards ``q_rand`` starts: the projection ``Pi_{C_i}(q_rand)``."""
        return self.geo.project(cell, sample)

    def _initial_step(self, cell: PolyhedralCell, z_star: np.ndarray, direction: np.ndarray, anchor: float) -> float:
        """First candidate step from ``x*`` along the unit lifted direction (later halved on failure)."""
        return self.config.step

    def _quick_portal(self, first: PolyhedralCell, second: PolyhedralCell, margin: float) -> Pose2D | None:
        """A point on the seed segment with slack ``>= margin`` in both cells, without optimisation."""
        a = self.geo.lift(first.pose, first.pose.yaw)
        b = self.geo.lift(second.pose, first.pose.yaw)
        for s in (0.5, 0.25, 0.75):
            pose = self.geo.pose(a + s * (b - a))
            if self.geo.slack(first, pose) >= margin and self.geo.slack(second, pose) >= margin:
                return pose
        z, done = self.geo._fast_projection(0.5 * (a + b), *self.geo._constraints((first, second), first.pose.yaw,
                                                                                    margin * (1.0 + 1e-6)))
        return self.geo.pose(z) if done else None

    def _overlap(self, first: PolyhedralCell, second: PolyhedralCell) -> Pose2D | None:
        """A common point with slack ``>= overlap_margin`` in both cells, or ``None`` if there is none."""
        margin = self.config.overlap_margin
        portal = self._quick_portal(first, second, margin)
        if portal is None:
            t, portal = self.geo.max_margin(first, second)
            portal = portal if t >= margin else None
        return portal

    def _neighbors(self, index: int) -> list[int]:
        """Existing regions (except ``index`` and its discovery parent) whose guard disks and yaw ranges meet
        ``C_index`` and that are not proven disjoint at the overlap margin; the ``max_neighbors`` closest seeds."""
        cell = self.cells[index]
        count = index
        dx, dy = self._x[:count] - cell.pose.x, self._y[:count] - cell.pose.y
        dyaw = self.rho * np.abs(np.remainder(self._t[:count] - cell.pose.yaw + math.pi, 2 * math.pi) - math.pi)
        mask = (np.hypot(dx, dy) < self._g[:count] + cell.guard) & (dyaw < self._lim[:count] + cell.rho * cell.yaw_limit)
        parent = self.discovery[index]
        if parent is not None:
            mask[parent] = False
        candidates = np.flatnonzero(mask)
        order = np.argsort(np.sqrt(dx[candidates] ** 2 + dy[candidates] ** 2 + dyaw[candidates] ** 2), kind="stable")
        candidates = candidates[order][:self.config.max_neighbors]
        self.stats["neighbor_candidates"] += len(candidates)
        kept = []
        for j in candidates.tolist():
            if self._separated(self.cells[j], cell):
                self.stats["neighbor_separated"] += 1
            else:
                kept.append(j)
        return kept

    def _separated(self, first: PolyhedralCell, second: PolyhedralCell) -> bool:
        """Cheap proof that no point has slack ``>= overlap_margin`` in both cells: slack is 1-Lipschitz in d_G and
        a cell lies in the d_G ball of its guard around its seed; or one row is negative on the whole other cell."""
        margin = self.config.overlap_margin
        if (self.geo.slack(first, second.pose) + second.guard < margin
                or self.geo.slack(second, first.pose) + first.guard < margin):
            return True
        vx, vy = second.pose.x - first.pose.x, second.pose.y - first.pose.y
        turn = self.rho * abs(wrap_to_pi(second.pose.yaw - first.pose.yaw))
        if self._turned:
            return (_turned_row_separates(first, second, vx, vy, turn)
                    or _turned_row_separates(second, first, -vx, -vy, turn))
        return (_row_separates(_rows(first), _rows(second), vx, vy, turn, second.guard)
                or _row_separates(_rows(second), _rows(first), -vx, -vy, turn, first.guard))

    # -- main loop ----------------------------------------------------------------------------------
    def plan(self) -> dict:
        config = self.config
        start_cell = self.model.make_cell(self.start)
        if not start_cell.valid:
            raise ValueError("start is in collision")
        self._add_cell(start_cell, None)
        self._init_root()
        if self.geo.goal_yaw(start_cell, self.goal_xy, self.start.yaw, config.overlap_margin) is not None:
            self.goal_cells.append(0)
        online = 0.0
        first: dict | None = None
        checkpoints: list[dict] = []
        pending = sorted(set(config.checkpoints))
        for iteration in range(1, config.max_iterations + 1):
            started = time.perf_counter()
            sample = self._sample()
            t = time.perf_counter()
            parent = self._select(sample)
            self.stats["select_time_s"] += time.perf_counter() - t
            cell, portal, _ = self._steer(parent, sample)
            if cell is not None:
                index = self._add_cell(cell, parent)
                t, solves = time.perf_counter(), self.geo.solves
                self._connect(index, parent, portal)
                self.stats["connect_time_s"] += time.perf_counter() - t
                self.stats["connect_solves"] += self.geo.solves - solves
                if self.geo.goal_yaw(cell, self.goal_xy, cell.pose.yaw, config.overlap_margin) is not None:
                    self.goal_cells.append(index)
            online += time.perf_counter() - started
            self.stats["iterations"] = iteration
            if first is None and self.goal_cells:
                first = self._checkpoint(iteration, online)
                first["kind"] = "first"
            while pending and pending[0] <= iteration:
                pending.pop(0)
                record = self._checkpoint(iteration, online)
                record["kind"] = "budget"
                checkpoints.append(record)
        return {"variant": self.variant, "success": bool(self.goal_cells), "first": first,
                "checkpoints": checkpoints, "regions": len(self.cells), "goal_cells": len(self.goal_cells),
                "online_time_s": online, "solver_failures": self.geo.failures,
                "stats": dict(self.stats)}

    def _checkpoint(self, iteration: int, online: float) -> dict:
        """Path phase at this iteration: region chain extraction + the shared chain optimiser (timed apart)."""
        record = {"iteration": iteration, "online_time_s": online, "regions": len(self.cells), "success": False}
        if not self.goal_cells:
            return record
        started = time.perf_counter()
        sequence, raw = self._region_chain()
        extract = time.perf_counter() - started
        started = time.perf_counter()
        cells = [self.cells[i] for i in sequence]
        margin = self.config.overlap_margin * (1.0 - 1e-6)
        solved = self.geo.chain(self.start, self.goal_xy, cells, margin)
        optimise = time.perf_counter() - started
        record.update(success=True, raw_cost=raw, chain_regions=len(sequence), extract_time_s=extract,
                      optimise_time_s=optimise, sequence=sequence,
                      optimised_cost=solved[0] if solved else math.nan,
                      route=[(p.x, p.y, p.yaw) for p in solved[1]] if solved else [])
        if solved:
            stages = {f"eps={eps:g}": dict(budget=solved[0] * (1.0 + max(eps, 1e-6))) for eps in self.config.turn_slacks}
            if self.config.turn_fixed_xy:
                stages["fixed_xy"] = dict(fixed=solved[1])
            post = {}
            for key, options in stages.items():
                started = time.perf_counter()
                second = self.geo.min_turn(self.start, self.goal_xy, cells, margin, **options)
                post[key] = dict(time_s=time.perf_counter() - started, cost=second[0] if second else math.nan,
                                 route=[(p.x, p.y, p.yaw) for p in second[1]] if second else [])
            if post:
                record["post"] = post
        return record

    def _init_root(self) -> None:
        raise NotImplementedError

    def _connect(self, index: int, parent: int, portal: Pose2D) -> None:
        raise NotImplementedError

    def _region_chain(self) -> tuple[list[int], float]:
        raise NotImplementedError


class SingleEntryRRTStar(_CorePlanner):
    """Online path optimisation: entry ``e_i`` and cost-to-entry ``J_i`` per region, continuous ChooseParent and
    whole-region Rewire.  A changed entry bumps the region's version; descendants re-attach lazily on access."""

    variant = "single_entry_rrtstar"

    def _init_root(self) -> None:
        self.parent: list[int | None] = [None]
        self.entry: list[Pose2D] = [self.start]
        self.cost: list[float] = [0.0]
        self.version: list[int] = [0]
        self.parent_version: list[int] = [0]
        self.stats.update(choose_parent_tests=0, choose_parent_improved=0, rewire_tests=0, rewires=0,
                          repairs=0, attach_solves=0)

    def _attach(self, entry: Pose2D, first: PolyhedralCell, second: PolyhedralCell) -> Pose2D | None:
        before = self.geo.solves
        z = self.geo.attach(entry, first, second, self.config.overlap_margin * (1.0 - 1e-6))
        self.stats["attach_solves"] += self.geo.solves - before
        return z

    def _bound(self, entry: Pose2D, first: PolyhedralCell, second: PolyhedralCell) -> float:
        """Lower bound on ``d_W(entry, C_first ∩ C_second)``: the slack is sqrt(2)-Lipschitz in ``d_W``."""
        slack = min(self.geo.slack(first, entry), self.geo.slack(second, entry))
        return max(0.0, self.config.overlap_margin - slack) / math.sqrt(2.0)

    def _refresh(self, index: int) -> None:
        chain = []
        cursor = index
        while self.parent[cursor] is not None:
            chain.append(cursor)
            cursor = self.parent[cursor]
        for node in reversed(chain):
            p = self.parent[node]
            if self.parent_version[node] == self.version[p]:
                continue
            z = self._attach(self.entry[p], self.cells[p], self.cells[node])
            if z is not None:
                self.entry[node] = z
            self.cost[node] = self.cost[p] + self.geo.distance(self.entry[p], self.entry[node])
            self.parent_version[node] = self.version[p]
            self.version[node] += 1
            self.stats["repairs"] += 1

    def _connect(self, index: int, parent: int, portal: Pose2D) -> None:
        cell = self.cells[index]
        neighbors = self._neighbors(index)
        candidates = [parent] + neighbors
        for j in candidates:
            self._refresh(j)
        best, best_cost, best_entry = None, math.inf, None
        for j in sorted(candidates, key=lambda j: (j != parent, self.cost[j])):
            if self.cost[j] + self._bound(self.entry[j], self.cells[j], cell) >= best_cost:
                continue
            self.stats["choose_parent_tests"] += 1
            z = self._attach(self.entry[j], self.cells[j], cell)
            if z is None:
                continue
            c = self.cost[j] + self.geo.distance(self.entry[j], z)
            if c < best_cost:
                self.stats["choose_parent_improved"] += int(best is not None)
                best, best_cost, best_entry = j, c, z
        if best is None:
            # Steering certified the overlap; keep the certified portal if the margin solve disagrees numerically.
            best, best_entry = parent, portal
            best_cost = self.cost[parent] + self.geo.distance(self.entry[parent], portal)
        self.parent.append(best)
        self.entry.append(best_entry)
        self.cost.append(best_cost)
        self.version.append(0)
        self.parent_version.append(self.version[best])
        self._after_choose(index)
        best, best_entry, best_cost = self.parent[index], self.entry[index], self.cost[index]
        ancestors = set()
        cursor = best
        while cursor is not None:
            ancestors.add(cursor)
            cursor = self.parent[cursor]
        for k in neighbors + ([parent] if parent != best else []):
            if k in ancestors:
                continue
            self._refresh(k)
            if best_cost + self._bound(best_entry, cell, self.cells[k]) >= self.cost[k] - 1e-9:
                continue
            self.stats["rewire_tests"] += 1
            z = self._attach(best_entry, cell, self.cells[k])
            if z is None:
                continue
            c = best_cost + self.geo.distance(best_entry, z)
            if c < self.cost[k] - 1e-9:
                self.parent[k], self.entry[k], self.cost[k] = index, z, c
                self.parent_version[k] = self.version[index]
                self.version[k] += 1
                self.stats["rewires"] += 1

    def _after_choose(self, index: int) -> None:
        """Hook between ChooseParent and Rewire of a new region."""

    def _goal_cost(self, g: int) -> float:
        self._refresh(g)
        yaw = self.geo.goal_yaw(self.cells[g], self.goal_xy, self.entry[g].yaw, self.config.overlap_margin)
        return self.cost[g] + self.geo.distance(self.entry[g], Pose2D(self.goal_xy[0], self.goal_xy[1], yaw))

    def _region_chain(self) -> tuple[list[int], float]:
        costs = [self._goal_cost(g) for g in self.goal_cells]
        k = int(np.argmin(costs))
        sequence, cursor = [], self.goal_cells[k]
        while cursor is not None:
            sequence.append(cursor)
            cursor = self.parent[cursor]
        return sequence[::-1], float(costs[k])


class AncestorRRTStar(SingleEntryRRTStar):
    """A + ancestor connection: after ChooseParent the new region ``n`` tries the ancestors above its parent (up to
    ``depth`` levels); ``a`` is taken when ``e' = argmin_{z in C_a ∩ C_n} d_W(e_a, z)`` (delta-overlap) gives
    ``J_a + d_W(e_a, e') + d_W(e', e_n) < J_n``, so no descendant cost can increase."""

    variant = "single_entry_ancestor"
    depth = 3

    def _init_root(self) -> None:
        super()._init_root()
        self.stats.update(ancestor_screened=0, ancestor_solves=0, ancestor_reconnects=0, ancestor_gain=0.0)

    def _after_choose(self, index: int) -> None:
        cell, entry, cost = self.cells[index], self.entry[index], self.cost[index]
        best = None
        cursor = self.parent[self.parent[index]]
        for _ in range(self.depth):
            if cursor is None:
                break
            a = cursor
            cursor = self.parent[a]
            self.stats["ancestor_screened"] += 1
            if self.cost[a] + self.geo.distance(self.entry[a], entry) >= cost - 1e-9:
                continue
            self.stats["ancestor_solves"] += 1
            z = self._attach(self.entry[a], self.cells[a], cell)
            if z is None:
                continue
            c = self.cost[a] + self.geo.distance(self.entry[a], z)
            if c + self.geo.distance(z, entry) < cost - 1e-9 and (best is None or c < best[1]):
                best = (a, c, z)
        if best is not None:
            a, c, z = best
            self.stats["ancestor_reconnects"] += 1
            self.stats["ancestor_gain"] += cost - c
            self.parent[index], self.entry[index], self.cost[index] = a, z, c
            self.parent_version[index] = self.version[a]
            self.version[index] += 1


class EntryRayRRTStar(SingleEntryRRTStar):
    """A + entry-ray launch: the step towards ``q_rand`` starts where the ray ``e_i -> q_rand`` leaves ``C_i``
    (``q_rand`` itself if it is inside), instead of ``Pi_{C_i}(q_rand)``."""

    variant = "single_entry_entry_ray"

    def _steer(self, parent: int, sample: Pose2D):
        self._refresh(parent)
        self._launch_entry = self.entry[parent]
        return super()._steer(parent, sample)

    def _launch(self, cell: PolyhedralCell, sample: Pose2D) -> Pose2D:
        if self.geo.slack(cell, sample) >= 0.0:
            return sample
        anchor = cell.pose.yaw
        e, q = self.geo.lift(self._launch_entry, anchor), self.geo.lift(sample, anchor)
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.geo.slack(cell, self.geo.pose(e + mid * (q - e))) >= 0.0:
                lo = mid
            else:
                hi = mid
        return self.geo.pose(e + lo * (q - e))


class AncestorEntryRayRRTStar(EntryRayRRTStar, AncestorRRTStar):
    variant = "single_entry_ancestor_entry_ray"


class ConsistentRRTStar(SingleEntryRRTStar):
    """A with cost comparisons at a common state.  ``consistent_rewire``: ``k`` is rewired only if
    ``J_k' + d_W(e_k', e_k) < J_k`` (the new branch also reaches the old entry more cheaply).  ``reference_parent``:
    ChooseParent ranks candidates by ``F_j = J_j + min_{z in C_j ∩ C_n} [d_W(e_j, z) + d_W(z, q_n)]`` (``q_n`` the
    construction state of ``C_n``) and enters at the minimiser ``z``; ``J_n`` stays the cost to ``e_n``."""

    variant = "single_entry_consistent"
    consistent_rewire = True
    reference_parent = True

    def _init_root(self) -> None:
        super()._init_root()
        self.stats.update(via_solves=0)

    def _connect(self, index: int, parent: int, portal: Pose2D) -> None:
        cell = self.cells[index]
        q_n = cell.pose
        margin = self.config.overlap_margin * (1.0 - 1e-6)
        neighbors = self._neighbors(index)
        candidates = [parent] + neighbors
        for j in candidates:
            self._refresh(j)
        best, best_rank, best_cost, best_entry = None, math.inf, math.inf, None
        for j in sorted(candidates, key=lambda j: (j != parent, self.cost[j])):
            if self.reference_parent:
                if self.cost[j] + self.geo.distance(self.entry[j], q_n) >= best_rank:
                    continue
                self.stats["choose_parent_tests"] += 1
                self.stats["via_solves"] += 1
                solved = self.geo.via(self.entry[j], self.cells[j], cell, q_n, margin)
                if solved is None:
                    continue
                rank, z = self.cost[j] + solved[0], solved[1]
                c = self.cost[j] + self.geo.distance(self.entry[j], z)
            else:
                if self.cost[j] + self._bound(self.entry[j], self.cells[j], cell) >= best_rank:
                    continue
                self.stats["choose_parent_tests"] += 1
                z = self._attach(self.entry[j], self.cells[j], cell)
                if z is None:
                    continue
                c = rank = self.cost[j] + self.geo.distance(self.entry[j], z)
            if rank < best_rank:
                self.stats["choose_parent_improved"] += int(best is not None)
                best, best_rank, best_cost, best_entry = j, rank, c, z
        if best is None:
            best, best_entry = parent, portal
            best_cost = self.cost[parent] + self.geo.distance(self.entry[parent], portal)
        self.parent.append(best)
        self.entry.append(best_entry)
        self.cost.append(best_cost)
        self.version.append(0)
        self.parent_version.append(self.version[best])
        ancestors = set()
        cursor = best
        while cursor is not None:
            ancestors.add(cursor)
            cursor = self.parent[cursor]
        for k in neighbors + ([parent] if parent != best else []):
            if k in ancestors:
                continue
            self._refresh(k)
            if best_cost + self._bound(best_entry, cell, self.cells[k]) >= self.cost[k] - 1e-9:
                continue
            self.stats["rewire_tests"] += 1
            z = self._attach(best_entry, cell, self.cells[k])
            if z is None:
                continue
            c = best_cost + self.geo.distance(best_entry, z)
            reach = c + self.geo.distance(z, self.entry[k]) if self.consistent_rewire else c
            if reach < self.cost[k] - 1e-9:
                self.parent[k], self.entry[k], self.cost[k] = index, z, c
                self.parent_version[k] = self.version[index]
                self.version[k] += 1
                self.stats["rewires"] += 1


class ConsistentRewireRRTStar(ConsistentRRTStar):
    variant = "single_entry_consistent_rewire"
    reference_parent = False


class ReferenceParentRRTStar(ConsistentRRTStar):
    variant = "single_entry_reference_parent"
    consistent_rewire = False


class AdaptiveStepRRTStar(SingleEntryRRTStar):
    """A with the first candidate step ``clip(kappa l^-(u), eta_min, eta_max)`` (capped by ``d_W(x*, q_rand)`` in
    the loop), ``l^-(u) = max{t >= 0 : z(x*) - t u in C_i}`` the depth of ``C_i`` behind ``x*`` along the step
    direction; the halving back-off is unchanged."""

    variant = "single_entry_adaptive_step"
    kappa, eta_min, eta_max = 1.0, 0.3, 1.0

    def _initial_step(self, cell: PolyhedralCell, z_star: np.ndarray, direction: np.ndarray, anchor: float) -> float:
        inside = lambda t: self.geo.slack(cell, self.geo.pose(z_star - t * direction)) >= -1e-9
        lo, hi = 0.0, 2.0 * cell.guard
        if inside(hi):
            lo = hi
        else:
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if inside(mid):
                    lo = mid
                else:
                    hi = mid
        self.stats["depth_sum"] = self.stats.get("depth_sum", 0.0) + lo
        return min(max(self.kappa * lo, self.eta_min), self.eta_max)


class AdaptiveStep05RRTStar(AdaptiveStepRRTStar):
    variant = "single_entry_adaptive_step05"
    kappa = 0.5


class AdaptiveStep15RRTStar(AdaptiveStepRRTStar):
    variant = "single_entry_adaptive_step15"
    kappa = 1.5


class SeedLaunchRRTStar(SingleEntryRRTStar):
    """Ablation of A: the step towards ``q_rand`` starts at the construction state ``q_i`` instead of
    ``Pi_{C_i}(q_rand)``; everything else is unchanged."""

    variant = "single_entry_seed_launch"

    def _launch(self, cell: PolyhedralCell, sample: Pose2D) -> Pose2D:
        return cell.pose


class SeedLaunchCoveredRRTStar(SeedLaunchRRTStar):
    """Auxiliary ablation: seed launch, but samples already inside ``C_i`` are skipped as in the full method, so the
    two differ only in where the step starts."""

    variant = "single_entry_seed_launch_covered"

    def _launch(self, cell: PolyhedralCell, sample: Pose2D) -> Pose2D:
        return sample if self.geo.slack(cell, sample) >= 0.0 else cell.pose


class RayLaunchRRTStar(SingleEntryRRTStar):
    """Ablation of A with a strong seed launch: leave ``C_i`` along the ray ``q_i -> q_rand`` and step ``eta`` from
    the exit point (samples inside ``C_i`` are skipped as in the full method).  It differs from A only in the
    boundary point the step starts from: ray exit instead of the closest point ``Pi_{C_i}(q_rand)``."""

    variant = "single_entry_ray_launch"

    def _launch(self, cell: PolyhedralCell, sample: Pose2D) -> Pose2D:
        anchor = cell.pose.yaw
        seed, target = self.geo.lift(cell.pose, anchor), self.geo.lift(sample, anchor)
        v = target - seed
        d = float(np.linalg.norm(v))
        if d < 1e-12 or self.geo.slack(cell, sample) >= 0.0:
            return sample
        u = v / d
        turn = abs(float(u[2]))
        reach = cell.guard / (math.hypot(u[0], u[1]) + turn)
        if turn > 0.0:
            reach = min(reach, cell.rho * cell.yaw_limit / turn)
        for nx, ny, e in _rows(cell):
            rate = turn - nx * u[0] - ny * u[1]
            if rate > 1e-15:
                reach = min(reach, e / rate)
        return self.geo.pose(seed + min(max(reach, 0.0), d) * u)


class HeadingHoldRRTStar(SingleEntryRRTStar):
    """A with heading-held expansion: ``c* = argmin ||c - c_rand||`` over the section of ``C_i`` at the entry yaw
    ``theta_i^e``, translational steps ``c* + eta u`` at that yaw; only when every step fails is the yaw moved
    towards ``theta_rand`` by at most ``max_turn`` and the steps retried."""

    variant = "single_entry_heading_hold"
    max_turn = math.pi / 4.0

    def _init_root(self) -> None:
        super()._init_root()
        self.stats.update(turn_retries=0, turn_successes=0)

    def _section_projection(self, cell: PolyhedralCell, target: Pose2D, yaw: float) -> np.ndarray:
        anchor = cell.pose.yaw
        t = self.geo.lift(Pose2D(0.0, 0.0, yaw), anchor)[2]
        p = np.array((target.x, target.y, t))
        a, b, cones = self.geo._constraints((cell,), anchor, 0.0)
        a = np.vstack((a, (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))
        b = np.concatenate((b, (t + 1e-9, -t + 1e-9)))
        if RegionGeometry._feasible(p, a, b, cones):
            return p
        z = self.geo._solve_projection(p, a, b, cones)
        if z is None:
            raise RuntimeError("section projection onto a valid cell failed")
        return z

    def _try_steps(self, cell, c_star, direction, length, yaw):
        step = self.config.step
        for attempt in range(1, self.config.steer_attempts + 1):
            c = c_star + min(step, length) * direction
            new = self.model.make_cell(Pose2D(float(c[0]), float(c[1]), yaw))
            self.stats["steer_cells"] += 1
            if new.valid and not self._separated(cell, new):
                portal = self._overlap(cell, new)
                if portal is not None:
                    return new, portal, attempt
            step *= 0.5
        return None, None, self.config.steer_attempts

    def _steer(self, parent: int, sample: Pose2D) -> tuple[PolyhedralCell | None, Pose2D | None, int]:
        cell = self.cells[parent]
        before = self.geo.solves
        started = time.perf_counter()
        self._refresh(parent)
        yaw = self.entry[parent].yaw
        z = self._section_projection(cell, sample, yaw)
        c_star = z[:2]
        self.stats["select_time_s"] += time.perf_counter() - started
        started = time.perf_counter()
        direction = np.array((sample.x, sample.y)) - c_star
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            self.stats["covered_samples"] += 1
            self.stats["core_solves"] += self.geo.solves - before
            self.stats["steer_time_s"] += time.perf_counter() - started
            return None, None, 0
        direction /= length
        result = self._try_steps(cell, c_star, direction, length, yaw)
        if result[0] is None:
            self.stats["turn_retries"] += 1
            turned = wrap_to_pi(yaw + max(-self.max_turn, min(self.max_turn, wrap_to_pi(sample.yaw - yaw))))
            result = self._try_steps(cell, c_star, direction, length, turned)
            self.stats["turn_successes"] += int(result[0] is not None)
        self.stats["core_solves"] += self.geo.solves - before
        self.stats["steer_time_s"] += time.perf_counter() - started
        new, _, attempt = result
        if new is None:
            self.stats["steer_failed"] += 1
        else:
            self.stats["steer_first_try"] += int(attempt == 1)
            self.stats["backoffs"] += attempt - 1
        return result


class LexTurnRRTStar(SingleEntryRRTStar):
    """A with node cost ``(J_i, Theta_i)``, ``Theta_i`` the accumulated turning of the entries along the tree path.
    ChooseParent / Rewire / goal choice: a clearly shorter candidate (``J_a (1 + eps_J) < J_b``) wins; otherwise,
    within the length tolerance, the one with less turning."""

    variant = "single_entry_lex_turn"
    eps_j = 0.01

    def _init_root(self) -> None:
        super()._init_root()
        self.turn: list[float] = [0.0]

    def _better(self, ja: float, ta: float, jb: float, tb: float) -> bool:
        if ja * (1.0 + self.eps_j) < jb:
            return True
        return ja <= jb * (1.0 + self.eps_j) and ta < tb - 1e-12

    def _turn(self, a: Pose2D, b: Pose2D) -> float:
        return abs(wrap_to_pi(b.yaw - a.yaw))

    def _refresh(self, index: int) -> None:
        chain = []
        cursor = index
        while self.parent[cursor] is not None:
            chain.append(cursor)
            cursor = self.parent[cursor]
        for node in reversed(chain):
            p = self.parent[node]
            if self.parent_version[node] == self.version[p]:
                continue
            z = self._attach(self.entry[p], self.cells[p], self.cells[node])
            if z is not None:
                self.entry[node] = z
            self.cost[node] = self.cost[p] + self.geo.distance(self.entry[p], self.entry[node])
            self.turn[node] = self.turn[p] + self._turn(self.entry[p], self.entry[node])
            self.parent_version[node] = self.version[p]
            self.version[node] += 1
            self.stats["repairs"] += 1

    def _connect(self, index: int, parent: int, portal: Pose2D) -> None:
        cell = self.cells[index]
        neighbors = self._neighbors(index)
        candidates = [parent] + neighbors
        for j in candidates:
            self._refresh(j)
        best, best_cost, best_turn, best_entry = None, math.inf, math.inf, None
        for j in sorted(candidates, key=lambda j: (j != parent, self.cost[j])):
            if self.cost[j] + self._bound(self.entry[j], self.cells[j], cell) > best_cost * (1.0 + self.eps_j):
                continue
            self.stats["choose_parent_tests"] += 1
            z = self._attach(self.entry[j], self.cells[j], cell)
            if z is None:
                continue
            c, t = self.cost[j] + self.geo.distance(self.entry[j], z), self.turn[j] + self._turn(self.entry[j], z)
            if self._better(c, t, best_cost, best_turn):
                self.stats["choose_parent_improved"] += int(best is not None)
                best, best_cost, best_turn, best_entry = j, c, t, z
        if best is None:
            best, best_entry = parent, portal
            best_cost = self.cost[parent] + self.geo.distance(self.entry[parent], portal)
            best_turn = self.turn[parent] + self._turn(self.entry[parent], portal)
        self.parent.append(best)
        self.entry.append(best_entry)
        self.cost.append(best_cost)
        self.turn.append(best_turn)
        self.version.append(0)
        self.parent_version.append(self.version[best])
        ancestors = set()
        cursor = best
        while cursor is not None:
            ancestors.add(cursor)
            cursor = self.parent[cursor]
        for k in neighbors + ([parent] if parent != best else []):
            if k in ancestors:
                continue
            self._refresh(k)
            if best_cost + self._bound(best_entry, cell, self.cells[k]) > self.cost[k] * (1.0 + self.eps_j):
                continue
            self.stats["rewire_tests"] += 1
            z = self._attach(best_entry, cell, self.cells[k])
            if z is None:
                continue
            c, t = best_cost + self.geo.distance(best_entry, z), best_turn + self._turn(best_entry, z)
            if self._better(c, t, self.cost[k], self.turn[k]):
                self.parent[k], self.entry[k], self.cost[k], self.turn[k] = index, z, c, t
                self.parent_version[k] = self.version[index]
                self.version[k] += 1
                self.stats["rewires"] += 1

    def _goal_value(self, g: int) -> tuple[float, float]:
        self._refresh(g)
        yaw = self.geo.goal_yaw(self.cells[g], self.goal_xy, self.entry[g].yaw, self.config.overlap_margin)
        goal = Pose2D(self.goal_xy[0], self.goal_xy[1], yaw)
        return self.cost[g] + self.geo.distance(self.entry[g], goal), self.turn[g] + self._turn(self.entry[g], goal)

    def _region_chain(self) -> tuple[list[int], float]:
        values = [self._goal_value(g) for g in self.goal_cells]
        k = 0
        for i in range(1, len(values)):
            if self._better(*values[i], *values[k]):
                k = i
        sequence, cursor = [], self.goal_cells[k]
        while cursor is not None:
            sequence.append(cursor)
            cursor = self.parent[cursor]
        return sequence[::-1], float(values[k][0])


class LexTurn5RRTStar(LexTurnRRTStar):
    variant = "single_entry_lex_turn5"
    eps_j = 0.05


class NoRewireRRTStar(SingleEntryRRTStar):
    """Ablation of A: no ChooseParent / Rewire.  The new region hangs on the region it was grown from, entering at
    the projection of that region's entry onto the intersection."""

    variant = "single_entry_no_rewire"

    def _connect(self, index: int, parent: int, portal: Pose2D) -> None:
        z = self._attach(self.entry[parent], self.cells[parent], self.cells[index])
        entry = portal if z is None else z
        self.parent.append(parent)
        self.entry.append(entry)
        self.cost.append(self.cost[parent] + self.geo.distance(self.entry[parent], entry))
        self.version.append(0)
        self.parent_version.append(self.version[parent])


class ProxyRRTStar(_CorePlanner):
    """Single-parent tree without entries: ChooseParent and Rewire compare the proxy cost
    ``J~_j + d_W(q_j, q_new)`` over regions that overlap at the margin; a rewire shifts the proxy cost of the whole
    subtree.  The path phase optimises the tree's region sequence once."""

    variant = "proxy_rrtstar"

    def _init_root(self) -> None:
        self.parent: list[int | None] = [None]
        self.children: list[set[int]] = [set()]
        self.cost: list[float] = [0.0]
        self.stats.update(choose_parent_tests=0, choose_parent_improved=0, rewire_tests=0, rewires=0)

    def _proxy(self, a: int, b: int) -> float:
        return self.geo.distance(self.cells[a].pose, self.cells[b].pose)

    def _connect(self, index: int, parent: int, portal: Pose2D) -> None:
        cell = self.cells[index]
        neighbors = self._neighbors(index)
        best, best_cost = parent, self.cost[parent] + self._proxy(parent, index)
        for j in sorted(neighbors, key=lambda j: self.cost[j] + self._proxy(j, index)):
            c = self.cost[j] + self._proxy(j, index)
            if c >= best_cost:
                break
            self.stats["choose_parent_tests"] += 1
            if self._overlap(self.cells[j], cell) is not None:
                self.stats["choose_parent_improved"] += 1
                best, best_cost = j, c
                break
        self.parent.append(best)
        self.children.append(set())
        self.children[best].add(index)
        self.cost.append(best_cost)
        ancestors = set()
        cursor: int | None = best
        while cursor is not None:
            ancestors.add(cursor)
            cursor = self.parent[cursor]
        for k in neighbors + ([parent] if parent != best else []):
            if k in ancestors:
                continue
            c = best_cost + self._proxy(index, k)
            if c >= self.cost[k] - 1e-9:
                continue
            self.stats["rewire_tests"] += 1
            if k != parent and self._overlap(cell, self.cells[k]) is None:
                continue
            self.children[self.parent[k]].discard(k)
            self.parent[k] = index
            self.children[index].add(k)
            shift = c - self.cost[k]
            stack = [k]
            while stack:
                node = stack.pop()
                self.cost[node] += shift
                stack.extend(self.children[node])
            self.stats["rewires"] += 1

    def _goal_cost(self, g: int) -> float:
        pose = self.cells[g].pose
        yaw = self.geo.goal_yaw(self.cells[g], self.goal_xy, pose.yaw, self.config.overlap_margin)
        return self.cost[g] + self.geo.distance(pose, Pose2D(self.goal_xy[0], self.goal_xy[1], yaw))

    def _region_chain(self) -> tuple[list[int], float]:
        costs = [self._goal_cost(g) for g in self.goal_cells]
        k = int(np.argmin(costs))
        sequence, cursor = [], self.goal_cells[k]
        while cursor is not None:
            sequence.append(cursor)
            cursor = self.parent[cursor]
        return sequence[::-1], float(costs[k])


class PortalRRTStar(_CorePlanner):
    """Single-parent tree whose entry ``w_i`` is the portal the connection test already returns (no optimisation):
    ``J~_new = J~_j + d_W(w_j, w_{j,new})``.  Both points lie in ``C_j``, so ``J~_i`` is the length of a feasible
    polyline start -> w -> ... -> w_i.  A rewire changes ``w_k``: direct children recompute their first leg and
    shift their subtrees."""

    variant = "portal_rrtstar"

    def _init_root(self) -> None:
        self.parent: list[int | None] = [None]
        self.children: list[set[int]] = [set()]
        self.entry: list[Pose2D] = [self.start]
        self.cost: list[float] = [0.0]
        self.stats.update(choose_parent_tests=0, choose_parent_improved=0, rewire_tests=0, rewires=0)

    def _bound(self, entry: Pose2D, cell: PolyhedralCell) -> float:
        return max(0.0, self.config.overlap_margin - self.geo.slack(cell, entry)) / math.sqrt(2.0)

    def _shift(self, root: int, delta: float) -> None:
        stack = [root]
        while stack:
            node = stack.pop()
            self.cost[node] += delta
            stack.extend(self.children[node])

    def _connect(self, index: int, parent: int, portal: Pose2D) -> None:
        cell = self.cells[index]
        neighbors = self._neighbors(index)
        best, best_entry = parent, portal
        best_cost = self.cost[parent] + self.geo.distance(self.entry[parent], portal)
        for j in sorted(neighbors, key=lambda j: self.cost[j]):
            if self.cost[j] + self._bound(self.entry[j], cell) >= best_cost:
                continue
            self.stats["choose_parent_tests"] += 1
            w = self._overlap(self.cells[j], cell)
            if w is None:
                continue
            c = self.cost[j] + self.geo.distance(self.entry[j], w)
            if c < best_cost:
                self.stats["choose_parent_improved"] += 1
                best, best_cost, best_entry = j, c, w
        self.parent.append(best)
        self.children.append(set())
        self.children[best].add(index)
        self.entry.append(best_entry)
        self.cost.append(best_cost)
        ancestors = set()
        cursor: int | None = best
        while cursor is not None:
            ancestors.add(cursor)
            cursor = self.parent[cursor]
        for k in neighbors + ([parent] if parent != best else []):
            if k in ancestors or best_cost + self._bound(best_entry, self.cells[k]) >= self.cost[k] - 1e-9:
                continue
            self.stats["rewire_tests"] += 1
            w = portal if k == parent else self._overlap(cell, self.cells[k])
            if w is None:
                continue
            c = best_cost + self.geo.distance(best_entry, w)
            if c >= self.cost[k] - 1e-9:
                continue
            self.children[self.parent[k]].discard(k)
            self.parent[k], self.entry[k], self.cost[k] = index, w, c
            self.children[index].add(k)
            for child in self.children[k]:
                self._shift(child, c + self.geo.distance(w, self.entry[child]) - self.cost[child])
            self.stats["rewires"] += 1

    def _goal_cost(self, g: int) -> float:
        yaw = self.geo.goal_yaw(self.cells[g], self.goal_xy, self.entry[g].yaw, self.config.overlap_margin)
        return self.cost[g] + self.geo.distance(self.entry[g], Pose2D(self.goal_xy[0], self.goal_xy[1], yaw))

    def _region_chain(self) -> tuple[list[int], float]:
        costs = [self._goal_cost(g) for g in self.goal_cells]
        k = int(np.argmin(costs))
        sequence, cursor = [], self.goal_cells[k]
        while cursor is not None:
            sequence.append(cursor)
            cursor = self.parent[cursor]
        return sequence[::-1], float(costs[k])


class AnchorRRTStar(_CorePlanner):
    """Single-parent tree whose cost ``J~_i`` is the length of a certified path to the seed ``q_i``.  Edge cost:
    ``d_W(q_i, q_j)`` when the seed segment meets ``C_i^delta ∩ C_j^delta`` (then by convexity it lies in
    ``C_i ∪ C_j``; with seeds in ``C^delta`` this is ``r_i^+ + r_j^- >= d``), else ``d_W(q_i, w) + d_W(w, q_j)``
    through the overlap witness ``w``.  Edge costs never change, so a rewire shifts the whole subtree by one
    constant."""

    variant = "anchor_rrtstar"

    def _init_root(self) -> None:
        self.parent: list[int | None] = [None]
        self.children: list[set[int]] = [set()]
        self.cost: list[float] = [0.0]
        self.stats.update(choose_parent_tests=0, choose_parent_improved=0, rewire_tests=0, rewires=0,
                          edges_direct=0, edges_detour=0)

    def _lift(self, i: int) -> np.ndarray:
        return self.geo.lift(self.cells[i].pose, self.cells[i].pose.yaw)

    def _interval(self, cell: PolyhedralCell, u: np.ndarray) -> tuple[float, float]:
        """``{alpha >= 0 : seed + alpha u in C^delta}`` (``u`` in z).  The seed itself may have slack below delta,
        so the interval need not start at 0."""
        delta = self.config.overlap_margin
        turn = abs(float(u[2]))
        lo, hi = 0.0, (cell.guard - delta) / (math.hypot(u[0], u[1]) + turn)
        if turn > 0.0:
            hi = min(hi, (cell.rho * cell.yaw_limit - delta) / turn)
        for nx, ny, e in _rows(cell):
            rate = turn - nx * u[0] - ny * u[1]
            if rate > 1e-15:
                hi = min(hi, (e - delta) / rate)
            elif rate < -1e-15:
                lo = max(lo, (e - delta) / rate)
            elif e < delta:
                return math.inf, -math.inf
        return lo, hi

    def _edge(self, i: int, j: int, witness: Pose2D | None = None) -> float | None:
        """Certified path length ``q_i -> q_j`` through ``C_i ∪ C_j``, or ``None`` if the regions do not connect."""
        a, b = self.cells[i].pose, self.cells[j].pose
        v = np.array((b.x - a.x, b.y - a.y, self.rho * wrap_to_pi(b.yaw - a.yaw)))
        d = float(np.linalg.norm(v))
        if d < 1e-12:
            return 0.0
        u = v / d
        lo_i, hi_i = self._interval(self.cells[i], u)
        lo_j, hi_j = self._interval(self.cells[j], -u)
        if max(lo_i, d - hi_j, 0.0) <= min(hi_i, d - lo_j, d):
            self.stats["edges_direct"] += 1
            return d
        if witness is None:
            witness = self._overlap(self.cells[i], self.cells[j])
            if witness is None:
                return None
        self.stats["edges_detour"] += 1
        return self.geo.distance(a, witness) + self.geo.distance(witness, b)

    def _connect(self, index: int, parent: int, portal: Pose2D) -> None:
        neighbors = self._neighbors(index)
        best, best_cost = parent, self.cost[parent] + self._edge(parent, index, portal)
        seed = self.cells[index].pose
        for j in sorted(neighbors, key=lambda j: self.cost[j] + self.geo.distance(self.cells[j].pose, seed)):
            if self.cost[j] + self.geo.distance(self.cells[j].pose, seed) >= best_cost:
                break
            self.stats["choose_parent_tests"] += 1
            c = self._edge(j, index)
            if c is not None and self.cost[j] + c < best_cost:
                self.stats["choose_parent_improved"] += 1
                best, best_cost = j, self.cost[j] + c
        self.parent.append(best)
        self.children.append(set())
        self.children[best].add(index)
        self.cost.append(best_cost)
        ancestors = set()
        cursor: int | None = best
        while cursor is not None:
            ancestors.add(cursor)
            cursor = self.parent[cursor]
        for k in neighbors + ([parent] if parent != best else []):
            if k in ancestors or best_cost + self.geo.distance(seed, self.cells[k].pose) >= self.cost[k] - 1e-9:
                continue
            self.stats["rewire_tests"] += 1
            c = self._edge(index, k, portal if k == parent else None)
            if c is None or best_cost + c >= self.cost[k] - 1e-9:
                continue
            self.children[self.parent[k]].discard(k)
            self.parent[k] = index
            self.children[index].add(k)
            shift = best_cost + c - self.cost[k]
            stack = [k]
            while stack:
                node = stack.pop()
                self.cost[node] += shift
                stack.extend(self.children[node])
            self.stats["rewires"] += 1

    def _goal_cost(self, g: int) -> float:
        pose = self.cells[g].pose
        yaw = self.geo.goal_yaw(self.cells[g], self.goal_xy, pose.yaw, self.config.overlap_margin)
        return self.cost[g] + self.geo.distance(pose, Pose2D(self.goal_xy[0], self.goal_xy[1], yaw))

    def _region_chain(self) -> tuple[list[int], float]:
        costs = [self._goal_cost(g) for g in self.goal_cells]
        k = int(np.argmin(costs))
        sequence, cursor = [], self.goal_cells[k]
        while cursor is not None:
            sequence.append(cursor)
            cursor = self.parent[cursor]
        return sequence[::-1], float(costs[k])


class CostNearRRTStar(SingleEntryRRTStar):
    """A with cost-guided region selection: with probability ``p_cost`` the ``k`` smallest region gaps are
    candidates, each is projected, ``x_i = Pi_{C_i}(q_rand)``, and the one minimising
    ``S_i = J_i + d_W(e_i, x_i) + d_W(x_i, q_rand)`` is expanded from ``x_i``; otherwise A's gap rule.  The choice
    uses its own random stream, so the samples are those of A."""

    variant = "single_entry_cost_near"
    k = 3
    p_cost = 0.8
    after_first = False
    """Use A's gap rule until the first goal region exists."""
    guided_iterations: tuple[int, float] = (0, math.inf)
    """Cost-guided selection only for iterations ``lo < it <= hi`` (switch at a fixed sample count)."""

    def _init_root(self) -> None:
        super()._init_root()
        self.choice_rng = np.random.default_rng((self.config.seed, 7))
        self._launch_cache: tuple[int, Pose2D, Pose2D] | None = None
        self._iteration = 0
        self.stats.update(cost_selects=0, cost_changed=0, cost_projections=0, cost_pruned=0)

    def _guided(self) -> bool:
        lo, hi = self.guided_iterations
        return lo < self._iteration <= hi and not (self.after_first and not self.goal_cells)

    def _select(self, sample: Pose2D) -> int:
        self._launch_cache = None
        self._iteration += 1
        if self.choice_rng.random() >= self.p_cost or not self._guided():
            return super()._select(sample)
        idx, gap = self._gap_candidates(sample, self.k)
        order = idx[np.argsort(gap, kind="stable")[:self.k]].tolist()
        # S_i >= J_i + d_W(e_i, q_rand): candidates whose bound exceeds the best score need no projection.
        bounds = []
        for rank, i in enumerate(order):
            self._refresh(i)
            bounds.append((self.cost[i] + self.geo.distance(self.entry[i], sample), rank, i))
        best, best_key, best_x = None, (math.inf, math.inf), None
        for bound, rank, i in sorted(bounds):
            if bound > best_key[0] + 1e-12:
                self.stats["cost_pruned"] += 1
                continue
            x = self.geo.project(self.cells[i], sample)
            key = (self.cost[i] + self.geo.distance(self.entry[i], x) + self.geo.distance(x, sample), rank)
            self.stats["cost_projections"] += 1
            if key < best_key:
                best, best_key, best_x = i, key, x
        self.stats["cost_selects"] += 1
        self.stats["cost_changed"] += int(best != order[0])
        self._launch_cache = (best, sample, best_x)
        return best

    def _launch(self, cell: PolyhedralCell, sample: Pose2D) -> Pose2D:
        cache = self._launch_cache
        if cache is not None and cache[1] is sample and self.cells[cache[0]] is cell:
            return cache[2]
        return super()._launch(cell, sample)


class CostNear5RRTStar(CostNearRRTStar):
    variant = "single_entry_cost_near5"
    k = 5


class CostNearMixRRTStar(CostNearRRTStar):
    variant = "single_entry_cost_near_mix"
    p_cost = 0.5


class CostNearFullRRTStar(CostNearRRTStar):
    variant = "single_entry_cost_near_full"
    p_cost = 1.0


class CostNear5FullRRTStar(CostNearRRTStar):
    variant = "single_entry_cost_near5_full"
    k = 5
    p_cost = 1.0


class CostNear5AfterRRTStar(CostNearRRTStar):
    variant = "single_entry_cost_near5_after"
    k = 5
    after_first = True


class CostNear5FullAfterRRTStar(CostNearRRTStar):
    variant = "single_entry_cost_near5_full_after"
    k = 5
    p_cost = 1.0
    after_first = True


def _switch_variant(n: int, guided_first: bool) -> type:
    """Cost-Near-5 (p 0.8) for the first ``n`` iterations then A's rule, or A first and cost-guided after."""
    name = f"single_entry_cost{n}_then_a" if guided_first else f"single_entry_a{n}_then_cost"
    window = (0, n) if guided_first else (n, math.inf)
    return type(f"Switch{n}{'CostA' if guided_first else 'ACost'}RRTStar", (CostNearRRTStar,),
                dict(variant=name, k=5, guided_iterations=window))


SWITCH_VARIANTS = tuple(_switch_variant(n, first) for n in (100, 250, 500) for first in (True, False))


class NormalTurnRRTStar(SingleEntryRRTStar):
    """A on ``NormalTurnCellModel`` cells: each obstacle row bounds only the normal component of its robot's
    rotation displacement (``C_old ⊆ C_new`` for the same seed); everything else as in A."""

    variant = "single_entry_normal_turn"
    cell_model = NormalTurnCellModel


PLANNERS = {cls.variant: cls for cls in (SingleEntryRRTStar, SeedLaunchRRTStar, SeedLaunchCoveredRRTStar,
                                         RayLaunchRRTStar, HeadingHoldRRTStar, AncestorRRTStar, EntryRayRRTStar, AncestorEntryRayRRTStar, ConsistentRRTStar,
                                    ConsistentRewireRRTStar, ReferenceParentRRTStar, AdaptiveStepRRTStar,
                                    AdaptiveStep05RRTStar, AdaptiveStep15RRTStar, LexTurnRRTStar, LexTurn5RRTStar, NoRewireRRTStar, ProxyRRTStar, PortalRRTStar,
                                         AnchorRRTStar, NormalTurnRRTStar, CostNearRRTStar, CostNear5RRTStar,
                                         CostNearMixRRTStar, CostNearFullRRTStar, CostNear5FullRRTStar,
                                         CostNear5AfterRRTStar, CostNear5FullAfterRRTStar) + SWITCH_VARIANTS}
