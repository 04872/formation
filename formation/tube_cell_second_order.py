"""Second-order formation cells.

    U_k = {(u, phi) : ||u + J a_i phi|| + beta_i phi^2 < d_k  for all i},   beta_i = ||r_i|| / 2,

with ``d_k = eta * d_obs(q_k)``.  Since ``(R(phi) - I) r_i = J r_i phi + eps_i(phi)`` and
``||eps_i(phi)|| <= ||r_i|| phi^2 / 2``, every ``q`` in ``U_k`` moves every robot by less than
``d_k <= d_obs(q_k)``: the cell is collision free, not just to first order.

Two cells overlap iff some ``q = (x, y, theta)`` lies in both.  In the chart of cell ``a`` this is
the SOCP

    max s  s.t.  ||u + J a_i phi|| + beta_i z_a <= d_a (1 - s),
                 ||u - Δ + J b_i (phi - δ)|| + beta_i z_b <= d_b (1 - s),
                 z_a >= phi^2,  z_b >= (phi - δ)^2   (rotated second-order cones),

and the cells overlap iff ``s* > 0``; the maximiser is the portal ``q_p``.  Cheap necessary
bounds reject most pairs and a few candidate portals on the chart segment accept most of the
rest, so the solver (Clarabel) only runs on the undecided pairs.
"""
from __future__ import annotations

import math

import clarabel
import numpy as np
import scipy.sparse as sp

from formation.tube_cell_first_order import PortalCertificate, FirstOrderCellModel, TubeCell, interpolate_pose
from formation.types import Pose2D, wrap_to_pi

_SEGMENT_CANDIDATES = (0.5, 0.25, 0.75)


class SecondOrderCellModel(FirstOrderCellModel):
    name = "second_order"

    def __init__(self, slots: np.ndarray, eta: float = 0.98) -> None:
        super().__init__(slots, eta)
        count = len(self._slots)
        self._cones = [clarabel.NonnegativeConeT(4)] + [clarabel.SecondOrderConeT(3)] * (2 * count + 2)
        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False
        self._objective = np.zeros(6)
        self._objective[5] = -1.0
        self._hessian = sp.csc_matrix((6, 6))

    @staticmethod
    def _beta(slots: np.ndarray) -> np.ndarray:
        return 0.5 * np.linalg.norm(slots, axis=1)

    def phi_limit(self, radius: float) -> float:
        limit = super().phi_limit(radius)
        return min(limit, math.sqrt(radius / self._beta_max)) if self._beta_max > 0.0 else limit

    def inner_step(self, cell: TubeCell, target: Pose2D) -> float:
        # Along the ray every term is <= s + beta_max (c s)^2 with c = |phi| / ||.||_F per unit length.
        center = cell.pose
        length = self.distance(center, target)
        if length <= 0.0 or cell.radius <= 0.0:
            return cell.radius
        rate = self._beta_max * (abs(wrap_to_pi(target.yaw - center.yaw)) / length) ** 2
        return 2.0 * cell.radius / (1.0 + math.sqrt(1.0 + 4.0 * rate * cell.radius))

    def _robots_within(self, a: Pose2D, b: Pose2D, reach: float) -> bool:
        """Necessary for overlap: a common q moves each robot < d_a from q_a and < d_b from q_b."""
        ca, sa, cb, sb = math.cos(a.yaw), math.sin(a.yaw), math.cos(b.yaw), math.sin(b.yaw)
        limit = reach * reach
        for rx, ry in self._slots:
            ex = b.x + cb * rx - sb * ry - a.x - ca * rx + sa * ry
            ey = b.y + sb * rx + cb * ry - a.y - sa * rx - ca * ry
            if ex * ex + ey * ey >= limit:
                return False
        return True

    def _slack_offset(self, cell: TubeCell, dx: float, dy: float, phi: float) -> float:
        return 1.0 - max(self._terms(dx, dy, phi, cell.pose.yaw)) / cell.radius

    def overlap(self, first: TubeCell, second: TubeCell) -> PortalCertificate | None:
        self.stats["overlap_calls"] += 1
        d_a, d_b = first.radius, second.radius
        if d_a <= 0.0 or d_b <= 0.0:
            return None
        a, b = first.pose, second.pose
        dx, dy, delta = b.x - a.x, b.y - a.y, wrap_to_pi(b.yaw - a.yaw)
        limit_a, limit_b = self.phi_limit(d_a), self.phi_limit(d_b)
        if abs(delta) >= limit_a + limit_b or not self._robots_within(a, b, d_a + d_b):
            self.stats["quick_reject"] += 1
            return None

        best_t, best_slack = None, 0.0
        for t in (1.0, 0.0, *_SEGMENT_CANDIDATES):
            slack_a = 1.0 if t == 0.0 else self._slack_offset(first, t * dx, t * dy, t * delta)
            if slack_a <= best_slack:
                continue
            u = t - 1.0
            slack_b = 1.0 if t == 1.0 else self._slack_offset(second, u * dx, u * dy, u * delta)
            slack = min(slack_a, slack_b)
            if slack > best_slack:
                best_t, best_slack = t, slack
        if best_t is not None:
            self.stats["quick_accept"] += 1
            portal = interpolate_pose(a, b, best_t)
            return self._certificate(first, second, portal, best_slack)
        return self._solve_overlap(first, second, dx, dy, delta, limit_a, limit_b)

    def _solve_overlap(self, first: TubeCell, second: TubeCell, dx: float, dy: float, delta: float,
                       limit_a: float, limit_b: float) -> PortalCertificate | None:
        self.stats["socp_calls"] += 1
        d_a, d_b = first.radius, second.radius
        # x = (u_x, u_y, phi, z_a, z_b, s) in the chart of ``first``; rows follow the cone order.
        rows: list[tuple[float, float, float, float, float, float]] = [
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
        ]
        rhs = [limit_a, limit_a, limit_b + delta, limit_b - delta]
        for cell, z_column, offset_x, offset_y, shift in ((first, 3, 0.0, 0.0, 0.0), (second, 4, dx, dy, delta)):
            c, s = math.cos(cell.pose.yaw), math.sin(cell.pose.yaw)
            for (rx, ry), beta in zip(self._slots, self._betas):
                ax, ay = c * rx - s * ry, s * rx + c * ry
                cone_t = [0.0] * 6
                cone_t[z_column], cone_t[5] = beta, cell.radius
                rows += [tuple(cone_t), (-1.0, 0.0, ay, 0.0, 0.0, 0.0), (0.0, -1.0, -ax, 0.0, 0.0, 0.0)]
                rhs += [cell.radius, -offset_x + ay * shift, -offset_y - ax * shift]
        for z_column, shift in ((3, 0.0), (4, delta)):
            lifted = [0.0] * 6
            lifted[z_column] = -1.0
            rows += [tuple(lifted), (0.0, 0.0, -2.0, 0.0, 0.0, 0.0), tuple(lifted)]
            rhs += [1.0, -2.0 * shift, -1.0]
        solver = clarabel.DefaultSolver(self._hessian, self._objective, sp.csc_matrix(np.asarray(rows)),
                                        np.asarray(rhs), self._cones, self._settings)
        solution = solver.solve()
        if str(solution.status) not in ("Solved", "AlmostSolved") or -solution.obj_val <= 0.0:
            return None
        ux, uy, phi = solution.x[0], solution.x[1], solution.x[2]
        slack = min(self._slack_offset(first, ux, uy, phi),
                    self._slack_offset(second, ux - dx, uy - dy, phi - delta))
        if slack <= 1e-9:
            return None
        self.stats["socp_accept"] += 1
        a = first.pose
        portal = Pose2D(a.x + ux, a.y + uy, wrap_to_pi(a.yaw + phi))
        return self._certificate(first, second, portal, slack)
