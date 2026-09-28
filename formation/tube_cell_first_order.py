"""First-order formation cells.

For a node ``q_k = (x_k, y_k, theta_k)`` and a query ``q`` write the local chart
``xi = (u, phi)`` with ``u = q.xy - q_k.xy`` and ``phi = wrap(q.theta - theta_k)``.
With world-frame slots ``a_i = R(theta_k) r_i`` and ``J`` the 90 degree rotation, the
first-order displacement of robot ``i`` is ``u + J a_i phi`` and

    ||xi||_F = max_i ||u + J a_i phi||,      U_k = {q : ||q - q_k||_F < rho_k},

with ``rho_k = eta * d_obs(q_k)``.  ``||u + J a_i phi|| = ||R(-theta_k) u + J r_i phi||``,
so the norm is evaluated in the body frame of the chart node.

Cells of different nodes use different charts (``a_i`` depends on ``theta_k``).  Along the
chart segment ``q(t) = q_a + t (q_b - q_a)`` the two norms scale linearly, so the
homothetic overlap ``||q_a - q_b||_F < rho_a + rho_b`` generalises to

    rho_a / n_a + rho_b / n_b > 1,   n_a = ||q_b - q_a||_{F,a},  n_b = ||q_a - q_b||_{F,b},

which is exactly the homothetic test when ``theta_a = theta_b`` and a sufficient test otherwise.
The first-order cell neglects the ``(R(phi) - I) r_i - J r_i phi`` remainder, so it is only
approximately collision free; see ``tube_cell_second_order`` for the certified version.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from formation.types import Pose2D, wrap_to_pi


@dataclass(frozen=True)
class TubeCell:
    pose: Pose2D
    clearance: float
    radius: float
    """``rho_k`` of a first-order cell or ``d_k`` of a second-order cell."""


@dataclass(frozen=True)
class PortalCertificate:
    """Portal ``q_p`` in both cells; the certified edge route is ``q_a -> q_p -> q_b``."""

    portal: Pose2D
    slack: float
    """Smallest relative slack ``1 - f(q_p) / radius`` over the two cells."""
    route_length: float
    """``||q_p - q_a||_{F,a} + ||q_b - q_p||_{F,b}``."""


def interpolate_pose(first: Pose2D, second: Pose2D, alpha: float) -> Pose2D:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return Pose2D(first.x + alpha * (second.x - first.x),
                  first.y + alpha * (second.y - first.y),
                  wrap_to_pi(first.yaw + alpha * wrap_to_pi(second.yaw - first.yaw)))


class FirstOrderCellModel:
    name = "first_order"

    def __init__(self, slots: np.ndarray, eta: float = 0.98) -> None:
        slots = np.asarray(slots, dtype=float)
        if not 0.0 < eta <= 1.0:
            raise ValueError("cell eta must lie in (0, 1]")
        self.eta = float(eta)
        self.slots = slots
        self._slot_x = slots[:, 0].copy()
        self._slot_y = slots[:, 1].copy()
        self._slots = [(float(x), float(y)) for x, y in slots]
        self.beta = self._beta(slots)
        self._betas = [float(value) for value in self.beta]
        self._beta_max = float(np.max(self.beta)) if len(self.beta) else 0.0
        spread = max((float(np.linalg.norm(a - b)) for i, a in enumerate(slots) for b in slots[i + 1:]), default=0.0)
        if spread <= 0.0:
            raise ValueError("formation needs at least two distinct slots")
        # |phi| * spread <= ||u + J a_i phi|| + ||u + J a_j phi||, so |phi| < 2 rho / spread in a cell.
        # Capping rho at pi * spread / 2 keeps every cell inside its (-pi, pi) yaw chart.
        self._spread = spread
        self._radius_cap = math.pi * spread / 2.0 * (1.0 - 1e-9)
        self.stats = {"overlap_calls": 0, "quick_reject": 0, "quick_accept": 0, "socp_calls": 0, "socp_accept": 0}

    @staticmethod
    def _beta(slots: np.ndarray) -> np.ndarray:
        return np.zeros(len(slots))

    def make_cell(self, pose: Pose2D, clearance: float) -> TubeCell:
        return TubeCell(pose, float(clearance), min(self.eta * max(float(clearance), 0.0), self._radius_cap))

    def phi_limit(self, radius: float) -> float:
        return min(2.0 * radius / self._spread, math.pi)

    # -- chart norms -----------------------------------------------------------------------------
    def _terms(self, dx: float, dy: float, phi: float, chart_yaw: float) -> list[float]:
        """``||u + J a_i phi|| + beta_i phi^2`` for every robot (pure Python: N is tiny)."""
        c, s = math.cos(chart_yaw), math.sin(chart_yaw)
        vx, vy = c * dx + s * dy, -s * dx + c * dy
        phi2 = phi * phi
        return [math.hypot(vx - ry * phi, vy + rx * phi) + beta * phi2
                for (rx, ry), beta in zip(self._slots, self._betas)]

    def norm(self, dx: float, dy: float, phi: float, chart_yaw: float) -> float:
        c, s = math.cos(chart_yaw), math.sin(chart_yaw)
        vx, vy = c * dx + s * dy, -s * dx + c * dy
        return max(math.hypot(vx - ry * phi, vy + rx * phi) for rx, ry in self._slots)

    def distance(self, first: Pose2D, second: Pose2D) -> float:
        """``||second - first||_F`` in the chart of ``first``."""
        return self.norm(second.x - first.x, second.y - first.y, wrap_to_pi(second.yaw - first.yaw), first.yaw)

    def distances_to(self, xs: np.ndarray, ys: np.ndarray, cos_yaws: np.ndarray, sin_yaws: np.ndarray,
                     yaws: np.ndarray, pose: Pose2D) -> np.ndarray:
        """``||pose - q_k||_{F,k}`` for many chart nodes ``k`` at once (nearest / neighbour queries)."""
        # ||v + J r_i phi||^2 = ||u||^2 + phi^2 ||r_i||^2 + 2 phi (r_i x v), with v = R(-theta_k) u.
        dx, dy = pose.x - xs, pose.y - ys
        phi = np.remainder(pose.yaw - yaws + math.pi, 2.0 * math.pi) - math.pi
        px = phi * (cos_yaws * dx + sin_yaws * dy)
        py = phi * (cos_yaws * dy - sin_yaws * dx)
        phi *= phi
        best = None
        for rx, ry in self._slots:
            value = (rx * rx + ry * ry) * phi + 2.0 * (rx * py - ry * px)
            best = value if best is None else np.maximum(best, value, out=best)
        best += dx * dx + dy * dy
        return np.sqrt(np.maximum(best, 0.0, out=best), out=best)

    def slack(self, cell: TubeCell, pose: Pose2D) -> float:
        """Relative slack ``1 - max_i f_i / radius``; positive exactly when ``pose`` is in the cell."""
        if cell.radius <= 0.0:
            return -math.inf
        center = cell.pose
        terms = self._terms(pose.x - center.x, pose.y - center.y, wrap_to_pi(pose.yaw - center.yaw), center.yaw)
        return 1.0 - max(terms) / cell.radius

    def inner_step(self, cell: TubeCell, target: Pose2D) -> float:
        """F-norm length along the chart ray towards ``target`` that stays strictly inside ``cell``."""
        return cell.radius

    # -- overlap -----------------------------------------------------------------------------------
    def _certificate(self, first: TubeCell, second: TubeCell, portal: Pose2D, slack: float) -> PortalCertificate:
        route = self.distance(first.pose, portal) + self.distance(portal, second.pose)
        return PortalCertificate(portal, float(slack), float(route))

    def overlap(self, first: TubeCell, second: TubeCell) -> PortalCertificate | None:
        self.stats["overlap_calls"] += 1
        if first.radius <= 0.0 or second.radius <= 0.0:
            return None
        n_a = self.distance(first.pose, second.pose)
        n_b = self.distance(second.pose, first.pose)
        if n_a <= 1e-15 or n_b <= 1e-15:
            return PortalCertificate(first.pose, 1.0, 0.0)
        share_a, share_b = first.radius / n_a, second.radius / n_b
        ratio = share_a + share_b
        if ratio <= 1.0:
            return None
        t = share_a / ratio
        portal = interpolate_pose(first.pose, second.pose, t)
        return PortalCertificate(portal, 1.0 - 1.0 / ratio, t * n_a + (1.0 - t) * n_b)

    # -- drawing helpers ---------------------------------------------------------------------------
    def slice_discs(self, cell: TubeCell, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Fixed-yaw xy slice ``D_k(theta) = cap_i B(q_k.xy - J a_i phi, radius - beta_i phi^2)``."""
        phi = wrap_to_pi(theta - cell.pose.yaw)
        c, s = math.cos(cell.pose.yaw), math.sin(cell.pose.yaw)
        ax, ay = c * self._slot_x - s * self._slot_y, s * self._slot_x + c * self._slot_y
        centers = np.column_stack((cell.pose.x + ay * phi, cell.pose.y - ax * phi))
        return centers, cell.radius - self.beta * phi * phi

    def slice_outline(self, cell: TubeCell, theta: float, samples: int = 90) -> np.ndarray | None:
        """Boundary polygon of the fixed-yaw slice, or ``None`` if the slice is empty."""
        centers, radii = self.slice_discs(cell, theta)
        if np.any(radii <= 0.0):
            return None
        candidates = np.vstack((centers.mean(axis=0), centers))
        depth = np.min(radii[None, :] - np.linalg.norm(candidates[:, None, :] - centers[None, :, :], axis=2), axis=1)
        if depth.max() <= 0.0:
            return None
        origin = candidates[int(np.argmax(depth))]
        angles = np.linspace(0.0, 2.0 * math.pi, samples)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        offset = origin[None, :] - centers
        along = directions @ offset.T
        reach = -along + np.sqrt(np.maximum(along * along - np.sum(offset * offset, axis=1) + radii * radii, 0.0))
        return origin + directions * reach.min(axis=1)[:, None]
