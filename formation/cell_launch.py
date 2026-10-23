"""Current-cell constraint release and lexicographic convex launch selection.

All geometry is taken from the selected certified cell. No hypothetical cells,
volume integrals, finite differences or nonconvex launch optimisation are used.
The local chart is z=(dx,dy,rho*dtheta), so W distances are Euclidean here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import clarabel
import numpy as np
import scipy.sparse as sp

from formation.polyhedral_cell import PolyhedralCell, PolyhedralCellModel
from formation.types import Pose2D, wrap_to_pi


@dataclass
class LaunchConfig:
    internal_distance: float = 1.0
    threshold: float = 0.02
    facet_epsilon: float = 1e-4
    progress_epsilon: float = 0.05


class CellLaunchSearch:
    def __init__(self, cells: PolyhedralCellModel, config: LaunchConfig) -> None:
        self.cells, self.config = cells, config
        self.scale = np.array((1., 1., cells.rho))
        self.last_solve = {}
        self.support_queries = 0

    def offset(self, origin: Pose2D, pose: Pose2D) -> np.ndarray:
        return np.array((pose.x-origin.x, pose.y-origin.y,
                         wrap_to_pi(pose.yaw-origin.yaw))) * self.scale

    def pose(self, origin: Pose2D, z: np.ndarray) -> Pose2D:
        d = z / self.scale
        return Pose2D(origin.x+float(d[0]), origin.y+float(d[1]), wrap_to_pi(origin.yaw+float(d[2])))

    @staticmethod
    def facet_rows(cell):
        n = len(cell.offsets)
        rows = np.vstack((np.column_stack((-cell.normals, np.ones(n))),
                          np.column_stack((-cell.normals, -np.ones(n)))))
        return rows, np.tile(cell.offsets, 2)

    def _solve(self, cell, objective, entry_z=None, *, projection=False, omit=None, progress=None, sample_ball=None, intersection=None):
        rows, rhs = self.facet_rows(cell)
        if omit is not None:
            keep = np.arange(len(rhs)) != omit
            rows, rhs = rows[keep], rhs[keep]
        rows = np.vstack((rows, [0., 0., 1.], [0., 0., -1.]))
        rhs = np.r_[rhs, cell.rho*cell.yaw_limit, cell.rho*cell.yaw_limit]
        if progress is not None:
            facet, minimum = progress
            rows = np.vstack((rows, -facet))
            rhs = np.r_[rhs, -minimum]
        # guard: ||z_xy|| <= guard +/- z_theta; all guards stay in support queries.
        guard = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        other = guard.copy()
        other[0, 2] = -1.
        blocks = [rows, guard, other]
        bounds = [rhs, np.array([cell.guard, 0., 0.]), np.array([cell.guard, 0., 0.])]
        cones = [clarabel.NonnegativeConeT(len(rhs)), clarabel.SecondOrderConeT(3),
                 clarabel.SecondOrderConeT(3)]
        if entry_z is not None:
            blocks.append(np.vstack((np.zeros(3), -np.eye(3))))
            bounds.append(np.r_[self.config.internal_distance, -entry_z])
            cones.append(clarabel.SecondOrderConeT(4))
        if sample_ball is not None:
            center, radius = sample_ball
            blocks.append(np.vstack((np.zeros(3), -np.eye(3))))
            bounds.append(np.r_[radius, -center])
            cones.append(clarabel.SecondOrderConeT(4))
        if intersection is not None:
            second, shift = intersection
            r, b = self.facet_rows(second)
            r = np.vstack((r, [0., 0., 1.], [0., 0., -1.]))
            b = np.r_[b, second.rho*second.yaw_limit, second.rho*second.yaw_limit]
            blocks.extend((r, guard, other))
            bounds.extend((b+r@shift, np.array([second.guard, 0., 0.])+guard@shift,
                           np.array([second.guard, 0., 0.])+other@shift))
            cones.extend((clarabel.NonnegativeConeT(len(b)), clarabel.SecondOrderConeT(3),
                          clarabel.SecondOrderConeT(3)))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.static_regularization_constant = 1e-10
        settings.tol_gap_abs = settings.tol_gap_rel = 1e-8
        settings.tol_feas = 1e-8
        settings.reduced_tol_feas = 1e-7
        settings.reduced_tol_gap_abs = settings.reduced_tol_gap_rel = 1e-7
        matrix, rhs = np.vstack(blocks), np.concatenate(bounds)
        if projection:
            # Norm epigraph: min t subject to ||z-target|| <= t.
            # This avoids a quadratic objective on thin cell intersections.
            matrix = np.column_stack((matrix, np.zeros(len(rhs))))
            epigraph = np.zeros((4, 4))
            epigraph[0, 3] = -1.
            epigraph[1:, :3] = -np.eye(3)
            matrix = np.vstack((matrix, epigraph))
            rhs = np.r_[rhs, 0., -objective]
            cones.append(clarabel.SecondOrderConeT(4))
            q = np.array([0., 0., 0., 1.])
        else:
            q = -objective
        result = clarabel.DefaultSolver(sp.csc_matrix((len(q), len(q))), q,
                                       sp.csc_matrix(matrix), rhs, cones, settings).solve()
        if intersection is not None and str(result.status) in ('PrimalInfeasible', 'AlmostPrimalInfeasible'):
            return None
        if str(result.status) not in ('Solved', 'AlmostSolved'):
            raise RuntimeError(f'Convex solve failed: {result.status}; primal={result.r_prim:.3g}, '
                               f'gap={abs(result.obj_val-result.obj_val_dual):.3g}')
        self.support_queries += int(not projection)
        return np.array(result.x[:3])

    def _project_z(self, cell, entry_z, target_z):
        return self._solve(cell, target_z, entry_z, projection=True)

    def projection(self, cell, target, entry):
        return self.pose(cell.pose, self._project_z(cell, self.offset(cell.pose, entry), self.offset(cell.pose, target)))

    def attachment(self, first, second, entry):
        """Closest continuous portal to entry in the COMPLETE cell intersection."""
        if self.cells.slack(second, entry) >= 0.:
            return entry
        z = self._solve(first, self.offset(first.pose, entry), projection=True,
                        intersection=(second, self.offset(first.pose, second.pose)))
        return None if z is None else self.pose(first.pose, z)

    def release_geometry(self, cell):
        """Cache per signed facet (unit outward vector, leverage, clearance rate).

        n.dc-|z_theta|+m>=0 has TWO planar facets. Removing one facet
        removes exactly one signed inequality, preserving its sibling and guard.
        Zero support change identifies redundant/non-limiting rows.
        """
        if cell._release_geometry is None:
            rows, _ = self.facet_rows(cell)
            outward = rows / np.sqrt(2.)
            # Reflection z_theta -> -z_theta maps the two signed facets
            # and their leave-one-row-out domains exactly onto one another.
            leverage = np.empty(len(cell.offsets))
            for k, u in enumerate(outward[:len(cell.offsets)]):
                full = self._solve(cell, u)
                without = self._solve(cell, u, omit=k)
                leverage[k] = max(0., float(u @ (without-full)))
            leverage[leverage < 1e-7] = 0.
            leverage = np.tile(leverage, 2)
            robots = np.array([r for r, _ in cell.pairs], dtype=int)
            rotated = self.cells.robot_positions(cell.pose)[robots] - (cell.pose.x, cell.pose.y)
            angular = np.einsum('ij,ij->i', cell.normals,
                                np.column_stack((-rotated[:, 1], rotated[:, 0]))) / cell.rho
            rates = np.column_stack((cell.normals, angular))
            cell._release_geometry = (rows, leverage, np.tile(rates, (2, 1)))
        return cell._release_geometry

    def release_utilities(self, cell, direction):
        rows, leverage, rates = self.release_geometry(cell)
        sigma = rates @ direction
        return rows, leverage, sigma, leverage*np.maximum(sigma, 0.)

    def _constraints(self, cell, entry_z):
        rows, rhs = self.facet_rows(cell)
        def values(z):
            return np.r_[rhs-rows@z, cell.guard-np.linalg.norm(z[:2])-abs(z[2]),
                         cell.rho*cell.yaw_limit-abs(z[2]),
                         self.config.internal_distance**2-(z-entry_z)@(z-entry_z)]
        return {'fun': values}

    def select(self, cell: PolyhedralCell, target: Pose2D, mode: str, entry: Pose2D):
        started, before = time.perf_counter(), self.support_queries
        entry_z = self.offset(cell.pose, entry)
        target_z = self.offset(cell.pose, target)
        projected = self._project_z(cell, entry_z, target_z)
        projected_distance = float(np.linalg.norm(projected-target_z))
        sample_ball = (target_z, projected_distance+self.config.progress_epsilon)
        # The random direction is fixed at the current seed, as specified in zonghe.md.
        direction = self.offset(cell.pose, target)
        length = np.linalg.norm(direction)
        direction = direction/length if length > 1e-12 else np.zeros(3)
        utility, tendency, selected, alpha, progress = 0., 0., None, None, None
        cache_hit = cell._release_geometry is not None
        geometry_time = 0.
        if mode == 'opening':
            t = time.perf_counter()
            rows, leverage, sigma, utilities = self.release_utilities(cell, direction)
            geometry_time = time.perf_counter()-t
            tendency = float(utilities.sum())
            if len(utilities):
                k = int(np.argmax(utilities))
                utility = float(utilities[k])
                if utility > self.config.threshold and self.config.progress_epsilon > 0.:
                    selected = k
                    a = rows[k]
                    first = self._solve(cell, a, entry_z, sample_ball=sample_ball)
                    alpha = float(a@first)
                    # Pull back the specified physical d_s^T x covector to z.
                    sample_progress = direction / self.scale**2
                    z = self._solve(cell, sample_progress, entry_z,
                                    progress=(a, alpha-self.config.facet_epsilon), sample_ball=sample_ball)
                    progress = float(a@z)
        active = selected is not None
        if not active:
            z = projected
        n = len(cell.offsets)
        self.last_solve = dict(solve_time_s=time.perf_counter()-started, selection_valid=True,
                              local_method='projection_constrained_release_socp' if active else 'projection_socp',
                              projection_distance=projected_distance,
                              selected_sample_distance=float(np.linalg.norm(z-target_z)),
                              progress_epsilon=self.config.progress_epsilon,
                              opening_tendency=tendency, maximum_release_utility=utility,
                              selected_facet=selected,
                              selected_pair=list(cell.pairs[selected % n]) if active else None,
                              facet_alpha=alpha, facet_progress=progress,
                              release_cache_hit=cache_hit, release_geometry_time_s=geometry_time,
                              support_queries=self.support_queries-before, temporary_cells=0,
                              projection_fallback=not active,
                              constraint_violation=max(0., -float(self._constraints(cell, entry_z)['fun'](z).min())))
        return self.pose(cell.pose, z), tendency, active
