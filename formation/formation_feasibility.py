from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from formation.assignment import compute_best_assignment
from formation.mpc_controller import query_distance_field
from formation.swept_band import SweptBand, SweptBandBuilder
from formation.types import (
    AssignmentResult,
    CurveBand,
    EmbeddingQPResult,
    FormationSpec,
    LocalPreviewPath,
    MapData,
    Point2D,
    RobotState,
)


@dataclass
class FormationFeasibilityResult:
    formation_name: str
    is_feasible: bool
    center_points_xy: list[Point2D]
    heading_rads: list[float]
    slot_points_by_step_xy: list[list[Point2D]]
    min_corridor_margin_m: float
    corridor_violation_cost: float
    min_slot_clearance_m: float
    mean_clearance_m: float
    safety_margin_m: float
    offset_cost: float
    heading_cost: float
    lateral_offsets_m: list[float]
    heading_offsets_rad: list[float]
    assignment: AssignmentResult | None
    embedding_qp_result: EmbeddingQPResult | None
    failure_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeasibilityConfig:
    max_heading_offset_rad: float = 0.30
    heading_grid_size: int = 31
    lateral_grid_size: int = 31
    mode: str = "swept_band_v2"
    feasibility_tol_m: float = 0.05
    bspline_control_count: int = 6
    collocation_points: int = 30
    verification_points: int = 60
    embed_margin_m: float = 0.0
    ref_weight: float = 1.0
    smooth2_weight: float = 0.5
    smooth3_weight: float = 0.2
    clearance_weight: float = 0.05
    max_opt_iters: int = 200
    feasibility_tol: float = 1e-4


def _make_preview_from_curve(pts_xy, source_mode: str = "v2_cr") -> LocalPreviewPath:
    """Build a LocalPreviewPath from a polyline using its own Frenet frame."""
    pts = np.asarray(pts_xy, dtype=float)
    n = len(pts)

    arc = np.zeros(n, dtype=float)
    if n >= 2:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc[1:] = np.cumsum(seg)

    tangents = np.zeros_like(pts)
    for j in range(n):
        if n <= 1:
            d = np.array([1.0, 0.0])
        elif j == 0:
            d = pts[1] - pts[0]
        elif j == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[j + 1] - pts[j - 1]
        dn = float(np.linalg.norm(d))
        tangents[j] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])

    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) if n > 0 else np.zeros((0, 2), dtype=float)
    normals_lp = normals.copy()
    for j in range(n):
        lo, hi = max(0, j - 1), min(n, j + 2)
        avg = np.mean(normals[lo:hi], axis=0)
        an = float(np.linalg.norm(avg))
        if an > 1e-9:
            normals_lp[j] = avg / an

    return LocalPreviewPath(
        points_xy=[(float(p[0]), float(p[1])) for p in pts],
        arc_lengths=[float(s) for s in arc],
        tangents_xy=[(float(t[0]), float(t[1])) for t in tangents],
        normals_xy=[(float(nn[0]), float(nn[1])) for nn in normals_lp],
        curvatures=[0.0] * n,
        source_mode=source_mode,
        is_safe=True,
        min_clearance=0.5,
    )


class FormationFeasibility:
    """V2-only swept-band feasibility checker."""

    def __init__(self, config: FeasibilityConfig | None = None) -> None:
        self.config = config or FeasibilityConfig()
        self._swept_builder = SweptBandBuilder()
        self._v2_cache: dict[str, Any] = {}
        self._band_recenter_history: list[list[Point2D]] = []
        self._bezier_A_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def check(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand | None,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_states: list[RobotState] | None = None,
    ) -> FormationFeasibilityResult:
        del curve_band
        return self._check_swept_v2(
            map_data,
            preview_path,
            formation,
            robot_radius,
            safety_margin,
            current_formation,
            current_states,
        )

    def check_multi(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand | None,
        formations: list[FormationSpec],
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_states: list[RobotState] | None = None,
        *,
        preference_order: list[str] | None = None,
        stop_at_first_feasible: bool = True,
    ) -> list[FormationFeasibilityResult]:
        del curve_band
        return self._check_multi_v2(
            map_data,
            preview_path,
            formations,
            robot_radius,
            safety_margin,
            current_formation,
            current_states,
            preference_order=preference_order,
            stop_at_first_feasible=stop_at_first_feasible,
        )

    def optimize_centerline(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
    ) -> FormationFeasibilityResult:
        if self._v2_cache:
            cached = self._v2_cache
            return self._build_swept_result(
                map_data,
                formation.name,
                cached["band"],
                cached["C_star"],
                formation,
                robot_radius,
                safety_margin,
                None,
                None,
                metadata={
                    "check": "v2_cached",
                    "delta_L": cached.get("delta_L", 0.0),
                    "delta_R": cached.get("delta_R", 0.0),
                },
            )
        results = self._check_multi_v2(
            map_data,
            preview_path,
            [formation],
            robot_radius,
            safety_margin,
            None,
            None,
            stop_at_first_feasible=False,
        )
        if self._v2_cache:
            cached = self._v2_cache
            return self._build_swept_result(
                map_data,
                formation.name,
                cached["band"],
                cached["C_star"],
                formation,
                robot_radius,
                safety_margin,
                None,
                None,
                metadata={
                    "check": "v2_cached",
                    "delta_L": cached.get("delta_L", 0.0),
                    "delta_R": cached.get("delta_R", 0.0),
                },
            )
        return results[0]

    def _check_swept_v2(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None,
        current_states: list[RobotState] | None,
    ) -> FormationFeasibilityResult:
        clearance = robot_radius + safety_margin
        band = self._swept_builder.build(map_data, preview_path, clearance)
        C0 = np.asarray(preview_path.points_xy, dtype=float)
        if len(C0) < 2:
            return _infeasible(formation.name, "empty_preview")

        bz = _CenterlineBezier.from_reference(
            C0,
            control_count=self.config.bspline_control_count,
            bezier_tension=0.35,
        )
        _, C_R = self._band_recenter_qp(map_data, band, bz, C0, clearance)
        δL, δR = self._compute_widths(map_data, C_R, clearance)
        δ = self.config.embed_margin_m - self.config.feasibility_tol_m

        preview_R = _make_preview_from_curve(C_R, source_mode="v2_cr")
        band_star = self._swept_builder.build(map_data, preview_R, clearance)

        feasible = self._check_envelope(formation, δL, δR)
        if feasible:
            feasible, _ = self._eval_slots_rigid(map_data, C_R, formation, clearance, δ)

        return self._build_swept_result(
            map_data,
            formation.name,
            band_star,
            C_R,
            formation,
            robot_radius,
            safety_margin,
            current_formation,
            current_states,
            metadata={
                "check": "v2_pass" if feasible else "v2_fail",
                "delta_L": δL,
                "delta_R": δR,
            },
        )

    def _check_multi_v2(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        formations: list[FormationSpec],
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None,
        current_states: list[RobotState] | None,
        *,
        preference_order: list[str] | None = None,
        stop_at_first_feasible: bool = True,
    ) -> list[FormationFeasibilityResult]:
        clearance = robot_radius + safety_margin
        band = self._swept_builder.build(map_data, preview_path, clearance)
        C0 = np.asarray(preview_path.points_xy, dtype=float)
        if len(C0) < 2:
            return [_infeasible(fm.name, "empty_preview") for fm in formations]

        bz = _CenterlineBezier.from_reference(
            C0,
            control_count=self.config.bspline_control_count,
            bezier_tension=0.35,
        )
        _, C_R = self._band_recenter_qp(map_data, band, bz, C0, clearance)
        self._band_recenter_history.append([(float(p[0]), float(p[1])) for p in C_R])

        δL, δR = self._compute_widths(map_data, C_R, clearance)
        preview_R = _make_preview_from_curve(C_R, source_mode="v2_cr")
        band_star = self._swept_builder.build(map_data, preview_R, clearance)
        self._v2_cache = {
            "C_star": C_R,
            "C_R": C_R,
            "band": band_star,
            "delta_L": δL,
            "delta_R": δR,
        }
        δ = self.config.embed_margin_m - self.config.feasibility_tol_m

        if preference_order is not None:
            rank = {name: i for i, name in enumerate(preference_order)}
            ordered = sorted(formations, key=lambda f: rank.get(f.name, 99))
        else:
            ordered = sorted(formations, key=lambda f: f.lateral_half_width, reverse=True)

        results: list[FormationFeasibilityResult] = []
        for fm in ordered:
            feasible = self._check_envelope(fm, δL, δR)
            if feasible:
                ok_s, worst_s = self._eval_slots_rigid(map_data, C_R, fm, clearance, δ)
                if not ok_s:
                    print(
                        f"    [eval fail] {fm.name}: envelope_ok slot_worst={worst_s:.3f} "
                        f"δL={δL:.3f} δR={δR:.3f}",
                        flush=True,
                    )
                feasible = ok_s
            results.append(
                self._build_swept_result(
                    map_data,
                    fm.name,
                    band_star,
                    C_R,
                    fm,
                    robot_radius,
                    safety_margin,
                    current_formation,
                    current_states,
                    metadata={
                        "check": "v2_pass" if feasible else "v2_fail",
                        "delta_L": δL,
                        "delta_R": δR,
                    },
                )
            )
            if stop_at_first_feasible and feasible:
                break
        return results

    def _band_recenter_qp(
        self,
        map_data: MapData,
        band: SweptBand,
        bezier: _CenterlineBezier,
        C_init: np.ndarray,
        clearance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the locally convexified band-recenter problem."""
        try:
            import cvxpy as cp
        except ImportError:
            Q = bezier.initial_controls()
            print("    [Clarabel] unavailable; keeping initial centerline", flush=True)
            return Q, C_init

        Q = bezier.initial_controls().copy()
        M = self.config.collocation_points
        n_free = len(bezier.free_indices)
        N_c = bezier.control_count
        A = self._bezier_A_matrix(bezier, M)

        C_init_eval = bezier.evaluate(Q, M)
        d_end = C_init_eval[-1] - C_init_eval[-2]
        dn = float(np.linalg.norm(d_end))
        t_T = d_end / dn if dn > 1e-9 else np.array([1.0, 0.0])
        z_prog = C_init_eval[-1].copy()

        w_m = 10.0
        w_W = 2.0
        w_bal = 1.0
        w_sm = 0.5
        w_prog = 5.0
        w_c = 5.0
        w_b = 3.0
        w_step = 1.0
        ρ = clearance
        d_max = 1.50
        r_q = 0.15
        r_lin = 0.12
        l_min = 0.1
        θ_max = 0.6

        nd = n_free * 2
        dq = cp.Variable(nd, name="dq")
        dL = cp.Variable(name="dL")
        dR = cp.Variable(name="dR")
        m = cp.Variable(name="m")
        xiL = cp.Variable(M, name="xiL")
        xiR = cp.Variable(M, name="xiR")
        xic = cp.Variable(M, name="xic")

        q_current = cp.Parameter(2 * N_c, name="q_current")
        left_base = cp.Parameter(M, name="left_base")
        left_dq = cp.Parameter((M, nd), name="left_dq")
        left_dL = cp.Parameter(M, name="left_dL")
        right_base = cp.Parameter(M, name="right_base")
        right_dq = cp.Parameter((M, nd), name="right_dq")
        right_dR = cp.Parameter(M, name="right_dR")
        center_base = cp.Parameter(M, name="center_base")
        center_dq = cp.Parameter((M, nd), name="center_dq")
        deltaL_current = cp.Parameter(nonneg=True, name="deltaL_current")
        deltaR_current = cp.Parameter(nonneg=True, name="deltaR_current")
        progress_base = cp.Parameter(name="progress_base")
        progress_start_base = cp.Parameter(name="progress_start_base")
        progress_dq = cp.Parameter(nd, name="progress_dq")
        a0_base = cp.Parameter(name="a0_base")
        a0_dq = cp.Parameter(nd, name="a0_dq")
        b0_base = cp.Parameter(name="b0_base")
        b0_dq = cp.Parameter(nd, name="b0_dq")
        end_base = cp.Parameter(name="end_base")
        end_dq = cp.Parameter(nd, name="end_dq")

        control_map = np.zeros((2 * N_c, nd), dtype=float)
        for fi, ci in enumerate(bezier.free_indices):
            control_map[2 * ci:2 * ci + 2, 2 * fi:2 * fi + 2] = np.eye(2)
        D2 = np.zeros((max(0, N_c - 4), N_c), dtype=float)
        for rr, cidx in enumerate(range(2, N_c - 2)):
            D2[rr, cidx - 2:cidx + 3] = [1.0, -4.0, 6.0, -4.0, 1.0]
        A_rows = A.reshape(M, 2, nd)
        u0_map = control_map[2:4, :] - control_map[0:2, :]
        uend_map = control_map[-2:, :] - control_map[-4:-2, :]

        Qp = cp.reshape(q_current + control_map @ dq, (N_c, 2), order="C")
        dq_points = cp.reshape(dq, (n_free, 2), order="C")
        dC = cp.reshape(A @ dq, (M, 2), order="C")
        smooth_cost = cp.sum_squares(D2 @ Qp) if D2.shape[0] > 0 else 0.0
        progress_cost = cp.square(progress_base + progress_dq @ dq)
        objective = cp.Minimize(
            -w_m * m
            - w_W * (dL + dR)
            + w_bal * cp.square(dL - dR)
            + w_sm * smooth_cost
            + w_prog * progress_cost
            + w_step * cp.sum_squares(dq)
            + w_c * cp.sum_squares(xic)
            + w_b * (cp.sum_squares(xiL) + cp.sum_squares(xiR))
        )

        pred_L = left_base + left_dq @ dq + cp.multiply(left_dL, dL)
        pred_R = right_base + right_dq @ dq + cp.multiply(right_dR, dR)
        pred_c = center_base + center_dq @ dq

        constraints = [
            dL >= m,
            dR >= m,
            dL >= 0.0,
            dR >= 0.0,
            dL <= d_max,
            dR <= d_max,
            dL >= deltaL_current - 0.30,
            dL <= deltaL_current + 0.30,
            dR >= deltaR_current - 0.30,
            dR <= deltaR_current + 0.30,
            xiL >= 0.0,
            xiR >= 0.0,
            xic >= 0.0,
            pred_L - ρ + xiL >= 0.0,
            pred_R - ρ + xiR >= 0.0,
            pred_c - ρ + xic >= 0.0,
            progress_start_base + progress_dq @ dq >= l_min,
        ]
        constraints.extend(
            cp.norm2(dC[j, :]) <= r_lin
            for j in range(M)
        )
        constraints.extend(
            cp.norm2(dq_points[fi, :]) <= r_q
            for fi in range(n_free)
        )

        constraints.extend(
            [
                a0_base + a0_dq @ dq >= 0.01,
                cp.abs(b0_base + b0_dq @ dq)
                <= θ_max * (a0_base + a0_dq @ dq + 0.01),
                end_base + end_dq @ dq >= 0.02,
            ]
        )

        problem = cp.Problem(objective, constraints)

        for k in range(5):
            C = bezier.evaluate(Q, M)
            nn = self._normals(C)
            nn_lp = self._lowpass_normals(nn, window=3)
            δL_cur = self._max_uniform_offset_df(map_data, C, nn_lp, +1.0, clearance)
            δR_cur = self._max_uniform_offset_df(map_data, C, nn_lp, -1.0, clearance)
            print(
                f"    [Clarabel k={k}] δL={δL_cur:.3f} δR={δR_cur:.3f} "
                f"μ={(δL_cur - δR_cur) / 2:.4f}",
                flush=True,
            )

            e = 0.02
            x_L = C + δL_cur * nn_lp
            x_R = C - δR_cur * nn_lp
            all_pts = np.empty((9 * M, 2), dtype=float)
            all_pts[0::9] = x_L
            all_pts[1::9] = x_L + np.array([e, 0.0])
            all_pts[2::9] = x_L + np.array([0.0, e])
            all_pts[3::9] = x_R
            all_pts[4::9] = x_R + np.array([e, 0.0])
            all_pts[5::9] = x_R + np.array([0.0, e])
            all_pts[6::9] = C
            all_pts[7::9] = C + np.array([e, 0.0])
            all_pts[8::9] = C + np.array([0.0, e])
            vals = self._bilinear_batch(map_data, all_pts).reshape(M, 9)
            D_L = vals[:, 0]
            g_L = np.column_stack(((vals[:, 1] - D_L) / e, (vals[:, 2] - D_L) / e))
            D_R = vals[:, 3]
            g_R = np.column_stack(((vals[:, 4] - D_R) / e, (vals[:, 5] - D_R) / e))
            D_c = vals[:, 6]
            g_c = np.column_stack(((vals[:, 7] - D_c) / e, (vals[:, 8] - D_c) / e))

            if M >= 2:
                e0 = C[1] - C[0]
                e0n = float(np.linalg.norm(e0))
                e0 = e0 / e0n if e0n > 1e-9 else np.array([1.0, 0.0])
                n0 = np.array([-e0[1], e0[0]])
                e_end = C[-1] - C[-2]
                en = float(np.linalg.norm(e_end))
                e_end = e_end / en if en > 1e-9 else np.array([1.0, 0.0])
            else:
                e0 = e_end = np.array([1.0, 0.0])
                n0 = np.array([0.0, 1.0])

            q_current.value = Q.ravel()
            left_base.value = D_L + np.sum(g_L * (C - x_L), axis=1)
            left_dq.value = np.einsum("mi,mij->mj", g_L, A_rows)
            left_dL.value = np.einsum("mi,mi->m", g_L, nn_lp)
            right_base.value = D_R + np.sum(g_R * (C - x_R), axis=1)
            right_dq.value = np.einsum("mi,mij->mj", g_R, A_rows)
            right_dR.value = -np.einsum("mi,mi->m", g_R, nn_lp)
            center_base.value = D_c
            center_dq.value = np.einsum("mi,mij->mj", g_c, A_rows)
            deltaL_current.value = max(float(δL_cur), 0.0)
            deltaR_current.value = max(float(δR_cur), 0.0)
            progress_base.value = float(np.dot(C[-1] - z_prog, t_T))
            progress_start_base.value = float(np.dot(C[-1] - C_init_eval[0], t_T))
            progress_dq.value = t_T @ A[-2:, :]
            a0_current = Q[1] - Q[0]
            a0_base.value = float(np.dot(a0_current, e0))
            a0_dq.value = e0 @ u0_map
            b0_base.value = float(np.dot(a0_current, n0))
            b0_dq.value = n0 @ u0_map
            end_current = Q[-1] - Q[-2]
            end_base.value = float(np.dot(end_current, e_end))
            end_dq.value = e_end @ uend_map

            if k == 0:
                shift_init = min(0.5 * abs((δL_cur - δR_cur) / 2), 0.06)
                direction = np.sign((δL_cur - δR_cur) / 2)
                dc_d = np.zeros(M * 2)
                for j in range(M):
                    chi = 1.0 if j > 0 else 0.0
                    dc_d[j * 2] = direction * shift_init * nn_lp[j, 0]
                    dc_d[j * 2 + 1] = direction * shift_init * nn_lp[j, 1]
                dq.value = A.T @ dc_d
                dL.value = float(δL_cur)
                dR.value = float(δR_cur)
                m.value = min(float(δL_cur), float(δR_cur))
                xiL.value = np.zeros(M)
                xiR.value = np.zeros(M)
                xic.value = np.zeros(M)

            try:
                problem.solve(
                    solver=cp.CLARABEL,
                    warm_start=True,
                    max_iter=max(1, int(self.config.max_opt_iters)),
                    tol_gap_abs=1e-6,
                    tol_gap_rel=1e-6,
                    tol_feas=1e-6,
                    verbose=False,
                )
            except (cp.error.SolverError, ValueError) as exc:
                print(f"    [Clarabel] solve failed: {exc}", flush=True)
                break

            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or dq.value is None:
                print(f"    [Clarabel] status={problem.status}; keeping current centerline", flush=True)
                break

            dq_solution = np.asarray(dq.value, dtype=float).reshape(n_free, 2)
            for fi, ci in enumerate(bezier.free_indices):
                Q[ci] += dq_solution[fi]
            Q[0] = C_init_eval[0].copy()
            print(
                f"    [Clarabel] status={problem.status} dL*={float(dL.value):.3f} "
                f"dR*={float(dR.value):.3f} |dq|={float(np.linalg.norm(dq_solution)):.3f}",
                flush=True,
            )

        C_final = bezier.evaluate(Q, M)
        C_i = bezier.evaluate(bezier.initial_controls(), M)
        shift = float(np.max(np.linalg.norm(C_final - C_i, axis=1)))
        nn_f = self._normals(C_final)
        nn_flp = self._lowpass_normals(nn_f)
        δL_f = self._max_uniform_offset_df(map_data, C_final, nn_flp, +1.0, clearance)
        δR_f = self._max_uniform_offset_df(map_data, C_final, nn_flp, -1.0, clearance)
        mg = self._margin_batch(map_data, C_final, clearance)
        min_m = float(np.min(mg))
        n_bad = int(np.sum(mg < -0.02))
        has_rev = any(
            float(np.dot(C_final[j + 1] - C_final[j], C_final[1] - C_final[0])) < -0.005
            for j in range(M - 1)
        ) if M >= 2 else False
        st = "WARN" if (min_m < -0.02 or has_rev) else "OK"
        print(
            f"  [v2] band_recenter max_shift={shift:.3f}m final δL={δL_f:.3f} δR={δR_f:.3f} "
            f"min_m={min_m:.3f} n_bad={n_bad} rev={has_rev} [{st}]",
            flush=True,
        )
        return Q, C_final

    def _bezier_A_matrix(
        self,
        bezier: _CenterlineBezier,
        M: int,
        eps: float = 0.005,
    ) -> np.ndarray:
        key = (id(bezier), bezier.control_count, tuple(bezier.free_indices), int(M), float(eps))
        cached = self._bezier_A_cache.get(key)
        if cached is not None:
            return cached

        n_free = len(bezier.free_indices)
        Q0 = bezier.initial_controls().copy()
        C0 = bezier.evaluate(Q0, M)
        A = np.zeros((M * 2, n_free * 2), dtype=float)
        for fi, ci in enumerate(bezier.free_indices):
            for d in range(2):
                Qp = Q0.copy()
                Qp[ci, d] += eps
                dC = (bezier.evaluate(Qp, M) - C0) / eps
                A[0::2, fi * 2 + d] = dC[:, 0]
                A[1::2, fi * 2 + d] = dC[:, 1]

        self._bezier_A_cache[key] = A
        return A

    @staticmethod
    def _normals(cc: np.ndarray) -> np.ndarray:
        nn = np.zeros_like(cc)
        n = len(cc)
        for j in range(n):
            if j == 0:
                d = cc[1] - cc[0]
            elif j == n - 1:
                d = cc[-1] - cc[-2]
            else:
                d = cc[j + 1] - cc[j - 1]
            dn = float(np.linalg.norm(d))
            t = d / dn if dn > 1e-9 else np.array([1.0, 0.0])
            nn[j] = np.array([-t[1], t[0]])
        return nn

    @staticmethod
    def _lowpass_normals(nn: np.ndarray, window: int = 3) -> np.ndarray:
        n_lp = nn.copy()
        hw = window // 2
        for j in range(len(nn)):
            lo, hi = max(0, j - hw), min(len(nn), j + hw + 1)
            avg = np.mean(nn[lo:hi], axis=0)
            nrm = float(np.linalg.norm(avg))
            if nrm > 1e-9:
                n_lp[j] = avg / nrm
        return n_lp

    @staticmethod
    def _bilinear_batch(map_data: MapData, pts: np.ndarray) -> np.ndarray:
        origin_x, origin_y = map_data.origin_xy
        res = map_data.resolution
        grid_x = (pts[:, 0] - origin_x) / res - 0.5
        grid_y = (pts[:, 1] - origin_y) / res - 0.5
        x0 = np.floor(grid_x).astype(int)
        y0 = np.floor(grid_y).astype(int)
        x1 = np.minimum(x0 + 1, map_data.cols - 1)
        y1 = np.minimum(y0 + 1, map_data.rows - 1)
        wx = grid_x - x0
        wy = grid_y - y0
        df = map_data.distance_field
        valid = (grid_x >= 0) & (grid_y >= 0) & (grid_x <= map_data.cols - 1) & (grid_y <= map_data.rows - 1)
        x0c = np.clip(x0, 0, map_data.cols - 1)
        y0c = np.clip(y0, 0, map_data.rows - 1)
        x1c = np.clip(x1, 0, map_data.cols - 1)
        y1c = np.clip(y1, 0, map_data.rows - 1)
        v00 = df[y0c, x0c]
        v10 = df[y0c, x1c]
        v01 = df[y1c, x0c]
        v11 = df[y1c, x1c]
        result = ((1 - wy) * (1 - wx) * v00 + (1 - wy) * wx * v10 + wy * (1 - wx) * v01 + wy * wx * v11)
        result[~valid] = 0.0
        return result

    @staticmethod
    def _margin_batch(map_data: MapData, pts: np.ndarray, clearance: float) -> np.ndarray:
        return FormationFeasibility._bilinear_batch(map_data, pts) - clearance

    @staticmethod
    def _max_uniform_offset_df(
        map_data: MapData,
        cc: np.ndarray,
        nn: np.ndarray,
        sg: float,
        clearance: float,
    ) -> float:
        ds = np.linspace(0.0, 1.50, 16)
        pts = cc[None, :, :] + (sg * ds[:, None, None]) * nn[None, :, :]
        margins = FormationFeasibility._margin_batch(
            map_data,
            pts.reshape(-1, 2),
            clearance,
        ).reshape(len(ds), len(cc))
        bad = np.any(margins < -0.02, axis=1)
        if np.any(bad):
            d = float(ds[int(np.argmax(bad))])
            return max(0.0, d - 0.05)
        return 1.50

    def _compute_widths(
        self,
        map_data: MapData,
        curve: np.ndarray,
        clearance: float,
    ) -> tuple[float, float]:
        nn = self._normals(curve)
        nn_lp = self._lowpass_normals(nn)
        return (
            self._max_uniform_offset_df(map_data, curve, nn_lp, +1.0, clearance),
            self._max_uniform_offset_df(map_data, curve, nn_lp, -1.0, clearance),
        )

    @staticmethod
    def _eval_slots_rigid(
        map_data: MapData,
        centre_curve: np.ndarray,
        formation: FormationSpec,
        clearance: float,
        margin_req: float,
    ) -> tuple[bool, float]:
        n = len(centre_curve)
        if n < 2:
            return False, float("-inf")

        tangents = np.zeros((n, 2), dtype=float)
        for j in range(n):
            if j == 0:
                d = centre_curve[1] - centre_curve[0]
            elif j == n - 1:
                d = centre_curve[-1] - centre_curve[-2]
            else:
                d = centre_curve[j + 1] - centre_curve[j - 1]
            dn = float(np.linalg.norm(d))
            tangents[j] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])

        positions: list[tuple[float, float]] = []
        for j in range(n):
            cx, cy = centre_curve[j, 0], centre_curve[j, 1]
            tx, ty = tangents[j, 0], tangents[j, 1]
            nx, ny = -ty, tx
            for slot in formation.slots:
                ell, b = slot[0], slot[1]
                qx = cx + ell * tx + b * nx
                qy = cy + ell * ty + b * ny
                positions.append((float(qx), float(qy)))

        if not positions:
            return True, 0.0

        margins = FormationFeasibility._margin_batch(
            map_data,
            np.asarray(positions, dtype=float),
            clearance,
        )
        worst = float(np.min(margins))
        return worst >= margin_req, worst

    @staticmethod
    def _check_envelope(formation: FormationSpec, δL: float, δR: float) -> bool:
        slots = formation.slots
        B_L = max(s[1] for s in slots)
        B_R = max(-s[1] for s in slots)
        return B_L <= δL and B_R <= δR

    @staticmethod
    def _eval_slots(
        band: SweptBand,
        centre_curve: np.ndarray,
        formation: FormationSpec,
        margin_req: float,
    ) -> tuple[bool, float]:
        n = len(centre_curve)
        if n < 2:
            return False, float("-inf")
        total_len = float(sum(np.linalg.norm(centre_curve[i + 1] - centre_curve[i]) for i in range(n - 1))) or 1.0
        tangents = np.zeros((n, 2), dtype=float)
        for j in range(n):
            if j == 0:
                d = centre_curve[1] - centre_curve[0]
            elif j == n - 1:
                d = centre_curve[-1] - centre_curve[-2]
            else:
                d = centre_curve[j + 1] - centre_curve[j - 1]
            dn = float(np.linalg.norm(d))
            tangents[j] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])

        positions: list[np.ndarray] = []
        for j in range(n):
            tau = j / max(n - 1, 1)
            for slot in formation.slots:
                tau_eff = tau + slot[0] / total_len
                if tau_eff < 0.0 or tau_eff > 1.0:
                    continue
                j_float = tau_eff * (n - 1)
                j0 = max(0, min(int(j_float), n - 2))
                j1 = j0 + 1
                frac = j_float - j0
                c_eff = centre_curve[j0] + frac * (centre_curve[j1] - centre_curve[j0])
                t_eff = tangents[j0] + frac * (tangents[j1] - tangents[j0])
                tn = float(np.linalg.norm(t_eff))
                if tn < 1e-9:
                    continue
                t_eff /= tn
                n_eff = np.array([-t_eff[1], t_eff[0]])
                slot_xy = c_eff + slot[1] * n_eff
                positions.append(slot_xy)
        worst = float(np.min(band.margin_batch(np.asarray(positions, dtype=float)))) if positions else float("inf")
        return worst >= margin_req, float(worst)

    def _build_swept_result(
        self,
        map_data: MapData,
        name: str,
        band: SweptBand,
        centre_curve: np.ndarray,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None,
        current_states: list[RobotState] | None,
        metadata: dict[str, Any] | None = None,
    ) -> FormationFeasibilityResult:
        n = len(centre_curve)
        if n < 2:
            return _infeasible(name, "empty_centreline")

        required = robot_radius + safety_margin
        headings, slot_pts_by_step = self._compute_slots(centre_curve, formation)
        ok_swept, worst_m = self._eval_slots(band, centre_curve, formation, -self.config.feasibility_tol_m)
        min_margin = worst_m

        cl_samples = [
            float(query_distance_field(map_data, slot_pt))
            for step_slots in slot_pts_by_step
            for slot_pt in step_slots
        ]
        min_cl = min(cl_samples) if cl_samples else 0.0
        mean_cl = sum(cl_samples) / len(cl_samples) if cl_samples else 0.0
        feasible = ok_swept

        n_robots = len(formation.slots)
        if current_formation is None or current_formation.name == formation.name:
            assignment = AssignmentResult(
                assignment=tuple(range(n_robots)),
                total_cost=0.0,
                max_cost=0.0,
                per_robot_costs=[0.0] * n_robots,
            )
        else:
            if current_states:
                current_slots_xy = [(s.x, s.y) for s in current_states]
            else:
                current_slots_xy = _nominal_slots(formation, tuple(centre_curve[0]), headings[0])
            assignment = compute_best_assignment(current_slots_xy, slot_pts_by_step[0])

        return FormationFeasibilityResult(
            formation_name=name,
            is_feasible=feasible,
            center_points_xy=[(float(c[0]), float(c[1])) for c in centre_curve],
            heading_rads=headings,
            slot_points_by_step_xy=slot_pts_by_step,
            min_corridor_margin_m=min_margin,
            corridor_violation_cost=0.0 if feasible else -min_margin,
            min_slot_clearance_m=min_cl,
            mean_clearance_m=mean_cl,
            safety_margin_m=min_cl - required,
            offset_cost=0.0,
            heading_cost=0.0,
            lateral_offsets_m=[],
            heading_offsets_rad=[],
            assignment=assignment,
            embedding_qp_result=None,
            failure_reason="" if feasible else "slot_outside_swept_band",
            metadata=metadata or {},
        )

    @staticmethod
    def _compute_slots(
        centre_curve: np.ndarray,
        formation: FormationSpec,
    ) -> tuple[list[float], list[list[Point2D]]]:
        n = len(centre_curve)
        headings: list[float] = []
        slots: list[list[Point2D]] = []
        for j in range(n):
            c = centre_curve[j]
            if j == 0:
                d = centre_curve[1] - centre_curve[0]
            elif j == n - 1:
                d = centre_curve[-1] - centre_curve[-2]
            else:
                d = centre_curve[j + 1] - centre_curve[j - 1]
            dn = float(np.linalg.norm(d))
            if dn < 1e-9:
                d = np.array([1.0, 0.0])
                dn = 1.0
            t = d / dn
            h = math.atan2(t[1], t[0])
            headings.append(h)
            cos_h, sin_h = math.cos(h), math.sin(h)
            step_slots: list[Point2D] = []
            for slot in formation.slots:
                step_slots.append(
                    (
                        c[0] + cos_h * slot[0] - sin_h * slot[1],
                        c[1] + sin_h * slot[0] + cos_h * slot[1],
                    )
                )
            slots.append(step_slots)
        return headings, slots

    def _slot_clearance_stats(
        self,
        map_data: MapData,
        slot_points_by_step: list[list[Point2D]],
    ) -> tuple[float, float]:
        clearances = [
            float(query_distance_field(map_data, slot))
            for step_slots in slot_points_by_step
            for slot in step_slots
        ]
        if not clearances:
            return 0.0, 0.0
        return min(clearances), sum(clearances) / len(clearances)


def _infeasible(
    name: str,
    reason: str,
    embedding: EmbeddingQPResult | None = None,
) -> FormationFeasibilityResult:
    return FormationFeasibilityResult(
        formation_name=name,
        is_feasible=False,
        center_points_xy=[],
        heading_rads=[],
        slot_points_by_step_xy=[],
        min_corridor_margin_m=0.0 if embedding is None else embedding.min_corridor_margin_m,
        corridor_violation_cost=0.0 if embedding is None else embedding.corridor_violation_cost,
        min_slot_clearance_m=0.0,
        mean_clearance_m=0.0,
        safety_margin_m=0.0,
        offset_cost=0.0 if embedding is None else embedding.offset_cost,
        heading_cost=0.0 if embedding is None else embedding.heading_cost,
        lateral_offsets_m=[],
        heading_offsets_rad=[],
        assignment=None,
        embedding_qp_result=embedding,
        failure_reason=reason,
    )


def _nominal_slots(
    formation: FormationSpec,
    center: Point2D,
    heading: float,
) -> list[Point2D]:
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return [
        (
            center[0] + cos_h * s[0] - sin_h * s[1],
            center[1] + sin_h * s[0] + cos_h * s[1],
        )
        for s in formation.slots
    ]


class _CenterlineBezier:
    """Piecewise cubic Bézier centreline  c(τ; Q),  τ ∈ [0, 1]."""

    def __init__(
        self,
        control_count: int,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        ref_curve: np.ndarray,
        bezier_tension: float = 0.35,
    ) -> None:
        self.control_count = max(control_count, 3)
        self.start = start_xy.copy()
        self.end = end_xy.copy()
        self._ref = ref_curve
        self.tension = bezier_tension
        self.free_indices = list(range(1, self.control_count))

    @classmethod
    def from_reference(
        cls,
        ref_curve: np.ndarray,
        control_count: int,
        bezier_tension: float = 0.35,
    ) -> _CenterlineBezier:
        return cls(
            control_count=control_count,
            start_xy=ref_curve[0].copy(),
            end_xy=ref_curve[-1].copy(),
            ref_curve=ref_curve,
            bezier_tension=bezier_tension,
        )

    def initial_controls(self) -> np.ndarray:
        Q = np.zeros((self.control_count, 2), dtype=float)
        Q[0] = self.start
        Q[-1] = self.end
        n_ref = len(self._ref)
        for k in range(1, self.control_count - 1):
            t = k / (self.control_count - 1)
            idx = min(int(t * (n_ref - 1)), n_ref - 1)
            Q[k] = self._ref[idx]
        return Q

    def evaluate(self, Q: np.ndarray, num_points: int) -> np.ndarray:
        M = self.control_count
        if M <= 2:
            ts = np.linspace(0.0, 1.0, num_points)
            return np.outer(1 - ts, Q[0]) + np.outer(ts, Q[-1])

        tangents = _compute_tangents(Q)
        segs_per_step = max(1, 4)
        raw: list[tuple[float, float]] = []
        for seg_idx in range(M - 1):
            p0, p3 = Q[seg_idx], Q[seg_idx + 1]
            seg_len = float(np.linalg.norm(p3 - p0))
            if seg_len < 1e-9:
                continue
            cs = self.tension * seg_len
            p1 = p0 + tangents[seg_idx] * cs
            p2 = p3 - tangents[seg_idx + 1] * cs
            seg_pts = _sample_cubic_bezier(p0, p1, p2, p3, segs_per_step)
            if raw:
                seg_pts = seg_pts[1:]
            raw.extend((float(x), float(y)) for x, y in seg_pts)
        if not raw:
            raw = [(float(Q[0, 0]), float(Q[0, 1])), (float(Q[-1, 0]), float(Q[-1, 1]))]
        return _resample_polyline(np.asarray(raw, dtype=float), num_points)

    def bounding_box(self) -> tuple[float, float, float, float]:
        margin = 0.80
        return (
            float(min(self.start[0], self.end[0]) - margin),
            float(max(self.start[0], self.end[0]) + margin),
            float(min(self.start[1], self.end[1]) - margin),
            float(max(self.start[1], self.end[1]) + margin),
        )


def _compute_tangents(Q: np.ndarray) -> np.ndarray:
    M = len(Q)
    tangents = np.zeros_like(Q)
    for i in range(M):
        if i == 0:
            d = Q[1] - Q[0]
        elif i == M - 1:
            d = Q[-1] - Q[-2]
        else:
            d = Q[i + 1] - Q[i - 1]
        nrm = float(np.linalg.norm(d))
        tangents[i] = np.array([1.0, 0.0]) if nrm < 1e-9 else d / nrm
    return tangents


def _sample_cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    steps: int,
) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, steps + 1)
    omt = 1.0 - ts
    return (
        (omt ** 3)[:, None] * p0
        + (3.0 * omt ** 2 * ts)[:, None] * p1
        + (3.0 * omt * ts ** 2)[:, None] * p2
        + (ts ** 3)[:, None] * p3
    )


def _resample_polyline(pts: np.ndarray, num_points: int) -> np.ndarray:
    if len(pts) <= 1:
        return pts
    seg_vecs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(cum[-1])
    if total < 1e-9:
        return np.tile(pts[0], (num_points, 1))
    ts = np.linspace(0.0, total, num_points)
    result = np.zeros((num_points, 2), dtype=float)
    si = 0
    for i, arc_s in enumerate(ts):
        while si < len(seg_lens) - 1 and cum[si + 1] < arc_s:
            si += 1
        sl = seg_lens[si]
        if sl < 1e-9:
            result[i] = pts[si]
        else:
            r = (arc_s - cum[si]) / sl
            result[i] = pts[si] + r * seg_vecs[si]
    return result
