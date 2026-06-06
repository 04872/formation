"""Swept-band v2 feasibility: band-recenter QP → fixed centreline C* → envelope + slot check.

The centreline C* is computed *once* (shared across all formations).
Each formation is judged via lateral-envelope quick-check and discrete slot validation.
No per-formation optimisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from formation.formation_feasibility import (
    FeasibilityConfig,
    FormationFeasibility,
    FormationFeasibilityResult,
    SweptBand,
    _CenterlineBezier,
    _compute_tangents,
    _resample_polyline,
    _infeasible,
)
from formation.mpc_controller import query_distance_field
from formation.types import FormationSpec, LocalPreviewPath, MapData


def _make_preview_from_curve(pts_xy, source_mode: str = "v2_cr"):
    """Build a LocalPreviewPath from a polyline using its own Frenet frame."""
    pts = np.asarray(pts_xy, dtype=float)
    n = len(pts)

    arc = np.zeros(n, dtype=float)
    if n >= 2:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc[1:] = np.cumsum(seg)

    tangents = np.zeros_like(pts)
    for j in range(n):
        if n == 0:
            d = np.array([1.0, 0.0])
        elif n == 1:
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


class FormationFeasibilityV2(FormationFeasibility):
    """Extended feasibility checker using band-recenter + envelope logic."""

    def __init__(self, config=None):
        super().__init__(config)
        self._v2_cache: dict = {}
        self._band_recenter_history: list = []
        # Cache the constant linear map dC = A @ dq for a given Bezier
        # sampling pattern.  For a Bezier curve, evaluate(Q, M) is linear
        # in Q, so this matrix does not depend on the current Q.
        # It intentionally preserves the old evaluate()-based derivative
        # convention instead of assuming a different internal basis.
        self._bezier_A_cache: dict = {}

    def optimize_centerline(
        self, map_data, preview_path, formation, robot_radius, safety_margin,
    ):
        if self.config.mode == "swept_band_v2" and self._v2_cache:
            c = self._v2_cache
            return self._build_swept_result(
                map_data, formation.name, c["band"], c["C_star"], formation,
                robot_radius, safety_margin, None, None,
                metadata={"check": "v2_cached"})
        return super().optimize_centerline(
            map_data, preview_path, formation, robot_radius, safety_margin)

    def check(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_states=None,
    ) -> FormationFeasibilityResult:
        if self.config.mode == "swept_band_v2":
            return self._check_swept_v2(
                map_data, preview_path, formation,
                robot_radius, safety_margin,
                current_formation, current_states,
            )
        return super().check(
            map_data, preview_path, curve_band, formation,
            robot_radius, safety_margin,
            current_formation, current_states,
        )

    def check_multi(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band,
        formations: list[FormationSpec],
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_states=None,
        *,
        preference_order: list[str] | None = None,
        stop_at_first_feasible: bool = True,
    ) -> list[FormationFeasibilityResult]:
        if self.config.mode == "swept_band_v2":
            return self._check_multi_v2(
                map_data, preview_path, formations,
                robot_radius, safety_margin,
                current_formation, current_states,
                preference_order=preference_order,
                stop_at_first_feasible=stop_at_first_feasible,
            )
        return super().check_multi(
            map_data, preview_path, curve_band, formations,
            robot_radius, safety_margin,
            current_formation, current_states,
            preference_order=preference_order,
            stop_at_first_feasible=stop_at_first_feasible,
        )

    # ═══════════════════════════════════════════════════════════════
    #  v2 pipeline
    # ═══════════════════════════════════════════════════════════════

    def _check_swept_v2(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None,
        current_states,
    ) -> FormationFeasibilityResult:
        clearance = robot_radius + safety_margin
        band = self._swept_builder.build(map_data, preview_path, clearance)
        C0 = np.asarray(preview_path.points_xy, dtype=float)
        if len(C0) < 2:
            return _infeasible(formation.name, "empty_preview")

        bz = _CenterlineBezier.from_reference(C0, control_count=6, bezier_tension=0.35)
        _, C_R = self._band_recenter_qp(map_data, band, bz, C0, clearance)
        δL, δR = self._compute_widths(map_data, C_R, clearance)
        δ = self.config.embed_margin_m - self.config.feasibility_tol_m

        preview_R = _make_preview_from_curve(C_R, source_mode="v2_cr")
        band_star = self._swept_builder.build(map_data, preview_R, clearance)

        feasible = self._check_envelope(formation, δL, δR)
        if feasible:
            feasible, _ = self._eval_slots_rigid(map_data, C_R, formation, clearance, δ)

        return self._build_swept_result(
            map_data, formation.name, band_star, C_R, formation,
            robot_radius, safety_margin, current_formation, current_states,
            metadata={"check": "v2_pass" if feasible else "v2_fail",
                       "delta_L": δL, "delta_R": δR},
        )

    def _check_multi_v2(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        formations: list[FormationSpec],
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None,
        current_states,
        *,
        preference_order: list[str] | None = None,
        stop_at_first_feasible: bool = True,
    ) -> list[FormationFeasibilityResult]:
        clearance = robot_radius + safety_margin
        band = self._swept_builder.build(map_data, preview_path, clearance)
        C0 = np.asarray(preview_path.points_xy, dtype=float)
        if len(C0) < 2:
            return [_infeasible(fm.name, "empty_preview") for fm in formations]

        # ── 1. band‑recenter ───────────────────────────────────
        bz = _CenterlineBezier.from_reference(C0, control_count=6, bezier_tension=0.35)
        Q_R, C_R = self._band_recenter_qp(map_data, band, bz, C0, clearance)
        self._band_recenter_history.append([(float(p[0]), float(p[1])) for p in C_R])

        # ── 2. compute widths and rebuild band from C_R ──────────
        δL, δR = self._compute_widths(map_data, C_R, clearance)
        preview_R = _make_preview_from_curve(C_R, source_mode="v2_cr")
        band_star = self._swept_builder.build(map_data, preview_R, clearance)
        self._v2_cache = {"C_star": C_R, "C_R": C_R, "band": band_star, "delta_L": δL, "delta_R": δR}
        δ = self.config.embed_margin_m - self.config.feasibility_tol_m

        if preference_order is not None:
            rank = {name: i for i, name in enumerate(preference_order)}
            ordered = sorted(formations, key=lambda f: rank.get(f.name, 99))
        else:
            ordered = sorted(formations, key=lambda f: f.lateral_half_width, reverse=True)

        # ── per‑formation: envelope + slot validation ─────────────
        results: list[FormationFeasibilityResult] = []
        for fm in ordered:
            feasible = self._check_envelope(fm, δL, δR)
            if feasible:
                ok_s, worst_s = self._eval_slots_rigid(map_data, C_R, fm, clearance, δ)
                if not ok_s: print(f"    [eval fail] {fm.name}: envelope_ok slot_worst={worst_s:.3f} δL={δL:.3f} δR={δR:.3f}", flush=True)
                feasible = ok_s
            results.append(self._build_swept_result(
                map_data, fm.name, band_star, C_R, fm,
                robot_radius, safety_margin, current_formation, current_states,
                metadata={"check": "v2_pass" if feasible else "v2_fail",
                           "delta_L": δL, "delta_R": δR},
            ))
            if stop_at_first_feasible and feasible:
                break
        return results

    # ═══════════════════════════════════════════════════════════════
    #  band‑recenter QP micro‑loop
    # ═══════════════════════════════════════════════════════════════

    def _band_recenter_qp(
        self, map_data: MapData, band: SweptBand, bezier: _CenterlineBezier,
        C_init: np.ndarray, clearance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """QP micro‑loop: push curve toward wider side of band."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            Q = bezier.initial_controls()
            return Q, C_init

        Q = bezier.initial_controls().copy()
        M = self.config.collocation_points
        n_free = len(bezier.free_indices)
        N_c = bezier.control_count

        # Equivalent speed-up: Bezier sampling is linear in control points,
        # so the Jacobian used by SLSQP is constant across the 5 outer
        # iterations.  The old code recomputed it numerically every round.
        A = self._bezier_A_matrix(bezier, M)

        def _max_off(cc, nn, sg):
            return self._max_uniform_offset_df(map_data, cc, nn, sg, clearance)

        # pre‑compute z_prog, t_T once
        C_init_eval = bezier.evaluate(Q, M)
        d_end = C_init_eval[-1] - C_init_eval[-2]
        dn = float(np.linalg.norm(d_end))
        t_T = d_end / dn if dn > 1e-9 else np.array([1.0, 0.0])
        z_prog = C_init_eval[-1].copy()

        for k in range(5):
            C = bezier.evaluate(Q, M)
            nn = self._normals(C)
            nn_lp = self._lowpass_normals(nn, window=3)
            δL_cur = _max_off(C, nn_lp, +1.0); δR_cur = _max_off(C, nn_lp, -1.0)
            print(f"    [QP k={k}] δL={δL_cur:.3f} δR={δR_cur:.3f} μ={(δL_cur-δR_cur)/2:.4f}", flush=True)

            # ── linearisation points + EDT gradients (batch) ─────
            # Preserve the old forward-difference gradient definition exactly:
            #   gx = (D(x+e,y)-D(x,y))/e, gy = (D(x,y+e)-D(x,y))/e
            # but build the 9M query points without Python per-point assignment.
            e = 0.02; bif = self._bilinear_batch
            x_L = C + δL_cur * nn_lp
            x_R = C - δR_cur * nn_lp
            all_pts = np.empty((9*M, 2), dtype=float)
            all_pts[0::9] = x_L
            all_pts[1::9] = x_L + np.array([e, 0.0])
            all_pts[2::9] = x_L + np.array([0.0, e])
            all_pts[3::9] = x_R
            all_pts[4::9] = x_R + np.array([e, 0.0])
            all_pts[5::9] = x_R + np.array([0.0, e])
            all_pts[6::9] = C
            all_pts[7::9] = C + np.array([e, 0.0])
            all_pts[8::9] = C + np.array([0.0, e])
            vals = bif(map_data, all_pts).reshape(M, 9)
            D_L = vals[:, 0]
            g_L = np.column_stack(((vals[:, 1] - D_L) / e, (vals[:, 2] - D_L) / e))
            D_R = vals[:, 3]
            g_R = np.column_stack(((vals[:, 4] - D_R) / e, (vals[:, 5] - D_R) / e))
            D_c = vals[:, 6]
            g_c = np.column_stack(((vals[:, 7] - D_c) / e, (vals[:, 8] - D_c) / e))

            # A is the same matrix the old numerical Jacobian produced,
            # precomputed once before the outer loop.

            # start/end directions
            if M >= 2:
                e0 = C[1]-C[0]; e0n=float(np.linalg.norm(e0))
                e0 = e0/e0n if e0n>1e-9 else np.array([1.,0.])
                n0 = np.array([-e0[1], e0[0]])
                e_end = C[-1]-C[-2]; en=float(np.linalg.norm(e_end))
                e_end = e_end/en if en>1e-9 else np.array([1.,0.])
            else:
                e0=e_end=np.array([1.,0.]); n0=np.array([0.,1.])

            # ── weights ──────────────────────────────────────────
            w_m=10.0; w_W=2.0; w_bal=1.0; w_sm=0.5; w_prog=5.0
            w_c=5.0; w_b=3.0; w_step=1.0
            ρ=clearance; d_max=1.50; r_q=0.15; r_lin=0.12; l_min=0.1; θ_max=0.6

            # vars: dq[n_free*2], d_L, d_R, m, xi_L[M], xi_R[M], xi_c[M]
            n_vars = n_free*2 + 3 + 3*M
            Q_free = np.zeros((n_free, 2))
            for fi, ci in enumerate(bezier.free_indices): Q_free[fi] = Q[ci]

            # Analytic derivatives for SLSQP.  These keep the old problem
            # structure unchanged but avoid SLSQP's finite-difference
            # objective/constraint Jacobian over ~100 variables.
            nd = n_free * 2
            i_dL = nd
            i_dR = nd + 1
            i_m = nd + 2
            i_xiL = nd + 3
            i_xiR = i_xiL + M
            i_xic = i_xiR + M
            free_to_fi = {ci: fi for fi, ci in enumerate(bezier.free_indices)}

            D2 = np.zeros((max(0, N_c - 4), N_c), dtype=float)
            for rr, cidx in enumerate(range(2, N_c - 2)):
                D2[rr, cidx-2:cidx+3] = [1.0, -4.0, 6.0, -4.0, 1.0]

            def _add_cp_vec(row_vec, cp_idx, vec, sign=1.0):
                """Add sign*vec to derivative wrt a free control point."""
                fi = free_to_fi.get(cp_idx)
                if fi is not None:
                    row_vec[2*fi:2*fi+2] += sign * vec

            def _obj(x):
                dq=x[:n_free*2].reshape(n_free,2)
                dLv=x[n_free*2]; dRv=x[n_free*2+1]; mv=x[n_free*2+2]
                xi_L=x[n_free*2+3:n_free*2+3+M]
                xi_R=x[n_free*2+3+M:n_free*2+3+2*M]
                xi_c=x[n_free*2+3+2*M:]
                dq_full=np.zeros((N_c,2))
                for fi,ci in enumerate(bezier.free_indices): dq_full[ci]=dq[fi]
                Qp=Q.copy()
                for fi,ci in enumerate(bezier.free_indices): Qp[ci]+=dq[fi]
                J_sm=sum(float(np.sum((Qp[c-2]-4*Qp[c-1]+6*Qp[c]-4*Qp[c+1]+Qp[c+2])**2))
                         for c in range(2,N_c-2))
                dC=(A@dq.ravel()).reshape(M,2)
                C_new=C+dC
                J_prog=float(np.dot(C_new[-1]-z_prog,t_T)**2)
                J_step=float(np.sum(dq**2))
                return (-w_m*mv - w_W*(dLv+dRv) + w_bal*(dLv-dRv)**2
                        + w_sm*J_sm + w_prog*J_prog + w_step*J_step
                        + w_c*float(np.sum(xi_c**2))
                        + w_b*float(np.sum(xi_L**2)+np.sum(xi_R**2)))


            def _obj_jac(x):
                dq_flat = x[:nd]
                dq = dq_flat.reshape(n_free, 2)
                dLv = x[i_dL]
                dRv = x[i_dR]
                xi_L = x[i_xiL:i_xiR]
                xi_R = x[i_xiR:i_xic]
                xi_c = x[i_xic:]

                grad = np.zeros(n_vars, dtype=float)

                # width / minimum-width terms
                grad[i_dL] = -w_W + 2.0 * w_bal * (dLv - dRv)
                grad[i_dR] = -w_W - 2.0 * w_bal * (dLv - dRv)
                grad[i_m] = -w_m

                # slack penalties
                grad[i_xiL:i_xiR] = 2.0 * w_b * xi_L
                grad[i_xiR:i_xic] = 2.0 * w_b * xi_R
                grad[i_xic:] = 2.0 * w_c * xi_c

                # step penalty
                grad[:nd] += 2.0 * w_step * dq_flat

                # smoothness term: ||D2 (Q + dq)||^2, not merely ||D2 dq||^2
                if D2.shape[0] > 0:
                    Qp = Q.copy()
                    for fi, cp_idx in enumerate(bezier.free_indices):
                        Qp[cp_idx] += dq[fi]
                    L2 = D2 @ Qp
                    grad_Q = 2.0 * (D2.T @ L2)
                    for fi, cp_idx in enumerate(bezier.free_indices):
                        grad[2*fi:2*fi+2] += w_sm * grad_Q[cp_idx]

                # terminal progress soft objective
                A_end = A[-2:, :]  # 2 x nd
                dC_end = A_end @ dq_flat
                s_prog = float(np.dot(C[-1] + dC_end - z_prog, t_T))
                grad[:nd] += 2.0 * w_prog * s_prog * (A_end.T @ t_T)
                return grad

            def _constraints(x):
                dq=x[:n_free*2].reshape(n_free,2)
                dLv=x[n_free*2]; dRv=x[n_free*2+1]; mv=x[n_free*2+2]
                xi_L=x[n_free*2+3:n_free*2+3+M]
                xi_R=x[n_free*2+3+M:n_free*2+3+2*M]
                xi_c=x[n_free*2+3+2*M:]
                Qp=Q.copy()
                for fi,ci in enumerate(bezier.free_indices): Qp[ci]+=dq[fi]
                dC=(A@dq.ravel()).reshape(M,2)
                C_new=C+dC
                vals=[]
                vals.append(dLv-mv); vals.append(dRv-mv)
                vals.append(d_max-dLv); vals.append(d_max-dRv)
                # width delta bounds
                vals.append(dLv-(δL_cur-0.30)); vals.append((δL_cur+0.30)-dLv)
                vals.append(dRv-(δR_cur-0.30)); vals.append((δR_cur+0.30)-dRv)
                # left/right boundary linearised
                for j in range(M):
                    pred=D_L[j]+float(np.dot(g_L[j],C_new[j]+dLv*nn_lp[j]-x_L[j]))
                    vals.append(pred-ρ+xi_L[j])
                for j in range(M):
                    pred=D_R[j]+float(np.dot(g_R[j],C_new[j]-dRv*nn_lp[j]-x_R[j]))
                    vals.append(pred-ρ+xi_R[j])
                # centreline safety
                for j in range(M):
                    pred=D_c[j]+float(np.dot(g_c[j],C_new[j]-C[j]))
                    vals.append(pred-ρ+xi_c[j])
                # trust region on dC and dQ
                for j in range(M):
                    vals.append(r_lin**2-float(np.sum(dC[j]**2)))
                for fi in range(n_free):
                    vals.append(r_q**2-float(np.sum(dq[fi]**2)))
                # overall progress
                vals.append(float(np.dot(C_new[-1]-C_init_eval[0],t_T))-l_min)
                # start derivative: q₁ ahead of q₀, limit lateral
                if n_free>=1:
                    q1_new=Qp[1]; q0=Qp[0]
                    vals.append(float(np.dot(q1_new-q0,e0))-0.01)
                    vals.append(θ_max*abs(float(np.dot(q1_new-q0,e0))+0.01)
                                -abs(float(np.dot(q1_new-q0,n0))))
                # end derivative
                if n_free>=1:
                    vals.append(float(np.dot(Qp[-1]-Qp[-2],e_end))-0.02)
                return np.array(vals)


            def _constraints_jac(x):
                dq_flat = x[:nd]
                dq = dq_flat.reshape(n_free, 2)
                dLv = x[i_dL]
                dRv = x[i_dR]

                # Match the exact vector order returned by _constraints.
                n_cons = 8 + 3*M + M + n_free + 1
                if n_free >= 1:
                    n_cons += 3
                Jc = np.zeros((n_cons, n_vars), dtype=float)
                row = 0

                # dL-m, dR-m, dmax-dL, dmax-dR
                Jc[row, i_dL] = 1.0; Jc[row, i_m] = -1.0; row += 1
                Jc[row, i_dR] = 1.0; Jc[row, i_m] = -1.0; row += 1
                Jc[row, i_dL] = -1.0; row += 1
                Jc[row, i_dR] = -1.0; row += 1

                # width delta bounds
                Jc[row, i_dL] = 1.0; row += 1
                Jc[row, i_dL] = -1.0; row += 1
                Jc[row, i_dR] = 1.0; row += 1
                Jc[row, i_dR] = -1.0; row += 1

                # left boundary linearised constraints
                for j in range(M):
                    A_j = A[2*j:2*j+2, :]
                    Jc[row, :nd] = g_L[j] @ A_j
                    Jc[row, i_dL] = float(np.dot(g_L[j], nn_lp[j]))
                    Jc[row, i_xiL + j] = 1.0
                    row += 1

                # right boundary linearised constraints
                for j in range(M):
                    A_j = A[2*j:2*j+2, :]
                    Jc[row, :nd] = g_R[j] @ A_j
                    Jc[row, i_dR] = -float(np.dot(g_R[j], nn_lp[j]))
                    Jc[row, i_xiR + j] = 1.0
                    row += 1

                # centreline safety constraints
                for j in range(M):
                    A_j = A[2*j:2*j+2, :]
                    Jc[row, :nd] = g_c[j] @ A_j
                    Jc[row, i_xic + j] = 1.0
                    row += 1

                # sampled-curve trust region: r_lin^2 - ||dC_j||^2
                dC = (A @ dq_flat).reshape(M, 2)
                for j in range(M):
                    A_j = A[2*j:2*j+2, :]
                    Jc[row, :nd] = -2.0 * (dC[j] @ A_j)
                    row += 1

                # control-point trust region: r_q^2 - ||dq_i||^2
                for fi in range(n_free):
                    Jc[row, 2*fi:2*fi+2] = -2.0 * dq[fi]
                    row += 1

                # overall progress
                A_end = A[-2:, :]
                Jc[row, :nd] = t_T @ A_end
                row += 1

                if n_free >= 1:
                    Qp = Q.copy()
                    for fi, cp_idx in enumerate(bezier.free_indices):
                        Qp[cp_idx] += dq[fi]
                    u10 = Qp[1] - Qp[0]
                    a0 = float(np.dot(u10, e0))
                    b0 = float(np.dot(u10, n0))

                    # start forward derivative: (q1-q0)·e0 - 0.01
                    _add_cp_vec(Jc[row, :nd], 1, e0, +1.0)
                    _add_cp_vec(Jc[row, :nd], 0, e0, -1.0)
                    row += 1

                    # start lateral cone with the same nonsmooth abs expression
                    # used by the old SLSQP implementation.  At zero use a
                    # zero subgradient.
                    sa = 1.0 if (a0 + 0.01) > 0.0 else (-1.0 if (a0 + 0.01) < 0.0 else 0.0)
                    sb = 1.0 if b0 > 0.0 else (-1.0 if b0 < 0.0 else 0.0)
                    theta_vec = θ_max * sa * e0 - sb * n0
                    _add_cp_vec(Jc[row, :nd], 1, theta_vec, +1.0)
                    _add_cp_vec(Jc[row, :nd], 0, theta_vec, -1.0)
                    row += 1

                    # end derivative: (q_N-q_{N-1})·e_end - 0.02
                    _add_cp_vec(Jc[row, :nd], N_c - 1, e_end, +1.0)
                    _add_cp_vec(Jc[row, :nd], N_c - 2, e_end, -1.0)
                    row += 1

                return Jc

            bnd = ([(None,None)]*(n_free*2)
                   +[(0.0,d_max)]*2+[(0.0,None)]
                   +[(0.0,None)]*(3*M))
            x0=np.zeros(n_vars)
            x0[n_free*2]=δL_cur; x0[n_free*2+1]=δR_cur
            x0[n_free*2+2]=min(δL_cur,δR_cur)
            # warm start: push toward wider side
            shift_init=min(0.5*abs((δL_cur-δR_cur)/2),0.06)
            direction=np.sign((δL_cur-δR_cur)/2)
            dc_d=np.zeros(M*2)
            for j in range(M):
                chi=1.0 if j>0 else 0.0
                dc_d[j*2]=direction*shift_init*chi*nn_lp[j,0]
                dc_d[j*2+1]=direction*shift_init*chi*nn_lp[j,1]
            x0[:n_free*2]=A.T@dc_d

            res=minimize(_obj, x0, method="SLSQP", jac=_obj_jac, bounds=bnd,
                         constraints={"type":"ineq", "fun":_constraints, "jac":_constraints_jac},
                         options={"maxiter":100, "ftol":1e-6})
            dq=res.x[:n_free*2].reshape(n_free,2)
            for fi,ci in enumerate(bezier.free_indices): Q[ci]+=dq[fi]
            Q[0]=C_init_eval[0].copy()
            print(f"    [SLSQP] ok={res.success} dL*={res.x[n_free*2]:.3f} dR*={res.x[n_free*2+1]:.3f} |dq|={float(np.linalg.norm(dq)):.3f}", flush=True)

        # ── post‑QP real verification ────────────────────────────
        C_final=bezier.evaluate(Q,M)
        C_i=bezier.evaluate(bezier.initial_controls(),M)
        shift=float(np.max(np.linalg.norm(C_final-C_i,axis=1)))
        nn_f=self._normals(C_final); nn_flp=self._lowpass_normals(nn_f)
        δL_f=self._max_uniform_offset_df(map_data,C_final,nn_flp,+1.0,clearance)
        δR_f=self._max_uniform_offset_df(map_data,C_final,nn_flp,-1.0,clearance)
        mg=self._margin_batch(map_data,C_final,clearance)
        min_m=float(np.min(mg)); n_bad=int(np.sum(mg<-0.02))
        has_rev=any(float(np.dot(C_final[j+1]-C_final[j],C_final[1]-C_final[0]))<-0.005 for j in range(M-1)) if M>=2 else False
        st="WARN" if(min_m<-0.02 or has_rev) else "OK"
        print(f"  [v2] band_recenter max_shift={shift:.3f}m final δL={δL_f:.3f} δR={δR_f:.3f} "
              f"min_m={min_m:.3f} n_bad={n_bad} rev={has_rev} [{st}]",flush=True)
        return Q,C_final

    # ═══════════════════════════════════════════════════════════════
    #  final polish QP
    # ═══════════════════════════════════════════════════════════════

    def _final_polish_qp(
        self, map_data: MapData, band: SweptBand, bezier: _CenterlineBezier,
        Q_R: np.ndarray, C_R: np.ndarray, clearance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """QP polish: smoothness + feasibility + terminal + band non‑regression."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            return Q_R, C_R

        M = self.config.collocation_points
        n_free = len(bezier.free_indices)
        Q = Q_R.copy()

        def _normals(cc):
            nn = np.zeros_like(cc); n = len(cc)
            for j in range(n):
                if j == 0: d = cc[1] - cc[0]
                elif j == n-1: d = cc[-1] - cc[-2]
                else: d = cc[j+1] - cc[j-1]
                dn = float(np.linalg.norm(d)); t = d/dn if dn > 1e-9 else np.array([1.0,0.0])
                nn[j] = np.array([-t[1], t[0]])
            return nn

        nn_R = self._normals(C_R); nn_R_lp = self._lowpass_normals(nn_R)
        δL_R, δR_R = self._compute_widths(map_data, C_R, clearance)
        W_R = δL_R + δR_R
        σ_R = 1.0 if δL_R > δR_R else -1.0
        balanced = abs(δL_R - δR_R) < 0.03

        sg = C_R[-1]
        d_end = C_R[-1] - C_R[-2]
        dn = float(np.linalg.norm(d_end))
        t_T = d_end / dn if dn > 1e-9 else np.array([1.0, 0.0])

        J = self._bezier_jacobian(bezier, Q, M)
        A = np.zeros((M * 2, n_free * 2))
        for j in range(M):
            for dim in range(2):
                for fi in range(n_free):
                    for pd in range(2):
                        A[j*2+dim, fi*2+pd] = J[j, dim, fi, pd]

        n_vars = n_free * 2 + M + M  # dq + xi_b + xi_c

        def _obj(x):
            dq = x[:n_free*2].reshape(n_free, 2)
            xi_b = x[n_free*2:n_free*2+M]; xi_c = x[n_free*2+M:]
            dq_full = np.zeros((bezier.control_count, 2))
            for fi, ci in enumerate(bezier.free_indices): dq_full[ci] = dq[fi]
            J_sm = 0.0
            for ci in range(2, bezier.control_count-2):
                L = dq_full[ci-2]-4*dq_full[ci-1]+6*dq_full[ci]-4*dq_full[ci+1]+dq_full[ci+2]
                J_sm += float(np.sum(L**2))
            # feasibility: control point spacing
            J_feas = 0.0
            Q_c = Q.copy()
            for fi, ci in enumerate(bezier.free_indices): Q_c[ci] += dq[fi]
            for ci in range(bezier.control_count-1):
                d = float(np.linalg.norm(Q_c[ci+1]-Q_c[ci]))
                if d > 0.3: J_feas += (d-0.3)**2
            # terminal gate
            dC = (A @ dq.ravel()).reshape(M, 2)
            end_disp = dC[-1] + C_R[-1] - sg
            J_end = float(np.dot(end_disp, t_T)**2)
            return (0.5*J_sm + 0.2*J_feas + 5.0*J_end
                    + 1.0*float(np.sum(xi_b**2)) + 0.5*float(np.sum(xi_c**2)))

        def _constraints(x):
            dq = x[:n_free*2].reshape(n_free, 2)
            xi_b = x[n_free*2:n_free*2+M]; xi_c = x[n_free*2+M:]
            dC = (A @ dq.ravel()).reshape(M, 2)
            C_c = C_R + dC
            margins = self._margin_batch(map_data, C_c, clearance)
            vals = []
            # collision: margin ≥ 0
            for j in range(M):
                vals.append(float(margins[j]) + xi_c[j])
            # band non‑regression (skip if already balanced)
            if not balanced:
                eps_b = 0.02
                for j in range(M):
                    vals.append(σ_R * float(np.dot(nn_R_lp[j], dC[j])) + eps_b + xi_b[j])
            return np.array(vals)

        bounds = ([(None, None)] * (n_free*2) + [(0.0, None)] * M + [(0.0, None)] * M)
        x0 = np.zeros(n_vars)

        res = minimize(_obj, x0, method="SLSQP", bounds=bounds,
                       constraints={"type": "ineq", "fun": _constraints},
                       options={"maxiter": 60, "ftol": 1e-6})
        dq = res.x[:n_free*2].reshape(n_free, 2)

        # verify + η-blend
        for eta in [1.0, 0.5, 0.25]:
            Q_cand = Q.copy()
            for fi, ci in enumerate(bezier.free_indices): Q_cand[ci] += eta * dq[fi]
            Q_cand[0] = C_R[0].copy()
            C_cand = bezier.evaluate(Q_cand, M)
            if np.any(self._margin_batch(map_data, C_cand, clearance) < -0.02):
                continue
            nn_c = self._normals(C_cand); nn_c_lp = self._lowpass_normals(nn_c)
            δL_c, δR_c = self._compute_widths(map_data, C_cand, clearance)
            if δL_c + δR_c < W_R - 0.02:
                continue
            if any(abs(self._curvature_at(C_cand, j)) > 3.0 for j in range(1, M-1)):
                continue
            Q = Q_cand; C_R = C_cand
            break

        C_final = bezier.evaluate(Q, M)
        C_init_eval = bezier.evaluate(bezier.initial_controls(), M)
        shift = float(np.max(np.linalg.norm(C_final - C_init_eval, axis=1)))
        print(f"  [v2] band_recenter max_shift={shift:.3f}m", flush=True)
        return Q, C_final

    # ═══════════════════════════════════════════════════════════════
    #  helpers
    # ═══════════════════════════════════════════════════════════════

    def _bezier_A_matrix(
        self, bezier: _CenterlineBezier, M: int, eps: float = 0.005,
    ) -> np.ndarray:
        """Return A such that dC_flat = A @ dq_free_flat.

        This is an equivalent speed-up of the old numerical Jacobian.
        The old code recomputed ∂C/∂Q at every outer SQP iteration.
        For Bezier sampling, C(s;Q) is linear in Q, so the same matrix is
        valid for all Q.  To avoid assuming details of _CenterlineBezier's
        internal parameterization, the matrix is extracted once using the
        same evaluate()-based finite-difference convention as the old code,
        then cached.
        """
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

    def _bezier_jacobian(
        self, bezier: _CenterlineBezier, Q: np.ndarray, M: int, eps: float = 0.005,
    ) -> np.ndarray:
        """Compatibility wrapper: ∂C/∂Q[free]: [M × 2 × n_free × 2].

        Kept for final_polish_qp and any external call sites.  It returns
        the same tensor layout as the old numerical implementation, but uses
        the cached linear map.  The Q argument is intentionally unused because
        Bezier sampling is linear in Q.
        """
        n_free = len(bezier.free_indices)
        A = self._bezier_A_matrix(bezier, M, eps=eps)
        J = np.zeros((M, 2, n_free, 2), dtype=float)
        for fi in range(n_free):
            for d in range(2):
                col = A[:, fi * 2 + d].reshape(M, 2)
                J[:, :, fi, d] = col
        return J

    @staticmethod
    def _normals(cc):
        nn = np.zeros_like(cc); n = len(cc)
        for j in range(n):
            if j == 0: d = cc[1] - cc[0]
            elif j == n - 1: d = cc[-1] - cc[-2]
            else: d = cc[j + 1] - cc[j - 1]
            dn = float(np.linalg.norm(d)); t = d / dn if dn > 1e-9 else np.array([1.0, 0.0])
            nn[j] = np.array([-t[1], t[0]])
        return nn

    @staticmethod
    def _curve_tangents(cc):
        M = len(cc); tC = np.zeros((M, 2))
        for j in range(M):
            if j == 0: d = cc[1] - cc[0]
            elif j == M - 1: d = cc[-1] - cc[-2]
            else: d = cc[j + 1] - cc[j - 1]
            dn = float(np.linalg.norm(d)); tC[j] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])
        return tC

    @staticmethod
    def _lowpass_normals(nn: np.ndarray, window: int = 3) -> np.ndarray:
        """Boxcar low‑pass on normal vectors."""
        n_lp = nn.copy(); hw = window // 2
        for j in range(len(nn)):
            lo, hi = max(0, j-hw), min(len(nn), j+hw+1)
            avg = np.mean(nn[lo:hi], axis=0)
            nrm = float(np.linalg.norm(avg))
            if nrm > 1e-9: n_lp[j] = avg / nrm
        return n_lp

    @staticmethod
    def _bilinear_batch(map_data: MapData, pts: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of distance_field at array of points."""
        origin_x, origin_y = map_data.origin_xy; res = map_data.resolution
        grid_x = (pts[:,0]-origin_x)/res - 0.5; grid_y = (pts[:,1]-origin_y)/res - 0.5
        x0=np.floor(grid_x).astype(int); y0=np.floor(grid_y).astype(int)
        x1=np.minimum(x0+1,map_data.cols-1); y1=np.minimum(y0+1,map_data.rows-1)
        wx=grid_x-x0; wy=grid_y-y0; df=map_data.distance_field
        valid=(grid_x>=0)&(grid_y>=0)&(grid_x<=map_data.cols-1)&(grid_y<=map_data.rows-1)
        x0c=np.clip(x0,0,map_data.cols-1); y0c=np.clip(y0,0,map_data.rows-1)
        x1c=np.clip(x1,0,map_data.cols-1); y1c=np.clip(y1,0,map_data.rows-1)
        v00=df[y0c,x0c]; v10=df[y0c,x1c]; v01=df[y1c,x0c]; v11=df[y1c,x1c]
        r=((1-wy)*(1-wx)*v00+(1-wy)*wx*v10+wy*(1-wx)*v01+wy*wx*v11)
        r[~valid]=0.0; return r

    @staticmethod
    def _margin_batch(map_data: MapData, pts: np.ndarray, clearance: float) -> np.ndarray:
        return FormationFeasibilityV2._bilinear_batch(map_data, pts) - clearance
        """Batch query: signed margin relative to clearance threshold."""
        margins = np.array([query_distance_field(map_data, (float(p[0]), float(p[1])))
                            for p in pts], dtype=float)
        return margins - clearance  # positive = safe

    @staticmethod
    def _max_uniform_offset_df(map_data: MapData, cc: np.ndarray, nn: np.ndarray,
                                sg: float, clearance: float) -> float:
        """Max uniform offset δ such that all cc[j] + δ*sg*nn[j] are safe.

        Equivalent to the old loop over np.linspace(0, 1.50, 16), including
        the same `return d - 0.05` policy at the first unsafe offset.
        It only batches the distance-field queries.
        """
        ds = np.linspace(0.0, 1.50, 16)
        pts = cc[None, :, :] + (sg * ds[:, None, None]) * nn[None, :, :]
        margins = FormationFeasibilityV2._margin_batch(
            map_data, pts.reshape(-1, 2), clearance
        ).reshape(len(ds), len(cc))
        bad = np.any(margins < -0.02, axis=1)
        if np.any(bad):
            d = float(ds[int(np.argmax(bad))])
            return max(0.0, d - 0.05)
        return 1.50

    def _compute_widths(
        self, map_data: MapData, curve: np.ndarray, clearance: float,
    ) -> tuple[float, float]:
        K = len(curve)
        nn = self._normals(curve); nn_lp = self._lowpass_normals(nn)
        return (self._max_uniform_offset_df(map_data, curve, nn_lp, +1.0, clearance),
                self._max_uniform_offset_df(map_data, curve, nn_lp, -1.0, clearance))

    @staticmethod
    @staticmethod
    def _eval_slots_rigid(
        map_data: MapData, centre_curve: np.ndarray,
        formation: FormationSpec, clearance: float, margin_req: float,
    ) -> tuple[bool, float]:
        r"""Rigid-body slot validation.

        q_i(s_j) = C_R(s_j) + ℓ_i·t(s_j) + b_i·n(s_j)

        where t, n are the centreline tangent/normal at s_j.
        """
        n = len(centre_curve)
        if n < 2:
            return False, float("-inf")

        tangents = np.zeros((n, 2), dtype=float)
        for j in range(n):
            if j == 0:       d = centre_curve[1] - centre_curve[0]
            elif j == n - 1: d = centre_curve[-1] - centre_curve[-2]
            else:            d = centre_curve[j + 1] - centre_curve[j - 1]
            dn = float(np.linalg.norm(d))
            tangents[j] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])

        positions: list[tuple[float, float]] = []
        for j in range(n):
            cx, cy = centre_curve[j, 0], centre_curve[j, 1]
            tx, ty = tangents[j, 0], tangents[j, 1]
            nx, ny = -ty, tx  # normal (rotate tangent by +90°)
            for slot in formation.slots:
                ell, b = slot[0], slot[1]
                qx = cx + ell * tx + b * nx
                qy = cy + ell * ty + b * ny
                positions.append((float(qx), float(qy)))

        if not positions:
            return True, 0.0

        margins = FormationFeasibilityV2._margin_batch(
            map_data, np.asarray(positions, dtype=float), clearance)
        worst = float(np.min(margins))
        return worst >= margin_req, worst

    @staticmethod
    def _check_envelope(formation: FormationSpec, δL: float, δR: float) -> bool:
        """Check if formation's lateral envelope fits within available widths."""
        slots = formation.slots
        B_L = max(s[1] for s in slots)
        B_R = max(-s[1] for s in slots)
        return B_L <= δL and B_R <= δR

    @staticmethod
    def _curvature_at(curve: np.ndarray, j: int) -> float:
        """Discrete curvature (rad/m) at point j."""
        d1 = curve[j] - curve[j-1]; d2 = curve[j+1] - curve[j]
        n1 = float(np.linalg.norm(d1)); n2 = float(np.linalg.norm(d2))
        if n1 < 1e-9 or n2 < 1e-9: return 0.0
        cos_a = max(-1.0, min(1.0, float(np.dot(d1, d2))/(n1*n2)))
        return math.acos(cos_a) / (0.5*(n1+n2))
