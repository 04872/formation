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
import osqp as _osqp
from scipy import sparse as _sparse

import osqp as _osqp
from scipy import sparse as _sparse


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
        """QP micro‑loop: precomputed gradient + Bernstein basis + OSQP."""
        Q = bezier.initial_controls().copy()
        M = self.config.collocation_points
        n_free = len(bezier.free_indices)
        N_c = bezier.control_count

        grad_x, grad_y = self._precompute_grad(map_data)
        A_tensor = self._bernstein_basis(bezier, M)  # [M, 2, N_c, 2]

        C_init_eval = bezier.evaluate(Q, M)
        d_end = C_init_eval[-1] - C_init_eval[-2]
        dn = float(np.linalg.norm(d_end))
        t_T = d_end / dn if dn > 1e-9 else np.array([1.0, 0.0])
        z_prog = C_init_eval[-1].copy()

        w_m=10.0; w_W=2.0; w_bal=1.0; w_sm=0.5; w_prog=5.0
        w_c=5.0; w_b=3.0; w_step=1.0
        ρ=clearance; d_max=1.50; r_q=0.15; θ_max=0.6

        # D2 matrix for control-point bi-Laplacian
        D2 = np.zeros((N_c-4, N_c))
        for c in range(2, N_c-2): D2[c-2, c-2:c+3] = [1, -4, 6, -4, 1]
        D2f = D2[:, bezier.free_indices]
        H_sm_block = w_sm * (D2f.T @ D2f)

        # Flatten A for dq (free CPs only)
        A_free = np.zeros((M, 2, n_free, 2))
        for fi, ci in enumerate(bezier.free_indices):
            A_free[:, :, fi, :] = A_tensor[:, :, ci, :]
        A_flat = np.zeros((M*2, n_free*2))
        for j in range(M):
            for dim in range(2):
                for fi in range(n_free):
                    for pd in range(2):
                        A_flat[j*2+dim, fi*2+pd] = A_free[j, dim, fi, pd]

        for k in range(5):
            C = bezier.evaluate(Q, M)
            nn = self._normals(C); nn_lp = self._lowpass_normals(nn, window=3)
            δL_cur = self._max_uniform_offset_batch(map_data, C, nn_lp, +1.0, clearance)
            δR_cur = self._max_uniform_offset_batch(map_data, C, nn_lp, -1.0, clearance)
            μ = (δL_cur - δR_cur) / 2.0
            print(f"    [QP k={k}] δL={δL_cur:.3f} δR={δR_cur:.3f} μ={μ:.4f}", flush=True)

            # Batch query D and gradient at linearisation points
            x_L = C + δL_cur * nn_lp; x_R = C - δR_cur * nn_lp
            all_pts = np.vstack([x_L, x_R, C])
            D_all = self._bilinear_batch(map_data, all_pts)
            gx_all = self._bilinear_query(map_data, grad_x, all_pts)
            gy_all = self._bilinear_query(map_data, grad_y, all_pts)
            D_L = D_all[0:M]; D_R = D_all[M:2*M]; D_c = D_all[2*M:3*M]
            g_L = np.column_stack([gx_all[0:M], gy_all[0:M]])
            g_R = np.column_stack([gx_all[M:2*M], gy_all[M:2*M]])
            g_c = np.column_stack([gx_all[2*M:3*M], gy_all[2*M:3*M]])

            # Start/end
            if M >= 2:
                e0 = C[1]-C[0]; e0n=float(np.linalg.norm(e0))
                e0 = e0/e0n if e0n>1e-9 else np.array([1.,0.])
                n0 = np.array([-e0[1], e0[0]])
                e_end = C[-1]-C[-2]; en=float(np.linalg.norm(e_end))
                e_end = e_end/en if en>1e-9 else np.array([1.,0.])
            else:
                e0=e_end=np.array([1.,0.]); n0=np.array([0.,1.])

            # ── Assemble sparse QP ──────────────────────────────
            n_vars = n_free*2 + 3 + 3  # dq, dL, dR, m, xi_L, xi_R, xi_c
            i_dL=n_free*2; i_dR=n_free*2+1; i_m=n_free*2+2
            i_xiL=n_free*2+3; i_xiR=n_free*2+4; i_xic=n_free*2+5

            from scipy.sparse import lil_matrix, csc_matrix
            H_sp = lil_matrix((n_vars, n_vars))
            # CP smoothness + step: H_sm_block ⊗ I₂
            for fi in range(n_free):
                for fj in range(n_free):
                    v = H_sm_block[fi, fj]
                    if abs(v) > 1e-15:
                        H_sp[fi*2,   fj*2]   += v
                        H_sp[fi*2+1, fj*2+1] += v
                    if fi == fj:
                        H_sp[fi*2,   fi*2]   += w_step
                        H_sp[fi*2+1, fi*2+1] += w_step
            # dL, dR, m: w_bal*(dL-dR)² = 2w_bal*dL² + 2w_bal*dR² - 4w_bal*dL*dR
            H_sp[i_dL, i_dL] += 2*w_bal; H_sp[i_dR, i_dR] += 2*w_bal
            H_sp[i_dL, i_dR] -= 2*w_bal; H_sp[i_dR, i_dL] -= 2*w_bal
            # Slack penalties
            H_sp[i_xiL, i_xiL] += 2*w_b; H_sp[i_xiR, i_xiR] += 2*w_b
            H_sp[i_xic, i_xic] += 2*w_c

            # progress term: w_prog * |t_T·(C[-1]+A_last·dq - z_prog)|²
            A_last = A_flat[-2:].T  # [n_free*2, 2]
            tA = A_last @ t_T  # [n_free*2]
            for ii in range(n_free*2):
                for jj in range(n_free*2):
                    vv = 2*w_prog * tA[ii] * tA[jj]
                    if abs(vv) > 1e-15: H_sp[ii, jj] += vv

            f_vec = np.zeros(n_vars)
            f_vec[i_dL] = -w_W; f_vec[i_dR] = -w_W; f_vec[i_m] = -w_m
            f_vec[:n_free*2] += 2*w_prog * float(np.dot(C[-1]-z_prog, t_T)) * tA

            H_sp = csc_matrix(H_sp)

            # ── Constraints ─────────────────────────────────────
            Gl, Gu, Gv = [], [], []
            lb, ub = [], []
            def _add(ci, vi, coeff, lo, hi):
                if abs(coeff)>1e-15: Gl.append(ci); Gv.append(coeff); Gu.append(vi)
                return lo, hi

            ci = 0
            # m ≤ dL, m ≤ dR, bounds
            lo,_=_add(ci,i_dL,1.,0,1e9); _,_=_add(ci,i_m,-1.,0,1e9); lb.append(lo);ub.append(1e9);ci+=1
            lo,_=_add(ci,i_dR,1.,0,1e9); _,_=_add(ci,i_m,-1.,0,1e9); lb.append(lo);ub.append(1e9);ci+=1
            lo,_=_add(ci,i_dL,1.,0,d_max); lb.append(0);ub.append(d_max);ci+=1
            lo,_=_add(ci,i_dR,1.,0,d_max); lb.append(0);ub.append(d_max);ci+=1
            lo,_=_add(ci,i_dL,1.,δL_cur-0.30,δL_cur+0.30);lb.append(δL_cur-0.30);ub.append(δL_cur+0.30);ci+=1
            lo,_=_add(ci,i_dR,1.,δR_cur-0.30,δR_cur+0.30);lb.append(δR_cur-0.30);ub.append(δR_cur+0.30);ci+=1

            for j in range(M):
                Adq_j = A_free[j]  # [2, n_free, 2]
                A_j_flat = np.zeros((2, n_free*2))
                for fi in range(n_free):
                    A_j_flat[0, fi*2] = Adq_j[0, fi, 0]; A_j_flat[0, fi*2+1] = Adq_j[0, fi, 1]
                    A_j_flat[1, fi*2] = Adq_j[1, fi, 0]; A_j_flat[1, fi*2+1] = Adq_j[1, fi, 1]
                # Left boundary
                coeff_L = g_L[j,0]*A_j_flat[0] + g_L[j,1]*A_j_flat[1]
                c_dL = float(np.dot(g_L[j], nn_lp[j]))
                rhs_L = ρ + float(np.dot(g_L[j], x_L[j])) - D_L[j] - float(np.dot(g_L[j], C[j]))
                lo=0; hi=0
                for ii in range(n_free*2): lo,_=_add(ci,ii,coeff_L[ii],0,0)
                lo,_=_add(ci,i_dL,c_dL,0,0); lo,_=_add(ci,i_xiL,1.,0,0)
                lb.append(rhs_L); ub.append(1e9); ci+=1
                # Right boundary
                coeff_R = g_R[j,0]*A_j_flat[0] + g_R[j,1]*A_j_flat[1]
                c_dR = -float(np.dot(g_R[j], nn_lp[j]))
                rhs_R = ρ + float(np.dot(g_R[j], x_R[j])) - D_R[j] - float(np.dot(g_R[j], C[j]))
                for ii in range(n_free*2): lo,_=_add(ci,ii,coeff_R[ii],0,0)
                lo,_=_add(ci,i_dR,c_dR,0,0); lo,_=_add(ci,i_xiR,1.,0,0)
                lb.append(rhs_R); ub.append(1e9); ci+=1
                # Centreline
                coeff_c = g_c[j,0]*A_j_flat[0] + g_c[j,1]*A_j_flat[1]
                rhs_c = ρ - D_c[j]
                for ii in range(n_free*2): lo,_=_add(ci,ii,coeff_c[ii],0,0)
                lo,_=_add(ci,i_xic,1.,0,0)
                lb.append(rhs_c); ub.append(1e9); ci+=1

            # Trust region on dq
            for fi in range(n_free):
                lo,_=_add(ci,fi*2,1.,-r_q,r_q); lo,_=_add(ci,fi*2+1,1.,-r_q,r_q)
                lb.append(-r_q); ub.append(r_q); ci+=1
                lb.append(-r_q); ub.append(r_q); ci+=1

            # Start/end derivative
            if n_free >= 1:
                lo,_=_add(ci,0,e0[0],0.01,1e9); lo,_=_add(ci,1,e0[1],0.01,1e9)
                lb.append(0.01); ub.append(1e9); ci+=1
                lo,_=_add(ci,0,n0[0]-e0[0]*θ_max,-1e9,1e9)
                lo,_=_add(ci,1,n0[1]-e0[1]*θ_max,-1e9,1e9)
                lb.append(-θ_max*0.01); ub.append(1e9); ci+=1

            # Slacks ≥ 0
            lo,_=_add(ci,i_xiL,1.,0,1e9); lb.append(0); ub.append(1e9); ci+=1
            lo,_=_add(ci,i_xiR,1.,0,1e9); lb.append(0); ub.append(1e9); ci+=1
            lo,_=_add(ci,i_xic,1.,0,1e9); lb.append(0); ub.append(1e9); ci+=1

            G_sp = csc_matrix((Gv, (Gl, Gu)), shape=(ci, n_vars))
            l_arr = np.array(lb); u_arr = np.array(ub)

            # ── OSQP ────────────────────────────────────────────
            prob = _osqp.OSQP()
            prob.setup(H_sp, f_vec, G_sp, l_arr, u_arr,
                       eps_abs=1e-5, eps_rel=1e-5, max_iter=200,
                       verbose=False)
            res = prob.solve()
            if res.info.status_val in (1, 2):
                dq = res.x[:n_free*2].reshape(n_free, 2)
                for fi, ci in enumerate(bezier.free_indices): Q[ci] += dq[fi]
                Q[0] = C_init_eval[0].copy()
            dq_n = float(np.linalg.norm(res.x[:n_free*2])) if res.x is not None else 0
            print(f"    [OSQP] ok={res.info.status_val in (1,2)} dL*={res.x[i_dL]:.3f} dR*={res.x[i_dR]:.3f} |dq|={dq_n:.3f}", flush=True)

        C_final = bezier.evaluate(Q, M)
        C_i = bezier.evaluate(bezier.initial_controls(), M)
        shift = float(np.max(np.linalg.norm(C_final - C_i, axis=1)))
        nn_f = self._normals(C_final); nn_flp = self._lowpass_normals(nn_f)
        δL_f = self._max_uniform_offset_batch(map_data, C_final, nn_flp, +1.0, clearance)
        δR_f = self._max_uniform_offset_batch(map_data, C_final, nn_flp, -1.0, clearance)
        mg = self._margin_batch(map_data, C_final, clearance)
        min_m = float(np.min(mg)); n_bad = int(np.sum(mg < -0.02))
        has_rev = any(float(np.dot(C_final[j+1]-C_final[j], C_final[1]-C_final[0])) < -0.005 for j in range(M-1)) if M >= 2 else False
        st = "WARN" if (min_m < -0.02 or has_rev) else "OK"
        print(f"  [v2] band_recenter max_shift={shift:.3f}m final δL={δL_f:.3f} δR={δR_f:.3f} "
              f"min_m={min_m:.3f} n_bad={n_bad} rev={has_rev} [{st}]", flush=True)
        return Q, C_final

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


    @staticmethod
    def _precompute_grad(map_data: MapData):
        """Precompute distance-field gradient grids."""
        df = map_data.distance_field
        r = map_data.resolution
        rows, cols = df.shape
        gx = np.zeros_like(df); gy = np.zeros_like(df)
        gx[:, 1:-1] = (df[:, 2:] - df[:, :-2]) / (2 * r)
        gy[1:-1, :] = (df[2:, :] - df[:-2, :]) / (2 * r)
        return gx, gy

    @staticmethod
    def _bilinear_query(map_data: MapData, grid: np.ndarray, pts: np.ndarray):
        """Bilinear interpolation on an arbitrary grid."""
        ox, oy = map_data.origin_xy; res = map_data.resolution
        gx_v = (pts[:, 0] - ox) / res - 0.5; gy_v = (pts[:, 1] - oy) / res - 0.5
        x0 = np.floor(gx_v).astype(int); y0 = np.floor(gy_v).astype(int)
        x1 = np.minimum(x0 + 1, map_data.cols - 1); y1 = np.minimum(y0 + 1, map_data.rows - 1)
        wx = gx_v - x0; wy = gy_v - y0
        x0c = np.clip(x0, 0, map_data.cols - 1); y0c = np.clip(y0, 0, map_data.rows - 1)
        x1c = np.clip(x1, 0, map_data.cols - 1); y1c = np.clip(y1, 0, map_data.rows - 1)
        v00 = grid[y0c, x0c]; v10 = grid[y0c, x1c]; v01 = grid[y1c, x0c]; v11 = grid[y1c, x1c]
        return (1 - wy) * (1 - wx) * v00 + (1 - wy) * wx * v10 + wy * (1 - wx) * v01 + wy * wx * v11

    @staticmethod
    def _bernstein_basis(bezier: _CenterlineBezier, M: int):
        """Precompute Bernstein basis A: C(s_j; Q) = A_j @ Q for all j."""
        n_cp = bezier.control_count
        Q0 = bezier.initial_controls()
        A_full = np.zeros((M * 2, n_cp * 2))
        eps = 0.005
        for ci in range(n_cp):
            for d in range(2):
                Qp = Q0.copy(); Qp[ci, d] += eps
                Cp = bezier.evaluate(Qp, M)
                C0 = bezier.evaluate(Q0, M)
                # Each CP affects C_j via ∂C_j/∂q_ci. Build full matrix.
                for j in range(M):
                    A_full[j*2 + 0, ci*2 + d] += (Cp[j, 0] - C0[j, 0]) / eps
                    A_full[j*2 + 0, ci*2 + 0:ci*2+2] = 0  # reset — only [ci,d] column
        # Proper build: one column at a time
        A_full = np.zeros((M * 2, n_cp * 2))
        for ci in range(n_cp):
            for d in range(2):
                Qp = Q0.copy(); Qp[ci, d] += eps
                Cp = bezier.evaluate(Qp, M)
                C0_val = bezier.evaluate(Q0, M)
                for j in range(M):
                    A_full[j*2 + d, ci*2 + d] = (Cp[j, d] - C0_val[j, d]) / eps
        # Compact form: [M, 2, n_cp, 2]
        A_tensor = np.zeros((M, 2, n_cp, 2))
        for ci in range(n_cp):
            for d in range(2):
                Qp = Q0.copy(); Qp[ci, d] += eps
                Cp = bezier.evaluate(Qp, M)
                C0_val = bezier.evaluate(Q0, M)
                A_tensor[:, :, ci, d] = (Cp - C0_val) / eps
        return A_tensor

    @staticmethod
    def _max_uniform_offset_batch(map_data: MapData, cc: np.ndarray, nn: np.ndarray,
                                   sg: float, clearance: float, n_steps: int = 16):
        """Vectorized max uniform offset: batch all d values into one query."""
        ds = np.linspace(0.0, 1.50, n_steps)
        M = len(cc)
        # Build [n_steps, M, 2] then reshape
        pts_all = cc[None, :, :] + ds[:, None, None] * (sg * nn)[None, :, :]
        pts_flat = pts_all.reshape(-1, 2)
        margins = FormationFeasibilityV2._margin_batch(map_data, pts_flat, clearance)
        margins_2d = margins.reshape(n_steps, M)
        unsafe = np.any(margins_2d < -0.02, axis=1)
        idx = np.argmax(unsafe)
        if idx == 0 or not unsafe[idx]:
            return ds[-1]
        return max(0.0, ds[idx - 1])

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
        """Max uniform offset δ such that all cc[j] + δ*sg*nn[j] are safe."""
        for d in np.linspace(0.0, 1.50, 16):
            pts = cc + (sg * d) * nn
            margins = FormationFeasibilityV2._margin_batch(map_data, pts, clearance)
            if np.any(margins < -0.02):  # 2cm tolerance for distance-field discretisation
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
