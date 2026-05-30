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


def _make_temp_preview(pts_xy):
    """Minimal LocalPreviewPath from a list of (x, y) tuples."""
    n = len(pts_xy)
    return LocalPreviewPath(
        points_xy=pts_xy,
        arc_lengths=[0.0] * n,
        tangents_xy=[(1.0, 0.0)] * n,
        normals_xy=[(0.0, 1.0)] * n,
        curvatures=[0.0] * n,
        source_mode="v2_temp",
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
        _, C_star = self._band_recenter_qp(map_data, band, bz, C0, clearance)
        δL, δR = self._compute_widths(map_data, C_star, clearance)
        δ = self.config.embed_margin_m - self.config.feasibility_tol_m

        C_star_pts = [(float(p[0]), float(p[1])) for p in C_star]
        band_star = self._swept_builder.build(map_data, _make_temp_preview(C_star_pts), clearance)

        feasible = self._check_envelope(formation, δL, δR)
        if feasible:
            feasible, _ = self._eval_slots(band_star, C_star, formation, δ)

        return self._build_swept_result(
            map_data, formation.name, band_star, C_star, formation,
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

        # ── 2. final polish ─────────────────────────────────────
        Q_star, C_star = self._final_polish_qp(map_data, band, bz, Q_R, C_R, clearance)

        # ── 3. compute widths and rebuild band ───────────────────
        δL, δR = self._compute_widths(map_data, C_star, clearance)
        C_star_pts = [(float(p[0]), float(p[1])) for p in C_star]
        band_star = self._swept_builder.build(map_data, _make_temp_preview(C_star_pts), clearance)
        self._v2_cache = {"C_star": C_star, "C_R": C_R, "band": band_star, "delta_L": δL, "delta_R": δR}
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
                feasible, _ = self._eval_slots(band_star, C_star, fm, δ)
            results.append(self._build_swept_result(
                map_data, fm.name, band_star, C_star, fm,
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
        prev_W = 0.0

        def _max_off(cc, nn, sg):
            return self._max_uniform_offset_df(map_data, cc, nn, sg, clearance)

        for k in range(5):
            C = bezier.evaluate(Q, M)
            nn = self._normals(C)
            nn_lp = self._lowpass_normals(nn, window=3)
            δL = _max_off(C, nn_lp, +1.0); δR = _max_off(C, nn_lp, -1.0)
            W = δL + δR; μ = (δL - δR) / 2.0
            print(f"    [QP k={k}] δL={δL:.3f} δR={δR:.3f} μ={μ:.4f}", flush=True)
            direction = np.sign(μ)
            prev_W = W

            J = self._bezier_jacobian(bezier, Q, M)
            A = np.zeros((M * 2, n_free * 2))
            for j in range(M):
                for dim in range(2):
                    for fi in range(n_free):
                        for pd in range(2):
                            A[j*2+dim, fi*2+pd] = J[j, dim, fi, pd]

            t_C = np.zeros((M, 2))
            for j in range(M):
                if j == 0: d = C[1] - C[0]
                elif j == M-1: d = C[-1] - C[-2]
                else: d = C[j+1] - C[j-1]
                dn = float(np.linalg.norm(d))
                t_C[j] = d/dn if dn > 1e-9 else np.array([1.0, 0.0])

            n_vars = n_free * 2 + M  # dq + xi

            def _obj(x):
                dq = x[:n_free*2].reshape(n_free, 2); xi = x[n_free*2:]
                dq_full = np.zeros((bezier.control_count, 2))
                for fi, ci in enumerate(bezier.free_indices): dq_full[ci] = dq[fi]
                J_sm = 0.0
                for ci in range(2, bezier.control_count - 2):
                    L = dq_full[ci-2] - 4*dq_full[ci-1] + 6*dq_full[ci] - 4*dq_full[ci+1] + dq_full[ci+2]
                    J_sm += float(np.sum(L**2))
                dC = A @ dq.ravel()
                J_long = sum(float(np.dot(t_C[j], dC[j*2:j*2+2])**2) for j in range(M))
                J_start = float(np.sum(dq[0]**2))
                J_step  = float(np.sum(dq**2))
                return (0.5*J_sm + 0.3*J_long + 0.5*J_start
                        + 0.05*J_step + 2.0*float(np.sum(xi**2)))

            def _constraints(x):
                dq = x[:n_free*2].reshape(n_free, 2); xi = x[n_free*2:]
                dC = (A @ dq.ravel()).reshape(M, 2)
                vals = np.zeros(M)
                for j in range(M):
                    tau_j = j / max(M-1, 1)
                    chi = 0.0 if j == 0 else 1.0
                    vals[j] = direction * float(np.dot(nn_lp[j], dC[j])) - chi * 0.5 * abs(μ) + xi[j]
                return vals

            bounds = [(None, None)] * (n_free*2) + [(0.0, None)] * M
            x0 = np.zeros(n_vars)
            shift = min(0.5 * abs(μ), 0.06)
            dc_desired = np.zeros(M * 2)
            for j in range(M):
                chi = 1.0 if j > 0 else 0.0
                dc_desired[j*2]     = direction * shift * chi * nn_lp[j, 0]
                dc_desired[j*2 + 1] = direction * shift * chi * nn_lp[j, 1]
            x0[:n_free*2] = A.T @ dc_desired

            res = minimize(_obj, x0, method="SLSQP", bounds=bounds,
                           constraints={"type": "ineq", "fun": _constraints},
                           options={"maxiter": 50, "ftol": 1e-6})
            dq = res.x[:n_free*2].reshape(n_free, 2)
            print(f"    [SLSQP] success={res.success} |dq|={float(np.linalg.norm(dq)):.4f} |xi|={float(np.linalg.norm(res.x[n_free*2:])):.3f}", flush=True)

            # verify candidate with η-blend
            for eta in [1.0, 0.5, 0.25]:
                Q_cand = Q.copy()
                for fi, ci in enumerate(bezier.free_indices): Q_cand[ci] += eta * dq[fi]
                Q_cand[0] = C_init[0].copy()
                C_cand = bezier.evaluate(Q_cand, M)
                if np.any(self._margin_batch(map_data, C_cand, clearance) < -0.02):
                    continue
                nn_c = self._normals(C_cand); nn_lp_c = self._lowpass_normals(nn_c)
                δL_c = self._max_uniform_offset_df(map_data, C_cand, nn_lp_c, +1.0, clearance)
                δR_c = self._max_uniform_offset_df(map_data, C_cand, nn_lp_c, -1.0, clearance)
                if any(abs(self._curvature_at(C_cand, j)) > 3.0 for j in range(1, M-1)):
                    continue
                if δL_c + δR_c >= W - 0.02:
                    Q = Q_cand; C = C_cand
                    break

        C_final = bezier.evaluate(Q, M)
        C_init_eval = bezier.evaluate(bezier.initial_controls(), M)
        shift = float(np.max(np.linalg.norm(C_final - C_init_eval, axis=1)))
        print(f"  [v2] band_recenter max_shift={shift:.3f}m", flush=True)
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

    def _bezier_jacobian(
        self, bezier: _CenterlineBezier, Q: np.ndarray, M: int, eps: float = 0.005,
    ) -> np.ndarray:
        """Numerical Jacobian ∂C/∂Q[free]: [M × 2 × n_free × 2]."""
        n_free = len(bezier.free_indices)
        C0 = bezier.evaluate(Q, M)
        J = np.zeros((M, 2, n_free, 2))
        for fi, ci in enumerate(bezier.free_indices):
            for d in range(2):
                Qp = Q.copy(); Qp[ci, d] += eps
                Cp = bezier.evaluate(Qp, M)
                J[:, :, fi, d] = (Cp - C0) / eps
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
    def _margin_batch(map_data: MapData, pts: np.ndarray, clearance: float) -> np.ndarray:
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
