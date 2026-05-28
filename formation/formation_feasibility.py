from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from formation.assignment import compute_best_assignment
from formation.curve_band import CurveBandBuilder
from formation.embedding_qp import EmbeddingQPSolver
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
    # ── legacy (EmbeddingQP) ──────────────────────────────────────
    max_heading_offset_rad: float = 0.30
    heading_grid_size: int = 31
    lateral_grid_size: int = 31

    # ── mode ──────────────────────────────────────────────────────
    mode: str = "legacy"          # "legacy" | "swept_band"

    # ── swept‑band ────────────────────────────────────────────────
    feasibility_tol_m: float = 0.05      # SDF discretisation tolerance (~0.4 cell at 5cm resolution)
    bspline_control_count: int = 6
    collocation_points: int = 30
    verification_points: int = 60
    embed_margin_m: float = 0.0           # δ_emb
    ref_weight: float = 1.0               # w_ref
    smooth2_weight: float = 0.5           # w₁  (c'' penalty)
    smooth3_weight: float = 0.2           # w₂  (c''' penalty)
    clearance_weight: float = 0.05        # w_m  (optional)
    max_opt_iters: int = 200
    feasibility_tol: float = 1e-4


class FormationFeasibility:
    """Check whether a candidate formation fits inside the curve band.

    Two modes:
      - ``legacy``      – EmbeddingQPSolver + discrete strip cells
      - ``swept_band``  – SweptBand + B‑spline centreline optimisation
    """

    def __init__(self, config: FeasibilityConfig | None = None) -> None:
        self.config = config or FeasibilityConfig()
        self._embedding_solver = EmbeddingQPSolver(
            max_heading_offset_rad=self.config.max_heading_offset_rad,
            heading_grid_size=self.config.heading_grid_size,
            lateral_grid_size=self.config.lateral_grid_size,
        )
        self._swept_builder = SweptBandBuilder()

    # ── public ────────────────────────────────────────────────────

    def check(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_states: list[RobotState] | None = None,
    ) -> FormationFeasibilityResult:
        if self.config.mode == "swept_band":
            return self._check_swept(
                map_data, preview_path, curve_band, formation,
                robot_radius, safety_margin,
                current_formation, current_states,
            )
        return self._check_legacy(
            map_data, preview_path, curve_band, formation,
            robot_radius, safety_margin,
            current_formation, current_states,
        )

    def check_multi(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand,
        formations: list[FormationSpec],
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_states: list[RobotState] | None = None,
        *,
        preference_order: list[str] | None = None,
        stop_at_first_feasible: bool = True,
    ) -> list[FormationFeasibilityResult]:
        """Evaluate formations in *preference_order* (or as‑given).

        With *stop_at_first_feasible*, the first feasible formation
        short‑circuits the remaining (lower‑priority) candidates."""
        if preference_order is not None:
            rank = {name: i for i, name in enumerate(preference_order)}
            ordered = sorted(formations, key=lambda f: rank.get(f.name, 99))
        else:
            ordered = sorted(formations, key=lambda f: f.lateral_half_width, reverse=True)
        results: list[FormationFeasibilityResult] = []
        for fm in ordered:
            r = self.check(
                map_data, preview_path, curve_band, fm,
                robot_radius, safety_margin,
                current_formation, current_states,
            )
            results.append(r)
            if stop_at_first_feasible and r.is_feasible:
                break
        return results

    # ═══════════════════════════════════════════════════════════════
    #  swept‑band implementation  (Bézier centreline)
    # ═══════════════════════════════════════════════════════════════

    def _check_swept(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None,
        current_states: list[RobotState] | None,
    ) -> FormationFeasibilityResult:
        clearance_threshold = robot_radius + safety_margin
        band = self._swept_builder.build(map_data, preview_path, clearance_threshold)
        ref_curve = np.asarray(preview_path.points_xy, dtype=float)
        if len(ref_curve) < 2:
            return _infeasible(formation.name, "empty_preview")
        δ = self.config.embed_margin_m

        # ── min‑violation SLSQP: determine feasibility ───────────────
        bezier = _CenterlineBezier.from_reference(
            ref_curve,
            control_count=self.config.bspline_control_count,
            bezier_tension=0.35,
        )
        Q_feas, _ = self._min_violation(band, bezier, formation, δ)
        curve_feas = bezier.evaluate(Q_feas, self.config.collocation_points)
        ok_feas, worst_feas = self._eval_slots(band, curve_feas, formation, δ - self.config.feasibility_tol_m)
        if ok_feas:
            return self._build_swept_result(
                map_data, formation.name, band, curve_feas, formation,
                robot_radius, safety_margin,
                current_formation, current_states,
                metadata={"check": "min_violation_pass", "worst_margin": worst_feas},
            )

        # ── infeasible ───────────────────────────────────────────────
        return self._build_swept_result(
            map_data, formation.name, band, curve_feas, formation,
            robot_radius, safety_margin,
            current_formation, current_states,
            metadata={"check": "infeasible", "worst_margin": worst_feas},
        )

    def optimize_centerline(
        self, map_data: MapData, preview_path: LocalPreviewPath,
        formation: FormationSpec, robot_radius: float, safety_margin: float,
    ) -> FormationFeasibilityResult:
        """Bézier control‑point centreline optimisation.

        Endpoint is constrained to the transversal line at the subgoal
        (free laterally, fixed longitudinally).  All interior control
        points are free.  Hard constraint: all slots must stay inside the
        swept band.
        """
        clearance_threshold = robot_radius + safety_margin
        band = self._swept_builder.build(map_data, preview_path, clearance_threshold)
        ref_curve = np.asarray(preview_path.points_xy, dtype=float)
        bezier = _CenterlineBezier.from_reference(
            ref_curve, control_count=self.config.bspline_control_count, bezier_tension=0.35,
        )
        subgoal = bezier._ref[-1]
        # terminal tangent / normal from reference curve
        d_end = ref_curve[-1] - ref_curve[-2]
        dn = float(np.linalg.norm(d_end))
        t_T = d_end / dn if dn > 1e-9 else np.array([1.0, 0.0])
        n_T = np.array([-t_T[1], t_T[0]])

        Q_opt, feasible = self._optimal_centerline_hard(
            band, bezier, formation, self.config.embed_margin_m,
            subgoal_xy=tuple(subgoal), t_T=t_T, n_T=n_T,
        )
        curve_opt = bezier.evaluate(Q_opt, self.config.collocation_points)
        return self._build_swept_result(
            map_data, formation.name, band, curve_opt, formation,
            robot_radius, safety_margin, None, None,
            metadata={"check": "optimal_centerline" if feasible else "optimal_centerline_infeasible"},
        )

    def _optimal_centerline_beta(
        self,
        band: SweptBand,
        ref_curve: np.ndarray,
        preview_path: LocalPreviewPath,
        formation: FormationSpec,
        margin_req: float,
    ) -> tuple[np.ndarray, bool]:
        r"""Minimise lateral offsets β along the preview curve subject to
        all slots staying inside the band.

        Variable:  β[j]  (scalar lateral offset at each curve point).
        c_j(β) = γ_j + β_j · n_j
        q_{i,j}(β) = c(s_j + ξ_i) + η_i · n_c(s_j + ξ_i)

        Objective:
          J = w_β·Σ β² + w₁·Σ (Δβ/Δs)² + w₂·Σ (Δ²β/Δs²)²
              + w_c·Σ [m_tar − m_B(q)]₊²
        s.t.
          m_B(q_{i,j}(β)) ≥ margin_req   ∀ i,j
          |β_j| ≤ β_max
          |β_{j+1} − β_j| ≤ d_max
        """
        n_full = len(ref_curve)
        if n_full < 2:
            return ref_curve.copy(), False

        # save full‑resolution reference + normals
        ref_full = ref_curve.copy()
        norms_full = np.asarray(preview_path.normals_xy, dtype=float)

        # ── subsample to ~18 collocation points ────────────────────
        n = min(n_full, 18)
        idx = np.linspace(0, n_full - 1, n).astype(int)
        ref_curve = ref_curve[idx]

        # ── pre‑compute tangents / normals / arc‑length ──────────
        tangents = np.zeros((n, 2), dtype=float)
        arc = np.zeros(n, dtype=float)
        for j in range(n):
            if j == 0:
                d = ref_curve[1] - ref_curve[0]
            elif j == n - 1:
                d = ref_curve[-1] - ref_curve[-2]
            else:
                d = ref_curve[j + 1] - ref_curve[j - 1]
            dn = float(np.linalg.norm(d))
            tangents[j] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])
            if j > 0:
                arc[j] = arc[j - 1] + float(np.linalg.norm(ref_curve[j] - ref_curve[j - 1]))
        arc_total = arc[-1] if arc[-1] > 1e-6 else 1.0
        ds = arc_total / max(n - 1, 1)
        norms = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        # ── weights and bounds ────────────────────────────────────
        w_beta = 0.05
        w1 = 1.0
        w2 = 2.0
        w_c = 0.3
        m_tar = 0.10    # target margin above the hard threshold
        beta_max = 0.60
        d_max = 0.20

        def _slot_centers(beta: np.ndarray) -> np.ndarray:
            """Return all slot world positions [n_slots, 2] for the current β."""
            total_len = float(np.sum(
                np.linalg.norm(ref_curve[i + 1] - ref_curve[i])
                for i in range(n - 1)
            )) or 1.0
            # centreline: c_j = γ_j + β_j * n_j
            cl = ref_curve + beta[:, None] * norms
            # cl tangents for slot orientation
            cl_tan = np.zeros_like(cl)
            for j in range(n):
                if j == 0:
                    d = cl[1] - cl[0]
                elif j == n - 1:
                    d = cl[-1] - cl[-2]
                else:
                    d = cl[j + 1] - cl[j - 1]
                dn = float(np.linalg.norm(d))
                cl_tan[j] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])
            cl_n = np.column_stack([-cl_tan[:, 1], cl_tan[:, 0]])
            positions = []
            for j in range(n):
                tau = j / max(n - 1, 1)
                cj = cl[j]; ctan = cl_tan[j]; cn = cl_n[j]
                for slot in formation.slots:
                    te = tau + slot[0] / total_len
                    if te < 0.0 or te > 1.0:
                        continue
                    jf = min(int(te * (n - 1)), n - 2)
                    frac = te * (n - 1) - jf
                    c_eff = cl[jf] + frac * (cl[jf + 1] - cl[jf])
                    cn_eff = cl_n[jf] + frac * (cl_n[jf + 1] - cl_n[jf])
                    cn_eff = cn_eff / float(np.linalg.norm(cn_eff)) if float(np.linalg.norm(cn_eff)) > 1e-9 else cn_eff
                    q = c_eff + slot[1] * cn_eff
                    positions.append(q)
            return np.asarray(positions) if positions else np.zeros((0, 2))

        try:
            from scipy.optimize import minimize
        except ImportError:
            return ref_curve.copy(), False

        x0 = np.zeros(n, dtype=float)
        bounds = [(0.0, 0.0)] + [(-beta_max, beta_max)] * (n - 1)

        def objective(x):
            beta = np.asarray(x, dtype=float)
            J = w_beta * float(np.sum(beta ** 2))
            for j in range(n - 1):
                J += w1 * ((beta[j + 1] - beta[j]) / ds) ** 2
            for j in range(1, n - 1):
                J += w2 * ((beta[j + 1] - 2 * beta[j] + beta[j - 1]) / (ds * ds)) ** 2
            slots_xy = _slot_centers(beta)
            if len(slots_xy) > 0:
                margins = band.margin_batch(slots_xy)
                for m in margins:
                    deficit = float(m_tar - m)
                    if deficit > 0:
                        J += w_c * deficit * deficit
            return float(J)

        def constraint(x):
            beta = np.asarray(x, dtype=float)
            slots_xy = _slot_centers(beta)
            if len(slots_xy) == 0:
                return 1.0
            margins = band.margin_batch(slots_xy)
            return float(np.min(margins)) - margin_req  # ≥ 0

        res = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraint},
            options={"maxiter": 100, "ftol": 1e-4},
        )
        beta_opt = np.asarray(res.x, dtype=float)
        # Interpolate β back to full resolution
        beta_full = np.interp(np.linspace(0, 1, n_full), np.linspace(0, 1, n), beta_opt)
        curve_opt = (ref_full + beta_full[:, None] * norms_full)

        # Check feasibility at full resolution
        ok, _ = self._eval_slots(band, curve_opt, formation, margin_req - 1e-4)
        return curve_opt, ok

    def _optimal_centerline_hard(
        self,
        band: SweptBand,
        bezier: "_CenterlineBezier",
        formation: FormationSpec,
        margin_req: float,
        subgoal_xy: tuple[float, float],
        t_T: np.ndarray | None = None,
        n_T: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        r"""Solve min_Q J_len+J_sm+J_feas+J_end s.t. all slots embeddable.

        J_end = [t_T·(c(1)-subgoal)]²  (longitudinal error only)
        Endpoint is constrained to the transversal line through subgoal:
            t_T·(Q[-1] - subgoal) = 0
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            return bezier.initial_controls().copy(), False

        Q0 = bezier.initial_controls().copy()
        K = self.config.collocation_points
        bbox = bezier.bounding_box()
        sg = np.asarray(subgoal_xy, dtype=float)
        if t_T is None:
            t_T = np.array([1.0, 0.0])

        def objective(x):
            Q = Q0.copy()
            Q[bezier.free_indices] = x.reshape(-1, 2)
            curve = bezier.evaluate(Q, K)
            # J_end: longitudinal error only
            J_end = float((np.dot(curve[-1] - sg, t_T)) ** 2)
            J_sm = 0.0
            for ci in range(bezier.control_count - 3):
                j = Q[ci] - 3*Q[ci+1] + 3*Q[ci+2] - Q[ci+3]
                J_sm += float(np.sum(j**2))
            J_feas = 0.0
            for ci in range(bezier.control_count - 1):
                d = float(np.linalg.norm(Q[ci+1] - Q[ci]))
                if d > 0.3: J_feas += (d - 0.3)**2
            for ci in range(1, bezier.control_count - 1):
                v0 = Q[ci] - Q[ci-1]; v1 = Q[ci+1] - Q[ci]
                d0=float(np.linalg.norm(v0)); d1=float(np.linalg.norm(v1))
                if d0<1e-6 or d1<1e-6: continue
                cos_th = float(np.dot(v0,v1))/(d0*d1)
                cos_th = max(-1.0, min(1.0, cos_th))
                th = math.acos(cos_th)
                if th > 0.5: J_feas += (th - 0.5)**2
            J_len = float(sum(
                np.linalg.norm(Q[ci+1] - Q[ci]) for ci in range(bezier.control_count - 1)
            ))
            # ── J_lat: lateral smoothness penalty ──
            # β_j = lateral offset from straight baseline (curve[0]→[-1]).
            J_lat = 0.0
            c0 = curve[0]; cT = curve[-1]
            b_vec = cT - c0
            b_len = float(np.linalg.norm(b_vec))
            if b_len > 1e-6:
                n_b = np.array([-b_vec[1], b_vec[0]]) / b_len
                betas = np.zeros(K, dtype=float)
                for j in range(K):
                    tau_j = j / max(K - 1, 1)
                    b_j = c0 + tau_j * b_vec
                    betas[j] = float(np.dot(curve[j] - b_j, n_b))
                # Δβ: penalises rapid lateral changes
                for j in range(K - 1):
                    J_lat += (betas[j + 1] - betas[j]) ** 2
                # Δ²β: penalises lateral acceleration / S-shaped wandering
                for j in range(1, K - 1):
                    J_lat += 2.0 * (betas[j + 1] - 2.0 * betas[j] + betas[j - 1]) ** 2
            return float(J_end + 0.5*J_len + 2.0*J_sm + 0.5*J_feas + 1.0*J_lat)

        def constraint(x):
            Q = Q0.copy()
            Q[bezier.free_indices] = x.reshape(-1, 2)
            curve = bezier.evaluate(Q, K)
            _, worst = self._eval_slots(band, curve, formation, margin_req)
            return float(worst)  # must be >= margin_req

        def endpoint_eq_constraint(x):
            """Endpoint must stay on transversal line: t_T·(Q[-1] - sg) = 0"""
            Q = Q0.copy()
            Q[bezier.free_indices] = x.reshape(-1, 2)
            return float(np.dot(Q[-1] - sg, t_T))

        x0 = Q0[bezier.free_indices].ravel()
        bounds = [(bbox[0], bbox[1]), (bbox[2], bbox[3])] * len(bezier.free_indices)

        constraints = [
            {"type": "ineq", "fun": lambda x: constraint(x) - margin_req},
            {"type": "eq", "fun": endpoint_eq_constraint},
        ]

        res = minimize(
            objective, x0, method="SLSQP", bounds=bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-6},
        )
        Q_opt = Q0.copy()
        Q_opt[bezier.free_indices] = res.x.reshape(-1, 2)
        feasible = constraint(res.x) >= margin_req - 1e-4
        return Q_opt, feasible

    # ── violation functional ──────────────────────────────────────

    @staticmethod
    def _eval_slots(
        band: SweptBand,
        centre_curve: np.ndarray,   # [N, 2]  centreline c(τ)
        formation: FormationSpec,
        margin_req: float,
    ) -> tuple[bool, float]:
        r"""Check whether all formation slots stay inside the band.

        Each slot is evaluated at its true longitudinal position
        τ_eff = τ + ξ/L, using the curve tangent at τ_eff.
        """
        n = len(centre_curve)
        if n < 2:
            return False, float("-inf")
        total_len = float(sum(
            np.linalg.norm(centre_curve[i+1] - centre_curve[i])
            for i in range(n - 1)
        )) or 1.0
        # Pre‑compute tangents (unit) at each curve point
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

    @staticmethod
    def _violation(
        band: SweptBand,
        centre_curve: np.ndarray,
        formation: FormationSpec,
        margin_req: float,
    ) -> float:
        r"""Robust interval-based violation.

            m̄_{i,j} = (m_{i,j} + m_{i,j+1} − ℓ_{i,j}) / 2 − ε_sdf

        Only margins at τ_j are batch‑queried; midpoints are computed solely
        for ℓ_{i,j} (no extra SDF look‑ups).
        """
        n = len(centre_curve)
        if n < 2:
            return 0.0
        total_len = float(sum(
            np.linalg.norm(centre_curve[i+1] - centre_curve[i])
            for i in range(n - 1)
        )) or 1.0
        tangents = np.zeros((n, 2), dtype=float)
        for j in range(n):
            if j == 0: d = centre_curve[1] - centre_curve[0]
            elif j == n - 1: d = centre_curve[-1] - centre_curve[-2]
            else: d = centre_curve[j + 1] - centre_curve[j - 1]
            dn = float(np.linalg.norm(d))
            tangents[j] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])

        eps_sdf = 0.05
        slots = formation.slots
        n_slots = len(slots)

        # ── slot positions at τ_j (batch margin query) ─────────────
        q_j: list[list[np.ndarray | None]] = []
        all_q: list[np.ndarray] = []
        for slot in slots:
            row: list[np.ndarray | None] = []
            for j in range(n):
                tau = j / max(n - 1, 1)
                tau_eff = tau + slot[0] / total_len
                if tau_eff < 0.0 or tau_eff > 1.0:
                    row.append(None); continue
                jf = tau_eff * (n - 1)
                j0 = max(0, min(int(jf), n - 2)); j1 = j0 + 1
                frac = jf - j0
                c = centre_curve[j0] + frac * (centre_curve[j1] - centre_curve[j0])
                t = tangents[j0] + frac * (tangents[j1] - tangents[j0])
                tn = float(np.linalg.norm(t))
                if tn < 1e-9: row.append(None); continue
                t /= tn; n_f = np.array([-t[1], t[0]])
                p = c + slot[1] * n_f
                row.append(p); all_q.append(p)
            q_j.append(row)

        if not all_q:
            return 0.0

        margins = band.margin_batch(np.asarray(all_q, dtype=float))
        idx = 0
        m_j: list[list[float | None]] = [[None] * n for _ in range(n_slots)]
        for si in range(n_slots):
            for j in range(n):
                if q_j[si][j] is not None:
                    m_j[si][j] = float(margins[idx]); idx += 1

        # ── midpoint positions (ℓ only, no margin query) ────────────
        def _midpt(slot_idx: int, j: int) -> np.ndarray | None:
            slot = slots[slot_idx]
            tau = (j + 0.5) / max(n - 1, 1)
            tau_eff = tau + slot[0] / total_len
            if tau_eff < 0.0 or tau_eff > 1.0:
                return None
            jf = tau_eff * (n - 1)
            j0 = max(0, min(int(jf), n - 2)); j1 = j0 + 1
            frac = jf - j0
            c = centre_curve[j0] + frac * (centre_curve[j1] - centre_curve[j0])
            t = tangents[j0] + frac * (tangents[j1] - tangents[j0])
            tn = float(np.linalg.norm(t))
            if tn < 1e-9: return None
            t /= tn; n_f = np.array([-t[1], t[0]])
            return c + slot[1] * n_f

        # ── robust interval violation ──
        total = 0.0
        for si in range(n_slots):
            for j in range(n - 1):
                qa = q_j[si][j]; qb = q_j[si][j+1]
                ma = m_j[si][j]; mb = m_j[si][j+1]
                if qa is None or qb is None or ma is None or mb is None:
                    continue
                qm = _midpt(si, j)
                if qm is None:
                    continue
                ell = float(np.linalg.norm(qa - qm)) + float(np.linalg.norm(qm - qb))
                m_bar = (ma + mb - ell) / 2.0 - eps_sdf
                deficit = margin_req - m_bar
                if deficit > 0:
                    total += deficit * deficit
        return total

    # ── min‑violation ─────────────────────────────────────────────

    def _min_violation(
        self,
        band: SweptBand,
        bezier: "_CenterlineBezier",
        formation: FormationSpec,
        margin_req: float,
    ) -> tuple[np.ndarray, float]:
        r"""Solve  inf_Q  V_k[Q]  via SLSQP.

        Variables: free Bézier control points (2·(M−2) scalars).
        Objective:  V_k[Q] = Σ_{τ,i} [δ − m_B(q_i)]_+²  (sum of squares,
        smooth and differentiable).
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            return bezier.initial_controls().copy(), float("inf")

        Q0 = bezier.initial_controls().copy()
        K = self.config.collocation_points
        free_flat = Q0[bezier.free_indices].ravel()
        bbox = bezier.bounding_box()

        ref_curve = bezier.evaluate(Q0, K)
        sg = bezier._ref[-1]  # subgoal = last reference point

        def objective(x: np.ndarray) -> float:
            Q = Q0.copy()
            Q[bezier.free_indices] = x.reshape(-1, 2)
            curve = bezier.evaluate(Q, K)
            viol = FormationFeasibility._violation(band, curve, formation, margin_req)
            end_err = float(np.sum((curve[-1] - sg) ** 2))
            # Jerk smoothness (same as preview planner)
            J_sm = 0.0
            for ci in range(bezier.control_count - 3):
                j = Q[ci] - 3*Q[ci+1] + 3*Q[ci+2] - Q[ci+3]
                J_sm += float(np.sum(j**2))
            # Feasibility: spacing + turning
            J_feas = 0.0
            for ci in range(bezier.control_count - 1):
                d = float(np.linalg.norm(Q[ci+1] - Q[ci]))
                if d > 0.3: J_feas += (d - 0.3)**2
            for ci in range(1, bezier.control_count - 1):
                v0 = Q[ci] - Q[ci-1]; v1 = Q[ci+1] - Q[ci]
                d0=float(np.linalg.norm(v0)); d1=float(np.linalg.norm(v1))
                if d0<1e-6 or d1<1e-6: continue
                cos_th = float(np.dot(v0,v1))/(d0*d1)
                cos_th = max(-1.0, min(1.0, cos_th))
                th = math.acos(cos_th)
                if th > 0.5: J_feas += (th - 0.5)**2
            return 10.0 * viol + 0.5*end_err + 0.5*J_sm + 0.3*J_feas

        res = minimize(
            objective, free_flat, method="SLSQP",
            bounds=[(bbox[0], bbox[1]), (bbox[2], bbox[3])] * len(bezier.free_indices),
            options={"maxiter": 80, "ftol": 1e-6},
        )
        Q_opt = Q0.copy()
        Q_opt[bezier.free_indices] = res.x.reshape(-1, 2)
        return Q_opt, float(res.fun)

    # ── optimal centreline ────────────────────────────────────────

    def _optimal_centerline(
        self,
        band: SweptBand,
        bezier: "_CenterlineBezier",
        formation: FormationSpec,
        margin_req: float,
        warm_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        r"""Solve  min_Q  J[Q]  s.t.  V_k[Q] ≤ 0.

        J = w_ref·Σ|c − γ_ref|²  +  w_sm·Σ‖LQ‖²

        Uses SLSQP with a constraint that all slots stay inside the band.
        """
        try:
            from scipy.optimize import NonlinearConstraint, minimize
        except ImportError:
            Q = warm_start if warm_start is not None else bezier.initial_controls().copy()
            return Q, False

        Q0 = warm_start if warm_start is not None else bezier.initial_controls().copy()
        K = self.config.collocation_points
        bbox = bezier.bounding_box()
        ref_curve = bezier.evaluate(bezier.initial_controls(), K)

        def objective(x):
            Q = Q0.copy()
            Q[bezier.free_indices] = x.reshape(-1, 2)
            curve = bezier.evaluate(Q, K)
            # J = w_ref·|c−γ_ref|² + w_sm·Σ‖LQ‖²
            J_ref = self.config.ref_weight * float(np.sum((curve - ref_curve) ** 2)) / K
            J_sm = 0.0
            for ci in range(bezier.control_count):
                residual = np.zeros(2)
                stencil = [(ci - 2, 1.0), (ci - 1, -4.0), (ci, 6.0),
                           (ci + 1, -4.0), (ci + 2, 1.0)]
                for ri, coeff in stencil:
                    clamped = min(max(ri, 0), bezier.control_count - 1)
                    residual += coeff * Q[clamped]
                J_sm += float(np.sum(residual ** 2))
            return float(J_ref + self.config.smooth2_weight * J_sm)

        def constraint(x):
            Q = Q0.copy()
            Q[bezier.free_indices] = x.reshape(-1, 2)
            curve = bezier.evaluate(Q, K)
            _, worst = self._eval_slots(band, curve, formation, margin_req)
            return float(worst)  # must be ≥ margin_req

        x0 = Q0[bezier.free_indices].ravel()
        bounds = [(bbox[0], bbox[1]), (bbox[2], bbox[3])] * len(bezier.free_indices)

        res = minimize(
            objective, x0, method="SLSQP", bounds=bounds,
            constraints={"type": "ineq", "fun": lambda x: constraint(x) - margin_req},
            options={"maxiter": 30, "ftol": 1e-6},
        )
        Q_opt = Q0.copy()
        Q_opt[bezier.free_indices] = res.x.reshape(-1, 2)
        curve_opt = bezier.evaluate(Q_opt, K)
        feasible = constraint(res.x) >= margin_req - 1e-4
        return Q_opt, feasible

    # ── build result from swept‑band output ────────────────────────

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
        metadata: dict | None = None,
    ) -> FormationFeasibilityResult:
        n = len(centre_curve)
        if n < 2:
            return _infeasible(name, "empty_centreline")

        required = robot_radius + safety_margin
        headings, slot_pts_by_step = self._compute_slots(centre_curve, formation)
        # Use the same longitudinal check for feasibility
        ok_swept, worst_m = self._eval_slots(band, centre_curve, formation, -self.config.feasibility_tol_m)
        min_margin = worst_m
        # Fallback mean margin from all sampled slots (for metadata)
        margin_samples = [
            band.margin(slot_pt)
            for step_slots in slot_pts_by_step
            for slot_pt in step_slots
        ]
        mean_margin = sum(margin_samples) / len(margin_samples) if margin_samples else 0.0

        # Ground-truth clearance via distance field (for metadata / downstream)
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
                assignment=tuple(range(n_robots)), total_cost=0.0, max_cost=0.0,
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
                d = np.array([1.0, 0.0]); dn = 1.0
            t = d / dn
            h = math.atan2(t[1], t[0])
            headings.append(h)
            cos_h, sin_h = math.cos(h), math.sin(h)
            step_slots: list[Point2D] = []
            for slot in formation.slots:
                step_slots.append((
                    c[0] + cos_h * slot[0] - sin_h * slot[1],
                    c[1] + sin_h * slot[0] + cos_h * slot[1],
                ))
            slots.append(step_slots)
        return headings, slots

    # ── legacy implementation (unchanged) ──────────────────────────

    def _check_legacy(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None,
        current_states: list[RobotState] | None,
    ) -> FormationFeasibilityResult:
        if not preview_path.points_xy or not curve_band.samples:
            return _infeasible(formation.name, "empty_preview_or_band")
        embedding = self._embedding_solver.solve(preview_path, curve_band, formation)
        if not embedding.is_feasible:
            return _infeasible(formation.name, embedding.failure_reason or "embedding_infeasible", embedding=embedding)
        min_cl, mean_cl = self._slot_clearance_stats(map_data, embedding.slot_points_by_step_xy)
        required = robot_radius + safety_margin
        safety_m = min_cl - required
        feasible = (
            embedding.min_corridor_margin_m >= -1e-9
            and embedding.corridor_violation_cost <= 1e-9
        )
        if current_formation is None or current_formation.name == formation.name:
            n = len(embedding.slot_points_by_step_xy[0])
            assignment = AssignmentResult(
                assignment=tuple(range(n)), total_cost=0.0, max_cost=0.0,
                per_robot_costs=[0.0] * n,
            )
        else:
            if current_states:
                current_slots_xy = [(s.x, s.y) for s in current_states]
            else:
                current_slots_xy = _nominal_slots(
                    current_formation,
                    embedding.center_points_xy[0],
                    embedding.heading_rads[0],
                )
            assignment = compute_best_assignment(current_slots_xy, embedding.slot_points_by_step_xy[0])
        return FormationFeasibilityResult(
            formation_name=formation.name, is_feasible=feasible,
            center_points_xy=embedding.center_points_xy,
            heading_rads=embedding.heading_rads,
            slot_points_by_step_xy=embedding.slot_points_by_step_xy,
            min_corridor_margin_m=embedding.min_corridor_margin_m,
            corridor_violation_cost=embedding.corridor_violation_cost,
            min_slot_clearance_m=min_cl, mean_clearance_m=mean_cl,
            safety_margin_m=safety_m,
            offset_cost=embedding.offset_cost, heading_cost=embedding.heading_cost,
            lateral_offsets_m=list(embedding.lateral_offsets_m),
            heading_offsets_rad=list(embedding.heading_offsets_rad),
            assignment=assignment, embedding_qp_result=embedding,
            failure_reason="" if feasible else _failure_reason(feasible, safety_m),
            metadata={
                "inside_slot_count": int(embedding.metadata.get("inside_slot_count", 0)),
                "total_slot_count": int(embedding.metadata.get("total_slot_count", 0)),
                "inside_slot_ratio": float(embedding.metadata.get("inside_slot_ratio", 0.0)),
            },
        )

    def _slot_clearance_stats(
        self, map_data: MapData, slot_points_by_step: list[list[Point2D]],
    ) -> tuple[float, float]:
        clearances = [
            float(query_distance_field(map_data, slot))
            for step_slots in slot_points_by_step
            for slot in step_slots
        ]
        if not clearances:
            return 0.0, 0.0
        return min(clearances), sum(clearances) / len(clearances)


# ── helpers ────────────────────────────────────────────────────────

def _infeasible(
    name: str, reason: str, embedding: EmbeddingQPResult | None = None,
) -> FormationFeasibilityResult:
    return FormationFeasibilityResult(
        formation_name=name, is_feasible=False,
        center_points_xy=[], heading_rads=[], slot_points_by_step_xy=[],
        min_corridor_margin_m=0.0 if embedding is None else embedding.min_corridor_margin_m,
        corridor_violation_cost=0.0 if embedding is None else embedding.corridor_violation_cost,
        min_slot_clearance_m=0.0, mean_clearance_m=0.0, safety_margin_m=0.0,
        offset_cost=0.0 if embedding is None else embedding.offset_cost,
        heading_cost=0.0 if embedding is None else embedding.heading_cost,
        lateral_offsets_m=[], heading_offsets_rad=[],
        assignment=None, embedding_qp_result=embedding,
        failure_reason=reason,
    )


def _failure_reason(band_feasible: bool, safety_margin_m: float) -> str:
    if not band_feasible:
        return "slot_outside_safe_corridor"
    if safety_margin_m < -1e-9:
        return "slot_clearance_below_threshold"
    return ""


def _nominal_slots(
    formation: FormationSpec, center: Point2D, heading: float,
) -> list[Point2D]:
    cos_h = math.cos(heading); sin_h = math.sin(heading)
    return [
        (center[0] + cos_h * s[0] - sin_h * s[1],
         center[1] + sin_h * s[0] + cos_h * s[1])
        for s in formation.slots
    ]


# ── Bézier centreline ────────────────────────────────────────────

class _CenterlineBezier:
    """Piecewise cubic Bézier centreline  c(τ; Q),  τ ∈ [0, 1].

    Control points  Q = [q_0, …, q_{M−1}]  where
      q_0 = start (fixed),   q_{M−1} = end (fixed),
      interior controls are the free optimisation variables.

    Uses the same cubic‑Bézier evaluator as the preview planner,
    keeping the representation consistent across the frontend.
    """

    def __init__(
        self,
        control_count: int,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        ref_curve: np.ndarray,           # [N, 2]
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
    ) -> "_CenterlineBezier":
        return cls(
            control_count=control_count,
            start_xy=ref_curve[0].copy(),
            end_xy=ref_curve[-1].copy(),
            ref_curve=ref_curve,
            bezier_tension=bezier_tension,
        )

    def initial_controls(self) -> np.ndarray:
        """Subsample the reference curve to get M initial control points."""
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
        """Evaluate the Bézier spline at *num_points* uniformly‑spaced τ.

        Returns [num_points, 2] array of curve positions."""
        M = self.control_count
        if M <= 2:
            ts = np.linspace(0.0, 1.0, num_points)
            return np.outer(1 - ts, Q[0]) + np.outer(ts, Q[-1])

        tangents = _compute_tangents(Q)
        segs_per_step = max(1, 4)  # fixed steps per segment, resample after
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

    def _sample_at(self, Q: np.ndarray, tau: float) -> np.ndarray:
        """Evaluate a single point at parameter τ ∈ [0, 1]."""
        M = self.control_count
        if M <= 2 or tau <= 0.0:
            return Q[0].copy()
        if tau >= 1.0:
            return Q[-1].copy()
        tangents = _compute_tangents(Q)
        seg_float = tau * (M - 1)
        seg_idx = min(int(seg_float), M - 2)
        local_t = seg_float - seg_idx
        p0, p3 = Q[seg_idx], Q[seg_idx + 1]
        seg_len = float(np.linalg.norm(p3 - p0))
        if seg_len < 1e-9:
            return p0.copy()
        cs = self.tension * seg_len
        p1 = p0 + tangents[seg_idx] * cs
        p2 = p3 - tangents[seg_idx + 1] * cs
        t = local_t; omt = 1.0 - t
        return omt**3 * p0 + 3.0 * omt**2 * t * p1 + 3.0 * omt * t**2 * p2 + t**3 * p3

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) for control clamping."""
        margin = 0.80
        return (
            float(min(self.start[0], self.end[0]) - margin),
            float(max(self.start[0], self.end[0]) + margin),
            float(min(self.start[1], self.end[1]) - margin),
            float(max(self.start[1], self.end[1]) + margin),
        )


# ── helpers ──────────────────────────────────────────────────────

def _sample_ref(ref: np.ndarray, tau: float) -> np.ndarray:
    """Sample the reference curve at parameter τ ∈ [0, 1]."""
    n = len(ref)
    idx = tau * (n - 1)
    i0 = min(int(idx), n - 1)
    i1 = min(i0 + 1, n - 1)
    t = idx - i0
    return ref[i0] * (1.0 - t) + ref[i1] * t


def _compute_tangents(Q: np.ndarray) -> np.ndarray:
    """Compute unit tangent vectors for each control point."""
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
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, steps: int,
) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, steps + 1)
    omt = 1.0 - ts
    return (
        (omt**3)[:, None] * p0
        + (3.0 * omt**2 * ts)[:, None] * p1
        + (3.0 * omt * ts**2)[:, None] * p2
        + (ts**3)[:, None] * p3
    )


def _polyline_len(Q: np.ndarray) -> float:
    return float(sum(np.linalg.norm(Q[i + 1] - Q[i]) for i in range(len(Q) - 1)))


def _curve_to_bezier_controls(
    curve: np.ndarray, bezier: "_CenterlineBezier",
) -> np.ndarray:
    """Fit Bézier controls to a dense centreline curve."""
    Q = bezier.initial_controls().copy()
    K = len(curve)
    for k in range(1, bezier.control_count - 1):
        t = k / (bezier.control_count - 1)
        idx = min(int(t * (K - 1)), K - 1)
        Q[k] = curve[idx]
    return Q


def _resample_polyline(pts: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a polyline to exactly *num_points* uniformly‑spaced points."""
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
