from __future__ import annotations

import heapq
import math

import numpy as np

from formation.mpc_controller import query_distance_field
from formation.types import LocalPathWindow, LocalPreviewPath, MapData, Point2D, PreviewCurveConfig


class PreviewCurvePlanner:
    def __init__(self, config: PreviewCurveConfig | None = None) -> None:
        self.config = config or PreviewCurveConfig()
        self._last_optimizer_stats: dict[str, float | int] = {
            "outer_iterations": 0, "inner_iterations": 0,
            "unsafe_sample_count": 0, "safe_plane_count": 0,
            "max_plane_slack_m": 0.0, "mean_plane_slack_m": 0.0,
            "alpha": 0.0, "min_curve_clearance_m": 0.0,
        }

    # ── public entry ──────────────────────────────────────────────

    def plan(
        self, map_data: MapData, ref_xy: Point2D,
        waypoint_window: list[Point2D] | LocalPathWindow,
        initial_heading: float | None = None,
    ) -> LocalPreviewPath:
        window = self._normalize_window(waypoint_window)
        local_subgoal = window.local_subgoal_xy
        if self._distance(ref_xy, local_subgoal) < 1e-9:
            self._reset_optimizer_stats()
            return self._empty_result(
                source_mode="ego_warm", waypoint_window=window.points_xy,
                truncated_window=[ref_xy, local_subgoal],
                local_subgoal=local_subgoal,
                observation_distance=window.accumulated_distance_m,
                failure_reason="ref_equals_subgoal",
            )
        clearance_threshold = (
            self.config.clearance_threshold_m
            if self.config.clearance_threshold_m is not None
            else map_data.inflation_radius
        )
        truncated_window = self._prepare_window(ref_xy, window)
        if len(truncated_window) < 2:
            self._reset_optimizer_stats()
            return self._empty_result(
                source_mode="keypoint_opt", waypoint_window=window.points_xy,
                truncated_window=truncated_window, local_subgoal=local_subgoal,
                observation_distance=window.accumulated_distance_m,
                failure_reason="insufficient_points",
            )
        seed_controls = np.asarray(truncated_window, dtype=float)
        # Resample to ~8 control points for consistent DOF
        target_ct = 8
        if len(seed_controls) > 2 and len(seed_controls) < target_ct:
            pts = [(float(p[0]), float(p[1])) for p in seed_controls]
            total = self._polyline_length(pts)
            if total > 1e-6:
                spaced = self._resample_polyline(pts, total / max(target_ct - 1, 1))
                if len(spaced) >= 3:
                    spaced[0] = (float(seed_controls[0,0]), float(seed_controls[0,1]))
                    spaced[-1] = (float(seed_controls[-1,0]), float(seed_controls[-1,1]))
                    seed_controls = np.asarray(spaced, dtype=float)
        optimized_controls, _, dense_curve = self._optimize_unified_curve(
            seed_controls, map_data, local_subgoal,
            clearance_threshold, initial_heading,
        )
        if not dense_curve:
            dense_curve = [(float(p[0]), float(p[1])) for p in optimized_controls]
        sampled_points = self._resample_polyline(dense_curve, self.config.sample_spacing_m)
        if sampled_points:
            sampled_points[0] = ref_xy
        return self._build_result(
            map_data=map_data, points_xy=sampled_points,
            waypoint_window=window.points_xy, truncated_window=truncated_window,
            local_subgoal_xy=local_subgoal, source_mode="keypoint_opt",
            used_tension=self.config.bezier_tension,
            observation_distance_m=window.accumulated_distance_m,
            clearance_threshold=clearance_threshold,
        )

    # ── unified optimizer ─────────────────────────────────────────

    def _optimize_unified_curve(
        self, seed_controls: np.ndarray, map_data: MapData,
        local_subgoal: Point2D, clearance_threshold: float,
        initial_heading: float | None,
    ) -> tuple[np.ndarray, float, list[Point2D]]:
        control_count = len(seed_controls)
        if control_count <= 2:
            dense_curve = self._build_piecewise_bezier(
                [(float(p[0]), float(p[1])) for p in seed_controls],
                self.config.bezier_tension, self.config.bezier_segment_steps,
                initial_heading=initial_heading,
            )
            clearances = [self._query_clearance(map_data, p) for p in dense_curve]
            self._last_optimizer_stats = {
                "outer_iterations": 0, "inner_iterations": 0,
                "unsafe_sample_count": 0, "safe_plane_count": 0,
                "max_plane_slack_m": 0.0, "mean_plane_slack_m": 0.0,
                "alpha": 0.0,
                "min_curve_clearance_m": min(clearances) if clearances else 0.0,
            }
            return seed_controls.copy(), 0.0, dense_curve

        q = seed_controls.copy()
        q[0] = seed_controls[0]
        z_xy = np.asarray(local_subgoal, dtype=float).copy()
        goal_xy = np.asarray(local_subgoal, dtype=float)
        comfort_clearance = max(clearance_threshold, self.config.curve_pref_clearance_m)
        max_outer = 8
        total_inner = 0
        outer = 0

        for outer_idx in range(max_outer):
            outer = outer_idx + 1
            curve, seg_ids = self._sample_curve_with_segments(q, initial_heading=initial_heading)
            rebounds = self._build_rebound_guides(
                q, curve, seg_ids, map_data, clearance_threshold,
            )
            q, z_xy, inner, max_shift = self._run_inner_updates(
                q=q, z_xy=z_xy, seed_controls=seed_controls, goal_xy=goal_xy,
                rebounds=rebounds, map_data=map_data,
                comfort_clearance=comfort_clearance, initial_heading=initial_heading,
            )
            total_inner += inner
            upd_curve, upd_seg_ids = self._sample_curve_with_segments(q, initial_heading=initial_heading)
            _, upd_unsafe = self._find_unsafe_samples(upd_curve, map_data, clearance_threshold)
            if upd_unsafe == 0 and max_shift < 5e-3:
                break

        final_curve, final_seg_ids = self._sample_curve_with_segments(q, initial_heading=initial_heading)
        _, final_unsafe = self._find_unsafe_samples(final_curve, map_data, clearance_threshold)
        clearances = [self._query_clearance(map_data, p) for p in final_curve]
        self._last_optimizer_stats = {
            "outer_iterations": outer, "inner_iterations": total_inner,
            "unsafe_sample_count": final_unsafe,
            "safe_plane_count": 0,  # rebound replaces safe‑plane
            "max_plane_slack_m": 0.0, "mean_plane_slack_m": 0.0,
            "z_xy": str(tuple(z_xy)), "alpha": 0.0,
            "min_curve_clearance_m": min(clearances) if clearances else 0.0,
        }
        return q, 0.0, final_curve

    # ── inner gradient update ─────────────────────────────────────

    def _run_inner_updates(
        self, *, q, z_xy, seed_controls, goal_xy,
        rebounds, map_data, comfort_clearance, initial_heading,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        N = len(q)
        λ_s = 0.5    # smoothness (jerk)
        λ_c = 5.0    # collision (rebound)
        λ_f = 0.3    # feasibility (spacing + turning)
        λ_e = 0.5    # endpoint
        λ_za = 0.3   # z anchor
        λ_zo = 1.5   # z obstacle
        d_safe = 0.02
        d_max = 0.30  # ~ v_max·dt × 1.9 (v_max=0.8, dt=0.2)
        th_max = 0.50 # ~ omega_max·dt × 2.1 (omega_max=1.2, dt=0.2)
        step_q = 0.02
        step_z = 0.03
        tol = 2e-3
        max_inner = 60
        min_x = map_data.origin_xy[0] - 0.5
        max_x = min_x + map_data.width_m + 1.0
        min_y = map_data.origin_xy[1] - 0.5
        max_y = min_y + map_data.height_m + 1.0

        qo = q.copy()
        zo = z_xy.copy()
        last = float("inf")
        iters = 0

        for inner_idx in range(max_inner):
            iters = inner_idx + 1
            gq = np.zeros_like(qo)
            gz = np.zeros(2)

            # ── J_smooth: jerk on control points ──────────────────
            for ci in range(N - 3):
                jerk = qo[ci] - 3.0 * qo[ci + 1] + 3.0 * qo[ci + 2] - qo[ci + 3]
                for j, coeff in ((ci, 1.0), (ci + 1, -3.0), (ci + 2, 3.0), (ci + 3, -1.0)):
                    if j != 0:
                        gq[j] += λ_s * 2.0 * coeff * jerk

            # ── J_feas: control-point spacing ─────────────────────
            for ci in range(N - 1):
                seg = qo[ci + 1] - qo[ci]
                dist = float(np.linalg.norm(seg))
                if dist > d_max:
                    gq[ci] += λ_f * 2.0 * (dist - d_max) * (-seg) / max(dist, 1e-9)
                    gq[ci + 1] += λ_f * 2.0 * (dist - d_max) * seg / max(dist, 1e-9)
            # ── J_feas: turning angle ─────────────────────────────
            for ci in range(1, N - 1):
                v0 = qo[ci] - qo[ci - 1]; v1 = qo[ci + 1] - qo[ci]
                d0 = float(np.linalg.norm(v0)); d1 = float(np.linalg.norm(v1))
                if d0 < 1e-6 or d1 < 1e-6: continue
                cos_th = float(np.dot(v0, v1)) / (d0 * d1)
                cos_th = max(-1.0, min(1.0, cos_th))
                th = math.acos(cos_th)
                if th > th_max:
                    # approximate gradient of angle w.r.t. control points
                    for j, sign in ((ci - 1, -1.0), (ci, 1.0), (ci + 1, 1.0)):
                        if j != 0:
                            gq[j] += λ_f * 2.0 * (th - th_max) * sign * 0.1 * v1 / max(d1, 1e-9)

            # ── J_end: endpoint pulled toward z_xy ─────────────────
            gq[-1] += 2.0 * λ_e * (qo[-1] - zo)
            gz += 2.0 * λ_za * (zo - goal_xy)

            # ── J_zobs: z_xy away from obstacles ───────────────────
            zc = self._query_clearance(map_data, (float(zo[0]), float(zo[1])))
            zd = comfort_clearance - zc
            if zd > 0:
                zg = self._distance_gradient(map_data, (float(zo[0]), float(zo[1])))
                zgn = float(np.linalg.norm(zg))
                if zgn > 1e-9:
                    zg /= zgn
                    gz += -2.0 * λ_zo * zd * zg

            # ── J_coll: one‑sided hinge rebound ───────────────────
            for rb in rebounds:
                v = rb["v"]; y = rb["y"]; supp = rb["support"]
                if not supp: continue
                ss = 1.0 / len(supp)
                for ci in supp:
                    slack = d_safe - float(np.dot(v, qo[ci] - y))
                    if slack <= 0: continue
                    gq[ci] += (-2.0 * λ_c * slack * ss) * v

            # apply with q0 freeze
            qn = qo.copy()
            qn[1:] -= step_q * gq[1:]
            qn[0] = seed_controls[0]
            qn[:, 0] = np.clip(qn[:, 0], min_x, max_x)
            qn[:, 1] = np.clip(qn[:, 1], min_y, max_y)
            qn[0] = seed_controls[0]

            zn = zo - step_z * gz
            zn[0] = min(max(zn[0], min_x), max_x)
            zn[1] = min(max(zn[1], min_y), max_y)

            ctrl_shift = float(np.max(np.linalg.norm(qn[1:] - qo[1:], axis=1))) if N > 1 else 0.0
            ms = max(ctrl_shift, float(np.linalg.norm(zn - zo)))
            qo, zo = qn, zn
            if ms < tol and last < tol:
                last = ms
                break
            last = ms

        return qo, zo, iters, (0.0 if last == float("inf") else last)

    # ═══════════════════════════════════════════════════════════════
    #  EGO‑style rebound guide generation
    # ═══════════════════════════════════════════════════════════════

    def _build_rebound_guides(
        self, control_points: np.ndarray,
        dense_curve: list[Point2D], segment_ids: list[int],
        map_data: MapData, clearance_threshold: float,
    ) -> list[dict[str, object]]:
        """For each unsafe curve segment, generate a local collision‑free
        guide path and compute rebound directions for the affected controls."""
        # ── find collision intervals ──────────────────────────────
        intervals = self._find_collision_intervals(dense_curve, map_data, clearance_threshold)
        if not intervals:
            return []

        rebounds: list[dict[str, object]] = []
        for s_a, s_b in intervals:
            # Safe points just outside the collision interval
            x_a = np.asarray(dense_curve[max(0, s_a - 1)])
            x_b = np.asarray(dense_curve[min(len(dense_curve) - 1, s_b + 1)])
            # Skip if safe points are too close (degenerate)
            if np.linalg.norm(x_b - x_a) < 0.02:
                continue

            # ── local A* within a cropped bbox around the collision ──
            cmin = np.minimum(x_a, x_b) - 0.3
            cmax = np.maximum(x_a, x_b) + 0.3
            guide_path = self._local_astar(map_data, tuple(x_a), tuple(x_b), bbox=(cmin, cmax))
            if not guide_path or len(guide_path) < 2:
                continue

            # ── affected control indices ──────────────────────────
            seg_a = segment_ids[s_a]
            seg_b = segment_ids[min(s_b, len(segment_ids) - 1)]
            affected = set()
            for seg in range(seg_a, min(seg_b + 1, len(control_points) - 1)):
                affected.update(self._control_support_for_segment(seg, len(control_points)))

            # ── one consistent rebound direction per segment ──────
            mid_idx = (s_a + s_b) // 2
            mid_curve = np.asarray(dense_curve[mid_idx])
            # Forward direction at collision midpoint
            fwd_a = max(mid_idx - 1, 0)
            fwd_b = min(mid_idx + 1, len(dense_curve) - 1)
            fwd = np.asarray(dense_curve[fwd_b]) - np.asarray(dense_curve[fwd_a])
            fwd_n = float(np.linalg.norm(fwd))
            if fwd_n < 1e-9:
                continue
            fwd /= fwd_n

            mid_guide = self._closest_on_path(mid_curve, guide_path)
            diff = mid_guide - mid_curve
            dist = float(np.linalg.norm(diff))
            if dist < 1e-6:
                continue
            v_seg = diff / dist

            # Reject if rebound pushes backwards (angle > 90° to fwd)
            if float(np.dot(v_seg, fwd)) < 0:
                continue

            # Anchor: point on the guide path (safe side)
            a_seg = mid_guide.copy()
            # All affected controls (m ≠ 0) share the same rebound
            support = sorted(m for m in affected if m != 0)
            if support:
                rebounds.append({
                    "v": v_seg,
                    "y": a_seg,
                    "support": support,
                })
        return rebounds

    @staticmethod
    def _find_collision_intervals(
        dense_curve: list[Point2D], map_data: MapData,
        clearance_threshold: float,
    ) -> list[tuple[int, int]]:
        """Return contiguous index ranges of hard‑collision curve samples (D < d_hard)."""
        d_hard = 0.15
        unsafe = [
            float(query_distance_field(map_data, pt)) + 1e-9 < d_hard
            for pt in dense_curve
        ]
        intervals: list[tuple[int, int]] = []
        i = 0
        while i < len(unsafe):
            if unsafe[i]:
                j = i
                while j < len(unsafe) and unsafe[j]:
                    j += 1
                intervals.append((i, j - 1))
                i = j
            else:
                i += 1
        return intervals

    @staticmethod
    def _find_unsafe_samples(
        dense_curve: list[Point2D], map_data: MapData,
        clearance_threshold: float,
    ) -> tuple[list[dict[str, object]], int]:
        count = sum(
            1 for pt in dense_curve
            if float(query_distance_field(map_data, pt)) + 1e-9 < clearance_threshold
        )
        return [], count

    def _local_astar(
        self, map_data: MapData, start: tuple[float, float],
        end: tuple[float, float],
        bbox: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        r"""A* with pre‑computed clearance‑weighted cost map.

        Cost per cell:  1 + λ_clear · φ(D(x))  where
        φ(D) = [max(0, d_pref − D)]²  for D ≥ d_hard,  ∞ for D < d_hard.
        """
        ox, oy = map_data.origin_xy
        res = map_data.resolution
        rows, cols = map_data.inflated_occupancy.shape
        d_hard = 0.15
        d_pref = self.config.curve_pref_clearance_m
        λ_clear = 25.0

        sx = int((start[0] - ox) / res); sy = int((start[1] - oy) / res)
        gx = int((end[0] - ox) / res); gy = int((end[1] - oy) / res)

        if bbox is not None:
            bx0 = max(0, int((bbox[0][0] - ox) / res))
            bx1 = min(cols - 1, int((bbox[1][0] - ox) / res))
            by0 = max(0, int((bbox[0][1] - oy) / res))
            by1 = min(rows - 1, int((bbox[1][1] - oy) / res))
        else:
            bx0, bx1 = 0, cols - 1
            by0, by1 = 0, rows - 1

        def _in(i, j): return bx0 <= i <= bx1 and by0 <= j <= by1

        if not _in(sx, sy) or not _in(gx, gy):
            return [np.asarray(start), np.asarray(end)]

        # ── pre‑compute local cost map ────────────────────────────
        bb_w = bx1 - bx0 + 1; bb_h = by1 - by0 + 1
        cost_map = np.full((bb_h, bb_w), np.inf, dtype=float)
        for j in range(by0, by1 + 1):
            cy = oy + (j + 0.5) * res
            for i in range(bx0, bx1 + 1):
                cx = ox + (i + 0.5) * res
                d = float(query_distance_field(map_data, (cx, cy)))
                if d < d_hard:
                    continue  # stays inf
                deficit = max(0.0, d_pref - d)
                cost_map[j - by0, i - bx0] = 1.0 + λ_clear * deficit * deficit

        # Check start/goal are not blocked
        if not np.isfinite(cost_map[sy - by0, sx - bx0]):
            return [np.asarray(start), np.asarray(end)]
        if not np.isfinite(cost_map[gy - by0, gx - bx0]):
            return [np.asarray(start), np.asarray(end)]

        nb8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        def h(ix, iy): return math.hypot(gx - ix, gy - iy)

        open_set = [(h(sx, sy), 0.0, sx, sy)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {(sx, sy): 0.0}
        closed = set()
        expanded = 0; max_expand = 4000

        while open_set and expanded < max_expand:
            _, g, cx, cy = heapq.heappop(open_set)
            if (cx, cy) in closed: continue
            closed.add((cx, cy)); expanded += 1
            if cx == gx and cy == gy:
                path = [(cx, cy)]
                while (cx, cy) in came_from:
                    cx, cy = came_from[(cx, cy)]
                    path.append((cx, cy))
                path.reverse()
                return [np.asarray((ox + (px + 0.5) * res, oy + (py + 0.5) * res)) for px, py in path]
            for dx, dy in nb8:
                nx, ny = cx + dx, cy + dy
                if not _in(nx, ny): continue
                cell_cost = cost_map[ny - by0, nx - bx0]
                if not np.isfinite(cell_cost): continue
                ng = g + math.hypot(dx, dy) * cell_cost
                if ng < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = ng
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_set, (ng + h(nx, ny), ng, nx, ny))
        return [np.asarray(start), np.asarray(end)]

    @staticmethod
    def _closest_on_path(x: np.ndarray, path: list[np.ndarray]) -> np.ndarray:
        best = path[0]
        best_d2 = float("inf")
        for p in path:
            d2 = float(np.sum((p - x) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best = p
        return best.copy()

    # ── curve sampling ────────────────────────────────────────────

    def _sample_curve_with_segments_lite(
        self, control_points: np.ndarray, *, initial_heading: float | None,
    ) -> tuple[list[Point2D], list[int]]:
        """Like _sample_curve_with_segments but with only ~4 steps per segment."""
        if len(control_points) <= 2:
            return self._sample_curve_with_segments(control_points, initial_heading=initial_heading)
        tangents = self._compute_tangent_vectors(control_points, initial_heading=initial_heading)
        sampled: list[Point2D] = []
        seg_ids: list[int] = []
        for seg_idx in range(len(control_points) - 1):
            p0, p3 = control_points[seg_idx], control_points[seg_idx + 1]
            seg_len = float(np.linalg.norm(p3 - p0))
            if seg_len < 1e-9: continue
            cs = self.config.bezier_tension * seg_len
            p1, p2 = p0 + tangents[seg_idx] * cs, p3 - tangents[seg_idx + 1] * cs
            seg_pts = self._sample_cubic_bezier(p0, p1, p2, p3, steps=4)
            if sampled: seg_pts = seg_pts[1:]
            sampled.extend((float(x), float(y)) for x, y in seg_pts)
            seg_ids.extend([seg_idx] * len(seg_pts))
        if not sampled:
            sampled = [(float(p[0]), float(p[1])) for p in control_points]
            seg_ids = [0] * len(sampled)
        return sampled, seg_ids

    def _sample_curve_with_segments(
        self, control_points: np.ndarray, *, initial_heading: float | None,
    ) -> tuple[list[Point2D], list[int]]:
        if len(control_points) == 0:
            return [], []
        if len(control_points) == 1:
            pt = (float(control_points[0, 0]), float(control_points[0, 1]))
            return [pt], [0]
        if len(control_points) == 2:
            pts = [(float(control_points[0, 0]), float(control_points[0, 1])),
                   (float(control_points[1, 0]), float(control_points[1, 1]))]
            sampled = self._resample_polyline(pts, self.config.sample_spacing_m)
            return sampled, [0] * len(sampled)
        tangents = self._compute_tangent_vectors(control_points, initial_heading=initial_heading)
        sampled: list[Point2D] = []
        seg_ids: list[int] = []
        spacing = max(self.config.sample_spacing_m, 1e-3)
        for seg_idx in range(len(control_points) - 1):
            p0 = control_points[seg_idx]; p3 = control_points[seg_idx + 1]
            seg_len = float(np.linalg.norm(p3 - p0))
            if seg_len < 1e-9:
                continue
            cs = self.config.bezier_tension * seg_len
            p1 = p0 + tangents[seg_idx] * cs
            p2 = p3 - tangents[seg_idx + 1] * cs
            steps = 6
            seg_pts = self._sample_cubic_bezier(p0, p1, p2, p3, steps=steps)
            if sampled:
                seg_pts = seg_pts[1:]
            sampled.extend((float(x), float(y)) for x, y in seg_pts)
            seg_ids.extend([seg_idx] * len(seg_pts))
        if not sampled:
            sampled = [(float(p[0]), float(p[1])) for p in control_points]
            seg_ids = [max(min(i, len(control_points) - 2), 0) for i in range(len(control_points))]
        return sampled, seg_ids

    def _build_piecewise_bezier(
        self, points: list[Point2D], tension: float, steps: int,
        initial_heading: float | None = None,
    ) -> list[Point2D]:
        if len(points) <= 2:
            return points
        cp = np.asarray(points, dtype=float)
        tangents = self._compute_tangent_vectors(cp, initial_heading=initial_heading)
        result: list[Point2D] = []
        for idx in range(len(cp) - 1):
            p0, p3 = cp[idx], cp[idx + 1]
            seg_len = float(np.linalg.norm(p3 - p0))
            if seg_len < 1e-9:
                continue
            cs = tension * seg_len
            p1, p2 = p0 + tangents[idx] * cs, p3 - tangents[idx + 1] * cs
            seg_pts = self._sample_cubic_bezier(p0, p1, p2, p3, steps=steps)
            if result:
                seg_pts = seg_pts[1:]
            result.extend((float(x), float(y)) for x, y in seg_pts)
        return result if result else points

    def _compute_tangent_vectors(
        self, points: np.ndarray, initial_heading: float | None = None,
    ) -> np.ndarray:
        tangents = np.zeros_like(points)
        for idx in range(len(points)):
            if idx == 0:
                if initial_heading is not None:
                    direction = np.array([math.cos(initial_heading), math.sin(initial_heading)])
                else:
                    direction = points[1] - points[0]
            elif idx == len(points) - 1:
                direction = points[-1] - points[-2]
            else:
                direction = points[idx + 1] - points[idx - 1]
            norm = np.linalg.norm(direction)
            tangents[idx] = np.array([1.0, 0.0]) if norm < 1e-9 else direction / norm
        return tangents

    def _sample_cubic_bezier(
        self, p0, p1, p2, p3, steps: int,
    ) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, steps + 1)
        omt = 1.0 - ts
        return ((omt**3)[:, None] * p0 + (3.0 * omt**2 * ts)[:, None] * p1
                + (3.0 * omt * ts**2)[:, None] * p2 + (ts**3)[:, None] * p3)

    # ── helpers ───────────────────────────────────────────────────

    def _control_support_for_segment(self, seg_idx: int, N: int) -> list[int]:
        return list(range(max(1, seg_idx - 1), min(N - 1, seg_idx + 2) + 1))

    def _distance_gradient(self, map_data: MapData, pt: Point2D) -> np.ndarray:
        eps = max(map_data.resolution * 0.5, 1e-3)
        px, py = pt
        gx = (self._query_clearance(map_data, (px + eps, py))
              - self._query_clearance(map_data, (px - eps, py))) / (2.0 * eps)
        gy = (self._query_clearance(map_data, (px, py + eps))
              - self._query_clearance(map_data, (px, py - eps))) / (2.0 * eps)
        return np.asarray([gx, gy], dtype=float)

    def _prepare_window(self, ref_xy: Point2D, window: LocalPathWindow) -> list[Point2D]:
        pts: list[Point2D] = [ref_xy]
        for p in window.points_xy:
            if self._distance(pts[-1], p) >= self.config.min_point_spacing_m:
                pts.append(p)
        pts = self._deduplicate_points(pts)
        if self.config.include_ref_point and len(pts) >= 3:
            pts.pop(1)
        return pts

    def _deduplicate_points(self, pts: list[Point2D]) -> list[Point2D]:
        out: list[Point2D] = []
        for p in pts:
            if not out or self._distance(out[-1], p) >= self.config.min_point_spacing_m:
                out.append(p)
        return out

    def _resample_polyline(self, pts: list[Point2D], spacing: float) -> list[Point2D]:
        if len(pts) <= 1:
            return pts
        arr = np.asarray(pts, dtype=float)
        seg_vecs = arr[1:] - arr[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total = float(cum[-1])
        if total < 1e-9:
            return [pts[0], pts[-1]] if len(pts) > 1 else pts
        sp = max(spacing, 1e-3)
        positions = np.arange(0.0, total, sp)
        if positions.size == 0 or positions[-1] < total:
            positions = np.append(positions, total)
        result: list[Point2D] = []
        si = 0
        for arc_s in positions:
            while si < len(seg_lens) - 1 and cum[si + 1] < arc_s:
                si += 1
            sl = seg_lens[si]
            if sl < 1e-9:
                pt = arr[si]
            else:
                r = (arc_s - cum[si]) / sl
                pt = arr[si] + r * seg_vecs[si]
            result.append((float(pt[0]), float(pt[1])))
        return self._deduplicate_points(result)

    def _query_clearance(self, map_data: MapData, pt: Point2D) -> float:
        return float(query_distance_field(map_data, pt))

    def _polyline_length(self, pts: list[Point2D]) -> float:
        return sum(self._distance(pts[i - 1], pts[i]) for i in range(1, len(pts)))

    def _distance(self, a: Point2D, b: Point2D) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _reset_optimizer_stats(self) -> None:
        self._last_optimizer_stats = {
            "outer_iterations": 0, "inner_iterations": 0,
            "unsafe_sample_count": 0, "safe_plane_count": 0,
            "max_plane_slack_m": 0.0, "mean_plane_slack_m": 0.0,
            "alpha": 0.0, "min_curve_clearance_m": 0.0,
        }

    # ── result building ───────────────────────────────────────────

    def _build_result(
        self, *, map_data, points_xy, waypoint_window, truncated_window,
        local_subgoal_xy, source_mode, used_tension, observation_distance_m,
        clearance_threshold,
    ) -> LocalPreviewPath:
        arc_lengths = self._compute_arc_lengths(points_xy)
        tangents = self._compute_tangents(points_xy)
        normals = [(-t[1], t[0]) for t in tangents]
        curvatures = self._compute_curvatures(points_xy)
        cl_samples = [self._query_clearance(map_data, p) for p in points_xy]
        min_cl = min(cl_samples) if cl_samples else 0.0
        mean_cl = sum(cl_samples) / len(cl_samples) if cl_samples else 0.0
        poly_len = self._polyline_length(truncated_window)
        curve_len = self._polyline_length(points_xy)
        len_gap = abs(curve_len - poly_len)
        align_err = self._mean_polyline_distance(points_xy, truncated_window)
        is_safe = bool(cl_samples) and min_cl + 1e-3 >= clearance_threshold
        failure = "" if is_safe else "clearance_below_threshold"
        return LocalPreviewPath(
            points_xy=points_xy, arc_lengths=arc_lengths,
            tangents_xy=tangents, normals_xy=normals, curvatures=curvatures,
            source_mode=source_mode, is_safe=is_safe, min_clearance=min_cl,
            local_subgoal_xy=local_subgoal_xy,
            truncated_window_xy=truncated_window,
            observation_distance_m=observation_distance_m,
            curve_end_distance_m=curve_len,
            waypoint_window_xy=waypoint_window,
            clearance_samples=cl_samples, failure_reason=failure,
            used_tension=used_tension,
            metadata={
                "clearance_threshold_m": clearance_threshold,
                "configured_preview_distance_m": self.config.preview_distance_m,
                "mean_clearance_m": mean_cl, "alignment_error_m": align_err,
                "length_gap_m": len_gap, "curve_length_m": curve_len,
                "polyline_length_m": poly_len,
                "curve_refine_iterations": 0,
                "curve_refine_max_iteration_shift_m": 0.0,
                "curve_refine_max_net_shift_m": 0.0,
                "curve_refine_fixed_prefix_points": 1,
                "curve_refine_max_abs_raw_offset_m": 0.0,
                "curve_refine_max_abs_smoothed_offset_m": 0.0,
                "curve_refine_mean_abs_smoothed_offset_m": 0.0,
                "optimizer_outer_iterations": int(self._last_optimizer_stats.get("outer_iterations", 0)),
                "optimizer_inner_iterations": int(self._last_optimizer_stats.get("inner_iterations", 0)),
                "optimizer_unsafe_sample_count": int(self._last_optimizer_stats.get("unsafe_sample_count", 0)),
                "optimizer_safe_plane_count": int(self._last_optimizer_stats.get("safe_plane_count", 0)),
                "optimizer_max_plane_slack_m": float(self._last_optimizer_stats.get("max_plane_slack_m", 0.0)),
                "optimizer_mean_plane_slack_m": float(self._last_optimizer_stats.get("mean_plane_slack_m", 0.0)),
                "optimizer_alpha": float(self._last_optimizer_stats.get("alpha", 0.0)),
                "optimizer_min_curve_clearance_m": float(
                    self._last_optimizer_stats.get("min_curve_clearance_m", min_cl)
                ),
            },
        )

    def _compute_arc_lengths(self, pts: list[Point2D]) -> list[float]:
        if not pts: return []
        al = [0.0]
        for i in range(1, len(pts)):
            al.append(al[-1] + self._distance(pts[i - 1], pts[i]))
        return al

    def _compute_tangents(self, pts: list[Point2D]) -> list[Point2D]:
        if len(pts) == 1: return [(1.0, 0.0)]
        arr = np.asarray(pts, dtype=float)
        out: list[Point2D] = []
        for i in range(len(arr)):
            if i == 0: d = arr[1] - arr[0]
            elif i == len(arr) - 1: d = arr[-1] - arr[-2]
            else: d = arr[i + 1] - arr[i - 1]
            norm = np.linalg.norm(d)
            if norm < 1e-9: out.append((1.0, 0.0))
            else: d = d / norm; out.append((float(d[0]), float(d[1])))
        return out

    def _compute_curvatures(self, pts: list[Point2D]) -> list[float]:
        if len(pts) < 3: return [0.0] * len(pts)
        arr = np.asarray(pts, dtype=float)
        curv = [0.0]
        for i in range(1, len(arr) - 1):
            a = np.linalg.norm(arr[i] - arr[i - 1])
            b = np.linalg.norm(arr[i + 1] - arr[i])
            c = np.linalg.norm(arr[i + 1] - arr[i - 1])
            area2 = abs(np.cross(arr[i] - arr[i - 1], arr[i + 1] - arr[i - 1]))
            curv.append(float(2.0 * area2 / max(a * b * c, 1e-9)))
        curv.append(0.0)
        return curv

    def _mean_polyline_distance(self, pts, poly) -> float:
        if not pts or len(poly) < 2: return 0.0
        return sum(self._distance_to_polyline(p, poly) for p in pts) / len(pts)

    def _distance_to_polyline(self, pt, poly) -> float:
        best = math.inf
        for a, b in zip(poly[:-1], poly[1:]):
            proj = self._project_to_segment(pt, a, b)
            best = min(best, self._distance(pt, proj))
        return 0.0 if best is math.inf else best

    def _project_to_segment(self, pt, a, b) -> Point2D:
        sx, sy = b[0] - a[0], b[1] - a[1]
        l2 = sx * sx + sy * sy
        if l2 <= 1e-12: return a
        r = max(0.0, min(1.0, ((pt[0] - a[0]) * sx + (pt[1] - a[1]) * sy) / l2))
        return (a[0] + r * sx, a[1] + r * sy)

    def _normalize_window(self, w) -> LocalPathWindow:
        if isinstance(w, LocalPathWindow): return w
        pts = list(w)
        sg = pts[-1] if pts else (0.0, 0.0)
        return LocalPathWindow(
            points_xy=pts, local_subgoal_xy=sg,
            accumulated_distance_m=self._polyline_length(pts),
            source_start_index=0, source_end_index=max(len(pts) - 1, 0),
        )

    def _empty_result(self, **kw) -> LocalPreviewPath:
        return LocalPreviewPath(
            points_xy=[], arc_lengths=[], tangents_xy=[], normals_xy=[],
            curvatures=[], source_mode=kw.get("source_mode", ""),
            is_safe=False, min_clearance=0.0,
            local_subgoal_xy=kw.get("local_subgoal"),
            truncated_window_xy=kw.get("truncated_window", []),
            observation_distance_m=kw.get("observation_distance", 0.0),
            curve_end_distance_m=0.0,
            waypoint_window_xy=kw.get("waypoint_window", []),
            clearance_samples=[], failure_reason=kw.get("failure_reason", ""),
            used_tension=0.0, metadata={},
        )
