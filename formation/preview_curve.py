from __future__ import annotations

import math

import numpy as np

from formation.mpc_controller import query_distance_field
from formation.types import LocalPathWindow, LocalPreviewPath, MapData, Point2D, PreviewCurveConfig


class PreviewCurvePlanner:
    def __init__(self, config: PreviewCurveConfig | None = None) -> None:
        self.config = config or PreviewCurveConfig()
        self._last_refine_stats = {
            "iterations": 0,
            "max_iteration_shift_m": 0.0,
            "max_net_shift_m": 0.0,
            "fixed_prefix_points": 0,
            "max_abs_raw_offset_m": 0.0,
            "max_abs_smoothed_offset_m": 0.0,
            "mean_abs_smoothed_offset_m": 0.0,
        }

    def plan(
        self,
        map_data: MapData,
        ref_xy: Point2D,
        waypoint_window: list[Point2D] | LocalPathWindow,
        initial_heading: float | None = None,
    ) -> LocalPreviewPath:
        window = self._normalize_window(waypoint_window)
        local_subgoal = window.local_subgoal_xy
        if self._distance(ref_xy, local_subgoal) < 1e-9:
            self._warm_q = None
            return self._empty_result(
                source_mode="ego_warm",
                waypoint_window=window.points_xy,
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

        # ── Optimize A* keypoints directly (5 control points) ──
        keypoints = [ref_xy] + window.points_xy
        q = np.asarray(keypoints, dtype=float)
        q = self._push_out_of_obstacles(q, map_data, clearance_threshold)
        if len(q) < 2:
            return self._empty_result(
                source_mode="keypoint_opt",
                waypoint_window=window.points_xy,
                truncated_window=keypoints,
                local_subgoal=local_subgoal,
                observation_distance=window.accumulated_distance_m,
                failure_reason="insufficient_points",
            )
        q = self._optimize_with_target(q, local_subgoal, map_data, clearance_threshold)

        # Phase 2: densify to ~0.3m for Bezier anchor points, light obstacle push
        pts = [(float(p[0]), float(p[1])) for p in q]
        denser = self._resample_polyline(pts, 0.3)
        qd = np.asarray(denser, dtype=float)
        qd = self._push_out_of_obstacles(qd, map_data, clearance_threshold)
        qd = self._optimize_keypoints(qd, local_subgoal, map_data, clearance_threshold)
        points_xy = [(float(p[0]), float(p[1])) for p in qd]

        # ── Bezier smooth ──
        bezier_points = self._build_piecewise_bezier(
            points_xy, self.config.bezier_tension,
            steps=self.config.bezier_segment_steps,
        )
        sampled = self._resample_polyline(bezier_points, self.config.sample_spacing_m)

        # ── Final safety push ──
        q_sampled = np.asarray(sampled, dtype=float)
        q_sampled = self._push_out_of_obstacles(q_sampled, map_data, clearance_threshold)
        sampled = [(float(p[0]), float(p[1])) for p in q_sampled]

        curve_length = self._polyline_length(sampled)
        return self._build_result(
            map_data, sampled, window.points_xy, keypoints,
            local_subgoal, "keypoint_opt", self.config.bezier_tension,
            window.accumulated_distance_m, curve_length, clearance_threshold,
        )

    def _push_out_of_obstacles(
        self,
        q: "np.ndarray",
        map_data: MapData,
        clearance_threshold: float,
    ) -> "np.ndarray":
        """Gradient-ascent push: move each point into free space (D ≥ threshold)."""
        result = [q[0]]  # q_0 frozen
        eps = map_data.resolution * 0.5
        for i in range(1, len(q)):
            px, py = float(q[i, 0]), float(q[i, 1])
            for _ in range(20):
                cl = float(query_distance_field(map_data, (px, py)))
                if cl >= clearance_threshold - 1e-9:
                    break
                gx = (float(query_distance_field(map_data, (px + eps, py)))
                      - float(query_distance_field(map_data, (px - eps, py)))) / (2.0 * eps)
                gy = (float(query_distance_field(map_data, (px, py + eps)))
                      - float(query_distance_field(map_data, (px, py - eps)))) / (2.0 * eps)
                gn = math.hypot(gx, gy)
                if gn < 1e-9:
                    break
                step = (clearance_threshold - cl) * 0.6
                px += (gx / gn) * step
                py += (gy / gn) * step
            result.append(np.array([px, py]))
        return np.asarray(result, dtype=float)

    def _optimize_curve_points(
        self,
        q: "np.ndarray",
        guide_points: "np.ndarray",
        local_subgoal: Point2D,
        map_data: MapData,
        clearance_threshold: float,
    ) -> "np.ndarray":
        r"""Multi-objective gradient descent on q_1..q_N.

        J = w_obs·Σ max(0, d_pref - D)²  [obstacle: push toward comfort zone]
          + w_smooth·Σ ‖Lq‖²             [smoothness]
          + Σ α_i·w_guide·max(0, d_i-r)² [tube: weak at start, ramp up]
          + w_goal·‖q_N - g‖²            [soft endpoint]
        """
        N = len(q)
        if N <= 2:
            return q

        d_pref = self.config.curve_pref_clearance_m  # ~0.45 m
        w_obs = 4.0
        w_smooth = 1.2
        w_guide = 0.1
        w_goal = 0.4
        step_size = 0.025
        max_iters = 200
        tol = 0.002
        eps = map_data.resolution * 0.5
        r_tube = 0.6

        q_opt = q.copy()
        goal_arr = np.array(local_subgoal, dtype=float)
        guide_polyline = [(float(p[0]), float(p[1])) for p in guide_points]

        # Position-dependent tube weight: near zero at start, ramps up after 30%
        def tube_weight(i: int) -> float:
            if N <= 1:
                return 0.0
            t = i / (N - 1)             # 0 at start, 1 at end
            if t < 0.3:
                return 0.0               # first 30%: no tube pull
            return ((t - 0.3) / 0.7) ** 2  # ramp up quadratically

        last_max_shift = float("inf")
        for _ in range(max_iters):
            grad = np.zeros_like(q_opt)
            for i in range(1, N):
                g = np.zeros(2, dtype=float)
                px, py = float(q_opt[i, 0]), float(q_opt[i, 1])

                # (1) Obstacle: push toward d_pref
                D_i = float(query_distance_field(map_data, (px, py)))
                deficit = d_pref - D_i
                if deficit > 0:
                    dx = (float(query_distance_field(map_data, (px + eps, py)))
                          - float(query_distance_field(map_data, (px - eps, py)))) / (2.0 * eps)
                    dy = (float(query_distance_field(map_data, (px, py + eps)))
                          - float(query_distance_field(map_data, (px, py - eps)))) / (2.0 * eps)
                    gn = math.hypot(dx, dy)
                    if gn > 1e-9:
                        g -= (2.0 * w_obs * deficit) * np.array([dx, dy]) / gn

                # (2) Guide: tube constraint with position-dependent weight
                a_i = tube_weight(i)
                if a_i > 0:
                    d_i = self._distance_to_polyline((px, py), guide_polyline)
                    if d_i > r_tube:
                        cp = self._closest_point_on_polyline((px, py), guide_polyline)
                        to_q = np.array([px - cp[0], py - cp[1]])
                        d = math.hypot(to_q[0], to_q[1])
                        if d > 1e-9:
                            g += a_i * w_guide * 2.0 * (d_i - r_tube) * to_q / d

                # (3) Smoothness: bi-Laplacian
                sg = np.zeros(2, dtype=float)
                _c = lambda j: q_opt[max(0, min(j, N - 1))]
                sg += _c(i - 2) + (-4.0) * _c(i - 1) + 6.0 * q_opt[i] + (-4.0) * _c(i + 1) + _c(i + 2)
                g += (2.0 * w_smooth) * sg

                # (4) Soft goal (q_N only)
                if i == N - 1:
                    g += (2.0 * w_goal) * (q_opt[i] - goal_arr)

                grad[i] = g

            q_opt[1:] -= step_size * grad[1:]
            min_x = map_data.origin_xy[0] - 0.5
            max_x = min_x + map_data.width_m + 1.0
            min_y = map_data.origin_xy[1] - 0.5
            max_y = min_y + map_data.height_m + 1.0
            q_opt[:, 0] = np.clip(q_opt[:, 0], min_x, max_x)
            q_opt[:, 1] = np.clip(q_opt[:, 1], min_y, max_y)

            max_shift = float(np.max(np.linalg.norm(step_size * grad[1:], axis=1))) if N > 1 else 0.0
            if max_shift < tol and last_max_shift < tol:
                break
            last_max_shift = max_shift

        return q_opt

    def _closest_point_on_polyline(
        self,
        warm_q: "np.ndarray",
        ref_xy: Point2D,
        astar_polyline: list[Point2D],
        spacing: float,
    ) -> "np.ndarray":
        """Trim passed segment, extend tail via A* corridor. No blend hack."""
        ref_arr = np.array(ref_xy, dtype=float)
        dists = np.linalg.norm(warm_q - ref_arr, axis=1)
        anchor_idx = int(np.argmin(dists))
        q = warm_q[anchor_idx:].copy()
        q[0] = ref_arr  # anchor start to current position
        if len(q) < 2:
            return q

        current_length = self._polyline_length([(float(p[0]), float(p[1])) for p in q])
        target_length = 3.0
        if current_length < target_length:
            tail_pts = self._resample_polyline(astar_polyline, spacing)
            tail_arr = np.asarray(tail_pts, dtype=float)
            tail_dists = np.linalg.norm(tail_arr - q[-1], axis=1)
            ahead = tail_arr[tail_dists > spacing * 0.5]
            if len(ahead) == 0:
                ahead = np.array([q[-1] + np.array([spacing, 0.0])])
            added = 0.0
            extra = []
            for pt in ahead:
                if current_length + added >= target_length:
                    break
                extra.append(pt)
                added += float(np.linalg.norm(pt - (extra[-2] if len(extra) >= 2 else q[-1]))) if extra else float(np.linalg.norm(pt - q[-1]))
            if extra:
                q = np.vstack([q, np.array(extra)])
        return q

    def _local_obstacle_fix(
        self,
        q: "np.ndarray",
        map_data: MapData,
        clearance_threshold: float,
    ) -> "np.ndarray":
        """Push only unsafe points (D < threshold) and their neighbours."""
        N = len(q)
        if N <= 1:
            return q
        eps = map_data.resolution * 0.5
        q_fixed = q.copy()
        max_push = 0.3  # max displacement per fix pass
        for _ in range(5):  # a few local passes
            fixed_any = False
            for i in range(1, N - 1):  # skip endpoints
                D_i = float(query_distance_field(map_data, (float(q_fixed[i, 0]), float(q_fixed[i, 1]))))
                if D_i >= clearance_threshold - 1e-9:
                    continue
                deficit = clearance_threshold - D_i
                dx = (float(query_distance_field(map_data, (float(q_fixed[i, 0]) + eps, float(q_fixed[i, 1]))))
                      - float(query_distance_field(map_data, (float(q_fixed[i, 0]) - eps, float(q_fixed[i, 1]))))) / (2.0 * eps)
                dy = (float(query_distance_field(map_data, (float(q_fixed[i, 0]), float(q_fixed[i, 1]) + eps)))
                      - float(query_distance_field(map_data, (float(q_fixed[i, 0]), float(q_fixed[i, 1]) - eps)))) / (2.0 * eps)
                gn = math.hypot(dx, dy)
                if gn < 1e-9:
                    continue
                push = min(deficit * 0.5, max_push) * np.array([dx, dy]) / gn
                q_fixed[i] += push
                # Also nudge neighbours
                if i > 1:
                    q_fixed[i - 1] += push * 0.3
                if i < N - 2:
                    q_fixed[i + 1] += push * 0.3
                fixed_any = True
            if not fixed_any:
                break
        return q_fixed

    def _optimize_with_target(
        self,
        q: "np.ndarray",
        local_subgoal: Point2D,
        map_data: MapData,
        clearance_threshold: float,
    ) -> "np.ndarray":
        r"""Optimize q_1..q_N with optimizable local target z = g + α·n.

        min  w_obs·Σ hinge(d_pref - D(q_i))²     [obstacle]
           + w_smooth·Σ ‖Lq‖²                     [smoothness]
           + w_end·‖q_N - z‖²                     [endpoint tracks z]
           + w_alpha·α²                           [anchor: z stays near g]
           + w_zobs·hinge(d_pref - D(z))²         [z avoids obstacles]

        where z = g_astar + α·n,  n is the local normal at the A* target.
        """
        N = len(q)
        if N <= 2:
            return q

        d_pref = self.config.curve_pref_clearance_m
        w_obs = 5.0
        w_smooth = 1.2
        w_end = 1.0
        w_alpha = 0.3
        w_zobs = 2.0
        step_q = 0.025
        step_alpha = 0.04
        max_iters = 200
        tol = 0.002
        alpha_max = 1.0
        eps = map_data.resolution * 0.5

        g_astar = np.array(local_subgoal, dtype=float)  # A* anchor
        # Normal at target: use the last two A* waypoints or tangent
        q_end = q[-1]
        to_target = g_astar - q_end
        dist_to_target = float(np.linalg.norm(to_target))
        if dist_to_target > 1e-9:
            tangent = to_target / dist_to_target
        else:
            tangent = np.array([1.0, 0.0])
        normal = np.array([-tangent[1], tangent[0]])  # lateral direction

        q_opt = q.copy()
        alpha = 0.0  # optimizable lateral offset
        last_max_shift = float("inf")

        for _ in range(max_iters):
            z = g_astar + alpha * normal  # current optimized target
            grad_q = np.zeros_like(q_opt)
            grad_alpha = 0.0

            for i in range(1, N):
                g = np.zeros(2, dtype=float)
                px, py = float(q_opt[i, 0]), float(q_opt[i, 1])

                # Obstacle on q_i
                D_i = float(query_distance_field(map_data, (px, py)))
                deficit = d_pref - D_i
                if deficit > 0:
                    dx = (float(query_distance_field(map_data, (px + eps, py)))
                          - float(query_distance_field(map_data, (px - eps, py)))) / (2.0 * eps)
                    dy = (float(query_distance_field(map_data, (px, py + eps)))
                          - float(query_distance_field(map_data, (px, py - eps)))) / (2.0 * eps)
                    gn = math.hypot(dx, dy)
                    if gn > 1e-9:
                        g -= (2.0 * w_obs * deficit) * np.array([dx, dy]) / gn

                # Smoothness
                sg = np.zeros(2, dtype=float)
                _c = lambda j: q_opt[max(0, min(j, N - 1))]
                sg += _c(i - 2) + (-4.0) * _c(i - 1) + 6.0 * q_opt[i] + (-4.0) * _c(i + 1) + _c(i + 2)
                g += (2.0 * w_smooth) * sg

                # Endpoint tracks z (q_N only)
                if i == N - 1:
                    g_qN_to_z = 2.0 * w_end * (q_opt[i] - z)
                    g += g_qN_to_z
                    # alpha gradient from ‖q_N - z‖²:
                    #   ∂/∂α ‖q_N - (g + αn)‖² = -2·(q_N - z)·n
                    grad_alpha += (-2.0 * w_end) * float(np.dot(q_opt[i] - z, normal))

                grad_q[i] = g

            # Anchor cost: α², ∂/∂α = 2·α
            grad_alpha += 2.0 * w_alpha * alpha

            # Obstacle on z
            zx, zy = float(z[0]), float(z[1])
            D_z = float(query_distance_field(map_data, (zx, zy)))
            deficit_z = d_pref - D_z
            if deficit_z > 0:
                dx_z = (float(query_distance_field(map_data, (zx + eps, zy)))
                        - float(query_distance_field(map_data, (zx - eps, zy)))) / (2.0 * eps)
                dy_z = (float(query_distance_field(map_data, (zx, zy + eps)))
                        - float(query_distance_field(map_data, (zx, zy - eps)))) / (2.0 * eps)
                gn_z = math.hypot(dx_z, dy_z)
                if gn_z > 1e-9:
                    # ∂/∂α hinge(d_pref-D(z))² = -2·deficit·∇D·n
                    grad_alpha += (-2.0 * w_zobs * deficit_z) * (dx_z * normal[0] + dy_z * normal[1]) / gn_z

            # Update q
            q_opt[1:] -= step_q * grad_q[1:]
            min_x = map_data.origin_xy[0] - 0.5
            max_x = min_x + map_data.width_m + 1.0
            min_y = map_data.origin_xy[1] - 0.5
            max_y = min_y + map_data.height_m + 1.0
            q_opt[:, 0] = np.clip(q_opt[:, 0], min_x, max_x)
            q_opt[:, 1] = np.clip(q_opt[:, 1], min_y, max_y)

            # Update α
            alpha -= step_alpha * grad_alpha
            alpha = max(-alpha_max, min(alpha_max, alpha))

            max_shift_q = float(np.max(np.linalg.norm(step_q * grad_q[1:], axis=1))) if N > 1 else 0.0
            max_shift = max(max_shift_q, abs(step_alpha * grad_alpha))
            if max_shift < tol and last_max_shift < tol:
                break
            last_max_shift = max_shift

        return q_opt

    def _smoothness_only_optimize(self, q: "np.ndarray") -> "np.ndarray":
        """Light smoothness optimization — no A* tube, no obstacle cost.
        Only preserves shape while removing noise from the local fix."""
        N = len(q)
        if N <= 2:
            return q
        w_smooth = 1.0
        step = 0.015
        q_opt = q.copy()
        for _ in range(30):
            grad = np.zeros_like(q_opt)
            for i in range(1, N - 1):
                sg = np.zeros(2, dtype=float)
                _c = lambda j: q_opt[max(0, min(j, N - 1))]
                sg += _c(i - 2) + (-4.0) * _c(i - 1) + 6.0 * q_opt[i] + (-4.0) * _c(i + 1) + _c(i + 2)
                grad[i] = (2.0 * w_smooth) * sg
            q_opt[1:-1] -= step * grad[1:-1]
            max_shift = float(np.max(np.linalg.norm(step * grad[1:-1], axis=1))) if N > 2 else 0.0
            if max_shift < 0.002:
                break
        return q_opt

    def _optimize_keypoints(
        self, q: "np.ndarray", local_subgoal: Point2D,
        map_data: MapData, clearance_threshold: float,
    ) -> "np.ndarray":
        """Light obstacle + smoothness optimization for denser control points."""
        N = len(q)
        if N <= 2:
            return q
        w_obs = 10.0
        w_smooth = 0.8
        w_goal = 0.3
        step = 0.02
        eps = map_data.resolution * 0.5
        goal_arr = np.array(local_subgoal, dtype=float)
        q_opt = q.copy()
        for _ in range(60):
            grad = np.zeros_like(q_opt)
            for i in range(1, N):
                g = np.zeros(2, dtype=float)
                px, py = float(q_opt[i, 0]), float(q_opt[i, 1])
                D_i = float(query_distance_field(map_data, (px, py)))
                deficit = clearance_threshold * 3.0 - D_i
                if deficit > 0:
                    dx = (float(query_distance_field(map_data, (px+eps, py))) - float(query_distance_field(map_data, (px-eps, py)))) / (2*eps)
                    dy = (float(query_distance_field(map_data, (px, py+eps))) - float(query_distance_field(map_data, (px, py-eps)))) / (2*eps)
                    gn = math.hypot(dx, dy)
                    if gn > 1e-9:
                        g -= (2.0 * w_obs * deficit) * np.array([dx, dy]) / gn
                sg = np.zeros(2, dtype=float)
                _c = lambda j: q_opt[max(0, min(j, N-1))]
                sg += _c(i-2) + (-4.0)*_c(i-1) + 6.0*q_opt[i] + (-4.0)*_c(i+1) + _c(i+2)
                g += (2.0 * w_smooth) * sg
                if i == N - 1:
                    g += (2.0 * w_goal) * (q_opt[i] - goal_arr)
                grad[i] = g
            q_opt[1:] -= step * grad[1:]
        return q_opt

    def _closest_point_on_polyline(
        self, point_xy: Point2D, polyline_xy: list[Point2D],
    ) -> Point2D:
        best = polyline_xy[0]
        best_dist = math.inf
        for a, b in zip(polyline_xy[:-1], polyline_xy[1:]):
            proj = self._project_point_to_segment(point_xy, a, b)
            d = self._distance(point_xy, proj)
            if d < best_dist:
                best_dist = d
                best = proj
        return best

    def _plan_fallback(
        self,
        map_data: MapData,
        ref_xy: Point2D,
        window: LocalPathWindow,
        clearance_threshold: float,
        initial_heading: float | None,
    ) -> LocalPreviewPath:
        """A*-waypoint fallback for cases where the straight-line seed is
        too far from free space for gradient-based recentering to recover."""
        truncated_points = self._prepare_window(ref_xy, window)
        if len(truncated_points) < 2:
            return self._empty_result(
                source_mode="polyline_fallback",
                waypoint_window=window.points_xy,
                truncated_window=truncated_points,
                local_subgoal=window.local_subgoal_xy,
                observation_distance=window.accumulated_distance_m,
                failure_reason="insufficient_points",
            )

        fallback_window = self._truncate_points_by_distance(
            truncated_points,
            window.accumulated_distance_m,
            min_points=self.config.min_window_points,
        )
        local_subgoal = fallback_window[-1]
        dense_polyline = self._densify_polyline(fallback_window, self.config.densify_spacing_m)
        safe_polyline = self._build_safe_corridor(dense_polyline, map_data, clearance_threshold)
        curve_points = self._build_piecewise_bezier(
            safe_polyline, self.config.bezier_tension,
            steps=self.config.bezier_segment_steps,
            initial_heading=initial_heading,
        )
        sampled_points = self._resample_polyline(curve_points, self.config.sample_spacing_m)
        recentered_points = self._recenter_curve_samples(
            map_data, sampled_points, local_subgoal, clearance_threshold,
        )
        curve_length = self._polyline_length(recentered_points)
        return self._build_result(
            map_data,
            recentered_points,
            window.points_xy,
            fallback_window,
            local_subgoal,
            "polyline_fallback",
            0.0,
            window.accumulated_distance_m,
            curve_length,
            clearance_threshold,
        )

    def _normalize_window(self, waypoint_window: list[Point2D] | LocalPathWindow) -> LocalPathWindow:
        if isinstance(waypoint_window, LocalPathWindow):
            return waypoint_window
        points = list(waypoint_window)
        local_subgoal = points[-1] if points else (0.0, 0.0)
        return LocalPathWindow(
            points_xy=points,
            local_subgoal_xy=local_subgoal,
            accumulated_distance_m=self._polyline_length(points),
            source_start_index=0,
            source_end_index=max(len(points) - 1, 0),
        )

    def _prepare_window(self, ref_xy: Point2D, window: LocalPathWindow) -> list[Point2D]:
        points: list[Point2D] = [ref_xy] if self.config.include_ref_point else []
        for point in window.points_xy:
            if not points or self._distance(points[-1], point) >= self.config.min_point_spacing_m:
                points.append(point)
        if len(points) >= 2 and self._distance(points[0], points[1]) < self.config.min_point_spacing_m:
            points = points[1:]
            if self.config.include_ref_point:
                points.insert(0, ref_xy)
        # When ref_xy is prepended, the first window point is the projection of
        # ref_xy onto the global path. The segment ref_xy -> projection is nearly
        # perpendicular to the path direction, which creates a sharp ~90° turn
        # in the Bezier curve. Skip the projection so the curve goes from ref_xy
        # directly toward points further along the path.
        if self.config.include_ref_point and len(points) >= 3:
            points.pop(1)
        if self.config.use_subgoal_truncation and window.local_subgoal_xy is not None:
            if not points or self._distance(points[-1], window.local_subgoal_xy) >= self.config.min_point_spacing_m:
                points.append(window.local_subgoal_xy)
            else:
                points[-1] = window.local_subgoal_xy
        return self._deduplicate_points(points)

    def _truncate_points_by_distance(
        self,
        points: list[Point2D],
        max_distance_m: float,
        min_points: int,
    ) -> list[Point2D]:
        if len(points) <= 2:
            return points
        truncated = [points[0]]
        accumulated = 0.0
        for idx in range(1, len(points)):
            segment_start = points[idx - 1]
            segment_end = points[idx]
            segment_length = self._distance(segment_start, segment_end)
            if segment_length < 1e-12:
                continue
            remaining = max_distance_m - accumulated
            if remaining > 1e-12 and accumulated + segment_length > max_distance_m:
                ratio = remaining / segment_length
                partial = (
                    segment_start[0] + ratio * (segment_end[0] - segment_start[0]),
                    segment_start[1] + ratio * (segment_end[1] - segment_start[1]),
                )
                truncated.append(partial)
                break
            accumulated += segment_length
            truncated.append(segment_end)
            if accumulated >= max_distance_m and len(truncated) >= max(min_points, 2):
                break
        if len(truncated) < max(min_points, 2):
            return points[:max(min_points, 2)]
        return self._deduplicate_points(truncated)

    def _deduplicate_points(self, points: list[Point2D]) -> list[Point2D]:
        deduplicated: list[Point2D] = []
        for point in points:
            if not deduplicated or self._distance(deduplicated[-1], point) >= self.config.min_point_spacing_m:
                deduplicated.append(point)
        return deduplicated

    def _densify_polyline(self, points: list[Point2D], spacing: float) -> list[Point2D]:
        if len(points) <= 2:
            return points
        return self._resample_polyline(points, max(spacing, self.config.min_point_spacing_m))

    def _build_piecewise_bezier(
        self,
        points: list[Point2D],
        tension: float,
        steps: int,
        initial_heading: float | None = None,
    ) -> list[Point2D]:
        if len(points) <= 2:
            return points

        control_points = np.asarray(points, dtype=float)
        tangents = self._compute_tangent_vectors(control_points, initial_heading=initial_heading)
        sampled_segments: list[Point2D] = []
        for idx in range(len(control_points) - 1):
            p0 = control_points[idx]
            p3 = control_points[idx + 1]
            segment_length = float(np.linalg.norm(p3 - p0))
            if segment_length < 1e-9:
                continue
            control_scale = tension * segment_length
            p1 = p0 + tangents[idx] * control_scale
            p2 = p3 - tangents[idx + 1] * control_scale
            segment_points = self._sample_cubic_bezier(p0, p1, p2, p3, steps=steps)
            if sampled_segments:
                segment_points = segment_points[1:]
            sampled_segments.extend((float(x), float(y)) for x, y in segment_points)
        return sampled_segments if sampled_segments else points

    def _compute_tangent_vectors(self, points: np.ndarray, initial_heading: float | None = None) -> np.ndarray:
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
            if norm < 1e-9:
                tangents[idx] = np.array([1.0, 0.0])
            else:
                tangents[idx] = direction / norm
        return tangents

    def _sample_cubic_bezier(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        steps: int,
    ) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, steps + 1)
        omt = 1.0 - ts
        curve = (
            (omt**3)[:, None] * p0
            + (3.0 * omt**2 * ts)[:, None] * p1
            + (3.0 * omt * ts**2)[:, None] * p2
            + (ts**3)[:, None] * p3
        )
        return curve

    def _resample_polyline(self, points: list[Point2D], sample_spacing_m: float) -> list[Point2D]:
        if len(points) <= 1:
            return points
        arrays = np.asarray(points, dtype=float)
        segment_vectors = arrays[1:] - arrays[:-1]
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        if total_length < 1e-9:
            return [points[0], points[-1]] if len(points) > 1 else points

        spacing = max(sample_spacing_m, 1e-3)
        sample_positions = np.arange(0.0, total_length, spacing)
        if sample_positions.size == 0 or sample_positions[-1] < total_length:
            sample_positions = np.append(sample_positions, total_length)

        resampled: list[Point2D] = []
        segment_index = 0
        for arc_s in sample_positions:
            while segment_index < len(segment_lengths) - 1 and cumulative[segment_index + 1] < arc_s:
                segment_index += 1
            local_length = segment_lengths[segment_index]
            if local_length < 1e-9:
                point = arrays[segment_index]
            else:
                ratio = (arc_s - cumulative[segment_index]) / local_length
                point = arrays[segment_index] + ratio * segment_vectors[segment_index]
            resampled.append((float(point[0]), float(point[1])))
        return self._deduplicate_points(resampled)

    def _recenter_curve_samples(
        self,
        map_data: MapData,
        points_xy: list[Point2D],
        local_subgoal_xy: Point2D,
        clearance_threshold: float,
    ) -> list[Point2D]:
        if len(points_xy) < 3 or self.config.curve_refine_iterations <= 0:
            self._last_refine_stats = {
                "iterations": 0,
                "max_iteration_shift_m": 0.0,
                "max_net_shift_m": 0.0,
                "fixed_prefix_points": min(self.config.curve_refine_fixed_prefix_points, len(points_xy)),
                "max_abs_raw_offset_m": 0.0,
                "max_abs_smoothed_offset_m": 0.0,
                "mean_abs_smoothed_offset_m": 0.0,
            }
            return points_xy
        current_points = [tuple(point) for point in points_xy]
        initial_points = np.asarray(points_xy, dtype=float)
        fixed_prefix = min(self.config.curve_refine_fixed_prefix_points, len(current_points))
        performed_iterations = 0
        last_shift_norm = 0.0
        max_abs_raw_offset = 0.0
        max_abs_smoothed_offset = 0.0
        mean_abs_smoothed_offset = 0.0

        for _ in range(self.config.curve_refine_iterations):
            tangents = self._compute_tangents(current_points)
            normals = np.asarray([(-tangent[1], tangent[0]) for tangent in tangents], dtype=float)
            raw_offsets = np.zeros(len(current_points), dtype=float)
            for idx, (point_xy, normal_xy) in enumerate(zip(current_points, normals)):
                if idx < fixed_prefix:
                    continue  # q_0 hard frozen
                point_arr = np.array(point_xy)
                if idx == len(current_points) - 1:
                    # q_N: soft goal attraction, projected onto normal
                    to_goal = np.array([local_subgoal_xy[0] - point_arr[0],
                                       local_subgoal_xy[1] - point_arr[1]])
                    raw_offsets[idx] = float(np.dot(to_goal, normal_xy)) * self.config.curve_refine_goal_weight
                    continue
                # q_1..q_{N-1}: obstacle center push + weak guide attraction
                left_clearance = self._scan_normal_clearance(
                    map_data, point_xy,
                    (float(normal_xy[0]), float(normal_xy[1])),
                    direction=1.0, required_clearance=clearance_threshold,
                )
                right_clearance = self._scan_normal_clearance(
                    map_data, point_xy,
                    (float(normal_xy[0]), float(normal_xy[1])),
                    direction=-1.0, required_clearance=clearance_threshold,
                )
                obstacle_offset = 0.5 * (left_clearance - right_clearance)
                guide_vector = initial_points[idx] - point_arr
                guide_offset = float(np.dot(guide_vector, normal_xy)) * self.config.curve_refine_guide_weight
                raw_offsets[idx] = obstacle_offset + guide_offset

            smoothed_offsets = self._solve_recenter_offset_field(raw_offsets, fixed_prefix)
            shifted_points = initial_points + smoothed_offsets[:, None] * normals
            shifted_points[:fixed_prefix] = initial_points[:fixed_prefix]
            current_points = [(float(point[0]), float(point[1])) for point in shifted_points]
            performed_iterations += 1
            if smoothed_offsets.size > 0:
                last_shift_norm = float(np.max(np.abs(smoothed_offsets)))
                max_abs_raw_offset = float(np.max(np.abs(raw_offsets)))
                max_abs_smoothed_offset = float(np.max(np.abs(smoothed_offsets)))
                mean_abs_smoothed_offset = float(np.mean(np.abs(smoothed_offsets)))
            if last_shift_norm < self.config.curve_refine_convergence_tol_m:
                break

        current_array = np.asarray(current_points, dtype=float)
        net_shift = np.linalg.norm(current_array - initial_points, axis=1)
        self._last_refine_stats = {
            "iterations": performed_iterations,
            "max_iteration_shift_m": last_shift_norm,
            "max_net_shift_m": float(np.max(net_shift)) if net_shift.size else 0.0,
            "fixed_prefix_points": fixed_prefix,
            "max_abs_raw_offset_m": max_abs_raw_offset,
            "max_abs_smoothed_offset_m": max_abs_smoothed_offset,
            "mean_abs_smoothed_offset_m": mean_abs_smoothed_offset,
        }
        return current_points

    def _solve_recenter_offset_field(self, raw_offsets: np.ndarray, fixed_prefix: int) -> np.ndarray:
        sample_count = int(raw_offsets.size)
        if sample_count == 0:
            return raw_offsets
        lambda_a = self.config.curve_refine_lambda_a
        lambda_s = self.config.curve_refine_lambda_s
        lambda_d = self.config.curve_refine_lambda_d
        target = lambda_a * raw_offsets
        system = np.zeros((sample_count, sample_count), dtype=float)
        rhs = target.copy()

        for idx in range(sample_count):
            if idx < fixed_prefix:
                # q_0: hard frozen
                system[idx, idx] = 1.0
                rhs[idx] = 0.0
            elif idx == sample_count - 1:
                # q_N: damped identity — responds to goal attraction, no smoothing
                system[idx, idx] = 1.0 + lambda_d
            else:
                # q_1..q_{N-1}: smooth + damp
                system[idx, idx] += 1.0 + lambda_d
                if idx > fixed_prefix:
                    system[idx, idx] += lambda_s
                    system[idx, idx - 1] -= lambda_s
                if idx < sample_count - 2:
                    system[idx, idx] += lambda_s
                    system[idx, idx + 1] -= lambda_s

        return np.linalg.solve(system, rhs)

    def _scan_normal_clearance(
        self,
        map_data: MapData,
        center_xy: Point2D,
        normal_xy: Point2D,
        *,
        direction: float,
        required_clearance: float,
    ) -> float:
        step_m = max(map_data.resolution * 0.5, 0.02)
        max_search_m = math.hypot(map_data.width_m, map_data.height_m)
        best_distance = 0.0
        distance = step_m
        while distance <= max_search_m + 1e-12:
            point_xy = (
                center_xy[0] + direction * distance * normal_xy[0],
                center_xy[1] + direction * distance * normal_xy[1],
            )
            if self._query_clearance(map_data, point_xy) + 1e-12 < required_clearance:
                break
            best_distance = distance
            distance += step_m
        return best_distance

    def _build_result(
        self,
        map_data: MapData,
        points_xy: list[Point2D],
        waypoint_window: list[Point2D],
        truncated_window: list[Point2D],
        local_subgoal_xy: Point2D,
        source_mode: str,
        used_tension: float,
        observation_distance_m: float,
        curve_end_distance_m: float,
        clearance_threshold: float,
    ) -> LocalPreviewPath:
        arc_lengths = self._compute_arc_lengths(points_xy)
        tangents = self._compute_tangents(points_xy)
        normals = [(-tangent[1], tangent[0]) for tangent in tangents]
        curvatures = self._compute_curvatures(points_xy)
        clearance_samples = [self._query_clearance(map_data, point) for point in points_xy]
        min_clearance = min(clearance_samples) if clearance_samples else 0.0
        mean_clearance = sum(clearance_samples) / len(clearance_samples) if clearance_samples else 0.0
        polyline_length = self._polyline_length(truncated_window)
        curve_length = self._polyline_length(points_xy)
        length_gap = abs(curve_length - polyline_length)
        alignment_error = self._mean_polyline_distance(points_xy, truncated_window)
        is_safe = bool(clearance_samples) and min_clearance + 1e-3 >= clearance_threshold
        failure_reason = "" if is_safe else "clearance_below_threshold"
        return LocalPreviewPath(
            points_xy=points_xy,
            arc_lengths=arc_lengths,
            tangents_xy=tangents,
            normals_xy=normals,
            curvatures=curvatures,
            source_mode=source_mode,
            is_safe=is_safe,
            min_clearance=min_clearance,
            local_subgoal_xy=local_subgoal_xy,
            truncated_window_xy=truncated_window,
            observation_distance_m=observation_distance_m,
            curve_end_distance_m=curve_end_distance_m,
            waypoint_window_xy=waypoint_window,
            clearance_samples=clearance_samples,
            failure_reason=failure_reason,
            used_tension=used_tension,
            metadata={
                "clearance_threshold_m": clearance_threshold,
                "configured_preview_distance_m": self.config.preview_distance_m,
                "mean_clearance_m": mean_clearance,
                "alignment_error_m": alignment_error,
                "length_gap_m": length_gap,
                "curve_length_m": curve_length,
                "polyline_length_m": polyline_length,
                "curve_refine_iterations": int(self._last_refine_stats.get("iterations", 0)),
                "curve_refine_max_iteration_shift_m": float(
                    self._last_refine_stats.get("max_iteration_shift_m", 0.0)
                ),
                "curve_refine_max_net_shift_m": float(self._last_refine_stats.get("max_net_shift_m", 0.0)),
                "curve_refine_fixed_prefix_points": int(
                    self._last_refine_stats.get("fixed_prefix_points", 0)
                ),
                "curve_refine_max_abs_raw_offset_m": float(
                    self._last_refine_stats.get("max_abs_raw_offset_m", 0.0)
                ),
                "curve_refine_max_abs_smoothed_offset_m": float(
                    self._last_refine_stats.get("max_abs_smoothed_offset_m", 0.0)
                ),
                "curve_refine_mean_abs_smoothed_offset_m": float(
                    self._last_refine_stats.get("mean_abs_smoothed_offset_m", 0.0)
                ),
            },
        )

    def _safe_candidate_score(self, result: LocalPreviewPath) -> tuple[float, float, float, float, float]:
        return (
            result.curve_end_distance_m,
            result.min_clearance,
            float(result.metadata.get("mean_clearance_m", 0.0)),
            -float(result.metadata.get("alignment_error_m", 0.0)),
            -float(result.metadata.get("length_gap_m", 0.0)),
        )

    def _unsafe_candidate_score(self, result: LocalPreviewPath) -> tuple[float, float, float, float]:
        return (
            result.min_clearance,
            float(result.metadata.get("mean_clearance_m", 0.0)),
            -float(result.metadata.get("alignment_error_m", 0.0)),
            -float(result.metadata.get("length_gap_m", 0.0)),
        )

    def _compute_arc_lengths(self, points_xy: list[Point2D]) -> list[float]:
        if not points_xy:
            return []
        arc_lengths = [0.0]
        for idx in range(1, len(points_xy)):
            arc_lengths.append(arc_lengths[-1] + self._distance(points_xy[idx - 1], points_xy[idx]))
        return arc_lengths

    def _compute_tangents(self, points_xy: list[Point2D]) -> list[Point2D]:
        if len(points_xy) == 1:
            return [(1.0, 0.0)]
        points = np.asarray(points_xy, dtype=float)
        tangents: list[Point2D] = []
        for idx in range(len(points)):
            if idx == 0:
                direction = points[1] - points[0]
            elif idx == len(points) - 1:
                direction = points[-1] - points[-2]
            else:
                direction = points[idx + 1] - points[idx - 1]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                tangents.append((1.0, 0.0))
            else:
                direction /= norm
                tangents.append((float(direction[0]), float(direction[1])))
        return tangents

    def _compute_curvatures(self, points_xy: list[Point2D]) -> list[float]:
        if len(points_xy) < 3:
            return [0.0 for _ in points_xy]
        points = np.asarray(points_xy, dtype=float)
        curvatures = [0.0]
        for idx in range(1, len(points) - 1):
            p_prev = points[idx - 1]
            p_curr = points[idx]
            p_next = points[idx + 1]
            a = np.linalg.norm(p_curr - p_prev)
            b = np.linalg.norm(p_next - p_curr)
            c = np.linalg.norm(p_next - p_prev)
            area2 = abs(np.cross(p_curr - p_prev, p_next - p_prev))
            denominator = max(a * b * c, 1e-9)
            curvatures.append(float(2.0 * area2 / denominator))
        curvatures.append(0.0)
        return curvatures

    def _build_safe_corridor(
        self,
        points: list[Point2D],
        map_data: MapData,
        clearance_threshold: float,
    ) -> list[Point2D]:
        if not points:
            return points
        eps = map_data.resolution * 0.5
        adjusted: list[Point2D] = []
        for point in points:
            px, py = point
            for _ in range(20):
                clearance = self._query_clearance(map_data, (px, py))
                if clearance >= clearance_threshold:
                    break
                gx = (self._query_clearance(map_data, (px + eps, py)) - self._query_clearance(map_data, (px - eps, py))) / (2.0 * eps)
                gy = (self._query_clearance(map_data, (px, py + eps)) - self._query_clearance(map_data, (px, py - eps))) / (2.0 * eps)
                grad_norm = math.hypot(gx, gy)
                if grad_norm < 1e-9:
                    break
                deficit = clearance_threshold - clearance
                step = deficit * 0.6
                px += (gx / grad_norm) * step
                py += (gy / grad_norm) * step
            adjusted.append((px, py))
        return adjusted

    def _query_clearance(self, map_data: MapData, point_xy: Point2D) -> float:
        return float(query_distance_field(map_data, point_xy))

    def _mean_polyline_distance(self, points_xy: list[Point2D], polyline_xy: list[Point2D]) -> float:
        if not points_xy or len(polyline_xy) < 2:
            return 0.0
        distances = [self._distance_to_polyline(point, polyline_xy) for point in points_xy]
        return sum(distances) / len(distances)

    def _distance_to_polyline(self, point_xy: Point2D, polyline_xy: list[Point2D]) -> float:
        best_distance = math.inf
        for start_xy, end_xy in zip(polyline_xy[:-1], polyline_xy[1:]):
            projected = self._project_point_to_segment(point_xy, start_xy, end_xy)
            best_distance = min(best_distance, self._distance(point_xy, projected))
        return 0.0 if best_distance is math.inf else best_distance

    def _project_point_to_segment(self, point_xy: Point2D, start_xy: Point2D, end_xy: Point2D) -> Point2D:
        segment_x = end_xy[0] - start_xy[0]
        segment_y = end_xy[1] - start_xy[1]
        segment_length_sq = segment_x * segment_x + segment_y * segment_y
        if segment_length_sq <= 1e-12:
            return start_xy
        ratio = (
            (point_xy[0] - start_xy[0]) * segment_x + (point_xy[1] - start_xy[1]) * segment_y
        ) / segment_length_sq
        clamped_ratio = min(max(ratio, 0.0), 1.0)
        return (
            start_xy[0] + clamped_ratio * segment_x,
            start_xy[1] + clamped_ratio * segment_y,
        )

    def _empty_result(
        self,
        source_mode: str,
        waypoint_window: list[Point2D],
        truncated_window: list[Point2D],
        local_subgoal: Point2D | None,
        observation_distance: float,
        failure_reason: str,
    ) -> LocalPreviewPath:
        return LocalPreviewPath(
            points_xy=[],
            arc_lengths=[],
            tangents_xy=[],
            normals_xy=[],
            curvatures=[],
            source_mode=source_mode,
            is_safe=False,
            min_clearance=0.0,
            local_subgoal_xy=local_subgoal,
            truncated_window_xy=truncated_window,
            observation_distance_m=observation_distance,
            curve_end_distance_m=0.0,
            waypoint_window_xy=waypoint_window,
            clearance_samples=[],
            failure_reason=failure_reason,
            used_tension=0.0,
            metadata={},
        )

    def _polyline_length(self, points: list[Point2D]) -> float:
        total = 0.0
        for idx in range(1, len(points)):
            total += self._distance(points[idx - 1], points[idx])
        return total

    def _distance(self, a: Point2D, b: Point2D) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
