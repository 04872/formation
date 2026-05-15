from __future__ import annotations

import math

import numpy as np

from formation.types import LocalPathWindow, LocalPreviewPath, MapData, Point2D, PreviewCurveConfig


class PreviewCurvePlanner:
    def __init__(self, config: PreviewCurveConfig | None = None) -> None:
        self.config = config or PreviewCurveConfig()

    def plan(
        self,
        map_data: MapData,
        ref_xy: Point2D,
        waypoint_window: list[Point2D] | LocalPathWindow,
    ) -> LocalPreviewPath:
        window = self._normalize_window(waypoint_window)
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

        clearance_threshold = (
            self.config.clearance_threshold_m
            if self.config.clearance_threshold_m is not None
            else map_data.inflation_radius
        )

        candidate_distances = [window.accumulated_distance_m]
        shortened_distance = max(
            self.config.fallback_min_distance_m,
            window.accumulated_distance_m * self.config.short_preview_scale,
        )
        if shortened_distance + 1e-6 < window.accumulated_distance_m:
            candidate_distances.append(shortened_distance)

        bezier_candidates = [
            (self.config.bezier_tension, "bezier"),
            (self.config.reduced_tension, "bezier"),
        ]
        best_result: LocalPreviewPath | None = None

        for candidate_distance in candidate_distances:
            candidate_window = self._truncate_points_by_distance(
                truncated_points,
                candidate_distance,
                min_points=self.config.min_window_points,
            )
            local_subgoal = candidate_window[-1]
            for tension, source_mode in bezier_candidates:
                dense_polyline = self._densify_polyline(candidate_window, self.config.densify_spacing_m)
                curve_points = self._build_piecewise_bezier(
                    dense_polyline,
                    tension,
                    steps=self.config.bezier_segment_steps,
                )
                sampled_points = self._resample_polyline(curve_points, self.config.sample_spacing_m)
                result = self._build_result(
                    map_data,
                    sampled_points,
                    window.points_xy,
                    candidate_window,
                    local_subgoal,
                    source_mode,
                    tension,
                    window.accumulated_distance_m,
                    candidate_distance,
                    clearance_threshold,
                )
                if result.is_safe:
                    return result
                if best_result is None or result.min_clearance > best_result.min_clearance:
                    best_result = result

        if self.config.fallback_to_polyline:
            fallback_window = self._truncate_points_by_distance(
                truncated_points,
                candidate_distances[-1],
                min_points=self.config.min_window_points,
            )
            fallback_points = self._resample_polyline(fallback_window, self.config.sample_spacing_m)
            fallback_result = self._build_result(
                map_data,
                fallback_points,
                window.points_xy,
                fallback_window,
                fallback_window[-1],
                "polyline_fallback",
                0.0,
                window.accumulated_distance_m,
                self._polyline_length(fallback_window),
                clearance_threshold,
            )
            if fallback_result.is_safe or best_result is None:
                return fallback_result
            if fallback_result.min_clearance >= best_result.min_clearance:
                return fallback_result

        assert best_result is not None
        return best_result

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
    ) -> list[Point2D]:
        if len(points) <= 2:
            return points

        control_points = np.asarray(points, dtype=float)
        tangents = self._compute_tangent_vectors(control_points)
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

    def _compute_tangent_vectors(self, points: np.ndarray) -> np.ndarray:
        tangents = np.zeros_like(points)
        for idx in range(len(points)):
            if idx == 0:
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
        is_safe = bool(clearance_samples) and min_clearance + 1e-12 >= clearance_threshold
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
            metadata={"clearance_threshold_m": clearance_threshold},
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

    def _query_clearance(self, map_data: MapData, point_xy: Point2D) -> float:
        try:
            rc = map_data.world_to_grid(point_xy)
        except ValueError:
            return 0.0
        return float(map_data.distance_field[rc])

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
