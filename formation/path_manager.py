from __future__ import annotations

import math

from formation.types import GlobalPath, LocalPathWindow, Point2D


class PathManager:
    def __init__(self, global_path: GlobalPath) -> None:
        if not global_path.waypoints_xy:
            raise ValueError("GlobalPath must contain at least one waypoint.")
        self.global_path = global_path
        self.current_waypoint_index = 0

    def find_closest_waypoint_index(self, ref_xy: Point2D) -> int:
        best_index = min(
            range(len(self.global_path.waypoints_xy)),
            key=lambda idx: self._distance(ref_xy, self.global_path.waypoints_xy[idx]),
        )
        self.current_waypoint_index = best_index
        return best_index

    def get_local_window(self, ref_xy: Point2D, window_size: int = 8) -> list[Point2D]:
        start_index = self.find_closest_waypoint_index(ref_xy)
        end_index = min(len(self.global_path.waypoints_xy), start_index + max(window_size, 1))
        return self.global_path.waypoints_xy[start_index:end_index]

    def get_local_window_by_distance(
        self,
        ref_xy: Point2D,
        preview_distance_m: float,
        min_points: int = 4,
    ) -> list[Point2D]:
        return self.get_local_path_window(ref_xy, preview_distance_m, min_points).points_xy

    def get_local_path_window(
        self,
        ref_xy: Point2D,
        preview_distance_m: float,
        min_points: int = 4,
    ) -> LocalPathWindow:
        if min_points < 1:
            raise ValueError("min_points must be at least 1.")

        start_index = self.find_closest_waypoint_index(ref_xy)
        return self._build_local_path_window(
            start_point=self.global_path.waypoints_xy[start_index],
            start_segment_index=start_index,
            preview_distance_m=preview_distance_m,
            min_points=min_points,
        )

    def get_local_path_window_from_projection(
        self,
        ref_xy: Point2D,
        preview_distance_m: float,
        min_points: int = 4,
    ) -> LocalPathWindow:
        if min_points < 1:
            raise ValueError("min_points must be at least 1.")

        start_segment_index, projected_point = self._project_point_to_polyline(ref_xy)
        self.current_waypoint_index = start_segment_index
        return self._build_local_path_window(
            start_point=projected_point,
            start_segment_index=start_segment_index,
            preview_distance_m=preview_distance_m,
            min_points=min_points,
        )

    def _build_local_path_window(
        self,
        start_point: Point2D,
        start_segment_index: int,
        preview_distance_m: float,
        min_points: int,
    ) -> LocalPathWindow:
        waypoints = self.global_path.waypoints_xy
        if len(waypoints) == 1:
            only_point = waypoints[0]
            return LocalPathWindow(
                points_xy=[only_point],
                local_subgoal_xy=only_point,
                accumulated_distance_m=0.0,
                source_start_index=0,
                source_end_index=0,
            )

        target_distance = max(preview_distance_m, 0.0)
        if target_distance <= 1e-12:
            return LocalPathWindow(
                points_xy=[start_point],
                local_subgoal_xy=start_point,
                accumulated_distance_m=0.0,
                source_start_index=start_segment_index,
                source_end_index=start_segment_index,
            )

        points = [start_point]
        accumulated = 0.0
        end_index = start_segment_index
        local_subgoal = start_point
        current_start = start_point

        for idx in range(start_segment_index + 1, len(waypoints)):
            segment_end = waypoints[idx]
            segment_length = self._distance(current_start, segment_end)
            if segment_length < 1e-12:
                current_start = segment_end
                continue

            remaining = target_distance - accumulated
            if remaining > 1e-12 and accumulated + segment_length > target_distance:
                ratio = remaining / segment_length
                local_subgoal = (
                    current_start[0] + ratio * (segment_end[0] - current_start[0]),
                    current_start[1] + ratio * (segment_end[1] - current_start[1]),
                )
                points.append(local_subgoal)
                accumulated = target_distance
                end_index = idx
                break

            accumulated += segment_length
            points.append(segment_end)
            local_subgoal = segment_end
            end_index = idx
            current_start = segment_end
            if accumulated >= target_distance - 1e-12:
                break

        if len(points) < min_points:
            points = self._densify_polyline(points, target_count=min_points)

        return LocalPathWindow(
            points_xy=points,
            local_subgoal_xy=local_subgoal,
            accumulated_distance_m=accumulated,
            source_start_index=start_segment_index,
            source_end_index=end_index,
        )

    def _project_point_to_polyline(self, ref_xy: Point2D) -> tuple[int, Point2D]:
        waypoints = self.global_path.waypoints_xy
        if len(waypoints) == 1:
            return 0, waypoints[0]

        best_segment_index = 0
        best_point = waypoints[0]
        best_distance = float("inf")
        for idx in range(len(waypoints) - 1):
            projected_point = self._project_point_to_segment(ref_xy, waypoints[idx], waypoints[idx + 1])
            distance = self._distance(ref_xy, projected_point)
            if distance < best_distance:
                best_distance = distance
                best_segment_index = idx
                best_point = projected_point
        return best_segment_index, best_point

    def _project_point_to_segment(self, point: Point2D, segment_start: Point2D, segment_end: Point2D) -> Point2D:
        dx = segment_end[0] - segment_start[0]
        dy = segment_end[1] - segment_start[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            return segment_start
        t = ((point[0] - segment_start[0]) * dx + (point[1] - segment_start[1]) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        return (
            segment_start[0] + t * dx,
            segment_start[1] + t * dy,
        )

    def get_local_subgoal(self, ref_xy: Point2D, lookahead_index: int = 5) -> Point2D:
        start_index = self.find_closest_waypoint_index(ref_xy)
        target_index = min(
            len(self.global_path.waypoints_xy) - 1,
            start_index + max(lookahead_index, 1),
        )
        return self.global_path.waypoints_xy[target_index]

    def _densify_polyline(self, points: list[Point2D], target_count: int) -> list[Point2D]:
        if len(points) >= target_count or len(points) <= 1:
            return points

        total_length = self._polyline_length(points)
        if total_length < 1e-12:
            return points

        cumulative = [0.0]
        for idx in range(1, len(points)):
            cumulative.append(cumulative[-1] + self._distance(points[idx - 1], points[idx]))

        densified: list[Point2D] = []
        sample_positions = [total_length * idx / (target_count - 1) for idx in range(target_count)]
        segment_index = 0
        for sample_s in sample_positions:
            while segment_index < len(cumulative) - 2 and cumulative[segment_index + 1] < sample_s:
                segment_index += 1
            start = points[segment_index]
            end = points[segment_index + 1]
            segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
            if segment_length < 1e-12:
                densified.append(start)
                continue
            ratio = (sample_s - cumulative[segment_index]) / segment_length
            densified.append(
                (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                )
            )
        return densified

    def _polyline_length(self, points: list[Point2D]) -> float:
        total = 0.0
        for idx in range(1, len(points)):
            total += self._distance(points[idx - 1], points[idx])
        return total

    def _distance(self, a: Point2D, b: Point2D) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
