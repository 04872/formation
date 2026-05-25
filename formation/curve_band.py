from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from formation.mpc_controller import query_distance_field
from formation.types import CurveBand, CurveBandSample, CurveBandStripCell, LocalPreviewPath, MapData, Point2D

_MAX_SCAN_DIST_M = 2.0


@dataclass(frozen=True)
class _ChordCandidate:
    sample_index: int
    candidate_index: int
    preview_xy: Point2D
    center_xy: Point2D
    direction_xy: Point2D
    left_xy: Point2D
    right_xy: Point2D
    half_width_m: float
    direction_offset_rad: float
    anchor_offset_m: float
    node_cost: float


class CurveBandBuilder:
    def __init__(
        self,
        lateral_step_m: float | None = None,
        chord_angle_offsets_deg: tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0),
        center_offset_count: int = 3,
        node_width_weight: float = 1.0,
        node_distance_weight: float = 6.0,
        node_angle_weight: float = 0.8,
        edge_center_smooth_weight: float = 4.0,
        edge_angle_smooth_weight: float = 0.8,
    ) -> None:
        self.lateral_step_m = lateral_step_m
        self.chord_angle_offsets_deg = chord_angle_offsets_deg
        self.center_offset_count = max(1, center_offset_count)
        self.node_width_weight = node_width_weight
        self.node_distance_weight = node_distance_weight
        self.node_angle_weight = node_angle_weight
        self.edge_center_smooth_weight = edge_center_smooth_weight
        self.edge_angle_smooth_weight = edge_angle_smooth_weight

    def build(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        robot_radius: float,
        safety_margin: float,
    ) -> CurveBand:
        required_clearance = robot_radius + safety_margin
        step_m = self.lateral_step_m or max(map_data.resolution * 0.5, 0.02)
        if not preview_path.points_xy:
            return CurveBand(
                samples=[],
                strip_cells=[],
                source_mode=preview_path.source_mode,
                metadata={
                    "required_clearance_m": required_clearance,
                    "lateral_step_m": step_m,
                    "candidate_counts": [],
                    "selected_candidate_indices": [],
                },
            )

        max_search_m = math.hypot(map_data.width_m, map_data.height_m)
        angle_offsets_rad = tuple(math.radians(value) for value in self.chord_angle_offsets_deg)
        center_offsets_m = self._center_offsets(map_data, required_clearance)

        candidates_by_sample: list[list[_ChordCandidate]] = []
        for sample_index, (arc_length_s, preview_xy, tangent_xy, normal_xy) in enumerate(
            zip(
                preview_path.arc_lengths,
                preview_path.points_xy,
                preview_path.tangents_xy,
                preview_path.normals_xy,
            )
        ):
            candidates = self._generate_candidates(
                map_data,
                sample_index=sample_index,
                arc_length_s=arc_length_s,
                preview_xy=preview_xy,
                tangent_xy=tangent_xy,
                normal_xy=normal_xy,
                angle_offsets_rad=angle_offsets_rad,
                center_offsets_m=center_offsets_m,
                required_clearance=required_clearance,
                max_search_m=max_search_m,
                step_m=step_m,
            )
            if not candidates and self._query_clearance(map_data, preview_xy) + 1e-12 >= required_clearance:
                candidates = [
                    _ChordCandidate(
                        sample_index=sample_index,
                        candidate_index=0,
                        preview_xy=preview_xy,
                        center_xy=preview_xy,
                        direction_xy=normal_xy,
                        left_xy=preview_xy,
                        right_xy=preview_xy,
                        half_width_m=0.0,
                        direction_offset_rad=0.0,
                        anchor_offset_m=0.0,
                        node_cost=0.0,
                    )
                ]
            candidates_by_sample.append(candidates)

        if any(not candidates for candidates in candidates_by_sample):
            return CurveBand(
                samples=[],
                strip_cells=[],
                source_mode=preview_path.source_mode,
                metadata={
                    "required_clearance_m": required_clearance,
                    "lateral_step_m": step_m,
                    "candidate_counts": [len(candidates) for candidates in candidates_by_sample],
                    "failure_reason": "missing_safe_chord_candidates",
                },
            )

        selected_candidates, strip_cells, total_cost, feasible_transition_count, total_transition_count = self._run_dp(
            map_data,
            candidates_by_sample,
            required_clearance=required_clearance,
            sample_step_m=step_m,
        )

        centerline_xy = [candidate.center_xy for candidate in selected_candidates]
        arc_lengths = self._compute_arc_lengths(centerline_xy)
        tangents_xy = self._compute_tangents(centerline_xy)
        normals_xy = [(-tangent[1], tangent[0]) for tangent in tangents_xy]
        samples = [
            CurveBandSample(
                arc_length_s=arc_length_s,
                center_xy=center_xy,
                tangent_xy=tangent_xy,
                normal_xy=normal_xy,
                left_xy=candidate.left_xy,
                right_xy=candidate.right_xy,
                half_width_m=candidate.half_width_m,
                direction_offset_rad=candidate.direction_offset_rad,
                metadata={
                    "candidate_index": candidate.candidate_index,
                    "anchor_offset_m": candidate.anchor_offset_m,
                    "preview_xy": candidate.preview_xy,
                    "chord_direction_xy": candidate.direction_xy,
                    "node_cost": candidate.node_cost,
                },
            )
            for arc_length_s, center_xy, tangent_xy, normal_xy, candidate in zip(
                arc_lengths,
                centerline_xy,
                tangents_xy,
                normals_xy,
                selected_candidates,
            )
        ]

        return CurveBand(
            samples=samples,
            strip_cells=strip_cells,
            source_mode=preview_path.source_mode,
            metadata={
                "required_clearance_m": required_clearance,
                "lateral_step_m": step_m,
                "max_search_m": max_search_m,
                "candidate_counts": [len(candidates) for candidates in candidates_by_sample],
                "candidate_angle_offsets_deg": list(self.chord_angle_offsets_deg),
                "candidate_center_offsets_m": list(center_offsets_m),
                "selected_candidate_indices": [candidate.candidate_index for candidate in selected_candidates],
                "selected_direction_offsets_rad": [candidate.direction_offset_rad for candidate in selected_candidates],
                "selected_half_widths_m": [candidate.half_width_m for candidate in selected_candidates],
                "selected_node_costs": [candidate.node_cost for candidate in selected_candidates],
                "dp_total_cost": total_cost,
                "total_candidate_count": sum(len(candidates) for candidates in candidates_by_sample),
                "total_transition_count": total_transition_count,
                "feasible_transition_count": feasible_transition_count,
                "arc_lengths": np.asarray(arc_lengths, dtype=float),
                "refined_band_centerline": centerline_xy,
                "safe_strip_cells": [cell.vertices_xy for cell in strip_cells],
            },
        )

    def _generate_candidates(
        self,
        map_data: MapData,
        *,
        sample_index: int,
        arc_length_s: float,
        preview_xy: Point2D,
        tangent_xy: Point2D,
        normal_xy: Point2D,
        angle_offsets_rad: tuple[float, ...],
        center_offsets_m: tuple[float, ...],
        required_clearance: float,
        max_search_m: float,
        step_m: float,
    ) -> list[_ChordCandidate]:
        del arc_length_s  # keep signature aligned with sampled preview semantics
        candidates: list[_ChordCandidate] = []
        candidate_index = 0
        for angle_offset_rad in angle_offsets_rad:
            cos_angle = math.cos(angle_offset_rad)
            sin_angle = math.sin(angle_offset_rad)
            direction_xy = (
                cos_angle * normal_xy[0] + sin_angle * tangent_xy[0],
                cos_angle * normal_xy[1] + sin_angle * tangent_xy[1],
            )
            direction_norm = math.hypot(direction_xy[0], direction_xy[1])
            if direction_norm <= 1e-9:
                continue
            direction_xy = (direction_xy[0] / direction_norm, direction_xy[1] / direction_norm)
            for center_offset_m in center_offsets_m:
                anchor_xy = (
                    preview_xy[0] + center_offset_m * normal_xy[0],
                    preview_xy[1] + center_offset_m * normal_xy[1],
                )
                if self._query_clearance(map_data, anchor_xy) + 1e-12 < required_clearance:
                    continue
                left_extent = self._scan_safe_extent(
                    map_data,
                    anchor_xy,
                    direction_xy,
                    direction=1.0,
                    required_clearance=required_clearance,
                    max_search_m=max_search_m,
                    step_m=step_m,
                )
                right_extent = self._scan_safe_extent(
                    map_data,
                    anchor_xy,
                    direction_xy,
                    direction=-1.0,
                    required_clearance=required_clearance,
                    max_search_m=max_search_m,
                    step_m=step_m,
                )
                if left_extent + right_extent <= 1e-9:
                    continue
                left_xy = (
                    anchor_xy[0] + left_extent * direction_xy[0],
                    anchor_xy[1] + left_extent * direction_xy[1],
                )
                right_xy = (
                    anchor_xy[0] - right_extent * direction_xy[0],
                    anchor_xy[1] - right_extent * direction_xy[1],
                )
                if not self._segment_is_safe(map_data, left_xy, right_xy, required_clearance, step_m):
                    continue
                center_xy = ((left_xy[0] + right_xy[0]) * 0.5, (left_xy[1] + right_xy[1]) * 0.5)
                half_width_m = 0.5 * math.hypot(left_xy[0] - right_xy[0], left_xy[1] - right_xy[1])
                node_cost = (
                    -self.node_width_weight * half_width_m
                    + self.node_distance_weight * (
                        (center_xy[0] - preview_xy[0]) ** 2 + (center_xy[1] - preview_xy[1]) ** 2
                    )
                    + self.node_angle_weight * (angle_offset_rad**2)
                )
                candidates.append(
                    _ChordCandidate(
                        sample_index=sample_index,
                        candidate_index=candidate_index,
                        preview_xy=preview_xy,
                        center_xy=center_xy,
                        direction_xy=direction_xy,
                        left_xy=left_xy,
                        right_xy=right_xy,
                        half_width_m=half_width_m,
                        direction_offset_rad=angle_offset_rad,
                        anchor_offset_m=center_offset_m,
                        node_cost=node_cost,
                    )
                )
                candidate_index += 1
        candidates.sort(key=lambda candidate: candidate.node_cost)
        return candidates

    def _run_dp(
        self,
        map_data: MapData,
        candidates_by_sample: list[list[_ChordCandidate]],
        *,
        required_clearance: float,
        sample_step_m: float,
    ) -> tuple[list[_ChordCandidate], list[CurveBandStripCell], float, int, int]:
        layer_costs: list[list[float]] = []
        backpointers: list[list[int]] = []
        strip_cache: dict[tuple[int, int, int], CurveBandStripCell | None] = {}
        feasible_transition_count = 0
        total_transition_count = 0

        for sample_index, candidates in enumerate(candidates_by_sample):
            if sample_index == 0:
                layer_costs.append([candidate.node_cost for candidate in candidates])
                backpointers.append([-1 for _ in candidates])
                continue

            prev_candidates = candidates_by_sample[sample_index - 1]
            prev_costs = layer_costs[-1]
            current_costs = [math.inf for _ in candidates]
            current_backpointers = [-1 for _ in candidates]
            for candidate_index, candidate in enumerate(candidates):
                best_cost = math.inf
                best_parent = -1
                # Only check top-5 geometrically nearest parents
                parent_dists = [
                    (
                        math.hypot(
                            candidate.center_xy[0] - p.center_xy[0],
                            candidate.center_xy[1] - p.center_xy[1],
                        ),
                        pi,
                        p,
                    )
                    for pi, p in enumerate(prev_candidates)
                    if math.hypot(candidate.center_xy[0] - p.center_xy[0], candidate.center_xy[1] - p.center_xy[1]) <= 1.5
                ]
                parent_dists.sort(key=lambda item: item[0])
                nearest_parents = [(pi, p) for _, pi, p in parent_dists[:5]]
                for parent_index, parent in nearest_parents:
                    total_transition_count += 1
                    if (
                        abs(candidate.half_width_m - parent.half_width_m) > 0.8
                        and min(candidate.half_width_m, parent.half_width_m) < 0.3
                    ):
                        continue
                    cache_key = (sample_index - 1, parent.candidate_index, candidate.candidate_index)
                    strip_cell = strip_cache.get(cache_key)
                    if strip_cell is None and cache_key not in strip_cache:
                        strip_cell = self._build_strip_cell(parent, candidate)
                        if not self._strip_cell_is_safe(map_data, strip_cell, required_clearance, sample_step_m):
                            strip_cell = None
                        strip_cache[cache_key] = strip_cell
                    if strip_cell is None:
                        continue
                    feasible_transition_count += 1
                    edge_cost = (
                        self.edge_center_smooth_weight
                        * (
                            (candidate.center_xy[0] - parent.center_xy[0]) ** 2
                            + (candidate.center_xy[1] - parent.center_xy[1]) ** 2
                        )
                        + self.edge_angle_smooth_weight
                        * ((candidate.direction_offset_rad - parent.direction_offset_rad) ** 2)
                    )
                    path_cost = prev_costs[parent_index] + candidate.node_cost + edge_cost
                    if path_cost < best_cost:
                        best_cost = path_cost
                        best_parent = parent_index
                current_costs[candidate_index] = best_cost
                current_backpointers[candidate_index] = best_parent
            layer_costs.append(current_costs)
            backpointers.append(current_backpointers)

        final_costs = layer_costs[-1]
        end_index = min(range(len(final_costs)), key=lambda index: final_costs[index])
        total_cost = final_costs[end_index]
        if not math.isfinite(total_cost):
            fallback_candidates = [candidates[0] for candidates in candidates_by_sample]
            fallback_cells = [
                self._build_strip_cell(left_candidate, right_candidate)
                for left_candidate, right_candidate in zip(fallback_candidates[:-1], fallback_candidates[1:])
            ]
            return fallback_candidates, fallback_cells, math.inf, feasible_transition_count, total_transition_count

        selected_indices = [0 for _ in candidates_by_sample]
        selected_indices[-1] = end_index
        for sample_index in range(len(candidates_by_sample) - 1, 0, -1):
            selected_indices[sample_index - 1] = backpointers[sample_index][selected_indices[sample_index]]
        selected_candidates = [
            candidates_by_sample[sample_index][candidate_index]
            for sample_index, candidate_index in enumerate(selected_indices)
        ]
        strip_cells = [
            self._build_strip_cell(left_candidate, right_candidate)
            for left_candidate, right_candidate in zip(selected_candidates[:-1], selected_candidates[1:])
        ]
        return selected_candidates, strip_cells, total_cost, feasible_transition_count, total_transition_count

    def _build_strip_cell(self, start: _ChordCandidate, end: _ChordCandidate) -> CurveBandStripCell:
        return CurveBandStripCell(
            start_index=start.sample_index,
            end_index=end.sample_index,
            left_start_xy=start.left_xy,
            right_start_xy=start.right_xy,
            right_end_xy=end.right_xy,
            left_end_xy=end.left_xy,
            metadata={
                "start_candidate_index": start.candidate_index,
                "end_candidate_index": end.candidate_index,
                "start_half_width_m": start.half_width_m,
                "end_half_width_m": end.half_width_m,
            },
        )

    def _batch_query_clearance(self, map_data: MapData, points_xy: np.ndarray) -> np.ndarray:
        """Batch query distance field. points_xy: (N, 2) array. Returns (N,) array of clearances."""
        if points_xy.size == 0:
            return np.zeros(0, dtype=float)
        origin_x, origin_y = map_data.origin_xy
        res = map_data.resolution
        xs = points_xy[:, 0]
        ys = points_xy[:, 1]
        grid_x = (xs - origin_x) / res - 0.5
        grid_y = (ys - origin_y) / res - 0.5
        valid = (grid_x >= 0) & (grid_y >= 0) & (grid_x <= map_data.cols - 1) & (grid_y <= map_data.rows - 1)
        x0 = np.floor(grid_x).astype(int)
        y0 = np.floor(grid_y).astype(int)
        x1 = np.minimum(x0 + 1, map_data.cols - 1)
        y1 = np.minimum(y0 + 1, map_data.rows - 1)
        wx = grid_x - x0
        wy = grid_y - y0
        df = map_data.distance_field
        result = np.where(
            valid,
            (1 - wx) * (1 - wy) * df[y0, x0]
            + wx * (1 - wy) * df[y0, x1]
            + (1 - wx) * wy * df[y1, x0]
            + wx * wy * df[y1, x1],
            0.0,
        )
        return result

    def _scan_safe_extent(
        self,
        map_data: MapData,
        anchor_xy: Point2D,
        direction_xy: Point2D,
        *,
        direction: float,
        required_clearance: float,
        max_search_m: float,
        step_m: float,
    ) -> float:
        _ = step_m  # step_m kept for compatibility, binary search uses its own precision
        max_dist = min(max_search_m, _MAX_SCAN_DIST_M)
        # exponential probe to find upper bound
        last_safe = 0.0
        probe = 0.1
        while probe <= max_dist:
            point_xy = (
                anchor_xy[0] + direction * probe * direction_xy[0],
                anchor_xy[1] + direction * probe * direction_xy[1],
            )
            if self._query_clearance(map_data, point_xy) + 1e-12 < required_clearance:
                break
            last_safe = probe
            probe *= 2.0
        hi = min(probe, max_dist)
        # binary search for precise boundary
        for _ in range(14):
            mid = (last_safe + hi) * 0.5
            point_xy = (
                anchor_xy[0] + direction * mid * direction_xy[0],
                anchor_xy[1] + direction * mid * direction_xy[1],
            )
            if self._query_clearance(map_data, point_xy) + 1e-12 >= required_clearance:
                last_safe = mid
            else:
                hi = mid
        return min(last_safe, max_dist)

    def _segment_is_safe(
        self,
        map_data: MapData,
        start_xy: Point2D,
        end_xy: Point2D,
        required_clearance: float,
        sample_step_m: float,
    ) -> bool:
        length_m = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
        sample_count = max(8, int(math.ceil(length_m / max(sample_step_m, 1e-6))) + 1)
        alphas = np.linspace(0.0, 1.0, sample_count)
        points = np.column_stack([
            (1.0 - alphas) * start_xy[0] + alphas * end_xy[0],
            (1.0 - alphas) * start_xy[1] + alphas * end_xy[1],
        ])
        clearances = self._batch_query_clearance(map_data, points)
        return bool(np.all(clearances + 1e-12 >= required_clearance))

    def _strip_cell_is_safe(
        self,
        map_data: MapData,
        strip_cell: CurveBandStripCell,
        required_clearance: float,
        sample_step_m: float,
    ) -> bool:
        _ = sample_step_m
        a, b, c, d = strip_cell.vertices_xy
        # Quick geometric prune
        max_edge = max(
            math.hypot(b[0] - a[0], b[1] - a[1]),
            math.hypot(c[0] - b[0], c[1] - b[1]),
            math.hypot(d[0] - c[0], d[1] - c[1]),
            math.hypot(a[0] - d[0], a[1] - d[1]),
        )
        if max_edge > 3.0:
            return False
        avg_width = 0.5 * (
            math.hypot(b[0] - a[0], b[1] - a[1]) + math.hypot(c[0] - d[0], c[1] - d[1])
        )
        if avg_width < required_clearance * 2.0 - 1e-9:
            return False

        # Coarse check: 4x3 grid with tighter threshold
        coarse_margin = 0.04
        coarse_threshold = required_clearance - coarse_margin
        u_coarse = 4
        v_coarse = 3
        u_vals_c = np.linspace(0.0, 1.0, u_coarse)
        v_vals_c = np.linspace(0.0, 1.0, v_coarse)
        uu_c, vv_c = np.meshgrid(u_vals_c, v_vals_c)
        uf_c = uu_c.ravel()
        vf_c = vv_c.ravel()
        lx_c = (1.0 - uf_c) * a[0] + uf_c * d[0]
        ly_c = (1.0 - uf_c) * a[1] + uf_c * d[1]
        rx_c = (1.0 - uf_c) * b[0] + uf_c * c[0]
        ry_c = (1.0 - uf_c) * b[1] + uf_c * c[1]
        pts_c = np.column_stack([
            (1.0 - vf_c) * lx_c + vf_c * rx_c,
            (1.0 - vf_c) * ly_c + vf_c * ry_c,
        ])
        clearances_c = self._batch_query_clearance(map_data, pts_c)
        if bool(np.all(clearances_c + 1e-12 >= coarse_threshold)):
            return True

        # Fine check: 10x7 grid
        u_fine = 10
        v_fine = 7
        u_vals_f = np.linspace(0.0, 1.0, u_fine)
        v_vals_f = np.linspace(0.0, 1.0, v_fine)
        uu_f, vv_f = np.meshgrid(u_vals_f, v_vals_f)
        uf_f = uu_f.ravel()
        vf_f = vv_f.ravel()
        lx_f = (1.0 - uf_f) * a[0] + uf_f * d[0]
        ly_f = (1.0 - uf_f) * a[1] + uf_f * d[1]
        rx_f = (1.0 - uf_f) * b[0] + uf_f * c[0]
        ry_f = (1.0 - uf_f) * b[1] + uf_f * c[1]
        pts_f = np.column_stack([
            (1.0 - vf_f) * lx_f + vf_f * rx_f,
            (1.0 - vf_f) * ly_f + vf_f * ry_f,
        ])
        clearances_f = self._batch_query_clearance(map_data, pts_f)
        return bool(np.all(clearances_f + 1e-12 >= required_clearance))

    def _center_offsets(self, map_data: MapData, required_clearance: float) -> tuple[float, ...]:
        if self.center_offset_count <= 1:
            return (0.0,)
        max_offset = max(map_data.resolution, required_clearance * 0.75)
        return tuple(np.linspace(-max_offset, max_offset, self.center_offset_count))

    def _compute_arc_lengths(self, points_xy: list[Point2D]) -> list[float]:
        if not points_xy:
            return []
        arc_lengths = [0.0]
        for prev_xy, next_xy in zip(points_xy[:-1], points_xy[1:]):
            arc_lengths.append(arc_lengths[-1] + math.hypot(next_xy[0] - prev_xy[0], next_xy[1] - prev_xy[1]))
        return arc_lengths

    def _compute_tangents(self, points_xy: list[Point2D]) -> list[Point2D]:
        if len(points_xy) == 1:
            return [(1.0, 0.0)]
        points = np.asarray(points_xy, dtype=float)
        tangents: list[Point2D] = []
        for index in range(len(points)):
            if index == 0:
                direction = points[1] - points[0]
            elif index == len(points) - 1:
                direction = points[-1] - points[-2]
            else:
                direction = points[index + 1] - points[index - 1]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                tangents.append((1.0, 0.0))
                continue
            direction /= norm
            tangents.append((float(direction[0]), float(direction[1])))
        return tangents

    def _query_clearance(self, map_data: MapData, point_xy: Point2D) -> float:
        return float(query_distance_field(map_data, point_xy))
