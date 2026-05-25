from __future__ import annotations

import math

import numpy as np

from formation.types import CurveBand, CurveBandSample, CurveBandStripCell, EmbeddingQPResult, FormationSpec, LocalPreviewPath, Point2D


class EmbeddingQPSolver:
    def __init__(
        self,
        *,
        max_heading_offset_rad: float = 0.30,
        heading_grid_size: int = 31,
        lateral_grid_size: int = 31,
        lateral_tracking_weight: float = 1.0,
        lateral_preview_weight: float = 0.15,
        lateral_smooth_weight: float = 8.0,
        heading_tracking_weight: float = 0.6,
        heading_smooth_weight: float = 4.0,
        violation_weight: float = 400.0,
        iterations: int = 180,
        alternating_iterations: int = 4,
    ) -> None:
        self.max_heading_offset_rad = max_heading_offset_rad
        self.heading_grid_size = max(3, heading_grid_size if heading_grid_size % 2 == 1 else heading_grid_size + 1)
        self.lateral_grid_size = max(3, lateral_grid_size if lateral_grid_size % 2 == 1 else lateral_grid_size + 1)
        self.lateral_tracking_weight = lateral_tracking_weight
        self.lateral_preview_weight = lateral_preview_weight
        self.lateral_smooth_weight = lateral_smooth_weight
        self.heading_tracking_weight = heading_tracking_weight
        self.heading_smooth_weight = heading_smooth_weight
        self.violation_weight = violation_weight
        self.iterations = iterations
        self.alternating_iterations = alternating_iterations

    def _build_corridor_margin_field(self, curve_band: CurveBand, grid_resolution_m: float = 0.05):
        points: list[tuple[float, float]] = []
        for cell in curve_band.strip_cells:
            points.extend(cell.vertices_xy)
        for sample in curve_band.samples:
            points.extend([sample.left_xy, sample.right_xy, sample.center_xy])
        if not points:
            return None
        min_x = min(p[0] for p in points) - 0.15
        max_x = max(p[0] for p in points) + 0.15
        min_y = min(p[1] for p in points) - 0.15
        max_y = max(p[1] for p in points) + 0.15
        nx = max(3, int((max_x - min_x) / grid_resolution_m) + 1)
        ny = max(3, int((max_y - min_y) / grid_resolution_m) + 1)
        margin_grid = np.full((ny, nx), -np.inf, dtype=np.float32)
        for iy in range(ny):
            y = min_y + iy * grid_resolution_m
            for ix in range(nx):
                x = min_x + ix * grid_resolution_m
                best = -math.inf
                for cell in curve_band.strip_cells:
                    m = self._polygon_signed_margin((x, y), cell.vertices_xy)
                    if m > best:
                        best = m
                margin_grid[iy, ix] = best
        return {"min_x": min_x, "min_y": min_y, "resolution_m": grid_resolution_m, "nx": nx, "ny": ny, "grid": margin_grid}

    def _margin_field_query(self, field: dict | None, point_xy):
        if field is None:
            return None
        fx = (point_xy[0] - field["min_x"]) / field["resolution_m"]
        fy = (point_xy[1] - field["min_y"]) / field["resolution_m"]
        ix = int(fx)
        iy = int(fy)
        if ix < 0 or iy < 0 or ix + 1 >= field["nx"] or iy + 1 >= field["ny"]:
            return None
        wx = fx - ix
        wy = fy - iy
        grid = field["grid"]
        try:
            result = (
                (1 - wx) * (1 - wy) * float(grid[iy, ix])
                + wx * (1 - wy) * float(grid[iy, ix + 1])
                + (1 - wx) * wy * float(grid[iy + 1, ix])
                + wx * wy * float(grid[iy + 1, ix + 1])
            )
            return result
        except IndexError:
            return None

    def _batch_margin_field_query(self, field: dict | None, points_xy: np.ndarray) -> np.ndarray:
        """Vectorized bilinear interpolation for N points. Returns (N,) array of margins."""
        if field is None:
            return np.full(len(points_xy), -math.inf, dtype=float)
        fx = (points_xy[:, 0] - field["min_x"]) / field["resolution_m"]
        fy = (points_xy[:, 1] - field["min_y"]) / field["resolution_m"]
        ix = np.floor(fx).astype(int)
        iy = np.floor(fy).astype(int)
        wx = fx - ix
        wy = fy - iy
        grid = field["grid"]
        ny, nx = grid.shape
        valid = (ix >= 0) & (iy >= 0) & (ix + 1 < nx) & (iy + 1 < ny)
        result = np.full(len(points_xy), -math.inf, dtype=float)
        if not np.any(valid):
            return result
        vx = ix[valid]
        vy = iy[valid]
        result[valid] = (
            (1 - wx[valid]) * (1 - wy[valid]) * grid[vy, vx]
            + wx[valid] * (1 - wy[valid]) * grid[vy, vx + 1]
            + (1 - wx[valid]) * wy[valid] * grid[vy + 1, vx]
            + wx[valid] * wy[valid] * grid[vy + 1, vx + 1]
        )
        return result

    def _batch_chord_margin_at_step(self, sample: CurveBandSample, points_xy: np.ndarray) -> np.ndarray:
        """Chord margin for N points at a single band sample step."""
        direction_x = sample.left_xy[0] - sample.right_xy[0]
        direction_y = sample.left_xy[1] - sample.right_xy[1]
        direction_norm = math.hypot(direction_x, direction_y)
        if direction_norm <= 1e-9:
            return np.full(len(points_xy), -math.inf, dtype=float)
        dx = direction_x / direction_norm
        dy = direction_y / direction_norm
        lateral_projection = (
            (points_xy[:, 0] - sample.center_xy[0]) * dx
            + (points_xy[:, 1] - sample.center_xy[1]) * dy
        )
        return sample.half_width_m - np.abs(lateral_projection)

    def _embedding_lattice_dp(
        self,
        curve_band: CurveBand,
        band_samples: list[CurveBandSample],
        formation: FormationSpec,
        lower_z: np.ndarray,
        upper_z: np.ndarray,
        base_headings: np.ndarray,
        n_z: int = 11,
        n_phi: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, list[float], list[list[float]], list[list[float]], float, float]:
        sample_count = len(band_samples)
        if sample_count <= 0:
            return np.zeros(0), np.zeros(0), [], [], [], 0.0, 0.0

        phi_grid_base = np.linspace(-self.max_heading_offset_rad, self.max_heading_offset_rad, n_phi)
        state_count = n_z * n_phi
        max_float_val = 1e30
        dp = np.full((sample_count, state_count), max_float_val, dtype=float)
        back = np.full((sample_count, state_count), -1, dtype=int)
        all_z = np.zeros((sample_count, n_z), dtype=float)
        all_phi = np.zeros((sample_count, n_phi), dtype=float)
        slots_arr = np.asarray(formation.slots, dtype=float)  # (4, 2)
        n_slots = slots_arr.shape[0]

        # Pre-compute (n_z, n_phi) grid indices for flat state access
        zi_indices = np.arange(n_z)
        pi_indices = np.arange(n_phi)
        zi_grid, pi_grid = np.meshgrid(zi_indices, pi_indices, indexing="ij")  # each (n_z, n_phi)
        zi_flat = zi_grid.ravel()  # (state_count,)
        pi_flat = pi_grid.ravel()  # (state_count,)

        per_step_slot_margins: list[list[list[float]]] = [[] for _ in range(sample_count)]
        all_node_margins: list[np.ndarray] = []

        w_m = self.violation_weight * 0.01
        w_z = self.lateral_preview_weight
        w_phi = self.heading_tracking_weight
        w_dz = self.lateral_smooth_weight
        w_dphi = self.heading_smooth_weight

        for m, sample in enumerate(band_samples):
            z_vals = np.linspace(float(lower_z[m]), float(upper_z[m]), n_z)
            all_z[m] = z_vals
            all_phi[m] = phi_grid_base.copy()
            base_heading = float(base_headings[m])

            # --- Batch compute all state centers, headings, and slot positions ---
            Z = z_vals[zi_grid]   # (n_z, n_phi)
            Phi = phi_grid_base[pi_grid]  # (n_z, n_phi)
            Z_flat = Z.ravel()
            Phi_flat = Phi.ravel()

            centers_x = sample.center_xy[0] + Z_flat * sample.normal_xy[0]  # (state_count,)
            centers_y = sample.center_xy[1] + Z_flat * sample.normal_xy[1]
            headings = base_heading + Phi_flat

            cos_h = np.cos(headings)  # (state_count,)
            sin_h = np.sin(headings)

            # Transform all slots at once: (state_count, n_slots) for x and y
            slot_x_all = (centers_x[:, None]
                          + cos_h[:, None] * slots_arr[None, :, 0]
                          - sin_h[:, None] * slots_arr[None, :, 1])  # (state_count, n_slots)
            slot_y_all = (centers_y[:, None]
                          + sin_h[:, None] * slots_arr[None, :, 0]
                          + cos_h[:, None] * slots_arr[None, :, 1])

            # Flatten all slot points for batch query: (state_count * n_slots, 2)
            all_points = np.column_stack([slot_x_all.ravel(), slot_y_all.ravel()])

            # Batch margin field query
            field_margins = self._batch_margin_field_query(self._margin_field, all_points)
            field_margins = field_margins.reshape(state_count, n_slots)

            # Batch chord margin at this step
            chord_margins = self._batch_chord_margin_at_step(sample, all_points)
            chord_margins = chord_margins.reshape(state_count, n_slots)

            # Combined margin per slot
            slot_margins_per_state = np.maximum(field_margins, chord_margins)  # (state_count, n_slots)
            node_margins = np.min(slot_margins_per_state, axis=1)  # (state_count,)
            all_node_margins.append(node_margins)

            # Store per-step slot margins for traceback (as lists for downstream compatibility)
            for si in range(state_count):
                per_step_slot_margins[m].append([float(slot_margins_per_state[si, j]) for j in range(n_slots)])

            # --- Node costs ---
            violation = np.maximum(0.0, -node_margins)
            dp[m] = (
                -node_margins * 0.5
                + w_z * (Z_flat ** 2)
                + w_phi * (Phi_flat ** 2)
                + w_m * (violation ** 2)
            )

            # --- Vectorized DP transition ---
            if m > 0:
                prev_Z = all_z[m - 1][zi_flat]  # (state_count,) z values from prev step
                prev_Phi = all_phi[m - 1][pi_flat]

                # (state_count, state_count) diff matrices
                dZ = np.abs(Z_flat[None, :] - prev_Z[:, None])  # (curr, prev)
                dPhi = Phi_flat[None, :] - prev_Phi[:, None]

                trans_cost = w_dz * (dZ ** 2) + w_dphi * (dPhi ** 2)
                mask = dZ <= 1.0

                # dp[m-1] is (state_count,); broadcast to (state_count, state_count)
                total = dp[m - 1][None, :] + trans_cost  # (curr, prev)
                total_masked = np.where(mask, total, max_float_val)
                best_prev_idx = np.argmin(total_masked, axis=1)  # (state_count,)
                best_prev = total_masked[np.arange(state_count), best_prev_idx]

                valid = best_prev < max_float_val
                dp[m, valid] += best_prev[valid]
                back[m, valid] = best_prev_idx[valid]

        # --- Traceback ---
        best_final_si = int(np.argmin(dp[-1]))
        state_sequence: list[int] = [best_final_si]
        for m in range(sample_count - 1, 0, -1):
            prev = back[m][state_sequence[-1]]
            if prev < 0:
                prev = 0
            state_sequence.append(prev)
        state_sequence.reverse()

        lateral_offsets = np.zeros(sample_count, dtype=float)
        heading_offsets = np.zeros(sample_count, dtype=float)
        total_violation = 0.0
        min_margin_overall = math.inf
        slot_corridor_margins_m: list[list[float]] = []
        per_step_final_margins: list[float] = []

        for m in range(sample_count):
            si = state_sequence[m]
            z_val = float(all_z[m][si // n_phi])
            phi_val = float(all_phi[m][si % n_phi])
            lateral_offsets[m] = z_val
            heading_offsets[m] = phi_val
            margin = float(all_node_margins[m][si]) if m < len(all_node_margins) and si < len(all_node_margins[m]) else -math.inf
            per_step_final_margins.append(margin)
            total_violation += max(0.0, -margin) ** 2
            min_margin_overall = min(min_margin_overall, margin)
            slot_margins = per_step_slot_margins[m][si] if m < len(per_step_slot_margins) and si < len(per_step_slot_margins[m]) else []
            slot_corridor_margins_m.append(slot_margins)

        offset_cost = float(
            self.lateral_tracking_weight * np.mean(np.square(lateral_offsets))
            + self.lateral_smooth_weight * (np.mean(np.square(np.diff(lateral_offsets))) if sample_count > 1 else 0.0)
        )
        heading_cost = float(
            self.heading_tracking_weight * np.mean(np.square(heading_offsets))
            + self.heading_smooth_weight * (np.mean(np.square(np.diff(heading_offsets))) if sample_count > 1 else 0.0)
        )

        return lateral_offsets, heading_offsets, per_step_final_margins, slot_corridor_margins_m, per_step_slot_margins, offset_cost, heading_cost

    def solve(
        self,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand,
        formation: FormationSpec,
    ) -> EmbeddingQPResult:
        sample_count = min(len(curve_band.samples), len(preview_path.points_xy))
        if sample_count <= 0:
            return EmbeddingQPResult(is_feasible=False, failure_reason="empty_preview_or_band")

        if getattr(self, "_margin_field_band_id", None) != id(curve_band):
            self._margin_field = self._build_corridor_margin_field(curve_band)
            self._margin_field_band_id = id(curve_band)
        band_samples = curve_band.samples[:sample_count]
        base_headings = np.asarray(
            [math.atan2(sample.tangent_xy[1], sample.tangent_xy[0]) for sample in band_samples],
            dtype=float,
        )
        lower_z, upper_z = self._lateral_search_bounds(curve_band, sample_count)

        lateral_offsets, heading_offsets, per_step_margins, slot_corridor_margins_m, _, offset_cost, heading_cost = (
            self._embedding_lattice_dp(
                curve_band, band_samples, formation, lower_z, upper_z, base_headings,
            )
        )

        center_points_xy = self._center_points_from_offsets(band_samples, lateral_offsets)
        heading_rads = [float(base_headings[index] + heading_offsets[index]) for index in range(sample_count)]
        slot_points_by_step_xy = [
            self._transform_slots(formation, center_xy, heading_rad)
            for center_xy, heading_rad in zip(center_points_xy, heading_rads)
        ]

        total_violation_cost = float(sum(max(0.0, -margin) ** 2 for margin in per_step_margins))
        min_corridor_margin_m = min(per_step_margins) if per_step_margins else 0.0
        per_step_violation_costs = [float(max(0.0, -margin) ** 2) for margin in per_step_margins]
        per_step_inside_counts = [sum(1 for sm in step_slots if sm >= 0.0) for step_slots in slot_corridor_margins_m]
        total_slot_count = sum(len(step_slots) for step_slots in slot_corridor_margins_m)
        inside_slot_count = sum(per_step_inside_counts)

        local_subgoal_xy = preview_path.local_subgoal_xy or preview_path.points_xy[min(sample_count - 1, len(preview_path.points_xy) - 1)]
        phi_reference = self._build_terminal_heading_reference(
            band_samples,
            base_headings,
            lateral_offsets,
            local_subgoal_xy,
            preview_path,
        )
        final_heading_error = 0.0
        if center_points_xy:
            target_heading = math.atan2(
                local_subgoal_xy[1] - center_points_xy[-1][1],
                local_subgoal_xy[0] - center_points_xy[-1][0],
            )
            final_heading_error = math.atan2(
                math.sin(target_heading - heading_rads[-1]),
                math.cos(target_heading - heading_rads[-1]),
            )
        preview_alignment_cost = float(np.mean(np.square(lateral_offsets))) if sample_count > 0 else 0.0

        return EmbeddingQPResult(
            lateral_offsets_m=lateral_offsets.tolist(),
            heading_offsets_rad=heading_offsets.tolist(),
            center_points_xy=center_points_xy,
            heading_rads=heading_rads,
            slot_points_by_step_xy=slot_points_by_step_xy,
            offset_cost=float(offset_cost),
            heading_cost=float(heading_cost),
            min_corridor_margin_m=float(min_corridor_margin_m),
            corridor_violation_cost=float(total_violation_cost),
            is_feasible=True,
            metadata={
                "preview_alignment_cost": preview_alignment_cost,
                "mean_lateral_offset_m": float(np.mean(lateral_offsets)) if sample_count > 0 else 0.0,
                "max_abs_lateral_offset_m": float(np.max(np.abs(lateral_offsets))) if sample_count > 0 else 0.0,
                "max_abs_heading_offset_rad": float(np.max(np.abs(heading_offsets))) if sample_count > 0 else 0.0,
                "mean_abs_heading_offset_rad": float(np.mean(np.abs(heading_offsets))) if sample_count > 0 else 0.0,
                "terminal_heading_offset_rad": float(heading_offsets[-1]) if heading_offsets.size else 0.0,
                "phi_reference_terminal_rad": float(phi_reference[-1]) if phi_reference.size else 0.0,
                "terminal_heading_error_rad": abs(float(final_heading_error)),
                "per_step_margins_m": per_step_margins,
                "per_step_violation_costs": per_step_violation_costs,
                "per_step_inside_counts": per_step_inside_counts,
                "slot_corridor_margins_m": slot_corridor_margins_m,
                "inside_slot_count": inside_slot_count,
                "total_slot_count": total_slot_count,
                "inside_slot_ratio": (inside_slot_count / total_slot_count) if total_slot_count > 0 else 0.0,
                "lateral_reference_m": lateral_offsets.tolist(),
                "heading_reference_rad": heading_offsets.tolist(),
            },
        )

    def _repair_trajectory(
        self,
        curve_band: CurveBand,
        band_samples: list[CurveBandSample],
        formation: FormationSpec,
        lower_z: np.ndarray,
        upper_z: np.ndarray,
        base_headings: np.ndarray,
        preview_path: LocalPreviewPath,
        local_subgoal_xy: Point2D,
        lateral_offsets: np.ndarray,
        heading_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        repaired_z = np.asarray(lateral_offsets, dtype=float).copy()
        repaired_phi = np.asarray(heading_offsets, dtype=float).copy()
        sample_count = repaired_z.size
        if sample_count <= 0:
            return repaired_z, repaired_phi, repaired_z.copy(), repaired_phi.copy()

        lower_phi = np.full(sample_count, -self.max_heading_offset_rad, dtype=float)
        upper_phi = np.full(sample_count, self.max_heading_offset_rad, dtype=float)
        repair_margin_threshold_m = 0.01

        def trajectory_stats(z_values: np.ndarray, phi_values: np.ndarray) -> tuple[list[float], float, float]:
            per_step_margins: list[float] = []
            total_violation = 0.0
            min_margin = math.inf
            for index, sample in enumerate(band_samples):
                center_xy = (
                    sample.center_xy[0] + float(z_values[index]) * sample.normal_xy[0],
                    sample.center_xy[1] + float(z_values[index]) * sample.normal_xy[1],
                )
                heading_rad = float(base_headings[index] + phi_values[index])
                slot_points_xy = self._transform_slots(formation, center_xy, heading_rad)
                step_margin_m, step_violation_cost, _, _ = self._evaluate_step(
                    curve_band,
                    index,
                    slot_points_xy,
                )
                per_step_margins.append(step_margin_m)
                total_violation += step_violation_cost
                min_margin = min(min_margin, step_margin_m)
            return per_step_margins, float(total_violation), (0.0 if min_margin is math.inf else float(min_margin))

        def trajectory_rank(z_values: np.ndarray, phi_values: np.ndarray) -> tuple[float, float, float, float]:
            _, total_violation, min_margin = trajectory_stats(z_values, phi_values)
            smooth_cost = self._trajectory_cost(
                z_values,
                np.zeros_like(z_values),
                tracking_weight=self.lateral_tracking_weight,
                smooth_weight=self.lateral_smooth_weight,
                center_weight=self.lateral_preview_weight,
            ) + self._trajectory_cost(
                phi_values,
                np.zeros_like(phi_values),
                tracking_weight=self.heading_tracking_weight,
                smooth_weight=self.heading_smooth_weight,
            )
            feasible_flag = 0.0 if (total_violation <= 1e-12 and min_margin >= -1e-9) else 1.0
            return feasible_flag, total_violation, -min_margin, smooth_cost

        original_z = repaired_z.copy()
        original_phi = repaired_phi.copy()

        for repair_iter in range(2):
            per_step_margins_m, total_violation, min_margin = trajectory_stats(repaired_z, repaired_phi)
            if total_violation <= 1e-12 and min_margin >= -1e-9:
                break
            if total_violation > 10.0:
                break

            active_indices: set[int] = set()
            for index, step_margin_m in enumerate(per_step_margins_m):
                if step_margin_m < repair_margin_threshold_m:
                    active_indices.update(range(max(0, index - 1), min(sample_count, index + 2)))
            if not active_indices:
                break

            phi_reference = self._build_terminal_heading_reference(
                band_samples,
                base_headings,
                repaired_z,
                local_subgoal_xy,
                preview_path,
            )
            improved = False
            for index in sorted(active_indices, key=lambda item: per_step_margins_m[item]):
                sample = band_samples[index]
                prev_z = float(repaired_z[index - 1]) if index > 0 else None
                next_z = float(repaired_z[index + 1]) if index + 1 < sample_count else None
                prev_phi = float(repaired_phi[index - 1]) if index > 0 else None
                next_phi = float(repaired_phi[index + 1]) if index + 1 < sample_count else None

                def candidate_rank(z_value: float, phi_value: float) -> tuple[float, float, float, float]:
                    center_xy = (
                        sample.center_xy[0] + z_value * sample.normal_xy[0],
                        sample.center_xy[1] + z_value * sample.normal_xy[1],
                    )
                    heading_rad = float(base_headings[index] + phi_value)
                    slot_points_xy = self._transform_slots(formation, center_xy, heading_rad)
                    step_margin_m, step_violation_cost, _, _ = self._evaluate_step(
                        curve_band,
                        index,
                        slot_points_xy,
                    )
                    local_objective = (
                        self.violation_weight * step_violation_cost
                        - step_margin_m
                        + self.lateral_preview_weight * (z_value**2)
                        + self.heading_tracking_weight * ((phi_value - float(phi_reference[index])) ** 2)
                    )
                    if prev_z is not None and prev_phi is not None:
                        local_objective += 0.5 * self.lateral_smooth_weight * ((z_value - prev_z) ** 2)
                        local_objective += 0.5 * self.heading_smooth_weight * ((phi_value - prev_phi) ** 2)
                    if next_z is not None and next_phi is not None:
                        local_objective += 0.5 * self.lateral_smooth_weight * ((next_z - z_value) ** 2)
                        local_objective += 0.5 * self.heading_smooth_weight * ((next_phi - phi_value) ** 2)
                    feasible_flag = 0.0 if (step_violation_cost <= 1e-12 and step_margin_m >= -1e-9) else 1.0
                    return feasible_flag, float(step_violation_cost), -float(step_margin_m), float(local_objective)

                current_z = float(repaired_z[index])
                current_phi = float(repaired_phi[index])
                best_z = current_z
                best_phi = current_phi
                best_rank = candidate_rank(current_z, current_phi)

                phi_candidates = list(self._coarse_to_fine_grid(float(lower_phi[index]), float(upper_phi[index])))
                for z_value in self._coarse_to_fine_grid(float(lower_z[index]), float(upper_z[index])):
                    for phi_value in phi_candidates:
                        rank = candidate_rank(float(z_value), float(phi_value))
                        if rank < best_rank:
                            best_rank = rank
                            best_z = float(z_value)
                            best_phi = float(phi_value)

                if abs(best_z - current_z) > 1e-6 or abs(best_phi - current_phi) > 1e-6:
                    repaired_z[index] = best_z
                    repaired_phi[index] = best_phi
                    improved = True

            if not improved:
                break

        if trajectory_rank(repaired_z, repaired_phi) > trajectory_rank(original_z, original_phi):
            repaired_z = original_z
            repaired_phi = original_phi

        repaired_z = np.clip(repaired_z, lower_z, upper_z)
        repaired_phi = np.clip(repaired_phi, lower_phi, upper_phi)
        phi_reference = self._build_terminal_heading_reference(
            band_samples,
            base_headings,
            repaired_z,
            local_subgoal_xy,
            preview_path,
        )
        preferred_z = self._build_lateral_reference(
            curve_band,
            band_samples,
            formation,
            lower_z,
            upper_z,
            repaired_phi,
            base_headings,
        )
        preferred_phi = self._build_heading_reference(
            curve_band,
            band_samples,
            formation,
            repaired_z,
            base_headings,
            phi_reference,
        )
        return repaired_z, repaired_phi, preferred_z, preferred_phi

    def _build_lateral_reference(
        self,
        curve_band: CurveBand,
        band_samples: list[CurveBandSample],
        formation: FormationSpec,
        lower_z: np.ndarray,
        upper_z: np.ndarray,
        heading_offsets: np.ndarray,
        base_headings: np.ndarray,
    ) -> np.ndarray:
        preferred = np.zeros(len(band_samples), dtype=float)
        for index, sample in enumerate(band_samples):
            best_value = 0.0
            best_objective = math.inf
            heading_offset = float(heading_offsets[index])
            for lateral_offset in self._coarse_to_fine_grid(float(lower_z[index]), float(upper_z[index])):
                center_xy = (
                    sample.center_xy[0] + lateral_offset * sample.normal_xy[0],
                    sample.center_xy[1] + lateral_offset * sample.normal_xy[1],
                )
                heading_rad = float(base_headings[index] + heading_offset)
                slot_points_xy = self._transform_slots(formation, center_xy, heading_rad)
                step_margin_m, step_violation_cost, _, _ = self._evaluate_step(
                    curve_band,
                    index,
                    slot_points_xy,
                )
                objective = (
                    self.violation_weight * step_violation_cost
                    - step_margin_m
                    + self.lateral_preview_weight * (lateral_offset**2)
                )
                if objective < best_objective:
                    best_objective = objective
                    best_value = lateral_offset
            preferred[index] = best_value
        return preferred

    def _build_heading_reference(
        self,
        curve_band: CurveBand,
        band_samples: list[CurveBandSample],
        formation: FormationSpec,
        lateral_offsets: np.ndarray,
        base_headings: np.ndarray,
        phi_reference: np.ndarray,
    ) -> np.ndarray:
        preferred = np.zeros(len(band_samples), dtype=float)
        for index, sample in enumerate(band_samples):
            center_xy = (
                sample.center_xy[0] + float(lateral_offsets[index]) * sample.normal_xy[0],
                sample.center_xy[1] + float(lateral_offsets[index]) * sample.normal_xy[1],
            )
            best_value = float(phi_reference[index])
            best_objective = math.inf
            for heading_offset in self._coarse_to_fine_grid(-self.max_heading_offset_rad, self.max_heading_offset_rad):
                heading_rad = float(base_headings[index] + heading_offset)
                slot_points_xy = self._transform_slots(formation, center_xy, heading_rad)
                step_margin_m, step_violation_cost, _, _ = self._evaluate_step(
                    curve_band,
                    index,
                    slot_points_xy,
                )
                objective = (
                    self.violation_weight * step_violation_cost
                    - step_margin_m
                    + self.heading_tracking_weight * ((heading_offset - float(phi_reference[index])) ** 2)
                )
                if objective < best_objective:
                    best_objective = objective
                    best_value = heading_offset
            preferred[index] = best_value
        return preferred

    def _build_terminal_heading_reference(
        self,
        band_samples: list[CurveBandSample],
        base_headings: np.ndarray,
        lateral_offsets: np.ndarray,
        local_subgoal_xy: Point2D,
        preview_path: LocalPreviewPath,
    ) -> np.ndarray:
        sample_count = len(band_samples)
        phi_reference = np.zeros(sample_count, dtype=float)
        if sample_count <= 0:
            return phi_reference

        preview_distance_m = float(preview_path.metadata.get("configured_preview_distance_m", preview_path.observation_distance_m))
        terminal_turn_ratio = 1.0 - min(max(preview_path.curve_end_distance_m / max(preview_distance_m, 1e-6), 0.0), 1.0)
        if terminal_turn_ratio <= 0.05:
            terminal_turn_ratio = 0.0
        tail_start = max(0, sample_count - max(3, sample_count // 3))

        center_points_xy = self._center_points_from_offsets(band_samples, lateral_offsets)
        for index, center_xy in enumerate(center_points_xy):
            dx_goal = local_subgoal_xy[0] - center_xy[0]
            dy_goal = local_subgoal_xy[1] - center_xy[1]
            if math.hypot(dx_goal, dy_goal) <= 1e-9:
                target_heading = float(base_headings[index - 1] if index > 0 else base_headings[index])
            else:
                target_heading = math.atan2(dy_goal, dx_goal)
            desired_phi = math.atan2(
                math.sin(target_heading - float(base_headings[index])),
                math.cos(target_heading - float(base_headings[index])),
            )
            if index >= tail_start and terminal_turn_ratio > 0.0:
                ramp = (index - tail_start + 1) / max(sample_count - tail_start, 1)
                phi_reference[index] = max(
                    -self.max_heading_offset_rad,
                    min(self.max_heading_offset_rad, terminal_turn_ratio * ramp * desired_phi),
                )
        return phi_reference

    def _evaluate_step(
        self,
        curve_band: CurveBand,
        index: int,
        slot_points_xy: list[Point2D],
    ) -> tuple[float, float, list[float], int]:
        slot_margins_m = [
            self._corridor_margin(curve_band, index, slot_xy)
            for slot_xy in slot_points_xy
        ]
        if not slot_margins_m:
            return 0.0, 0.0, [], 0
        step_margin_m = min(slot_margins_m)
        step_violation_cost = float(sum(max(0.0, -margin) ** 2 for margin in slot_margins_m))
        inside_count = sum(margin >= 0.0 for margin in slot_margins_m)
        return step_margin_m, step_violation_cost, slot_margins_m, inside_count

    def _corridor_margin(self, curve_band: CurveBand, index: int, point_xy: Point2D) -> float:
        field_val = self._margin_field_query(self._margin_field, point_xy)
        chord_val = -math.inf
        if 0 <= index < len(curve_band.samples):
            sample = curve_band.samples[index]
            direction_xy = (
                sample.left_xy[0] - sample.right_xy[0],
                sample.left_xy[1] - sample.right_xy[1],
            )
            direction_norm = math.hypot(direction_xy[0], direction_xy[1])
            if direction_norm > 1e-9:
                direction_xy = (direction_xy[0] / direction_norm, direction_xy[1] / direction_norm)
                lateral_projection = (
                    (point_xy[0] - sample.center_xy[0]) * direction_xy[0]
                    + (point_xy[1] - sample.center_xy[1]) * direction_xy[1]
                )
                chord_val = float(sample.half_width_m - abs(lateral_projection))
        if field_val is None:
            return chord_val
        return max(field_val, chord_val)

    def _adjacent_strip_cells(self, curve_band: CurveBand, index: int) -> list[CurveBandStripCell]:
        strip_cell_count = len(curve_band.strip_cells)
        if strip_cell_count <= 0:
            return []
        window = min(2, strip_cell_count)
        start = max(0, index - window)
        end = min(strip_cell_count, index + window + 1)
        return [curve_band.strip_cells[i] for i in range(start, end)]

    def _polygon_signed_margin(
        self,
        point_xy: Point2D,
        vertices_xy: tuple[Point2D, Point2D, Point2D, Point2D],
    ) -> float:
        points = list(vertices_xy)
        signed_area = 0.0
        for start_xy, end_xy in zip(points, points[1:] + points[:1]):
            signed_area += start_xy[0] * end_xy[1] - end_xy[0] * start_xy[1]
        orientation = 1.0 if signed_area >= 0.0 else -1.0
        min_margin = math.inf
        for start_xy, end_xy in zip(points, points[1:] + points[:1]):
            edge_x = end_xy[0] - start_xy[0]
            edge_y = end_xy[1] - start_xy[1]
            edge_length = math.hypot(edge_x, edge_y)
            if edge_length <= 1e-9:
                continue
            signed_cross = orientation * (
                edge_x * (point_xy[1] - start_xy[1]) - edge_y * (point_xy[0] - start_xy[0])
            )
            min_margin = min(min_margin, signed_cross / edge_length)
        return -math.inf if min_margin is math.inf else min_margin

    def _lateral_search_bounds(self, curve_band: CurveBand, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(sample_count, dtype=float)
        upper = np.zeros(sample_count, dtype=float)
        for index in range(sample_count):
            sample = curve_band.samples[index]
            points_xy = [sample.center_xy, sample.left_xy, sample.right_xy]
            for cell in self._adjacent_strip_cells(curve_band, index):
                points_xy.extend(cell.vertices_xy)
            projections = [
                (point_xy[0] - sample.center_xy[0]) * sample.normal_xy[0]
                + (point_xy[1] - sample.center_xy[1]) * sample.normal_xy[1]
                for point_xy in points_xy
            ]
            lower[index] = min(projections, default=0.0)
            upper[index] = max(projections, default=0.0)
            if lower[index] > upper[index]:
                lower[index], upper[index] = upper[index], lower[index]
        return lower, upper

    def _center_points_from_offsets(
        self,
        band_samples: list[CurveBandSample],
        lateral_offsets: np.ndarray,
    ) -> list[Point2D]:
        return [
            (
                sample.center_xy[0] + float(lateral_offsets[index]) * sample.normal_xy[0],
                sample.center_xy[1] + float(lateral_offsets[index]) * sample.normal_xy[1],
            )
            for index, sample in enumerate(band_samples)
        ]

    def _scalar_grid(self, lower: float, upper: float, grid_size: int) -> np.ndarray:
        if upper <= lower + 1e-12:
            return np.asarray([0.5 * (lower + upper)], dtype=float)
        return np.linspace(lower, upper, grid_size)

    def _coarse_to_fine_grid(self, lower: float, upper: float, best: float | None = None) -> list[float]:
        if upper <= lower + 1e-12:
            return [0.5 * (lower + upper)]
        center = best if best is not None else 0.5 * (lower + upper)
        all_values: list[float] = []
        seen: set[float] = set()
        half_range = 0.5 * (upper - lower)
        for fraction in (0.5, 0.25, 0.125):
            half = half_range * fraction
            lo = max(lower, center - half)
            hi = min(upper, center + half)
            for v in np.linspace(lo, hi, 9):
                vf = float(v)
                if vf not in seen:
                    seen.add(vf)
                    all_values.append(vf)
        return all_values

    def _iterative_grid_search(
        self,
        curve_band,
        formation,
        lower: float,
        upper: float,
        base_heading: float,
        sample,
        index: int,
        initial_z: float,
        heading_offset: float,
        phi_reference_value: float,
        prev_z: float | None,
        next_z: float | None,
        prev_phi: float | None,
        next_phi: float | None,
    ) -> tuple[float, float]:
        def objective(z_value: float, phi_value: float) -> float:
            center_xy = (
                sample.center_xy[0] + z_value * sample.normal_xy[0],
                sample.center_xy[1] + z_value * sample.normal_xy[1],
            )
            heading_rad = float(base_heading + phi_value)
            slot_points_xy = self._transform_slots(formation, center_xy, heading_rad)
            step_margin_m, step_violation_cost, _, _ = self._evaluate_step(
                curve_band, index, slot_points_xy,
            )
            obj_val = (
                self.violation_weight * step_violation_cost
                - step_margin_m
                + self.lateral_preview_weight * (z_value**2)
                + self.heading_tracking_weight * ((phi_value - phi_reference_value) ** 2)
            )
            if prev_z is not None and prev_phi is not None:
                obj_val += 0.5 * self.lateral_smooth_weight * ((z_value - prev_z) ** 2)
                obj_val += 0.5 * self.heading_smooth_weight * ((phi_value - prev_phi) ** 2)
            if next_z is not None and next_phi is not None:
                obj_val += 0.5 * self.lateral_smooth_weight * ((next_z - z_value) ** 2)
                obj_val += 0.5 * self.heading_smooth_weight * ((next_phi - phi_value) ** 2)
            return float(obj_val)

        center_z = initial_z
        center_phi = heading_offset
        for fraction in (0.5, 0.25, 0.125):
            half_z = 0.5 * (upper - lower) * fraction
            lo_z = max(lower, center_z - half_z)
            hi_z = min(upper, center_z + half_z)
            half_phi = 0.3 * fraction
            lo_phi = max(-self.max_heading_offset_rad, center_phi - half_phi)
            hi_phi = min(self.max_heading_offset_rad, center_phi + half_phi)
            best_obj = math.inf
            best_z = float(center_z)
            best_phi = float(center_phi)
            for z_val in np.linspace(lo_z, hi_z, 9):
                for phi_val in np.linspace(lo_phi, hi_phi, 9):
                    obj_val = objective(float(z_val), float(phi_val))
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_z = float(z_val)
                        best_phi = float(phi_val)
            center_z = float(best_z)
            center_phi = float(best_phi)
        return float(center_z), float(center_phi)

    def _solve_box_qp(
        self,
        reference: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        *,
        tracking_weight: float,
        smooth_weight: float,
        center_weight: float = 0.0,
    ) -> np.ndarray:
        values = np.clip(reference, lower_bounds, upper_bounds)
        if values.size == 0:
            return values

        lipschitz = 2.0 * (tracking_weight + center_weight) + 4.0 * smooth_weight + 1e-9
        step = 1.0 / lipschitz
        for _ in range(self.iterations):
            gradient = 2.0 * tracking_weight * (values - reference) + 2.0 * center_weight * values
            if values.size > 1:
                gradient[0] += 2.0 * smooth_weight * (values[0] - values[1])
                gradient[-1] += 2.0 * smooth_weight * (values[-1] - values[-2])
            if values.size > 2:
                gradient[1:-1] += 2.0 * smooth_weight * (2.0 * values[1:-1] - values[:-2] - values[2:])
            updated = np.clip(values - step * gradient, lower_bounds, upper_bounds)
            if np.max(np.abs(updated - values)) <= 1e-7:
                values = updated
                break
            values = updated
        return values

    def _trajectory_cost(
        self,
        values: np.ndarray,
        reference: np.ndarray,
        *,
        tracking_weight: float,
        smooth_weight: float,
        center_weight: float = 0.0,
    ) -> float:
        tracking_cost = tracking_weight * float(np.mean(np.square(values - reference)))
        center_cost = center_weight * float(np.mean(np.square(values)))
        smooth_cost = 0.0
        if values.size > 1:
            smooth_cost = smooth_weight * float(np.mean(np.square(np.diff(values))))
        return tracking_cost + center_cost + smooth_cost

    def _transform_slots(
        self,
        formation: FormationSpec,
        center_xy: Point2D,
        heading_rad: float,
    ) -> list[Point2D]:
        cos_heading = math.cos(heading_rad)
        sin_heading = math.sin(heading_rad)
        return [
            (
                center_xy[0] + cos_heading * float(slot[0]) - sin_heading * float(slot[1]),
                center_xy[1] + sin_heading * float(slot[0]) + cos_heading * float(slot[1]),
            )
            for slot in formation.slots
        ]
