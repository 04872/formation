from __future__ import annotations

import math
import unittest

from formation import (
    CurveBandBuilder,
    EmbeddingQPSolver,
    FormationLibrary,
    GlobalPlanner,
    MapBuilder,
    NarrowingCorridorConfig,
    PathManager,
    PreviewCurvePlanner,
    RightAngleCorridorConfig,
)


class EmbeddingQPSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = MapBuilder()
        self.global_planner = GlobalPlanner()
        self.preview_planner = PreviewCurvePlanner()
        self.band_builder = CurveBandBuilder()
        self.library = FormationLibrary.build_default(robot_radius=0.09, inter_robot_margin=0.10)
        self.solver = EmbeddingQPSolver()
        self.robot_radius = 0.09
        self.safety_margin = 0.06

    def _build_right_angle_inputs(self):
        config = RightAngleCorridorConfig(robot_radius=self.robot_radius)
        map_data = self.builder.build("right_angle_corridor", config)
        global_path = self.global_planner.plan(map_data)
        window = PathManager(global_path).get_local_path_window(map_data.start_xy, preview_distance_m=3.5)
        preview = self.preview_planner.plan(map_data, map_data.start_xy, window)
        curve_band = self.band_builder.build(map_data, preview, self.robot_radius, self.safety_margin)
        return map_data, preview, curve_band

    def _build_right_angle_turn_inputs(self):
        config = RightAngleCorridorConfig(robot_radius=self.robot_radius)
        map_data = self.builder.build("right_angle_corridor", config)
        global_path = self.global_planner.plan(map_data)
        ref_xy = (3.6, 2.6)
        window = PathManager(global_path).get_local_path_window_from_projection(ref_xy, preview_distance_m=3.5)
        preview = self.preview_planner.plan(map_data, ref_xy, window)
        curve_band = self.band_builder.build(map_data, preview, self.robot_radius, self.safety_margin)
        return map_data, preview, curve_band

    def _build_narrowing_inputs(self):
        config = NarrowingCorridorConfig(robot_radius=self.robot_radius)
        map_data = self.builder.build("narrowing_corridor", config)
        global_path = self.global_planner.plan(map_data)
        ref_xy = (2.8, config.centerline_slope * 2.8 + config.centerline_intercept)
        window = PathManager(global_path).get_local_path_window_from_projection(ref_xy, preview_distance_m=3.5)
        preview = self.preview_planner.plan(map_data, ref_xy, window)
        curve_band = self.band_builder.build(map_data, preview, self.robot_radius, self.safety_margin)
        return map_data, preview, curve_band

    def _assert_slot_clearance(self, map_data, result, threshold: float) -> None:
        for slot_points_xy in result.slot_points_by_step_xy:
            for slot_xy in slot_points_xy:
                clearance = self.band_builder._query_clearance(map_data, slot_xy)
                self.assertGreaterEqual(clearance, threshold)

    def _assert_slot_points_in_corridor(self, curve_band, result) -> None:
        for index, slot_points_xy in enumerate(result.slot_points_by_step_xy):
            for slot_xy in slot_points_xy:
                margin = self.solver._corridor_margin(curve_band, index, slot_xy)
                self.assertGreaterEqual(margin, -1e-9,
                    f"slot point {slot_xy} at step {index} outside corridor, margin={margin}")

    def test_solver_keeps_near_zero_offset_in_symmetric_corridor(self) -> None:
        _, preview, curve_band = self._build_right_angle_inputs()
        result = self.solver.solve(preview, curve_band, self.library.get("compact"))

        self.assertTrue(result.is_feasible)
        self.assertEqual(len(result.lateral_offsets_m), preview.sample_count)
        self.assertLess(max(abs(value) for value in result.lateral_offsets_m), 0.25)
        self.assertLess(max(abs(value) for value in result.heading_offsets_rad), 0.20)

    def test_solver_biases_offsets_toward_open_side(self) -> None:
        _, preview, curve_band = self._build_narrowing_inputs()
        result = self.solver.solve(preview, curve_band, self.library.get("compact"))

        self.assertTrue(result.is_feasible)
        self.assertGreater(sum(abs(value) for value in result.lateral_offsets_m), 1e-3)
        self.assertGreaterEqual(result.min_corridor_margin_m, -1e-9)
        self.assertEqual(result.corridor_violation_cost, 0.0)

    def test_solver_returns_smooth_sequences(self) -> None:
        _, preview, curve_band = self._build_narrowing_inputs()
        result = self.solver.solve(preview, curve_band, self.library.get("square"))

        self.assertTrue(result.is_feasible)
        if len(result.lateral_offsets_m) > 1:
            self.assertLess(max(abs(a - b) for a, b in zip(result.lateral_offsets_m[:-1], result.lateral_offsets_m[1:])), 0.35)
        if len(result.heading_offsets_rad) > 1:
            self.assertLess(max(abs(a - b) for a, b in zip(result.heading_offsets_rad[:-1], result.heading_offsets_rad[1:])), 0.20)

    def test_solver_turns_terminal_heading_toward_local_subgoal(self) -> None:
        _, preview, curve_band = self._build_right_angle_turn_inputs()
        result = self.solver.solve(preview, curve_band, self.library.get("square"))

        self.assertTrue(result.is_feasible)
        self.assertGreaterEqual(result.min_corridor_margin_m, -1e-9)
        self.assertGreater(abs(result.metadata["phi_reference_terminal_rad"]), 0.05)
        for value in result.heading_offsets_rad:
            self.assertLessEqual(abs(value), self.solver.max_heading_offset_rad + 1e-9)
        goal_heading = math.atan2(
            preview.local_subgoal_xy[1] - result.center_points_xy[-1][1],
            preview.local_subgoal_xy[0] - result.center_points_xy[-1][0],
        )
        final_terminal_error = abs(
            math.atan2(
                math.sin(goal_heading - result.heading_rads[-1]),
                math.cos(goal_heading - result.heading_rads[-1]),
            )
        )
        self.assertAlmostEqual(final_terminal_error, result.metadata["terminal_heading_error_rad"], places=6)

    def test_solver_slot_points_in_step_local_corridor(self) -> None:
        map_data, preview, curve_band = self._build_right_angle_turn_inputs()
        formation = self.library.get("compact")
        result = self.solver.solve(preview, curve_band, formation)

        self.assertTrue(result.is_feasible)
        self._assert_slot_points_in_corridor(curve_band, result)
        self.assertGreaterEqual(result.min_corridor_margin_m, -1e-9)
        self.assertEqual(result.corridor_violation_cost, 0.0)

    def test_solver_reports_corridor_margin_from_step_local_strip_cells(self) -> None:
        _, preview, curve_band = self._build_right_angle_inputs()
        formation = self.library.get("compact")
        result = self.solver.solve(preview, curve_band, formation)

        self.assertTrue(result.is_feasible)
        self.assertGreaterEqual(result.min_corridor_margin_m, -1e-9)
        self.assertEqual(result.corridor_violation_cost, 0.0)
        self.assertGreater(result.metadata["inside_slot_ratio"], 0.95)
        computed_min_margin = math.inf
        for index, slot_points_xy in enumerate(result.slot_points_by_step_xy):
            for slot_xy in slot_points_xy:
                margin = self.solver._corridor_margin(curve_band, index, slot_xy)
                computed_min_margin = min(computed_min_margin, margin)
        self.assertAlmostEqual(result.min_corridor_margin_m, computed_min_margin, places=6)


if __name__ == "__main__":
    unittest.main()
