from __future__ import annotations

import unittest

from formation import (
    FormationLibrary,
    FormationSelector,
    GlobalPlanner,
    MapBuilder,
    NarrowingCorridorConfig,
    PathManager,
    PreviewCurvePlanner,
    RightAngleCorridorConfig,
)
from formation.formation_selector import SelectorConfig, SelectorWeights


class FormationSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = MapBuilder()
        self.global_planner = GlobalPlanner()
        self.preview_planner = PreviewCurvePlanner()
        self.library = FormationLibrary.build_default(robot_radius=0.18, inter_robot_margin=0.10)
        self.selector = FormationSelector()
        self.robot_radius = 0.18
        self.safety_margin = 0.06

    def _build_right_angle_preview(self, preview_distance_m: float):
        config = RightAngleCorridorConfig(robot_radius=self.robot_radius)
        map_data = self.builder.build("right_angle_corridor", config)
        global_path = self.global_planner.plan(map_data)
        window = PathManager(global_path).get_local_path_window(
            map_data.start_xy,
            preview_distance_m=preview_distance_m,
        )
        preview = self.preview_planner.plan(map_data, map_data.start_xy, window)
        return map_data, preview

    def _build_narrowing_preview(self, x: float, preview_distance_m: float):
        config = NarrowingCorridorConfig(robot_radius=self.robot_radius)
        map_data = self.builder.build("narrowing_corridor", config)
        global_path = self.global_planner.plan(map_data)
        ref_xy = (x, config.centerline_slope * x + config.centerline_intercept)
        window = PathManager(global_path).get_local_path_window_from_projection(
            ref_xy,
            preview_distance_m=preview_distance_m,
        )
        preview = self.preview_planner.plan(map_data, ref_xy, window)
        return map_data, preview

    def test_selector_returns_safe_result_on_right_angle_corridor(self) -> None:
        map_data, preview = self._build_right_angle_preview(preview_distance_m=3.5)

        result = self.selector.select_target_formation(
            map_data,
            preview,
            self.library.list(),
            self.robot_radius,
            self.safety_margin,
        )

        evaluations = {evaluation.formation_name: evaluation for evaluation in result.evaluations}
        self.assertTrue(any(evaluation.band_feasible for evaluation in result.evaluations))
        self.assertEqual(result.selected_formation.name, result.selected_evaluation.formation_name)
        self.assertEqual(result.guide.formation_name, result.selected_formation.name)
        self.assertEqual(len(result.guide.guide_samples), preview.sample_count)
        self.assertIsNotNone(result.selected_evaluation.embedding_qp_result)
        self.assertTrue(result.selected_evaluation.embedding_qp_result.is_feasible)
        self.assertGreaterEqual(result.selected_evaluation.score_breakdown.offset_cost, 0.0)
        self.assertGreaterEqual(result.selected_evaluation.score_breakdown.heading_cost, 0.0)
        self.assertAlmostEqual(
            result.selected_evaluation.score_breakdown.offset_cost,
            result.selected_evaluation.embedding_qp_result.offset_cost,
            places=6,
        )
        self.assertAlmostEqual(
            result.selected_evaluation.score_breakdown.heading_cost,
            result.selected_evaluation.embedding_qp_result.heading_cost,
            places=6,
        )

    def test_selector_prefers_widest_safe_in_narrowing_corridor(self) -> None:
        map_data, preview = self._build_narrowing_preview(x=2.8, preview_distance_m=3.5)

        result = self.selector.select_target_formation(
            map_data,
            preview,
            self.library.list(),
            self.robot_radius,
            self.safety_margin,
        )

        evaluations = {evaluation.formation_name: evaluation for evaluation in result.evaluations}
        self.assertTrue(any(evaluation.is_safe for evaluation in result.evaluations))
        self.assertTrue(any(not evaluation.band_feasible for evaluation in result.evaluations))
        self.assertTrue(result.selected_evaluation.band_feasible)
        offsets = result.selected_evaluation.embedding_qp_result.lateral_offsets_m
        self.assertGreater(sum(abs(value) for value in offsets), 1e-3)
        self.assertLessEqual(abs(result.selected_evaluation.heading_offset_rad), self.selector.config.max_heading_offset_rad)

    def test_selector_exposes_terminal_alignment_metadata_near_goal(self) -> None:
        map_data, _ = self._build_right_angle_preview(preview_distance_m=3.5)
        global_path = self.global_planner.plan(map_data)
        ref_xy = (3.6, 2.6)
        window = PathManager(global_path).get_local_path_window_from_projection(ref_xy, preview_distance_m=3.5)
        preview = self.preview_planner.plan(map_data, ref_xy, window)

        result = self.selector.select_target_formation(
            map_data,
            preview,
            self.library.list(),
            self.robot_radius,
            self.safety_margin,
        )

        self.assertTrue(any(ev.is_safe for ev in result.evaluations))
        self.assertIn("terminal_heading_error_rad", result.selected_evaluation.embedding_qp_result.metadata)
        self.assertIn("phi_reference_terminal_rad", result.selected_evaluation.embedding_qp_result.metadata)
        self.assertGreater(
            abs(result.selected_evaluation.embedding_qp_result.metadata["phi_reference_terminal_rad"]), 0.01,
        )
        self.assertIn("terminal_heading_error_rad", result.guide.metadata)
        self.assertIn("phi_reference_terminal_rad", result.guide.metadata)

    def test_high_switch_cost_prefers_current_square(self) -> None:
        map_data, preview = self._build_right_angle_preview(preview_distance_m=3.5)
        square = self.library.get("square")
        compact = self.library.get("compact")
        selector = FormationSelector(
            SelectorConfig(
                weights=SelectorWeights(
                    corridor_margin=8.0,
                    embedding_cost=1.5,
                    corridor_violation=400.0,
                    switch_cost=4.0,
                    task_utility=1.0,
                )
            )
        )

        result = selector.select_target_formation(
            map_data,
            preview,
            [square, compact],
            self.robot_radius,
            self.safety_margin,
            current_formation=square,
        )

        evaluations = {evaluation.formation_name: evaluation for evaluation in result.evaluations}
        self.assertTrue(evaluations["square"].is_safe)
        self.assertEqual(evaluations["square"].score_breakdown.switch_cost, 0.0)
        self.assertEqual(result.selected_formation.name, "square")
        self.assertFalse(result.guide.switched)


if __name__ == "__main__":
    unittest.main()
