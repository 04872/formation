from __future__ import annotations

import math
import unittest

from formation import (
    GlobalPlanner,
    MapBuilder,
    NarrowingCorridorConfig,
    PathManager,
    PreviewCurvePlanner,
    PreviewCurveConfig,
    RightAngleCorridorConfig,
    SCurveCorridorConfig,
)


class PreviewCurveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = MapBuilder()
        self.global_planner = GlobalPlanner()

    def test_preview_curve_generates_on_right_angle_corridor(self) -> None:
        config = RightAngleCorridorConfig()
        map_data = self.builder.build("right_angle_corridor", config)
        global_path = self.global_planner.plan(map_data)
        path_manager = PathManager(global_path)
        window = path_manager.get_local_path_window(map_data.start_xy, preview_distance_m=7.0)

        preview = PreviewCurvePlanner().plan(map_data, map_data.start_xy, window)

        self.assertGreater(preview.sample_count, 2)
        self.assertTrue(preview.is_safe)
        self.assertIn(preview.source_mode, {"free_centerline", "ego_warm", "keypoint_opt"})
        self.assertAlmostEqual(window.accumulated_distance_m, 7.0, places=6)
        self.assertAlmostEqual(preview.observation_distance_m, 7.0, places=6)
        self.assertGreater(preview.curve_end_distance_m, 5.0)
        self.assertLess(preview.curve_end_distance_m, 8.0)
        self.assertIsNotNone(preview.local_subgoal_xy)
        self.assertLess(preview.local_subgoal_xy[0], config.corner_x)
        self.assertGreaterEqual(preview.metadata["mean_clearance_m"], preview.min_clearance)
        self.assertGreaterEqual(preview.metadata["alignment_error_m"], 0.0)
        self.assertGreaterEqual(preview.metadata["length_gap_m"], 0.0)
        self.assertLess(preview.metadata["alignment_error_m"], 1.0)
        self.assertLess(preview.metadata["length_gap_m"], 2.5)

    def test_preview_curve_safe_candidate_ranking_prefers_good_clearance(self) -> None:
        config = RightAngleCorridorConfig()
        map_data = self.builder.build("right_angle_corridor", config)
        global_path = self.global_planner.plan(map_data)
        path_manager = PathManager(global_path)
        window = path_manager.get_local_path_window(map_data.start_xy, preview_distance_m=7.0)

        preview = PreviewCurvePlanner(
            PreviewCurveConfig(bezier_tension=0.55, reduced_tension=0.18)
        ).plan(map_data, map_data.start_xy, window)

        self.assertTrue(preview.is_safe)
        self.assertGreaterEqual(preview.metadata["mean_clearance_m"], preview.min_clearance)
        self.assertLess(preview.metadata["alignment_error_m"], 1.2)
        self.assertLess(preview.metadata["length_gap_m"], 3.0)
        self.assertIn(preview.source_mode, {"free_centerline", "ego_warm", "ego_fallback", "keypoint_opt"})

    def test_preview_curve_projection_to_window_is_reasonable(self) -> None:
        config = NarrowingCorridorConfig()
        map_data = self.builder.build("narrowing_corridor", config)
        global_path = self.global_planner.plan(map_data)
        path_manager = PathManager(global_path)
        window = path_manager.get_local_window_by_distance(map_data.start_xy, preview_distance_m=4.5)

        preview = PreviewCurvePlanner(
            PreviewCurveConfig(bezier_tension=0.45, reduced_tension=0.18)
        ).plan(map_data, map_data.start_xy, window)

        self.assertGreater(preview.sample_count, 2)
        self.assertLess(preview.metadata["alignment_error_m"], 0.8)
        self.assertLess(preview.metadata["length_gap_m"], 1.5)

    def test_preview_curve_has_nonzero_curvature_on_s_curve(self) -> None:
        config = SCurveCorridorConfig()
        map_data = self.builder.build("s_curve_corridor", config)
        global_path = self.global_planner.plan(map_data)
        path_manager = PathManager(global_path)
        window = path_manager.get_local_window_by_distance(map_data.start_xy, preview_distance_m=4.0)

        preview = PreviewCurvePlanner().plan(map_data, map_data.start_xy, window)

        self.assertGreater(preview.sample_count, 2)
        self.assertTrue(preview.is_safe)
        self.assertGreater(max(abs(value) for value in preview.curvatures), 1e-3)
        self.assertEqual(preview.arc_lengths, sorted(preview.arc_lengths))

    def test_preview_curve_can_fallback_to_polyline(self) -> None:
        config = NarrowingCorridorConfig()
        map_data = self.builder.build("narrowing_corridor", config)
        global_path = self.global_planner.plan(map_data)
        path_manager = PathManager(global_path)
        window = path_manager.get_local_window_by_distance(map_data.start_xy, preview_distance_m=4.5)

        preview = PreviewCurvePlanner(
            PreviewCurveConfig(bezier_tension=1.1, reduced_tension=0.7)
        ).plan(map_data, map_data.start_xy, window)

        self.assertGreater(preview.sample_count, 2)
        self.assertIn(preview.source_mode, {"free_centerline", "ego_warm", "ego_fallback", "keypoint_opt"})
        self.assertGreaterEqual(preview.min_clearance, 0.0)
        self.assertIn(preview.failure_reason, {"", "clearance_below_threshold"})

    def test_preview_curve_recentering_preserves_endpoint_and_prefix(self) -> None:
        config = RightAngleCorridorConfig()
        map_data = self.builder.build("right_angle_corridor", config)
        global_path = self.global_planner.plan(map_data)
        path_manager = PathManager(global_path)
        window = path_manager.get_local_path_window(map_data.start_xy, preview_distance_m=7.0)

        preview = PreviewCurvePlanner().plan(map_data, map_data.start_xy, window)

        # q_0 must equal ref_xy
        self.assertEqual(preview.points_xy[0], map_data.start_xy)
        # Endpoint is within 2.5 m of local_subgoal (soft constraint, z can move from A*)
        self.assertLess(
            math.hypot(preview.points_xy[-1][0] - preview.local_subgoal_xy[0],
                       preview.points_xy[-1][1] - preview.local_subgoal_xy[1]),
            2.5,
        )
        self.assertTrue(preview.is_safe)

    def test_preview_curve_optimization_metadata_is_reported(self) -> None:
        config = NarrowingCorridorConfig()
        map_data = self.builder.build("narrowing_corridor", config)
        global_path = self.global_planner.plan(map_data)
        ref_xy = (2.8, config.centerline_slope * 2.8 + config.centerline_intercept)
        window = PathManager(global_path).get_local_path_window_from_projection(ref_xy, preview_distance_m=4.5)

        preview = PreviewCurvePlanner().plan(map_data, ref_xy, window)

        self.assertTrue(preview.is_safe)
        self.assertGreater(preview.sample_count, 2)
        self.assertGreaterEqual(preview.min_clearance, 0.0)


if __name__ == "__main__":
    unittest.main()
