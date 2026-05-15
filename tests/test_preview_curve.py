from __future__ import annotations

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
        self.assertEqual(preview.source_mode, "bezier")
        self.assertAlmostEqual(window.accumulated_distance_m, 7.0, places=6)
        self.assertAlmostEqual(preview.observation_distance_m, 7.0, places=6)
        self.assertAlmostEqual(preview.curve_end_distance_m, 7.0, places=6)
        self.assertIsNotNone(preview.local_subgoal_xy)
        self.assertLess(preview.local_subgoal_xy[0], config.corner_x)

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
        self.assertIn(preview.source_mode, {"bezier", "polyline_fallback"})
        self.assertGreaterEqual(preview.min_clearance, 0.0)
        self.assertIn(preview.failure_reason, {"", "clearance_below_threshold"})


if __name__ == "__main__":
    unittest.main()
