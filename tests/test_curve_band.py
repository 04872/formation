from __future__ import annotations

import unittest

from formation import GlobalPlanner, MapBuilder, PathManager, PreviewCurvePlanner, RightAngleCorridorConfig, SCurveCorridorConfig
from formation.curve_band import CurveBandBuilder


class CurveBandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = MapBuilder()
        self.global_planner = GlobalPlanner()
        self.preview_planner = PreviewCurvePlanner()
        self.band_builder = CurveBandBuilder()
        self.robot_radius = 0.09
        self.safety_margin = 0.06

    def _build_preview(self, map_name: str, config, preview_distance_m: float):
        map_data = self.builder.build(map_name, config)
        global_path = self.global_planner.plan(map_data)
        window = PathManager(global_path).get_local_path_window(
            map_data.start_xy,
            preview_distance_m=preview_distance_m,
        )
        preview = self.preview_planner.plan(map_data, map_data.start_xy, window)
        return map_data, preview

    def _assert_band_has_valid_strip_cells(self, curve_band) -> None:
        self.assertGreater(len(curve_band.samples), 0)
        self.assertEqual(len(curve_band.strip_cells), curve_band.sample_count - 1)
        for sample in curve_band.samples:
            self.assertGreater(sample.half_width_m, 0.0)
            self.assertTrue(sample.arc_length_s >= 0.0)
        for cell in curve_band.strip_cells:
            self.assertTrue(cell.start_index >= 0)
            self.assertEqual(cell.end_index, cell.start_index + 1)

    def test_curve_band_builds_on_right_angle_corridor(self) -> None:
        map_data, preview = self._build_preview(
            "right_angle_corridor",
            RightAngleCorridorConfig(robot_radius=self.robot_radius),
            preview_distance_m=7.0,
        )
        curve_band = self.band_builder.build(map_data, preview, self.robot_radius, self.safety_margin)

        self.assertEqual(curve_band.sample_count, preview.sample_count)
        self.assertEqual(curve_band.source_mode, preview.source_mode)
        self.assertAlmostEqual(
            curve_band.metadata["required_clearance_m"],
            self.robot_radius + self.safety_margin,
            places=6,
        )
        self._assert_band_has_valid_strip_cells(curve_band)
        widths = [sample.half_width_m for sample in curve_band.samples]
        self.assertLess(min(widths), max(widths))
        self.assertGreater(max(widths), 0.5)
        self.assertTrue(any(abs(sample.direction_offset_rad) > 1e-6 for sample in curve_band.samples))
        self.assertGreater(curve_band.metadata["max_search_m"], max(map_data.width_m, map_data.height_m))
        self.assertGreater(curve_band.metadata["lateral_step_m"], 0.0)
        self.assertTrue(any(w > 0.3 for w in widths))
        self.assertTrue(all(w >= 0.0 for w in widths))
        self.assertTrue(all(sample.arc_length_s >= 0.0 for sample in curve_band.samples))
        self.assertEqual(len(curve_band.strip_cells), curve_band.sample_count - 1)
        self.assertIn("selected_candidate_indices", curve_band.metadata)
        self.assertIn("selected_half_widths_m", curve_band.metadata)
        self.assertIn("dp_total_cost", curve_band.metadata)

    def test_curve_band_builds_on_s_curve(self) -> None:
        map_data, preview = self._build_preview(
            "s_curve_corridor",
            SCurveCorridorConfig(robot_radius=self.robot_radius),
            preview_distance_m=4.0,
        )
        curve_band = self.band_builder.build(map_data, preview, self.robot_radius, self.safety_margin)

        self.assertEqual(curve_band.sample_count, preview.sample_count)
        self.assertEqual(curve_band.source_mode, preview.source_mode)
        self._assert_band_has_valid_strip_cells(curve_band)


if __name__ == "__main__":
    unittest.main()
