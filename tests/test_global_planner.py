from __future__ import annotations

import unittest

from formation import (
    GlobalPlanner,
    MapBuilder,
    NarrowEntranceConfig,
    NarrowingCorridorConfig,
    ObstacleClusterConfig,
    RightAngleCorridorConfig,
    SCurveCorridorConfig,
)


class GlobalPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = MapBuilder()
        self.planner = GlobalPlanner()

    def test_planner_runs_on_all_maps(self) -> None:
        scenarios = {
            "right_angle_corridor": RightAngleCorridorConfig(),
            "s_curve_corridor": SCurveCorridorConfig(),
            "obstacle_cluster": ObstacleClusterConfig(),
            "narrow_entrance": NarrowEntranceConfig(),
            "narrowing_corridor": NarrowingCorridorConfig(),
        }

        for name, config in scenarios.items():
            with self.subTest(map_type=name):
                map_data = self.builder.build(name, config)
                path = self.planner.plan(map_data)
                self.assertGreater(len(path.grid_path_rc), 1)
                self.assertGreater(len(path.waypoints_xy), 1)
                self.assertLessEqual(len(path.waypoints_xy), len(path.grid_path_rc))
                self.assertEqual(path.grid_path_rc[0], map_data.world_to_grid(map_data.start_xy))
                self.assertEqual(path.grid_path_rc[-1], map_data.world_to_grid(map_data.goal_xy))
                for rc in path.grid_path_rc:
                    self.assertFalse(map_data.is_occupied(rc, inflated=True))
                for rc in path.waypoint_grid_rc:
                    self.assertFalse(map_data.is_occupied(rc, inflated=True))

    def test_right_angle_corridor_path_turns_at_corner(self) -> None:
        config = RightAngleCorridorConfig()
        map_data = self.builder.build("right_angle_corridor", config)
        path = self.planner.plan(map_data)
        self.assertGreaterEqual(len(path.waypoints_xy), 3)
        self.assertTrue(any(y <= config.corner_y + 0.8 for _, y in path.waypoints_xy[1:-1]))
        self.assertTrue(any(x >= config.corner_x - 0.8 for x, _ in path.waypoints_xy[1:-1]))
        self.assertLessEqual(len(path.waypoints_xy), 5)

    def test_s_curve_keeps_multiple_turning_waypoints(self) -> None:
        map_data = self.builder.build("s_curve_corridor", SCurveCorridorConfig())
        path = self.planner.plan(map_data)
        self.assertGreaterEqual(len(path.waypoints_xy), 4)

    def test_narrow_entrance_path_passes_through_neck(self) -> None:
        map_data = self.builder.build("narrow_entrance", NarrowEntranceConfig())
        path = self.planner.plan(map_data)
        self.assertTrue(
            any(
                abs(row - map_data.rows // 2) <= 2 and abs(col - map_data.cols // 2) <= 25
                for row, col in path.grid_path_rc
            )
        )

    def test_narrowing_corridor_path_reaches_tail(self) -> None:
        config = NarrowingCorridorConfig()
        map_data = self.builder.build("narrowing_corridor", config)
        path = self.planner.plan(map_data)
        self.assertTrue(any(x >= 4.3 for x, _ in path.waypoints_xy))


if __name__ == "__main__":
    unittest.main()
