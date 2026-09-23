from __future__ import annotations

import math
import unittest

import numpy as np

from formation import (
    MapBuilder,
    NarrowEntranceConfig,
    NarrowingCorridorConfig,
    ObstacleClusterConfig,
    PostFenceConfig,
    RightAngleCorridorConfig,
    RandomCirclesConfig,
    SCurveCorridorConfig,
    SinglePostConfig,
)


class MapBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = MapBuilder()

    def test_all_maps_build_with_valid_fields(self) -> None:
        scenarios = {
            "right_angle_corridor": RightAngleCorridorConfig(),
            "s_curve_corridor": SCurveCorridorConfig(),
            "obstacle_cluster": ObstacleClusterConfig(),
            "narrow_entrance": NarrowEntranceConfig(),
            "narrowing_corridor": NarrowingCorridorConfig(),
            "random_circles": RandomCirclesConfig(seed=31, obstacle_count=8),
            "post_fence": PostFenceConfig(),
            "single_post": SinglePostConfig(),
        }

        for name, config in scenarios.items():
            with self.subTest(map_type=name):
                map_data = self.builder.build(name, config)
                self.assertEqual(map_data.occupancy.ndim, 2)
                self.assertEqual(map_data.occupancy.shape, map_data.inflated_occupancy.shape)
                self.assertEqual(map_data.occupancy.shape, map_data.distance_field.shape)
                self.assertGreater(map_data.rows, 0)
                self.assertGreater(map_data.cols, 0)
                self.assertTrue(np.any(map_data.occupancy))
                self.assertTrue(np.any(~map_data.occupancy))

                start_rc = map_data.world_to_grid(map_data.start_xy)
                goal_rc = map_data.world_to_grid(map_data.goal_xy)
                self.assertFalse(map_data.is_occupied(start_rc))
                self.assertFalse(map_data.is_occupied(goal_rc))
                self.assertFalse(map_data.is_occupied(start_rc, inflated=True))
                self.assertFalse(map_data.is_occupied(goal_rc, inflated=True))

                self.assertTrue(np.all(map_data.distance_field[map_data.occupancy] == 0.0))
                self.assertGreater(float(np.max(map_data.distance_field[~map_data.occupancy])), 0.0)

    def test_random_circles_are_deterministic_bounded_and_clear_of_terminals(self) -> None:
        config = RandomCirclesConfig(seed=31, obstacle_count=8)
        first = self.builder.build("random_circles", config)
        second = self.builder.build("random_circles", config)
        self.assertEqual(first.obstacle_primitives, second.obstacle_primitives)
        self.assertEqual(len(first.obstacle_primitives), config.obstacle_count)
        self.assertGreater(
            len({round(float(primitive["radius"]), 12) for primitive in first.obstacle_primitives}),
            1,
        )
        for primitive in first.obstacle_primitives:
            self.assertEqual(primitive["type"], "circle")
            center = np.asarray(primitive["center_xy"], dtype=float)
            radius = float(primitive["radius"])
            self.assertGreaterEqual(radius, config.radius_min)
            self.assertLessEqual(radius, config.radius_max)
            self.assertGreaterEqual(center[0] - radius, -config.width_m / 2.0)
            self.assertLessEqual(center[0] + radius, config.width_m / 2.0)
            self.assertGreaterEqual(center[1] - radius, -config.height_m / 2.0)
            self.assertLessEqual(center[1] + radius, config.height_m / 2.0)
            self.assertGreater(np.linalg.norm(center - config.start_xy), config.start_clearance_radius + radius - 1e-12)
            self.assertGreater(np.linalg.norm(center - config.goal_xy), config.goal_clearance_radius + radius - 1e-12)

    def test_post_fence_spans_map_and_single_post_blocks_straight_line(self) -> None:
        config = PostFenceConfig()
        fence = self.builder.build("post_fence", config)
        ys = sorted(float(p["center_xy"][1]) for p in fence.obstacle_primitives)
        self.assertTrue(all(p["type"] == "circle" and p["center_xy"][0] == config.fence_x for p in fence.obstacle_primitives))
        np.testing.assert_allclose(np.diff(ys), config.post_spacing)
        self.assertIn(0.0, ys)
        self.assertLess(ys[0] + config.height_m / 2.0, config.post_spacing)
        self.assertLess(config.height_m / 2.0 - ys[-1], config.post_spacing)

        post = self.builder.build("single_post", SinglePostConfig())
        self.assertEqual(post.obstacle_primitives, [{"type": "circle", "center_xy": (0.0, 0.0), "radius": 0.15}])
        self.assertTrue(post.is_occupied(post.world_to_grid((0.0, 0.0))))

    def test_right_angle_corridor_legs_and_corner_are_free(self) -> None:
        config = RightAngleCorridorConfig()
        map_data = self.builder.build("right_angle_corridor", config)
        samples = [
            (-4.0, config.horizontal_center_y),
            (1.0, config.horizontal_center_y),
            (config.corner_x, config.corner_y),
            (config.vertical_center_x, 0.0),
            (config.vertical_center_x, 3.5),
        ]
        for point in samples:
            rc = map_data.world_to_grid(point)
            self.assertFalse(map_data.is_occupied(rc))

    def test_s_curve_centerline_samples_are_free(self) -> None:
        config = SCurveCorridorConfig()
        map_data = self.builder.build("s_curve_corridor", config)
        for x in (-5.0, -2.5, 0.0, 2.5, 5.0):
            phase = 2.0 * math.pi * (x - config.start_xy[0]) / (config.goal_xy[0] - config.start_xy[0])
            y = config.amplitude * math.sin(phase)
            rc = map_data.world_to_grid((x, y))
            self.assertFalse(map_data.is_occupied(rc))

    def test_obstacle_cluster_contains_center_obstacles(self) -> None:
        config = ObstacleClusterConfig()
        map_data = self.builder.build("obstacle_cluster", config)
        x_grid, y_grid = np.meshgrid(
            np.linspace(map_data.origin_xy[0], map_data.origin_xy[0] + map_data.width_m, map_data.cols, endpoint=False),
            np.linspace(map_data.origin_xy[1], map_data.origin_xy[1] + map_data.height_m, map_data.rows, endpoint=False),
        )
        center_mask = (np.abs(x_grid) <= 2.5) & (np.abs(y_grid) <= 2.0)
        self.assertGreater(int(np.count_nonzero(map_data.occupancy & center_mask)), 0)

    def test_narrow_entrance_has_tight_neck(self) -> None:
        config = NarrowEntranceConfig()
        map_data = self.builder.build("narrow_entrance", config)
        neck_center = map_data.world_to_grid((0.0, 0.0))
        self.assertFalse(map_data.is_occupied(neck_center))

        neck_upper = map_data.world_to_grid((0.0, 0.8))
        self.assertTrue(map_data.is_occupied(neck_upper))

        left_room_upper = map_data.world_to_grid((-3.0, 0.8))
        self.assertFalse(map_data.is_occupied(left_room_upper))

    def test_narrowing_corridor_shrinks_from_wide_end_to_throat(self) -> None:
        config = NarrowingCorridorConfig()
        map_data = self.builder.build("narrowing_corridor", config)

        def free_band_half_width(x: float) -> float:
            center_y = config.centerline_slope * x + config.centerline_intercept
            y_samples = np.arange(map_data.origin_xy[1], map_data.origin_xy[1] + map_data.height_m, map_data.resolution)
            free_mask = []
            for y in y_samples:
                rc = map_data.world_to_grid((x, y))
                free_mask.append(not map_data.is_occupied(rc))
            free_mask = np.asarray(free_mask, dtype=bool)
            free_indices = np.flatnonzero(free_mask)
            self.assertGreater(len(free_indices), 0)
            free_ys = y_samples[free_indices]
            return float(np.max(np.abs(free_ys - center_y)))

        wide_half_width = free_band_half_width(-3.8)
        throat_half_width = free_band_half_width(3.2)
        tail_half_width = free_band_half_width(4.6)

        self.assertGreater(wide_half_width, throat_half_width)
        self.assertGreaterEqual(tail_half_width + 0.05, throat_half_width)
        self.assertFalse(map_data.is_occupied(map_data.world_to_grid(config.start_xy)))
        self.assertFalse(map_data.is_occupied(map_data.world_to_grid(config.goal_xy)))


if __name__ == "__main__":
    unittest.main()
