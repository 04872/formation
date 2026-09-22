import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from formation import MapData, Pose2D, TubeRRTConfig, TubeRRTNode, TubeRRTPlanner, project_robot_paths, transform_slots
from formation.tube_rrt import interpolate_pose


class TubeRRTTest(unittest.TestCase):
    def map_data(self, primitives=None):
        occupancy = np.zeros((160, 200), dtype=bool)
        occupancy[[0, -1], :] = True
        occupancy[:, [0, -1]] = True
        return MapData("test", 0.1, 20.0, 16.0, (-10.0, -8.0), occupancy, occupancy.copy(), np.ones_like(occupancy, dtype=float), (-8.0, 0.0), (8.0, 0.0), primitives or [], 0.15, 0.09, 0.06)

    def planner(self, primitives=None, **kwargs):
        slots = np.asarray([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
        return TubeRRTPlanner(self.map_data(primitives), slots, Pose2D(-8.0, 0.0, 0.0), config=TubeRRTConfig(**kwargs))

    def test_rigid_transform_projection_metric_and_interpolation(self):
        slots = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        points = transform_slots(Pose2D(2.0, 3.0, np.pi / 2), slots)
        np.testing.assert_allclose(points, [[2.0, 4.0], [1.0, 3.0]])
        self.assertEqual([len(path) for path in project_robot_paths([Pose2D(0, 0), Pose2D(1, 0)], slots)], [2, 2])
        planner = self.planner()
        self.assertAlmostEqual(planner.metric(Pose2D(0, 0, 0), Pose2D(3, 4, np.pi)), 5.0 + planner.formation_radius * np.pi)
        midpoint = interpolate_pose(Pose2D(0, 0, np.pi - 0.1), Pose2D(0, 0, -np.pi + 0.1), 0.5)
        self.assertAlmostEqual(abs(midpoint.yaw - np.pi), 0.0, places=7)

    def test_exact_circle_collision_and_boundary(self):
        planner = self.planner([{"type": "circle", "center_xy": (-7.5, -0.5), "radius": 0.2}])
        self.assertLess(planner.clearance(Pose2D(-7.0, 0.0)), 0.0)
        self.assertGreater(planner.clearance(Pose2D(-4.0, 0.0)), 0.0)
        self.assertLess(planner.clearance(Pose2D(-9.6, 0.0)), 0.0)
        gap = self.planner([{"type": "circle", "center_xy": (-4.0, 0.0), "radius": 0.2}])
        self.assertGreater(gap.clearance(Pose2D(-4.0, 0.0)), 0.0)

    def test_overlap_is_strict(self):
        planner = self.planner()
        a = TubeRRTNode(Pose2D(0, 0), 0.5, None, 0)
        b = TubeRRTNode(Pose2D(1.01, 0), 0.5, None, 0)
        self.assertFalse(planner.tube_overlap(a, b))
        c = TubeRRTNode(Pose2D(1.0, 0), 0.5, None, 0)
        self.assertFalse(planner.tube_overlap(a, c))

        obstacle_planner = self.planner([{"type": "circle", "center_xy": (0.0, 0.0), "radius": 0.2}])
        left = Pose2D(-0.2, 0.0, 0.0)
        right = Pose2D(0.2, 0.0, 0.0)
        left_node = TubeRRTNode(left, obstacle_planner.safety_radius(left), None, 0)
        right_node = TubeRRTNode(right, obstacle_planner.safety_radius(right), None, 0)
        self.assertTrue(obstacle_planner.tube_overlap(left_node, right_node))
        for alpha in np.linspace(0.0, 1.0, 101):
            self.assertGreaterEqual(obstacle_planner.clearance(interpolate_pose(left, right, alpha)), 0.0)

    def test_center_circle_between_slots_does_not_inflate_formation_hull(self):
        planner = self.planner(
            [{"type": "circle", "center_xy": (0.0, 0.0), "radius": 0.2}],
            metric_step=0.2,
            goal_connect_distance=0.25,
            max_iterations=100,
        )
        planner._sample_pose = lambda: Pose2D(8.0, 0.0, 0.0)
        result = planner.plan()
        self.assertTrue(result.success)
        nearest = min(result.path_poses, key=lambda pose: pose.x**2 + pose.y**2)
        self.assertAlmostEqual(nearest.x, 0.0)
        self.assertAlmostEqual(nearest.y, 0.0)
        self.assertGreater(planner.clearance(nearest), 0.0)

    def test_progress_reporting_and_default_silence(self):
        planner = self.planner(metric_step=0.2, goal_connect_distance=0.25, max_iterations=100, progress_interval=2)
        planner._sample_pose = lambda: Pose2D(8.0, 0.0, 0.0)
        output = StringIO()
        with redirect_stdout(output):
            result = planner.plan()
        self.assertTrue(result.success)
        self.assertIn("progress iteration=2/100", output.getvalue())
        self.assertIn("goal connected iteration=", output.getvalue())

        silent_planner = self.planner(metric_step=0.2, goal_connect_distance=0.25, max_iterations=100)
        silent_planner._sample_pose = lambda: Pose2D(8.0, 0.0, 0.0)
        output = StringIO()
        with redirect_stdout(output):
            self.assertTrue(silent_planner.plan().success)
        self.assertEqual(output.getvalue(), "")

    def test_descendant_costs_follow_rewired_parent(self):
        planner = self.planner()
        nodes = [
            TubeRRTNode(Pose2D(0.0, 0.0), 1.0, None, 0.0),
            TubeRRTNode(Pose2D(1.0, 0.0), 1.0, 0, 10.0),
            TubeRRTNode(Pose2D(2.0, 0.0), 1.0, 1, 20.0),
        ]
        nodes[1].parent = 0
        nodes[1].cost = planner.metric(nodes[0].pose, nodes[1].pose)
        planner._update_descendant_costs(nodes, 1)
        self.assertAlmostEqual(nodes[2].cost, nodes[1].cost + planner.metric(nodes[1].pose, nodes[2].pose))

        planner = self.planner(seed=4, max_iterations=1800, metric_step=0.6, goal_bias=0.2)
        first = planner.plan()
        second = self.planner(seed=4, max_iterations=1800, metric_step=0.6, goal_bias=0.2).plan()
        self.assertTrue(first.success)
        self.assertEqual(first.path_poses, second.path_poses)
        self.assertEqual(first.path_poses[0], Pose2D(-8.0, 0.0, 0.0))
        self.assertEqual((first.path_poses[-1].x, first.path_poses[-1].y), (8.0, 0.0))
        self.assertEqual(first.path_poses[-1].yaw, first.path_poses[-2].yaw)
        self.assertNotEqual(first.path_poses[-1].yaw, 0.0)
        self.assertTrue(any(node.parent is not None and node.parent > index for index, node in enumerate(first.tree_nodes)))
        self.assertTrue(all(first.tree_nodes[a].parent == b for a, b in first.tree_edges))
        for index, parent in first.tree_edges:
            self.assertTrue(first.tree_nodes[index].radius > 0.0)
            self.assertTrue(first.tree_nodes[parent].radius > 0.0)
            self.assertTrue(planner.tube_overlap(first.tree_nodes[index], first.tree_nodes[parent]))
            self.assertAlmostEqual(
                first.tree_nodes[index].cost,
                first.tree_nodes[parent].cost + planner.metric(first.tree_nodes[parent].pose, first.tree_nodes[index].pose),
            )
        failed = self.planner(seed=4, max_iterations=0).plan()
        self.assertFalse(failed.success)
        self.assertEqual(failed.failure_reason, "iteration budget exhausted")
        blocked = TubeRRTPlanner(self.map_data([{"type": "circle", "center_xy": (-8.5, -0.5), "radius": 0.2}]), np.asarray([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]), Pose2D(-8.0, 0.0, 0.0), config=TubeRRTConfig(max_iterations=1)).plan()
        self.assertFalse(blocked.success)
        self.assertEqual(blocked.failure_reason, "start is in collision")


if __name__ == "__main__":
    unittest.main()
