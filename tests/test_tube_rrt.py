import math
import unittest

import numpy as np

from formation import (
    FirstOrderCellModel,
    FormationLibrary,
    MapBuilder,
    ChartCellTubeRRTPlanner,
    Pose2D,
    RandomCirclesConfig,
    SecondOrderCellModel,
    SinglePostConfig,
    TubeRRTConfig,
    TubeRRTPlanner,
    make_tube_rrt_planner,
)
from formation.tube_cell_first_order import interpolate_pose
from formation.types import wrap_to_pi

SLOTS = FormationLibrary.build_default(0.113).get("square").slots


def rotation(theta: float) -> np.ndarray:
    return np.array(((math.cos(theta), -math.sin(theta)), (math.sin(theta), math.cos(theta))))


def random_pose(rng: np.random.Generator, spread: float = 0.6) -> Pose2D:
    return Pose2D(*rng.uniform(-spread, spread, 2), float(rng.uniform(-math.pi, math.pi)))


def nearby_pose(rng: np.random.Generator, pose: Pose2D, spread: float = 0.5, yaw_spread: float = 1.0) -> Pose2D:
    return Pose2D(pose.x + rng.uniform(-spread, spread), pose.y + rng.uniform(-spread, spread),
                  wrap_to_pi(pose.yaw + rng.uniform(-yaw_spread, yaw_spread)))


class CellModelTest(unittest.TestCase):
    def test_first_order_norm_is_max_linearised_robot_displacement(self) -> None:
        model = FirstOrderCellModel(SLOTS)
        rng = np.random.default_rng(0)
        for _ in range(50):
            center, query = random_pose(rng), random_pose(rng)
            u = np.array((query.x - center.x, query.y - center.y))
            phi = wrap_to_pi(query.yaw - center.yaw)
            a = SLOTS @ rotation(center.yaw).T
            expected = max(np.linalg.norm(u + phi * np.array((-ai[1], ai[0]))) for ai in a)
            self.assertAlmostEqual(model.distance(center, query), expected, places=12)
            xs, ys, yaws = (np.array([v]) for v in (center.x, center.y, center.yaw))
            batch = model.distances_to(xs, ys, np.cos(yaws), np.sin(yaws), yaws, query)
            self.assertAlmostEqual(float(batch[0]), expected, places=12)

    def test_first_order_overlap_is_homothetic_for_equal_yaw(self) -> None:
        model = FirstOrderCellModel(SLOTS, eta=1.0)
        a = model.make_cell(Pose2D(0.0, 0.0, 0.4), 0.3)
        for gap, expected in ((0.49, True), (0.51, False)):
            b = model.make_cell(Pose2D(gap, 0.0, 0.4), 0.2)
            self.assertEqual(model.overlap(a, b) is not None, expected)

    def test_portals_lie_in_both_cells(self) -> None:
        rng = np.random.default_rng(1)
        for model in (FirstOrderCellModel(SLOTS), SecondOrderCellModel(SLOTS)):
            accepted = 0
            for _ in range(300):
                a = model.make_cell(random_pose(rng), float(rng.uniform(0.05, 0.5)))
                b = model.make_cell(nearby_pose(rng, a.pose), float(rng.uniform(0.05, 0.5)))
                certificate = model.overlap(a, b)
                if certificate is None:
                    continue
                accepted += 1
                self.assertGreater(model.slack(a, certificate.portal), 0.0)
                self.assertGreater(model.slack(b, certificate.portal), 0.0)
            self.assertGreater(accepted, 20, model.name)

    def test_only_second_order_cell_bounds_true_robot_displacement(self) -> None:
        rng = np.random.default_rng(2)
        worst = {}
        for model in (FirstOrderCellModel(SLOTS, eta=1.0), SecondOrderCellModel(SLOTS, eta=1.0)):
            cell = model.make_cell(Pose2D(0.0, 0.0, 0.0), 0.4)
            ratios = []
            for _ in range(30000):
                query = Pose2D(*rng.uniform(-0.45, 0.45, 2), float(rng.uniform(-1.6, 1.6)))
                if model.slack(cell, query) > 0.0:
                    moved = SLOTS @ rotation(query.yaw).T + (query.x, query.y)
                    ratios.append(float(np.max(np.linalg.norm(moved - SLOTS, axis=1))) / cell.radius)
            self.assertGreater(len(ratios), 500, model.name)
            worst[model.name] = max(ratios)
        self.assertGreater(worst["first_order"], 1.0)
        self.assertLess(worst["second_order"], 1.0)

    def test_second_order_overlap_matches_brute_force(self) -> None:
        model = SecondOrderCellModel(SLOTS, eta=1.0)
        rng = np.random.default_rng(3)
        found_count = 0
        for _ in range(250):
            a = model.make_cell(random_pose(rng), float(rng.uniform(0.1, 0.4)))
            b = model.make_cell(nearby_pose(rng, a.pose, 0.5, 1.0), float(rng.uniform(0.1, 0.4)))
            samples = [nearby_pose(rng, a.pose, 0.5, 1.2) for _ in range(1000)]
            if any(model.slack(a, q) > 0.02 and model.slack(b, q) > 0.02 for q in samples):
                found_count += 1
                self.assertIsNotNone(model.overlap(a, b))
        self.assertGreater(found_count, 20)
        self.assertGreater(model.stats["socp_accept"], 0)
        far = model.make_cell(Pose2D(a.pose.x + 1.0, a.pose.y, a.pose.yaw), 0.3)
        self.assertIsNone(model.overlap(model.make_cell(a.pose, 0.3), far))
        self.assertGreater(model.stats["quick_reject"], 0)

    def test_second_order_inner_step_stays_inside(self) -> None:
        model = SecondOrderCellModel(SLOTS)
        cell = model.make_cell(Pose2D(0.0, 0.0, 0.0), 0.3)
        target = Pose2D(0.1, 0.05, 2.0)
        step = model.inner_step(cell, target)
        pose = interpolate_pose(cell.pose, target, 0.999 * step / model.distance(cell.pose, target))
        self.assertGreater(model.slack(cell, pose), 0.0)


class PlannerTest(unittest.TestCase):
    def build(self, cell_model: str, **overrides):
        map_data = MapBuilder().build("single_post", SinglePostConfig(robot_radius=0.113, safety_margin=0.06))
        config = TubeRRTConfig(cell_model=cell_model, **overrides)
        return make_tube_rrt_planner(map_data, SLOTS * 2.0, Pose2D(*map_data.start_xy, 0.0), config=config)

    def test_factory_keeps_orientation_cell_as_default(self) -> None:
        self.assertIs(type(self.build("orientation")), TubeRRTPlanner)
        self.assertIs(type(self.build("second_order")), ChartCellTubeRRTPlanner)
        with self.assertRaises(ValueError):
            TubeRRTPlanner(self.build("orientation").map_data, SLOTS, Pose2D(0.0, 0.0, 0.0),
                           config=TubeRRTConfig(cell_model="first_order"))

    def test_orientation_cell_reproduces_original_run(self) -> None:
        # scripts/visualize_tube_rrt.py --map random_circles --seed 7 --slot-scale 2 --anytime
        map_data = MapBuilder().build("random_circles", RandomCirclesConfig(seed=7, robot_radius=0.113, safety_margin=0.06))
        config = TubeRRTConfig(seed=7, max_iterations=2500, stop_on_first_goal=False, yaw_slices=16, cell_eta=0.98)
        result = make_tube_rrt_planner(map_data, SLOTS * 2.0, Pose2D(*map_data.start_xy, 0.0), config=config).plan()
        self.assertTrue(result.success)
        self.assertEqual((len(result.tree_nodes), result.first_goal_iteration), (2143, 236))
        self.assertAlmostEqual(result.path_cost, 12.406, places=3)

    def test_both_cell_models_reach_goal_deterministically(self) -> None:
        for cell_model in ("first_order", "second_order"):
            first = self.build(cell_model).plan()
            second = self.build(cell_model).plan()
            self.assertTrue(first.success, cell_model)
            self.assertEqual(first.path_poses, second.path_poses)
            self.assertAlmostEqual(first.path_poses[-1].x, self.build(cell_model).goal_xy[0])

    def test_second_order_route_is_certified_and_collision_free(self) -> None:
        planner = self.build("second_order", stop_on_first_goal=False, max_iterations=800)
        result = planner.plan()
        self.assertTrue(result.success)
        self.assertGreater(result.bottleneck, 0.0)
        nodes = result.tree_nodes
        for parent, child in zip(result.path_nodes, result.path_nodes[1:]):
            portal = nodes[child].certificate.portal
            self.assertGreater(planner.cells.slack(nodes[parent].cell, portal), 0.0)
            self.assertGreater(planner.cells.slack(nodes[child].cell, portal), 0.0)
        costs = [nodes[i].cost for i in result.path_nodes]
        self.assertEqual(costs, sorted(costs))
        self.assertAlmostEqual(result.path_cost, costs[-1])

    def test_rewired_descendants_keep_consistent_costs(self) -> None:
        planner = self.build("first_order", stop_on_first_goal=False, max_iterations=600, record_trace=True)
        result = planner.plan()
        self.assertTrue(any(event.rewires for event in result.trace))
        for node in result.tree_nodes:
            if node.parent is not None:
                parent = result.tree_nodes[node.parent]
                expected = parent.cost + planner.edge_cost(parent, node, node.certificate)
                self.assertAlmostEqual(node.cost, expected, places=9)

    def test_rejects_unknown_cell_model_and_colliding_start(self) -> None:
        with self.assertRaises(ValueError):
            self.build("third_order")
        planner = self.build("second_order")
        planner.start = Pose2D(*planner.map_data.origin_xy, 0.0)
        result = planner.plan()
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "start is in collision")


if __name__ == "__main__":
    unittest.main()
