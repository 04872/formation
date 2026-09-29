import math
import unittest

import numpy as np
from matplotlib.path import Path

from formation import (
    FormationLibrary,
    FrontierConfig,
    MapBuilder,
    PolyhedralFrontierPlanner,
    Pose2D,
    RandomCirclesConfig,
    SinglePostConfig,
    TubeRRTConfig,
    make_tube_rrt_planner,
)
from formation.tube_rrt_frontier import overlap_preference

# Square formation with 1 m between neighbouring robots (slots at (+-0.5, +-0.5)).
SLOTS = FormationLibrary.build_default(0.113).get("square").slots * 2.0


def random_circles_planner(**overrides) -> PolyhedralFrontierPlanner:
    map_data = MapBuilder().build("random_circles", RandomCirclesConfig(seed=7, robot_radius=0.113, safety_margin=0.06))
    config = TubeRRTConfig(cell_model="polyhedral", seed=7, **overrides)
    return make_tube_rrt_planner(map_data, SLOTS, Pose2D(*map_data.start_xy, 0.0), config=config)


def valid_cells(planner: PolyhedralFrontierPlanner, count: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    ox, oy = planner.map_data.origin_xy
    cells = []
    while len(cells) < count:
        pose = Pose2D(float(rng.uniform(ox, ox + planner.map_data.width_m)),
                      float(rng.uniform(oy, oy + planner.map_data.height_m)), float(rng.uniform(-math.pi, math.pi)))
        cell = planner.cells.make_cell(pose)
        if cell.valid:
            cells.append(cell)
    return cells


def box_points(rng: np.random.Generator, cell, count: int) -> np.ndarray:
    g, limit = cell.guard, cell.yaw_limit
    return np.column_stack((cell.pose.x + rng.uniform(-g, g, count), cell.pose.y + rng.uniform(-g, g, count),
                            cell.pose.yaw + rng.uniform(-limit, limit, count)))


class PolyhedralCellTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = random_circles_planner()
        self.model = self.planner.cells

    def test_common_rotation_radius_is_formation_radius(self) -> None:
        self.assertAlmostEqual(self.model.rho, math.sqrt(0.5), places=12)
        self.assertTrue(np.allclose(np.linalg.norm(SLOTS, axis=1), self.model.rho))

    def test_rows_follow_proximity_queries_and_guard(self) -> None:
        d_s = self.model.safety_distance
        for cell in valid_cells(self.planner, 40):
            clearance, normals = self.model.proximity(self.model.robot_positions(cell.pose))
            obstacle_rows = np.flatnonzero(~cell.guard_rows)
            for row, (robot, component) in zip(obstacle_rows, cell.pairs):
                self.assertTrue(np.allclose(cell.normals[row], normals[robot, component]))
                self.assertAlmostEqual(cell.offsets[row], clearance[robot, component] - d_s)
                self.assertEqual(cell.kappa[row], 1.0)
            active = np.zeros(clearance.shape, dtype=bool)
            for robot, component in cell.pairs:
                active[robot, component] = True
            inactive = clearance[~active]
            expected = min(float(inactive.min()) - d_s, self.model.max_extent)
            self.assertAlmostEqual(cell.guard, expected)
            self.assertLessEqual(len(cell.pairs), self.model.max_active_per_robot * len(SLOTS))
            self.assertAlmostEqual(cell.clearance, float(clearance.min()))

    def test_every_configuration_in_a_cell_is_collision_free(self) -> None:
        rng = np.random.default_rng(1)
        d_s = self.model.safety_distance
        inside_total = 0
        for cell in valid_cells(self.planner, 60):
            points = np.vstack((box_points(rng, cell, 3000), cell.samples))
            inside = points[self.model.contains(cell, points)]
            inside_total += len(inside)
            clearance = self.planner.clearances(inside[:, 0], inside[:, 1], inside[:, 2])
            self.assertGreaterEqual(float(clearance.min()), d_s - 1e-9)
        self.assertGreater(inside_total, 20000)

    def test_translational_extent_reaches_the_boundary(self) -> None:
        rng = np.random.default_rng(2)
        for cell in valid_cells(self.planner, 30):
            angles = rng.uniform(-math.pi, math.pi, 16)
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
            dtheta = float(rng.uniform(-0.9, 0.9)) * cell.yaw_reach
            lengths = self.model.translational_extent(cell, directions, dtheta)
            self.assertTrue(np.all(lengths > 0.0))
            for u, length in zip(directions, lengths):
                at = lambda t: Pose2D(cell.pose.x + t * u[0], cell.pose.y + t * u[1], cell.pose.yaw + dtheta)
                self.assertGreater(self.model.slack(cell, at(0.999 * length)), 0.0)
                self.assertLess(self.model.slack(cell, at(1.001 * length)), 0.0)
                self.assertAlmostEqual(self.model.slack(cell, at(length)), 0.0, places=9)

    def test_ray_extent_stays_inside(self) -> None:
        rng = np.random.default_rng(3)
        for cell in valid_cells(self.planner, 30):
            dc, dphi = rng.normal(size=2), float(rng.uniform(-1.0, 1.0))
            t = self.model.ray_extent(cell, dc, dphi)
            at = lambda s: Pose2D(cell.pose.x + s * dc[0], cell.pose.y + s * dc[1], cell.pose.yaw + s * dphi)
            self.assertGreater(self.model.slack(cell, at(0.999 * t)), 0.0)
            self.assertLess(self.model.slack(cell, at(1.001 * t)), 1e-12)

    def test_slice_polygon_equals_fixed_yaw_section(self) -> None:
        rng = np.random.default_rng(4)
        for cell in valid_cells(self.planner, 20):
            dtheta = float(rng.uniform(-0.95, 0.95)) * cell.yaw_limit
            polygon, labels = self.model.slice_polygon(cell, dtheta)
            points = box_points(rng, cell, 4000)
            points[:, 2] = cell.pose.yaw + dtheta
            slack = self.model.slack_many(cell, points)
            clear = np.abs(slack) > 1e-6
            if not len(polygon):
                self.assertFalse(np.any(slack[clear] > 0.0))
                continue
            self.assertEqual(len(labels), len(polygon))
            inside = Path(polygon).contains_points(points[:, :2])
            np.testing.assert_array_equal(inside[clear], slack[clear] > 0.0)

    def test_overlap_portal_is_strictly_inside_both_cells(self) -> None:
        rng = np.random.default_rng(5)
        found, accepted = 0, 0
        for cell in valid_cells(self.planner, 150):
            offset = rng.uniform(-1.0, 1.0, 2) * (cell.guard + 0.3)
            other = self.model.make_cell(Pose2D(cell.pose.x + offset[0], cell.pose.y + offset[1],
                                                cell.pose.yaw + float(rng.uniform(-0.8, 0.8))))
            if not other.valid:
                continue
            certificate = self.model.overlap(cell, other)
            common = self.model.contains(other, cell.samples).any() or self.model.contains(cell, other.samples).any()
            if common:
                found += 1
                self.assertIsNotNone(certificate)
            if certificate is not None:
                accepted += 1
                self.assertGreater(self.model.slack(cell, certificate.portal), 0.0)
                self.assertGreater(self.model.slack(other, certificate.portal), 0.0)
        self.assertGreater(found, 30)
        self.assertGreater(self.model.stats["lp_accept"], 0)

    def test_overlap_preference_rewards_the_band(self) -> None:
        band = (0.1, 0.5)
        self.assertEqual(overlap_preference(0.3, band), 1.0)
        self.assertAlmostEqual(overlap_preference(0.05, band), 0.5)
        self.assertAlmostEqual(overlap_preference(0.75, band), 0.5)
        self.assertEqual(overlap_preference(1.0, band), 0.0)


class FrontierPlannerTest(unittest.TestCase):
    def test_factory_builds_the_frontier_planner(self) -> None:
        self.assertIsInstance(random_circles_planner(), PolyhedralFrontierPlanner)

    def test_route_is_certified_collision_free_and_deterministic(self) -> None:
        planner = random_circles_planner(stop_on_first_goal=False, max_iterations=400)
        result = planner.plan()
        again = random_circles_planner(stop_on_first_goal=False, max_iterations=400).plan()
        self.assertTrue(result.success)
        self.assertEqual(result.path_poses, again.path_poses)
        self.assertAlmostEqual(result.path_poses[-1].x, planner.goal_xy[0])
        self.assertGreaterEqual(result.bottleneck, planner.cells.safety_distance - 1e-9)
        nodes = result.tree_nodes
        for parent, child in zip(result.path_nodes, result.path_nodes[1:]):
            portal = nodes[child].certificate.portal
            self.assertGreater(planner.cells.slack(nodes[parent].cell, portal), 0.0)
            self.assertGreater(planner.cells.slack(nodes[child].cell, portal), 0.0)
        for node in nodes:
            if node.parent is not None:
                parent = nodes[node.parent]
                self.assertAlmostEqual(node.cost, parent.cost + planner.edge_cost(parent, node, node.certificate), places=9)
        self.assertGreater(result.overlap_stats["frontier_nodes"], result.overlap_stats["uniform_nodes"])
        self.assertGreater(result.overlap_stats["uniform_iterations"], 0)

    def test_live_frontier_is_exposed_boundary(self) -> None:
        planner = random_circles_planner(max_iterations=150, stop_on_first_goal=False)
        result = planner.plan()
        snapshot = planner.frontier_snapshot()
        self.assertGreater(len(snapshot["points"]), 50)
        for point, owner in zip(snapshot["points"][::7], snapshot["cell"][::7]):
            slack = planner.cells.slack(result.tree_nodes[owner].cell, Pose2D(*point))
            self.assertAlmostEqual(slack, 0.0, places=9)
            for index, node in enumerate(result.tree_nodes):
                if index != owner:
                    self.assertFalse(planner.cells.contains(node.cell, point[None, :])[0])

    def test_uniform_only_search_still_works(self) -> None:
        planner = random_circles_planner(max_iterations=1500)
        planner.frontier_config = FrontierConfig(uniform_probability=1.0)
        result = planner.plan()
        self.assertTrue(result.success)
        self.assertEqual(result.overlap_stats["frontier_iterations"], 0)

    def test_rejects_colliding_start(self) -> None:
        map_data = MapBuilder().build("single_post", SinglePostConfig(robot_radius=0.113, safety_margin=0.06))
        planner = make_tube_rrt_planner(map_data, SLOTS, Pose2D(*map_data.origin_xy, 0.0),
                                        config=TubeRRTConfig(cell_model="polyhedral"))
        result = planner.plan()
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "start is in collision")


if __name__ == "__main__":
    unittest.main()
