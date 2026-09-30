import math
import unittest
from dataclasses import replace

import numpy as np
from matplotlib.path import Path

from formation import (
    FormationLibrary,
    FrontierConfig,
    MapBuilder,
    PolyhedralFrontierPlanner,
    Pose2D,
    PostFenceConfig,
    RandomCirclesConfig,
    SinglePostConfig,
    TubeRRTConfig,
    make_tube_rrt_planner,
)
from formation.tube_rrt_frontier import FRONTIER_GUARD
from formation.types import wrap_to_pi

# Square formation with 1 m between neighbouring robots (slots at (+-0.5, +-0.5)).
SLOTS = FormationLibrary.build_default(0.113).get("square").slots * 2.0


def random_circles_planner(frontier_config: FrontierConfig | None = None, **overrides) -> PolyhedralFrontierPlanner:
    map_data = MapBuilder().build("random_circles", RandomCirclesConfig(seed=7, robot_radius=0.113, safety_margin=0.06))
    config = TubeRRTConfig(cell_model="polyhedral", seed=7, **overrides)
    return make_tube_rrt_planner(map_data, SLOTS, Pose2D(*map_data.start_xy, 0.0), config=config,
                                 frontier_config=frontier_config)


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


def box_points(rng: np.random.Generator, cell, count: int, yaw_half_width: float | None = None) -> np.ndarray:
    g = cell.guard
    limit = cell.yaw_limit if yaw_half_width is None else yaw_half_width
    return np.column_stack((cell.pose.x + rng.uniform(-g, g, count), cell.pose.y + rng.uniform(-g, g, count),
                            cell.pose.yaw + rng.uniform(-limit, limit, count)))


class PolyhedralCellTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = random_circles_planner()
        self.model = self.planner.cells

    def test_common_rotation_radius_is_formation_radius(self) -> None:
        self.assertAlmostEqual(self.model.rho, math.sqrt(0.5), places=12)
        self.assertTrue(np.allclose(np.linalg.norm(SLOTS, axis=1), self.model.rho))

    def test_rows_come_from_broadphase_queries_and_guard_from_the_rest(self) -> None:
        d_s = self.model.safety_distance
        for cell in valid_cells(self.planner, 40):
            clearance, normals = self.model.proximity(self.model.robot_positions(cell.pose))
            self.assertEqual(cell.broadphase_pairs, int(np.count_nonzero(clearance < self.model.active_range)))
            self.assertLessEqual(cell.active_count, cell.broadphase_pairs)
            for row, (robot, component) in enumerate(cell.pairs):
                self.assertLess(clearance[robot, component], self.model.active_range)
                self.assertTrue(np.allclose(cell.normals[row], normals[robot, component]))
                self.assertLessEqual(cell.offsets[row], clearance[robot, component] - d_s + 1e-12)
            rest = clearance[clearance >= self.model.active_range]
            self.assertAlmostEqual(cell.guard, min(float(rest.min()) - d_s, self.model.max_extent))
            self.assertAlmostEqual(cell.clearance, float(clearance.min()))

    def test_pruned_rows_are_conservative_and_nearly_exact(self) -> None:
        rng = np.random.default_rng(6)
        d_s = self.model.safety_distance
        pruned_total, mismatch = 0, 0
        for cell in valid_cells(self.planner, 60):
            clearance, normals = self.model.proximity(self.model.robot_positions(cell.pose))
            robots, components = np.nonzero(clearance < self.model.active_range)
            full = replace(cell, normals=normals[robots, components].reshape((-1, 2)),
                           offsets=clearance[robots, components] - d_s, yaw_limit=self.model.yaw_chart)
            pruned_total += full.active_count - cell.active_count
            points = box_points(rng, cell, 4000, self.model.yaw_chart)
            inside, inside_full = self.model.contains(cell, points), self.model.contains(full, points)
            self.assertFalse(np.any(inside & ~inside_full))
            mismatch += int(np.count_nonzero(inside_full & ~inside))
        self.assertGreater(pruned_total, 0)
        self.assertLess(mismatch, 0.01 * 60 * 4000)

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

    def test_yaw_interval_is_decided_by_the_constraints(self) -> None:
        rng = np.random.default_rng(7)
        shrunk = 0
        for cell in valid_cells(self.planner, 60):
            self.assertAlmostEqual(cell.yaw_limit, min(self.model.yaw_chart, cell.inradius / self.model.rho))
            self.assertGreaterEqual(cell.inradius, cell.radius - 1e-9)
            center = cell.center_offset + (cell.pose.x, cell.pose.y)
            inner = Pose2D(center[0], center[1], cell.pose.yaw + 0.98 * cell.yaw_limit)
            self.assertGreater(self.model.slack(cell, inner), 0.0)
            if cell.yaw_limit < self.model.yaw_chart - 1e-6:
                shrunk += 1
                points = box_points(rng, cell, 3000)
                points[:, 2] = cell.pose.yaw + 1.02 * cell.yaw_limit
                self.assertFalse(np.any(self.model.contains(cell, points)))
        self.assertGreater(shrunk, 20)

    def test_translational_extent_reaches_the_boundary(self) -> None:
        rng = np.random.default_rng(2)
        guard_bound = 0
        for cell in valid_cells(self.planner, 30):
            angles = rng.uniform(-math.pi, math.pi, 16)
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
            dtheta = float(rng.uniform(-0.9, 0.9)) * cell.yaw_reach
            lengths, guard_edges = self.model.translational_extent(cell, directions, dtheta)
            guard_bound += int(guard_edges.sum())
            self.assertTrue(np.all(lengths > 0.0))
            for u, length in zip(directions, lengths):
                at = lambda t: Pose2D(cell.pose.x + t * u[0], cell.pose.y + t * u[1], cell.pose.yaw + dtheta)
                self.assertGreater(self.model.slack(cell, at(0.999 * length)), 0.0)
                self.assertLess(self.model.slack(cell, at(1.001 * length)), 0.0)
                self.assertAlmostEqual(self.model.slack(cell, at(length)), 0.0, places=9)
        self.assertGreater(guard_bound, 0)

    def test_ray_extent_stays_inside(self) -> None:
        rng = np.random.default_rng(3)
        for cell in valid_cells(self.planner, 30):
            dc, dphi = rng.normal(size=2), float(rng.uniform(-1.0, 1.0))
            t = self.model.ray_extent(cell, dc, dphi)
            at = lambda s: Pose2D(cell.pose.x + s * dc[0], cell.pose.y + s * dc[1], cell.pose.yaw + s * dphi)
            self.assertGreater(self.model.slack(cell, at(0.999 * t)), 0.0)
            self.assertLess(self.model.slack(cell, at(1.001 * t)), 1e-12)

    def test_boundary_radii_trace_the_3d_surface(self) -> None:
        for cell in valid_cells(self.planner, 20):
            angles = np.linspace(-math.pi, math.pi, 12, endpoint=False)
            dthetas = np.linspace(-0.9, 0.9, 5) * cell.yaw_limit
            radii = self.model.boundary_radii(cell, angles, dthetas)
            center = cell.center_offset + (cell.pose.x, cell.pose.y)
            for dtheta, row in zip(dthetas, radii):
                for angle, radius in zip(angles, row):
                    at = lambda s: Pose2D(center[0] + s * math.cos(angle), center[1] + s * math.sin(angle),
                                          cell.pose.yaw + dtheta)
                    self.assertGreater(self.model.slack(cell, at(0.999 * radius)), 0.0)
                    self.assertLess(self.model.slack(cell, at(1.001 * radius)), 0.0)

    def test_slice_polygon_matches_the_fixed_yaw_section(self) -> None:
        rng = np.random.default_rng(4)
        for cell in valid_cells(self.planner, 20):
            dtheta = float(rng.uniform(-0.95, 0.95)) * cell.yaw_limit
            polygon, labels = self.model.slice_polygon(cell, dtheta)
            points = box_points(rng, cell, 4000)
            points[:, 2] = cell.pose.yaw + dtheta
            slack = self.model.slack_many(cell, points)
            self.assertEqual(len(labels), len(polygon))
            inside = Path(polygon).contains_points(points[:, :2])
            self.assertFalse(np.any(inside & (slack < -1e-9)))
            self.assertFalse(np.any(~inside & (slack > 0.01 * cell.guard)))

    def test_overlap_portal_is_strictly_inside_both_cells(self) -> None:
        rng = np.random.default_rng(5)
        found = 0
        for cell in valid_cells(self.planner, 200):
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
                self.assertGreater(self.model.slack(cell, certificate.portal), 0.0)
                self.assertGreater(self.model.slack(other, certificate.portal), 0.0)
        self.assertGreater(found, 30)
        stats = self.model.stats
        self.assertGreater(stats["lp_calls"], 0)
        self.assertGreater(stats["lp_accept"] + stats["socp_accept"], 0)


class FrontierPlannerTest(unittest.TestCase):
    def test_factory_builds_the_frontier_planner(self) -> None:
        self.assertIsInstance(random_circles_planner(), PolyhedralFrontierPlanner)

    def test_region_probability_schedules(self) -> None:
        switch = FrontierConfig()
        self.assertEqual((switch.region_probability_at(10, False), switch.region_probability_at(10, True)), (0.5, 0.2))
        constant = FrontierConfig(region_schedule="constant", region_probability=0.6)
        self.assertEqual(constant.region_probability_at(1000, True), 0.6)
        decay = FrontierConfig(region_schedule="exp", region_min=0.2, region_max=0.6, region_decay=0.01)
        self.assertAlmostEqual(decay.region_probability_at(0, False), 0.6)
        self.assertAlmostEqual(decay.region_probability_at(100, True), 0.2 + 0.4 * math.exp(-1.0))
        with self.assertRaises(ValueError):
            FrontierConfig(region_schedule="linear")

    def test_one_cell_per_iteration_and_certified_route(self) -> None:
        planner = random_circles_planner(stop_on_first_goal=False, max_iterations=400)
        result = planner.plan()
        again = random_circles_planner(stop_on_first_goal=False, max_iterations=400).plan()
        stats = result.overlap_stats
        self.assertTrue(result.success)
        self.assertEqual(result.path_poses, again.path_poses)
        self.assertEqual(stats["cells_built"], 1 + result.iterations - stats["rejected_no_progress"])
        self.assertEqual(stats["pose_queries"], stats["cells_built"])
        self.assertGreater(stats["region_iterations"], 0)
        self.assertGreater(stats["uniform_iterations"], stats["region_iterations"])
        self.assertLessEqual(stats["first_goal_pose_queries"], stats["pose_queries"])
        self.assertGreater(stats["first_goal_time_s"], 0.0)
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

    def test_only_guard_and_yaw_cap_boundaries_are_expanded(self) -> None:
        planner = random_circles_planner(max_iterations=300, stop_on_first_goal=False)
        result = planner.plan()
        store, f = planner.frontier, planner.frontier_config
        kinds = set()
        for index in range(store.size):
            cell = result.tree_nodes[store.cell[index]].cell
            point = store.points[index]
            kinds.add(int(store.kind[index]))
            local = point - (cell.pose.x, cell.pose.y, 0.0)
            turn = cell.rho * abs(wrap_to_pi(point[2] - cell.pose.yaw))
            if store.kind[index] == FRONTIER_GUARD:
                self.assertAlmostEqual(math.hypot(local[0], local[1]) + turn, cell.guard, places=9)
            else:
                self.assertAlmostEqual(turn, cell.rho * cell.yaw_limit, places=9)
            if len(cell.offsets):
                self.assertGreater(float(planner.cells.directional_slack(cell, point, outer=False)[0]), 0.0)
            self.assertGreaterEqual(float(planner.cells.directional_slack(cell, point)[0]), f.min_sample_offset)
        self.assertIn(FRONTIER_GUARD, kinds)
        self.assertGreater(result.overlap_stats["frontier_obstacle_limited"], store.size)

    def test_region_seeds_leave_the_certificate_but_not_into_known_obstacles(self) -> None:
        planner = random_circles_planner(FrontierConfig(region_schedule="constant", region_probability=0.8),
                                         max_iterations=400, stop_on_first_goal=False)
        result = planner.plan()
        region = [e for e in planner.expansion_log if e["mode"] == "region"]
        self.assertGreater(len(region), 30)
        for entry in region:
            source = result.tree_nodes[entry["source"]].cell
            node = result.tree_nodes[entry["node"]]
            q = np.array((node.pose.x, node.pose.y, node.pose.yaw))
            self.assertFalse(planner.cells.contains(source, q)[0])
            self.assertGreater(float(planner.cells.directional_slack(source, q)[0]), 0.0)
            self.assertLessEqual(planner.metric(Pose2D(*entry["frontier_point"]), node.pose),
                                 planner.frontier_config.sample_offset + 1e-9)
            self.assertTrue(node.cell.valid)
        stats = result.overlap_stats
        self.assertEqual(stats["region_reject_obstacle"], 0)
        self.assertEqual(stats["rejected_collision"], 0)

    def test_extreme_region_probabilities(self) -> None:
        uniform = random_circles_planner(FrontierConfig(region_schedule="constant", region_probability=0.0),
                                         max_iterations=1500).plan()
        self.assertTrue(uniform.success)
        self.assertEqual(uniform.overlap_stats["region_iterations"], 0)
        region = random_circles_planner(FrontierConfig(region_schedule="constant", region_probability=1.0),
                                        max_iterations=1500).plan()
        self.assertTrue(region.success)
        self.assertEqual(region.overlap_stats["uniform_iterations"], region.overlap_stats["region_fallback"])

    def test_post_fence_route_lets_a_post_pass_between_robots(self) -> None:
        map_data = MapBuilder().build("post_fence", PostFenceConfig(robot_radius=0.113, safety_margin=0.06))
        planner = make_tube_rrt_planner(map_data, SLOTS, Pose2D(*map_data.start_xy, 0.0),
                                        config=TubeRRTConfig(cell_model="polyhedral", seed=7))
        result = planner.plan()
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.bottleneck, planner.cells.safety_distance - 1e-9)

    def test_rejects_colliding_start(self) -> None:
        map_data = MapBuilder().build("single_post", SinglePostConfig(robot_radius=0.113, safety_margin=0.06))
        planner = make_tube_rrt_planner(map_data, SLOTS, Pose2D(*map_data.origin_xy, 0.0),
                                        config=TubeRRTConfig(cell_model="polyhedral"))
        result = planner.plan()
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "start is in collision")


if __name__ == "__main__":
    unittest.main()
