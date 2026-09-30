import math
import unittest
from unittest import mock

import numpy as np

from formation import (
    FormationLibrary,
    MapBuilder,
    Pose2D,
    PostFenceConfig,
    RandomCirclesConfig,
    RegionTubeConfig,
    RegionTubeRRTPlanner,
    SinglePostConfig,
    TubeRRTConfig,
    make_tube_rrt_planner,
)
from formation import polyhedral_cell
from formation.types import wrap_to_pi

SLOTS = FormationLibrary.build_default(0.113).get("square").slots * 2.0
MAPS = {"random_circles": RandomCirclesConfig, "single_post": SinglePostConfig, "post_fence": PostFenceConfig}


def region_tube_planner(map_name: str = "random_circles", tube_config: RegionTubeConfig | None = None,
                        **overrides) -> RegionTubeRRTPlanner:
    extra = {"seed": 7} if map_name == "random_circles" else {}
    map_data = MapBuilder().build(map_name, MAPS[map_name](robot_radius=0.113, safety_margin=0.06, **extra))
    config = TubeRRTConfig(cell_model="polyhedral_tube", seed=7, **overrides)
    return make_tube_rrt_planner(map_data, SLOTS, Pose2D(*map_data.start_xy, 0.0), config=config,
                                 tube_config=tube_config)


def random_poses(planner: RegionTubeRRTPlanner, count: int, seed: int) -> list[Pose2D]:
    rng = np.random.default_rng(seed)
    ox, oy = planner.map_data.origin_xy
    return [Pose2D(float(rng.uniform(ox, ox + planner.map_data.width_m)),
                   float(rng.uniform(oy, oy + planner.map_data.height_m)), float(rng.uniform(-math.pi, math.pi)))
            for _ in range(count)]


def valid_cells(planner: RegionTubeRRTPlanner, count: int, seed: int = 0) -> list:
    cells = []
    for pose in random_poses(planner, 20 * count, seed):
        cell = planner.cells.make_cell(pose)
        if cell.valid:
            cells.append(cell)
            if len(cells) == count:
                break
    return cells


class RegionTubeGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.planner = region_tube_planner(max_iterations=400, stop_on_first_goal=False)
        cls.result = cls.planner.plan()
        cls.model = cls.planner.cells

    def test_analytic_incircle_matches_socp(self) -> None:
        cells = valid_cells(self.planner, 150, seed=3)
        self.assertTrue(any(len(cell.offsets) >= 2 for cell in cells))
        for cell in cells:
            _, analytic = self.model._inscribed_circle(cell)
            with mock.patch.object(polyhedral_cell, "_ANALYTIC_INCIRCLE_ROWS", -1):
                _, reference = self.model._inscribed_circle(cell)
            self.assertGreaterEqual(analytic, reference - 1e-7)
            self.assertLessEqual(analytic, reference + 1e-6)

    def test_lazy_cells_are_the_unpruned_cells_and_skip_the_geometry(self) -> None:
        eager = region_tube_planner(tube_config=RegionTubeConfig(lazy_geometry=False)).cells
        rng = np.random.default_rng(2)
        compared = 0
        for pose in random_poses(self.planner, 600, seed=4):
            lazy_cell, eager_cell = self.model.make_cell(pose), eager.make_cell(pose)
            self.assertEqual(lazy_cell.valid, eager_cell.valid)
            if not lazy_cell.valid:
                continue
            self.assertIsNone(lazy_cell._inradius)
            self.assertIsNone(lazy_cell._samples)
            self.assertGreaterEqual(lazy_cell.yaw_limit, eager_cell.yaw_limit - 1e-12)
            points = np.column_stack((pose.x + rng.uniform(-1.6, 1.6, 300), pose.y + rng.uniform(-1.6, 1.6, 300),
                                      pose.yaw + rng.uniform(-1.6, 1.6, 300)))
            lazy_slack, eager_slack = self.model.slack_many(lazy_cell, points), eager.slack_many(eager_cell, points)
            self.assertFalse(np.any((eager_slack > 1e-9) & (lazy_slack <= 0.0)))
            self.assertLess(np.mean((lazy_slack > 0.0) != (eager_slack > 0.0)), 0.01)
            self.assertAlmostEqual(lazy_cell.inradius, eager_cell.inradius, delta=0.02)
            compared += 1
        self.assertGreater(compared, 100)

    def test_approximate_center_is_inside_the_section(self) -> None:
        for cell in valid_cells(self.planner, 200, seed=8):
            point = self.model.approximate_center(cell)
            pose = Pose2D(cell.pose.x + point[0], cell.pose.y + point[1], cell.pose.yaw)
            self.assertGreater(self.model.slack(cell, pose), 0.0)

    def test_vectorised_extents_and_slack_match_the_cell_model(self) -> None:
        planner, rng = self.planner, np.random.default_rng(1)
        nodes = np.arange(len(planner._nodes))
        target = Pose2D(0.3, -0.4, 0.7)
        u, length = planner._directions(nodes, target)
        self.assertTrue(np.allclose(np.hypot(u[:, 0], u[:, 1]) + planner.formation_radius * np.abs(u[:, 2]), 1.0))
        extents = planner._node_extents(nodes, u)
        count = len(nodes)
        points = np.column_stack((planner._x[:count] + rng.normal(0, 0.3, count),
                                  planner._y[:count] + rng.normal(0, 0.3, count),
                                  planner._yaw[:count] + rng.normal(0, 0.4, count)))
        slack = planner._node_slack(nodes, points)
        for i in rng.choice(nodes, size=min(80, len(nodes)), replace=False):
            cell = planner._nodes[i].cell
            self.assertAlmostEqual(extents[i], self.model.ray_extent(cell, u[i, :2], u[i, 2]), places=9)
            self.assertAlmostEqual(slack[i], self.model.slack(cell, Pose2D(*points[i])), places=9)
            other = planner._cell_extents(cell, -u[i:i + 1])[0]
            self.assertAlmostEqual(other, self.model.ray_extent(cell, -u[i, :2], -u[i, 2]), places=9)

    def test_gap_nearest_matches_brute_force(self) -> None:
        planner = self.planner
        for pose in random_poses(planner, 40, seed=5):
            cell = self.model.make_cell(pose)
            if not cell.valid:
                continue
            best, _, length, forward = planner._nearest(pose, cell)
            gaps = []
            for i, node in enumerate(planner._nodes):
                v = (pose.x - node.pose.x, pose.y - node.pose.y, wrap_to_pi(pose.yaw - node.pose.yaw))
                d = math.hypot(v[0], v[1]) + planner.formation_radius * abs(v[2])
                if i in planner._goal_nodes or d < planner.config.min_metric_step:
                    gaps.append(math.inf)
                    continue
                forward_i = self.model.ray_extent(node.cell, (v[0] / d, v[1] / d), v[2] / d)
                backward = self.model.ray_extent(cell, (-v[0] / d, -v[1] / d), -v[2] / d)
                gaps.append(d - forward_i - backward)
            self.assertAlmostEqual(gaps[best], min(gaps), places=9)

    def test_every_tree_edge_portal_is_strictly_inside_both_cells(self) -> None:
        self.assertTrue(self.result.success)
        kinds = set()
        for index, node in enumerate(self.result.tree_nodes):
            if node.parent is None:
                continue
            parent = self.result.tree_nodes[node.parent]
            portal = node.certificate.portal
            slack = min(self.model.slack(parent.cell, portal), self.model.slack(node.cell, portal))
            self.assertGreater(slack, 0.0)
            self.assertLessEqual(node.certificate.slack, slack + 1e-9)
            kinds.add(self.planner.edge_kind[index])
        self.assertLessEqual(kinds, {"steer", "line", "center", "exact", "goal"})
        self.assertIn("steer", kinds)
        self.assertGreater(self.result.bottleneck, 0.0)

    def test_line_witness_route_length_is_the_seed_distance(self) -> None:
        planner = self.planner
        nodes = np.arange(1, min(len(planner._nodes), 60))
        cell = planner._nodes[0].cell
        witnesses, lengths = planner._witnesses(nodes, cell)
        for k, witness in enumerate(witnesses):
            if witness is None:
                continue
            self.assertAlmostEqual(witness.route_length, lengths[k], places=12)
            self.assertGreater(self.model.slack(cell, witness.portal), 0.0)
            self.assertGreater(self.model.slack(planner._nodes[nodes[k]].cell, witness.portal), 0.0)


class RegionTubePlannerTest(unittest.TestCase):
    def test_factory_dispatches_and_rejects_bad_options(self) -> None:
        planner = region_tube_planner(max_iterations=10)
        self.assertIsInstance(planner, RegionTubeRRTPlanner)
        with self.assertRaises(ValueError):
            RegionTubeConfig(nearest="other")
        with self.assertRaises(ValueError):
            RegionTubeConfig(colliding="other")
        with self.assertRaises(ValueError):
            RegionTubeConfig(max_step=0.0)

    def test_first_solution_on_all_maps(self) -> None:
        for name in MAPS:
            with self.subTest(map=name):
                result = region_tube_planner(name, max_iterations=2500, stop_on_first_goal=True).plan()
                self.assertTrue(result.success)
                self.assertGreater(result.bottleneck, 0.0)
                self.assertGreater(result.overlap_stats["samples_colliding"], 0)

    def test_without_exact_overlap_no_lp_is_solved(self) -> None:
        result = region_tube_planner(tube_config=RegionTubeConfig(exact_overlap=False), max_iterations=600,
                                     stop_on_first_goal=False).plan()
        self.assertTrue(result.success)
        self.assertEqual(result.overlap_stats["lp_calls"], 0)
        self.assertEqual(result.overlap_stats["socp_calls"], 0)
        self.assertEqual(result.overlap_stats["near_exact_calls"] + result.overlap_stats["rewire_exact_calls"], 0)

    def test_point_nearest_and_extend_variants_plan(self) -> None:
        for tube_config in (RegionTubeConfig(nearest="point"), RegionTubeConfig(colliding="extend")):
            with self.subTest(nearest=tube_config.nearest, colliding=tube_config.colliding):
                planner = region_tube_planner(tube_config=tube_config, max_iterations=2500, stop_on_first_goal=True)
                result = planner.plan()
                self.assertTrue(result.success)
                if tube_config.nearest == "point":
                    self.assertEqual(result.overlap_stats["nearest_differs"], 0)

    def test_anytime_cost_is_monotone_and_matches_the_path(self) -> None:
        planner = region_tube_planner(max_iterations=800, stop_on_first_goal=False)
        result = planner.plan()
        self.assertTrue(result.success)
        costs = [cost for _, cost in result.cost_history]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(costs, costs[1:])))
        self.assertAlmostEqual(costs[-1], result.path_cost, places=12)
        length = sum(result.tree_nodes[i].certificate.route_length for i in result.path_nodes[1:])
        self.assertAlmostEqual(length, result.path_cost, places=9)


if __name__ == "__main__":
    unittest.main()
