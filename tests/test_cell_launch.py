"""Check launch geometry, path accounting and matched sampling together."""
import unittest
from unittest import mock

import numpy as np

from formation import (FormationLibrary, MapBuilder, MapData, Pose2D, RegionTubeConfig, SinglePostConfig,
                       TubeRRTConfig, TubeRRTNode, make_tube_rrt_planner)


class CellLaunchTest(unittest.TestCase):
    def test_release_rates_cache_and_lexicographic_launch(self):
        m = MapBuilder().build('single_post', SinglePostConfig(robot_radius=.113, safety_margin=.06))
        slots = FormationLibrary.build_default(.113).get('square').slots * 2.
        p = make_tube_rrt_planner(m, slots, Pose2D(*m.start_xy, 0.),
            config=TubeRRTConfig(cell_model='polyhedral_tube'), tube_config=RegionTubeConfig(expansion='opening'))
        a, cell = p.launch_search, p.cells.make_cell(p.start)
        rows, leverage, rates = a.release_geometry(cell)
        self.assertTrue((leverage >= 0.).all())
        # Independent rigid formation motion verifies the sign and angular term.
        direction = np.array([.4, -.2, .3]); direction /= np.linalg.norm(direction)
        eps = 1e-6
        robots = np.array([r for r, _ in cell.pairs])
        moved = p.cells.robot_positions(a.pose(cell.pose, eps*direction))
        original = p.cells.robot_positions(cell.pose)
        finite = np.einsum('ij,ij->i', cell.normals, (moved-original)[robots])/eps
        np.testing.assert_allclose(rates[:len(robots)]@direction, finite, atol=1e-6)
        # Confirm one signed row is omitted, with the sibling and all guards kept.
        for k in np.flatnonzero(leverage):
            u = rows[k]/np.linalg.norm(rows[k])
            expected = u@(a._solve(cell, u, omit=k)-a._solve(cell, u))
            self.assertAlmostEqual(leverage[k], expected, places=7)
        target = Pose2D(*m.goal_xy, .4)
        entry = a.pose(cell.pose, np.array([.1, .02, 0.]))
        with mock.patch.object(p.cells, 'make_cell', side_effect=AssertionError('hypothetical cell')), \
             mock.patch.object(a, 'release_geometry', wraps=a.release_geometry):
            # Force activation to exercise BOTH lexicographic stages independently of threshold.
            a.config.threshold = -1.
            launch, _, active = a.select(cell, target, 'opening', entry)
            self.assertTrue(active)
            info = a.last_solve
            self.assertTrue(info['release_cache_hit'])
            self.assertEqual(info['support_queries'], 2)
            z = a.offset(cell.pose, launch)
            self.assertGreaterEqual(a._constraints(cell, a.offset(cell.pose, entry))['fun'](z).min(), -1e-7)
            self.assertGreaterEqual(info['facet_progress'], info['facet_alpha']-a.config.facet_epsilon-1e-7)
            # No weighted surrogate: compare against the second support optimum.
            d = a.offset(cell.pose, target); d /= np.linalg.norm(d)
            d /= a.scale**2
            facet = rows[info['selected_facet']]
            projected = a.offset(cell.pose, a.projection(cell, target, entry))
            target_z = a.offset(cell.pose, target)
            radius = np.linalg.norm(projected-target_z)+a.config.progress_epsilon
            self.assertLessEqual(np.linalg.norm(z-target_z), radius+1e-7)
            ref = a._solve(cell, d, a.offset(cell.pose, entry),
                           progress=(facet, info['facet_alpha']-a.config.facet_epsilon),
                           sample_ball=(target_z, radius))
            self.assertAlmostEqual(d@z, d@ref, places=7)
            a.config.threshold = float('inf')
            launch, _, active = a.select(cell, target, 'opening', entry)
            self.assertFalse(active)
            self.assertEqual(a.last_solve['support_queries'], 0)
            self.assertLess(p.metric(launch, a.projection(cell, target, entry)), 1e-8)

    def test_rewire_lazily_reoptimizes_descendant_portal(self):
        grid = np.zeros((80, 120), dtype=bool)
        m = MapData('attachment', .1, 12., 8., (0., 0.), grid, grid, grid.astype(float),
                    (3., 3.), (9., 4.), [], robot_radius=.08, safety_margin=.02)
        slots = np.array([[-.18, -.18], [-.18, .18], [.18, -.18], [.18, .18]])
        p = make_tube_rrt_planner(m, slots, Pose2D(*m.start_xy, 0.),
            config=TubeRRTConfig(cell_model='polyhedral_tube'), tube_config=RegionTubeConfig())
        p._reset()
        p._store(TubeRRTNode(p.start, p.cells.make_cell(p.start), None, 0.))
        def add(parent_index, pose):
            parent, cell = p._nodes[parent_index], p.cells.make_cell(pose)
            cert = p._attachment(parent.cell, cell, parent.entry_pose)
            return p._store(TubeRRTNode(pose, cell, parent_index,
                                       parent.cost+p.metric(parent.entry_pose, cert.portal), cert))
        detour = add(0, Pose2D(3., 5., 0.))
        branch = add(detour, Pose2D(5., 4.5, 0.))
        child = add(branch, Pose2D(6., 4.5, 0.))
        old_entry, old_cost = p._nodes[child].entry_pose, p._nodes[branch].cost
        cert = p._attachment(p._nodes[0].cell, p._nodes[branch].cell, p.start)
        self.assertLess(cert.route_length, old_cost)
        p._rewire_entry(branch, 0, cert)
        self.assertEqual(p._nodes[child].entry_pose, old_entry)  # Deferred, not scalar-only repair.
        self.assertNotEqual(p._nodes[child].parent_version, p._nodes[branch].version)
        p._refresh_node(child)
        node, parent = p._nodes[child], p._nodes[branch]
        self.assertGreater(p.metric(old_entry, node.entry_pose), 1e-4)
        # Independent closed-form projection onto the child's guard disk.
        np.testing.assert_allclose([parent.entry_pose.x, parent.entry_pose.y, parent.entry_pose.yaw],
                                   [3.8, 3.6, 0.], atol=1e-6)
        delta = np.array([2.2, .9])
        xy = np.array([6., 4.5])-1.5*delta/np.linalg.norm(delta)
        self.assertLess(p.metric(Pose2D(*xy, 0.), node.entry_pose), 1e-4)
        self.assertAlmostEqual(node.cost, 1.+np.linalg.norm(delta)-1.5, delta=1e-7)
        self.assertAlmostEqual(node.cost, parent.cost+p.metric(parent.entry_pose, node.entry_pose), places=7)
        self.assertEqual(node.parent_version, parent.version)
        self.assertGreater(p._stats['descendant_repairs'], 0)

    def test_launch_certificate_before_near_connect_in_a_wrapped_yaw_chart(self):
        m = MapBuilder().build('single_post', SinglePostConfig(robot_radius=.113, safety_margin=.06))
        slots = FormationLibrary.build_default(.113).get('square').slots * 2.
        for mode in ('projection', 'opening'):
            p = make_tube_rrt_planner(m, slots, Pose2D(*m.start_xy, np.pi - .05),
                config=TubeRRTConfig(cell_model='polyhedral_tube'), tube_config=RegionTubeConfig(expansion=mode))
            p._reset()
            root = TubeRRTNode(p.start, p.cells.make_cell(p.start), None, 0.)
            p._store(root)
            pose, cell, cert, _, _ = p._launch_extend(0, Pose2D(*m.goal_xy, -np.pi + .2))
            self.assertIsNotNone(cert)
            if mode == "projection":
                self.assertGreater(p.metric(root.pose, cert.launch), .01)
            self.assertGreaterEqual(p.cells.slack(root.cell, cert.launch), -1e-7)
            self.assertGreaterEqual(p.cells.slack(root.cell, cert.portal), -1e-7)
            self.assertGreater(p.cells.slack(cell, cert.portal), 0.)
            self.assertLessEqual(np.linalg.norm(p.launch_search.offset(root.pose, cert.launch)), 1.000001)
            route = [root.entry_pose, cert.launch, cert.portal]
            self.assertGreater(p.route_clearance(route), 0.)
            self.assertAlmostEqual(cert.route_length, sum(p.metric(a, b) for a, b in zip(route, route[1:])), places=7)

    def test_matched_samples_and_certified_routes_after_rewiring(self):
        m = MapBuilder().build('single_post', SinglePostConfig(robot_radius=.113, safety_margin=.06))
        slots = FormationLibrary.build_default(.113).get('square').slots * 2.
        streams = []
        for mode in ('point', 'projection', 'opening'):
            with self.subTest(mode=mode):
                p = make_tube_rrt_planner(m, slots, Pose2D(*m.start_xy, 0.),
                    config=TubeRRTConfig(cell_model='polyhedral_tube', seed=0, max_iterations=60,
                                         record_trace=True, stop_on_first_goal=False),
                    tube_config=RegionTubeConfig(expansion=mode))
                r = p.plan()
                streams.append([e.sample for e in r.trace if e.status != 'goal'])
                for node in r.tree_nodes[1:]:
                    parent, cert = r.tree_nodes[node.parent], node.certificate
                    self.assertGreaterEqual(p.cells.slack(parent.cell, cert.portal), -1e-7)
                    self.assertGreaterEqual(p.cells.slack(node.cell, cert.portal), -1e-7)
                    self.assertEqual(node.entry_pose, cert.portal)
                    route = [parent.entry_pose]
                    if cert.launch is not None:
                        self.assertGreaterEqual(p.cells.slack(parent.cell, cert.launch), -1e-7)
                        if mode != "point":
                            self.assertLessEqual(np.linalg.norm(p.launch_search.offset(parent.entry_pose, cert.launch)), 1.000001)
                        route.append(cert.launch)
                    route += [cert.portal]
                    self.assertGreater(p.route_clearance(route), 0.)
                    self.assertAlmostEqual(cert.route_length, sum(p.metric(a, b) for a, b in zip(route, route[1:])), places=7)
                    self.assertAlmostEqual(node.cost, parent.cost + p.edge_cost(parent, node, cert), places=7)
                self.assertTrue(any(n.entry_pose != n.pose for n in r.tree_nodes[1:]))
                if r.success:
                    self.assertAlmostEqual(r.path_cost, sum(p.metric(a, b) for a, b in
                                                          zip(r.path_poses, r.path_poses[1:])), places=7)
                    costs = [c for _, c in r.cost_history]
                    self.assertAlmostEqual(costs[-1], r.path_cost, places=7)
        self.assertEqual(streams[0], streams[1])
        self.assertEqual(streams[0], streams[2])


if __name__ == '__main__':
    unittest.main()
