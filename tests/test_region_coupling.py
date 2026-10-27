import math

import numpy as np

from formation.region_coupling import (AnchorRRTStar, CouplingConfig, PortalRRTStar, ProxyRRTStar,
                                       SingleEntryRRTStar)
from formation.types import MapData, Pose2D

SLOTS = np.array([[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]])


def _map(obstacles):
    grid = np.zeros((80, 120), dtype=bool)
    return MapData("test", 0.1, 12.0, 8.0, (-6.0, -4.0), grid, grid, grid.astype(float), (-4.0, 0.0), (4.0, 0.0),
                   obstacles, robot_radius=0.1, safety_margin=0.05)


def test_chain_in_free_space_is_the_straight_line():
    planner = ProxyRRTStar(_map([]), SLOTS, Pose2D(-4.0, 0.0, 0.0), CouplingConfig())
    cells = [planner.model.make_cell(Pose2D(x, 0.0, 0.0)) for x in np.arange(-4.0, 4.01, 1.0)]
    length, route = planner.geo.chain(planner.start, planner.goal_xy, cells, 0.02)
    assert math.isclose(length, 8.0, rel_tol=1e-5)
    assert all(abs(p.y) < 1e-5 and abs(p.yaw) < 1e-5 for p in route)


def test_attach_is_the_projection_onto_the_intersection():
    planner = ProxyRRTStar(_map([]), SLOTS, Pose2D(-4.0, 0.0, 0.0), CouplingConfig())
    first, second = (planner.model.make_cell(Pose2D(x, 0.0, 0.0)) for x in (0.0, 1.5))
    inside = Pose2D(0.75, 0.0, 0.0)
    assert planner.geo.attach(inside, first, second, 0.02) == inside
    portal = planner.geo.attach(Pose2D(-0.5, 0.0, 0.0), first, second, 0.02)
    assert min(planner.geo.slack(first, portal), planner.geo.slack(second, portal)) >= 0.02 - 1e-6
    assert abs(portal.y) < 1e-6 and abs(portal.yaw) < 1e-6
    assert math.isclose(portal.x, 1.5 - second.guard + 0.02, abs_tol=1e-5)


def test_disjoint_cells_have_no_attachment_and_negative_margin():
    planner = ProxyRRTStar(_map([]), SLOTS, Pose2D(-4.0, 0.0, 0.0), CouplingConfig())
    first, second = (planner.model.make_cell(Pose2D(x, 0.0, 0.0)) for x in (-2.0, 2.0))
    assert planner.geo.attach(Pose2D(0.0, 0.0, 0.0), first, second, 0.02) is None
    t, _ = planner.geo.max_margin(first, second)
    assert math.isclose(t, (4.0 - first.guard - second.guard) / -2.0, abs_tol=1e-5)
    assert planner.geo.fallbacks == 0


def test_both_couplings_build_the_same_regions_and_valid_chains():
    obstacles = [dict(type="circle", center_xy=(0.0, y), radius=0.3) for y in (-2.4, -1.2, 1.2, 2.4)]
    config = CouplingConfig(max_iterations=150, checkpoints=(150,), seed=3)
    results = {}
    for cls in (SingleEntryRRTStar, ProxyRRTStar):
        planner = cls(_map(obstacles), SLOTS, Pose2D(-4.0, 0.0, 0.0), config)
        results[cls.variant] = (planner.plan(), [c.pose for c in planner.cells])
    (a, cells_a), (b, cells_b) = results.values()
    assert cells_a == cells_b
    for result in (a, b):
        final = result["checkpoints"][-1]
        assert final["success"]
        assert final["optimised_cost"] <= final["raw_cost"] + 1e-6
        assert final["optimised_cost"] >= 8.0 - 1e-6


def test_proxy_costs_follow_the_parent_chain_after_rewiring():
    obstacles = [dict(type="circle", center_xy=(0.0, y), radius=0.3) for y in (-2.4, -1.2, 1.2, 2.4)]
    planner = ProxyRRTStar(_map(obstacles), SLOTS, Pose2D(-4.0, 0.0, 0.0),
                           CouplingConfig(max_iterations=400, checkpoints=(400,), seed=5))
    planner.plan()
    assert planner.stats["rewires"] > 0
    for i, p in enumerate(planner.parent):
        expected = 0.0 if p is None else planner.cost[p] + planner._proxy(p, i)
        assert math.isclose(planner.cost[i], expected, abs_tol=1e-9)
        assert p is None or i in planner.children[p]


def test_portal_costs_are_feasible_polyline_lengths_after_rewiring():
    obstacles = [dict(type="circle", center_xy=(0.0, y), radius=0.3) for y in (-2.4, -1.2, 1.2, 2.4)]
    planner = PortalRRTStar(_map(obstacles), SLOTS, Pose2D(-4.0, 0.0, 0.0),
                            CouplingConfig(max_iterations=400, checkpoints=(400,), seed=5))
    planner.plan()
    assert planner.stats["rewires"] > 0
    margin = planner.config.overlap_margin
    for i, p in enumerate(planner.parent):
        if p is None:
            continue
        w = planner.entry[i]
        assert math.isclose(planner.cost[i], planner.cost[p] + planner.geo.distance(planner.entry[p], w), abs_tol=1e-9)
        assert min(planner.geo.slack(planner.cells[p], w), planner.geo.slack(planner.cells[i], w)) >= margin - 1e-6


def test_anchor_costs_are_certified_seed_to_seed_lengths():
    obstacles = [dict(type="circle", center_xy=(0.0, y), radius=0.3) for y in (-2.4, -1.2, 1.2, 2.4)]
    planner = AnchorRRTStar(_map(obstacles), SLOTS, Pose2D(-4.0, 0.0, 0.0),
                            CouplingConfig(max_iterations=400, checkpoints=(400,), seed=5))
    planner.plan()
    assert planner.stats["rewires"] > 0 and planner.stats["edges_direct"] > 0
    for i, p in enumerate(planner.parent):
        assert planner.geo.slack(planner.cells[i], planner.cells[i].pose) >= 0.0
        if p is None:
            continue
        edge = planner.cost[i] - planner.cost[p]
        a, b = planner.cells[p].pose, planner.cells[i].pose
        assert edge >= planner.geo.distance(a, b) - 1e-9
        if math.isclose(edge, planner.geo.distance(a, b), abs_tol=1e-9):
            za, zb = planner.geo.lift(a, a.yaw), planner.geo.lift(b, a.yaw)
            both = []
            for t in np.linspace(0.0, 1.0, 201):
                pose = planner.geo.pose(za + t * (zb - za))
                sp, si = planner.geo.slack(planner.cells[p], pose), planner.geo.slack(planner.cells[i], pose)
                assert max(sp, si) >= -1e-9
                both.append(min(sp, si))
            assert max(both) >= 0.0
