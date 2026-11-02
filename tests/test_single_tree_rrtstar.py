import math

from formation.single_tree_rrtstar import SingleTreeConfig, SingleTreeRRTStar
from formation.types import Pose2D
from test_region_coupling import SLOTS, _map


def make_planner(**kwargs):
    p = SingleTreeRRTStar(_map([]), SLOTS, Pose2D(-4., 0., 0.), SingleTreeConfig(**kwargs))
    p.initialize()
    return p


def add_state(p, pose, owner, parent=0):
    node = len(p.anchor)
    p.anchor.append(pose)
    p.owner.append(owner)
    p.parent.append(parent)
    p.children.append(set())
    p.children[parent].add(node)
    p.edge_cost.append(p.geo.distance(p.anchor[parent], pose))
    p.cost.append(p.cost[parent] + p.edge_cost[-1])
    p.portal.append(None)
    p.owner_nodes[owner].append(node)
    return node


def test_covered_low_slack_state_is_kept_and_exact_duplicate_reconsidered():
    p = make_planner(seed=2)
    cell = p.cells[0]
    pose = Pose2D(cell.pose.x + cell.guard - 0.001, cell.pose.y, 0.)
    assert 0. < p.geo.slack(cell, pose) < p.config.overlap_margin
    p._sample = lambda: pose
    p.step()
    assert len(p.cells) == 1 and p.anchor[-1] == pose
    assert p.owner[-1] == 0
    n = len(p.anchor)
    p.step()
    assert len(p.anchor) == n and p.stats["exact_reconsider"] == 1


def test_all_covering_owners_are_returned_and_selected_fairly():
    p = make_planner(seed=7)
    second = p.model.make_cell(Pose2D(-3.5, 0., 0.))
    i = p._add_cell(second, 0)
    p.owner_nodes.append([])
    add_state(p, second.pose, i)
    sample = Pose2D(-3.75, 0., 0.)
    assert p._covering(sample) == [0, 1]
    p._sample = lambda: sample
    for _ in range(30):
        p.step()
    assert p.owner.count(0) == 2 and p.owner.count(1) == 2
    assert len(p.cells) == 2


def test_cross_owner_endpoint_fast_path_and_chart_seam_cost():
    p = make_planner()
    # Root state stays reachable; two extra certificates only exercise edges.
    for yaw in (0., 2.):
        cell = p.model.make_cell(Pose2D(0., 0., yaw))
        p._add_cell(cell, 0)
        p.owner_nodes.append([])
    a = add_state(p, Pose2D(0., 0., -.8), 1)
    b = add_state(p, Pose2D(0., 0., 2.8), 2)
    edge = p._edge(a, b)
    assert edge is not None
    assert math.isclose(edge[0], 3.6*p.rho, abs_tol=1e-6)
    assert edge[0] > p.geo.distance(p.anchor[a], p.anchor[b])
    endpoint = add_state(p, Pose2D(0., 0., 1.), 1)
    edge = p._edge(endpoint, b)
    assert edge[1] == p.anchor[endpoint]
    assert p.stats["endpoint_edges"] > 0


def test_fixed_anchors_certificates_costs_cycles_and_monotone_tree_goal():
    obstacles = [dict(type="circle", center_xy=(0., y), radius=.3) for y in (-2.4, -1.2, 1.2, 2.4)]
    p = SingleTreeRRTStar(_map(obstacles), SLOTS, Pose2D(-4., 0., 0.), SingleTreeConfig(seed=5))
    p.initialize()
    saved, best = [], math.inf
    for iteration in range(500):
        p.step()
        assert p.anchor[:len(saved)] == saved
        saved = list(p.anchor)
        if iteration % 25 == 0:
            current = p.current_path()
            if current:
                assert current[0] <= best + 1e-8
                best = current[0]
    assert p.stats["rewires"] > 0 and p.stats["covered_insertions"] > 0
    assert len(p.anchor) > len(p.cells)
    assert math.isfinite(best)
    for v, parent in enumerate(p.parent):
        assert p.geo.slack(p.cells[p.owner[v]], p.anchor[v]) >= -1e-7
        assert len(p._ancestors(v)) <= len(p.parent)
        if parent is None:
            assert v == 0
            continue
        assert v in p.children[parent]
        assert math.isclose(p.cost[v], p.cost[parent]+p.edge_cost[v], abs_tol=1e-9)
        if p.owner[v] != p.owner[parent]:
            assert p._portal_valid(p.portal[v], p.owner[parent], p.owner[v])
        else:
            assert p.portal[v] is None
    from scripts.compare_coupling import route_clearance
    route = [(q.x, q.y, q.yaw) for q in p.current_path()[1]]
    assert route_clearance(p.map_data, p.slots, route) >= p.config.safety_distance - 1e-6


def test_owner_fallback_outside_radius_still_inserts():
    p = make_planner(neighbor_radius=0.01)
    pose = Pose2D(-3., 0., 0.)
    p._sample = lambda: pose
    p.step()
    assert p.parent[1] == 0 and p.anchor[1] == pose
    assert math.isclose(p.cost[1], 1.)
