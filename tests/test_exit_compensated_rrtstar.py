import math

from formation.exit_compensated_rrtstar import ExitCompensatedConfig, ExitCompensatedRRTStar
from formation.types import Pose2D
from test_region_coupling import SLOTS, _map


def make_planner(**kwargs):
    p = ExitCompensatedRRTStar(_map([]), SLOTS, Pose2D(0., 0., 0.), ExitCompensatedConfig(**kwargs))
    p.initialize()
    return p


def append_region(p, entry, parent):
    i = p._add_cell(p.model.make_cell(Pose2D(0., 0., 0.)), parent)
    p.parent.append(parent)
    p.entry.append(entry)
    p.cost.append(p.cost[parent]+p._distance_in(parent, p.entry[parent], entry))
    p.children.append(set())
    p.children[parent].add(i)
    p.version.append(0)
    p.parent_version.append(p.version[parent])
    p.geometry_version.append(0)
    p.exit_version.append(0)
    p.local_signature.append(None)
    return i


def assert_tree(p):
    for i, parent in enumerate(p.parent):
        seen, cursor = set(), i
        while cursor is not None:
            assert cursor not in seen
            seen.add(cursor)
            cursor = p.parent[cursor]
        assert p.children[i] == {j for j, q in enumerate(p.parent) if q == i}
        if parent is not None:
            assert p._portal_valid(parent, i, p.entry[i])
            assert math.isclose(p.cost[i], p.cost[parent]+p._distance_in(parent, p.entry[parent], p.entry[i]), abs_tol=1e-9)


def test_compensation_accepts_higher_entry_cost_and_freezes_all_suffixes():
    p = make_planner(enable_local=False)
    chain, parent = [], 0
    for x in (-.2, .2, -.2, .2, -.2, 0.):
        parent = append_region(p, Pose2D(x, 0., 0.), parent)
        chain.append(parent)
    node = chain[-1]
    via = append_region(p, Pose2D(.2, .35, 0.), chain[3])
    new_parent = append_region(p, Pose2D(.2, 0., 0.), via)
    a = append_region(p, Pose2D(.4, 0., 0.), node)
    b = append_region(p, Pose2D(.6, 0., 0.), node)
    tail = append_region(p, Pose2D(.7, 0., 0.), b)
    entries, costs = list(p.entry), list(p.cost)
    exits = p._exits(node)
    candidate = p._candidate(node, new_parent, p.entry[new_parent], exits)
    assert candidate is not None and candidate[0] > costs[node]
    p._commit(node, new_parent, p.entry[new_parent], candidate, exits, "rewire")
    assert all(q == entries[i] for i, q in enumerate(p.entry) if i != node)
    assert all(math.isclose(p.cost[i], costs[i]-.1, abs_tol=1e-9) for i in (a, b, tail))
    assert p.stats["compensated_incoming_increases"] == 1
    assert p.stats["compensation_min"] < 0.
    assert_tree(p)


def test_protected_sum_finds_partial_improvement_at_zero_minimax():
    p = make_planner()
    node = append_region(p, Pose2D(0., .5, 0.), 0)
    append_region(p, Pose2D(0., 1., 0.), node)
    append_region(p, Pose2D(1., 0., 0.), node)
    exits = p._exits(node)
    z = p._protected_entry(node, 0, exits, "local")
    assert z is not None
    accepted = p._candidate(node, 0, z, exits)
    assert accepted is not None
    assert max(accepted[1]) <= p.config.protection_tolerance
    assert min(accepted[1]) < -.5
    p._commit(node, 0, z, accepted, exits, "local")
    assert_tree(p)


def test_covered_samples_service_changed_entries_without_extra_sampling():
    p = make_planner()
    node = append_region(p, Pose2D(0., .5, 0.), 0)
    append_region(p, Pose2D(1., 0., 0.), node)
    p._mark_local(node)
    calls = []
    p._sample = lambda: (calls.append(1) or p.start)
    p.step()
    assert len(calls) == 1 and p.stats["sample_calls"] == 1
    assert p.stats["local_accepted"] == 1 and len(p.cells) == 3
    assert p.stats["covered_samples"] == 1
    solves = p.stats["local_solves"]
    p._mark_local(node)
    p.step()
    assert len(calls) == 2 and p.stats["local_solves"] == solves
    assert_tree(p)
