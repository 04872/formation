#!/usr/bin/env python3
"""Recover certified RRT edges from stored accepted-node insertion order."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from formation.local_affine.geometry import certify_edge_cached, distance, region_membership, robot_positions
from formation.local_affine.rrt import Node, edge_cost
from formation.paper_formation.world import World

run = ROOT / 'results/three_routes/joint_path_h1_closed_loop/history_deadend/rrt_affine/seed_0'
out = Path('/tmp/obs_boundary_rebuilt')
out.mkdir(exist_ok=True)
saved = json.loads((run / 'planning.json').read_text())
traj = np.load(run / 'trajectory.npz')
reg = np.load(run / 'regions.npz')
ref = traj['centered_reference']
goal = np.r_[traj['goal_center'], np.eye(2).ravel()]

for t, old in enumerate(saved['frontend_metadata']):
    records = old['tree']
    regions = []
    for r in range(6):
        i = t * 6 + r
        v = reg['coefficients'][reg['offsets'][i]:reg['offsets'][i+1]]
        regions.append((v[:, :2], v[:, 2]))
    world = World('history_deadend', obstacle_count=3, map_seed=1, pillar_width=.3)
    world.known = traj['known_history'][t].copy()
    world.occupied = world.known & world.truth
    assert np.array_equal(world.truth, traj['occupancy'])
    q0 = np.array(records[0]['q'])
    nodes = [Node(q0, -1, 0., 0., None)]
    children = [set()]
    pos = [robot_positions(q0, ref)]
    membership = [region_membership(pos[0], regions)]
    certified = []
    for index, record in enumerate(records[1:], 1):
        q = np.array(record['q'])
        qp = robot_positions(q, ref)
        qm = region_membership(qp, regions)
        ds = np.linalg.norm((np.asarray(pos) - qp).reshape(len(pos), -1), axis=1)
        near = np.flatnonzero(ds <= np.sqrt(len(ref)))
        options, edge_costs = [], {}
        best = np.inf
        for k in near:
            lower = ds[k]/np.sqrt(len(ref)) + .08*np.linalg.norm(q[2:]-nodes[k].q[2:])
            if nodes[k].cost + lower > best + 1e-12:
                continue
            cert = certify_edge_cached(nodes[k].q, q, pos[k], qp, membership[k], qm)
            if cert is not None:
                cost = edge_cost(nodes[k].q, q, ref, world)
                edge_costs[int(k)] = cost
                options.append((nodes[k].cost+cost, int(k), cost, cert))
                best = min(best, nodes[k].cost+cost)
        assert options, (t, index)
        total, parent, cost, cert = min(options, key=lambda z: z[0])
        nodes.append(Node(q, parent, total, cost, cert))
        children.append(set()); children[parent].add(index)
        pos.append(qp); membership.append(qm)
        certified.extend((k, index) for _, k, _, _ in options)
        for k0 in near:
            k = int(k0)
            if k == 0 or k == parent:
                continue
            lower = ds[k]/np.sqrt(len(ref)) + .08*np.linalg.norm(q[2:]-nodes[k].q[2:])
            if total + lower >= nodes[k].cost - 1e-12:
                continue
            cert2 = certify_edge_cached(q, nodes[k].q, qp, pos[k], qm, membership[k])
            if cert2 is None:
                continue
            certified.append((index, k))
            new = edge_costs.get(k)
            if new is None:
                new = edge_cost(q, nodes[k].q, ref, world)
            difference = total + new - nodes[k].cost
            if difference >= -1e-9:
                continue
            children[nodes[k].parent].remove(k); children[index].add(k)
            nodes[k].parent = index; nodes[k].edge_cost = new; nodes[k].certificate = cert2
            stack = [k]
            while stack:
                child = stack.pop(); nodes[child].cost += difference; stack.extend(children[child])
    assert [n.parent for n in nodes] == [v['parent'] for v in records], t
    cerr = max(abs(n.cost-v['cost']) for n, v in zip(nodes, records))
    assert cerr < 1e-9, (t, cerr)
    eligible = [i for i, n in enumerate(nodes[1:], 1)
                if distance(q0, n.q, ref) / np.sqrt(len(ref)) >= .4]
    assert set(eligible) == {c['endpoint'] for c in old['statistics']['history_decision']['candidates']}, t
    routes = []
    for endpoint in eligible:
        i = endpoint; route = []
        while i >= 0:
            route.append(i); i = nodes[i].parent
        routes.append(route[::-1])
    stats = {'candidate_full_routes': routes, 'certified_edges': certified,
             'baseline_route': old['route']}
    # Clear reconstructible clearance caches before pickling.
    for key in ('_blocked_mask', '_dist', '_blocked_points', '_blocked_tree'):
        world.__dict__.pop(key, None)
    world._dirty = True
    data = dict(nodes=nodes, stats=stats, reference=ref, goal=goal, world=world,
                regions=regions, old=old['statistics']['history_decision'],
                route=old['route'], qerr=0., cerr=cerr)
    with (out / f'{t:03d}.pkl').open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{t:03d} nodes={len(nodes)} edges={len(certified)} cost_error={cerr:.3g}', flush=True)
