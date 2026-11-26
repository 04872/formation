"""Finite local RRT* in centered affine formation coordinates."""
from dataclasses import dataclass
import numpy as np
from .geometry import certify_edge_cached, distance, region_membership, robot_positions, shape_valid


@dataclass
class Node:
    q: np.ndarray
    parent: int
    cost: float
    edge_cost: float
    certificate: object


def edge_cost(a, b, reference, world):
    """Integrate length-weighted shape and obstacle risk along the edge."""
    length = distance(a, b, reference) / np.sqrt(len(reference))
    if length == 0:
        return 0.
    points = np.array([robot_positions((1-t)*a+t*b, reference) for t in (0., .5, 1.)])
    risk = np.maximum(.8-world.clearance(points), 0.).mean()
    shape = sum(np.linalg.norm(((1-t)*a+t*b)[2:].reshape(2, 2)-np.eye(2))**2
                for t in (0., .5, 1.))/3
    variation = np.linalg.norm(b[2:]-a[2:])
    return float(length*(1+.3*shape+.3*risk)+.08*variation)


def plan(root, goal, reference, regions, world, rng, mode='affine', iterations=250,
         collect_history_candidates=False, collect_certified_edges=False):
    """Build and rewire a bounded tree; return its best goal-biased branch."""
    nodes = [Node(root.copy(), -1, 0., 0., None)]
    children = [set()]
    positions_cache = [robot_positions(root,reference)]
    membership_cache = [region_membership(positions_cache[0],regions)]
    # Preallocated storage keeps nearest and near queries vectorized while
    # preserving node ordering and the original sampling sequence.
    position_array=np.empty((iterations+1,len(reference),2))
    position_array[0]=positions_cache[0]
    accepted = rewires = 0
    certified_edges = []
    for _ in range(iterations):
        if rng.random() < .18:
            sample = goal.copy()
        else:
            direction = np.arctan2(goal[1]-root[1], goal[0]-root[0])
            angle = direction+rng.uniform(-np.pi, np.pi)
            # Forward bias retains both lateral and backward directions.
            if rng.random() < .7:
                angle = direction+rng.uniform(-np.pi/2, np.pi/2)
            radius = rng.uniform(.15, 2.2)
            center = root[:2] + radius*np.array([np.cos(angle), np.sin(angle)])
            if rng.random() < .5:
                matrix=root[2:].reshape(2,2).copy()
            elif mode == 'uniform':
                scale=(rng.uniform(.42,1.35) if rng.random()<.35 else
                       np.clip(root[2]+rng.normal(0,.035),.42,1.35))
                matrix = np.eye(2)*scale
            else:
                if rng.random() < .5:
                    matrix = root[2:].reshape(2,2) + rng.normal(0, .12, (2, 2))
                else:
                    theta=rng.uniform(-np.pi, np.pi)
                    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                    matrix=rot@np.array([[rng.uniform(.42,1.35),rng.uniform(-.6,.6)],
                                         [0.,rng.uniform(.42,1.35)]])
            sample = np.r_[center, matrix.ravel()]
            if not shape_valid(sample, mode):
                continue
        sample_positions=robot_positions(sample,reference)
        distances=np.linalg.norm((position_array[:len(nodes)]-sample_positions).reshape(len(nodes),-1),axis=1)
        near = int(np.argmin(distances))
        factor = min(1., .7*np.sqrt(len(reference))/max(distances[near], 1e-12))
        q = nodes[near].q + factor*(sample-nodes[near].q)
        if not shape_valid(q, mode):
            continue
        qpos=robot_positions(q,reference)
        qmembership=region_membership(qpos,regions)
        qdist=np.linalg.norm((position_array[:len(nodes)]-qpos).reshape(len(nodes),-1),axis=1)
        near_ids = np.flatnonzero(qdist <= 1.0*np.sqrt(len(reference)))
        options = []
        edge_costs={}
        best_cost=np.inf
        for k in near_ids:
            lower=qdist[k]/np.sqrt(len(reference))+.08*np.linalg.norm(q[2:]-nodes[k].q[2:])
            if nodes[k].cost+lower > best_cost+1e-12:
                continue
            cert = certify_edge_cached(nodes[k].q,q,positions_cache[k],qpos,membership_cache[k],qmembership)
            if cert is not None:
                cost = edge_cost(nodes[k].q, q, reference, world)
                edge_costs[int(k)]=cost
                options.append((nodes[k].cost+cost, int(k), cost, cert))
                best_cost=min(best_cost,nodes[k].cost+cost)
        if not options:
            continue
        total, parent, cost, cert = min(options, key=lambda z: z[0])
        index = len(nodes); nodes.append(Node(q, parent, total, cost, cert)); children.append(set())
        positions_cache.append(qpos);membership_cache.append(qmembership);position_array[index]=qpos
        children[parent].add(index); accepted += 1
        if collect_certified_edges:
            certified_edges.extend((k, index) for _, k, _, _ in options)
        for k in near_ids:
            k = int(k)
            if k == 0 or k == parent:
                continue
            lower=qdist[k]/np.sqrt(len(reference))+.08*np.linalg.norm(q[2:]-nodes[k].q[2:])
            if total+lower >= nodes[k].cost-1e-12:
                continue
            cert2 = certify_edge_cached(q,nodes[k].q,qpos,positions_cache[k],qmembership,membership_cache[k])
            if cert2 is None:
                continue
            if collect_certified_edges:
                certified_edges.append((index, k))
            new_edge = edge_costs.get(k)
            if new_edge is None:
                new_edge=edge_cost(q, nodes[k].q, reference, world)
            difference = total+new_edge-nodes[k].cost
            if difference >= -1e-9:
                continue
            # A positive edge cost and reduced cost rule exclude ancestor cycles.
            children[nodes[k].parent].remove(k);children[index].add(k)
            nodes[k].parent = index; nodes[k].edge_cost = new_edge;nodes[k].certificate = cert2
            stack = [k]
            while stack:
                child = stack.pop(); nodes[child].cost += difference; stack.extend(children[child])
            rewires += 1
    # Cost and terminal heuristic share the same workspace-distance units.
    scores = [n.cost + 2*distance(n.q,goal,reference)/np.sqrt(len(reference)) for n in nodes]
    # Select a branch with meaningful local movement, allowing sideways or
    # backwards motion when the best goal-directed connection is blocked.
    candidates=[i for i,n in enumerate(nodes[1:],1)
                if distance(root,n.q,reference)/np.sqrt(len(reference)) >= .4]
    ranked=[]
    for i in candidates:
        branch=[];k=i
        while k>=0:
            branch.append(k);k=nodes[k].parent
        branch=branch[::-1]
        prefix=branch[:4]
        length=sum(distance(nodes[a].q,nodes[b].q,reference)/np.sqrt(len(reference))
                   for a,b in zip(prefix[:-1],prefix[1:]))
        ranked.append((length < 1.1, scores[i], branch if collect_history_candidates else prefix))
    ranked.sort(key=lambda item:(item[0],item[1]))
    candidate_routes = []
    first_edges = set()
    for _, _, branch in ranked:
        if branch[1] not in first_edges:
            candidate_routes.append(branch[:4])
            first_edges.add(branch[1])
    best = min(candidates,key=lambda i:scores[i]) if candidates else 0
    route=[]
    while best >= 0:
        route.append(best); best=nodes[best].parent
    route.reverse()
    stats = {'iterations': iterations, 'accepted': accepted, 'rewires': rewires,
             'candidate_routes': candidate_routes}
    if collect_history_candidates:
        stats['candidate_full_routes'] = [branch for _, _, branch in ranked]
        stats['baseline_route'] = route.copy()
    if collect_certified_edges:
        stats['certified_edges'] = certified_edges
    return nodes, route, stats
