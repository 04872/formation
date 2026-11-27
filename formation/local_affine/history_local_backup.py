"""Finite-depth backups over ordered, previously certified search paths."""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import robot_positions
from .history_graph import HistoryGraph
from .history_joint_geometry import evaluate_pairs


class HistoryLocalBackup:
    def __init__(self, reference, epsilon=0.4, rho=0.4):
        self.reference = np.asarray(reference, dtype=float)
        self.epsilon, self.rho = float(epsilon), float(rho)
        if self.epsilon <= 0 or self.rho <= 0:
            raise ValueError('epsilon and rho must be positive')
        self.rounds = []
        self.heuristic = HistoryGraph()

    def add_round(self, round_id, paths):
        clean = []
        seen = set()
        for path in paths:
            qs = np.asarray(path['nodes'], dtype=float)
            ids = list(path['node_ids'])
            if len(qs) != len(ids):
                raise ValueError('node_ids and nodes lengths differ')
            if len(qs) < 2:
                continue
            key = (tuple(ids), bool(path.get('goal', False)))
            if key in seen:
                continue
            seen.add(key)
            clean.append((qs.copy(), ids, bool(path.get('goal', False))))
        self.rounds.append((int(round_id), clean))

    def evaluate(self, round_id, queries, goal, world, regions, successor_max_round=None, include_successor_options=False, depth=2):
        if depth not in (1, 2):
            raise ValueError('depth must be 1 or 2')
        queries = np.asarray(queries, dtype=float).reshape(-1, 6)
        goal = np.asarray(goal, dtype=float)
        # Deduplicate only within a round: the discovery time and original
        # node identity are essential for the strict chronological comparison.
        qs = [q.copy() for q in queries]
        vertices = {}
        entries = []
        suffixes = {}
        edges = {}
        for k, paths in self.rounds:
            if k >= round_id:
                continue
            for path, ids, is_goal in paths:
                route = []
                for q, node_id in zip(path, ids):
                    key = (k, int(node_id), q.tobytes())
                    if key not in vertices:
                        vertices[key] = len(qs)
                        qs.append(q.copy())
                        entries.append((len(qs)-1, k, int(node_id)))
                    route.append(vertices[key])
                for a, b in zip(route[:-1], route[1:]):
                    edges.setdefault((a, b), len(edges))
                for i, a in enumerate(route[:-1]):
                    suffix = (tuple(route[i:]), is_goal)
                    suffixes.setdefault(a, set()).add(suffix)
        qs = np.asarray(qs, dtype=float).reshape(-1, 6)
        pos = robot_positions(qs, self.reference)
        emb = pos.reshape(len(qs), -1) / np.sqrt(len(self.reference))
        self.heuristic.update(world)
        goals = robot_positions(goal, self.reference)
        if not entries:
            distances = np.asarray(self.heuristic.distances(pos, goals, details=False))
            fallback = not np.all(np.isfinite(distances))
            if fallback:
                distances = np.linalg.norm(pos-goals[None, :, :], axis=2)
            h = np.linalg.norm(distances, axis=1) / np.sqrt(len(self.reference))
            h[np.all(qs == goal, axis=1)] = 0.
            return [dict(h=float(h[i]), v1=float(h[i]), v2=float(h[i]),
                         witness1=None, witness2=None, fixed_v1_successor=None,
                         heuristic_fallback=fallback) for i in range(len(queries))]
        entry_ids = np.asarray([x[0] for x in entries], dtype=int)
        entry_k = np.asarray([x[1] for x in entries], dtype=int)
        entry_node = np.asarray([x[2] for x in entries], dtype=int)
        index_by_vertex = {int(v): i for i, v in enumerate(entry_ids)}
        edge_pairs = np.asarray(list(edges), dtype=int).reshape(-1, 2)
        _, edge_costs = evaluate_pairs(qs, edge_pairs, self.reference, world, regions, certify=False)
        start_ids = np.asarray(sorted(suffixes), dtype=int)
        tree = cKDTree(emb[start_ids])
        # Batch only configurations actually queried. In particular, the
        # second layer needs endpoints reached by a first-layer witness, not
        # every vertex of the complete historical trees.
        links = {}
        def prepare_links(query_ids, min_rounds=None):
            query_ids = [int(q) for q in query_ids if int(q) not in links]
            if not query_ids:
                return
            for q in query_ids:
                links[q] = []
            neighbors = tree.query_ball_point(emb[query_ids], self.epsilon)
            pairs = np.asarray([(q, int(start_ids[j])) for q, near in zip(query_ids, neighbors)
                                for j in near if min_rounds is None or
                                entry_k[index_by_vertex[int(start_ids[j])]] > min_rounds[q]],
                               dtype=int).reshape(-1, 2)
            for start in range(0, len(pairs), 32768):
                batch = pairs[start:start+32768]
                safe, costs = evaluate_pairs(qs, batch, self.reference, world, regions)
                safe &= costs <= self.epsilon
                for (q, a), c in zip(batch[safe], costs[safe]):
                    links[int(q)].append((int(a), float(c)))
        prepare_links(range(len(queries)))
        option_cache = {}
        def options(q):
            if q in option_cache:
                return option_cache[q]
            result = []
            for a, link_cost in links.get(q, ()):
                j = index_by_vertex[a]
                for route, is_goal in sorted(suffixes[a]):
                    route = np.asarray(route, dtype=int)
                    tails = route[1:]
                    da = np.linalg.norm(emb[tails]-emb[a], axis=1)
                    dq = np.linalg.norm(emb[tails]-emb[q], axis=1)
                    eligible = (da >= self.rho) & (dq >= self.rho)
                    if is_goal:
                        eligible[-1] = True
                    hit = np.flatnonzero(eligible)
                    if not len(hit):
                        continue
                    n = int(hit[0])+1
                    z = int(route[n])
                    cost = link_cost + sum(edge_costs[edges[(int(x), int(y))]]
                                           for x, y in zip(route[:n], route[1:n+1]))
                    result.append((float(cost), z, int(entry_k[j]), int(entry_node[j]),
                                   int(entry_node[index_by_vertex[z]]), bool(is_goal and n == len(route)-1), link_cost,
                                   [int(entry_node[index_by_vertex[int(v)]]) for v in route[:n+1]]))
            option_cache[q] = result
            return result
        def witness(o, successor=None):
            c, z, k, node, end_node, is_goal, link_cost, route = o
            return {'round': k, 'node': node, 'end_node': end_node,
                    'cost': c, 'c_link': link_cost, 'c_path': c-link_cost, 'h': float(h[z]), 'goal': is_goal,
                    'end_q': qs[z].tolist(),
                    'path_node_ids': route,
                    'successor': successor}
        successor_cache = {}
        def successor(o):
            _, z, k, _, _, is_goal, _, _ = o
            if is_goal:
                return 0., None
            if (z, k) in successor_cache:
                return successor_cache[(z, k)]
            candidates = [x for x in options(z) if x[2] > k and
                          (successor_max_round is None or x[2] <= successor_max_round)]
            if not candidates:
                successor_cache[(z, k)] = (float(h[z]), None)
                return successor_cache[(z, k)]
            best = min(candidates, key=lambda x: x[0] + (0. if x[5] else h[x[1]]))
            successor_cache[(z, k)] = (float(best[0] + (0. if best[5] else h[best[1]])), witness(best))
            return successor_cache[(z, k)]
        first_choices = [options(q) for q in range(len(queries))]
        needed = {}
        if depth == 2:
            for choices in first_choices:
                for x in choices:
                    if not x[5]:
                        needed[x[1]] = min(needed.get(x[1], x[2]), x[2])
            prepare_links(needed, needed)
        # The fallback applies to every historical vertex, even those that
        # cannot participate in this backup. Prove reachability cheaply via a
        # direct segment or a visible goal-connected graph vertex; only points
        # without such a certificate require the full exact distance search.
        fallback = False
        graph = self.heuristic
        for member, target in enumerate(goals):
            points = np.unique(pos[:, member, :], axis=0)
            direct, _ = graph._segments(points, np.broadcast_to(target, points.shape), classify=False)
            remaining = points[~direct]
            if not len(remaining):
                continue
            # Populate the same goal-connected vertex set used by _distances.
            graph._distances(remaining[:1, None, :], target[None, :], details=False)
            targets, _ = graph._goal_cache.get(tuple(target), (np.empty((0, 2)), None))
            if len(targets):
                _, nearest = cKDTree(targets).query(remaining)
                visible, _ = graph._segments(remaining, targets[nearest], classify=False)
                remaining = remaining[~visible]
            if len(remaining):
                exact = graph._distances(remaining[:, None, :], target[None, :], details=False)
                if not np.all(np.isfinite(exact)):
                    fallback = True
                    break
        required = set(range(len(queries)))
        for choices in first_choices:
            required.update(x[1] for x in choices)
        if depth == 2:
            for z in needed:
                required.update(x[1] for x in options(z))
        required = np.asarray(sorted(required), dtype=int)
        h = np.full(len(qs), np.nan)
        if fallback:
            distances = np.linalg.norm(pos[required]-goals[None, :, :], axis=2)
        else:
            distances = graph.distances(pos[required], goals, details=False)
        h[required] = np.linalg.norm(distances, axis=1) / np.sqrt(len(self.reference))
        h[np.all(qs == goal, axis=1)] = 0.
        output = []
        for q in range(len(queries)):
            choices = options(q)
            if not choices:
                output.append(dict(h=float(h[q]), v1=float(h[q]), v2=float(h[q]),
                                   witness1=None, witness2=None, fixed_v1_successor=None,
                                   heuristic_fallback=fallback))
                continue
            v1best = min(choices, key=lambda x: x[0] + (0. if x[5] else h[x[1]]))
            if depth == 1:
                output.append(dict(h=float(h[q]),
                    v1=float(v1best[0]+(0. if v1best[5] else h[v1best[1]])),
                    witness1=witness(v1best), heuristic_fallback=fallback))
                continue
            values = [(x[0]+successor(x)[0], x) for x in choices]
            value2, v2best = min(values, key=lambda x: x[0])
            fixed, fixed_w = successor(v1best)
            _, v2_w = successor(v2best)
            fixed_detail = {'value': fixed, 'witness': fixed_w}
            if include_successor_options:
                z, k = v1best[1], v1best[2]
                candidates = [] if v1best[5] else [x for x in options(z) if x[2] > k]
                fixed_detail['options'] = [dict(witness(x), value=float(x[0] + (0. if x[5] else h[x[1]])),
                    distance_from_original=float(np.linalg.norm(emb[x[1]]-emb[q]))) for x in candidates]
            output.append(dict(h=float(h[q]),
                               v1=float(v1best[0]+(0. if v1best[5] else h[v1best[1]])),
                               v2=float(value2), witness1=witness(v1best),
                               witness2=witness(v2best, v2_w),
                               fixed_v1_successor=fixed_detail,
                               heuristic_fallback=fallback))
        return output
