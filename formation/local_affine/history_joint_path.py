"""Reuse certified joint affine paths to define the rolling evaluation boundary."""
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.sparse.csgraph import dijkstra

from formation.paper_formation.world import PERCEPTION_RADIUS
from .geometry import certify_edge, robot_positions
from .history_graph import HistoryGraph
from .history_joint_geometry import evaluate_pairs


class HistoryJointPath:
    def __init__(self, epsilon=0.4, current_motion_only=False):
        if epsilon <= 0:
            raise ValueError('history link threshold must be positive')
        self.epsilon = float(epsilon)
        self.current_motion_only = bool(current_motion_only)
        self.paths = []
        self.eligible_endpoints_total = 0
        self.cached_links = np.empty((0, 2), dtype=np.int64)
        self.free_keys = np.empty(0, dtype=np.uint64)
        self.free_costs = np.empty(0)
        self.regions = []
        self.region_keys = set()
        self.known = self.occupied = None
        self.heuristic = HistoryGraph()

    def _update_map(self, world, regions):
        known = np.asarray(world.known, dtype=bool)
        occupied = np.asarray(world.occupied, dtype=bool)
        # Old SFC certificates remain valid under cumulative observations of a
        # static map. If an already observed cell changes, discard them: the
        # original convex regions can no longer certify free space.
        changed = (self.known is not None and
                   (np.any(self.known & ~known) or
                    np.any(self.known & (self.occupied != occupied))))
        if changed:
            self.paths.clear()
            self.cached_links = np.empty((0, 2), dtype=np.int64)
            self.free_keys = np.empty(0, dtype=np.uint64)
            self.free_costs = np.empty(0)
            self.regions.clear()
            self.region_keys.clear()
        self.known, self.occupied = known.copy(), occupied.copy()
        for h, b in regions:
            key = (np.asarray(h).tobytes(), np.asarray(b).tobytes())
            if key not in self.region_keys:
                self.region_keys.add(key)
                self.regions.append((np.asarray(h).copy(), np.asarray(b).copy()))

        return changed

    def record_direct(self, root, goal, world, regions):
        """Preserve a directly certified goal segment for a later replan."""
        changed = self._update_map(world, regions)
        if not self.current_motion_only:
            self.paths.append([np.asarray(root).copy(), np.asarray(goal).copy()])
        return {'status': 'direct_goal', 'current_goal_sources_count': 1,
                'eligible_endpoints_total': self.eligible_endpoints_total,
                'history_paths': len(self.paths), 'map_reset': bool(changed)}

    def select(self, nodes, stats, reference, goal, world, regions):
        started = time.perf_counter()
        changed = self._update_map(world, regions)
        routes = sorted(stats.get('candidate_full_routes', []), key=lambda p: p[-1])
        self.eligible_endpoints_total += len(routes)
        if not routes:
            current_goal_sources_count = 0
            for i, node in enumerate(nodes):
                if certify_edge(node.q, goal, self.regions, reference) is not None:
                    current_goal_sources_count += 1
                    branch = []
                    while i >= 0:
                        branch.append(nodes[i].q.copy())
                        i = nodes[i].parent
                    if not self.current_motion_only:
                        self.paths.append(branch[::-1] + [goal.copy()])
            return stats.get('baseline_route', [0]), {'status': 'no_candidates', 'history_paths': len(self.paths),
                    'current_goal_sources_count': current_goal_sources_count,
                    'eligible_endpoints_total': self.eligible_endpoints_total}
        # Share identical configurations and certified edges across paths.
        qs = []
        ids = {}
        def vertex(q):
            q = np.asarray(q, dtype=float)
            key = q.tobytes()
            if key not in ids:
                ids[key] = len(qs)
                qs.append(q.copy())
            return ids[key]
        paths = [[vertex(q) for q in path] for path in self.paths]
        historical_count = len(qs)
        current = [vertex(n.q) for n in nodes]
        # Canonical undirected pairs share the exact original certificate.
        certified = set()
        def add_pair(a, b):
            if a != b:
                certified.add((min(a, b), max(a, b)))
        for a, b in stats.get('certified_edges', []):
            add_pair(current[a], current[b])
        for i, n in enumerate(nodes[1:], 1):
            add_pair(current[n.parent], current[i])
        for path in paths:
            for a, b in zip(path[:-1], path[1:]):
                add_pair(a, b)
        goal_id = vertex(goal)
        pos = robot_positions(np.asarray(qs), reference)
        goal_pairs = np.asarray([(a, goal_id) for a in set(current) if a != goal_id], dtype=int).reshape(-1, 2)
        safe, _ = evaluate_pairs(qs, goal_pairs, reference, world, self.regions)
        current_goal_sources = []
        for (a, _), ok in zip(goal_pairs, safe):
            if ok:
                add_pair(a, goal_id)
                node_id = current.index(a)
                branch = []
                while node_id >= 0:
                    branch.append(current[node_id])
                    node_id = nodes[node_id].parent
                paths.append(branch[::-1] + [goal_id])
                current_goal_sources.append(int(a))
        embedding = pos.reshape(len(pos), -1) / np.sqrt(len(reference))
        near = cKDTree(embedding).query_pairs(self.epsilon, output_type='ndarray')
        known_pairs = np.asarray(sorted(certified), dtype=int).reshape(-1, 2)
        def pair_keys(p):
            return (p[:, 0].astype(np.uint64) << np.uint64(32)) | p[:, 1].astype(np.uint64)

        def weights_for(pairs):
            keys = pair_keys(pairs)
            lookup = np.searchsorted(self.free_keys, keys)
            hit = lookup < len(self.free_keys)
            if len(self.free_keys):
                hit &= self.free_keys[np.minimum(lookup, len(self.free_keys)-1)] == keys
            result = np.empty(len(pairs))
            result[hit] = self.free_costs[lookup[hit]]
            missing = np.flatnonzero(~hit)
            if len(missing):
                pp = pairs[missing]
                _, values = evaluate_pairs(qs, pp, reference, world, self.regions, certify=False)
                result[missing] = values
                # Risk-free weights remain fixed as cumulative free space grows.
                qa, qb = np.asarray(qs)[pp[:, 0]], np.asarray(qs)[pp[:, 1]]
                length = np.linalg.norm((pos[pp[:, 0]]-pos[pp[:, 1]]).reshape(len(pp), -1), axis=1)/np.sqrt(len(reference))
                shapes = np.stack((qa[:, 2:], (qa[:, 2:]+qb[:, 2:])/2, qb[:, 2:]), axis=1) - np.array([1., 0., 0., 1.])
                shape = np.sum(shapes*shapes, axis=(1, 2))/3
                base = length*(1+.3*shape)+.08*np.linalg.norm(qb[:, 2:]-qa[:, 2:], axis=1)
                free = (values == base) & np.all(pp < historical_count, axis=1)
                if np.any(free):
                    keys_all = np.concatenate((self.free_keys, keys[missing[free]]))
                    costs_all = np.concatenate((self.free_costs, values[free]))
                    order = np.argsort(keys_all)
                    self.free_keys, self.free_costs = keys_all[order], costs_all[order]
            return result

        known_costs = weights_for(known_pairs)
        # New short links require the full original certificate.  The cost
        # threshold is checked after certification, not merely distance.
        # Stable historical vertex IDs let certified old links be reused.
        # Uncertified old pairs are retried because new SFCs may enable them.
        key = pair_keys
        cached = self.cached_links
        already = np.concatenate((known_pairs, cached))
        pending = near[~np.isin(key(near), key(already))]
        # A nonnegative obstacle-risk term can only increase this edge cost.
        # Filter in chunks to avoid constructing large temporary tensors when
        # many historical configurations fall within the distance radius.
        keep = np.empty(len(pending), dtype=bool)
        configurations = np.asarray(qs)
        for first in range(0, len(pending), 65536):
            pairs = pending[first:first+65536]
            qa, qb = configurations[pairs[:, 0]], configurations[pairs[:, 1]]
            length = np.linalg.norm((pos[pairs[:, 0]]-pos[pairs[:, 1]]).reshape(len(pairs), -1), axis=1)/np.sqrt(len(reference))
            shapes = np.stack((qa[:, 2:], (qa[:, 2:]+qb[:, 2:])/2, qb[:, 2:]), axis=1) - np.array([1., 0., 0., 1.])
            shape = np.sum(shapes*shapes, axis=(1, 2))/3
            base = length*(1+.3*shape)+.08*np.linalg.norm(qb[:, 2:]-qa[:, 2:], axis=1)
            # edge_cost treats zero member displacement as a zero-cost edge.
            keep[first:first+len(pairs)] = (length == 0) | (base <= np.nextafter(self.epsilon, np.inf))
        pending = pending[keep]
        safe, costs = evaluate_pairs(qs, pending, reference, world, self.regions)
        failed = np.flatnonzero(~safe)
        if len(failed):
            reverse_safe, reverse_costs = evaluate_pairs(qs, pending[failed, ::-1], reference, world, self.regions)
            safe[failed] = reverse_safe
            costs[failed] = reverse_costs
        accepted = safe & (costs <= self.epsilon)
        new_links = pending[accepted]
        historical_links = new_links[np.all(new_links < historical_count, axis=1)]
        self.cached_links = np.unique(np.concatenate((cached, historical_links)), axis=0)
        cached_only = cached[~np.isin(key(cached), key(known_pairs))]
        cached_costs = weights_for(cached_only)
        edge_pairs = np.concatenate((known_pairs, cached_only, new_links))
        edge_costs = np.concatenate((known_costs, cached_costs, costs[accepted]))
        rows = np.concatenate((edge_pairs[:, 0], edge_pairs[:, 1]))
        cols = np.concatenate((edge_pairs[:, 1], edge_pairs[:, 0]))
        weights = np.concatenate((edge_costs, edge_costs))
        graph = coo_matrix((weights, (rows, cols)), shape=(len(qs), len(qs))).tocsr()

        # Only the map built from actual observations defines novelty. A
        # nearest-neighbour query over unknown cell centres is exactly the
        # union of the members' 3 m sensing disks, clipped to the grid.
        unknown = np.argwhere(~self.known)
        if len(unknown):
            unknown_centres = unknown * world.res + world.res / 2
            points = pos.reshape(-1, 2)
            _, nearest = cKDTree(unknown_centres).query(points)
            delta = unknown_centres[nearest] - points
            visible = delta[:, 0]**2 + delta[:, 1]**2 <= PERCEPTION_RADIUS**2
            observation = np.any(visible.reshape(len(qs), len(reference)), axis=1)
        else:
            observation = np.zeros(len(qs), dtype=bool)
        boundary_mask = observation.copy()
        boundary_mask[goal_id] = True
        boundary = np.flatnonzero(boundary_mask).tolist()
        self.heuristic.update(world)
        goals = robot_positions(goal, reference)
        candidate_vertices = [current[p[-1]] for p in routes]
        starts = pos[boundary + candidate_vertices]
        distances = np.asarray(self.heuristic.distances(starts, goals, details=False))
        # Match the old mode's all-or-nothing finite fallback within the round.
        if not np.all(np.isfinite(distances)):
            distances = np.linalg.norm(starts-goals[None, :, :], axis=2)
            fallback = True
        else:
            fallback = False
        terminal_values = np.linalg.norm(distances, axis=1)/np.sqrt(len(reference))
        values = terminal_values[:len(boundary)]
        # Add a virtual source with weighted edges to all terminals, then
        # solve all remaining values in one compiled sparse shortest path.
        values = np.asarray(values).copy()
        values[np.asarray(boundary) == goal_id] = 0.
        seed = csr_matrix((values, (np.zeros(len(boundary), dtype=int), boundary)), shape=(1, len(qs)))
        augmented = bmat([[graph.T, csr_matrix((len(qs), 1))], [seed, csr_matrix((1, 1))]], format='csr')
        h = dijkstra(augmented, directed=True, indices=len(qs))[:len(qs)]
        graph_scores = [nodes[p[-1]].cost + h[current[p[-1]]] for p in routes]
        original = [nodes[p[-1]].cost + value for p, value in zip(routes, terminal_values[len(boundary):])]
        round_fallback = not np.any(np.isfinite(graph_scores))
        scores = original if round_fallback else graph_scores
        baseline = min(range(len(routes)), key=lambda i: (original[i], routes[i][-1]))
        # Only prefix-related exact ties prefer the longer guide.
        best = 0
        for i in range(1, len(routes)):
            if scores[i] < scores[best] or (scores[i] == scores[best] and len(routes[i]) > len(routes[best]) and routes[i][:len(routes[best])] == routes[best]):
                best = i
        # Keep all eligible paths across all rounds; shared graph vertices and
        # edges are deduplicated at reconstruction, never by endpoint cost.
        # A candidate that is a prefix of another route is already fully
        # represented by that route's shared edges.
        if not self.current_motion_only:
            prefix_ids = {i for route in routes for i in route[:-1]}
            for route in routes:
                if route[-1] not in prefix_ids:
                    self.paths.append([nodes[i].q.copy() for i in route])
            for a in current_goal_sources:
                node_id = current.index(a)
                branch = []
                while node_id >= 0:
                    branch.append(nodes[node_id].q.copy())
                    node_id = nodes[node_id].parent
                self.paths.append(branch[::-1] + [goal.copy()])
        remaining = terminal_values[len(boundary):] if round_fallback else h[candidate_vertices]
        evidence = {'status': 'joint_path_scoring', 'epsilon': self.epsilon, 'terminal_heuristic_weight': 1.0,
                    'current_motion_only': self.current_motion_only,
                    'candidate_count': len(routes), 'eligible_endpoints_total': self.eligible_endpoints_total,
                    'current_goal_sources_count': len(current_goal_sources), 'selected_endpoint': int(routes[best][-1]),
                    'baseline_endpoint': int(routes[baseline][-1]), 'changed_selection': bool(best != baseline),
                    'closed_count': int(len(qs)-len(boundary)),
                    'selected_score': float(scores[best]), 'vertices': len(qs),
                    'edges': int(2*len(edge_pairs)), 'boundary': len(boundary),
                    'history_paths': len(self.paths), 'history_regions': len(self.regions),
                    'heuristic_fallback': fallback, 'round_fallback': bool(round_fallback),
                    'observation_boundary_count': int(np.count_nonzero(observation)),
                    'goal_boundary_count': 1, 'map_reset': bool(changed),
                    'seconds': time.perf_counter()-started,
                    'candidates': [
                        {'endpoint': int(p[-1]), 'tree_cost': float(nodes[p[-1]].cost),
                         'observation_boundary': bool(observation[v]),
                         'remaining_cost': float(r) if np.isfinite(r) else None,
                         'original_score': float(o), 'score': float(s) if np.isfinite(s) else None}
                        for p, v, r, o, s in zip(routes, candidate_vertices, remaining, original, scores)]}
        return routes[best], evidence
