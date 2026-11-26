"""Observed-map visibility graph used to estimate individual remaining paths.

Unknown cells are traversable hypotheses for estimation, never certificates for
execution. All collision checks use only the cumulative observed occupancy.
"""
from collections import OrderedDict

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import shapely
from shapely.strtree import STRtree
from formation.paper_formation.config import RADIUS


class HistoryGraph:
    def __init__(self, radius=RADIUS, spacing=0.8, safety_margin=0.):
        self.radius = float(radius) + float(safety_margin)
        self.spacing = float(spacing)
        self.nodes = []
        self._indices = {}
        self._edges = {}
        self._contour = set()
        self._obstacle_tree = None
        self._obstacle_union = None
        self._active = np.empty(0, dtype=bool)
        self._known = None
        self._occupied = None
        self._world = None
        self._free_dist = None
        self._known_dist = None
        self._graph = None
        self._ids = np.empty(0, dtype=int)
        self.last_query_stats = {}
        self.last_query_status = None
        self._goal_cache = {}
        self._distance_cache = OrderedDict()
        self._geometry_key = None

    def _segments(self, starts, ends, classify=True):
        """Return (traversable, fully observed) for straight segments.

        A centre-distance EDT minus two cell circumradii and half the sample
        interval is a lower bound on continuous clearance from blocked cells.
        """
        starts = np.asarray(starts, dtype=float).reshape(-1, 2)
        ends = np.asarray(ends, dtype=float).reshape(-1, 2)
        if not classify:
            res = self._world.res
            extent = np.asarray(self._world.size)*res
            inside = (np.all(starts >= self.radius, axis=1) & np.all(ends >= self.radius, axis=1)
                      & np.all(starts <= extent-self.radius, axis=1)
                      & np.all(ends <= extent-self.radius, axis=1))
            good = inside.copy()
            if self._obstacle_union is not None and np.any(inside):
                lines = shapely.linestrings(np.stack((starts[inside], ends[inside]), axis=1))
                good[inside] = ~shapely.dwithin(lines, self._obstacle_union, self.radius)
            return good, None
        good = np.empty(len(starts), dtype=bool)
        known = np.empty(len(starts), dtype=bool)
        res = self._world.res
        extent = np.asarray(self._world.size)*res
        threshold = self.radius + res*(np.sqrt(2)+.25)
        for first in range(0, len(starts), 512):
            a, b = starts[first:first+512], ends[first:first+512]
            count = np.maximum(1, np.ceil(np.linalg.norm(b-a, axis=1)/(res/2)).astype(int))
            owner = np.repeat(np.arange(len(a)), count+1)
            bases = np.repeat(np.cumsum(count+1)-(count+1), count+1)
            t = (np.arange(len(owner))-bases)/count[owner]
            p = a[owner]+t[:, None]*(b[owner]-a[owner])
            cells = np.floor(p/res).astype(int)
            inside = (np.all((cells >= 0) & (cells < np.asarray(self._world.size)), axis=1)
                      & np.all(p >= self.radius+res/4, axis=1)
                      & np.all(p <= extent-self.radius-res/4, axis=1))
            ok = np.zeros(len(p), dtype=bool)
            seen = np.zeros(len(p), dtype=bool)
            c = cells[inside]
            ok[inside] = self._free_dist[c[:, 0], c[:, 1]] >= threshold
            seen[inside] = self._known_dist[c[:, 0], c[:, 1]] >= threshold
            offsets = np.r_[0, np.cumsum(count+1)[:-1]]
            good[first:first+len(a)] = np.minimum.reduceat(ok, offsets)
            known[first:first+len(a)] = np.minimum.reduceat(seen, offsets)
        # EDT rejection is only a warning, not proof of collision. Resolve
        # ambiguous segments exactly against observed occupied cell squares.
        extent = np.asarray(self._world.size)*res
        in_bounds = (np.all(starts >= self.radius, axis=1) & np.all(ends >= self.radius, axis=1)
                     & np.all(starts <= extent-self.radius, axis=1)
                     & np.all(ends <= extent-self.radius, axis=1))
        uncertain = np.flatnonzero(~good & in_bounds)
        if len(uncertain):
            if self._obstacle_tree is None:
                good[uncertain] = True
            else:
                lines = shapely.linestrings(np.stack((starts[uncertain], ends[uncertain]), axis=1))
                hits = self._obstacle_tree.query(lines, predicate='dwithin', distance=self.radius)
                blocked = np.zeros(len(uncertain), dtype=bool)
                if hits.size:
                    blocked[hits[0]] = True
                good[uncertain] = ~blocked
        good &= in_bounds
        return good, known & good

    def update(self, world):
        self._world = world
        known = np.asarray(world.known, dtype=bool)
        occupied = known & np.asarray(world.occupied, dtype=bool)
        geometry_key = (float(world.res), tuple(world.size), self.radius)
        same_geometry = geometry_key == self._geometry_key
        if (same_geometry and self._known is not None and np.array_equal(known, self._known)
                and np.array_equal(occupied, self._occupied)):
            return
        old_ids = self._ids
        old_edges = self._edges
        old_occupied = self._occupied
        res = world.res
        if not same_geometry or old_occupied is None or not np.array_equal(occupied, old_occupied):
            obstacle_cells = np.argwhere(occupied)
            if len(obstacle_cells):
                lo = obstacle_cells*res
                hi = (obstacle_cells+1)*res
                boxes = shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1])
                self._obstacle_tree = STRtree(boxes)
                self._obstacle_union = shapely.simplify(shapely.coverage_union_all(boxes), tolerance=0)
            else:
                self._obstacle_tree = None
                self._obstacle_union = None
            self._free_dist = distance_transform_edt(np.pad(~occupied, 1, constant_values=False))[1:-1, 1:-1]*res
        self._known_dist = distance_transform_edt(np.pad(known & ~occupied, 1, constant_values=False))[1:-1, 1:-1]*res
        step = max(1, int(round(self.spacing/res)))
        self.spacing = step*res
        xs = np.arange(step//2, world.size[0], step)
        ys = np.arange(step//2, world.size[1], step)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        cells = np.column_stack((xx.ravel(), yy.ravel()))
        threshold = self.radius + res*(np.sqrt(2)+.25)
        valid = known[cells[:, 0], cells[:, 1]] & (self._known_dist[cells[:, 0], cells[:, 1]] >= threshold)
        for cell in cells[valid]:
            key = tuple(cell)
            if key not in self._indices:
                self._indices[key] = len(self.nodes)
                self.nodes.append((cell+.5)*res)
        # Sample observed obstacle boundary on a sparse spatial grid. Points
        # just outside the footprint may lie in unknown space and act only as
        # speculative visibility waypoints, never execution certificates.
        boundary = occupied & ~binary_erosion(occupied, border_value=0)
        boundary_cells = np.argwhere(boundary)
        if len(boundary_cells):
            buckets = boundary_cells//step
            _, first = np.unique(buckets, axis=0, return_index=True)
            boundary_cells = boundary_cells[np.sort(first)]
            angles = np.arange(8)*np.pi/4
            offsets = np.column_stack((np.cos(angles), np.sin(angles)))
            centers = (boundary_cells+.5)*res
            distance = self.radius + res*(np.sqrt(2)+1.5)
            contour_cells = np.floor((centers[:, None, :] + distance*offsets[None, :, :])/res).astype(int).reshape(-1, 2)
            inside = np.all((contour_cells >= 0) & (contour_cells < np.asarray(world.size)), axis=1)
            contour_cells = np.unique(contour_cells[inside], axis=0)
            points = (contour_cells+.5)*res
            safe, _ = self._segments(points, points)
            for cell in contour_cells[safe]:
                key = tuple(cell)
                if key not in self._indices:
                    self._indices[key] = len(self.nodes)
                    self.nodes.append((cell+.5)*res)
                self._contour.add(self._indices[key])
        coords = np.asarray(self.nodes, dtype=float).reshape(-1, 2)
        if len(coords):
            node_cells = np.floor(coords/res).astype(int)
            self._active = (known[node_cells[:, 0], node_cells[:, 1]] &
                            (self._known_dist[node_cells[:, 0], node_cells[:, 1]] >= threshold))
        else:
            self._active = np.empty(0, dtype=bool)
        if self._contour:
            contour_ids = np.asarray(sorted(self._contour), dtype=int)
            safe, _ = self._segments(coords[contour_ids], coords[contour_ids])
            self._active[contour_ids] |= safe
        self._ids = np.flatnonzero(self._active)
        # Retain old geometry and only discover sparse local pairs near new
        # nodes; all old edges are rechecked when new obstacles appear.
        changed_obstacle = self._occupied is None or not np.array_equal(occupied, self._occupied)
        edges = {pair: value for pair, value in self._edges.items()
                 if self._active[pair[0]] and self._active[pair[1]]}
        pairs = set()
        if len(self._ids):
            tree = cKDTree(coords[self._ids])
            for a, b in tree.query_pairs(2.5*self.spacing, output_type='ndarray'):
                pair = (int(self._ids[a]), int(self._ids[b]))
                if pair not in edges:
                    pairs.add(pair)
        if changed_obstacle:
            # Unknown edges become invalid when new occupied evidence arrives.
            pairs.update(edges)
            edges = {}
        else:
            # Observation can turn a speculative edge into a known-free edge.
            newly_seen = [pair for pair, (_, seen) in edges.items() if not seen]
            for pair in newly_seen:
                pairs.add(pair)
                del edges[pair]
        pairs = sorted(pairs)
        if pairs:
            traversable, seen = self._segments([coords[a] for a,b in pairs], [coords[b] for a,b in pairs])
            for (a,b), ok, is_known in zip(pairs, traversable, seen):
                if ok:
                    edges[(a,b)] = (float(np.linalg.norm(coords[a]-coords[b])), bool(is_known))
        self._edges = edges
        rows, cols, data = [], [], []
        for (a,b), (length, _) in edges.items():
            rows.extend((a,b)); cols.extend((b,a)); data.extend((length,length))
        self._graph = csr_matrix((data,(rows,cols)),shape=(len(coords),len(coords)))
        # Known/unknown labels affect diagnostics but not distance if the
        # active vertices, traversable edges and occupied geometry are unchanged.
        same_graph = (same_geometry and old_occupied is not None and
                      np.array_equal(occupied, old_occupied) and
                      np.array_equal(self._ids, old_ids) and
                      old_edges.keys() == edges.keys() and
                      all(old_edges[pair][0] == value[0] for pair, value in edges.items()))
        if not same_graph:
            self._goal_cache.clear()
            self._distance_cache.clear()
        self._geometry_key = geometry_key
        self._known = known.copy()
        self._occupied = occupied.copy()

    def distances(self, starts, goals, details=True):
        """Return member distances, reusing exact distance-only queries."""
        starts = np.asarray(starts, dtype=float)
        goals = np.asarray(goals, dtype=float)
        if starts.ndim != 3 or starts.shape[2] != 2 or goals.shape != starts.shape[1:]:
            raise ValueError('starts must be (candidates, members, 2), goals (members, 2)')
        if details:
            return self._distances(starts, goals, details=True)
        result = np.empty(starts.shape[:2], dtype=float)
        for member, goal in enumerate(goals):
            unique, inverse = np.unique(starts[:, member, :], axis=0, return_inverse=True)
            values = np.empty(len(unique), dtype=float)
            missing = []
            goal_key = goal.tobytes()
            for i, point in enumerate(unique):
                key = (goal_key, point.tobytes())
                if key in self._distance_cache:
                    values[i] = self._distance_cache[key]
                    self._distance_cache.move_to_end(key)
                else:
                    missing.append(i)
            if missing:
                missing = np.asarray(missing, dtype=int)
                computed = self._distances(unique[missing, None, :], goal[None, :], details=False)[:, 0]
                values[missing] = computed
                for i, value in zip(missing, computed):
                    key = (goal_key, unique[i].tobytes())
                    self._distance_cache[key] = float(value)
                    if len(self._distance_cache) > 100_000:
                        self._distance_cache.popitem(last=False)
            result[:, member] = values[inverse]
        self.last_query_status = None
        self.last_query_stats = {}
        return result

    def _distances(self, starts, goals, details=True):
        """Batch member distances, with np.inf when no connection is found.

        Each endpoint can connect directly to its goal or to any active graph
        vertex, including across unknown space if known obstacles permit it.
        Goal-to-graph distances are shared across all candidate endpoints and
        cached while the observed map is unchanged. Set details=False to skip
        observation-status diagnostics for large distance-only queries.
        """
        starts = np.asarray(starts, dtype=float)
        goals = np.asarray(goals, dtype=float)
        if starts.ndim != 3 or starts.shape[2] != 2 or goals.shape != starts.shape[1:]:
            raise ValueError('starts must be (candidates, members, 2), goals (members, 2)')
        count, members = starts.shape[:2]
        result = np.full((count, members), np.inf)
        direct_start = starts.reshape(-1, 2)
        direct_goals = np.broadcast_to(goals, starts.shape).reshape(-1, 2)
        direct, _ = self._segments(direct_start, direct_goals, classify=False)
        direct_known = np.zeros(len(direct), dtype=bool)
        if details and np.any(direct):
            _, direct_known[direct] = self._segments(direct_start[direct], direct_goals[direct])
        direct_lengths = np.linalg.norm(direct_start-direct_goals, axis=1)
        result.flat[direct] = direct_lengths[direct]
        direct_matrix = direct.reshape(count, members)
        coords = np.asarray(self.nodes, dtype=float).reshape(-1, 2)
        vertices = coords[self._ids]
        for member in range(members):
            need = np.flatnonzero(~direct_matrix[:, member])
            if not len(need) or not len(vertices):
                continue
            goal = goals[member]
            key = tuple(goal)
            cached = self._goal_cache.get(key)
            if cached is None:
                goal_ok, _ = self._segments(np.broadcast_to(goal, vertices.shape), vertices, classify=False)
                seeds = self._ids[goal_ok]
                if len(seeds):
                    # One augmented-graph multi-source Dijkstra per goal.
                    lengths = np.linalg.norm(vertices[goal_ok]-goal, axis=1)
                    extra = csr_matrix((lengths, (np.zeros(len(seeds), dtype=int), seeds)), shape=(1, len(coords)))
                    from scipy.sparse import bmat
                    augmented = bmat([[self._graph, extra.T], [extra, None]], format='csr')
                    distances = dijkstra(augmented, directed=False, indices=len(coords))[:len(coords)]
                    possible = np.isfinite(distances[self._ids])
                    ids = self._ids[possible]
                    cached = (coords[ids], distances[ids])
                else:
                    cached = (np.empty((0, 2)), np.empty(0))
                if len(self._goal_cache) >= 64:
                    self._goal_cache.pop(next(iter(self._goal_cache)))
                self._goal_cache[key] = cached
            targets, graph_distances = cached
            if not len(targets):
                continue
            # Bound temporary matrices even for thousands of endpoints and
            # vertices. Sorted lower bounds allow an exact stop after the
            # first batch containing a visible vertex for each endpoint.
            chunk_size = max(1, min(256, 1_000_000 // len(targets)))
            for first in range(0, len(need), chunk_size):
                indices = need[first:first+chunk_size]
                points = starts[indices, member]
                lower = np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=2)
                lower += graph_distances[None, :]
                order = np.argsort(lower, axis=1)
                remaining = np.arange(len(indices))
                offset = 0
                batch_sizes = iter((1, 4, 16, 64))
                while offset < len(targets) and len(remaining):
                    batch_limit = next(batch_sizes, 128)
                    choices = order[remaining, offset:offset+batch_limit]
                    offset += batch_limit
                    batch_size = choices.shape[1]
                    batch_points = np.repeat(points[remaining], batch_size, axis=0)
                    batch_targets = targets[choices.reshape(-1)]
                    visible, _ = self._segments(batch_points, batch_targets, classify=False)
                    visible = visible.reshape(len(remaining), batch_size)
                    found = np.any(visible, axis=1)
                    if np.any(found):
                        rows = remaining[found]
                        first_visible = np.argmax(visible[found], axis=1)
                        target_ids = choices[found, first_visible]
                        result[indices[rows], member] = lower[rows, target_ids]
                    remaining = remaining[~found]
        if not details:
            self.last_query_status = None
            self.last_query_stats = {}
            return result
        status = np.full((count, members), 'missing', dtype='<U14')
        status[np.isfinite(result)] = 'graph_path'
        status.reshape(-1)[direct & ~direct_known] = 'direct_attempt'
        status.reshape(-1)[direct & direct_known] = 'direct_known'
        self.last_query_status = status
        self.last_query_stats = {'direct_known': int(np.count_nonzero(direct & direct_known)),
                                 'direct_attempt': int(np.count_nonzero(direct & ~direct_known)),
                                 'missing': int(np.count_nonzero(~np.isfinite(result))),
                                 'nodes': len(self._ids), 'edges': len(self._edges)}
        return result
