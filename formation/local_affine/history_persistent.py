"""Budgeted persistent directed Bellman backups for affine branch selection.

Only certified segments are usable actions. Nearby pairs are a finite,
explicit set of possible actions; lack of a certificate is not proof of
infeasibility and is reconsidered when new regions become available.
"""
from collections import defaultdict, deque
import heapq
import time

import numpy as np
from scipy.spatial import cKDTree

from .geometry import robot_positions
from .history_graph import HistoryGraph
from .history_joint_geometry import evaluate_pairs


class PersistentHistoryDecision:
    def __init__(self, reference, epsilon=.4, rho=.4, backup_budget=256,
                 link_budget=128, max_neighbors=12, max_vertices=5000):
        # rho is accepted for compatibility with the finite-depth decision
        # constructor; the section-8 persistent action model does not use it.
        self.reference = np.asarray(reference, dtype=float)
        self.epsilon, self.rho = float(epsilon), float(rho)
        self.backup_budget, self.link_budget = int(backup_budget), int(link_budget)
        self.max_neighbors, self.max_vertices = int(max_neighbors), int(max_vertices)
        if min(self.epsilon, self.rho, self.backup_budget, self.link_budget,
               self.max_neighbors, self.max_vertices) <= 0:
            raise ValueError('thresholds, budgets and limits must be positive')
        self.heuristic = HistoryGraph()
        self.round_id = 0
        self.regions, self._region_keys = [], set()
        self.known = self.occupied = None
        self._reset_graph()

    def _reset_graph(self):
        self.qs, self.ids, self.h, self.labels, self.last_written = [], {}, [], [], []
        self.edges = defaultdict(dict)  # source -> target -> (cost, source)
        self.pending = defaultdict(dict)  # source -> target -> distance lower bound
        self.incoming = defaultdict(set)
        self.dependents = defaultdict(set)
        self.witness = {}
        self.queue, self.queued = deque(), set()
        self.goal_id = None
        self._last_world = None
        self._events = []
        self._attempted = {}
        self._too_expensive = {}
        self.first_seen, self.write_round, self.provenance = [], [], []
        self._cert_version = 0
        self._h_known = self._h_occupied = self._h_goal = None
        self._h_graph_values = np.empty((0, len(self.reference)))

    def _enqueue(self, x):
        if x not in self.queued and x != self.goal_id:
            self.queued.add(x)
            self.queue.append(x)

    def _invalidate(self, starts, exclude=()):
        stack, seen = list(starts), set(exclude)
        while stack:
            x = stack.pop()
            if x in seen or x == self.goal_id:
                continue
            seen.add(x)
            stack.extend(self.dependents.get(x, ()))
            # A predecessor that did not choose x can still become stale if
            # x's fallback decreased enough to change its minimizer.
            fallback = self.h[x]
            if np.isfinite(fallback):
                for p in self.incoming.get(x, ()):
                    if p in seen or self.labels[p] is None:
                        continue
                    edge = self.edges[p].get(x)
                    bound = edge[0] if edge is not None else self.pending[p].get(x)
                    if bound is not None and bound + fallback < self.labels[p] - 1e-10:
                        stack.append(p)
            old = self.witness.pop(x, None)
            if old is not None:
                self.dependents[old].discard(x)
            self.labels[x] = None
            self._enqueue(x)
        return len(seen)

    def add_vertex(self, q, h=None):
        q = np.asarray(q, dtype=float).reshape(6)
        key = q.tobytes()
        if key in self.ids:
            x = self.ids[key]
            if h is not None:
                self.h[x] = float(h)
            return x
        if len(self.qs) >= self.max_vertices:
            return None
        x = len(self.qs)
        self.ids[key] = x
        self.qs.append(q.copy())
        self.h.append(float(h) if h is not None else float('nan'))
        self.labels.append(None)
        self.last_written.append(None)
        self.first_seen.append(self.round_id)
        self.write_round.append(None)
        self.provenance.append(None)
        return x

    def set_goal(self, x):
        self.goal_id = x
        self.h[x] = 0.
        self.labels[x] = 0.

    def value(self, x):
        return float(self.labels[x] if self.labels[x] is not None else self.h[x])

    def _consider_action(self, a, b, lower_bound, witness_changed=False):
        """Invalidate only if a changed action can invalidate the minimum."""
        if self.labels[a] is None:
            self._enqueue(a)
            return
        successor = self.value(b)
        if (witness_changed or not np.isfinite(successor) or
                lower_bound + successor < self.labels[a] - 1e-10):
            self._invalidate([a])

    def add_edge(self, a, b, cost, source='search'):
        if a is None or b is None or a == b:
            return False
        cost = float(cost)
        if not np.isfinite(cost) or cost < 0:
            raise ValueError('edge cost must be finite and nonnegative')
        old = self.edges[a].get(b)
        was_pending = self.pending[a].pop(b, None) is not None
        if old is not None and old[0] == cost:
            if was_pending and self.labels[a] is None:
                self._enqueue(a)
            return False
        self.edges[a][b] = (cost, source)
        self.incoming[b].add(a)
        self._consider_action(a, b, cost,
                              witness_changed=(old is not None and self.witness.get(a) == b))
        return True

    def add_pending(self, a, b, lower_bound):
        if (a is None or b is None or a == b or b in self.edges[a] or
                self._too_expensive.get((a, b)) == self._cert_version):
            return False
        if b not in self.pending[a]:
            self.pending[a][b] = float(lower_bound)
            self.incoming[b].add(a)
            self._consider_action(a, b, float(lower_bound))
            return True
        return False

    def _cycle(self, x, successor):
        seen = {x}
        while successor in self.witness:
            if successor in seen:
                return True
            seen.add(successor)
            successor = self.witness[successor]
        return successor in seen

    def _resolve_cycle(self, x, successor):
        # Solve the dependency ring as a small simultaneous shortest-path
        # problem with fixed outside labels.  Require every pending bound to
        # be no better before committing any member of the ring.
        ring = [x]
        z = successor
        while z != x:
            if z in ring or z not in self.witness:
                return None
            ring.append(z)
            z = self.witness[z]
        members = set(ring)
        distance = {a: np.inf for a in ring}
        choice = {}
        reverse = defaultdict(list)
        heap = []
        for a in ring:
            for b, (cost, source) in self.edges[a].items():
                if b in members:
                    reverse[b].append((a, cost, source))
                    continue
                # Outside labels depending on this ring do not form exits.
                v, seen = b, set()
                while v in self.witness and v not in seen and v not in members:
                    seen.add(v)
                    v = self.witness[v]
                if v in members or v in seen:
                    return None
                val = cost + self.value(b)
                if np.isfinite(val) and val < distance[a]:
                    distance[a], choice[a] = val, (b, cost, source)
                    heapq.heappush(heap, (val, a))
        while heap:
            val, b = heapq.heappop(heap)
            if val != distance[b]:
                continue
            for a, cost, source in reverse[b]:
                candidate = val + cost
                if candidate < distance[a] - 1e-12:
                    distance[a], choice[a] = candidate, (b, cost, source)
                    heapq.heappush(heap, (candidate, a))
        if any(not np.isfinite(distance[a]) for a in ring):
            return None
        for a in ring:
            for b, lower in self.pending[a].items():
                target = distance[b] if b in members else self.value(b)
                if lower + target < distance[a] - 1e-10:
                    return None
        events = []
        for a in sorted(ring, key=lambda n: distance[n]):
            b, cost, source = choice[a]
            previous = self.last_written[a]
            old_witness = self.witness.get(a)
            if old_witness is not None:
                self.dependents[old_witness].discard(a)
            successor_source = 'goal' if b == self.goal_id else ('learned' if self.labels[b] is not None else 'initial')
            self.labels[a] = self.last_written[a] = float(distance[a])
            self.write_round[a] = self.round_id
            self.provenance[a] = dict(successor=b, edge_source=source,
                                      successor_source=successor_source,
                                      successor_write_round=self.write_round[b])
            self.witness[a] = b
            self.dependents[b].add(a)
            events.append(dict(vertex=a, value=float(distance[a]), previous=previous,
                               successor=b, edge_cost=cost, source=source))
        for a in ring:
            for p in self.incoming.get(a, ()):
                if p not in members:
                    self._invalidate([p], exclude=members)
        return events

    def backup_vertex(self, x):
        """Evaluate fixed successor labels; return bounds and writeback evidence."""
        if x == self.goal_id:
            return {'vertex': x, 'status': 'goal', 'L': 0., 'U': 0.}
        options = [(cost + self.value(z), z, cost, source)
                   for z, (cost, source) in self.edges[x].items()]
        options = [o for o in options if np.isfinite(o[0])]
        options.sort(key=lambda o: (o[0], o[1]))
        upper = options[0][0] if options else np.inf
        pending = [(d + self.value(z), z) for z, d in self.pending[x].items()
                   if np.isfinite(self.value(z))]
        lower = min([upper] + [o[0] for o in pending])
        result = {'vertex': x, 'L': float(lower) if np.isfinite(lower) else None,
                  'U': float(upper) if np.isfinite(upper) else None,
                  'pending': len(pending), 'status': 'unresolved'}
        if not np.isfinite(upper) or lower < upper - 1e-10:
            return result
        _, z, cost, source = options[0]
        if self._cycle(x, z):
            resolved = self._resolve_cycle(x, z)
            result['status'] = 'cycle_resolved' if resolved is not None else 'cyclic_dependency'
            if resolved is not None:
                result['cycle_writes'] = resolved
            return result
        old = self.last_written[x]
        old_effective = self.value(x)
        previous = self.witness.get(x)
        if previous is not None:
            self.dependents[previous].discard(x)
        successor_source = 'goal' if z == self.goal_id else ('learned' if self.labels[z] is not None else 'initial')
        successor_write_round = self.write_round[z]
        self.labels[x], self.witness[x] = float(upper), z
        self.last_written[x] = float(upper)
        self.write_round[x] = self.round_id
        self.provenance[x] = dict(successor=z, edge_source=source,
                                  successor_source=successor_source,
                                  successor_write_round=successor_write_round)
        self.dependents[z].add(x)
        result.update(status='written', value=float(upper), successor=int(z),
                      edge_cost=float(cost), source=source, previous=old,
                      successor_source=successor_source,
                      successor_write_round=successor_write_round)
        if not np.isclose(old_effective, upper, rtol=1e-12, atol=1e-12):
            # Any predecessor can switch its minimizing successor, even when
            # it did not previously depend on this particular successor.
            ancestors = {x}
            ancestor = z
            while ancestor not in ancestors:
                ancestors.add(ancestor)
                if ancestor not in self.witness:
                    break
                ancestor = self.witness[ancestor]
            affected = []
            for p in self.incoming.get(x, ()):
                if p in ancestors:
                    continue
                # A changed non-winning option only invalidates a committed
                # value if it can now undercut that value.
                edge = self.edges[p].get(x)
                lower = self.pending[p].get(x)
                bound = edge[0] if edge is not None else lower
                if (self.labels[p] is not None and
                    (self.witness.get(p) == x or
                     (bound is not None and bound + upper < self.labels[p] - 1e-10))):
                    affected.append(p)
                elif self.labels[p] is None:
                    self._enqueue(p)
            result['propagated'] = self._invalidate(affected, exclude=ancestors)

        return result

    def process_backups(self, budget=None):
        results = []
        for _ in range(self.backup_budget if budget is None else int(budget)):
            if not self.queue:
                break
            x = self.queue.popleft()
            if x not in self.queued:
                continue
            self.queued.discard(x)
            results.append(self.backup_vertex(x))
        return results

    def _update_map(self, world, regions):
        known = np.asarray(world.known, dtype=bool)
        occupied = np.asarray(world.occupied, dtype=bool)
        reset = (self.known is not None and
                 (self.known.shape != known.shape or np.any(self.known & ~known) or
                  np.any(self.known & (self.occupied != occupied))))
        changed = (self.known is None or reset or not np.array_equal(self.known, known) or
                   not np.array_equal(self.occupied, occupied))
        if reset:
            self._reset_graph()
            self.regions, self._region_keys = [], set()
        self.known, self.occupied = known.copy(), occupied.copy()
        regions_added = False
        for h, b in regions:
            key = (np.asarray(h).tobytes(), np.asarray(b).tobytes())
            if key not in self._region_keys:
                self._region_keys.add(key)
                self.regions.append((np.asarray(h).copy(), np.asarray(b).copy()))
                regions_added = True
        if changed or regions_added:
            self._cert_version += 1
        if changed and self.qs and self.edges:
            # Invalidate only edge costs that actually changed. Unrelated new
            # observations must not erase learned values across the graph.
            pairs = [(a, b) for a, targets in self.edges.items() for b in targets]
            _, costs = evaluate_pairs(self.qs, pairs, self.reference, world,
                                      self.regions, certify=False)
            affected = set()
            for (a, b), cost in zip(pairs, costs):
                old, source = self.edges[a][b]
                if not np.isclose(old, cost, rtol=1e-12, atol=1e-12):
                    self.edges[a][b] = (float(cost), source)
                    affected.add(a)
            self._invalidate(affected)
        self.heuristic.update(world)
        self._last_world = world
        return reset, changed, regions_added

    def _heuristics(self, goal):
        if not self.qs:
            return False
        pos = robot_positions(np.asarray(self.qs), self.reference)
        targets = robot_positions(np.asarray(goal), self.reference)
        unchanged = (self._h_known is not None and
                     np.array_equal(self._h_known, self.known) and
                     np.array_equal(self._h_occupied, self.occupied) and
                     np.array_equal(self._h_goal, targets) and
                     len(self._h_graph_values) <= len(pos))
        start = len(self._h_graph_values) if unchanged else 0
        new = (np.asarray(self.heuristic.distances(pos[start:], targets, details=False))
               if start < len(pos) else np.empty((0, len(self.reference))))
        distances = np.concatenate((self._h_graph_values[:start], new), axis=0)
        self._h_graph_values = distances.copy()
        self._h_known, self._h_occupied = self.known.copy(), self.occupied.copy()
        self._h_goal = targets.copy()
        fallback = not np.all(np.isfinite(distances))
        if fallback:
            distances = np.linalg.norm(pos-targets[None, :, :], axis=2)
        values = np.linalg.norm(distances, axis=1)/np.sqrt(len(self.reference))
        changed = []
        for i, value in enumerate(values):
            value = 0. if i == self.goal_id else float(value)
            old = self.h[i]
            if (self.labels[i] is None and
                    (not np.isfinite(old) or not np.isclose(old, value, rtol=1e-12, atol=1e-12))):
                changed.append(i)
            self.h[i] = value
        # Unknown H was treated as potentially competitive when the action was
        # inserted. Once initialized, revisit all its incoming actions too.
        for i in changed:
            for parent in tuple(self.incoming.get(i, ())):
                edge = self.edges[parent].get(i)
                lower = edge[0] if edge is not None else self.pending[parent].get(i)
                if lower is not None:
                    self._consider_action(parent, i, lower,
                                          witness_changed=self.witness.get(parent) == i)
        return fallback

    def _discover(self, ids):
        if len(self.qs) < 2:
            return
        pos = robot_positions(np.asarray(self.qs), self.reference)
        emb = pos.reshape(len(pos), -1)/np.sqrt(len(self.reference))
        tree = cKDTree(emb)
        # k-nearest with a distance cap avoids materializing dense radius pairs.
        k = min(len(self.qs), self.max_neighbors + 1)
        ds, near = tree.query(emb[ids], k=k, distance_upper_bound=self.epsilon)
        ds, near = np.asarray(ds).reshape(len(ids), k), np.asarray(near).reshape(len(ids), k)
        for a, distances, neighbors in zip(ids, ds, near):
            for d, b in zip(distances, neighbors):
                if np.isfinite(d) and b < len(self.qs) and a != b:
                    self.add_pending(a, int(b), d)
                    self.add_pending(int(b), a, d)

    def _certify_pending(self, world):
        # Half the checks follow the current competitive endpoints; the
        # remainder prevents old dependencies from starving indefinitely.
        priority = set(getattr(self, '_priority_ids', ()))
        prioritized, other = [], []
        rank = {x: i for i, x in enumerate(getattr(self, '_priority_ids', ()))}
        for a, targets in self.pending.items():
            for b, d in targets.items():
                if self._attempted.get((a, b)) == self._cert_version:
                    continue
                item = (d + self.value(b), a, b)
                (prioritized if a in priority else other).append(item)
        # Round-robin endpoints so one dense neighbourhood cannot consume the
        # dedicated budget before other contenders receive a single check.
        prioritized.sort(key=lambda t: (rank[t[1]], t[0], t[2]))
        groups = defaultdict(deque)
        for item in prioritized:
            groups[item[1]].append(item)
        first = []
        while len(first) < self.link_budget // 2 and groups:
            for a in list(groups):
                if len(first) >= self.link_budget // 2:
                    break
                first.append(groups[a].popleft())
                if not groups[a]:
                    del groups[a]
        rest = other + [v for group in groups.values() for v in group]
        checks = first + heapq.nsmallest(self.link_budget - len(first), rest)
        if not checks:
            return 0, 0
        pairs = [(a, b) for _, a, b in checks]
        safe, costs = evaluate_pairs(self.qs, pairs, self.reference, world, self.regions)
        accepted = 0
        for (_, a, b), ok, cost in zip(checks, safe, costs):
            self._attempted[a, b] = self._cert_version
            if ok and cost <= self.epsilon:
                accepted += bool(self.add_edge(a, b, cost, 'short_link'))
            elif ok:
                # It is outside the finite short-link action model at this
                # map version; reconsider if the clearance cost changes.
                self._too_expensive[a, b] = self._cert_version
                self.pending[a].pop(b, None)
                if b not in self.edges[a]:
                    self.incoming[b].discard(a)
                self._enqueue(a)
        return len(checks), accepted

    def _ingest(self, nodes, routes, stats, world):
        relevant = sorted({i for route in routes for i in route})
        for a, b in stats.get('certified_edges', []):
            relevant.extend((int(a), int(b)))
        relevant = sorted(set(relevant))
        mapping = {i: self.add_vertex(nodes[i].q) for i in relevant}
        pairs = set()
        for route in routes:
            pairs.update(zip(route[:-1], route[1:]))
        pairs.update(tuple(p) for p in stats.get('certified_edges', []))
        pairs = [(mapping[a], mapping[b]) for a, b in sorted(pairs)
                 if mapping[a] is not None and mapping[b] is not None and mapping[a] != mapping[b]]
        if pairs:
            _, costs = evaluate_pairs(self.qs, pairs, self.reference, world, self.regions, certify=False)
            for (a, b), cost in zip(pairs, costs):
                self.add_edge(a, b, cost, 'search')
        return mapping

    def select(self, nodes, stats, goal, world, regions):
        start = time.perf_counter()
        reset, changed, added = self._update_map(world, regions)
        routes = sorted(stats.get('candidate_full_routes', []), key=lambda p: p[-1])
        goal_changed = (self.goal_id is not None and
                        not np.array_equal(self.qs[self.goal_id], np.asarray(goal)))
        if goal_changed:
            self._reset_graph()
            self._last_world = world
        current_keys = {np.asarray(nodes[i].q, dtype=float).reshape(6).tobytes()
                        for route in routes for i in route}
        current_keys.update(np.asarray(nodes[i].q, dtype=float).reshape(6).tobytes()
                            for edge in stats.get('certified_edges', []) for i in edge)
        # Bounded generations, never permanently freeze out future trees.
        capacity_reset = len(self.qs) + len(current_keys - self.ids.keys()) + 1 > self.max_vertices
        if capacity_reset:
            self._reset_graph()
            self._last_world = world
        mapping = self._ingest(nodes, routes, stats, world)
        goal_id = self.add_vertex(goal)
        if goal_id is not None:
            self.set_goal(goal_id)
        fallback = self._heuristics(goal)
        eligible = [x for x in mapping.values() if x is not None]
        if eligible:
            self._discover(sorted(set(eligible)))
        ordered_routes = sorted(routes, key=lambda r: (float(nodes[r[-1]].cost) +
                                (self.h[mapping[r[-1]]] if mapping[r[-1]] is not None else np.inf), r[-1]))
        self._priority_ids = list(dict.fromkeys(mapping[r[-1]] for r in ordered_routes
                                               if mapping[r[-1]] is not None))
        checked, accepted = self._certify_pending(world)
        # Current endpoints receive dedicated compute before FIFO propagation;
        # remaining budget advances previous rounds' unresolved predecessors.
        priority = self._priority_ids
        updates = []
        for x in priority[:max(1, self.backup_budget // 2)]:
            self.queued.discard(x)
            updates.append(self.backup_vertex(x))
        updates.extend(self.process_backups(max(0, self.backup_budget - len(updates))))
        candidates = []
        for route in routes:
            i = route[-1]
            x = mapping[i]
            if x is None:
                p = robot_positions(np.asarray([nodes[i].q]), self.reference)
                g = robot_positions(np.asarray(goal), self.reference)
                d = np.asarray(self.heuristic.distances(p, g, details=False))
                if fallback or not np.all(np.isfinite(d)):
                    d = np.linalg.norm(p-g[None, :, :], axis=2)
                h = float(np.linalg.norm(d)/np.sqrt(len(self.reference)))
                value, source = h, 'initial_capacity'
            else:
                h, value = self.h[x], self.value(x)
                source = 'learned' if self.labels[x] is not None else 'initial'
            cost = float(nodes[i].cost)
            candidates.append(dict(endpoint=int(i), route=list(route), vertex=x,
                                   tree_cost=cost, h=float(h), value=float(value),
                                   value_source=source, original_score=cost+h,
                                   first_seen_round=self.first_seen[x] if x is not None else None,
                                   write_round=self.write_round[x] if x is not None and self.labels[x] is not None else None,
                                   provenance=self.provenance[x] if x is not None and self.labels[x] is not None else None,
                                   score=cost+value))
        baseline = stats.get('baseline_route', [0])
        if candidates:
            best = candidates[0]
            for c in candidates[1:]:
                if c['score'] < best['score'] or (c['score'] == best['score'] and
                        len(c['route']) > len(best['route']) and
                        c['route'][:len(best['route'])] == best['route']):
                    best = c
            original = candidates[0]
            for c in candidates[1:]:
                if (c['original_score'] < original['original_score'] or
                        (c['original_score'] == original['original_score'] and
                         len(c['route']) > len(original['route']) and
                         c['route'][:len(original['route'])] == original['route'])):
                    original = c
            selected = best['route']
        else:
            selected, best, original = list(baseline), None, None
        evidence = dict(status='history_persistent' if candidates else 'no_candidates',
                        round=self.round_id, candidates=candidates, candidate_count=len(candidates),
                        selected_route=selected, selected_endpoint=int(selected[-1]),
                        baseline_endpoint=original['endpoint'] if original else int(selected[-1]),
                        vertices=len(self.qs), edges=sum(map(len, self.edges.values())),
                        pending=sum(map(len, self.pending.values())), queue=len(self.queue),
                        checks=checked, accepted_links=accepted, backups=updates,
                        map_reset=bool(reset), goal_reset=bool(goal_changed),
                        capacity_reset=bool(capacity_reset),
                        map_changed=bool(changed), regions_added=bool(added),
                        heuristic_fallback=bool(fallback), history_regions=len(self.regions),
                        selection_seconds=time.perf_counter()-start)
        self.round_id += 1
        return selected, evidence

    def record_goal(self, nodes, route, goal):
        if not route or self._last_world is None:
            return
        ids = [self.add_vertex(nodes[j].q) for j in route]
        g = self.add_vertex(goal)
        if g is not None:
            self.set_goal(g)
        ids.append(g)
        pairs = [(a, b) for a, b in zip(ids[:-1], ids[1:]) if a is not None and b is not None and a != b]
        if pairs:
            _, costs = evaluate_pairs(self.qs, pairs, self.reference, self._last_world, self.regions, certify=False)
            for (a, b), cost in zip(pairs, costs):
                self.add_edge(a, b, cost, 'goal_route')

    def record_direct(self, root, goal, world, regions):
        reset, _, _ = self._update_map(world, regions)
        a, b = self.add_vertex(root), self.add_vertex(goal)
        if b is not None:
            self.set_goal(b)
        if a is not None and b is not None and a != b:
            _, costs = evaluate_pairs(self.qs, [(a, b)], self.reference, world, self.regions, certify=False)
            self.add_edge(a, b, costs[0], 'direct_goal')
        result = dict(status='direct_goal', round=self.round_id, selected_route=[0, -1],
                      history_regions=len(self.regions), map_reset=bool(reset))
        self.round_id += 1
        return result

    def record_no_root(self, world, regions):
        reset, _, _ = self._update_map(world, regions)
        result = dict(status='no_safe_nominal_root', round=self.round_id,
                      candidate_count=0, candidates=[], history_regions=len(self.regions), map_reset=bool(reset))
        self.round_id += 1
        return result
