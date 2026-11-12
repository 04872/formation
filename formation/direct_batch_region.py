"""Direct uniform configuration certificates and lazy ordered edge search.

No Nearest/Steer. Pending certificates survive batches, but cannot be parents.
The exact priority uses free-goal-yaw XY distance, not a fixed goal yaw.
Every batch finishes its eligible queue, or pauses at an incumbent lower bound.
This is a finite-proposal experiment, with no BIT* completeness/AO claim.
"""
import heapq
import math
import time
from collections import defaultdict

import numpy as np

from formation.exit_compensated_rrtstar import ExitCompensatedRRTStar
from formation.region_coupling import _Program
from formation.types import Pose2D


class DirectBatchRegion(ExitCompensatedRRTStar):
    batch_size = 64
    work_per_batch = None
    tail_work = None

    @property
    def variant(self):
        return 'direct_batch_region'

    def _init_root(self):
        super()._init_root()
        self.regions = [self.cells[0]]
        self.active = {0: 0}  # certificate id -> reachable tree node
        self.node_region = [0]
        self.adjacency = defaultdict(set)
        self.bins = defaultdict(list)
        self.bin_size = self.config.max_extent
        self.bins[self._bin(self.cells[0])].append(0)
        self.queue, self.queued = [], {}
        self.serial = 0
        self.impossible = set()
        self.stats.update(batch_candidates=0, batch_attached=0, batch_pops=0,
            batch_pending=0, batch_queue=0, batch_stale=0, batch_reopened=0,
            batch_flushes=0, direct_invalid=0, direct_broad_pairs=0,
            direct_geometry_pairs=0, direct_potential_solves=0,
            direct_exact_pops=0, direct_strict_infeasible=0,
            direct_queue_max=0, direct_tail_pops=0, direct_bound_batch_stops=0)

    def _sample(self):
        # Deliberately uniform, including yaw: no inherited goal-bias atom.
        ox, oy = self.map_data.origin_xy
        return Pose2D(float(self.rng.uniform(ox, ox+self.map_data.width_m)),
                      float(self.rng.uniform(oy, oy+self.map_data.height_m)),
                      float(self.rng.uniform(-math.pi, math.pi)))

    def _bin(self, cell):
        return (math.floor(cell.pose.x/self.bin_size), math.floor(cell.pose.y/self.bin_size))

    def _goal_lower(self, cell):
        return max(0., math.hypot(cell.pose.x-self.goal_xy[0],
                                 cell.pose.y-self.goal_xy[1])-cell.guard)

    def _signature(self, parent, rid):
        node = self.active.get(rid)
        return (self.version[parent], None if node is None else
                (self.version[node], self.exit_version[node], self.parent[node]))

    def _push(self, parent, rid):
        if self.node_region[parent] == rid or self.active.get(rid) == 0:
            return
        if tuple(sorted((self.node_region[parent], rid))) in self.impossible:
            return
        token = parent, rid
        signature = self._signature(parent, rid)
        if token in self.queued and self.queued[token][1] == signature:
            return
        key = (self.cost[parent]+self._bound(self.entry[parent], self.cells[parent], self.regions[rid])
               +self._goal_lower(self.regions[rid]))
        self._enqueue(key, parent, rid, signature, False)

    def _enqueue(self, key, parent, rid, signature, exact):
        self.serial += 1
        self.queued[parent, rid] = self.serial, signature
        heapq.heappush(self.queue, (key, self.serial, parent, rid, signature, exact))
        self.stats['direct_queue_max'] = max(self.stats['direct_queue_max'], len(self.queue))

    def _add_candidate(self, cell):
        rid = len(self.regions)
        self.regions.append(cell)
        bx, by = self._bin(cell)
        # Every guard <= max_extent: +/-2 bins includes every possible XY overlap.
        for x in range(bx-2, bx+3):
            for y in range(by-2, by+3):
                for other in self.bins.get((x, y), ()):
                    c = self.regions[other]
                    if (math.hypot(cell.pose.x-c.pose.x, cell.pose.y-c.pose.y) >= cell.guard+c.guard
                        or self.rho*abs((cell.pose.yaw-c.pose.yaw+math.pi)%(2*math.pi)-math.pi)
                           >= self.rho*(cell.yaw_limit+c.yaw_limit)):
                        continue
                    self.stats['direct_broad_pairs'] += 1
                    if self._separated(cell, c):
                        continue
                    self.stats['direct_geometry_pairs'] += 1
                    self.adjacency[rid].add(other)
                    self.adjacency[other].add(rid)
                    if other in self.active:
                        self._push(self.active[other], rid)
        self.bins[bx, by].append(rid)
        self.stats['batch_candidates'] += 1

    def _potential(self, parent, rid):
        """min_z d_W(e_parent,z) + ||z_xy-goal_xy||, strict portal SOCP."""
        first, second = self.cells[parent], self.regions[rid]
        anchor = first.pose.yaw
        program = _Program(5)
        margin = self.config.overlap_margin
        program.cell(first, self.geo.lift(first.pose, anchor), 0, margin)
        program.cell(second, self.geo.lift(second.pose, anchor), 0, margin)
        program.distance(3, 0, self.geo.lift(self.entry[parent], anchor))
        a = np.zeros((3, 5)); a[0, 4] = -1.; a[1, 0] = -1.; a[2, 1] = -1.
        program.cone(a, np.array((0., -self.goal_xy[0], -self.goal_xy[1])))
        self.stats['direct_potential_solves'] += 1
        self.geo.solves += 1
        try:
            result, status = program.solve(np.array((0., 0., 0., 1., 1.)), self.geo.settings)
        except Exception:
            self.geo.failures += 1
            return None
        if result is None:
            if status == 'infeasible':
                self.impossible.add(tuple(sorted((self.node_region[parent], rid))))
                self.stats['direct_strict_infeasible'] += 1
            else:
                self.geo.failures += 1
            return None
        pose = self.geo.pose(result[:3])
        if min(self.geo.slack(first, pose), self.geo.slack(second, pose)) < margin-1e-7:
            self.geo.failures += 1
            return None
        return self.cost[parent]+float(result[3]+result[4])

    def _reopen_incoming(self, node):
        rid = self.node_region[node]
        for other in self.adjacency[rid]:
            if other in self.active:
                self._push(self.active[other], rid)

    def _attach_pending(self, parent, rid):
        self.stats['choose_parent_tests'] += 1
        cell = self.regions[rid]
        z = self._attach(self.entry[parent], self.cells[parent], cell)
        if z is None or min(self.geo.slack(self.cells[parent], z), self.geo.slack(cell, z)) < self.config.overlap_margin*(1-1e-6)-1e-8:
            return
        idx = self._add_cell(cell, None)
        self.active[rid] = idx; self.node_region.append(rid)
        self.parent.append(parent); self.entry.append(z)
        self.cost.append(self.cost[parent]+self._distance_in(parent, self.entry[parent], z))
        self.version.append(0); self.parent_version.append(self.version[parent])
        self.children.append(set()); self.geometry_version.append(0)
        self.exit_version.append(0); self.local_signature.append(None)
        self.children[parent].add(idx); self._changed_exits(parent)
        self._reopen_incoming(parent)
        self.stats['children_links'] = len(self.parent)-1
        self.stats['children_max'] = max(self.stats['children_max'], len(self.children[parent]))
        self.stats['batch_attached'] += 1
        self._register_goal(idx)
        # A formerly pending certificate becomes a parent only here.
        for other in self.adjacency[rid]:
            self._push(idx, other)
            if other in self.active:
                self._push(self.active[other], rid)

    def _commit(self, node, parent, pose, candidate, outlets, kind):
        old_parent = self.parent[node]
        descendants = []; stack = [node]
        while stack:
            i = stack.pop(); descendants.append(i); stack.extend(self.children[i])
        super()._commit(node, parent, pose, candidate, outlets, kind)
        # Eager g propagation in the base also changes every descendant's queue keys.
        for i in descendants:
            if i != node: self.version[i] += 1
            for rid in self.adjacency[self.node_region[i]]:
                self._push(i, rid); self.stats['batch_reopened'] += 1
            self._reopen_incoming(i)
        for i in {old_parent, parent}:
            self._reopen_incoming(i)

    def flush(self, tail=False):
        self.stats['batch_flushes'] += 1
        # Finish current eligible opportunities. An incumbent lower-bound stop
        # retains the entire queue and every certificate for subsequent batches.
        reported = time.perf_counter()
        while self.queue:
            incumbent = min(self.goal_cost_cache.values(), default=math.inf)
            if self.queue[0][0] >= incumbent-self.config.improvement_tolerance:
                self.stats['direct_bound_batch_stops'] += 1
                break
            if time.perf_counter()-reported > 30.:
                print('direct batch progress', self.stats['sample_calls'],
                      'active',len(self.cells),'pending',len(self.regions)-len(self.cells),
                      'queue',len(self.queue),'J',incumbent,'pops',self.stats['batch_pops'], flush=True)
                reported = time.perf_counter()
            key, serial, parent, rid, signature, exact = heapq.heappop(self.queue)
            self.stats['batch_pops'] += 1
            self.stats['direct_tail_pops'] += int(tail)
            token = parent, rid
            if self.queued.get(token) != (serial, signature):
                self.stats['batch_stale'] += 1; continue
            del self.queued[token]
            if signature != self._signature(parent, rid):
                self.stats['batch_stale'] += 1; self._push(parent, rid); continue
            node = self.active.get(rid)
            if node is not None:
                if self._ancestor(node, parent): continue
                outlets = self._exits(node)
                if not self._can_improve(node, parent, outlets):
                    self.stats['rewire_lb_skipped'] += 1; continue
            if not exact:
                value = self._potential(parent, rid)
                if value is not None: self._enqueue(value, parent, rid, signature, True)
                continue
            self.stats['direct_exact_pops'] += 1
            if node is None:
                self._attach_pending(parent, rid)
                continue
            self.stats['rewire_tests'] += 1
            z = self._attach(self.entry[parent], self.cells[parent], self.cells[node])
            accepted = self._candidate(node, parent, z, outlets) if z is not None else None
            if accepted is None and outlets:
                z = self._protected_entry(node, parent, outlets, 'rewire')
                accepted = self._candidate(node, parent, z, outlets) if z is not None else None
            if accepted is not None: self._commit(node, parent, z, accepted, outlets, 'rewire')
        self.stats['batch_pending'] = len(self.regions)-len(self.cells)
        self.stats['batch_queue'] = len(self.queue)

    def step(self):
        q = self._sample(); self.stats['sample_calls'] += 1
        cell = self.model.make_cell(q)
        if cell.valid: self._add_candidate(cell)
        else: self.stats['direct_invalid'] += 1
        if self.stats['sample_calls'] % self.batch_size == 0: self.flush()
        self.stats['batch_pending'] = len(self.regions)-len(self.cells)
        self.stats['batch_queue'] = len(self.queue)
