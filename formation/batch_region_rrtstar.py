"""Bounded-work, cost-ordered region-tree experiment (no BIT* guarantees)."""
import heapq
import math
import time
from collections import defaultdict

from formation.exit_compensated_rrtstar import ExitCompensatedRRTStar


class BatchRegionRRTStar(ExitCompensatedRRTStar):
    batch_size = 32
    work_per_batch = 128

    @property
    def variant(self):
        return 'batch_region'

    def _init_root(self):
        super()._init_root()
        self.pending = {}
        self.next_pending = 0
        self.queue = []
        self.queued = {}
        self.edges = defaultdict(set)
        self.serial = 0
        self.stats.update(batch_candidates=0, batch_attached=0, batch_pops=0,
                          batch_pending=0, batch_queue=0, batch_stale=0,
                          batch_reopened=0, batch_flushes=0)

    def _goal_lower(self, cell):
        return max(0., math.hypot(cell.pose.x-self.goal_xy[0], cell.pose.y-self.goal_xy[1])-cell.guard)

    def _key(self, parent, cell):
        return self.cost[parent]+self._bound(self.entry[parent], self.cells[parent], cell)+self._goal_lower(cell)

    def _push(self, parent, target):
        if target[0] == 'p':
            if target[1] not in self.pending: return
            cell = self.pending[target[1]][0]
        else:
            if parent == target[1] or target[1] == 0: return
            cell = self.cells[target[1]]
        token = (parent, target)
        key = self._key(parent, cell)
        if token in self.queued and self.queued[token][0] <= key + 1e-12: return
        self.serial += 1
        heapq.heappush(self.queue, (key, self.serial, parent, target,
                                    self.version[parent], self.cost[parent]))
        self.queued[token] = (key, self.serial)

    def _adjacent_active(self, cell, discovery):
        # Same max-neighbour geometry filter as the original region tree.
        candidates = []
        for i, other in enumerate(self.cells):
            if i == discovery: continue
            dx, dy = cell.pose.x-other.pose.x, cell.pose.y-other.pose.y
            dyaw = self.rho*abs((cell.pose.yaw-other.pose.yaw+math.pi)%(2*math.pi)-math.pi)
            if math.hypot(dx,dy) < cell.guard+other.guard and dyaw < cell.rho*cell.yaw_limit+other.rho*other.yaw_limit:
                candidates.append((dx*dx+dy*dy+dyaw*dyaw, i))
        return [i for _,i in sorted(candidates)[:self.config.max_neighbors]
                if not self._separated(self.cells[i], cell)]

    def _add_pending(self, cell, parent, portal):
        ident = self.next_pending
        self.next_pending += 1
        self.pending[ident] = (cell, parent, portal)
        self.stats['batch_candidates'] += 1
        for p in [parent]+self._adjacent_active(cell,parent):
            self.edges[p].add(('p',ident))
            self._push(p,('p',ident))

    def _attach_pending(self, parent, ident):
        item = self.pending.get(ident)
        if item is None: return
        cell, discovery, portal = item
        self.stats['choose_parent_tests'] += 1
        z = self._attach(self.entry[parent], self.cells[parent], cell)
        # Discovery is a certified fallback if the numerical projection fails.
        if z is None and parent == discovery: z = portal
        if z is None or min(self.geo.slack(self.cells[parent], z), self.geo.slack(cell,z)) < self.config.overlap_margin*(1-1e-6)-1e-8: return
        del self.pending[ident]
        idx = self._add_cell(cell, discovery)
        self.parent.append(parent); self.entry.append(z)
        self.cost.append(self.cost[parent]+self._distance_in(parent,self.entry[parent],z))
        self.version.append(0); self.parent_version.append(self.version[parent])
        self.children.append(set()); self.geometry_version.append(0)
        self.exit_version.append(0); self.local_signature.append(None)
        self.children[parent].add(idx); self._changed_exits(parent)
        self._reopen_incoming(parent)
        self.stats['children_links'] = len(self.parent)-1
        self.stats['children_max'] = max(self.stats['children_max'],len(self.children[parent]))
        self.stats['batch_attached'] += 1
        self._register_goal(idx)
        # Existing queued alternatives become rewire opportunities.
        for p, targets in list(self.edges.items()):
            if ('p',ident) in targets:
                targets.remove(('p',ident)); targets.add(('n',idx))
                if p != parent: self._push(p,('n',idx))
                self.edges[idx].add(('n',p)); self._push(idx,('n',p))
        # New active region may connect geometries created in earlier batches.
        for other,(candidate,_,_) in list(self.pending.items()):
            if not self._separated(cell,candidate):
                self.edges[idx].add(('p',other)); self._push(idx,('p',other))

    def _reopen_incoming(self, node):
        for p, targets in self.edges.items():
            if ('n',node) in targets: self._push(p,('n',node))

    def _commit(self, node, parent, pose, candidate, outlets, kind):
        old_parent = self.parent[node]
        descendants = []
        stack = [node]
        while stack:
            i = stack.pop(); descendants.append(i); stack.extend(self.children[i])
        super()._commit(node,parent,pose,candidate,outlets,kind)
        for i in descendants:
            if i != node: self.version[i] += 1
            for target in self.edges[i]:
                self._push(i,target); self.stats['batch_reopened'] += 1
        # Changed exits can enable previously rejected incoming rewires.
        for changed in {node, old_parent, parent}:
            self._reopen_incoming(changed)

    def _process(self, limit):
        count = 0
        while self.queue and count < limit:
            _,serial,parent,target,version,cost = heapq.heappop(self.queue)
            token = (parent,target)
            if token not in self.queued or self.queued[token][1] != serial: continue
            del self.queued[token]; count += 1
            self.stats['batch_pops'] += 1
            if version != self.version[parent] or abs(cost-self.cost[parent]) > 1e-10:
                self.stats['batch_stale'] += 1; self._push(parent,target); continue
            if target[0] == 'p':
                self._attach_pending(parent,target[1]); continue
            node = target[1]
            if self._ancestor(node,parent): continue
            outlets = self._exits(node)
            if not self._can_improve(node,parent,outlets):
                self.stats['rewire_lb_skipped'] += 1; continue
            self.stats['rewire_tests'] += 1
            z = self._attach(self.entry[parent],self.cells[parent],self.cells[node])
            accepted = self._candidate(node,parent,z,outlets) if z is not None else None
            if accepted is None and outlets:
                z = self._protected_entry(node,parent,outlets,'rewire')
                accepted = self._candidate(node,parent,z,outlets) if z is not None else None
            if accepted is not None: self._commit(node,parent,z,accepted,outlets,'rewire')
        self.stats['batch_pending'] = len(self.pending)
        self.stats['batch_queue'] = len(self.queue)

    def flush(self, tail=False):
        self.stats['batch_flushes'] += 1
        # Tail processes the queue snapshot; newly opened work is retained.
        self._process(len(self.queue) if tail else self.work_per_batch)

    def step(self):
        sample = self._sample(); self.stats['sample_calls'] += 1
        t = time.perf_counter(); parent = self._select(sample)
        self.stats['select_time_s'] += time.perf_counter()-t
        cell,portal,_ = self._steer(parent,sample)
        if cell is not None: self._add_pending(cell,parent,portal)
        if self.stats['sample_calls'] % self.batch_size == 0: self.flush()
