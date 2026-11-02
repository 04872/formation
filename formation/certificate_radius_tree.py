"""Fixed geometric states, certificate straight edges and attempt-indexed radius.

Successful edges carry an actual convex-certificate corridor. The bounded local
refinement can return unknown; this implementation does not assert AO.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from formation.single_tree_rrtstar import SingleTreeConfig, SingleTreeRRTStar


@dataclass
class CertificateRadiusConfig(SingleTreeConfig):
    radius_max: float = 1.5
    radius_gamma: float = 1.5
    local_certificate_limit: int = 64
    local_depth_limit: int = 16


class CertificateRadiusTree(SingleTreeRRTStar):
    variant = "certificate_radius_tree"

    def initialize(self):
        super().initialize()
        self.attempts = 0
        self.radius = self.config.radius_max
        self.exact = {(self.start.x, self.start.y, self.start.yaw): 0}
        self.goal_nodes = [(u, goal, cost, 0) for u, goal, cost in self.goal_nodes]
        self.edge_certificates = {}
        self.cover_cache = {}
        self._state = np.zeros((256, 3))
        self._state[0] = (self.start.x, self.start.y, self.start.yaw)
        self.expansion_cells = [0]
        self.certificate_node = [0]
        self.stats.update(local_connect_calls=0, common_certificate_edges=0,
                          certificate_cover_edges=0, local_certificates=0,
                          local_certificate_builds=0, local_unknown=0,
                          local_invalid=0, local_cap_unknown=0, local_numerical_unknown=0,
                          certificate_tests=0, edge_cache_hits=0,
                          direct_samples=0, direct_rejected=0, attempts=0,
                          radius=self.radius)

    def _add_cell(self, cell, parent):
        index = super()._add_cell(cell, parent)
        if hasattr(self, "certificate_node"):
            self.certificate_node.append(None)
        return index

    def _distances(self, q):
        delta = self._state[:len(self.anchor)] - (q.x, q.y, q.yaw)
        delta[:, 2] = self.rho * ((delta[:, 2] + math.pi) % (2*math.pi) - math.pi)
        return np.linalg.norm(delta, axis=1)

    def _covers_segment(self, index, a, b, za, zb):
        self.stats["certificate_tests"] += 1
        cell = self.cells[index]
        # Both points must use the SAME branch as the selected short edge.
        seed = self.geo.lift(cell.pose, za[2]/self.rho)
        pa, pb = self.geo.lift(a, seed[2]/self.rho), self.geo.lift(b, seed[2]/self.rho)
        if abs(pa[2]-za[2]) > 1e-8 or abs(pb[2]-zb[2]) > 1e-8:
            return False
        return self.geo.slack(cell, a) >= 0.0 and self.geo.slack(cell, b) >= 0.0

    def _node_cover(self, node):
        previous, indices = self.cover_cache.get(node, (0, []))
        if previous < len(self.cells):
            # Filter only the new suffix; every certificate remains queryable.
            q, count = self.anchor[node], len(self.cells)
            dx, dy = q.x-self._x[previous:count], q.y-self._y[previous:count]
            turn = self.rho*np.abs((q.yaw-self._t[previous:count]+math.pi)%(2*math.pi)-math.pi)
            possible = np.flatnonzero((np.hypot(dx,dy)+turn <= self._g[previous:count]) &
                                      (turn <= self._lim[previous:count])) + previous
            indices = indices + [int(i) for i in possible if self.geo.slack(self.cells[int(i)],q)>=0.0]
            self.cover_cache[node] = (count, indices)
        return indices

    def _edge(self, u, v, limit=math.inf):
        a, b = self.anchor[u], self.anchor[v]
        za, zb = self.geo.lift(a, a.yaw), self.geo.lift(b, a.yaw)
        length = float(np.linalg.norm(zb-za))
        if length >= limit-1e-10:
            return None
        key = (u, v)
        if key in self.edge_certificates:
            self.stats["edge_cache_hits"] += 1
            corridor = self.edge_certificates[key]
            return length, corridor
        self.stats["local_connect_calls"] += 1
        preferred = list(dict.fromkeys((self.owner[u], self.owner[v])))
        for i in preferred:
            if self._covers_segment(i,a,b,za,zb):
                self.stats["common_certificate_edges"] += 1
                corridor = (i,)
                self.edge_certificates[key] = corridor
                return length, corridor
        for i in self._node_cover(u):
            if i not in preferred and self._covers_segment(i,a,b,za,zb):
                self.stats["common_certificate_edges"] += 1
                corridor = (i,)
                self.edge_certificates[key] = corridor
                return length, corridor
        built = 0
        failure = "cap"

        def certify(left, right, depth):
            nonlocal built, failure
            p, q = self.geo.pose(left), self.geo.pose(right)
            # Every acceptance certifies the entire subsegment by convexity.
            for i in self._covering(p):
                if self._covers_segment(i,p,q,left,right):
                    return [i]
            if depth >= self.config.local_depth_limit or built >= self.config.local_certificate_limit:
                return None
            midpoint = (left+right)*.5
            center = self.geo.pose(midpoint)
            cell = self.model.make_cell(center)
            built += 1
            self.stats["local_certificate_builds"] += 1
            if not cell.valid:
                failure = "invalid"
                return None
            i = self._add_cell(cell, None)
            self.stats["local_certificates"] += 1
            if self._covers_segment(i,p,q,left,right):
                return [i]
            # Seed slack / sqrt(2) is a certified d_W inner ball for this model.
            if float(np.linalg.norm(right-left)) < 1e-10:
                failure = "numerical"
                return None
            first = certify(left, midpoint, depth+1)
            if first is None:
                return None
            second = certify(midpoint, right, depth+1)
            return None if second is None else first+second

        corridor = certify(za,zb,0)
        if corridor is None:
            if failure == "invalid":
                self.stats["local_invalid"] += 1
            else:
                self.stats["local_unknown"] += 1
                self.stats["local_cap_unknown" if failure == "cap" else "local_numerical_unknown"] += 1
            return None
        corridor = tuple(i for j,i in enumerate(corridor) if j==0 or i != corridor[j-1])
        self.stats["certificate_cover_edges"] += 1
        self.edge_certificates[key] = corridor
        return length, corridor

    def _connect_state(self, node, fallback):
        distance = self._distances(self.anchor[node])
        near = np.flatnonzero(distance <= self.radius).tolist()
        candidates = set(near+[fallback])
        if self.parent[node] is not None:
            candidates.add(self.parent[node])
        forbidden, stack = set(), [node]
        while stack:
            k=stack.pop(); forbidden.add(k); stack.extend(self.children[k])
        self.stats["neighbor_candidates"] += len(candidates-forbidden)
        for u in sorted(candidates-forbidden, key=lambda k:self.cost[k]+distance[k]):
            if self.cost[u]+distance[u] >= self.cost[node]-1e-10:
                break
            self.stats["choose_parent_tests"] += 1
            edge=self._edge(u,node,self.cost[node]-self.cost[u])
            if edge is not None:
                self._reparent(node,u,edge)
                break  # exact straight-edge costs equal the sorting lower bound
        if not math.isfinite(self.cost[node]):
            return False
        ancestors=self._ancestors(node)
        for v in near:
            if v in ancestors or self.cost[node]+distance[v] >= self.cost[v]-1e-10:
                continue
            self.stats["rewire_tests"] += 1
            edge=self._edge(node,v,self.cost[v]-self.cost[node])
            if edge is not None:
                self._reparent(v,node,edge); self.stats["rewires"] += 1
        return True

    def _insert(self, anchor, owner, fallback):
        key=(anchor.x,anchor.y,anchor.yaw)
        fresh=key not in self.exact
        if fresh:
            node=len(self.anchor)
            self.exact[key]=node
            self.anchor.append(anchor); self.owner.append(owner)
            self.parent.append(None); self.children.append(set())
            self.cost.append(math.inf); self.edge_cost.append(math.inf); self.portal.append(None)
            if node == len(self._state):
                self._state=np.concatenate((self._state,np.zeros_like(self._state)))
            self._state[node]=(anchor.x,anchor.y,anchor.yaw)
        else:
            node=self.exact[key]
            self.stats["exact_reconsider"] += 1
        started, solves=time.perf_counter(),self.geo.solves
        connected=self._connect_state(node,fallback)
        self.stats["connect_time_s"] += time.perf_counter()-started
        self.stats["connect_solves"] += self.geo.solves-solves
        if not connected and fresh:
            for array in (self.anchor,self.owner,self.parent,self.children,self.cost,self.edge_cost,self.portal):
                array.pop()
            del self.exact[key]
            self.cover_cache.pop(node,None)
            # Failed attempts can leave cached edges only if one was accepted;
            # an unreachable state has no accepted edge, hence none are present.
            return False
        if fresh:
            self._add_goal(node)
        self.stats["nodes"]=len(self.anchor)
        return connected

    def step(self):
        sample=self._sample()
        self.attempts += 1
        candidate=self.config.radius_gamma*(math.log(self.attempts+math.e)**2/self.attempts)**.25
        self.radius=min(self.radius,self.config.radius_max,candidate)
        self.stats.update(attempts=self.attempts,radius=self.radius)
        started=time.perf_counter()
        distance=self._distances(sample)
        nearest=int(np.argmin(distance))
        covered=self._covering(sample)
        self.stats["coverage_time_s"] += time.perf_counter()-started
        fallback=nearest
        if covered:
            representatives=[self.certificate_node[i] for i in covered if self.certificate_node[i] is not None]
            if representatives:
                fallback=min(representatives,key=lambda k:self.cost[k]+distance[k])
        # Uniform samples near the tree enter unchanged; a projection does not
        # replace them. Distant inaccessible samples drive the same tree's cover.
        if covered or distance[nearest] <= self.radius:
            if not covered:
                cell=self.model.make_cell(sample)
                if not cell.valid:
                    self.stats["direct_rejected"] += 1
                    return
                owner=self._add_cell(cell,None)
                self.stats["local_certificates"] += 1
            else:
                owner=covered[0]
            self.stats["direct_samples"] += 1
            if self._insert(sample,owner,fallback):
                self.certificate_node[owner]=self.exact[(sample.x,sample.y,sample.yaw)]
                if owner not in self.expansion_cells:
                    self.expansion_cells.append(owner)
                return
            self.stats["direct_rejected"] += 1
        started=time.perf_counter()
        # Region gap remains an exploration heuristic; only reachable cells are
        # eligible and its witness goes through the identical tree connector.
        indices=np.asarray(self.expansion_cells,dtype=int)
        vector=np.column_stack((sample.x-self._x[indices],sample.y-self._y[indices],
                               self.rho*((sample.yaw-self._t[indices]+math.pi)%(2*math.pi)-math.pi)))
        length=np.linalg.norm(vector,axis=1)
        direction=vector/np.maximum(length[:,None],1e-300)
        turn=np.abs(direction[:,2])
        rate=turn[:,None]-np.einsum("ikj,ij->ik",self._n[indices],direction[:,:2])
        with np.errstate(divide="ignore",invalid="ignore"):
            reach=np.minimum(self._g[indices]/(np.linalg.norm(direction[:,:2],axis=1)+turn),
                             np.where(turn>0,self._lim[indices]/turn,np.inf))
            reach=np.minimum(reach,np.where(rate>1e-15,self._e[indices]/rate,np.inf).min(axis=1))
        source=int(indices[np.argmin(length-reach)])
        fallback=self.certificate_node[source]
        self.stats["select_time_s"] += time.perf_counter()-started
        cell,anchor,_=self._steer(source,sample)
        if cell is None:
            return
        owner=self._add_cell(cell,source)
        # The steering witness is in both cells; the straight edge may need a
        # certificate corridor, so no unverified owner fallback is installed.
        if self._insert(anchor,owner,fallback):
            self.certificate_node[owner]=self.exact[(anchor.x,anchor.y,anchor.yaw)]
            self.expansion_cells.append(owner)

    def _add_goal(self, node):
        if not hasattr(self, "cover_cache"):
            return super()._add_goal(node)
        p=self.anchor[node]
        candidates=[]
        for i in self._node_cover(node):
            yaw=self.geo.goal_yaw(self.cells[i],self.goal_xy,p.yaw,self.config.goal_margin)
            if yaw is not None:
                from formation.types import Pose2D
                goal=Pose2D(*self.goal_xy,yaw)
                candidates.append((node,goal,self._segment(p,goal,i),i))
        if candidates:
            self.goal_nodes.append(min(candidates,key=lambda g:g[2]))

    def current_path(self):
        if not self.goal_nodes:
            return None
        node,goal,terminal,terminal_certificate=min(self.goal_nodes,key=lambda g:self.cost[g[0]]+g[2])
        sequence,cursor=[],node
        while cursor is not None:
            sequence.append(cursor); cursor=self.parent[cursor]
        sequence.reverse()
        route=[self.anchor[k] for k in sequence]+[goal]
        corridor=[]
        for v in sequence[1:]:
            corridor.extend(self.portal[v])
        corridor.append(terminal_certificate)
        corridor=[i for j,i in enumerate(corridor) if j==0 or i!=corridor[j-1]]
        return self.cost[node]+terminal,route,corridor,sequence
