"""Single-entry region tree with protected exits and online entry optimisation.

Every accepted update protects all CURRENT child entries and the region's fixed
goal endpoint.  Entry costs are real path lengths, never compensated scores.
The optional single-entry SOCP minimises a positive sum under exit protection;
it is not the exact minimax ranking.  Fixed FIFO work budgets are finite-budget
choices, with no completeness or AO claim.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np

from formation.region_coupling import CouplingConfig, SingleEntryRRTStar, _Program
from formation.types import Pose2D


@dataclass
class ExitCompensatedConfig(CouplingConfig):
    enable_local: bool = True
    local_tasks_per_attempt: int = 1
    rewire_tasks_per_attempt: int = 1
    protection_tolerance: float = 1e-8
    improvement_tolerance: float = 1e-7


class ExitCompensatedRRTStar(SingleEntryRRTStar):
    @property
    def variant(self):
        return "exit_local" if self.config.enable_local else "exit_rewire"

    def _init_root(self):
        super()._init_root()
        self.children = [set()]
        self.goal_endpoints = {}
        self.goal_cost_cache = {}
        self.geometry_version = [0]
        self.exit_version = [0]
        self.local_signature = [None]
        self.local_queue = deque()
        self.local_queued = set()
        self.rewire_queue = deque()
        self.stats.update(sample_calls=0, local_attempts=0, local_solves=0,
                          local_accepted=0, local_improvement=0., local_time_s=0.,
                          local_unchanged=0, local_no_exits=0, local_lb_skipped=0,
                          local_pending=0, local_pending_max=0, goal_chain_seeds=0,
                          exit_checks=0, exit_rejected=0, protection_violations=0,
                          exit_max_accepted_delta=-math.inf,
                          compensated_incoming_increases=0, compensation_min=0.,
                          rewire_solves=0, rewire_queued=0, rewire_queue_pops=0,
                          rewire_pending=0, rewire_pending_max=0,
                          rewire_budget_deferred=0, rewire_lb_skipped=0,
                          rewire_stale_cycles=0, rewire_accepted=0,
                          protected_infeasible=0, local_infeasible=0, rewire_infeasible=0,
                          local_infeasible_feasible_fallback=0,
                          local_infeasible_invalid_fallback=0, local_fallback_max_residual=0.,
                          protected_unknown=0,
                          protected_invalid=0, propagation_time_s=0.,
                          propagated_nodes=0, children_links=0,
                          children_max=0, goal_cost_increases=0)
        self._observed_goal_cost = math.inf

    def initialize(self):
        if self.cells:
            raise RuntimeError("initialize must be called once on a fresh planner")
        cell = self.model.make_cell(self.start)
        if not cell.valid:
            raise ValueError("start is in collision")
        self._add_cell(cell, None)
        self._init_root()
        self._register_goal(0)

    def _refresh(self, index):
        # Costs are eagerly propagated.  In particular do NOT use A's greedy
        # descendant reattachment: protected child entries must stay fixed.
        return None

    def _point(self, index, pose):
        return self.geo.lift(pose, self.cells[index].pose.yaw)

    def _distance_in(self, index, first, second):
        return float(np.linalg.norm(self._point(index, first)-self._point(index, second)))

    def _portal_valid(self, parent, node, pose):
        margin = self.config.overlap_margin * (1.-1e-6)
        return min(self.geo.slack(self.cells[parent], pose), self.geo.slack(self.cells[node], pose)) >= margin-1e-8

    def _exits(self, node):
        """Fixed outlet poses, their REAL old prefix costs, and child ids."""
        outlets = [(self.entry[c], self.cost[c], c) for c in sorted(self.children[node])]
        if node in self.goal_endpoints:
            q = self.goal_endpoints[node]
            outlets.append((q, self.cost[node]+self._distance_in(node, self.entry[node], q), None))
        return outlets

    def _ancestor(self, ancestor, node):
        while node is not None:
            if node == ancestor:
                return True
            node = self.parent[node]
        return False

    def _mark_local(self, node):
        if (not self.config.enable_local or node is None or node == 0 or node in self.local_queued
                or not (self.children[node] or node in self.goal_endpoints)):
            return
        self.local_queued.add(node)
        self.local_queue.append(node)
        self.stats["local_pending_max"] = max(self.stats["local_pending_max"], len(self.local_queue))

    def _changed_exits(self, node):
        if node is not None:
            self.exit_version[node] += 1
            self._mark_local(node)

    def _register_goal(self, node):
        yaw = self.geo.goal_yaw(self.cells[node], self.goal_xy, self.entry[node].yaw,
                                self.config.overlap_margin)
        if yaw is None:
            return
        self.goal_cells.append(node)
        # Freeze this endpoint for EACH protection comparison.  After a
        # committed entry update it can move to the closest feasible goal yaw.
        # It belongs to E_i even when another region has the best solution.
        self.goal_endpoints[node] = Pose2D(*self.goal_xy, yaw)
        self._cache_goal(node)
        self._changed_exits(node)
        cursor = node
        while cursor is not None:
            self._mark_local(cursor)
            self.stats["goal_chain_seeds"] += int(cursor != 0)
            cursor = self.parent[cursor]

    def _candidate(self, node, parent, pose, outlets):
        self.stats["exit_checks"] += 1
        if not self._portal_valid(parent, node, pose):
            return None
        incoming = self.cost[parent]+self._distance_in(parent, self.entry[parent], pose)
        if not outlets:
            # A leaf has no protected current branch.  Use the exact minimum
            # incoming-cost rule, without claiming an outlet improvement.
            return (incoming, []) if incoming < self.cost[node]-self.config.improvement_tolerance else None
        delta = [incoming+self._distance_in(node, pose, q)-old for q, old, _ in outlets]
        if (max(delta) <= self.config.protection_tolerance
                and min(delta) < -self.config.improvement_tolerance):
            return incoming, delta
        self.stats["exit_rejected"] += 1
        return None

    def _can_improve(self, node, parent, outlets):
        """Necessary metric lower bounds only; current incoming g is no gate."""
        tol, eps = self.config.protection_tolerance, self.config.improvement_tolerance
        if not outlets:
            lower = self.cost[parent]+self._bound(self.entry[parent], self.cells[parent], self.cells[node])
            return lower < self.cost[node]-eps
        lower = [self.cost[parent]+self.geo.distance(self.entry[parent], q)-old for q, old, _ in outlets]
        return max(lower) <= tol and min(lower) < -eps

    def _protected_entry(self, node, parent, outlets, kind):
        """One SOCP, all outlets protected, strictly positive equal weights.

        Variables z[3], s >= ||e_parent-z||, t_j >= ||z-x_j||.
        Hard rows s+t_j <= b_j-J_parent.  Minimising s+mean(t)
        also finds partial improvements when the best max(delta) is zero.
        """
        if not outlets:
            return None
        anchor = self.cells[node].pose.yaw
        seed_i = self.geo.lift(self.cells[node].pose, anchor)
        seed_p = self.geo.lift(self.cells[parent].pose, anchor)
        # Lift the parent entry through ITS certificate, then align its seed.
        # A direct wrap around anchor could choose a different seam branch.
        parent_entry = self.geo.lift(self.entry[parent], seed_p[2]/self.rho)
        n = len(outlets)
        program = _Program(4+n)
        margin = self.config.overlap_margin*(1.-1e-6)
        program.cell(self.cells[node], seed_i, 0, margin)
        program.cell(self.cells[parent], seed_p, 0, margin)
        program.distance(3, 0, parent_entry)
        rows = np.zeros((n, 4+n))
        rhs = np.empty(n)
        for j, (pose, old, _) in enumerate(outlets):
            program.distance(4+j, 0, self.geo.lift(pose, anchor))
            rows[j, 3] = rows[j, 4+j] = 1.
            rhs[j] = old-self.cost[parent]+self.config.protection_tolerance
        program.nonneg(rows, rhs)
        objective = np.zeros(4+n)
        objective[3], objective[4:] = 1., 1./n
        self.stats[kind+"_solves"] += 1
        self.geo.solves += 1
        try:
            result, status = program.solve(objective, self.geo.settings)
        except Exception:
            self.geo.failures += 1
            self.stats["protected_unknown"] += 1
            return None
        if result is None:
            self.stats["protected_infeasible" if status == "infeasible" else "protected_unknown"] += 1
            if status == "infeasible":
                # A local task keeps the same parent and its old entry is a
                # feasible fallback; infeasibility there is a numerical/model
                # diagnostic, unlike a legitimately incompatible new parent.
                self.stats[kind+"_infeasible"] += 1
                if kind == "local":
                    z_old = self.geo.lift(self.entry[node], anchor)
                    a, b, cones = self.geo._constraints((self.cells[node], self.cells[parent]), anchor, margin)
                    residual = max(float(np.max(a@z_old-b)),
                        max(float(np.linalg.norm(z_old[:2]-c[:2])-(g-sign*(z_old[2]-c[2])))
                            for c, sign, g in cones),
                        max(self.cost[parent]+float(np.linalg.norm(z_old-parent_entry))
                            +float(np.linalg.norm(z_old-self.geo.lift(q, anchor)))-old
                            -self.config.protection_tolerance for q, old, _ in outlets),
                        float(np.linalg.norm(self.geo.lift(self.entry[node], seed_p[2]/self.rho)-z_old)))
                    self.stats["local_fallback_max_residual"] = max(self.stats["local_fallback_max_residual"], residual)
                    key = "local_infeasible_feasible_fallback" if residual <= 1e-8 else "local_infeasible_invalid_fallback"
                    self.stats[key] += 1
                    self.geo.failures += 1
            self.geo.failures += int(status != "infeasible")
            return None
        pose = self.geo.pose(result[:3])
        # Check the solver chart as well as physical slack and real costs.
        if (np.linalg.norm(self.geo.lift(pose, anchor)-result[:3]) > 1e-7
                or np.linalg.norm(self.geo.lift(pose, seed_p[2]/self.rho)-result[:3]) > 1e-7):
            self.stats["protected_invalid"] += 1
            return None
        return pose

    def _commit(self, node, parent, pose, candidate, outlets, kind):
        incoming, delta = candidate
        old_parent, old_cost = self.parent[node], self.cost[node]
        if delta and max(delta) > self.config.protection_tolerance:
            self.stats["protection_violations"] += 1
            raise AssertionError("unprotected exit update")
        if old_parent != parent:
            self.children[old_parent].remove(node)
            self.children[parent].add(node)
            self._changed_exits(old_parent)
            self._changed_exits(parent)
        self.parent[node], self.entry[node], self.cost[node] = parent, pose, incoming
        self.geometry_version[node] += 1
        self.version[node] += 1
        self.parent_version[node] = self.version[parent]
        # The old parent loses this outlet or observes its changed pose; the
        # new parent gains it.  Only direct children see a moved parent entry.
        if old_parent == parent:
            self._changed_exits(parent)
        self.local_signature[node] = None
        if kind != "local":
            self._mark_local(node)
        if node in self.goal_endpoints:
            old_goal = self.goal_endpoints[node]
            yaw = self.geo.goal_yaw(self.cells[node], self.goal_xy, pose.yaw, self.config.overlap_margin)
            goal = Pose2D(*self.goal_xy, yaw)
            # The protected old endpoint stays feasible, so free-yaw refresh
            # can only lower the resulting terminal cost.
            if self._distance_in(node, pose, goal) <= self._distance_in(node, pose, old_goal)+1e-10:
                self.goal_endpoints[node] = goal
                if goal != old_goal:
                    self._changed_exits(node)
        for child in self.children[node]:
            self._mark_local(child)
        if node in self.goal_endpoints:
            self._cache_goal(node)
        began = time.perf_counter()
        stack = list(self.children[node])
        while stack:
            child = stack.pop()
            p = self.parent[child]
            self.cost[child] = self.cost[p]+self._distance_in(p, self.entry[p], self.entry[child])
            self.parent_version[child] = self.version[p]
            if child in self.goal_endpoints:
                self._cache_goal(child)
            self.stats["propagated_nodes"] += 1
            stack.extend(self.children[child])
        self.stats["propagation_time_s"] += time.perf_counter()-began
        self.stats[kind+"_accepted"] += 1
        if kind == "rewire":
            self.stats["rewires"] += 1
        if delta:
            self.stats["exit_max_accepted_delta"] = max(self.stats["exit_max_accepted_delta"], max(delta))
            self.stats["compensation_min"] = min(self.stats["compensation_min"],
                max(self._distance_in(node, pose, q)-(old-old_cost) for q, old, _ in outlets))
            self.stats["compensated_incoming_increases"] += int(incoming > old_cost+self.config.improvement_tolerance)
            if kind == "local":
                self.stats["local_improvement"] += -sum(delta)
        self.stats["children_max"] = max(self.stats["children_max"], len(self.children[parent]))

    def _connect(self, index, parent, portal):
        # Identical A neighbourhood, continuous ChooseParent, and fallback.
        cell, neighbors = self.cells[index], self._neighbors(index)
        candidates = [parent]+neighbors
        best, best_cost, best_entry = None, math.inf, None
        for j in sorted(candidates, key=lambda j: (j != parent, self.cost[j])):
            if self.cost[j]+self._bound(self.entry[j], self.cells[j], cell) >= best_cost:
                continue
            self.stats["choose_parent_tests"] += 1
            z = self._attach(self.entry[j], self.cells[j], cell)
            if z is None or not self._portal_valid(j, index, z):
                continue
            cost = self.cost[j]+self._distance_in(j, self.entry[j], z)
            if cost < best_cost:
                self.stats["choose_parent_improved"] += int(best is not None)
                best, best_cost, best_entry = j, cost, z
        if best is None:
            best, best_entry = parent, portal
            best_cost = self.cost[parent]+self._distance_in(parent, self.entry[parent], portal)
        self.parent.append(best)
        self.entry.append(best_entry)
        self.cost.append(best_cost)
        self.version.append(0)
        self.parent_version.append(self.version[best])
        self.children.append(set())
        self.geometry_version.append(0)
        self.exit_version.append(0)
        self.local_signature.append(None)
        self.children[best].add(index)
        self._changed_exits(best)
        self.stats["children_links"] = len(self.parent)-1
        self.stats["children_max"] = max(self.stats["children_max"], len(self.children[best]))
        for node in neighbors+([parent] if parent != best else []):
            if node == 0 or self._ancestor(node, index):
                continue
            outlets = self._exits(node)
            if not self._can_improve(node, index, outlets):
                self.stats["rewire_lb_skipped"] += 1
                continue
            self.stats["rewire_tests"] += 1
            z = self._attach(self.entry[index], cell, self.cells[node])
            if z is None:
                continue
            accepted = self._candidate(node, index, z, outlets)
            if accepted is not None:
                self._commit(node, index, z, accepted, outlets, "rewire")
            elif outlets:
                # Failed projection is not proof against another entry.  FIFO
                # gives each generated opportunity a later bounded-work turn.
                self.rewire_queue.append((node, index))
                self.stats["rewire_queued"] += 1
        self.stats["rewire_pending_max"] = max(self.stats["rewire_pending_max"], len(self.rewire_queue))

    def _service_rewire(self):
        for _ in range(self.config.rewire_tasks_per_attempt):
            if not self.rewire_queue:
                break
            node, parent = self.rewire_queue.popleft()
            self.stats["rewire_queue_pops"] += 1
            if self._ancestor(node, parent):
                self.stats["rewire_stale_cycles"] += 1
                continue
            outlets = self._exits(node)
            if not self._can_improve(node, parent, outlets):
                self.stats["rewire_lb_skipped"] += 1
                continue
            z = self._protected_entry(node, parent, outlets, "rewire")
            if z is not None:
                accepted = self._candidate(node, parent, z, outlets)
                if accepted is not None:
                    self._commit(node, parent, z, accepted, outlets, "rewire")
        self.stats["rewire_pending"] = len(self.rewire_queue)
        self.stats["rewire_budget_deferred"] += len(self.rewire_queue)

    def _service_local(self):
        if not self.config.enable_local:
            return
        began = time.perf_counter()
        for _ in range(self.config.local_tasks_per_attempt):
            if not self.local_queue:
                break
            node = self.local_queue.popleft()
            self.local_queued.remove(node)
            parent = self.parent[node]
            signature = (parent, self.geometry_version[parent], self.exit_version[node])
            if signature == self.local_signature[node]:
                self.stats["local_unchanged"] += 1
                continue
            self.stats["local_attempts"] += 1
            self.local_signature[node] = signature
            outlets = self._exits(node)
            if not outlets:
                self.stats["local_no_exits"] += 1
                continue
            if not self._can_improve(node, parent, outlets):
                self.stats["local_lb_skipped"] += 1
                continue
            z = self._protected_entry(node, parent, outlets, "local")
            if z is not None:
                accepted = self._candidate(node, parent, z, outlets)
                if accepted is not None:
                    self._commit(node, parent, z, accepted, outlets, "local")
                    # Own new entry does not change this problem's optimum.
                    self.local_signature[node] = signature
        self.stats["local_pending"] = len(self.local_queue)
        self.stats["local_time_s"] += time.perf_counter()-began

    def step(self):
        sample = self._sample()
        self.stats["sample_calls"] += 1
        began = time.perf_counter()
        parent = self._select(sample)
        self.stats["select_time_s"] += time.perf_counter()-began
        cell, portal, _ = self._steer(parent, sample)
        if cell is not None:
            index = self._add_cell(cell, parent)
            began, solves = time.perf_counter(), self.geo.solves
            self._connect(index, parent, portal)
            self._register_goal(index)
            self.stats["connect_time_s"] += time.perf_counter()-began
            self.stats["connect_solves"] += self.geo.solves-solves
        else:
            # Covered attempts still run scheduled work; changed dependencies
            # can revisit this old region without re-solving identical geometry.
            self._mark_local(parent)
        began, solves = time.perf_counter(), self.geo.solves
        self._service_rewire()
        self.stats["connect_time_s"] += time.perf_counter()-began
        self.stats["connect_solves"] += self.geo.solves-solves
        self._service_local()

    def _cache_goal(self, node):
        value = self._goal_cost(node)
        old = self.goal_cost_cache.get(node, math.inf)
        self.stats["goal_cost_increases"] += int(value > old+1e-7)
        self.goal_cost_cache[node] = value
        self._observed_goal_cost = min(self._observed_goal_cost, value)

    def _goal_cost(self, node):
        return self.cost[node]+self._distance_in(node, self.entry[node], self.goal_endpoints[node])

    def plan(self):
        self.initialize()
        online, first, checkpoints = 0., None, []
        for iteration in range(1, self.config.max_iterations+1):
            began = time.perf_counter()
            self.step()
            online += time.perf_counter()-began
            self.stats["iterations"] = iteration
            if first is None and self.goal_cells:
                first = self._checkpoint(iteration, online)
                first["kind"] = "first"
            if iteration in self.config.checkpoints:
                checkpoints.append(self._checkpoint(iteration, online))
        return dict(variant=self.variant, success=bool(self.goal_cells), first=first,
                    checkpoints=checkpoints, regions=len(self.cells),
                    online_time_s=online, solver_failures=self.geo.failures, stats=dict(self.stats))
