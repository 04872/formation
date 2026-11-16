"""Generic sampling-based baselines on the same rigid-formation planning problem (method/对比实验.md).

The state is the formation pose ``q = (x, y, theta)``, the distance is ``d_W`` (Euclidean in the lifted chart
``z = (x, y, rho * theta)`` with the yaw difference wrapped), the goal is the goal position with free yaw.  A state
is valid when every robot ``c + R(theta) s_r`` keeps ``robot_radius + safety_margin + clearance`` from all obstacles
and the map border; edges are checked at a fixed ``d_W`` resolution.  The planners only see this validity interface.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

from formation.types import MapData, Pose2D, wrap_to_pi


@dataclass
class BaselineConfig:
    seed: int = 0
    time_budget: float = 10.0
    """Online seconds (checkpoints, path extraction and shortcutting excluded)."""
    checkpoints: tuple[float, ...] = ()
    goal_bias: float = 0.1
    """RRT*: probability of sampling the goal position (random yaw); BIT*: share of each batch at the goal."""
    clearance: float = 0.04
    """Required clearance beyond robot radius + safety margin (= what the region planner certifies)."""
    edge_resolution: float = 0.02
    """Edge collision check spacing [d_W]."""
    step: float = 0.6
    """RRT*: steering distance eta."""
    batch_size: int = 100
    """BIT*: samples per batch."""
    rewire_factor: float = 1.1
    shortcut_rounds: int = 100


class FormationValidity:
    """Per-robot state validity of the rigid formation; counts every checked state."""

    def __init__(self, map_data: MapData, slots: np.ndarray, clearance: float) -> None:
        self.slots = np.asarray(slots, dtype=float)
        self.rho = float(np.max(np.linalg.norm(self.slots, axis=1)))
        self.margin = float(map_data.robot_radius) + float(map_data.safety_margin) + clearance
        self.centers = np.array([p["center_xy"] for p in map_data.obstacle_primitives], dtype=float).reshape((-1, 2))
        self.radii = np.array([p["radius"] for p in map_data.obstacle_primitives], dtype=float)
        ox, oy = map_data.origin_xy
        self.low = np.array((ox, oy))
        self.high = np.array((ox + map_data.width_m, oy + map_data.height_m))
        self.states_checked = 0

    def clearance(self, q: np.ndarray) -> np.ndarray:
        """Guarded clearance (beyond the margin) of the worst robot for states ``q`` of shape (k, 3)."""
        q = np.atleast_2d(q)
        c, s = np.cos(q[:, 2])[:, None], np.sin(q[:, 2])[:, None]
        px = q[:, 0:1] + c * self.slots[None, :, 0] - s * self.slots[None, :, 1]
        py = q[:, 1:2] + s * self.slots[None, :, 0] + c * self.slots[None, :, 1]
        d = np.minimum.reduce((px - self.low[0], self.high[0] - px, py - self.low[1], self.high[1] - py))
        if len(self.radii):
            d = np.minimum(d, np.min(np.hypot(px[..., None] - self.centers[:, 0], py[..., None] - self.centers[:, 1])
                                     - self.radii, axis=2))
        return d.min(axis=1) - self.margin

    def valid(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q)
        self.states_checked += len(q)
        return self.clearance(q) >= 0.0


def distances(q: np.ndarray, Q: np.ndarray, rho: float) -> np.ndarray:
    dyaw = np.remainder(Q[:, 2] - q[2] + math.pi, 2 * math.pi) - math.pi
    return np.sqrt((Q[:, 0] - q[0]) ** 2 + (Q[:, 1] - q[1]) ** 2 + (rho * dyaw) ** 2)


def distance(a: np.ndarray, b: np.ndarray, rho: float) -> float:
    dyaw = wrap_to_pi(float(b[2] - a[2]))
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (rho * dyaw) ** 2)


def interpolate(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    dyaw = wrap_to_pi(float(b[2] - a[2]))
    s = np.asarray(s, dtype=float)[:, None]
    q = a[None, :] + s * np.array((b[0] - a[0], b[1] - a[1], dyaw))[None, :]
    q[:, 2] = np.remainder(q[:, 2] + math.pi, 2 * math.pi) - math.pi
    return q


class _Baseline:
    name = "baseline"

    def __init__(self, map_data: MapData, slots: np.ndarray, start: Pose2D, config: BaselineConfig) -> None:
        self.config = config
        self.check = FormationValidity(map_data, slots, config.clearance)
        self.rho = self.check.rho
        self.rng = np.random.default_rng(config.seed)
        self.start = np.array((start.x, start.y, start.yaw), dtype=float)
        self.goal_xy = np.asarray(map_data.goal_xy, dtype=float)
        self.low, self.high = self.check.low, self.check.high
        self.edges_checked = 0
        dim = 3
        measure = float(np.prod(self.high - self.low)) * 2 * math.pi * self.rho
        unit_ball = 4.0 / 3.0 * math.pi
        self.gamma = config.rewire_factor * 2.0 * (1.0 + 1.0 / dim) ** (1.0 / dim) * (measure / unit_ball) ** (1.0 / dim)

    def edge_valid(self, a: np.ndarray, b: np.ndarray, length: float | None = None) -> bool:
        self.edges_checked += 1
        length = distance(a, b, self.rho) if length is None else length
        n = max(1, int(math.ceil(length / self.config.edge_resolution)))
        return bool(np.all(self.check.valid(interpolate(a, b, np.arange(1, n + 1) / n))))

    def uniform(self) -> np.ndarray:
        return np.array((*self.rng.uniform(self.low, self.high), self.rng.uniform(-math.pi, math.pi)))

    def goal_state(self) -> np.ndarray:
        return np.array((*self.goal_xy, self.rng.uniform(-math.pi, math.pi)))

    # -- path phase (not timed as online) ---------------------------------------------------------------------------
    def path_length(self, path: list[np.ndarray]) -> float:
        return sum(distance(a, b, self.rho) for a, b in zip(path, path[1:]))

    def shortcut(self, path: list[np.ndarray]) -> list[np.ndarray]:
        """Greedy forward vertex shortcutting, then random shortcuts between points on the path (same validity
        check)."""
        path = [np.array(p) for p in path]
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = i + 1
            while j + 1 < len(path) and self.edge_valid(path[i], path[j + 1]):
                j += 1
            out.append(path[j])
            i = j
        path = out
        rng = np.random.default_rng(self.config.seed + 7919)
        for _ in range(self.config.shortcut_rounds):
            lengths = np.array([distance(a, b, self.rho) for a, b in zip(path, path[1:])])
            if len(lengths) < 2:
                break
            cum = np.concatenate([[0.0], np.cumsum(lengths)])
            s1, s2 = np.sort(rng.uniform(0.0, cum[-1], 2))
            k1, k2 = (min(int(np.searchsorted(cum, s, side="right")) - 1, len(lengths) - 1) for s in (s1, s2))
            if k1 == k2:
                continue
            p1 = interpolate(path[k1], path[k1 + 1], [(s1 - cum[k1]) / lengths[k1]])[0]
            p2 = interpolate(path[k2], path[k2 + 1], [(s2 - cum[k2]) / lengths[k2]])[0]
            if distance(p1, p2, self.rho) < s2 - s1 - 1e-9 and self.edge_valid(p1, p2):
                path = path[:k1 + 1] + [p1, p2] + path[k2 + 1:]
        return path

    def best_path(self) -> tuple[float, list[np.ndarray]] | None:
        raise NotImplementedError

    def checkpoint(self, online: float, iteration: int) -> dict:
        record = {"iteration": iteration, "online_time_s": online, "success": False, "regions": self.size()}
        best = self.best_path()
        if best is None:
            return record
        raw, path = best
        started = time.perf_counter()
        short = self.shortcut(path)
        record.update(success=True, raw_cost=raw, optimised_cost=self.path_length(short),
                      optimise_time_s=time.perf_counter() - started, extract_time_s=0.0, chain_regions=len(path),
                      route=[tuple(map(float, p)) for p in short])
        return record

    def size(self) -> int:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def plan(self) -> dict:
        if not self.check.valid(self.start)[0]:
            raise ValueError("start is invalid")
        config = self.config
        pending = sorted(set(config.checkpoints) | {config.time_budget})
        first, checkpoints, online, iteration = None, [], 0.0, 0
        while online < config.time_budget:
            started = time.perf_counter()
            self.step()
            online += time.perf_counter() - started
            iteration += 1
            if first is None and self.best_path() is not None:
                first = self.checkpoint(online, iteration)
                first["kind"] = "first"
            while pending and pending[0] <= online:
                pending.pop(0)
                record = self.checkpoint(online, iteration)
                record["kind"] = "budget"
                checkpoints.append(record)
        for record in checkpoints[:-1] + ([first] if first else []):
            record.pop("route", None)
        return {"variant": self.name, "success": checkpoints[-1]["success"], "first": first,
                "checkpoints": checkpoints, "regions": self.size(), "online_time_s": online, "solver_failures": 0,
                "stats": {"iterations": iteration, "states_checked": self.check.states_checked,
                          "edges_checked": self.edges_checked, "vertices": self.size()}}


class RRTStar(_Baseline):
    """RRT* (Karaman & Frazzoli 2011): steer by ``eta``, ChooseParent and Rewire within
    ``r_n = min(gamma (log n / n)^(1/3), eta)``; every new vertex tries a straight connection to the goal position
    at its own yaw."""

    name = "rrtstar"

    def __init__(self, *args) -> None:
        super().__init__(*args)
        cap = 1024
        self.Q = np.zeros((cap, 3))
        self.Q[0] = self.start
        self.cost = np.zeros(cap)
        self.parent = [-1]
        self.children: list[list[int]] = [[]]
        self.n = 1
        self.goal_nodes: list[int] = []
        self.goal_legs: list[float] = []

    def size(self) -> int:
        return self.n

    def _add(self, q: np.ndarray, parent: int, cost: float) -> int:
        if self.n == len(self.Q):
            self.Q = np.concatenate((self.Q, np.zeros_like(self.Q)))
            self.cost = np.concatenate((self.cost, np.zeros_like(self.cost)))
        i = self.n
        self.Q[i], self.cost[i] = q, cost
        self.parent.append(parent)
        self.children.append([])
        self.children[parent].append(i)
        self.n += 1
        return i

    def step(self) -> None:
        q_rand = self.goal_state() if self.rng.random() < self.config.goal_bias else self.uniform()
        Q = self.Q[:self.n]
        d = distances(q_rand, Q, self.rho)
        nearest = int(np.argmin(d))
        if d[nearest] < 1e-12:
            return
        q_new = interpolate(Q[nearest], q_rand, [min(1.0, self.config.step / d[nearest])])[0]
        if not self.check.valid(q_new)[0]:
            return
        radius = min(self.gamma * (math.log(self.n + 1) / (self.n + 1)) ** (1.0 / 3.0), self.config.step)
        dn = distances(q_new, Q, self.rho)
        near = np.flatnonzero(dn <= radius)
        if nearest not in near:
            near = np.append(near, nearest)
        order = near[np.argsort(self.cost[near] + dn[near], kind="stable")]
        parent = -1
        for j in order.tolist():
            if self.edge_valid(Q[j], q_new, dn[j]):
                parent = j
                break
        if parent < 0:
            return
        new_cost = self.cost[parent] + dn[parent]
        i = self._add(q_new, parent, new_cost)
        for k in near.tolist():
            if k == parent or new_cost + dn[k] >= self.cost[k] - 1e-12:
                continue
            if self.edge_valid(q_new, self.Q[k], dn[k]):
                self.children[self.parent[k]].remove(k)
                self.parent[k] = i
                self.children[i].append(k)
                shift = new_cost + dn[k] - self.cost[k]
                stack = [k]
                while stack:
                    node = stack.pop()
                    self.cost[node] += shift
                    stack.extend(self.children[node])
        goal = np.array((*self.goal_xy, q_new[2]))
        leg = distance(q_new, goal, self.rho)
        if self.edge_valid(q_new, goal, leg):
            self.goal_nodes.append(i)
            self.goal_legs.append(leg)

    def best_path(self):
        if not self.goal_nodes:
            return None
        costs = self.cost[self.goal_nodes] + np.array(self.goal_legs)
        k = int(np.argmin(costs))
        node, path = self.goal_nodes[k], []
        while node >= 0:
            path.append(self.Q[node].copy())
            node = self.parent[node]
        path = path[::-1] + [np.array((*self.goal_xy, self.Q[self.goal_nodes[k], 2]))]
        return float(costs[k]), path


class BITStar(_Baseline):
    """Batch Informed Trees (Gammell, Srinivasa & Barfoot 2015).  Heuristics: ``g^(x) = d_W(start, x)`` and
    ``h^(x) = ||x_xy - goal_xy||`` (the goal yaw is free, so this is the exact distance to the goal set).  Each batch
    adds ``batch_size`` valid samples, a ``goal_bias`` share of them at the goal position with random yaw, drawn
    from the informed set ``g^ + h^ < c_best`` once a solution exists; edges are collision-checked lazily in the
    order of the edge queue."""

    name = "bitstar"

    def __init__(self, *args) -> None:
        super().__init__(*args)
        cap = 1024
        self.Q = np.zeros((cap, 3))
        self.Q[0] = self.start
        self.g = np.full(cap, math.inf)
        self.g[0] = 0.0
        self.in_tree = np.zeros(cap, dtype=bool)
        self.in_tree[0] = True
        self.alive = np.zeros(cap, dtype=bool)
        self.alive[0] = True
        self.goal = np.zeros(cap, dtype=bool)
        self.parent = np.full(cap, -1)
        self.children: list[set[int]] = [set()]
        self.old = np.zeros(cap, dtype=bool)
        self.n = 1
        self.c_best = math.inf
        self.qv: list = []
        self.qe: list = []
        self.radius = math.inf
        self.batches = 0

    def size(self) -> int:
        return int(self.in_tree[:self.n].sum())

    def g_hat(self, idx) -> np.ndarray:
        return distances(self.start, self.Q[idx], self.rho)

    def h_hat(self, idx) -> np.ndarray:
        return np.hypot(self.Q[idx, 0] - self.goal_xy[0], self.Q[idx, 1] - self.goal_xy[1])

    def _grow(self) -> None:
        cap = len(self.Q)
        self.Q = np.concatenate((self.Q, np.zeros_like(self.Q)))
        self.g = np.concatenate((self.g, np.full(cap, math.inf)))
        for name in ("in_tree", "alive", "goal", "old"):
            setattr(self, name, np.concatenate((getattr(self, name), np.zeros(cap, dtype=bool))))
        self.parent = np.concatenate((self.parent, np.full(cap, -1)))

    def _sample(self, at_goal: bool) -> np.ndarray:
        if at_goal:
            return self.goal_state()
        if not math.isfinite(self.c_best):
            return self.uniform()
        a = self.start[:2]
        b = self.goal_xy
        center, focal = (a + b) / 2.0, float(np.linalg.norm(b - a))
        major = self.c_best / 2.0
        minor = math.sqrt(max(major ** 2 - (focal / 2.0) ** 2, 0.0))
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        rotation = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
        span = min(math.pi, self.c_best / self.rho)
        while True:
            r, phi = math.sqrt(self.rng.random()), self.rng.uniform(-math.pi, math.pi)
            xy = center + rotation @ np.array((major * r * math.cos(phi), minor * r * math.sin(phi)))
            if np.any(xy < self.low) or np.any(xy > self.high):
                continue
            q = np.array((*xy, wrap_to_pi(self.start[2] + self.rng.uniform(-span, span))))
            f = distance(self.start, q, self.rho) + float(np.hypot(*(xy - self.goal_xy)))
            if f < self.c_best:
                return q

    def _new_batch(self) -> None:
        self.batches += 1
        n = self.n
        if math.isfinite(self.c_best):
            idx = np.flatnonzero(self.alive[:n] & ~self.in_tree[:n])
            prune = idx[self.g_hat(idx) + self.h_hat(idx) >= self.c_best]
            self.alive[prune] = False
        self.old[:n] = self.in_tree[:n]
        goal_count = max(1, int(round(self.config.goal_bias * self.config.batch_size)))
        added = 0
        while added < self.config.batch_size:
            q = self._sample(added < goal_count)
            if not self.check.valid(q)[0]:
                continue
            if self.n == len(self.Q):
                self._grow()
            i = self.n
            self.Q[i], self.alive[i], self.goal[i] = q, True, added < goal_count
            self.children.append(set())
            self.n += 1
            added += 1
        count = int(self.alive[:self.n].sum())
        self.radius = self.gamma * (math.log(count) / count) ** (1.0 / 3.0)
        self.qe = []
        tree = np.flatnonzero(self.in_tree[:self.n])
        self.qv = [(float(self.g[v] + self.h_hat([v])[0]), int(v)) for v in tree]
        heapq.heapify(self.qv)

    def _expand(self, v: int) -> None:
        n = self.n
        d = distances(self.Q[v], self.Q[:n], self.rho)
        near = np.flatnonzero((d <= self.radius) & self.alive[:n])
        near = near[near != v]
        if not len(near):
            return
        h = self.h_hat(near)
        g_hat_v = float(self.g_hat([v])[0])
        samples = near[~self.in_tree[near]]
        mask = g_hat_v + d[samples] + h[~self.in_tree[near]] < self.c_best
        for x in samples[mask].tolist():
            heapq.heappush(self.qe, (self.g[v] + d[x] + float(np.hypot(*(self.Q[x, :2] - self.goal_xy))), v, x))
        if not self.old[v]:
            verts = near[self.in_tree[near]]
            hv = h[self.in_tree[near]]
            ok = (g_hat_v + d[verts] + hv < self.c_best) & (self.g[v] + d[verts] < self.g[verts])
            for w in verts[ok].tolist():
                if w != self.parent[v] and self.parent[w] != v:
                    heapq.heappush(self.qe, (self.g[v] + d[w] + float(np.hypot(*(self.Q[w, :2] - self.goal_xy))), v, w))

    def _set_parent(self, x: int, v: int, cost: float) -> None:
        if self.in_tree[x]:
            self.children[self.parent[x]].discard(x)
            shift = cost - self.g[x]
            stack = [x]
            while stack:
                node = stack.pop()
                self.g[node] += shift
                stack.extend(self.children[node])
        else:
            self.in_tree[x] = True
            self.g[x] = cost
            heapq.heappush(self.qv, (cost + float(self.h_hat([x])[0]), x))
        self.parent[x] = v
        self.children[v].add(x)
        goals = np.flatnonzero(self.goal[:self.n] & self.in_tree[:self.n])
        if len(goals):
            self.c_best = float(self.g[goals].min())

    def step(self) -> None:
        """One edge-queue pop (with the vertex expansions that must precede it), or a new batch."""
        if self.batches == 0 or (not self.qe and not self.qv):
            self._new_batch()
            return
        while self.qv and (not self.qe or self.qv[0][0] <= self.qe[0][0]):
            key, v = heapq.heappop(self.qv)
            current = float(self.g[v] + self.h_hat([v])[0])
            if current < key - 1e-12:
                heapq.heappush(self.qv, (current, v))
                continue
            self._expand(v)
        if not self.qe:
            return
        key, v, x = heapq.heappop(self.qe)
        if not self.in_tree[v] or not self.alive[x]:
            return
        c_hat = distance(self.Q[v], self.Q[x], self.rho)
        h_x = float(np.hypot(*(self.Q[x, :2] - self.goal_xy)))
        if self.g[v] + c_hat + h_x > key + 1e-12:
            heapq.heappush(self.qe, (self.g[v] + c_hat + h_x, v, x))
            return
        if self.g[v] + c_hat + h_x >= self.c_best:
            self.qe, self.qv = [], []
            return
        if self.g[v] + c_hat >= self.g[x]:
            return
        if float(self.g_hat([v])[0]) + c_hat + h_x >= self.c_best or not self.edge_valid(self.Q[v], self.Q[x], c_hat):
            return
        if self.g[v] + c_hat < self.g[x]:
            self._set_parent(x, v, self.g[v] + c_hat)

    def best_path(self):
        if not math.isfinite(self.c_best):
            return None
        goals = np.flatnonzero(self.goal[:self.n] & self.in_tree[:self.n])
        node = int(goals[np.argmin(self.g[goals])])
        path = []
        while node >= 0:
            path.append(self.Q[node].copy())
            node = int(self.parent[node])
        return self.c_best, path[::-1]


BASELINES = {RRTStar.name: RRTStar, BITStar.name: BITStar}
