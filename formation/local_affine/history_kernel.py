"""Tree-level residual learning with a compact, locally compatible kernel."""
import time
import numpy as np

from .geometry import robot_positions
from .history_graph import HistoryGraph
from .history_joint_geometry import evaluate_pairs


class HistoryKernel:
    """A zero-prior weighted mean of previous tree-level residual samples."""
    def __init__(self, reference, bandwidth=.4, prior_weight=1.):
        self.reference = np.asarray(reference, dtype=float)
        self.bandwidth = float(bandwidth)
        self.prior_weight = float(prior_weight)
        if self.bandwidth <= 0 or self.prior_weight <= 0:
            raise ValueError('kernel bandwidth and prior weight must be positive')
        self.samples = []

    def add_sample(self, root, target, round_id):
        sample_id = len(self.samples)
        self.samples.append(dict(q=np.asarray(root, dtype=float).copy(), target=float(target),
                                 round=int(round_id), sample_id=sample_id,
                                 positions=robot_positions(np.asarray(root, dtype=float), self.reference)))
        return sample_id

    def predict(self, q, goal, regions):
        q = np.asarray(q, dtype=float)
        if np.array_equal(q, np.asarray(goal)):
            return dict(residual=0., support=[], weight_sum=0., prior_weight_normalized=1.)
        positions = robot_positions(q, self.reference)
        support = []
        if self.samples:
            all_positions = np.asarray([sample['positions'] for sample in self.samples])
            distances = np.linalg.norm((all_positions - positions).reshape(len(self.samples), -1), axis=1) / np.sqrt(len(self.reference))
            near = np.flatnonzero(distances < self.bandwidth)
            if len(near):
                # Batch only nearby roots; each member may use a different region.
                compatible = np.zeros((len(near), len(self.reference)), dtype=bool)
                near_positions = all_positions[near]
                for h, b in regions:
                    query_inside = np.all(positions @ h.T <= b + 1e-9, axis=1)
                    if not np.any(query_inside):
                        continue
                    sample_inside = np.all(near_positions @ h.T <= b + 1e-9, axis=2)
                    compatible |= sample_inside & query_inside[None, :]
                    if np.all(compatible):
                        break
                for index in near[np.all(compatible, axis=1)]:
                    sample = self.samples[int(index)]
                    distance = float(distances[index])
                    weight = (1. - distance / self.bandwidth) ** 2
                    support.append(dict(sample_id=sample['sample_id'], round=sample['round'],
                                        distance=distance, raw_weight=weight, target=sample['target']))
        total_weight = sum(s['raw_weight'] for s in support)
        denominator = self.prior_weight + total_weight
        for item in support:
            item['weight'] = item['raw_weight'] / denominator
        return dict(residual=float(sum(s['weight'] * s['target'] for s in support)),
                    support=support, weight_sum=float(total_weight),
                    prior_weight_normalized=float(self.prior_weight / denominator))


class HistoryKernelDecision:
    def __init__(self, reference, bandwidth=.4, prior_weight=1., one_step=False):
        self.reference = np.asarray(reference, dtype=float)
        self.one_step = bool(one_step)
        self.model = HistoryKernel(reference, bandwidth, prior_weight)
        self.heuristic = HistoryGraph()
        self.regions = []
        self._region_keys = set()
        self.round_id = 0
        self.goal = None

    def _prepare(self, goal, world, regions):
        if self.goal is not None and not np.array_equal(self.goal, goal):
            self.model.samples.clear()
        self.goal = np.asarray(goal, dtype=float).copy()
        self._update_regions(regions)
        self.heuristic.update(world)

    def _update_regions(self, regions):
        for h, b in regions:
            h, b = np.asarray(h), np.asarray(b)
            key = (h.tobytes(), b.tobytes())
            if key not in self._region_keys:
                self._region_keys.add(key)
                self.regions.append((h.copy(), b.copy()))

    def _h(self, qs, goal):
        qs = np.asarray(qs, dtype=float).reshape(-1, 6)
        pos = robot_positions(qs, self.reference)
        target = robot_positions(np.asarray(goal), self.reference)
        distances = np.asarray(self.heuristic.distances(pos, target, details=False))
        fallback = not np.all(np.isfinite(distances))
        if fallback:
            distances = np.linalg.norm(pos - target[None, :, :], axis=2)
        h = np.linalg.norm(distances, axis=1) / np.sqrt(len(self.reference))
        h[np.all(qs == goal, axis=1)] = 0.
        return h, fallback

    def select(self, nodes, stats, goal, world, regions):
        start = time.perf_counter()
        self._prepare(goal, world, regions)
        routes = sorted(stats.get('candidate_full_routes', []), key=lambda r: r[-1])
        baseline = list(stats.get('baseline_route', [0]))
        before = len(self.model.samples)
        candidates = []
        target, sample_id, root_h = None, None, None
        fallback = False
        if routes:
            qs = [nodes[0].q] + [nodes[r[-1]].q for r in routes]
            hs, fallback = self._h(qs, goal)
            root_h = float(hs[0])
            for route, h in zip(routes, hs[1:]):
                pred = self.model.predict(nodes[route[-1]].q, goal, self.regions)
                cost = float(nodes[route[-1]].cost)
                candidates.append(dict(endpoint=int(route[-1]), route=list(route), tree_cost=cost,
                                       h=float(h), original_score=cost+float(h),
                                       learned_value=float(h)+pred['residual'], correction=pred['residual'],
                                       score=cost+float(h)+pred['residual'], prediction=pred))
            def choose(key):
                best = candidates[0]
                for c in candidates[1:]:
                    if c[key] < best[key] or (c[key] == best[key] and len(c['route']) > len(best['route']) and c['route'][:len(best['route'])] == best['route']):
                        best = c
                return best
            best, original = choose('score'), choose('original_score')
            selected = best['route']
            target = float((original['original_score'] if self.one_step else best['score']) - root_h)
            # Goal itself has a prescribed zero residual and is never trained.
            if not np.array_equal(nodes[0].q, goal):
                sample_id = self.model.add_sample(nodes[0].q, target, self.round_id)
        else:
            selected, best, original = baseline, None, None
        evidence = dict(status='history_kernel' if candidates else 'no_candidates', round=self.round_id,
                        candidates=candidates, candidate_count=len(candidates), selected_route=selected,
                        selected_endpoint=int(selected[-1]), baseline_endpoint=original['endpoint'] if original else int(selected[-1]),
                        baseline_route=baseline, frontend_baseline_endpoint=int(baseline[-1]),
                        root_h=root_h, root_target=target, sample_id=sample_id, target_mode='one_step' if self.one_step else 'recursive',
                        model_samples_before=before, model_samples_after=len(self.model.samples),
                        update_order='predict_all_then_select_then_add_root', heuristic_fallback=bool(fallback),
                        history_regions=len(self.regions), selection_seconds=time.perf_counter()-start)
        self.round_id += 1
        return selected, evidence

    def record_goal(self, nodes, route, goal):
        pass

    def record_direct(self, root, goal, world, regions):
        self._prepare(goal, world, regions)
        before = len(self.model.samples)
        target, sample_id, root_h = None, None, 0.
        if not np.array_equal(root, goal):
            hs, fallback = self._h([root, goal], goal)
            _, costs = evaluate_pairs(np.asarray([root, goal]), [(0, 1)], self.reference, world, self.regions, certify=False)
            root_h = float(hs[0])
            target = float(costs[0] - root_h)
            sample_id = self.model.add_sample(root, target, self.round_id)
        else:
            fallback = False
        evidence = dict(status='direct_goal', round=self.round_id, selected_route=[0, -1],
                        root_h=root_h, root_target=target, sample_id=sample_id, target_mode='one_step' if self.one_step else 'recursive', model_samples_before=before,
                        model_samples_after=len(self.model.samples), update_order='predict_all_then_select_then_add_root',
                        heuristic_fallback=fallback, history_regions=len(self.regions))
        self.round_id += 1
        return evidence

    def record_no_root(self, world, regions):
        # No nominal configuration or certified candidate exists to train on.
        self._update_regions(regions)
        self.heuristic.update(world)
        evidence = dict(status='no_safe_nominal_root', round=self.round_id, candidate_count=0,
                        candidates=[], sample_id=None, model_samples_before=len(self.model.samples),
                        model_samples_after=len(self.model.samples))
        self.round_id += 1
        return evidence
