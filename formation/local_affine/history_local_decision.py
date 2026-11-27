"""Closed-loop selection using finite-depth backups of prior RRT branches."""
import time
import numpy as np

from .history_local_backup import HistoryLocalBackup


class HistoryLocalDecision:
    def __init__(self, reference, depth=1, epsilon=.4, rho=.4):
        if depth not in (1, 2):
            raise ValueError('history local depth must be 1 or 2')
        self.depth = depth
        self.backup = HistoryLocalBackup(reference, epsilon, rho)
        self.regions = []
        self._region_keys = set()
        self.round_id = 0

    def _update_regions(self, regions):
        for h, b in regions:
            key = (np.asarray(h).tobytes(), np.asarray(b).tobytes())
            if key not in self._region_keys:
                self._region_keys.add(key)
                self.regions.append((np.asarray(h).copy(), np.asarray(b).copy()))

    @staticmethod
    def _paths(nodes, routes):
        prefixes = {j for route in routes for j in route[:-1]}
        return [dict(nodes=np.asarray([nodes[j].q for j in route]), node_ids=list(route), goal=False)
                for route in routes if len(route) >= 2 and route[-1] not in prefixes]

    def select(self, nodes, stats, goal, world, regions):
        start = time.perf_counter()
        self._update_regions(regions)
        t = self.round_id
        routes = sorted(stats.get('candidate_full_routes', []), key=lambda x: x[-1])
        baseline = stats.get('baseline_route', [0])
        if routes:
            values = self.backup.evaluate(t, [nodes[r[-1]].q for r in routes], goal,
                                          world, self.regions, depth=self.depth)
            candidates = []
            for route, value in zip(routes, values):
                cost = float(nodes[route[-1]].cost)
                candidates.append(dict(endpoint=int(route[-1]), route=list(route), tree_cost=cost,
                                       original_score=cost+value['h'],
                                       score=cost+value['v%d' % self.depth], **value))
            # Match the offline replay, including its prefix tie rule.
            best = candidates[0]
            for c in candidates[1:]:
                if (c['score'] < best['score'] or
                    (c['score'] == best['score'] and len(c['route']) > len(best['route'])
                     and c['route'][:len(best['route'])] == best['route'])):
                    best = c
            original = min(candidates, key=lambda c: (c['original_score'], c['endpoint']))
            selected = best['route']
            evidence = dict(status='history_local_backup', round=t, depth=self.depth,
                            candidate_count=len(candidates), candidates=candidates,
                            baseline_endpoint=original['endpoint'], selected_endpoint=best['endpoint'],
                            selected_score=best['score'], selected_route=selected)
        else:
            self.backup.heuristic.update(world)
            selected = baseline
            evidence = dict(status='no_candidates', round=t, depth=self.depth,
                            candidate_count=0, candidates=[], selected_route=list(selected),
                            selected_endpoint=int(selected[-1]))
        self.backup.add_round(t, self._paths(nodes, routes))
        self.round_id += 1
        evidence['history_regions'] = len(self.regions)
        evidence['selection_seconds'] = time.perf_counter()-start
        return selected, evidence

    def record_goal(self, nodes, route, goal):
        """Add a newly certified selected-route-to-goal to the just stored round."""
        if not route or not self.backup.rounds:
            return
        t, paths = self.backup.rounds[-1]
        qs = np.asarray([nodes[j].q for j in route] + [np.asarray(goal)])
        paths.append((qs, list(route)+[-1], True))

    def record_direct(self, root, goal, world, regions):
        self._update_regions(regions)
        self.backup.heuristic.update(world)
        t = self.round_id
        self.backup.add_round(t, [dict(nodes=[root, goal], node_ids=[0, -1], goal=True)])
        self.round_id += 1
        return dict(status='direct_goal', round=t, depth=self.depth,
                    selected_route=[0, -1], history_regions=len(self.regions))

    def record_no_root(self, world, regions):
        self._update_regions(regions)
        self.backup.heuristic.update(world)
        t = self.round_id
        self.backup.add_round(t, [])
        self.round_id += 1
        return dict(status='no_safe_nominal_root', round=t, depth=self.depth,
                    candidate_count=0, candidates=[], history_regions=len(self.regions))
