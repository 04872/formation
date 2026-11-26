"""Rank the current certified RRT branches using member visibility distances."""
import time

import numpy as np

from .geometry import distance, robot_positions
from .history_graph import HistoryGraph


def score_candidates(node_ids, tree_costs, euclidean, graph_distances, original_rms=None):
    """Use one scoring rule for the entire tree, with exact-ID tie breaking."""
    node_ids = np.asarray(node_ids, dtype=int)
    tree_costs = np.asarray(tree_costs, dtype=float)
    euclidean = np.asarray(euclidean, dtype=float)
    graph_distances = np.asarray(graph_distances, dtype=float)
    if (euclidean.ndim != 2 or euclidean.shape != graph_distances.shape
            or len(node_ids) != len(tree_costs) or len(node_ids) != len(euclidean)):
        raise ValueError('candidate scoring inputs have incompatible shapes')
    missing = ~np.isfinite(graph_distances)
    fallback = bool(missing.any())
    used = euclidean if fallback else graph_distances
    if original_rms is None:
        original_rms = np.linalg.norm(euclidean, axis=1) / np.sqrt(euclidean.shape[1])
    else:
        original_rms = np.asarray(original_rms, dtype=float)
        if original_rms.shape != tree_costs.shape:
            raise ValueError('original RMS distances have incompatible shape')
    original_scores = tree_costs + 2 * original_rms
    used_rms = np.linalg.norm(used, axis=1) / np.sqrt(used.shape[1])
    # Exact direct queries have the original geometry and should not reorder
    # near ties through a different floating-point reduction order.
    direct_rows = np.all(graph_distances == euclidean, axis=1)
    used_rms[direct_rows] = original_rms[direct_rows]
    scores = original_scores.copy() if fallback else tree_costs + 2 * used_rms
    original_index = min(range(len(node_ids)), key=lambda i: (original_scores[i], node_ids[i]), default=None)
    selected_index = min(range(len(node_ids)), key=lambda i: (scores[i], node_ids[i]), default=None)
    return {'fallback': fallback, 'missing': missing, 'original_scores': original_scores,
            'scores': scores, 'original_index': original_index, 'selected_index': selected_index}


class HistoryDecision:
    """Maintain one observed graph and score every eligible current tree node."""

    def __init__(self):
        self.graph = HistoryGraph()

    def select(self, nodes, stats, reference, goal, world):
        start = time.perf_counter()
        self.graph.update(world)
        graph_seconds = time.perf_counter() - start
        routes = stats.get('candidate_full_routes', [])
        baseline = stats.get('baseline_route', [0])
        if not routes:
            return baseline, {
                'status': 'no_candidates',
                'baseline_endpoint': int(baseline[-1]),
                'selected_endpoint': int(baseline[-1]),
                'graph_seconds': graph_seconds,
            }

        # Sort by terminal node, independently of the old goal-biased ranking.
        routes = sorted(routes, key=lambda route: route[-1])
        starts = np.asarray([robot_positions(nodes[route[-1]].q, reference) for route in routes])
        goals = robot_positions(goal, reference)
        euclidean = np.linalg.norm(starts - goals[None, :, :], axis=2)
        query_start = time.perf_counter()
        distances = np.asarray(self.graph.distances(starts, goals), dtype=float)
        if distances.shape != euclidean.shape:
            raise ValueError('history graph returned an invalid member-distance matrix')
        tree_costs = np.asarray([nodes[route[-1]].cost for route in routes], dtype=float)
        original_rms = [distance(nodes[route[-1]].q, goal, reference) / np.sqrt(len(reference))
                        for route in routes]
        result = score_candidates([route[-1] for route in routes], tree_costs, euclidean, distances, original_rms)
        missing = result['missing']
        fallback = result['fallback']
        original_scores = result['original_scores']
        selected_scores = result['scores']
        original_index = result['original_index']
        selected_index = result['selected_index']
        candidates = []
        for i, route in enumerate(routes):
            graph_row = distances[i]
            candidates.append({
                'endpoint': int(route[-1]),
                'tree_cost': float(tree_costs[i]),
                'euclidean_member_distances': euclidean[i].tolist(),
                'graph_member_distances': [float(value) if np.isfinite(value) else None
                                           for value in graph_row],
                'missing_members': np.flatnonzero(missing[i]).tolist(),
                'original_score': float(original_scores[i]),
                'score': float(selected_scores[i]),
            })
        evidence = {
            'status': 'euclidean_fallback' if fallback else 'history_graph_scoring',
            'candidate_count': len(routes),
            'baseline_endpoint': int(routes[original_index][-1]),
            'selected_endpoint': int(routes[selected_index][-1]),
            'baseline_score': float(original_scores[original_index]),
            'selected_score': float(selected_scores[selected_index]),
            'any_missing_distance': fallback,
            'missing_count': int(missing.sum()),
            'same_endpoint': bool(routes[original_index][-1] == routes[selected_index][-1]),
            'candidates': candidates,
            'graph_nodes': len(self.graph.nodes),
            'query_statistics': dict(self.graph.last_query_stats),
            'graph_seconds': graph_seconds,
            'query_and_scoring_seconds': time.perf_counter() - query_start,
        }
        return routes[selected_index], evidence
