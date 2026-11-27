#!/usr/bin/env python3
"""Paired depth-1/depth-2 replay on the same recorded closed-loop trees.

Each worker reconstructs state from round zero. The current tree is inserted
only after scoring, so every witness uses strictly earlier recorded trees.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from formation.local_affine.history_graph import HistoryGraph
from formation.local_affine.geometry import robot_positions
from formation.local_affine.history_local_backup import HistoryLocalBackup


def read(folder, t):
    with (folder / f'{t:03d}.pkl').open('rb') as f:
        return pickle.load(f)


def paths_for_tree(nodes, routes):
    # Prefix candidates are already represented by their longer path.
    prefixes = {j for route in routes for j in route[:-1]}
    return [{'nodes': np.asarray([nodes[j].q for j in route]),
             'node_ids': route, 'goal': False}
            for route in routes if route[-1] not in prefixes]


def json_value(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--input', type=Path, default=Path('/tmp/obs_boundary_rebuilt'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    first = read(args.input, 0)
    backup = HistoryLocalBackup(first['reference'], epsilon=0.4, rho=0.4)
    graph = HistoryGraph()
    region_keys, regions = set(), []
    prev_known = prev_occupied = None
    for t in range(args.end):
        d = first if t == 0 else read(args.input, t)
        world = d['world']
        known = np.asarray(world.known, dtype=bool)
        occupied = np.asarray(world.occupied, dtype=bool)
        if prev_known is not None:
            assert not np.any(prev_known & ~known), t
            assert not np.any(prev_known & (prev_occupied != occupied)), t
        prev_known, prev_occupied = known.copy(), occupied.copy()
        for h, b in d['regions']:
            key = (np.asarray(h).tobytes(), np.asarray(b).tobytes())
            if key not in region_keys:
                region_keys.add(key)
                regions.append((h, b))
        graph.update(world)
        # Synchronize the backup's own graph even during warmup.
        backup.heuristic.update(world)
        assert d['old']['current_goal_sources_count'] == 0, t
        if t >= args.start:
            routes = sorted(d['stats']['candidate_full_routes'], key=lambda x: x[-1])
            ids = [p[-1] for p in routes]
            queries = np.asarray([d['nodes'][i].q for i in ids])
            starts = robot_positions(queries, d['reference'])
            goals = robot_positions(d['goal'], d['reference'])
            distances = np.asarray(graph.distances(starts, goals, details=False))
            # The recorded heuristic used an all-or-nothing fallback over
            # candidates and boundary points; honor the recorded condition.
            if d['old']['heuristic_fallback']:
                distances = np.linalg.norm(starts-goals[None, :, :], axis=2)
            assert np.isfinite(distances).all(), t
            h = np.linalg.norm(distances, axis=1)/np.sqrt(len(d['reference']))
            old = {c['endpoint']: c for c in d['old']['candidates']}
            assert set(old) == set(ids), t
            errors = [abs(d['nodes'][i].cost + hv - old[i]['original_score'])
                      for i, hv in zip(ids, h)]
            maxerr = max(errors, default=0.)
            assert maxerr < 1e-8, (t, maxerr)
            evaluated = backup.evaluate(t, queries, d['goal'], world, regions)
            assert len(evaluated) == len(ids)
            backup_h_errors = [abs(d['nodes'][i].cost + result['h'] - old[i]['original_score'])
                               for i, result in zip(ids, evaluated)]
            backup_h_maxerr = max(backup_h_errors, default=0.)
            assert backup_h_maxerr < 1e-8, (t, backup_h_maxerr)
            candidates = []
            for i, route, hv, result in zip(ids, routes, h, evaluated):
                candidates.append({'endpoint': i, 'route': route,
                    'tree_cost': d['nodes'][i].cost, 'original_h': float(hv),
                    'original_score': old[i]['original_score'], **result})
            selected = {}
            for mode in ('v1', 'v2'):
                best = candidates[0]
                for c in candidates[1:]:
                    score, best_score = c['tree_cost'] + c[mode], best['tree_cost'] + best[mode]
                    if (score < best_score or (score == best_score and
                            len(c['route']) > len(best['route']) and
                            c['route'][:len(best['route'])] == best['route'])):
                        best = c
                selected[mode] = best['endpoint']
            out = {'round': t, 'recorded_endpoint': d['route'][-1],
                   'original_endpoint': min(candidates, key=lambda c: (c['original_score'], c['endpoint']))['endpoint'],
                   'selected': selected, 'old_jh_max_error': maxerr, 'backup_jh_max_error': backup_h_maxerr,
                   'tree_cost_max_error': d['cerr'], 'history_rounds': t,
                   'history_regions': len(regions), 'candidates': candidates}
            (args.output / f'{t:03d}.json').write_text(json.dumps(out, indent=2, default=json_value, allow_nan=False))
            print(f"round={t} n={len(candidates)} jh_error={maxerr:.3g} selected={selected}", flush=True)
        backup.add_round(t, paths_for_tree(d['nodes'], d['stats']['candidate_full_routes']))
        if t < args.start and t % 25 == 0:
            print(f'warm={t}', flush=True)

if __name__ == '__main__':
    main()
