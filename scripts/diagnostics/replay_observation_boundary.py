#!/usr/bin/env python3
"""Score fixed recorded trees with and without historical motion paths."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from formation.local_affine.history_joint_path import HistoryJointPath

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True, help='exclusive')
parser.add_argument('--input', type=Path, default=Path('/tmp/obs_boundary_rebuilt'))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

def read(i):
    with (args.input / f'{i:03d}.pkl').open('rb') as f:
        return pickle.load(f)

history = HistoryJointPath(.4)
current = HistoryJointPath(.4, current_motion_only=True)
for i in range(args.start):
    d = read(i)
    for selector in (history, current):
        selector._update_map(d['world'], d['regions'])
        selector.heuristic.update(d['world'])
        selector.eligible_endpoints_total += len(d['stats']['candidate_full_routes'])
    routes = d['stats']['candidate_full_routes']
    prefixes = {j for route in routes for j in route[:-1]}
    for route in routes:
        if route[-1] not in prefixes:
            history.paths.append([d['nodes'][j].q.copy() for j in route])
    assert d['old']['current_goal_sources_count'] == 0
    if i % 20 == 0:
        print(f'warm {i}', flush=True)

for i in range(args.start, args.end):
    d = read(i)
    out = {}
    for name, selector in [('history', history), ('current', current)]:
        route, evidence = selector.select(d['nodes'], d['stats'], d['reference'],
                                          d['goal'], d['world'], d['regions'])
        out[name] = {'route': route, 'evidence': evidence}
    old = {c['endpoint']: c for c in d['old']['candidates']}
    for name in ('history', 'current'):
        ev = out[name]['evidence']
        assert set(old) == {c['endpoint'] for c in ev['candidates']}
        err = max(abs(c['original_score'] - old[c['endpoint']]['original_score']) for c in ev['candidates'])
        out[name]['old_jh_max_error'] = err
        assert err < 1e-8, (i, name, err)
    assert out['history']['evidence']['history_regions'] == out['current']['evidence']['history_regions']
    assert out['current']['evidence']['history_paths'] == 0
    out['recorded'] = {'route': d['route'], 'evidence': d['old'],
                       'tree_q_max_error': d['qerr'], 'tree_cost_max_error': d['cerr']}
    (args.output / f'{i:03d}.json').write_text(json.dumps(out, indent=2, allow_nan=False))
    print(f"score {i} history={out['history']['route'][-1]} current={out['current']['route'][-1]} "
          f"old={d['route'][-1]} jh_error={out['history']['old_jh_max_error']:.3g}", flush=True)
