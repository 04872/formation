#!/usr/bin/env python3
"""Reconstruct one recorded round and expose the fixed V1 witness over time."""
import argparse
import json
from pathlib import Path

import numpy as np

from replay_local_backup_depth import read, paths_for_tree, json_value
from formation.local_affine.history_local_backup import HistoryLocalBackup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--endpoint', type=int, required=True)
    parser.add_argument('--input', type=Path, default=Path('/tmp/obs_boundary_rebuilt'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    first = read(args.input, 0)
    backup = HistoryLocalBackup(first['reference'])
    keys, regions = set(), []
    for t in range(args.round+1):
        d = first if t == 0 else read(args.input, t)
        backup.heuristic.update(d['world'])
        for h, b in d['regions']:
            key = (np.asarray(h).tobytes(), np.asarray(b).tobytes())
            if key not in keys:
                keys.add(key)
                regions.append((h, b))
        if t < args.round:
            if d['old']['current_goal_sources_count']:
                raise ValueError('goal paths must be explicitly reconstructed')
            backup.add_round(t, paths_for_tree(d['nodes'], d['stats']['candidate_full_routes']))
    routes = sorted(d['stats']['candidate_full_routes'], key=lambda x: x[-1])
    ids = [r[-1] for r in routes]
    q = np.asarray([d['nodes'][i].q for i in ids])
    result = backup.evaluate(args.round, q, d['goal'], d['world'], regions,
                             include_successor_options=True)[ids.index(args.endpoint)]
    first_witness = result['witness1']
    if first_witness is None:
        raise ValueError('candidate has no first-layer witness')
    detail = result['fixed_v1_successor']
    options = detail.pop('options')
    steps = []
    for cutoff in range(first_witness['round'], args.round):
        available = [w for w in options if w['round'] <= cutoff]
        best = min(available, key=lambda w: w['value']) if available else None
        u = best['value'] if best else (0. if first_witness['goal'] else first_witness['h'])
        steps.append({'cutoff_round': cutoff, 'successor_value': u,
                      'fixed_first_total': first_witness['cost']+u,
                      'successor': best})
    out = {'round': args.round, 'endpoint': args.endpoint,
           'query_q': q[ids.index(args.endpoint)].tolist(), 'result': result,
           'successor_options': options, 'cutoffs': steps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=json_value, allow_nan=False))
    print(f"round={args.round} endpoint={args.endpoint} options={len(options)} output={args.output}")


if __name__ == '__main__':
    main()
