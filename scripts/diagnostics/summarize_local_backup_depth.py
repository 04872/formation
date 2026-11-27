#!/usr/bin/env python3
"""Summarize the complete recorded-tree depth comparison."""
import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/three_routes/local_backup_depth_replay')
rows = [json.loads((folder / f'{t:03d}.json').read_text()) for t in range(110, 200)]
stats = {'rounds': len(rows), 'candidate_count': 0, 'strict_two_layer_witnesses': 0,
         'v1_v2_selection_differences': 0, 'v1_original_selection_differences': 0,
         'v2_original_selection_differences': 0, 'fixed_first_successor_changed': 0,
         'old_jh_max_error': 0., 'backup_jh_max_error': 0.}
examples = []
candidate_rows = []
for row in rows:
    t = row['round']
    stats['v1_v2_selection_differences'] += row['selected']['v1'] != row['selected']['v2']
    for v in ('v1', 'v2'):
        stats[f'{v}_original_selection_differences'] += row['selected'][v] != row['original_endpoint']
    stats['old_jh_max_error'] = max(stats['old_jh_max_error'], row['old_jh_max_error'])
    stats['backup_jh_max_error'] = max(stats['backup_jh_max_error'], row['backup_jh_max_error'])
    with (Path('/tmp/obs_boundary_rebuilt') / f'{t:03d}.pkl').open('rb') as f:
        data = pickle.load(f)
    root_x = data['nodes'][0].q[0]
    retreats = {c['endpoint']: float(root_x - data['nodes'][c['endpoint']].q[0]) for c in row['candidates']}
    row['any_retreat_0_5'] = any(value >= 0.5 for value in retreats.values())
    for mode in ('v1', 'v2', 'original'):
        endpoint = row['original_endpoint'] if mode == 'original' else row['selected'][mode]
        row[mode + '_retreat'] = retreats[endpoint]
    for c in row['candidates']:
        candidate_rows.append([t, c['endpoint'], retreats[c['endpoint']], c['tree_cost'], c['original_h'], c['v1'], c['v2']])
        stats['candidate_count'] += 1
        for key in ('witness1', 'witness2'):
            w = c[key]
            if w is not None:
                assert w['round'] < t
                if w['successor'] is not None:
                    assert w['round'] < w['successor']['round'] < t
                    stats['strict_two_layer_witnesses'] += 1
                    if len(examples) < 10:
                        examples.append({'round': t, 'endpoint': c['endpoint'], 'witness': w})
        first = c['witness1']
        fixed = c['fixed_v1_successor']
        if first is not None and fixed is not None:
            next_w = fixed['witness']
            if next_w is not None:
                assert first['round'] < next_w['round'] < t
                if abs(fixed['value'] - first['h']) > 1e-9:
                    stats['fixed_first_successor_changed'] += 1
stats['rounds_with_retreat_candidate_0_5'] = sum(row['any_retreat_0_5'] for row in rows)
for mode in ('original', 'v1', 'v2'):
    stats[mode + '_selected_retreat_0_5'] = sum(row[mode + '_retreat'] >= 0.5 for row in rows)
with (folder / 'candidates.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round', 'endpoint', 'retreat_x_m', 'tree_cost', 'original_h', 'v1', 'v2'])
    writer.writerows(candidate_rows)
with (folder / 'rounds.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round', 'candidates', 'recorded_endpoint', 'original_endpoint', 'v1_endpoint', 'v2_endpoint', 'jh_error', 'backup_jh_error', 'any_retreat_0_5', 'original_retreat_x_m', 'v1_retreat_x_m', 'v2_retreat_x_m'])
    for row in rows:
        writer.writerow([row['round'], len(row['candidates']), row['recorded_endpoint'], row['original_endpoint'], row['selected']['v1'], row['selected']['v2'], row['old_jh_max_error'], row['backup_jh_max_error'], row['any_retreat_0_5'], row['original_retreat'], row['v1_retreat'], row['v2_retreat']])
out = {'statistics': stats, 'strict_witness_examples': examples}
(folder / 'summary.json').write_text(json.dumps(out, indent=2, allow_nan=False))
print(json.dumps(stats, indent=2))
