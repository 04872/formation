#!/usr/bin/env python3
"""Controlled revelation of archived, certified actions on one fixed U-scene tree.

This deliberately retains only the archived witness action for candidate 15,
not every possible action. Rankings also show all original candidates, whose
unlearned labels remain their archived H. It is not a natural closed-loop run.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from formation.local_affine.history_persistent import PersistentHistoryDecision

DATA = ROOT / 'results/three_routes/local_backup_depth_replay'
OUT = ROOT / 'results/three_routes/persistent_learning_replay'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    output = parser.parse_args().output
    tree = json.loads((DATA / '117.json').read_text())
    detail = json.loads((DATA / 'cutoff_117_15.json').read_text())
    # Compact snapshot exported from the archived round 117 reconstruction.
    fixed = json.loads((OUT / 'round117_snapshot.json').read_text())
    entries = {c['endpoint']: c for c in tree['candidates']}
    target = entries[15]
    first = detail['result']['witness1']
    high = next(o for o in detail['successor_options'] if o['round'] == 105 and o['node'] == 41 and o['end_node'] == 43)
    low = next(o for o in detail['successor_options'] if o['round'] == 107 and o['node'] == 7 and o['end_node'] == 8)
    assert high['value'] > low['value']
    learner = PersistentHistoryDecision(fixed['reference'])
    x = learner.add_vertex(detail['query_q'], h=target['h'])
    z = learner.add_vertex(first['end_q'], h=first['h'])
    hi = learner.add_vertex(high['end_q'], h=high['h'])
    lo = learner.add_vertex(low['end_q'], h=low['h'])
    events, stages = [], []
    def snapshot(name):
        scores = {str(i): c['tree_cost'] + (learner.value(x) if i == 15 else c['h']) for i, c in entries.items()}
        stages.append({'stage': name, 'target_value': learner.value(x), 'successor_value': learner.value(z),
                       'target_label': learner.labels[x], 'successor_label': learner.labels[z],
                       'pair_winner_15_11': min((15, 11), key=lambda i: scores[str(i)]),
                       'all_winner': int(min(scores, key=scores.get)), 'scores': scores})
    def backup(v, name):
        learner.round_id += 1  # one queued backup per controlled model cycle
        processed = learner.process_backups(1)
        assert len(processed) == 1, processed
        e = processed[0]
        assert e['vertex'] == v and e['status'] == 'written', e
        events.append({'step': name, 'model_cycle': learner.round_id, **e})
    snapshot('initial')
    learner.add_edge(x, z, first['cost'], source='archive_round_104_certified_composite')
    backup(x, 'first_witness')
    snapshot('first_witness')
    learner.add_edge(z, hi, high['cost'], source='archive_round_105_certified_composite')
    backup(z, 'expensive_successor')
    backup(x, 'propagate_expensive_to_candidate')
    snapshot('expensive')
    learner.add_edge(z, lo, low['cost'], source='archive_round_107_certified_composite')
    backup(z, 'cheaper_successor')
    backup(x, 'propagate_cheaper_to_candidate')
    snapshot('cheaper')
    assert [s['pair_winner_15_11'] for s in stages] == [15, 15, 11, 15]
    assert [s['all_winner'] for s in stages] == [15, 7, 7, 7]
    assert abs(stages[-1]['target_value'] - (first['cost'] + low['value'])) < 1e-9
    result = {'classification': 'controlled revelation of previously archived certified actions; not natural closed-loop replay',
              'limitations': ['Only one first action for candidate 15 is retained: this is a constructed finite action model, not a complete production successor list.', 'The competing candidates use original H and are not learned.', 'Earlier sampled actions are revealed artificially after holding the round-117 tree and observed map fixed.', 'Composite action costs include a certified short link and an archived path, and their endpoint H is still an estimate.', 'The full candidate set does not recover candidate 15 as winner; no U-scene escape or timing claim follows.'],
              'sources': {'tree': str(DATA / '117.json'), 'successor_options': str(DATA / 'cutoff_117_15.json'), 'fixed_snapshot': str(OUT / 'round117_snapshot.json'), 'fixed_observed_map': str(OUT / 'round117_map.npz')},
              'fixed': {'round': 117, 'candidate_count': len(entries), 'target': 15, 'pair_comparator': 11,
                        'root_q': fixed['root_q'], 'goal_q': fixed['goal_q'],
                        'known_cells': fixed['known_cells'], 'occupied_known_cells': fixed['occupied_known_cells']},
              'actions': {'first': first, 'expensive': high, 'cheaper': low}, 'events': events, 'stages': stages}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'replay.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for s in stages:
        print(s['stage'], 'V15=', round(s['target_value'], 6), 'Vsuccessor=', round(s['successor_value'], 6), 'pair=', s['pair_winner_15_11'], 'all=', s['all_winner'])
    print('wrote', output / 'replay.json')

if __name__ == '__main__':
    main()
