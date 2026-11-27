#!/usr/bin/env python3
"""Compare two production replays and summarize frozen-H counterfactuals."""
import argparse
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);p=ap.parse_args().folder
    original=[json.loads(l) for l in (p/'natural.jsonl').open()]
    rows=[json.loads(l) for l in (p/'natural_ablation.jsonl').open()]
    assert len(original)==len(rows)==90
    mismatch=[a['round'] for a,b in zip(original,rows) if a['round']!=b['round'] or a['selected_endpoint']!=b['selected_endpoint'] or a['candidates']!=b['candidates'] or a['backups']!=b['backups']]
    flips=[{'round':d['round'],'actual':d['ablation']['actual_first_edge'],'frozen':d['ablation']['frozen_successor_h_first_edge'],'actual_endpoint':d['selected_endpoint'],'frozen_endpoint':d['ablation']['frozen_successor_h_selected']} for d in rows if d['ablation']['actual_first_edge']!=d['ablation']['frozen_successor_h_first_edge']]
    single=[{'round':d['round'],**a} for d in rows for a in d['ablation']['single_witness']]
    details=[a for d in rows for a in d['ablation']['details']]
    summary={'rounds':len(rows),'exact_repeat_mismatch_rounds':mismatch,'frozen_successor_h_first_edge_difference_count':len(flips),'frozen_successor_h_first_edge_differences':flips,'frozen_backup_resolved':sum(c['resolved'] for c in details),'frozen_backup_unresolved':sum(not c['resolved'] for c in details),'single_witness_count':len(single),'single_witness_first_edge_changed':sum(a['actual_first_edge']!=a['counterfactual_first_edge'] for a in single),'single_witness':single,'limitations':['Frozen H uses the same final current graph and pending action list with no learning or writeback, and falls back to H when L<U or U is infinite.','The frozen whole-graph decision is a comparison at the recorded state, not a rerun of counterfactual history or proof that cross-round learned values caused the difference.','Single-witness substitution holds the original chosen edge fixed, with other candidates untouched; it does not reselect the best witness.','successor_write_round is read only when successor_source is learned.']}
    (p/'ablation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('rounds',len(rows),'mismatches',len(mismatch),'frozen H first-edge changes',len(flips),'single-witness changes',summary['single_witness_first_edge_changed'])
if __name__=='__main__':main()
