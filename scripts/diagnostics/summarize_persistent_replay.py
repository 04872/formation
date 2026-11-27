#!/usr/bin/env python3
"""Summarize production select replay including learned-successor provenance."""
import argparse
import collections
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args()
    rows=[json.loads(l) for l in (a.folder/'natural.jsonl').open() if l.strip()]
    events=[(d['round'],e) for d in rows for e in d['backups']]
    cs=[(d['round'],x) for d in rows for x in d['candidates'] if x['value_source']=='learned']
    def cross(t,e):
        return e.get('successor_source')=='learned' and e.get('successor_write_round') is not None and e['successor_write_round']<t
    crosses=[{'round':t,**e} for t,e in events if e.get('status')=='written' and cross(t,e)]
    flips=[]
    for d in rows:
        if d['selected_endpoint']!=d['baseline_endpoint']:
            c={x['endpoint']:x for x in d['candidates']}
            flips.append({'round':d['round'],'baseline':c[d['baseline_endpoint']],'selected':c[d['selected_endpoint']]})
    changes=collections.Counter('up' if e['value']>e['previous']+1e-9 else 'down' if e['value']<e['previous']-1e-9 else 'equal' for _,e in events if e.get('status')=='written' and e.get('previous') is not None)
    summary={'classification':'chronological open-loop replay of recorded old trees, not a new closed-loop trajectory', 'rounds':len(rows),'round_range':[rows[0]['round'],rows[-1]['round']], 'modified_selection_rounds':len(flips),'modified_selections':flips,'learned_candidate_count':len(cs),'learned_candidate_successor_source':dict(collections.Counter(x['provenance']['successor_source'] for _,x in cs)), 'candidate_cross_round_learned_successor_count':sum(cross(t,x['provenance']) for t,x in cs), 'written_change_counts':dict(changes),'backup_cross_round_learned_successor_count':len(crosses),'backup_cross_round_examples':crosses[:5], 'capacity_reset_rounds':[d['round'] for d in rows if d['capacity_reset']], 'limitations':['Recorded candidate tree changes each round; no new MPPI rollout.', 'No proof of recovery at the same fork.', 'successor_write_round is counted only when successor_source is learned.', 'previous values can predate invalidation; differences summarize writes, not monotonic convergence.']}
    (a.folder/'natural_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print({k:v for k,v in summary.items() if k not in ('modified_selections','backup_cross_round_examples','limitations')})
if __name__=='__main__':main()
