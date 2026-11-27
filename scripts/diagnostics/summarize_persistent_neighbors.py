#!/usr/bin/env python3
"""Compare old and retained historical-neighbor audit records."""
import collections
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'results/three_routes'
old=json.loads((root/'persistent_learning_replay/neighbor_audit.json').read_text())['records']
new=json.loads((root/'persistent_learning_retained_replay/neighbor_audit.json').read_text())['records']
assert [r['round'] for r in old]==[r['round'] for r in new]
keys=('candidate_neighbor_pairs','near_survive_ingest','near_survive_discover','near_at_candidate_backup','near_at_end','near_late_recomputed')
def summarize(rows):
    pend=[v for r in rows for v in r['pending_actions']]
    pairs=[v for r in rows for v in r.get('live_pairs',[])]
    return {'rounds':[r['round'] for r in rows],**{k:sum(r[k] for r in rows) for k in keys},'pending_competitive_and_lost':{str(k):v for k,v in collections.Counter((x['competitive'],x['lost']) for x in pend).items()},'live_pair_status':dict(collections.Counter(x['status'] for x in pairs)),'live_pair_best_certified':sum(x['is_best_certified'] for x in pairs)}
result={'old':summarize(old),'retained':summarize(new),'scope':'20 identical rounds with unchanged observed occupancy maps; candidate-neighbor pairs, not unique vertices; SFC regions can grow.'}
(root/'persistent_learning_retained_replay/neighbor_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
