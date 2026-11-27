#!/usr/bin/env python3
"""Instrument 20 archived, map-stable rounds without changing the learner."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from formation.local_affine.history_persistent import PersistentHistoryDecision
from formation.local_affine.geometry import robot_positions

OUT=ROOT/'results/three_routes/persistent_learning_replay'
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folder',type=Path,default=OUT)
    ap.add_argument('--target-rounds',type=Path,default=OUT/'neighbor_audit.json')
    args=ap.parse_args()
    targets=[d['round'] for d in json.loads(args.target_rounds.read_text())['records']]
    assert len(targets)==20
    base=Path('/tmp/obs_boundary_rebuilt')
    with (base/'000.pkl').open('rb') as f: first=pickle.load(f)
    m=PersistentHistoryDecision(first['reference']); records=[]
    orig_discover=m._discover; orig_pending=m.add_pending; orig_edge=m.add_edge; orig_backup=m.backup_vertex
    state={'active':False}
    def pending(a,b,lower):
        if state['active'] and a is not None and b is not None and a < len(m.labels) and b < len(m.labels):
            old=m.labels[a]; legal=b not in m.edges[a] and b not in m.pending[a] and m._too_expensive.get((a,b)) != m._cert_version
            bound=lower+m.value(b)
            result=orig_pending(a,b,lower)
            if old is not None and legal and result:
                state['pending'].append({'a':a,'b':b,'bound':float(bound),'old':float(old),'competitive':bool(bound<old-1e-10),'lost':m.labels[a] is None})
            return result
        return orig_pending(a,b,lower)
    def edge(a,b,cost,source='search'):
        if state['active'] and a is not None and a < len(m.labels):
            old=m.labels[a]; result=orig_edge(a,b,cost,source)
            if old is not None and result: state['edges'].append({'a':a,'b':b,'source':source,'lost':m.labels[a] is None})
            return result
        return orig_edge(a,b,cost,source)
    def discover(ids):
        if state['active']:
            state['after_ingest']=[m.labels[i] is not None for i in state['old_ids']]
            pos=robot_positions(np.asarray(m.qs),m.reference).reshape(len(m.qs),-1)/np.sqrt(len(m.reference)); tree=cKDTree(pos)
            k=min(len(pos),m.max_neighbors+1)
            dd, nn=tree.query(pos[state['candidate_ids']], k=k, distance_upper_bound=m.epsilon)
            for x,ds,ns in zip(state['candidate_ids'],np.asarray(dd).reshape(-1,k),np.asarray(nn).reshape(-1,k)):
                state['near'][x]=[int(y) for y,d in zip(ns,ds) if np.isfinite(d) and y<len(pos) and y!=x and y in state['old_ids_set']]
        result=orig_discover(ids)
        if state['active']: state['after_discover']=[m.labels[i] is not None for i in state['old_ids']]
        return result
    def backup(x):
        if state['active'] and x in state['near'] and x not in state['at_candidate']:
            state['at_candidate'][x]=[i for i in state['near'][x] if m.labels[i] is not None]
            options=[cost+m.value(z) for z,(cost,_) in m.edges[x].items() if np.isfinite(m.value(z))]
            best=min(options,default=np.inf)
            for z in state['at_candidate'][x]:
                edge=m.edges[x].get(z)
                state['live_pairs'].append({'candidate':x,'neighbor':z,'status':'certified' if edge else ('pending' if z in m.pending[x] else 'absent'),'cost_plus_value':float(edge[0]+m.value(z)) if edge else None,'best_certified':float(best) if np.isfinite(best) else None,'is_best_certified':bool(edge and edge[0]+m.value(z)<=best+1e-10)})
        return orig_backup(x)
    m.add_pending=pending;m.add_edge=edge;m._discover=discover;m.backup_vertex=backup
    for t in range(max(targets)+1):
        if t:
            with (base/f'{t:03d}.pkl').open('rb') as f:d=pickle.load(f)
        else:d=first
        active=t in targets
        state={'active':active}
        if active:
            old_ids=[i for i,v in enumerate(m.labels) if v is not None and i!=m.goal_id]
            # Candidates are mapped during ingest; identify them in the wrapper after _ingest.
            state.update(old_ids=old_ids,old_ids_set=set(old_ids),candidate_ids=[],near={},pending=[],edges=[],at_candidate={},live_pairs=[])
            original_ingest=m._ingest
            def ingest(nodes,routes,stats,world):
                mapping=original_ingest(nodes,routes,stats,world)
                state['candidate_ids']=[mapping[r[-1]] for r in routes if mapping[r[-1]] is not None]
                return mapping
            m._ingest=ingest
        _,info=m.select(d['nodes'],d['stats'],d['goal'],d['world'],d['regions'])
        if active:
            m._ingest=original_ingest
            assert not info['map_changed'] and not info['capacity_reset']
            near_pairs=[(x,i) for x,neighbors in state['near'].items() for i in neighbors]
            before={i:bool(v) for i,v in zip(state['old_ids'],state['after_ingest'])}
            after={i:bool(v) for i,v in zip(state['old_ids'],state['after_discover'])}
            record={'round':t,'old_labeled':len(state['old_ids']),'candidate_neighbor_pairs':len(near_pairs),
                'near_survive_ingest':sum(before[i] for _,i in near_pairs),'near_survive_discover':sum(after[i] for _,i in near_pairs),
                'near_at_candidate_backup':sum(i in state['at_candidate'].get(x,[]) for x,i in near_pairs),
                'near_at_end':sum(m.labels[i] is not None for x,i in near_pairs),
                'near_late_recomputed':sum(i not in state['at_candidate'].get(x,[]) and m.labels[i] is not None for x,i in near_pairs),
                'candidate_backups_observed':len(state['at_candidate']),'candidate_count':len(state['candidate_ids']),
                'pending_actions':state['pending'],'edge_actions':state['edges'],'near_pairs':near_pairs,'live_pairs':state['live_pairs'],
                'selected':info['selected_endpoint'],'baseline':info['baseline_endpoint']}
            records.append(record);print(json.dumps({k:v for k,v in record.items() if k not in ('pending_actions','edge_actions','near_pairs','live_pairs')}),flush=True)
        elif t%25==0: print('warm',t,flush=True)
    args.folder.mkdir(parents=True,exist_ok=True)
    (args.folder/'neighbor_audit.json').write_text(json.dumps({'classification':'20 observed-map-stable recorded rounds; region sets can grow; first 20 eligible from 110 onward','records':records},indent=2)+'\n')
if __name__=='__main__':main()
