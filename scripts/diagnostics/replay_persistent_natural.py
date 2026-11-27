#!/usr/bin/env python3
"""Replay old U-scene trees in their recorded order, using production select.

Reconstruct input pickles first via scripts/diagnostics/rebuild_joint_inputs.py.
This is an open-loop replay of recorded trees, not a new execution rollout.
"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from formation.local_affine.history_persistent import PersistentHistoryDecision

def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def choose(candidates, scores):
    best = candidates[0]
    for c in candidates[1:]:
        a, b = scores[c['endpoint']], scores[best['endpoint']]
        if a < b or (a == b and len(c['route']) > len(best['route']) and c['route'][:len(best['route'])] == best['route']):
            best = c
    return best

def ablate(m, info):
    cs = info['candidates']
    if not cs: return None
    frozen, details = {}, []
    for c in cs:
        x=c['vertex']
        if x is None:
            frozen[c['endpoint']]=c['original_score']; continue
        edges=[cost + m.h[z] for z,(cost,_) in m.edges[x].items() if np.isfinite(m.h[z])]
        pending=[lower + m.h[z] for z,lower in m.pending[x].items() if np.isfinite(m.h[z])]
        U=min(edges,default=np.inf); L=min([U]+pending)
        valid=np.isfinite(U) and L>=U-1e-10
        value=U if valid else m.h[x]
        frozen[c['endpoint']]=c['tree_cost']+value
        details.append({'endpoint':c['endpoint'],'L':L,'U':U,'resolved':bool(valid),'frozen_value':value,'edges':len(edges),'pending':len(pending)})
    best=choose(cs,frozen); actual=choose(cs,{c['endpoint']:c['score'] for c in cs})
    individual=[]
    for c in cs:
        p=c['provenance']
        if not p or p['successor_source']!='learned' or p['successor_write_round'] is None or p['successor_write_round']>=info['round']: continue
        x,z=c['vertex'],p['successor']
        if z not in m.edges[x] or not np.isfinite(m.h[z]): continue
        cost=m.edges[x][z][0]
        alt=c['tree_cost']+cost+m.h[z]
        scores={v['endpoint']:v['score'] for v in cs}; scores[c['endpoint']]=alt
        other=choose(cs,scores)
        individual.append({'endpoint':c['endpoint'],'successor':z,'successor_write_round':p['successor_write_round'],'edge_cost':cost,'successor_h':m.h[z],'actual_value':c['value'],'replacement_value':cost+m.h[z], 'counterfactual_selected_endpoint':other['endpoint'],'counterfactual_first_edge':other['route'][1] if len(other['route'])>1 else None,'actual_first_edge':actual['route'][1] if len(actual['route'])>1 else None})
    return {'frozen_successor_h_selected':best['endpoint'],'frozen_successor_h_first_edge':best['route'][1] if len(best['route'])>1 else None,'actual_first_edge':actual['route'][1] if len(actual['route'])>1 else None,'details':details,'single_witness':individual,'interpretation':'Same final graph and current H, frozen-label backup without writeback; unresolved uses H. Single witness substitutes c+H only for one candidate, without reselecting its witness.'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/tmp/obs_boundary_rebuilt'))
    ap.add_argument('--output', type=Path, default=ROOT / 'results/three_routes/persistent_learning_replay/natural.jsonl')
    ap.add_argument('--end', type=int, default=200)
    ap.add_argument('--ablation', action='store_true', help='evaluate the identical finite edge/pending list with successor H and single-witness substitutions')
    a = ap.parse_args()
    with (a.input / '000.pkl').open('rb') as f:
        first = pickle.load(f)
    learner = PersistentHistoryDecision(first['reference'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w') as out:
        for t in range(a.end):
            if t:
                with (a.input / f'{t:03d}.pkl').open('rb') as f:
                    d = pickle.load(f)
            else:
                d = first
            _, info = learner.select(d['nodes'], d['stats'], d['goal'], d['world'], d['regions'])
            if a.ablation and t >= 110:
                info['ablation'] = ablate(learner, info)
            info['archived_recorded_endpoint'] = int(d['route'][-1])
            info['replay_kind'] = 'recorded tree, chronological open-loop select; not closed-loop'
            if t >= 110:
                out.write(json.dumps(finite_json(info), allow_nan=False) + '\n')
                out.flush()
            print(f"round={t} selected={info['selected_endpoint']} baseline={info['baseline_endpoint']} learned={sum(c['value_source']=='learned' for c in info['candidates'])} vertices={info['vertices']} updates={len(info['backups'])} sec={info['selection_seconds']:.2f}", flush=True)

if __name__ == '__main__':
    main()
