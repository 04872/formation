#!/usr/bin/env python3
"""Prospective chronological root-only fit and independently seeded two-tree U probes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, sys, time, argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from formation.local_affine.history_graph import HistoryGraph
from formation.local_affine.geometry import robot_positions
from formation.local_affine.rrt import plan
from formation.local_affine.history_kernel import HistoryKernel
from formation.paper_formation.world import World
BASE=ROOT/'results/three_routes/persistent_learning_retained_500s/rrt_affine/seed_0'
OUT=ROOT/'results/three_routes/kernel_holdout_u'
ROUNDS=(433,439,445,451,457,463,469)
SEEDS=tuple(range(8))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=json.loads((BASE/'planning.json').read_text()); frames=p['frontend_metadata']
    with np.load(BASE/'trajectory.npz') as z:
        ref=z['centered_reference'].copy(); known=z['known_history'].copy(); truth=z['occupancy'].copy()
    with np.load(BASE/'regions.npz') as z: coeff=z['coefficients'].copy(); offsets=z['offsets'].copy()
    goal=np.r_[[45.,20.],np.eye(2).ravel()]; gp=robot_positions(goal,ref)
    def regions(t):
        return [(v[:,:2],v[:,2]) for k in range(t*6,(t+1)*6) for v in [coeff[offsets[k]:offsets[k+1]]]]
    world=World('history_deadend',obstacle_count=3,map_seed=1,pillar_width=.3)
    assert np.array_equal(world.truth,truth)
    graph=HistoryGraph(); models={'recursive':HistoryKernel(ref,bandwidth=.4,prior_weight=1.),'one_step':HistoryKernel(ref,bandwidth=.4,prior_weight=1.)}
    cumulative=[]; seen=set(); rounds=[]; exclusions=[]; timings=[]; h_diffs=[]
    selected={}
    for t in ROUNDS:
        frame=frames[t]; tree=frame['tree']; candidates=frame['statistics']['history_decision']['candidates']; root=np.array(tree[0]['q'])
        groups=[[c for c in candidates if root[0]-tree[c['endpoint']]['q'][0]<.5], [c for c in candidates if root[0]-tree[c['endpoint']]['q'][0]>=.5]]
        if any(not g for g in groups): selected[t]=None; continue
        selected[t]=[min(g,key=lambda c:(c['tree_cost']+c['h'],c['endpoint'])) for g in groups]
    # Never train a held-out endpoint, even if it happens to coincide with a root.
    held=[np.array(frames[t]['tree'][c['endpoint']]['q']) for t in ROUNDS if selected[t] for c in selected[t]]
    def values(qs):
        pos=robot_positions(np.asarray(qs),ref); d=np.asarray(graph.distances(pos,gp,details=False))
        fallback=not np.all(np.isfinite(d))
        if fallback:d=np.linalg.norm(pos-gp[None,:,:],axis=2)
        return np.linalg.norm(d,axis=1)/np.sqrt(len(ref)),fallback
    grouped = {}
    query_cache = {}
    def relevant_regions(q):
        # Regions containing no query member cannot certify any member. Filter
        # exactly with the kernel's own inequality before calling its predict.
        key=np.asarray(q).tobytes()
        if key in query_cache:return query_cache[key]
        pos=robot_positions(q,ref); out=[]
        for H,B,indices in grouped.values():
            inside=np.all(np.einsum('rki,mi->rmk',H,pos)<=B[:,None,:]+1e-9,axis=2)
            out.extend(cumulative[indices[i]] for i in np.flatnonzero(inside.any(axis=1)))
        query_cache[key]=out
        return out
    def rebuild_region_index():
        grouped.clear();query_cache.clear()
        index={}
        for i,(h,b) in enumerate(cumulative):index.setdefault(len(b),[]).append(i)
        for k,ids in index.items():
            grouped[k]=(np.stack([cumulative[i][0] for i in ids]),np.stack([cumulative[i][1] for i in ids]),ids)
    def tree_eval(root, seed, regs):
        st=time.perf_counter(); nodes,_,stats=plan(root,goal,ref,regs,world,np.random.default_rng(seed),'affine',250,collect_history_candidates=True)
        routes=stats['candidate_full_routes']; ids=[r[-1] for r in routes]
        result={'seed':int(seed),'accepted':stats['accepted'],'candidate_count':len(ids),'seconds':time.perf_counter()-st}
        if not ids:return result,None
        h,fall=values([n.q for n in nodes]); scores=np.array([nodes[i].cost+h[i] for i in ids]); i=ids[int(np.argmin(scores))]
        result.update(endpoint=int(i),q=nodes[i].q.tolist(),cost=float(nodes[i].cost),h=float(h[i]),score=float(nodes[i].cost+h[i]),fallback=fall)
        return result,nodes[i].q.copy()
    start=time.perf_counter()
    for t in range(max(ROUNDS)+1):
        world.known=known[t].copy();world.occupied=world.known & world.truth;world._dirty=True;graph.update(world)
        for h,b in regions(t):
            key=(h.tobytes(),b.tobytes())
            if key not in seen:seen.add(key);cumulative.append((h,b))
        rebuild_region_index()
        f=frames[t];tree=f.get('tree',[]);cand=f.get('statistics',{}).get('history_decision',{}).get('candidates',[])
        if not tree or not cand:continue
        qs=np.asarray([tree[0]['q']]+[tree[c['endpoint']]['q'] for c in cand]); hs,fall=values(qs)
        old=np.array([c['h'] for c in cand]); h_diffs.append(float(np.max(np.abs(hs[1:]-old))))
        if t in ROUNDS:
            if selected[t] is None:
                rounds.append({'round':t,'missing_group':True});continue
            pair=selected[t]; rec={'round':t,'regions_current':len(regions(t)),'regions_cumulative_unique':len(cumulative),'branches':[],'pairs':[]}
            for kind,c in zip(('small_retreat','retreat_ge_0_5'),pair):
                q=np.array(tree[c['endpoint']]['q']); h=values([q])[0][0]
                branch={'group':kind,'endpoint':c['endpoint'],'q':q.tolist(),'retreat':float(qs[0,0]-q[0]),'J':c['tree_cost'],'H':float(h),'recorded_H':c['h'],'persistent':c['value'],'persistent_source':c['value_source'],'predictions':{},'trees':[]}
                for name,model in models.items():branch['predictions'][name]=model.predict(q,goal,relevant_regions(q))
                rec['branches'].append(branch)
            for k in SEEDS:
                for j,b in enumerate(rec['branches']):
                    q=np.array(b['q']); seed1=100000+t*1000+k*10+1; seed2=100000+t*1000+k*10+2
                    a,q2=tree_eval(q,seed1,regions(t));b['trees'].append({'first':a})
                    if q2 is not None:
                        z,_=tree_eval(q2,seed2,regions(t)); b['trees'][-1]['second']=z
                    print('probe',t,k,j,'accepted',a['accepted'],'seconds',round(time.perf_counter()-start,1),flush=True)
            for k in SEEDS:
                a,b=[x['trees'][k] for x in rec['branches']]
                one=(b['first'].get('score',None),a['first'].get('score',None))
                def two(x):
                    return x['first']['cost']+x['second']['score'] if 'second' in x and 'score' in x['second'] else None
                v0,v1=two(a),two(b)
                rec['pairs'].append({'seed_index':k,'one_step_retreat_minus_small':one[0]-one[1] if None not in one else None,'two_step_retreat_minus_small':v1-v0 if v0 is not None and v1 is not None else None})
            for key in ('one_step_retreat_minus_small','two_step_retreat_minus_small'):
                v=np.array([x[key] for x in rec['pairs'] if x[key] is not None]);rec[key+'_summary']={'n':len(v),'mean':float(v.mean()) if len(v) else None,'se':float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else None}
            b0,b1=rec['branches']; rec['prediction_differences']={'H':b1['H']-b0['H'],'persistent':b1['persistent']-b0['persistent']}
            for name in models:rec['prediction_differences'][name]=b1['H']+b1['predictions'][name]['residual']-b0['H']-b0['predictions'][name]['residual']
            rec['J_difference']=b1['J']-b0['J'];rounds.append(rec)
            (OUT/'partial.json').write_text(json.dumps({'rounds':rounds},indent=2,allow_nan=False))
        if any(np.array_equal(qs[0],q) for q in held):exclusions.append(t);continue
        j=np.array([c['tree_cost'] for c in cand])
        for name,model in models.items():
            pred=np.array([model.predict(q,goal,relevant_regions(q))['residual'] for q in qs[1:]]) if name=='recursive' else np.zeros(len(cand))
            model.add_sample(qs[0],float(np.min(j+hs[1:]+pred)-hs[0]),t)
        if t%50==0:print('train',t,'elapsed',round(time.perf_counter()-start,1),flush=True)
    result={'protocol':{'rounds':ROUNDS,'paired_seeds':SEEDS,'iterations_per_tree':250,'first_policy':'minimum J+H in first realized tree; full certified route','second_policy':'independent tree minimum J+H; no hindsight','regions_rrt':'6 saved SFCs from the same round','regions_kernel':'unique saved SFCs from all rounds through this round','H':'chronologically reconstructed HistoryGraph; whole batch falls back to Euclidean if any nonfinite','training':'one recorded tree sample per prior round, target frozen at collection, no held endpoint used as root','delta':'retreat minus small retreat; negative favors retreat','empty_tree':'report and exclude incomplete paired references; never silently replace'},'h_recorded_abs_diff_max':max(h_diffs,default=None),'excluded_training_rounds_exact_held_roots':exclusions,'rounds':rounds,'elapsed_seconds':time.perf_counter()-start}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
def snapshot_only():
    """Interpolate last observed written candidate labels on identical-map rounds.

    This is a recorded-label snapshot, not an authoritative live graph: a
    historical label could later have been invalidated without reappearing.
    """
    source=OUT/'partial.json'
    data=json.loads(source.read_text());frames=json.loads((BASE/'planning.json').read_text())['frontend_metadata']
    with np.load(BASE/'trajectory.npz') as z: ref=z['centered_reference'].copy();known=z['known_history'].copy()
    with np.load(BASE/'regions.npz') as z: coeff=z['coefficients'].copy();offsets=z['offsets'].copy()
    goal=np.r_[[45.,20.],np.eye(2).ravel()]; output=[]
    for rec in data['rounds']:
        t=rec['round']; latest={}; regions=[]; seen=set()
        for s in range(t+1):
            for i in range(s*6,(s+1)*6):
                v=coeff[offsets[i]:offsets[i+1]];key=v.tobytes()
                if key not in seen:seen.add(key);regions.append((v[:,:2],v[:,2]))
            if s>=t or s<394 or not np.array_equal(known[s],known[t]):continue
            f=frames[s];tree=f.get('tree',[])
            for c in f.get('statistics',{}).get('history_decision',{}).get('candidates',[]):
                if c['value_source']!='learned' or c.get('write_round') is None or c['write_round']>=t:continue
                q=np.asarray(tree[c['endpoint']]['q']);latest[q.tobytes()]=(q,float(c['value']-c['h']),s,int(c['write_round']))
        model=HistoryKernel(ref,bandwidth=.4,prior_weight=1.)
        for q,y,seen_round,write_round in latest.values():model.add_sample(q,y,seen_round)
        branches=[]
        for b in rec['branches']:
            q=np.array(b['q']); pred=model.predict(q,goal,regions)
            branches.append({'group':b['group'],'prediction':pred,'exact_snapshot_match':q.tobytes() in latest})
        output.append({'round':t,'sample_count':len(latest),'branches':branches,'delta_residual':branches[1]['prediction']['residual']-branches[0]['prediction']['residual']})
    result={'protocol':'last recorded candidate observation of a learned label, deduplicated by exact configuration, s<t, write_round<t, same observed map as target and same graph generation >=394; no guarantee later invalidation is visible','rounds':output}
    (OUT/'written_label_snapshot.json').write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    if '--snapshot-only' in sys.argv:snapshot_only()
    else:main()
