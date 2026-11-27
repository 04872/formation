#!/usr/bin/env python3
"""Bidirectional transfer test of actually supported holdout/root pairs.

Requires the chronological partial.json/results.json from kernel_holdout_u.py.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from formation.local_affine.geometry import robot_positions, region_membership, shape_valid, certify_edge
from formation.local_affine.history_graph import HistoryGraph
from formation.local_affine.rrt import plan, edge_cost
from formation.paper_formation.world import World
from scripts.diagnostics.kernel_holdout_u import BASE, OUT, ROUNDS


def main():
    frames=json.loads((BASE/'planning.json').read_text())['frontend_metadata']
    with np.load(BASE/'trajectory.npz') as z:
        ref=z['centered_reference'].copy(); known=z['known_history'].copy(); truth=z['occupancy'].copy()
    with np.load(BASE/'regions.npz') as z:
        coeff=z['coefficients'].copy(); offsets=z['offsets'].copy()
    goal=np.r_[[45.,20.],np.eye(2).ravel()]; gp=robot_positions(goal,ref)
    world=World('history_deadend',obstacle_count=3,map_seed=1,pillar_width=.3)
    assert np.array_equal(world.truth,truth)
    graph=HistoryGraph(); graph_round=-1; records=[]; completed=set()
    def regions(t):
        return [(v[:,:2],v[:,2]) for k in range(t*6,(t+1)*6) for v in [coeff[offsets[k]:offsets[k+1]]]]
    def H(qs):
        pos=robot_positions(np.asarray(qs),ref); d=np.asarray(graph.distances(pos,gp,details=False)); fall=not np.all(np.isfinite(d))
        if fall:d=np.linalg.norm(pos-gp[None,:,:],axis=2)
        return np.linalg.norm(d,axis=1)/np.sqrt(len(ref)),fall
    def fits(q,regs):
        return bool(shape_valid(q,'affine') and np.all(np.any(region_membership(robot_positions(q,ref),regs),axis=1)))
    def build(q,seed,regs):
        nodes,_,stats=plan(q,goal,ref,regs,world,np.random.default_rng(seed),'affine',250,collect_history_candidates=True)
        routes=stats['candidate_full_routes']
        if not routes:return None,{'accepted':stats['accepted'],'candidate_count':0}
        hs,fall=H([q]+[nodes[r[-1]].q for r in routes]); vals=[nodes[r[-1]].cost+h for r,h in zip(routes,hs[1:])]
        idx=int(np.argmin(vals)); route=routes[idx]
        return np.array([nodes[i].q for i in route]),{'accepted':stats['accepted'],'candidate_count':len(routes),'J':float(nodes[route[-1]].cost),'H_root':float(hs[0]),'H_end':float(hs[idx+1]),'score':float(vals[idx]),'heuristic_fallback':fall,'endpoint':int(route[-1]),'route':route}
    def transfer(path,other,regs):
        translated=path+(other-path[0]); invalid=[int(i) for i,q in enumerate(translated) if not shape_valid(q,'affine')]
        failed=[int(i) for i,(a,b) in enumerate(zip(translated[:-1],translated[1:])) if certify_edge(a,b,regs,ref) is None]
        safe=not invalid and not failed
        J=float(sum(edge_cost(a,b,ref,world) for a,b in zip(translated[:-1],translated[1:]))) if safe else None
        return translated,{'certified':safe,'invalid_shapes':invalid,'failed_edges':failed,'J':J}
    start=time.time()
    while len(completed)<len(ROUNDS):
        src=OUT/'results.json' if (OUT/'results.json').exists() else OUT/'partial.json'
        try:data=json.loads(src.read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
        pending=[row for row in data['rounds'] if row['round'] not in completed]
        if not pending:time.sleep(5);continue
        for row in pending:
            t=row['round']; world.known=known[t].copy();world.occupied=world.known & world.truth
            # Rebuild chronological history even if snapshots are accumulated masks.
            for k in range(graph_round+1,t+1):
                world.known=known[k].copy();world.occupied=world.known & world.truth;world._dirty=True;graph.update(world)
            graph_round=t
            regs=regions(t)
            for branch in row.get('branches',[]):
                pred=branch['predictions']['recursive']; support=pred['support']
                rec={'round':t,'group':branch['group'],'endpoint':branch['endpoint'],'support_count':len(support),'weight_sum':pred['weight_sum'],'status':None}
                if not support:
                    rec['status']='no_support';records.append(rec);continue
                best=max(support,key=lambda x:(x['weight'],-x['round'])); old_round=int(best['round'])
                q=np.array(branch['q']); old=np.array(frames[old_round]['tree'][0]['q']); seed=100000+t*1000+1
                rec.update(support=best,member_rms_m=float(np.linalg.norm(robot_positions(q,ref)-robot_positions(old,ref))/np.sqrt(len(ref))),seed=seed,old_root_local_sfc=fits(old,regs),query_local_sfc=fits(q,regs))
                if not rec['old_root_local_sfc'] or not rec['query_local_sfc']:
                    rec['status']='root_outside_current_sfc'; records.append(rec);continue
                paths=[];trees=[]
                for root in (q,old):
                    path,tree=build(root,seed,regs);paths.append(path);trees.append(tree)
                rec['trees']={'query':trees[0],'old_root':trees[1]}
                if any(p is None for p in paths):
                    rec['status']='empty_tree';records.append(rec);continue
                xy,a=transfer(paths[0],old,regs); yx,b=transfer(paths[1],q,regs)
                rec['transfer']={'query_to_old':a,'old_to_query':b}
                qs=[q,old,paths[0][-1],paths[1][-1],xy[-1],yx[-1]]; hs,fall=H(qs)
                for detail,src,dst in ((a,0,1),(b,1,0)):
                    if detail['certified']:
                        sourceJ=trees[src]['J']; source_end=2+src; translated_end=4+src
                        detail.update(J_difference=float(detail['J']-sourceJ),H_root_difference=float(hs[dst]-hs[src]),H_end_difference=float(hs[translated_end]-hs[source_end]),D_difference=float((detail['J']+hs[translated_end]-hs[dst])-(sourceJ+hs[source_end]-hs[src])))
                rec.update(status='evaluated',bidirectional=bool(a['certified'] and b['certified']),joint_H_fallback=fall,tree_D_difference=float((trees[1]['J']+hs[3]-hs[1])-(trees[0]['J']+hs[2]-hs[0])))
                records.append(rec)
            if row.get('missing_group'):records.append({'round':t,'status':'missing_group'})
            completed.add(t)
            evaluated=[r for r in records if r['status']=='evaluated']; directions=[v for r in evaluated for v in r['transfer'].values()]
            result={'protocol':{'rounds':list(ROUNDS),'pair_selection':'recursive kernel highest normalized-weight previously trained root for each of two holdout endpoints; missing support reported','same_seed':'100000 + 1000 * round + 1 for both independent trees','budget':250,'map':'chronologically accumulated observed map at query round','regions':'six current saved SFCs at query round','best':'minimum J+H among certified full routes from each new tree','transfer':'apply complete 6D displacement of every original edge at other root, validate shape and original edge certificate; recompute cost on current map','meaning':'operational feasibility of translated best motion; does not prove other random tree contains a similar action or theoretical interchangeability','labels':'prior historical root labels frozen at collection'},'records':records,'summary':{'records':len(records),'status_counts':{s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})},'bidirectional_success':sum(r['bidirectional'] for r in evaluated),'evaluated_pairs':len(evaluated),'directional_success':sum(x['certified'] for x in directions),'directional_attempts':len(directions)},'completed_rounds':sorted(completed),'elapsed_seconds':time.time()-start}
            (OUT/'support_transfer.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            print('round',t,'completed',len(completed),flush=True)
if __name__=='__main__':main()
