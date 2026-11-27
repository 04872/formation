#!/usr/bin/env python3
"""Audit saved U-wall candidate choices against split-group tip progress.

A positive signed tip coordinate means progress towards and around the upper
wall tip; it does not prove global feasibility or that MPPI follows a branch.
"""
import csv, json, sys, types
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
# Load the lightweight local modules without unrelated top-level solver imports.
package=types.ModuleType('formation'); package.__path__=[str(ROOT/'formation')];sys.modules['formation']=package
from formation.local_affine.geometry import robot_positions
from formation.paper_formation.world import World
BASE=ROOT/'results/three_routes/local_backup_closed_loop_500s'
EVENTS={'depth1':[(331,445,[0,1,3],'inner'),(435,458,[2,4,5],'outer')],
        'depth2':[(310,494,[0,1,2],'inner'),(400,425,[3],'member3')]}
EVENTS['depth1'].append((0,500,[0,1,3],'all_straddling'))
EVENTS['depth2'].append((0,500,[0,1,2],'all_straddling'))

def s(p):
    x,y=p[...,0],p[...,1]-25.75
    return np.where(x<14,y,np.sign(y)*np.hypot(np.maximum(x-14,0),y))

def components(c,tree,ref,world):
    z=np.zeros(4)
    for i,j in zip(c['route'][:-1],c['route'][1:]):
        a=np.array(tree[i]['q']); b=np.array(tree[j]['q'])
        length=np.linalg.norm(robot_positions(b,ref)-robot_positions(a,ref))/np.sqrt(len(ref))
        qs=np.array([(1-t)*a+t*b for t in (0,.5,1)])
        shape=np.mean([np.linalg.norm(q[2:].reshape(2,2)-np.eye(2))**2 for q in qs])
        points=np.array([robot_positions(q,ref) for q in qs]); risk=np.maximum(.8-world.clearance(points),0).mean()
        z += [length,.3*length*shape,.3*length*risk,.08*np.linalg.norm(b[2:]-a[2:])]
    if abs(z.sum()-c['tree_cost'])>1e-7: raise ValueError((z.sum(),c['tree_cost']))
    return z

def main():
    allrows=[]; report={}
    for name,events in EVENTS.items():
        path=BASE/name/'rrt_affine/seed_0'; p=json.loads((path/'planning.json').read_text())
        with np.load(path/'trajectory.npz') as f:
            dt=float(f['dt']); states=f['states'][:,:,:2]; ref=f['centered_reference']; known=f['known_history']; truth=f['occupancy']; leader_slot=int(f['leader_slot'])
        results=[]
        for start,end,group,label in events:
            other=[i for i in range(6) if i not in group]; rows=[]
            for k,tick in enumerate(p['ticks']):
                t=tick*dt
                if not start-5<=t<=end: continue
                if label=='all_straddling':
                    xy=states[tick]; inside=(xy[:,0]>14)&(xy[:,0]<30.5)&(xy[:,1]<25.34); outside=xy[:,1]>=26.16
                    if not (inside.any() and outside.any()): continue
                meta=p['frontend_metadata'][k]; d=meta.get('statistics',{}).get('history_decision',{}); tree=meta.get('tree',[]); candidates=d.get('candidates',[])
                # Actual motion over next 1 s, with explicit distinction from selection.
                n=min(tick+int(round(1/dt)),len(states)-1)
                actual=s(states[n])-s(states[tick]); act=float(actual[group].mean())
                row=dict(run=name,event=label,time_s=t,event_start_s=start,event_end_s=end,actual_group_delta_s_1s=act,actual_other_delta_s_1s=float(actual[other].mean()),status=d.get('status'),candidate_count=len(candidates))
                if not candidates or not tree:
                    row['winner_class']='no_candidates';rows.append(row);continue
                assignment=np.asarray(p['assignments'][k],int); root=np.asarray(tree[0]['q']); base=s(robot_positions(root,ref)[assignment])
                # Saved path q is leader-anchored; reconstruct assigned member targets.
                pathq=np.asarray(p['paths'][k]);
                if len(pathq):
                    def reference_members(q):
                        q=q.copy(); q[:2]-=q[2:].reshape(2,2)@ref[leader_slot]
                        return robot_positions(q,ref)[assignment]
                    rootref=reference_members(pathq[0]); nextref=reference_members(pathq[min(5,len(pathq)-1)])
                    row['ref_1s_vs_actual_group']=float((s(nextref)-s(states[tick]))[group].mean())
                    row['ref_1s_vs_refroot_group']=float((s(nextref)-s(rootref))[group].mean())
                    row['refroot_vs_actual_group']=float((s(rootref)-s(states[tick]))[group].mean())
                classified=[]
                for c in candidates:
                    q=np.asarray(tree[c['endpoint']]['q']); pos=robot_positions(q,ref)[assignment]; ds=s(pos)-base
                    g=float(ds[group].mean()); o=float(ds[other].mean()); translation=np.linalg.norm(q[:2]-root[:2]); rms=float(np.linalg.norm(pos-robot_positions(root,ref)[assignment])/np.sqrt(6))
                    # 0.1 m band filters tiny endpoint shifts. Both groups positive
                    # is the strict continuation of their prior joint progress.
                    if g>.1 and o>.1: cls='joint_continue'
                    elif g<-.1: cls='group_reverse'
                    elif translation<.15 and rms>.25: cls='shape_only'
                    elif g>.1: cls='group_continue_other_neutral_or_reverse'
                    else: cls='neutral_or_other'
                    classified.append((c,g,o,cls))
                win=next(x for x in classified if x[0]['endpoint']==d['selected_endpoint'])
                alt=[x for x in classified if x[1]>.1 and x[2]>=-.5]; cont=[x for x in classified if x[3]=='joint_continue']; best=min(cont,key=lambda x:(x[0]['score'],x[0]['endpoint'])) if cont else None
                cf=min(classified,key=lambda x:(x[0]['original_score'],x[0]['endpoint']))
                best_jh=min(cont,key=lambda x:(x[0]['original_score'],x[0]['endpoint'])) if cont else None
                row.update(winner_endpoint=win[0]['endpoint'],winner_class=win[3],winner_group_delta_s=win[1],winner_other_delta_s=win[2],continuation_candidates=len(cont),batched_candidates=len(alt),winner_batched=bool(win[1]>.1 and win[2]>=-.5),jh_winner_batched=bool(cf[1]>.1 and cf[2]>=-.5),jh_final_gap=cf[0]['score']-win[0]['score'],jh_winner_endpoint=cf[0]['endpoint'],jh_winner_class=cf[3])
                if best:
                    c=best[0];w=win[0]; row.update(best_jh_continue_endpoint=best_jh[0]['endpoint'],best_jh_continue_original_score=best_jh[0]['original_score'],jh_winner_original_score=cf[0]['original_score'],best_continue_endpoint=c['endpoint'],continue_group_delta_s=best[1],continue_other_delta_s=best[2],delta_score=c['score']-w['score'],delta_J=c['tree_cost']-w['tree_cost'],delta_H=c['h']-w['h'],delta_backup=(c['score']-c['original_score'])-(w['score']-w['original_score']))
                    assert abs(row['delta_score']-row['delta_J']-row['delta_H']-row['delta_backup'])<1e-7
                    if label!='all_straddling' and ((win[3]=='group_reverse' and start-5<=t<=end) or int(t)%10==0):
                        world=World('history_deadend');world.known=known[k].copy();world.occupied=world.known & truth;world._dirty=True
                        dc=components(c,tree,ref,world)-components(w,tree,ref,world)
                        row.update(dict(zip(['delta_length','delta_shape','delta_risk','delta_variation'],map(float,dc))))
                rows.append(row)
            selected=[r for r in rows if start<=r['time_s']<end]; op=[r for r in selected if r.get('continuation_candidates',0)>0]
            actualback=[r for r in selected if r['actual_group_delta_s_1s']<-.01]
            batched=[r for r in selected if r.get('batched_candidates',0)>0]; reverse_alt=[r for r in batched if r['winner_class']=='group_reverse'];
            results.append(dict(batched_opportunities=len(batched),batched_reverse_winners=len(reverse_alt),batched_reverse_jh_continue_flips=sum(r['jh_winner_batched'] and r['jh_final_gap']>1e-9 for r in reverse_alt),batched_reverse_jh_ties=sum(r['jh_winner_batched'] and abs(r['jh_final_gap'])<=1e-9 for r in reverse_alt),jh_batched_to_reverse=sum(r.get('jh_winner_batched',False) and r['winner_class']=='group_reverse' and r.get('jh_final_gap',0)>1e-9 for r in selected),jh_reverse_to_batched=sum(r.get('jh_winner_class')=='group_reverse' and r.get('winner_batched',False) and r.get('jh_final_gap',0)>1e-9 for r in selected),jh_batched_to_nonbatched=sum(r.get('jh_winner_batched',False) and not r.get('winner_batched',False) and r.get('jh_final_gap',0)>1e-9 for r in selected),jh_nonbatched_to_batched=sum(not r.get('jh_winner_batched',False) and r.get('winner_batched',False) and r.get('jh_final_gap',0)>1e-9 for r in selected),event=label,group=group,start=start,end=end,rounds=len(selected),winner_counts=dict(Counter(r['winner_class'] for r in selected)),with_continue=len(op),with_continue_winner_counts=dict(Counter(r['winner_class'] for r in op)),no_candidates=sum(r['winner_class']=='no_candidates' for r in selected),actual_back_rounds=len(actualback),actual_back_winner_counts=dict(Counter(r['winner_class'] for r in actualback)),reverse_winner_with_continue_jh_counts=dict(Counter(r['jh_winner_class'] for r in op if r['winner_class']=='group_reverse')),actual_back_joint_winner_reference_counts={str(eps):dict(reference_not_forward=sum(r.get('ref_1s_vs_actual_group',float('inf'))<=eps for r in actualback if r['winner_class']=='joint_continue'),reference_forward=sum(r.get('ref_1s_vs_actual_group',float('-inf'))>eps for r in actualback if r['winner_class']=='joint_continue')) for eps in [0.0,0.01]},jh_counts_given_continue=dict(Counter(r['jh_winner_class'] for r in op))))
            allrows.extend(rows)
        report[name]=results
    out=BASE/'u_split_choice.json';out.write_text(json.dumps({'method':'Upper tip signed progress; strict joint continuation requires mean endpoint gain >0.1m in each fixed group; actual back uses next 1s mean <-0.01m. Batched continuation requires group mean >0.1m, other mean >=-0.5m. Event summaries use [start,end); all_straddling requires an actual member with 14<x<30.5 and y<25.34 and another member with y>=26.16. Includes 5 pre-event rounds; hypotheses not causal proof.', 'events':report,'rows':allrows},indent=2)+'\n')
    fields=sorted(set().union(*(r.keys() for r in allrows)))
    with (BASE/'u_split_choice.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(allrows)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
