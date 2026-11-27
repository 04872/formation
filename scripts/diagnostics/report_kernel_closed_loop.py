#!/usr/bin/env python3
"""Compare recorded U-scene closed loops; no simulation or planner replay."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.summarize_closed_loop_backup import summarize
from scripts.diagnostics.summarize_persistent_learning import retreat_summary, timing
from scripts.paper_formation.audit_paper_formation import audit, audit_regions

ROOT = Path('results/three_routes')
CASES = {f'{name} seed {seed}': ROOT/f'{directory}/rrt_affine/seed_{seed}'
         for seed in (0, 1) for name,directory in (
           ('J+H','persistent_learning_raw_500s'),
           ('One step','kernel_one_step_closed_loop_500s'),
           ('Recursive','kernel_learning_closed_loop_500s'))}
CASES['Exact node seed 0'] = ROOT/'persistent_learning_retained_500s/rrt_affine/seed_0'
METHOD_KEYS = {'history_decision','history_joint_path','history_current_motion_only',
               'history_link_epsilon','history_local_depth','history_local_rho',
               'history_persistent','history_persistent_learning','history_kernel_learning', 'history_kernel', 'history_kernel_one_step','diagnostic_control','history_kernel_config'}

def load(path):
    with np.load(path/'trajectory.npz', allow_pickle=False) as z:
        d = {k:z[k] for k in z.files if k != 'known_history'}
    return d, *(json.loads((path/f).read_text()) for f in ('metrics.json','planning.json','configuration.json'))

def decisions(p):
    rows=[]
    for tick, m in zip(p['ticks'],p['frontend_metadata']):
        d=m.get('statistics',{}).get('history_decision',{})
        if d: rows.append((tick,d))
    return rows

def decision_summary(p, dt):
    rows=decisions(p); candidates=[c for _,d in rows for c in d.get('candidates',[])]
    counts={}; selected_counts={}; flip=first=initial_flip=0; compared=0; samples=[]; supported=selected_supported=0; first_unsupported_flip=0; unsupported_over_supported=0; unsupported_over_supported_edge=0; selected_rounds=0; prior=[]
    for tick,d in rows:
        cs=d.get('candidates',[]); lookup={c.get('endpoint'):c for c in cs}
        winner=lookup.get(d.get('selected_endpoint')); base=lookup.get(d.get('baseline_endpoint'))
        for c in cs:
            pred=c.get('prediction'); k=c.get('value_source', 'supported' if pred and pred.get('support') else 'zero_prior' if pred is not None else 'unrecorded'); counts[k]=counts.get(k,0)+1
            if pred is not None:
                supported+=bool(pred.get('support')); prior.append(pred.get('prior_weight_normalized',1.))
        if not winner: continue
        selected_rounds+=1
        pred=winner.get('prediction'); source=winner.get('value_source', 'supported' if pred and pred.get('support') else 'zero_prior' if pred is not None else 'unrecorded'); selected_counts[source]=selected_counts.get(source,0)+1
        selected_supported+=bool(pred and pred.get('support'))
        if base:
            compared+=1; changed=winner['endpoint']!=base['endpoint']; edge=winner.get('route',[])[:2]!=base.get('route',[])[:2]
            flip+=int(changed); first+=int(edge)
            initial_flip+=int(changed and source in ('initial','zero_prior')); first_unsupported_flip+=int(edge and source=='zero_prior')
            base_supported=bool((base.get('prediction') or {}).get('support'))
            unsupported_over_supported+=int(changed and source=='zero_prior' and base_supported)
            unsupported_over_supported_edge+=int(edge and source=='zero_prior' and base_supported)
            if changed:
                samples.append({'time_s':tick*dt,'baseline_endpoint':base['endpoint'],'selected_endpoint':winner['endpoint'],
                  'first_edge_changed':edge,'selected_value_source':source,
                  'baseline_original_score':base.get('original_score'),'selected_original_score':winner.get('original_score'),
                  'baseline_score':base.get('score'),'selected_score':winner.get('score')})
    return {'rounds_with_history_metadata':len(rows),'candidates':len(candidates),'candidate_value_sources':counts,
      'selected_value_sources':selected_counts,'supported_candidates':supported,'supported_candidate_rate':supported/len(candidates) if candidates else None,
      'selected_supported_rounds':selected_supported,'selected_support_rate':selected_supported/selected_rounds if selected_rounds else None,'mean_prior_weight':float(np.mean(prior)) if prior else None,'same_tree_baseline_comparable_rounds':compared,
      'selected_endpoint_changed_from_jh':flip,'first_edge_changed_from_jh':first,
      'changed_to_initial_or_zero_prior_source':initial_flip,'first_edge_changed_to_zero_prior':first_unsupported_flip,
      'zero_prior_wins_over_supported_jh_winner':unsupported_over_supported,
      'zero_prior_first_edge_wins_over_supported_jh_winner':unsupported_over_supported_edge,'selection_changes':samples,
      'note':'Same-tree J+H comparison only; initial source is not evidence of an actual underestimate. Value source support is a recorded label, not physical passage certification.'}

def compare_kernels(a, b):
    da,_,pa,_=a; db,_,pb,_=b; dt=float(da['dt'])
    if dt!=float(db['dt']): raise ValueError('different time steps')
    n=min(len(da['states']),len(db['states']))
    delta=np.max(np.abs(da['states'][:n]-db['states'][:n]),axis=(1,2))
    indices=np.flatnonzero(delta>1e-9)
    aa={tick:(m,q) for tick,m,q in zip(pa['ticks'],pa['frontend_metadata'],pa['initialization_q'])}
    bb={tick:(m,q) for tick,m,q in zip(pb['ticks'],pb['frontend_metadata'],pb['initialization_q'])}
    roots=[]; predictions=[]; endpoints=[]; edges=[]; same_tree=0; different_tree=0; labels=[]
    for tick in sorted(aa.keys() & bb.keys()):
        ma,qa=aa[tick]; mb,qb=bb[tick]
        ha=ma.get('statistics',{}).get('history_decision',{}); hb=mb.get('statistics',{}).get('history_decision',{})
        if ha.get('root_target') is not None and hb.get('root_target') is not None:
            labels.append((tick,abs(ha['root_target']-hb['root_target'])))
        ca={c['endpoint']:c for c in ha.get('candidates',[])};cb={c['endpoint']:c for c in hb.get('candidates',[])}
        if not ca or not cb: continue
        same=(np.allclose(qa,qb,rtol=0,atol=1e-9) and set(ca)==set(cb) and
              all(ca[k]['route']==cb[k]['route'] and abs(ca[k]['original_score']-cb[k]['original_score'])<1e-9 for k in ca))
        if not same: different_tree+=1; continue
        same_tree+=1
        for k in ca:
            if 'correction' in ca[k] and 'correction' in cb[k]: predictions.append((tick,k,abs(ca[k]['correction']-cb[k]['correction'])))
        if ha.get('selected_endpoint')!=hb.get('selected_endpoint'): endpoints.append(tick)
        if ha.get('selected_route',[])[:2]!=hb.get('selected_route',[])[:2]: edges.append(tick)
    nonzero_labels=[x for x in labels if x[1]>1e-9]; nonzero_predictions=[x for x in predictions if x[2]>1e-9]
    return {'common_samples':n,'maximum_absolute_state_component_difference':float(delta.max()),
      'first_state_component_difference_gt_1e-9_s':float(indices[0]*dt) if len(indices) else None,
      'complete_sequence_equal':bool(len(da['states'])==len(db['states']) and not len(indices)),
      'root_target_pairs':len(labels),'root_target_different_pairs':len(nonzero_labels),
      'root_target_max_abs_difference':max((x[1] for x in labels),default=None),
      'first_root_target_difference_s':nonzero_labels[0][0]*dt if nonzero_labels else None,
      'same_tree_comparable_rounds':same_tree,'different_tree_rounds':different_tree,
      'same_tree_prediction_pairs':len(predictions),'same_tree_prediction_different_pairs':len(nonzero_predictions),
      'same_tree_prediction_max_abs_difference':max((x[2] for x in predictions),default=None),
      'same_tree_endpoint_winner_difference_rounds':len(endpoints),'same_tree_first_edge_difference_rounds':len(edges),
      'first_same_tree_endpoint_difference_s':endpoints[0]*dt if endpoints else None,
      'first_same_tree_first_edge_difference_s':edges[0]*dt if edges else None,
      'same_tree_definition':'same saved root, candidate endpoints, routes, and J+H scores within 1e-9; not every unsampled tree node is compared',
      'label_comparison':'same simulation tick; if actual roots diverged, targets are not matched-state comparisons'}

def run(out, cases):
    data={name:load(path) for name,path in cases.items()}
    expected=np.array([[14,14,31,14.5],[14,25.5,31,26],[30.5,14,31,26]])
    if not np.allclose(data['J+H seed 0'][0]['rectangles'],expected): raise ValueError('U geometry mismatch')
    report={'definition':{'trajectory':'actual saved closed-loop execution','opening_exit':'each member first enters cavity, then center crosses x < 14 within 14.5 < y < 25.5',
      'time':'simulation and synchronous offline wall clock are distinct',
      'safety':'Euler states joined by line segments, and independent SFC checks; not a continuous jerk trajectory',
      'config':'all differing keys retained; method keys are listed separately, not silently discarded'},'runs':{}}
    report['kernel_mode_comparison']={f'seed_{seed}':compare_kernels(data[f'One step seed {seed}'],data[f'Recursive seed {seed}']) for seed in (0,1)}
    fig,axes=plt.subplots(2,3,figsize=(19,11),sharex=True,sharey=True,constrained_layout=True)
    for idx,(name,(d,m,p,c)) in enumerate(data.items()):
        ref=data['J+H seed 1' if name.endswith('seed 1') else 'J+H seed 0']
        ax=axes.flat[idx] if idx<6 else None
        for k in ('rectangles','goal_center','reference'):
            if not np.array_equal(d[k],ref[0][k]): raise ValueError(f'{name}: {k} mismatch')
        s=summarize(d,m); pos=d['states'][:,:,:2]; center=pos.mean(axis=1); dt=float(d['dt'])
        s['members_ever_exited_opening']=sum(x['first_exit_through_opening_s'] is not None for x in s['members'])
        s['members_final_in_cavity']=int(((pos[-1,:,0]>=14)&(pos[-1,:,0]<30.5)&(pos[-1,:,1]>14.5)&(pos[-1,:,1]<25.5)).sum())
        start=max(0,len(center)-1-round(100/dt)); x=center[start:]
        s['last_100s_center_net_m']=float(np.linalg.norm(x[-1]-x[0])); s['last_100s_center_path_m']=float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum())
        s['last_100s_member_net_m']=np.linalg.norm(pos[-1]-pos[start],axis=1).tolist()
        s['last_100s_member_path_m']=np.linalg.norm(np.diff(pos[start:],axis=0),axis=2).sum(axis=0).tolist()
        s['wall_s']=m.get('simulation_wall_seconds'); s['frontend_seconds']=timing([x.get('frontend') for x in m.get('planning_stage_seconds',[])])
        s['last_100_round_retreat']=retreat_summary(d,p); s['decision']=decision_summary(p,dt)
        diff={k:[ref[3].get(k),c.get(k)] for k in sorted(ref[3].keys()|c.keys()) if ref[3].get(k)!=c.get(k)}
        s['config_differences_from_jh']=diff
        s['config_nonmethod_differences_from_jh']={k:v for k,v in diff.items() if k not in METHOD_KEYS}
        s['source']=str(cases[name]); trace=cases[name]/'trajectory.npz'
        saved=cases[name]/'audit.json'
        if saved.exists():
            s['independent_safety']=json.loads(saved.read_text())[0]
        else:
            s['independent_safety']=audit(trace); s['independent_safety']['regions']=audit_regions(trace)
            saved.write_text(json.dumps([s['independent_safety']],indent=2)+'\n')
        report['runs'][name]=s
        if ax is None: continue
        for x0,y0,x1,y1 in d['rectangles']: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#343a45'))
        for i in range(pos.shape[1]): ax.plot(pos[:,i,0],pos[:,i,1],lw=.8,alpha=.7,color=plt.cm.tab10(i),label=f'member {i}')
        ax.plot(center[:,0],center[:,1],color='black',lw=1.6,label='center')
        ax.scatter(*center[0],color='green',s=35); ax.scatter(*center[-1],marker='x',color='red',s=55); ax.scatter(*d['goal_center'],marker='*',color='purple',s=90)
        ax.plot([14,14],[14.5,25.5],':',color='#d68b00')
        for t in np.arange(0,len(center)*dt,100):
            q=center[min(round(t/dt),len(center)-1)];ax.annotate(f'{t:.0f}s',q,fontsize=7)
        ax.set(title=f"{name}: exited {s['members_ever_exited_opening']}/6, final inside {s['members_final_in_cavity']}/6\ngoal={s['actual_goal_first_s'] is not None}, wall={s['wall_s']:.1f}s",xlabel='x (m)',ylabel='y (m)',aspect='equal',xlim=(0,50),ylim=(8,32)); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,loc='upper left',ncol=2); fig.suptitle('Executed paths: green start, red final, purple goal')
    out.mkdir(parents=True,exist_ok=True); fig.savefig(out/'kernel_executed_comparison.png',dpi=180);plt.close(fig)
    (out/'kernel_closed_loop_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,s in report['runs'].items(): print(name, 'duration',s['duration_s'],'exit',s['members_ever_exited_opening'],'inside',s['members_final_in_cavity'],'goal',s['actual_goal_first_s'],'wall',s['wall_s'],'P95',s['frontend_seconds']['p95_s'],'safety',s['independent_safety']['continuous_safety_certified_by_audit'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'kernel_learning_closed_loop_500s');args=ap.parse_args();run(args.output,CASES)
