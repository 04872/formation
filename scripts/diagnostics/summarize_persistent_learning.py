#!/usr/bin/env python3
"""Summarize recorded U-scene closed loops; never reruns the simulator."""
import argparse
from collections import Counter
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

ROOT = Path('results/three_routes')
CASES = {
    'learning': ROOT/'persistent_learning_closed_loop_500s/rrt_affine/seed_0',
    'depth1': ROOT/'local_backup_closed_loop_500s/depth1/rrt_affine/seed_0',
    'depth2': ROOT/'local_backup_closed_loop_500s/depth2/rrt_affine/seed_0',
    'J+H': ROOT/'persistent_learning_raw_500s/rrt_affine/seed_0',
}

def load(path):
    with np.load(path/'trajectory.npz', allow_pickle=False) as z:
        # known_history can be very large and is unnecessary for this comparison.
        d = {k: z[k] for k in z.files if k != 'known_history'}
    return d, *(json.loads((path/f).read_text()) for f in ('metrics.json', 'planning.json', 'configuration.json'))

def compare_config(reference, current):
    ignored = {'history_decision', 'history_joint_path', 'history_current_motion_only', 'history_link_epsilon', 'history_local_depth', 'history_local_rho', 'history_persistent_learning', 'history_persistent'}
    return {k: [reference.get(k),current.get(k)] for k in sorted(reference.keys()|current.keys()) if k not in ignored and reference.get(k)!=current.get(k)}

def timing(xs):
    x=np.asarray([v for v in xs if v is not None and np.isfinite(v)],dtype=float)
    return None if not len(x) else {'count':len(x), 'total_s':float(x.sum()), 'median_s':float(np.median(x)), 'p95_s':float(np.percentile(x,95)), 'max_s':float(x.max())}

def learning_summary(planning, dt):
    decisions=[m.get('statistics',{}).get('history_decision',{}) for m in planning['frontend_metadata']]
    rows=[d for d in decisions if d.get('status') == 'history_persistent']
    ticks={d.get('round'):tick for d,tick in zip(decisions,planning['ticks']) if d.get('round') is not None}
    updates=[b for d in rows for b in d.get('backups',[])]
    regular=[b for b in updates if b.get('status')=='written']
    cycles=[b for b in updates if b.get('status')=='cycle_resolved']
    cycle_writes=[w for b in cycles for w in b.get('cycle_writes',[])]
    written=regular+cycle_writes
    comparable=[b for b in written if b.get('previous') is not None]
    candidates=[(d,c) for d in rows for c in d.get('candidates',[])]
    learned=[(d,c) for d,c in candidates if c.get('value_source')=='learned']
    def provenance(c, r):
        p=c.get('provenance') or {}
        w=p.get('successor_write_round')
        return c.get('value_source')=='learned' and p.get('successor_source')=='learned' and w is not None and w<r
    def select(cs, score):
        best=cs[0]
        for c in cs[1:]:
            a,b=score(c),score(best)
            if a<b or (a==b and len(c['route'])>len(best['route']) and c['route'][:len(best['route'])]==best['route']): best=c
        return best
    changes=[]; cross_effects=[]
    for d in rows:
        cs=d['candidates']; r=d['round']
        if not cs: continue
        selected=next((c for c in cs if c['endpoint']==d['selected_endpoint']),None)
        original=next((c for c in cs if c['endpoint']==d['baseline_endpoint']),None)
        if not selected or not original: continue
        # Counterfactual only replaces current candidate values with H where the
        # recorded direct successor is learned from an earlier round. It does
        # not claim to reconstruct or remove any deeper historical dependency.
        without=select(cs,lambda c: c['original_score'] if provenance(c,r) else c['score'])
        item={'round':r,'simulation_time_s':ticks.get(r,0)*dt,
              'original_endpoint':original['endpoint'],'selected_endpoint':selected['endpoint'],
              'first_edge_changed':selected['route'][:2]!=original['route'][:2],
              'learning_score_advantage_over_original':float(original['score']-selected['score']),
              'original_jh_advantage_over_selected':float(selected['original_score']-original['original_score']),
              'selected_candidate':selected,'original_candidate':original}
        if selected['endpoint']!=original['endpoint']: changes.append(item)
        if without['endpoint']!=selected['endpoint']:
            item=dict(item,without_direct_cross_round_endpoint=without['endpoint'],
                      without_direct_cross_round_route=without['route'],
                      direct_cross_round_score_advantage=float((without['original_score'] if provenance(without,r) else without['score'])-(selected['original_score'] if provenance(selected,r) else selected['score'])),
                      cross_round_first_edge_changed=without['route'][:2]!=selected['route'][:2])
            cross_effects.append(item)
    read=[(d,c) for d,c in learned if (c.get('provenance') or {}).get('successor_source')=='learned']
    cross=[(d,c) for d,c in read if provenance(c,d['round'])]
    older_than_write=[(d,c) for d,c in cross if c.get('write_round') is not None and c['provenance']['successor_write_round']<c['write_round']]
    return {'rounds':len(rows), 'candidate_evaluations':len(candidates), 'learned_candidate_evaluations':len(learned),
            'learned_candidates_read_learned_successor':len(read),
            'candidates_read_learned_successor_written_before_current_round':len(cross),
            'rounds_read_learned_successor_written_before_current_round':len({d['round'] for d,c in cross}),
            'candidates_successor_written_before_candidate_write':len(older_than_write),
            'provenance_limit':'direct successor source and latest write round at candidate write only; deeper chain and exact graph-generation ancestry are not reconstructible from candidate provenance alone',
            'learned_value_above_current_h':sum(c['value']>c['h']+1e-9 for d,c in learned),
            'learned_value_below_current_h':sum(c['value']<c['h']-1e-9 for d,c in learned),
            'candidate_value_comparison':'candidate learned value versus its current H, not a label update',
            'selection_differs_from_same_tree_j_plus_h':len(changes),
            'first_edge_differs_from_same_tree_j_plus_h':sum(c['first_edge_changed'] for c in changes),
            'selection_changes':changes,
            'jh_to_selected_value_source_counts':dict(Counter(f"{c['original_candidate'].get('value_source', 'unknown')} -> {c['selected_candidate'].get('value_source', 'unknown')}" for c in changes)),
            'jh_to_selected_value_source_first_edge_counts':dict(Counter(f"{c['original_candidate'].get('value_source', 'unknown')} -> {c['selected_candidate'].get('value_source', 'unknown')}" for c in changes if c['first_edge_changed'])),
            'strict_selection_changes':sum(c['learning_score_advantage_over_original']>1e-9 for c in changes),
            'tied_selection_changes':sum(abs(c['learning_score_advantage_over_original'])<=1e-9 for c in changes),
            'direct_cross_round_counterfactual_changed_selection':len(cross_effects),
            'direct_cross_round_counterfactual_changed_first_edge':sum(c['cross_round_first_edge_changed'] for c in cross_effects),
            'cross_round_effect_and_jh_selection_changed':sum(c['original_endpoint']!=c['selected_endpoint'] for c in cross_effects),
            'cross_round_effect_and_jh_first_edge_changed':sum(c['original_endpoint']!=c['selected_endpoint'] and c['first_edge_changed'] and c['cross_round_first_edge_changed'] for c in cross_effects),
            'direct_cross_round_counterfactual_changes':cross_effects,
            'counterfactual_limit':'replace the entire learned value at qualifying candidates with their current H, including even a single-edge motion cost; this is not isolation of the cross-round part and cannot establish that cross-round contribution alone caused the change. Other values remain fixed and can contain deeper historical dependencies; same saved tree, not a new closed-loop rollout. Logs do not preserve the successor H at the exact writeback needed to substitute c+H(successor) without reconstruction.',
            'backup_evaluations':len(updates),'backup_written_total_including_cycles':len(written),
            'backup_written_regular':len(regular),'cycle_resolved_events':len(cycles),'cycle_writes':len(cycle_writes),
            'backup_written_first_label':sum(b.get('previous') is None for b in written),
            'backup_written_increased':sum(b['value']>b['previous']+1e-9 for b in comparable),
            'backup_written_decreased':sum(b['value']<b['previous']-1e-9 for b in comparable),
            'backup_previous_note':'previous is the last valid writeback even after invalidation; null means first write in this graph generation',
            'backup_unresolved':sum(b.get('status')=='unresolved' for b in updates),
            'backup_cycle_unresolved':sum(b.get('status')=='cyclic_dependency' for b in updates),
            'selection_seconds':timing([d.get('selection_seconds') for d in rows]),
            'checks':sum(d.get('checks',0) for d in rows), 'accepted_links':sum(d.get('accepted_links',0) for d in rows),
            'map_reset_rounds':[{'round':d['round'],'simulation_time_s':ticks.get(d['round'],0)*dt} for d in decisions if d.get('map_reset')],
            'capacity_reset_rounds':[{'round':d['round'],'simulation_time_s':ticks.get(d['round'],0)*dt} for d in decisions if d.get('capacity_reset')],
            'last_graph':{k:rows[-1].get(k) for k in ('vertices','edges','pending','queue')} if rows else None}

def retreat_summary(d, planning):
    pos=d['states'][:,:,:2]; center=pos.mean(axis=1); dt=float(d['dt'])
    start=max(0,len(center)-1-int(round(100/dt)))
    xy=center[start:]; endpos=pos[-1]
    inside=lambda p: (p[...,0]>=14)&(p[...,0]<30.5)&(p[...,1]>14.5)&(p[...,1]<25.5)
    samples=[]
    for tick,m in list(zip(planning['ticks'],planning['frontend_metadata']))[-100:]:
        if tick>=len(center) or not inside(center[tick]): continue
        dec=m.get('statistics',{}).get('history_decision',{}); tree=m.get('tree',[]); cs=dec.get('candidates',[])
        if not tree: continue
        root=np.asarray(tree[0]['q'][:2]); byid={c['endpoint']:c for c in cs}
        def retreat(c):
            if not c or c['endpoint']>=len(tree): return None
            return float(root[0]-tree[c['endpoint']]['q'][0])
        vals=[retreat(c) for c in cs]; vals=[x for x in vals if x is not None]
        jh=retreat(byid.get(dec.get('baseline_endpoint'))); selected=retreat(byid.get(dec.get('selected_endpoint')))
        samples.append({'tick':tick,'time_s':tick*dt,'candidate_count':len(vals),'max_candidate_retreat_m':max(vals) if vals else None,
                        'jh_selected_retreat_m':jh,'selected_retreat_m':selected})
    return {'final_members_inside_cavity':int(inside(endpos).sum()), 'final_members_inside_indices':np.flatnonzero(inside(endpos)).tolist(),
            'last_100s_center_net_displacement_xy_m':(xy[-1]-xy[0]).tolist(),
            'last_100s_center_net_distance_m':float(np.linalg.norm(xy[-1]-xy[0])),
            'last_100s_center_path_length_m':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
            'last_100_planning_rounds_center_in_cavity':len(samples),
            'rounds_any_candidate_endpoint_retreat_at_least_0_5m':sum(x['max_candidate_retreat_m'] is not None and x['max_candidate_retreat_m']>=.5 for x in samples),
            'rounds_jh_endpoint_retreat_at_least_0_5m':sum(x['jh_selected_retreat_m'] is not None and x['jh_selected_retreat_m']>=.5 for x in samples),
            'rounds_selected_endpoint_retreat_at_least_0_5m':sum(x['selected_retreat_m'] is not None and x['selected_retreat_m']>=.5 for x in samples),
            'definition':'endpoint x retreat = tree root q[0] - candidate endpoint q[0]; not actual first-second motion. Cavity is 14 <= x < 30.5 and 14.5 < y < 25.5, for centers.',
            'rounds':samples}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning-run',type=Path,default=CASES['learning'], help='seed_0 directory containing learning run artifacts')
    parser.add_argument('--output',type=Path,default=None, help='output directory; defaults to the learning experiment directory')
    args=parser.parse_args()
    CASES['learning']=args.learning_run
    if args.output is None: args.output=args.learning_run.parent.parent
    args.output.mkdir(parents=True,exist_ok=True)
    missing=[str(p) for p in CASES.values() if not all((p/f).exists() for f in ('trajectory.npz','metrics.json','planning.json','configuration.json'))]
    if missing: raise SystemExit('Incomplete recorded runs: '+', '.join(missing))
    cases={name:load(path) for name,path in CASES.items()}
    ref=cases['depth1'][0]
    for name,(d,_,_,_) in cases.items():
        for k in ('rectangles','goal_center','reference'):
            if not np.array_equal(ref[k],d[k]): raise ValueError(f'{name}: mismatched {k}')
    expected=np.array([[14,14,31,14.5],[14,25.5,31,26],[30.5,14,31,26]])
    if not np.allclose(ref['rectangles'],expected): raise ValueError('Unexpected U geometry: review exit criteria')
    report={'method':{'source':'recorded actual trajectories and per-replan metadata', 'timing':'offline synchronous; wall computation time does not advance simulation', 'safety':'sampled planar obstacle and pair disk clearances; does not certify between samples', 'baseline':'J+H native closed-loop reference, matched seed and recorded configuration except decision method'},'runs':{}}
    fig,axes=plt.subplots(2,2,figsize=(14,12),sharex=True,sharey=True,constrained_layout=True)
    for ax,(name,(d,m,p,c)) in zip(axes.flat,cases.items()):
        s=summarize(d,m)
        s['recorded_collision_free'] = not (m.get('obstacle_collision') or m.get('robot_collision'))
        s['recorded_min_obstacle_clearance_m'] = m.get('min_obstacle_clearance')
        s['recorded_min_robot_center_distance_m'] = m.get('min_robot_distance')
        s['sampled_collision_free'] = bool(s['sampled_min_obstacle_disk_clearance_m'] >= -1e-9 and s['sampled_min_pair_disk_clearance_m'] >= -1e-9 and s['recorded_collision_free'])
        s['success_actual_and_sampled_safe'] = bool(s['actual_goal_first_s'] is not None and s['sampled_collision_free'])
        s['planning_seconds']=timing(p.get('planning_seconds',[]))
        s['configuration_differences_from_depth1_excluding_method'] = compare_config(cases['depth1'][3],c)
        s['source'] = str(CASES[name])
        n=min(len(d['states']),len(ref['states']))
        delta=np.max(np.linalg.norm(d['states'][:n,:,:2]-ref['states'][:n,:,:2],axis=-1),axis=1)
        ix=np.flatnonzero(delta>1e-9)
        s['first_actual_state_difference_from_depth1_s'] = None if not len(ix) else float(ix[0]*float(d['dt']))
        s['retreat_diagnostics']=retreat_summary(d,p)
        if name == 'learning': s['learning_diagnostics']=learning_summary(p,float(d['dt']))
        report['runs'][name]=s
        pos=d['states'][:,:,:2]; center=pos.mean(axis=1); dt=float(d['dt'])
        for x0,y0,x1,y1 in d['rectangles']: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#343a45'))
        for i in range(pos.shape[1]): ax.plot(pos[:,i,0],pos[:,i,1],lw=.8,alpha=.7,color=plt.cm.tab10(i),label=f'member {i}')
        ax.plot(center[:,0],center[:,1],color='black',lw=1.8,label='center')
        ax.scatter(*center[0],color='green',s=40,zorder=5); ax.scatter(*center[-1],marker='x',color='red',s=60,zorder=5); ax.scatter(*d['goal_center'],marker='*',color='purple',s=100,zorder=5)
        ax.plot([14,14],[14.5,25.5],':',color='#d68b00',lw=1.5)
        for t in np.arange(0,len(center)*dt,100):
            q=center[min(int(round(t/dt)),len(center)-1)];ax.annotate(f'{t:.0f}s',q,fontsize=7)
        ax.set(title=f"{name} | {s['duration_s']:.0f}s | goal={s['actual_goal_first_s'] is not None} | safe={s['sampled_collision_free']}",xlabel='x (m)',ylabel='y (m)',aspect='equal',xlim=(0,50),ylim=(8,32));ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7,loc='upper left',ncol=2)
    base=cases['J+H'][0]
    for name,(d,_,_,_) in cases.items():
        n=min(len(d['states']),len(base['states']))
        delta=np.max(np.linalg.norm(d['states'][:n,:,:2]-base['states'][:n,:,:2],axis=-1),axis=1)
        ix=np.flatnonzero(delta>1e-9)
        report['runs'][name]['first_actual_state_difference_from_jh_s'] = None if not len(ix) else float(ix[0]*float(d['dt']))
    fig.suptitle('Executed member trajectories; green=start, red=end, purple=goal',fontsize=13)
    fig.savefig(args.output/'persistent_learning_executed_paths.png',dpi=180);plt.close(fig)
    (args.output/'persistent_learning_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('duration_s','actual_goal_first_s','sampled_collision_free','final_max_member_goal_error_m','first_actual_state_difference_from_depth1_s')} for k,v in report['runs'].items()},indent=2))
if __name__=='__main__': main()
