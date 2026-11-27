#!/usr/bin/env python3
"""Compare two completed U-scene closed-loop runs from recorded execution states."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path('results/three_routes/local_backup_closed_loop')

def first(mask, dt):
    i = np.flatnonzero(mask)
    return None if not len(i) else float(i[0]*dt)

def load(root, name):
    p = root/name/'rrt_affine/seed_0'
    with np.load(p/'trajectory.npz', allow_pickle=False) as a:
        d = {k: a[k] for k in a.files}
    return d, json.loads((p/'metrics.json').read_text()), json.loads((p/'planning.json').read_text())

def summarize(d, metrics):
    pos=d['states'][:,:,:2]; center=pos.mean(axis=1); dt=float(d['dt']); r=float(d['robot_radius'])
    # Center and all-member conditions intentionally do not imply disk clearance.
    in_y=(center[:,1]>14.5)&(center[:,1]<25.5)
    inside=(center[:,0]>=14)&(center[:,0]<30.5)&in_y
    entered=first(inside,dt)
    all_inside=(pos[:,:,0]>=14).all(axis=1)&(pos[:,:,0]<30.5).all(axis=1)&(pos[:,:,1]>14.5).all(axis=1)&(pos[:,:,1]<25.5).all(axis=1)
    after=np.arange(len(pos))*dt >= entered if entered is not None else np.zeros(len(pos),dtype=bool)
    exit_center=first(after&(center[:,0]<14)&in_y,dt)
    exit_all=first(after&(pos[:,:,0]<14).all(axis=1),dt)
    after_exit=np.arange(len(pos))*dt > exit_center if exit_center is not None else np.zeros(len(pos),dtype=bool)
    reentered=first(after_exit&inside,dt)
    post_entry=center[after]
    running=np.maximum.accumulate(center[:,0]); retreats=running-center[:,0]; j=int(np.argmax(retreats)); i=int(np.argmax(center[:j+1,0]))
    after_entry_retreat=None
    if entered is not None:
        x=center[int(round(entered/dt)):,0]
        after_entry_retreat=float((np.maximum.accumulate(x)-x).max())
    clear=np.full(pos.shape[:2],np.inf)
    for x0,y0,x1,y1 in d['rectangles']:
        dx=np.maximum(np.maximum(x0-pos[:,:,0],pos[:,:,0]-x1),0)
        dy=np.maximum(np.maximum(y0-pos[:,:,1],pos[:,:,1]-y1),0)
        clear=np.minimum(clear,np.hypot(dx,dy)-r)
    pair=np.inf
    for a in range(pos.shape[1]):
        for b in range(a): pair=min(pair,float(np.linalg.norm(pos[:,a]-pos[:,b],axis=1).min())-2*r)
    # Track each member independently: an earlier member can move around the
    # wall and return to x >= 14 before its teammates exit the opening.
    member_inside=(pos[:,:,0]>=14)&(pos[:,:,0]<30.5)&(pos[:,:,1]>14.5)&(pos[:,:,1]<25.5)
    member_exit_masks=[]; member_events=[]
    for k in range(pos.shape[1]):
        entry=first(member_inside[:,k],dt)
        after_member_entry=np.arange(len(pos))*dt > entry if entry is not None else np.zeros(len(pos),dtype=bool)
        exit_mask=after_member_entry&(pos[:,k,0]<14)&(pos[:,k,1]>14.5)&(pos[:,k,1]<25.5)
        exit_time=first(exit_mask,dt)
        after_member_exit=np.arange(len(pos))*dt > exit_time if exit_time is not None else np.zeros(len(pos),dtype=bool)
        member_exit_masks.append(np.maximum.accumulate(exit_mask))
        member_events.append({'member_index':k,'first_inside_s':entry,'first_exit_through_opening_s':exit_time,
          'first_reentry_s':first(after_member_exit&member_inside[:,k],dt),
          'sampled_min_obstacle_disk_clearance_m':float(clear[:,k].min())})
    escaped=np.stack(member_exit_masks,axis=1).all(axis=1)
    all_outside=escaped&(~member_inside).all(axis=1)
    all_outside_time=first(all_outside,dt)
    target=d['goal_center']+d.get('centered_reference',d['reference'])
    if 'assignment_history' in d: target=target[d['assignment_history']]
    errors=np.linalg.norm(pos-target,axis=-1).max(axis=1)
    speeds=d.get('velocity_history')
    if 'uav_states' in d: speeds=np.linalg.norm(d['uav_states'][:,:,3:6],axis=-1)
    hit=(errors<.4)&(np.max(np.abs(speeds),axis=1)<.1)
    known=d.get('known_history')
    side_lower=first(after_exit&(center[:,1]<14),dt)
    side_upper=first(after_exit&(center[:,1]>26),dt)
    beyond=first(after_exit&(center[:,0]>31)&((center[:,1]<14)|(center[:,1]>26)),dt)
    if hit.any() and beyond is not None:
        progress='detoured_and_reached_goal'
    elif hit.any():
        progress='reached_goal_without_observed_detour'
    elif exit_all is not None:
        progress='all_members_exited_opening_without_goal'
    elif exit_center is not None:
        progress='center_exited_opening_without_goal'
    elif entered is not None:
        progress='entered_without_exiting_opening'
    else:
        progress='center_not_entered'
    return {'progress_classification':progress,'duration_s':(len(pos)-1)*dt,'termination':metrics.get('termination'),'reported_success':metrics.get('success'),
      'center_first_inside_s':entered,'all_members_first_inside_s':first(all_inside,dt),
      'center_first_exited_opening_s':exit_center,'all_members_first_exited_opening_s':exit_all,
      'center_first_reentered_after_exit_s':reentered,
      'members':member_events,'all_members_first_outside_cavity_after_individual_opening_exits_s':all_outside_time,
      'all_members_outside_sample_min_obstacle_disk_clearance_m':float(clear[int(round(all_outside_time/dt))].min()) if all_outside_time is not None else None,
      'all_members_outside_sample_positions_xy_m':pos[int(round(all_outside_time/dt))].tolist() if all_outside_time is not None else None,
      'all_members_outside_criterion':'each member previously crossed the opening x < 14 with 14.5 < y < 25.5, and no member center is in 14 <= x < 30.5, 14.5 < y < 25.5; sampled disk clearances are reported separately',
      'center_min_x_after_entry_m':float(post_entry[:,0].min()) if len(post_entry) else None,
      'center_first_beyond_lower_side_after_exit_s':side_lower,'center_first_beyond_upper_side_after_exit_s':side_upper,
      'center_max_x_m':float(center[:,0].max()),'center_final_xy_m':center[-1].tolist(),
      'center_first_beyond_bottom_outside_s':beyond,
      'all_members_first_beyond_bottom_s':first((pos[:,:,0]>31).all(axis=1),dt),
      'max_center_retreat_x_m':float(retreats[j]),'max_center_retreat_after_entry_x_m':after_entry_retreat,'retreat_peak_s':i*dt,'retreat_end_s':j*dt,
      'actual_goal_first_s':first(hit,dt),'actual_goal_at_end':bool(hit[-1]),'final_max_member_goal_error_m':float(errors[-1]),
      'sampled_min_obstacle_disk_clearance_m':float(clear.min()),'sampled_min_pair_disk_clearance_m':pair,
      'known_grid_fraction_final':float(np.mean(known[-1])) if known is not None and len(known) else None,
      'goal_criterion':'every assigned member position error < 0.4 m and speed < 0.1 m/s',
      'safety_criterion':'sampled disk clearances, no continuous-time certification'}

def comparison(a,b,pa,pb):
    assert float(a['dt'])==float(b['dt'])
    dt=float(a['dt']); n=min(len(a['states']),len(b['states']))
    delta=np.linalg.norm(a['states'][:n,:,:2]-b['states'][:n,:,:2],axis=-1).max(axis=1)
    result={'first_member_position_difference_gt_1e-9_s':first(delta>1e-9,dt),
      'first_member_position_difference_gt_1cm_s':first(delta>.01,dt),
      'first_member_position_difference_gt_10cm_s':first(delta>.1,dt),'max_member_position_difference_m':float(delta.max()), 'common_initial_positions':bool(np.array_equal(a['states'][0],b['states'][0]))}
    if 'uav_controls' in a and 'uav_controls' in b:
        n=min(len(a['uav_controls']),len(b['uav_controls']))
        cd=np.max(np.abs(a['uav_controls'][:n]-b['uav_controls'][:n]),axis=(1,2))
        result['first_uav_command_difference_gt_1e-9_s']=first(cd>1e-9,dt)
    common_ticks=sorted(set(pa['ticks']) & set(pb['ticks']))
    ia={tick:i for i,tick in enumerate(pa['ticks'])}; ib={tick:i for i,tick in enumerate(pb['ticks'])}
    selection_diffs=[]; path_diffs=[]; backup_diffs=[]; selected_value_diffs=[]; strict_witnesses=[]; candidate_tree_mismatch=[]; candidate_total=0
    for tick in common_ticks:
        i,j=ia[tick],ib[tick]
        da=pa['frontend_metadata'][i].get('statistics',{}).get('history_decision',{})
        db=pb['frontend_metadata'][j].get('statistics',{}).get('history_decision',{})
        ca={c['endpoint']:c for c in da.get('candidates',[])}; cb={c['endpoint']:c for c in db.get('candidates',[])}
        candidate_total+=len(ca)
        if set(ca)!=set(cb) or any(ca[k].get('route')!=cb[k].get('route') or abs(ca[k]['tree_cost']-cb[k]['tree_cost'])>1e-9 or abs(ca[k]['original_score']-cb[k]['original_score'])>1e-9 for k in set(ca)&set(cb)):
            candidate_tree_mismatch.append(tick*dt)
        for k in set(ca)&set(cb):
            if ca[k].get('v1') is not None and cb[k].get('v2') is not None and abs(ca[k]['v1']-cb[k]['v2'])>1e-9:
                backup_diffs.append((tick*dt,k))
                if k==da.get('selected_endpoint'): selected_value_diffs.append((tick*dt,k))
            w=cb[k].get('witness2')
            if w is not None and w.get('successor') is not None: strict_witnesses.append((tick*dt,k))
        if da.get('selected_route')!=db.get('selected_route') or da.get('selected_endpoint')!=db.get('selected_endpoint'):
            selection_diffs.append(tick*dt)
        x=np.asarray(pa['paths'][i]);y=np.asarray(pb['paths'][j])
        if x.shape!=y.shape or not np.allclose(x,y,rtol=0,atol=1e-9): path_diffs.append(tick*dt)
    result.update({'candidate_tree_metadata_mismatch_rounds':len(candidate_tree_mismatch),
                   'candidate_total':candidate_total, 'candidate_backup_values_different':len(backup_diffs),
                   'candidate_backup_difference_rounds':len({x[0] for x in backup_diffs}),
                   'selected_backup_values_different':len(selected_value_diffs),
                   'strict_two_layer_witnesses':len(strict_witnesses),
                   'strict_two_layer_witness_rounds':len({x[0] for x in strict_witnesses})})
    result.update({'common_planning_rounds':len(common_ticks),'selected_route_different_rounds':len(selection_diffs),
                   'selected_route_difference_times_s':selection_diffs,'sampled_guidance_path_different_rounds':len(path_diffs),
                   'sampled_guidance_path_difference_times_s':path_diffs})
    def window(d,p,tick):
        i=int(np.searchsorted(p['ticks'],tick,side='right')-1)
        path=np.asarray(p['paths'][i]); pred=float(d['prediction_dt'])
        period=int(round((p['ticks'][1]-p['ticks'][0])*dt/pred))
        offset=int(round(tick*dt/pred))%period
        if not len(path): return np.repeat(d['states'][tick:tick+1,:,:2],35,axis=0)
        q=path[np.minimum(np.arange(35)+offset,len(path)-1)]
        targets=q[:,:2,None]+np.einsum('tij,nj->tin',q[:,2:].reshape(-1,2,2),d['reference'])
        targets=targets.transpose(0,2,1)
        return targets[:,np.asarray(p['assignments'][i]),:]
    common_backend=sorted(set(a['backend_ticks'].tolist()) & set(b['backend_ticks'].tolist()))
    window_diffs=[tick*dt for tick in common_backend if not np.allclose(window(a,pa,tick),window(b,pb,tick),rtol=0,atol=1e-9)]
    result.update({'common_backend_optimizations':len(common_backend),'reconstructed_35_point_mppi_windows_different':len(window_diffs),
                   'reconstructed_35_point_mppi_window_difference_times_s':window_diffs,
                   'window_reconstruction':'uses saved path, assignment, reference, timing and simulation.py indices; constant altitude omitted'})
    for tick in common_ticks:
        i,j=ia[tick],ib[tick]
        ma,mb=pa['frontend_metadata'][i],pb['frontend_metadata'][j]
        da=ma.get('statistics',{}).get('history_decision',{}); db=mb.get('statistics',{}).get('history_decision',{})
        sa=da.get('selected_endpoint'); sb=db.get('selected_endpoint')
        if sa is None or sb is None or (sa==sb and da.get('selected_route')==db.get('selected_route')): continue
        ca={c['endpoint']:c for c in da.get('candidates',[])}; cb={c['endpoint']:c for c in db.get('candidates',[])}
        ids=set(ca)==set(cb) and bool(ca)
        same=ids and all(ca[k].get('route') == cb[k].get('route') and abs(ca[k].get('tree_cost',float('inf'))-cb[k].get('tree_cost',float('-inf')))<1e-8 and abs(ca[k].get('original_score',float('inf'))-cb[k].get('original_score',float('-inf')))<1e-8 for k in ca)
        result['first_selected_endpoint_difference']={'round':i,'time_s':tick*dt,'matching_candidate_ids_routes_costs_original_scores':same,
          'actual_states_match_at_round':bool(np.allclose(a['states'][tick],b['states'][tick],rtol=0,atol=1e-9)),
          'frontend_roots_match':bool(np.allclose(pa['initialization_q'][i],pb['initialization_q'][j],rtol=0,atol=1e-9)),
          'depth1_selected':sa,'depth2_selected':sb,'depth1_candidates_at_both_selected':{str(k):ca.get(k) for k in (sa,sb)},
          'depth2_candidates_at_both_selected':{str(k):cb.get(k) for k in (sa,sb)}}
        break
    return result

def plot(root, data, reports):
    fig, axs=plt.subplots(1,2,figsize=(15,6),sharex=True,sharey=True,constrained_layout=True)
    ymin=min(8., *(float(d['states'][:,:,1].min())-1 for _,d in data))
    ymax=max(32., *(float(d['states'][:,:,1].max())+1 for _,d in data))
    for ax,(name,d),rep in zip(axs,data,reports):
        pos=d['states'][:,:,:2]; center=pos.mean(axis=1); dt=float(d['dt'])
        for x0,y0,x1,y1 in d['rectangles']: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#323b4a'))
        colors=plt.cm.tab10(np.linspace(0,1,pos.shape[1]))
        for i,c in enumerate(colors): ax.plot(pos[:,i,0],pos[:,i,1],lw=.8,alpha=.55,color=c)
        ax.plot(center[:,0],center[:,1],color='black',lw=2,label='Actual center')
        ax.scatter(*center[0],marker='o',s=60,color='#17944e',label='Start'); ax.scatter(*center[-1],marker='x',s=85,color='#cf2525',label='Final')
        ax.scatter(*d['goal_center'],marker='*',s=140,color='#9054bd',label='Goal center')
        ax.plot([14,14],[14.5,25.5],':',color='#d68b00',lw=2,label='Opening x=14')
        for t in np.arange(0,len(center)*dt,50):
            p=center[min(int(round(t/dt)),len(center)-1)];ax.annotate(f'{t:.0f}s',p,fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set(title=f"{name}: {rep['termination']} | {rep['duration_s']:.0f} s",xlabel='x (m)',ylabel='y (m)',aspect='equal',xlim=(0,50),ylim=(ymin,ymax));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Executed member paths (color) and actual formation center (black)',fontsize=13)
    fig.savefig(root/'executed_trajectory_comparison.png',dpi=180);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',nargs='?',type=Path,default=ROOT);args=ap.parse_args();root=args.root
    cases=[load(root,n) for n in ('depth1','depth2')]
    a,b=cases[0][0],cases[1][0]
    for k in ('rectangles','goal_center','reference'):
        if not np.array_equal(a[k],b[k]): raise ValueError(f'Unmatched {k}')
    expected=np.array([[14,14,31,14.5],[14,25.5,31,26],[30.5,14,31,26]])
    if not np.allclose(a['rectangles'],expected): raise ValueError('Unexpected U geometry; review opening and exit tests')
    reports=[summarize(d,m) for d,m,_ in cases]
    report={'geometry':{'opening_x_m':14,'opening_y_m':[14.5,25.5],'bottom_wall_x_m':[30.5,31]},
      'depth1':reports[0],'depth2':reports[1], 'paired':comparison(a,b,cases[0][2],cases[1][2])}
    (root/'execution_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(root,[('depth1',a),('depth2',b)],reports)
    print(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
