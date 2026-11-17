#!/usr/bin/env python3
"""Compare all routes using common goal, dynamics and safety criteria."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(root):
    rows=[];reference_case=None
    expected={f'{route}/seed_{seed}' for route in ('paper_uniform','paper_affine','rrt_affine') for seed in (0,1)}
    actual={str(p.parent.relative_to(root)) for p in root.glob('*/*/trajectory.npz')}
    if actual != expected:
        raise ValueError(f'Expected six matched runs; missing={expected-actual}, extra={actual-expected}')
    for p in sorted(root.glob('*/*/trajectory.npz')):
        case=p.parent;config=json.loads((case/'configuration.json').read_text())
        metrics=json.loads((case/'metrics.json').read_text())
        audit=json.loads((case/'audit.json').read_text())
        if isinstance(audit,list):audit=audit[0]
        with np.load(p,allow_pickle=False) as archive:
            d={key:archive[key] for key in archive.files}
        shared=(d['occupancy'],d['rectangles'],d['start_center'],d['goal_center'],d['states'][0,:,:2])
        if reference_case is None:reference_case=shared
        if not all(np.array_equal(x,y) for x,y in zip(shared,reference_case)):
            raise ValueError(f'Map, endpoints or initial formation mismatch: {case}')
        states=d['states'];dt=float(d['dt']);n=states.shape[1]
        is_paper=config['route'].startswith('paper_')
        if is_paper:
            targets=d['goal']+d['reference'][d['assignment_history']]
            speed=np.abs(d['velocity_history'])
            safe=bool(audit['continuous_safety_certified_by_audit'])
            clearance=audit['continuous_obstacle_clearance_lower_bound_m']
            pair_distance=audit['continuous_pair_clearance_lower_bound_m']+2*float(d['robot_radius'])
            w=np.abs(d['controls'][:,:,1]);ac=d['acceleration_controls'][:,:,0]
            peak_v=np.maximum(speed[:-1],speed[1:])
            max_a=float(np.sqrt(ac*ac+(peak_v*w)**2).max(initial=0))
            dynamic_ok=(audit['v_max_satisfied'] and audit['omega_max_satisfied'] and max_a<=1.+1e-8 and
                        audit['velocity_update_residual_max_m_s']<1e-8 and
                        audit['exact_unicycle_position_residual_max_m']<1e-8)
            regions=audit.get('regions',{})
            region_ok=all(regions.get(k,0)==0 for k in ('invalid_region_anchors','uncertified_guidance_edges','uncertified_actual_state_joins'))
            backend_updates=len(d['backend_ticks'])
            planning_count=len(d['planning_ticks'])
            frontend_times=[t['frontend'] for t in metrics['planning_stage_seconds']]
            native_wall=metrics['simulation_wall_seconds']
            backend_period=float(d['backend_period'])
            frontend_times_actual=d['planning_ticks']*dt
            backend_times_actual=d['backend_ticks']*dt
        else:
            targets=np.broadcast_to(d['goal_center']+d['reference'],states[:,:,:2].shape)
            speed=np.r_[np.zeros((1,n)),np.abs(d['controls'][:,:,0])]
            safe=bool(audit['passed']);region_ok=safe
            clearance=audit['known_grid_clearance_lower_bound']
            pair_distance=audit['minimum_pair_distance'];max_a=audit['max_acceleration']
            dynamic_ok=bool(audit['passed'])
            plans=json.loads((case/'plans.json').read_text())['plans']
            planning_count=len(plans);backend_updates=sum(bool(x.get('attempts')) for x in plans)
            frontend_times=[x['timings']['frontend'] for x in plans if 'timings' in x]
            native_wall=metrics['elapsed_seconds'];backend_period=float(d['replan_period'])
            frontend_times_actual=np.array([x['time'] for x in plans])
            backend_times_actual=np.array([x['time'] for x in plans if x.get('attempts')])
        if dt!=.02 or backend_period!=1. or config['frontend_period']!=1. or config['perception_period']!=.2:
            raise ValueError(f'Frequency mismatch: {case}')
        err=np.linalg.norm(states[:,:,:2]-targets,axis=-1).max(axis=1)
        hit=np.flatnonzero((err<.4)&(speed.max(axis=1)<.1))
        endpoint_hit=bool(err[-1]<.4 and speed[-1].max()<.1)
        center_dist=np.linalg.norm(states[:,:,:2].mean(axis=1)-d['goal_center'],axis=1)
        nearby=np.flatnonzero(center_dist<5.)
        stage=metrics['timings_seconds']
        row={'route':config['route'],'seed':config['seed'],'termination':metrics['termination'],
             'common_goal_met_at_end':endpoint_hit,'success_with_safety_and_dynamics':bool(endpoint_hit and safe and dynamic_ok and region_ok),
             'common_first_arrival_s':float(hit[0]*dt) if len(hit) else None,
             'first_within_5m_s':float(nearby[0]*dt) if len(nearby) else None,
             'simulated_time_s':(len(states)-1)*dt,'compute_time_s':native_wall,
             'final_center_error_m':float(center_dist[-1]),'final_max_robot_error_m':float(err[-1]),
             'final_max_speed_m_s':float(speed[-1].max()),
             'mean_robot_path_length_m':float(np.linalg.norm(np.diff(states[:,:,:2],axis=0),axis=-1).sum(axis=0).mean()),
             'continuous_safety_passed':safe,'dynamic_limits_passed':bool(dynamic_ok),'sfc_audit_passed':region_ok,
             'continuous_grid_clearance_lower_m':clearance,'minimum_pair_distance_m':pair_distance,
             'maximum_acceleration_m_s2':max_a,'planning_rounds':planning_count,'backend_updates':backend_updates,
             'frontend_mean_ms':float(np.mean(frontend_times)*1000),'frontend_p95_ms':float(np.percentile(frontend_times,95)*1000),
             'frontend_median_interval_s':float(np.median(np.diff(frontend_times_actual))),
             'backend_median_interval_s':float(np.median(np.diff(backend_times_actual))),
             'frontend_min_interval_s':float(np.min(np.diff(frontend_times_actual))),
             'backend_min_interval_s':float(np.min(np.diff(backend_times_actual))),
             'backend_mean_per_update_ms':stage['backend']/backend_updates*1000,
             'total_perception_s':stage['perception'],'total_sfc_s':stage['sfc'],
             'total_frontend_s':stage['frontend'],'total_backend_s':stage['backend'],
             'execution_hz':50,'perception_hz':5,'frontend_nominal_hz':1,'backend_nominal_hz':1,
             'result_dir':str(case)}
        rows.append(row)
        (case/'common_evaluation.json').write_text(json.dumps(row,indent=2))
    report={'protocol':{'map':'same 50 square pillars, map seed1, 50x40m; triangle6 spacing3.5m',
                        'rates_hz':{'execution':50,'perception':5,'frontend':1,'backend':1},
                        'max_time_s':500,'common_goal':'every assigned robot <0.4m, every speed <0.1m/s',
                        'fairness_notes':['All routes use the same optimized static observed-grid/FIRI/distance-query implementation.',
                            'Paper routes retain 20x35x256 frontend sampling and 1120x35 MPPI; RRT uses250 iterations. Search work budgets are deliberately reported, not equated.',
                            'MPPI prediction/control knots remain0.2s (7s prediction horizon); optimize every1s and execute five knots at0.02s substeps.',
                            'Centralized execution is an analytic continuous flat-output trajectory; MPPI execution is midpoint unicycle integration. No common noisy tracking-controller claim.',
                            'Periodic rates are nominal simulation-time schedules, with terminal commitment and tail-recovery events for RRT. Solver wall time does not advance simulated time.']},'runs':rows}
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(summarize(args.root),indent=2))
