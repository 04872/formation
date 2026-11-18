#!/usr/bin/env python3
"""Paper metrics on saved trajectories, including censored failures.

Quan et al. (ICRA 2022), Eqs. 1-4: complete graph with squared distance
weights, symmetric normalized Laplacian, squared Frobenius difference.
https://arxiv.org/html/2109.07682
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROUTES=('paper_uniform','paper_affine','rrt_affine')


def normalized_laplacian(points):
    """Complete-graph symmetric normalized Laplacian for (..., N, 2) points."""
    delta=points[..., :,None,:]-points[...,None,:,:]
    weights=np.sum(delta*delta,axis=-1)
    degree=weights.sum(axis=-1)
    return np.eye(points.shape[-2])-weights/np.sqrt(degree[..., :,None]*degree[...,None,:])


def formation_error(states, reference, assignment):
    """Squared Frobenius shape error, with desired slots in robot order."""
    target=reference[assignment]
    delta=normalized_laplacian(states)-normalized_laplacian(target)
    return np.sum(delta*delta,axis=(-2,-1))


def physical_obstacle_distance(states, rectangles, extent, wall_thickness):
    """Sampled minimum robot-center distance to obstacle sets and border walls."""
    output=np.empty(len(states))
    for first in range(0,len(states),2048):
        p=states[first:first+2048]
        x=np.maximum(np.maximum(rectangles[None,None,:,0]-p[:,:,None,0],p[:,:,None,0]-rectangles[None,None,:,2]),0.)
        y=np.maximum(np.maximum(rectangles[None,None,:,1]-p[:,:,None,1],p[:,:,None,1]-rectangles[None,None,:,3]),0.)
        rect=np.sqrt(x*x+y*y).min(axis=(1,2)) if len(rectangles) else np.full(len(p),np.inf)
        wall=np.minimum.reduce((p[:,:,0]-extent[0]-wall_thickness,
             extent[1]-wall_thickness-p[:,:,0],p[:,:,1]-extent[2]-wall_thickness,
             extent[3]-wall_thickness-p[:,:,1])).min(axis=1)
        output[first:first+len(p)]=np.minimum(rect,np.maximum(wall,0.))
    return output


def read_case(path,route,seed,common=None):
    metrics=json.loads((path/'metrics.json').read_text())
    if common is None:
        common=json.loads((path/'common_evaluation.json').read_text())
    with np.load(path/'trajectory.npz') as z:
        states=z['states'][:,:,:2];dt=float(z['dt']);radius=float(z['robot_radius'])
        reference=z['centered_reference'] if 'centered_reference' in z.files else z['reference']
        assignment=z['assignment_history'] if 'assignment_history' in z.files else np.tile(np.arange(len(reference)),(len(states),1))
        extent=[z['origin'][0],z['origin'][0]+z['occupancy'].shape[0]*float(z['resolution']),z['origin'][1],z['origin'][1]+z['occupancy'].shape[1]*float(z['resolution'])]
        distances=physical_obstacle_distance(states,z['rectangles'],extent,2*float(z['resolution']))
        errors=formation_error(states,reference,assignment)
        duration=(len(states)-1)*dt
        backend_updates=len(z['backend_ticks']) if 'backend_ticks' in z.files else None
    times=np.arange(len(states))*dt
    ef_mean=float(np.trapezoid(errors,times)/duration) if duration else float(errors[0])
    stages=metrics['timings_seconds']
    if backend_updates is None:
        plans=json.loads((path/'plans.json').read_text())['plans']
        backend=[p['timings']['backend'] for p in plans if p.get('attempts')]
        tsol_ms=float(np.mean(backend)*1000) if backend else None
        backend_updates=len(backend)
        tsol_note='mean per centralized backend invocation, including all candidate solves and any dynamics-specific refinement'
    else:
        # This total includes four cheap nominal-sequence advances per MPPI
        # update; it is an amortized per-optimization estimate, not isolated latency.
        tsol_ms=1000*stages['backend']/backend_updates if backend_updates else None
        tsol_note='backend total / MPPI optimization count; includes control-sequence reuse overhead'
    front_updates=common['planning_rounds']
    success=bool(common['success_with_safety_and_dynamics'])
    row={'route':route,'backend':common.get('backend','bezier' if route=='rrt_affine' else 'mppi'),'seed':seed,'termination':metrics['termination'],'success_with_safety_and_dynamics':success,
         'censored':not success,'observed_duration_s':duration,'t_fly_s':duration if success else None,
         'robot_radius_m':radius,
         'd_safe_center_to_obstacle_m':float(distances.min()),'d_safe_body_clearance_m':float(distances.min()-radius),
         'e_f_time_average':ef_mean,'e_f_max_sampled':float(errors.max()),
         't_sol_backend_amortized_ms':tsol_ms,'backend_updates':backend_updates,'t_sol_definition':tsol_note,
         't_front_mean_ms':1000*stages['frontend']/front_updates if front_updates else None,
         't_front_total_s':stages['frontend'],'t_back_total_s':stages['backend'],
         'sfc_total_s':stages['sfc'],'perception_total_s':stages['perception'],
         'execution_total_s':stages['execution'],'wall_time_s':common['compute_time_s'],
         'result_dir':str(path)}
    return row,times,errors,distances


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/three_routes/paper_metrics_20261117'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();out=args.output or args.root;out.mkdir(parents=True,exist_ok=True)
    traces=out/'metric_traces';traces.mkdir(exist_ok=True);rows=[]
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='col',constrained_layout=True)
    labels={'paper_uniform':'Paper uniform + MPPI','paper_affine':'Paper affine + MPPI','rrt_affine':'Local RRT* + Bezier'}
    rrt_config=json.loads((args.root/'rrt_affine/seed_0/configuration.json').read_text())
    if rrt_config.get('backend')=='mppi':labels['rrt_affine']='Local RRT* + MPPI'
    colors={'paper_uniform':'#d95f02','paper_affine':'#7570b3','rrt_affine':'#1b9e77'}
    for route in ROUTES:
        for seed in (0,1):
            path=args.root/route/f'seed_{seed}';row,t,e,d=read_case(path,route,seed);rows.append(row)
            np.savez_compressed(traces/f'{route}_seed{seed}.npz',time_s=t,e_f=e,d_obs_center_m=d,d_obs_body_m=d-row['robot_radius_m'])
            label=labels[route]+(' (censored)' if row['censored'] else '')
            for ax,series in ((axes[0,seed],e),(axes[1,seed],d)):
                ax.plot(t,series,label=label,color=colors[route],linewidth=1.2)
                if row['censored']:ax.plot(t[-1],series[-1],'x',color=colors[route],markersize=6)
    for seed in (0,1):
        axes[0,seed].set_title(f'Seed {seed}');axes[0,seed].set_ylabel('Formation similarity $e_f$');axes[1,seed].set_ylabel('Center-to-obstacle distance (m)');axes[1,seed].set_xlabel('Simulation time (s)')
        for ax in axes[:,seed]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'paper_metrics_timeseries.png',dpi=160);plt.close(fig)
    document={'definitions':{'e_f':'Quan et al. complete graph squared pair distances, squared Frobenius difference of symmetric normalized Laplacians; desired slots follow recorded assignment. Invariant to translation, rotation and uniform scale, but penalizes anisotropic affine deformation. https://arxiv.org/html/2109.07682',
      'e_f_average':'trapezoidal time integral over observed 0.02 s states divided by observed duration','e_f_max':'maximum of recorded samples; not a continuous-time upper bound',
      'd_safe':'minimum recorded center-to-physical-rectangle/border distance; paper does not specify body-size convention; separate body clearance subtracts radius. Continuous safety bounds are in independent audits.',
      't_fly':'observed simulation duration only for common safety-and-dynamics success; null for failure, whose remaining columns are censored-prefix statistics',
      't_sol':'local proxy: full-team backend time per optimization update; MPPI includes cheap control-sequence reuse. Paper does not resolve whether t_sol includes frontend; CPU serial full-team vs paper distributed GPU times are not directly comparable.',
      'timing':'front, backend, SFC and perception are separately reported; SFC includes lazy map indexing. Runtime concurrency, contention and any reruns are recorded in run_protocol.json.'},'runs':rows}
    (out/'paper_metrics.json').write_text(json.dumps(document,indent=2,ensure_ascii=False)+'\n')
    with (out/'paper_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
