#!/usr/bin/env python3
"""Report matched-map UAV MPPI comparisons without modifying run artifacts.

Usage: python scripts/diagnostics/report_paper_density.py EXPERIMENT_ROOT
Each EXPERIMENT_ROOT/{sparse,medium,dense}_mapN has route/seed_0,seed_1.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from paper_metrics import read_case

MAIN = ('paper_affine', 'rrt_affine')
UNIFORM = 'paper_uniform'
ROUTES = (*MAIN, UNIFORM)
FIELDS = ('d_safe_center_to_obstacle_m', 'e_f_time_average', 'e_f_max_sampled',
          't_fly_s', 't_front_mean_ms', 't_sol_backend_amortized_ms')
LABELS = {'paper_affine': 'Paper affine + MPPI', 'rrt_affine': 'RRT affine + MPPI'}
COLORS = {'paper_affine': '#7570b3', 'rrt_affine': '#1b9e77'}


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def case_common(path, route, seed):
    """Apply the existing common UAV success, audit and endpoint conditions."""
    m = json.loads((path/'metrics.json').read_text())
    cfg = json.loads((path/'configuration.json').read_text())
    audit = json.loads((path/'audit.json').read_text())
    if isinstance(audit, list):
        audit = audit[0]
    r = audit['regions']
    safety = bool(audit['continuous_safety_certified_by_audit'])
    dynamics = all(audit[k] for k in ('v_max_satisfied', 'a_max_satisfied', 'euler_model_satisfied'))
    sfc = r['regions_checked'] > 0 and all(r[k] == 0 for k in ('invalid_region_anchors', 'uncertified_guidance_edges', 'uncertified_actual_state_joins'))
    with np.load(path/'trajectory.npz', allow_pickle=False) as z:
        target = z['goal'] + z['reference'][z['assignment_history'][-1]]
        max_err = float(np.max(np.linalg.norm(z['states'][-1, :, :2]-target, axis=1)))
        max_speed = float(np.max(np.linalg.norm(z['uav_states'][-1, :, 3:6], axis=1)))
        center_err = float(np.linalg.norm(z['states'][-1, :, :2].mean(axis=0)-z['goal_center']))
        plans = len(z['planning_ticks'])
    endpoint = max_err < .4 and max_speed < .1
    common = {'planning_rounds': plans, 'backend': cfg['backend'],
              'success_with_safety_and_dynamics': bool(endpoint and safety and dynamics and sfc),
              'compute_time_s': m['simulation_wall_seconds']}
    reasons = []
    if not endpoint:
        reasons.append('endpoint_conditions_not_met')
    if not safety:
        reasons.append('continuous_safety_audit_failed')
    if not dynamics:
        reasons.append('dynamics_audit_failed')
    if not sfc:
        reasons.append('sfc_audit_failed')
    if m['termination'] != 'goal_reached':
        reasons.insert(0, m['termination'])
    return common, {'failure_reason': ';'.join(reasons), 'sfc_error': m.get('sfc_error', ''), 'common_goal_met_at_end': endpoint,
                    'continuous_safety_passed': safety, 'dynamic_limits_passed': dynamics,
                    'sfc_audit_passed': sfc, 'final_center_error_m': center_err,
                    'final_max_robot_error_m': max_err, 'final_max_speed_m_s': max_speed}


def identity(path):
    cfg = json.loads((path/'configuration.json').read_text())
    with np.load(path/'trajectory.npz', allow_pickle=False) as z:
        keys = ('occupancy', 'rectangles', 'start_center', 'goal_center', 'reference',
                'centered_reference', 'origin', 'resolution', 'robot_radius')
        geometry = {k: z[k].copy() for k in keys}
        geometry['initial_positions'] = z['states'][0, :, :2].copy()
    return cfg, geometry


def overview(scenarios, root):
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 3.5*len(scenarios)), squeeze=False,
                             constrained_layout=True)
    for ax, (name, geom, cfg) in zip(axes[:, 0], scenarios):
        for xmin, ymin, xmax, ymax in geom['rectangles']:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor='black'))
        start, goal = geom['start_center'], geom['goal_center']
        ax.scatter(*start, s=60, c='#1b9e77', label='Start center', zorder=3)
        ax.scatter(*goal, s=60, c='#d95f02', marker='*', label='Goal center', zorder=3)
        origin=geom['origin']; size=np.array(geom['occupancy'].shape)*float(geom['resolution'])
        ax.set(xlim=(origin[0], origin[0]+size[0]), ylim=(origin[1], origin[1]+size[1]),
               title=f"{name}: {len(geom['rectangles'])} pillars, map seed {cfg['map_seed']}",
               xlabel='x (m)', ylabel='y (m)')
        ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8)
    fig.savefig(root/'maps.png', dpi=150); plt.close(fig)


def plot_summary(summaries, out):
    specs = [('d_safe_center_to_obstacle_m', 'Min sampled center-obstacle distance (m)'),
             ('e_f_time_average', 'Mean formation error'), ('e_f_max_sampled', 'Max formation error'),
             ('t_fly_s', 'Flight duration (s)'), ('t_front_mean_ms', 'Frontend time / plan (ms)'),
             ('t_sol_backend_amortized_ms', 'Backend time / update (ms)')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    pos=np.arange(len(summaries))
    for ax, (metric, ylabel) in zip(axes.flat, specs):
        for j, route in enumerate(MAIN):
            values=[s[f'{route}_{metric}_mean'] if s[f'{route}_{metric}_mean'] is not None else np.nan for s in summaries]
            ax.bar(pos+(j-.5)*.34, values, width=.34, color=COLORS[route], label=LABELS[route])
        ax.set_xticks(pos, [f"{s['density']}\npaired n={s['common_success_pairs']}" for s in summaries]); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Paired common-success means; missing bars indicate no common successes')
    fig.savefig(out/'main_paired_metrics.png', dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, help='Report directory (default ROOT/report)')
    args=parser.parse_args(); root=args.root; out=args.output or root/'report'; out.mkdir(parents=True,exist_ok=True)
    scenarios=sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'(sparse|medium|dense)_map\d+',p.name))
    if not scenarios:
        raise ValueError(f'No scenario directories under {root}')
    rows=[]; maps=[]; global_geometry=None; density_obstacle_counts={}
    for scenario in scenarios:
        density,map_name=scenario.name.rsplit('_map',1); ref_geom=None; ref_cfg=None; observed=0
        for route in ROUTES:
            for seed in (0,1):
                path=scenario/route/f'seed_{seed}'
                base={'density':density,'map_seed':int(map_name),'route':route,'seed':seed,
                      'result_dir':str(path)}
                missing=[f for f in ('configuration.json','metrics.json','trajectory.npz','audit.json') if not (path/f).is_file()]
                if missing:
                    rows.append({**base,'record_status':'incomplete','success_with_safety_and_dynamics':False,
                                 'failure_reason':'missing: '+', '.join(missing),'censored':True})
                    continue
                cfg,geom=identity(path)
                if cfg['map_seed']!=int(map_name) or cfg['seed']!=seed or cfg['route']!=route:
                    raise ValueError(f'Configuration/scenario mismatch: {path}')
                if cfg['backend']!='mppi' or cfg['dynamics']!='uav':
                    raise ValueError(f'Expected UAV MPPI case: {path}')
                if len(geom['rectangles']) != cfg['obstacle_count']:
                    raise ValueError(f'Obstacle count mismatch: {path}')
                count=density_obstacle_counts.setdefault(density, cfg['obstacle_count'])
                if count != cfg['obstacle_count']:
                    raise ValueError(f'Inconsistent obstacle count for {density}: {path}')
                shared={k:geom[k] for k in ('start_center','goal_center','reference','centered_reference',
                                            'origin','resolution','robot_radius','initial_positions')}
                if global_geometry is None:global_geometry=shared
                elif any(not np.array_equal(shared[k],global_geometry[k]) for k in shared):
                    raise ValueError(f'Endpoints or initial formation mismatch across maps: {path}')
                if ref_geom is None:ref_geom,ref_cfg=geom,cfg
                elif any(not np.array_equal(geom[k],ref_geom[k]) for k in geom):
                    raise ValueError(f'Map, physical geometry, endpoints or initial formation mismatch: {path}')
                observed+=1
                common, extra=case_common(path,route,seed)
                row,_,_,_=read_case(path,route,seed,common=common)
                rows.append({**row,**base,**extra,'record_status':'complete'})
        if observed:maps.append((scenario.name,ref_geom,ref_cfg))
    mainrows=[x for x in rows if x['route'] in MAIN]; uniform=[x for x in rows if x['route']==UNIFORM]
    fields=list(dict.fromkeys(k for row in rows for k in row))
    write_csv(out/'main_runs.csv',mainrows,fields);write_csv(out/'uniform_runs.csv',uniform,fields)
    summaries=[]
    for density in ('sparse','medium','dense'):
        dr=[r for r in mainrows if r['density']==density]
        if not dr:continue
        keys=sorted({(r['map_seed'],r['seed']) for r in dr}); pairs=[]
        bykey={(r['map_seed'],r['seed'],r['route']):r for r in dr}
        for key in keys:
            pair=[bykey.get((*key,route)) for route in MAIN]
            if all(r and r['record_status']=='complete' and r['success_with_safety_and_dynamics'] for r in pair):pairs.append(pair)
        summary={'density':density,'expected_pairs':len(keys),'common_success_pairs':len(pairs)}
        for route in MAIN:
            candidates=[r for r in dr if r['route']==route]
            summary[f'{route}_successes']=sum(bool(r['success_with_safety_and_dynamics']) and r['record_status']=='complete' for r in candidates)
            summary[f'{route}_complete_runs']=sum(r['record_status']=='complete' for r in candidates)
            for metric in FIELDS:
                values=[pair[MAIN.index(route)].get(metric) for pair in pairs]
                if any(v is None for v in values):raise ValueError(f'Missing metric {metric} for success pair')
                summary[f'{route}_{metric}_mean']=float(np.mean(values)) if values else None
        summaries.append(summary)
    write_csv(out/'main_summary.csv',summaries)
    if maps:overview(maps,out)
    plot_summary(summaries,out)
    notes={'root':str(root),'matching':'density, map_seed, planning seed; exact occupancy and rectangles must agree within a map; origin, resolution, radius, endpoints, reference and initial positions must agree across all maps and routes',
           'eligibility':'endpoint (all robot errors <0.4m and speeds <0.1m/s) and continuous safety, dynamics and SFC audits; incomplete records are marked incomplete and are not counted as successes',
           'pair_means':'only pairs where both paper_affine and rrt_affine pass; same pair set for all metrics; no common-success pairs means blank CSV cells',
           'd_safe':'sampled robot-center to physical obstacle/wall minimum; body clearance subtracts radius; audit lower bound is separate',
           'e_f':'paper normalized Laplacian shape error; e_f_time_average integrates over recorded observed duration, e_f_max_sampled is sample maximum',
           't_fly':'observed simulated duration only on successful runs; unsuccessful runs are censored and keep prefix metrics',
           't_sol':'local full-team MPPI backend time divided by optimization count, includes sequence reuse overhead; not directly comparable to distributed GPU paper timings',
           't_front':'total frontend wall time / frontend planning rounds; does not include SFC construction',
           'main_routes':list(MAIN),'uniform':'reported only in uniform_runs.csv',
           'cases':rows,'summary':summaries}
    (out/'report.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(out),'scenarios':len(scenarios),'summary':summaries},indent=2))


if __name__=='__main__':main()
