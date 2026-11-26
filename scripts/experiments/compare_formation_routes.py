#!/usr/bin/env python3
"""Run one formation route under the common-rate comparison protocol."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route',required=True,choices=['paper_uniform','paper_affine','rrt_affine'])
    p.add_argument('--rrt-backend',choices=['mppi','bezier'],default='mppi')
    p.add_argument('--dynamics',choices=['differential_drive','uav'],default='differential_drive')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-time',type=float,default=500.)
    p.add_argument('--obstacle-count',type=int,default=50)
    p.add_argument('--map-seed',type=int,default=1)
    p.add_argument('--scene',choices=['random_pillars','history_deadend'],default='random_pillars')
    p.add_argument('--history-decision',action='store_true',help='use the history visibility graph to select RRT candidate branches')
    p.add_argument('--history-joint-path',action='store_true',help='reuse certified joint paths for boundary-based branch selection')
    p.add_argument('--history-current-motion-only',action='store_true',help='with joint history, use only current-round motion while retaining the map and SFC history')
    p.add_argument('--history-link-epsilon',type=float,default=.4,help='maximum certified short-link cost (default: 0.4 m)')
    p.add_argument('--output',type=Path,default=Path('results/three_routes/matched_rates'))
    p.add_argument('--progress',action='store_true')
    a=p.parse_args()
    if (a.history_decision or a.history_joint_path) and (a.route!='rrt_affine' or a.rrt_backend!='mppi'):
        p.error('history modes require --route rrt_affine --rrt-backend mppi')
    if a.history_decision and a.history_joint_path:
        p.error('history modes are mutually exclusive')
    if a.history_current_motion_only and not a.history_joint_path:
        p.error('--history-current-motion-only requires --history-joint-path')
    if a.history_link_epsilon <= 0:
        p.error('--history-link-epsilon must be positive')
    out=a.output/a.route/f'seed_{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    from formation.paper_formation.sfc import SFC_GEOMETRY_VERSION
    protocol={'route':a.route,'backend':a.rrt_backend if a.route=='rrt_affine' else 'mppi','frontend':'rrt' if a.route=='rrt_affine' else 'paper','dynamics':a.dynamics,'seed':a.seed,'map_seed':a.map_seed,
              'scene':a.scene,'obstacle_count':a.obstacle_count if a.scene=='random_pillars' else 3,'pillar_width':.3 if a.scene=='random_pillars' else None,
              'history_decision':a.history_decision,'history_joint_path':a.history_joint_path,'history_current_motion_only':a.history_current_motion_only,'history_link_epsilon':a.history_link_epsilon,'per_robot_perception_radius_m':3.,
              'formation':'triangle6','spacing':3.5,'start_center':[5.,20.],'goal_center':[45.,20.],
              'execution_dt':.02,'perception_period':.2,'frontend_period':1.,'backend_period':1.,
              'sfc_geometry_version':SFC_GEOMETRY_VERSION,
              'max_time':a.max_time,'stagnation_timeout':32.,'v_max':1.5 if a.dynamics=='uav' else .65,'a_max':1.,'omega_max':None if a.dynamics=='uav' else 1.5,
              'goal_max_robot_error':.4,'goal_max_robot_speed':.1,
              'execution_note':'Differential-drive kinematics. MPPI executes acceleration/yaw-rate in 0.02 s midpoint substeps; Bezier uses analytic flat-output execution. No actuator/tracking error model.',
              'timing_note':'Offline synchronous simulation; compute time does not advance simulation time.'}
    if a.dynamics=='uav':
        protocol['execution_note']='Paper Eq.4 jerk-driven integrator at fixed altitude. MPPI uses 0.02 s Euler execution and matching composed prediction steps; Bezier executes analytic p/v/a/jerk. Yaw is independent of translation.'
        protocol['parameter_provenance']='v_max=1.5 from paper; a_max=1, altitude=1 and formation spacing are experiment assumptions.'
    if protocol['backend']=='mppi':
        protocol['execution_note']=('Paper Eq.4 jerk-driven UAV; 0.02 s Euler execution with matching composed prediction steps and independent yaw.' if a.dynamics=='uav' else 'Differential-drive midpoint execution at 0.02 s.')
    (out/'configuration.json').write_text(json.dumps(protocol,indent=2))
    if a.route.startswith('paper_') or a.rrt_backend=='mppi':
        from formation.paper_formation.config import CONFIG
        from formation.paper_formation.simulation import simulate
        from formation.paper_formation.outputs import write_outputs
        mode='affine' if a.route=='rrt_affine' else a.route.removeprefix('paper_')
        result=simulate(a.scene,mode,a.seed,obstacle_count=a.obstacle_count,map_seed=a.map_seed,
                        formation='triangle6',formation_spacing=3.5,pillar_width=.3,
                        start_center=[5.,20.],goal_center=[45.,20.],progress=a.progress,
                        frontend='rrt' if a.route=='rrt_affine' else 'paper',rrt_iterations=250,
                        dynamics=a.dynamics,replan_period=1.,backend_period=1.,execution_dt=.02,
                        max_time=a.max_time,stagnation_timeout=32.,history_decision=a.history_decision,
                        history_joint_path=a.history_joint_path,history_link_epsilon=a.history_link_epsilon,
                        history_current_motion_only=a.history_current_motion_only)
        output_args=argparse.Namespace(scene=a.scene,mode=mode,seed=a.seed,
            map_seed=a.map_seed,obstacle_count=a.obstacle_count,output=str(out),exact_output=True,
            start_center=[5.,20.],goal_center=[45.,20.],sequences=CONFIG.sequences,
            steps=CONFIG.steps,candidates=CONFIG.candidates,samples=CONFIG.samples,horizon=CONFIG.horizon,
            wall_seconds=result.elapsed_seconds)
        write_outputs(output_args,result)
        protocol['budget']={'sequences':CONFIG.sequences,'sequence_steps':CONFIG.steps,
                            'candidates_per_step':CONFIG.candidates,'mppi_samples':CONFIG.samples,
                            'mppi_horizon':CONFIG.horizon,'mppi_prediction_dt':.2,
                            'executed_control_knots_per_optimization':5}
        if a.route.startswith('paper_'):
            protocol['budget']['min_radius'] = CONFIG.uniform_min_radius if mode == 'uniform' else 0.1
            protocol['budget']['radius_sampling'] = 'r = min_radius * (1 + gamma), gamma uniform [0,1)'
        if a.route=='rrt_affine':
            protocol['budget']={k:v for k,v in protocol['budget'].items() if k not in ('sequences','sequence_steps','candidates_per_step')}
            protocol['budget'].update(rrt_iterations=250,guidance_reference_speed_m_s=.55)
    else:
        from formation.local_affine.simulation import simulate
        from formation.local_affine.outputs import save
        result=simulate(scene=a.scene,mode='affine',formation='triangle6',spacing=3.5,
                        obstacle_count=a.obstacle_count,map_seed=a.map_seed,seed=a.seed,
                        dynamics=a.dynamics,dt=.02,perception_period=.2,replan_period=1.,max_time=a.max_time,progress=a.progress)
        save(result,str(out),a.scene,'affine',a.seed,a.map_seed,a.obstacle_count,250,.02,exact_output=True)
        protocol['budget']={'rrt_iterations':250,'max_bezier_segments':3,
                            'max_backend_route_candidates':3,
                            'translation_only_fraction_of_non_goal_samples':0.5,
                            'independent_braking_reserve_segments':0,
                            'time_allocation':'nonterminal max-robot edge length / 0.55 m/s; execution-step quantized; fixed initial controls screened before solve',
                            'minimum_nonterminal_horizon_seconds':1.02,
                            'local_terminal':'free position, velocity and acceleration; soft waypoint tracking',
                            'goal_terminal':'exact goal configuration, zero velocity and acceleration',
                            'angular_refinement_max_iterations':3,
                            'angular_refinement_target_rad_s':1.5,
                            'angular_refinement':'certificate violation locations; no relaxed retries; full post-solve constraint check',
                            'maximum_solver_calls_per_round':12,
                            'objective':'physical-time integrals of workspace position tracking, reference velocity tracking and acceleration'}
    if a.dynamics=='uav' and a.route=='rrt_affine' and a.rrt_backend=='bezier':
        protocol['budget'].update(angular_refinement_max_iterations=0,angular_refinement_target_rad_s=None,angular_refinement='not applicable: independent UAV yaw',maximum_solver_calls_per_round=3,time_allocation='per-edge max-robot displacement / 0.55 m/s (unchanged from DD); execution-step quantized with derivative feasibility checks',reference_speed_m_s=.55)
    (out/'configuration.json').write_text(json.dumps(protocol,indent=2))


if __name__=='__main__':
    main()
