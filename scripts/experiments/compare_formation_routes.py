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
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-time',type=float,default=500.)
    p.add_argument('--obstacle-count',type=int,default=50)
    p.add_argument('--map-seed',type=int,default=1)
    p.add_argument('--output',type=Path,default=Path('results/three_routes/matched_rates'))
    p.add_argument('--progress',action='store_true')
    a=p.parse_args();out=a.output/a.route/f'seed_{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    protocol={'route':a.route,'seed':a.seed,'map_seed':a.map_seed,
              'scene':'random_pillars','obstacle_count':a.obstacle_count,'pillar_width':.3,
              'formation':'triangle6','spacing':3.5,'start_center':[5.,20.],'goal_center':[45.,20.],
              'execution_dt':.02,'perception_period':.2,'frontend_period':1.,'backend_period':1.,
              'max_time':a.max_time,'stagnation_timeout':32.,'v_max':.65,'omega_max':1.5,
              'goal_max_robot_error':.4,'goal_max_robot_speed':.1,
              'execution_note':'Differential-drive kinematics. MPPI executes acceleration/yaw-rate in 0.02 s midpoint substeps; Bezier uses analytic flat-output execution. No actuator/tracking error model.',
              'timing_note':'Offline synchronous simulation; compute time does not advance simulation time.'}
    (out/'configuration.json').write_text(json.dumps(protocol,indent=2))
    if a.route.startswith('paper_'):
        from formation.paper_formation.config import CONFIG
        from formation.paper_formation.simulation import simulate
        from formation.paper_formation.outputs import write_outputs
        mode=a.route.removeprefix('paper_')
        result=simulate('random_pillars',mode,a.seed,obstacle_count=a.obstacle_count,map_seed=a.map_seed,
                        formation='triangle6',formation_spacing=3.5,pillar_width=.3,
                        start_center=[5.,20.],goal_center=[45.,20.],progress=a.progress,
                        replan_period=1.,backend_period=1.,execution_dt=.02,
                        max_time=a.max_time,stagnation_timeout=32.)
        output_args=argparse.Namespace(scene='random_pillars',mode=mode,seed=a.seed,
            map_seed=a.map_seed,obstacle_count=a.obstacle_count,output=str(out),exact_output=True,
            start_center=[5.,20.],goal_center=[45.,20.],sequences=CONFIG.sequences,
            steps=CONFIG.steps,candidates=CONFIG.candidates,samples=CONFIG.samples,horizon=CONFIG.horizon,
            wall_seconds=result.elapsed_seconds)
        write_outputs(output_args,result)
        protocol['budget']={'sequences':CONFIG.sequences,'sequence_steps':CONFIG.steps,
                            'candidates_per_step':CONFIG.candidates,'mppi_samples':CONFIG.samples,
                            'mppi_horizon':CONFIG.horizon,'mppi_prediction_dt':.2,
                            'executed_control_knots_per_optimization':5}
    else:
        from formation.local_affine.simulation import simulate
        from formation.local_affine.outputs import save
        result=simulate(scene='random_pillars',mode='affine',formation='triangle6',spacing=3.5,
                        obstacle_count=a.obstacle_count,map_seed=a.map_seed,seed=a.seed,
                        dt=.02,perception_period=.2,replan_period=1.,max_time=a.max_time,progress=a.progress)
        save(result,str(out),'random_pillars','affine',a.seed,a.map_seed,a.obstacle_count,250,.02,exact_output=True)
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
    (out/'configuration.json').write_text(json.dumps(protocol,indent=2))


if __name__=='__main__':
    main()
