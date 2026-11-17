import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import DT, RADIUS, CONFIG

def write_outputs(args, result):
    """Write recorded trajectories, planning evidence, metrics, and a plot."""
    (world, states, controls, qs, actual_q, paths, ticks, assignments, assignment_history, known_history, planning_seconds) = (result.world, result.states, result.controls, result.q_history, result.actual_fit_q, result.paths, result.planning_ticks, result.assignments, result.assignment_history, result.known_history, result.planning_seconds)
    geometry = result.geometry
    REF = geometry.leader_reference
    CENTERED_REF = geometry.centered_reference
    name = f'{args.scene}_{args.mode}_{geometry.name}_spacing{geometry.spacing:g}_seed{args.seed}'
    if result.dynamics == 'uav':
        name += '_uav'
    if args.scene in ('random_circles', 'random_pillars'):
        name += f'_mapseed{args.map_seed}_n{args.obstacle_count}'
    if args.scene == 'random_pillars':
        name += f'_width{world.pillar_width:g}'
    out = Path(args.output) if getattr(args,'exact_output',False) else Path(args.output) / name
    out.mkdir(parents=True, exist_ok=True)
    uav_fields = {'omega_max': CONFIG.max_angular_speed} if result.dynamics != 'uav' else {}
    if result.dynamics == 'uav':
        uav_fields = {'uav_states': result.uav_states, 'uav_controls': result.uav_controls,
                      'uav_altitude': result.uav_altitude, 'a_max': 1.0,
                      'control_labels': np.array(['jerk_x', 'jerk_y', 'jerk_z', 'yaw_rate'])}
    np.savez_compressed(out / 'trajectory.npz', dynamics=result.dynamics, **uav_fields, states=states, controls=controls, acceleration_controls=result.acceleration_controls, velocity_history=result.velocity_history, mppi_diagnostics=result.mppi_diagnostics, q_history=qs, actual_fit_q=actual_q, assignment_history=assignment_history, known_history=known_history, dt=result.execution_dt, prediction_dt=DT, backend_ticks=result.backend_ticks, backend_period=result.backend_period, perception_period=DT, occupancy=world.truth, origin=np.zeros(2), resolution=world.res, robot_radius=RADIUS, reference=REF, centered_reference=CENTERED_REF, leader_slot=geometry.leader_slot, formation_name=geometry.name, formation_spacing=geometry.spacing, formation_provenance=('Figure-inspired equilateral grid; spacing chosen, not specified in paper' if geometry.name == 'triangle6' else 'Legacy square4 reference; spacing chosen'), goal=world.goal+CENTERED_REF[geometry.leader_slot], start_center=world.start, goal_center=world.goal, requested_start_center=np.asarray(args.start_center if args.start_center is not None else [np.nan, np.nan]), requested_goal_center=np.asarray(args.goal_center if args.goal_center is not None else [np.nan, np.nan]), planning_ticks=ticks, v_max=1.5 if result.dynamics == 'uav' else CONFIG.max_speed, circles=world.circles, rectangles=world.rectangles, obstacle_centers=world.obstacle_centers, pillar_width=world.pillar_width, obstacle_provenance=('Axis-aligned square pillars; width chosen, not specified in paper' if args.scene == 'random_pillars' else 'Legacy circle radius 0.5m assumed'), map_seed=world.map_seed, obstacle_count=len(world.circles)+len(world.rectangles))
    regions = [np.c_[h, b] for group in result.regions for (h, b) in group]
    offsets = np.r_[0, np.cumsum([len(region) for region in regions])]
    np.savez_compressed(out / 'regions.npz', coefficients=np.concatenate(regions) if regions else np.empty((0,3)), offsets=offsets, regions_per_plan=len(REF))
    (out / 'planning.json').write_text(json.dumps({'ticks': ticks, 'paths': paths, 'assignments': assignments, 'planning_seconds': planning_seconds, 'status': result.failures, 'paths_include_initial': False, 'initialization_q': result.initialization_q, 'sequence_diagnostics': result.sequence_diagnostics}))
    from scipy.spatial import cKDTree
    occupied_centers = (np.argwhere(world.truth) + 0.5) * world.res
    xy = states[:, :, :2].reshape(-1, 2)
    half_diagonal = world.res * np.sqrt(2) / 2
    clear = cKDTree(occupied_centers).query(xy)[0] - half_diagonal
    outside = np.any((xy < 0) | (xy >= np.asarray(world.size) * world.res), axis=1)
    clear[outside] = np.minimum(clear[outside], -half_diagonal)
    pair = np.linalg.norm(states[:, :, None, :2] - states[:, None, :, :2], axis=-1) + np.eye(len(REF))[None, :, :] * 100
    progress = np.linalg.norm(states[-1, :, :2].mean(axis=0) - world.goal)
    errors = np.linalg.norm(states[-1, :, :2] - (world.goal + CENTERED_REF[geometry.leader_slot] + REF[assignment_history[-1]]), axis=1)
    terminal_speed = float(np.max(np.abs(result.velocity_history[-1])))
    metrics = {'scene': args.scene, 'mode': args.mode, 'seed': args.seed, 'map_seed': world.map_seed, 'obstacle_count': len(world.circles)+len(world.rectangles), 'formation': geometry.name, 'formation_spacing': geometry.spacing, 'leader_slot': geometry.leader_slot, 'pillar_width': world.pillar_width if args.scene == 'random_pillars' else None, 'success': bool(errors.max() < 0.4 and terminal_speed < 0.1), 'goal_distance': float(progress), 'robot_goal_errors': errors.tolist(), 'terminal_max_speed': terminal_speed, 'planning_seconds': planning_seconds, 'steps': len(controls), 'min_obstacle_clearance': float(clear.min()), 'min_robot_distance': float(pair.min()), 'obstacle_collision': bool(clear.min() < RADIUS), 'robot_collision': bool(pair.min() < 2 * RADIUS), 'budget': {'sequences': args.sequences, 'steps': args.steps, 'candidates': args.candidates, 'samples': args.samples, 'horizon': args.horizon, 'replan_steps': CONFIG.replan_steps, 'min_radius': 0.1}, 'dt': result.execution_dt, 'termination': result.termination, 'planning_status_counts': {status: result.failures.count(status) for status in set(result.failures)}}
    metrics['requested_start_center'] = args.start_center
    metrics['requested_goal_center'] = args.goal_center
    metrics['start_center'] = world.start.tolist()
    metrics['goal_center'] = world.goal.tolist()
    metrics['dynamics'] = result.dynamics
    metrics['mppi_diagnostic_fields'] = (['ess', 'best_sample_cost_dt'] if result.dynamics == 'uav' else ['ess', 'fraction_peer_clearance_ge_0.32', 'fraction_peer_clearance_ge_0.5', 'best_sample_min_peer_distance', 'weighted_prediction_min_peer_distance', 'best_sample_cost_dt'])
    metrics['obstacle_clearance_model'] = 'Actual state samples versus disks enclosing occupied grid cells; continuous execution requires independent audit'
    metrics['mppi_guidance_weight'] = result.guidance_weight
    metrics['mppi_smoothness_weight'] = result.smoothness_weight
    metrics['mppi_temperature'] = CONFIG.mppi_temperature
    metrics['mppi_noise_std'] = [0.22, 0.22, 0.0, 0.9] if result.dynamics == 'uav' else list(CONFIG.mppi_noise_std)
    if result.dynamics == 'uav':
        speed = np.linalg.norm(result.uav_states[..., 3:6], axis=-1)
        acceleration = np.linalg.norm(result.uav_states[..., 6:9], axis=-1)
        metrics['uav_model'] = {'integration': 'forward Euler interpretation of Eq.23', 'state_labels': ['px','py','pz','vx','vy','vz','ax','ay','az','yaw'], 'control_labels': ['jerk_x','jerk_y','jerk_z','yaw_rate'], 'states_projection': 'xy and yaw', 'acceleration_controls': 'post-step xyz acceleration', 'obstacles': '2D map extruded vertically', 'v_max_paper_m_s': 1.5, 'assumed_a_max_m_s2': 1.0, 'assumed_altitude_m': result.uav_altitude, 'assumed_downwash_factor': 0.5, 'noise_z': 0.0, 'assumed_dt_s': DT, 'assumed_temperature': 1000.0, 'limits': 'soft cost only; no clipping', 'max_speed_m_s': float(speed.max()), 'max_acceleration_m_s2': float(acceleration.max()), 'speed_limit_exceedances': int((speed > 1.5).sum()), 'acceleration_limit_exceedances': int((acceleration > 1.0).sum())}
    metrics['frontend_goal_weight'] = CONFIG.goal_weight
    metrics['suspension_penalty_per_remaining_step'] = CONFIG.goal_weight * float(np.linalg.norm(np.asarray(world.size)*world.res)) + CONFIG.suspension_margin
    metrics['sfc_seed'] = 'planar_disk'
    metrics['sfc_seed_radius_m'] = RADIUS + world.res * np.sqrt(2) / 2 + 1e-5
    metrics['sfc_geometry_version'] = 'gcopter_old_firi_planar_disk_seed_v1'
    metrics['sfc_error'] = result.sfc_error
    metrics['simulation_wall_seconds'] = result.elapsed_seconds
    metrics['timings_seconds'] = result.timings
    metrics['planning_stage_seconds'] = result.stage_times
    metrics['simulation_duration_seconds'] = len(controls)*result.execution_dt
    metrics['backend_period_seconds'] = result.backend_period
    metrics['prediction_dt_seconds'] = DT
    metrics['replan_period_seconds'] = result.replan_period
    metrics['budget']['replan_steps'] = int(round(result.replan_period/DT))
    metrics['stagnation_timeout_seconds'] = result.stagnation_timeout
    metrics['timing_note'] = 'Lazy map-index refresh is charged to the stage that queries the map; MPPI prediction and control-sequence knot step is 0.2 s; optimization update period and execution step are recorded separately.'
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    from .plotting import redraw
    redraw(out)
    print(json.dumps(metrics, indent=2), flush=True)
    return out
