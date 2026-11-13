import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import REF as CENTERED_REF, LEADER_REF as REF, DT, RADIUS, CONFIG

def write_outputs(args, result):
    """Write recorded trajectories, planning evidence, metrics, and a plot."""
    (world, states, controls, qs, actual_q, paths, ticks, assignments, assignment_history, known_history, planning_seconds) = (result.world, result.states, result.controls, result.q_history, result.actual_fit_q, result.paths, result.planning_ticks, result.assignments, result.assignment_history, result.known_history, result.planning_seconds)
    name = f'{args.scene}_{args.mode}_seed{args.seed}'
    if args.scene == 'random_circles':
        name += f'_mapseed{args.map_seed}_n{args.obstacle_count}'
    out = Path(args.output) / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'trajectory.npz', states=states, controls=controls, acceleration_controls=result.acceleration_controls, velocity_history=result.velocity_history, q_history=qs, actual_fit_q=actual_q, assignment_history=assignment_history, known_history=known_history, dt=DT, occupancy=world.truth, origin=np.zeros(2), resolution=world.res, robot_radius=RADIUS, reference=REF, goal=world.goal+CENTERED_REF[0], planning_ticks=ticks, v_max=CONFIG.max_speed, omega_max=CONFIG.max_angular_speed, circles=world.circles, map_seed=world.map_seed, obstacle_count=len(world.circles))
    regions = [np.c_[h, b] for group in result.regions for (h, b) in group]
    offsets = np.r_[0, np.cumsum([len(region) for region in regions])]
    np.savez_compressed(out / 'regions.npz', coefficients=np.concatenate(regions) if regions else np.empty((0,3)), offsets=offsets, regions_per_plan=len(REF))
    (out / 'planning.json').write_text(json.dumps({'ticks': ticks, 'paths': paths, 'assignments': assignments, 'planning_seconds': planning_seconds, 'status': result.failures}))
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~world.truth) * world.res - world.res * np.sqrt(2) / 2
    idx = np.floor(states[:, :, :2] / world.res).astype(int)
    idx = np.clip(idx, [0, 0], np.array(world.size) - 1)
    clear = dist[idx[:, :, 0], idx[:, :, 1]]
    pair = np.linalg.norm(states[:, :, None, :2] - states[:, None, :, :2], axis=-1) + np.eye(len(REF))[None, :, :] * 100
    progress = np.linalg.norm(states[-1, :, :2].mean(axis=0) - world.goal)
    errors = np.linalg.norm(states[-1, :, :2] - (world.goal + CENTERED_REF[0] + REF[assignment_history[-1]]), axis=1)
    terminal_speed = float(np.max(np.abs(result.velocity_history[-1])))
    metrics = {'scene': args.scene, 'mode': args.mode, 'seed': args.seed, 'map_seed': world.map_seed, 'obstacle_count': len(world.circles), 'success': bool(errors.max() < 0.4 and terminal_speed < 0.1), 'goal_distance': float(progress), 'robot_goal_errors': errors.tolist(), 'terminal_max_speed': terminal_speed, 'planning_seconds': planning_seconds, 'steps': len(controls), 'min_obstacle_clearance': float(clear.min()), 'min_robot_distance': float(pair.min()), 'obstacle_collision': bool(clear.min() < RADIUS), 'robot_collision': bool(pair.min() < 2 * RADIUS), 'budget': {'sequences': args.sequences, 'steps': args.steps, 'candidates': args.candidates, 'samples': args.samples, 'horizon': args.horizon, 'replan_steps': CONFIG.replan_steps, 'min_radius': 0.1}, 'dt': DT, 'termination': result.termination, 'planning_status_counts': {status: result.failures.count(status) for status in set(result.failures)}}
    metrics['sfc_error'] = result.sfc_error
    metrics['simulation_wall_seconds'] = args.wall_seconds
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    from .plotting import redraw
    redraw(out)
    print(json.dumps(metrics, indent=2), flush=True)
