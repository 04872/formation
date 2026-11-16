"""Independent audit of paper-Euler UAV traces in the fixed-height experiment."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def audit_uav(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        x, u = d['uav_states'], d['uav_controls']
        projected = d['states']
        dt, radius, res = float(d['dt']), float(d['robot_radius']), float(d['resolution'])
        occupied, origin = d['occupancy'], d['origin']
        boxes, circles = d['rectangles'], d['circles']
        vmax, amax = float(d['v_max']), float(d['a_max'])
        goal, reference, assignment = d['goal'], d['reference'], d['assignment_history'][-1]
    expected = x[:-1].copy()
    expected[..., :3] += dt * x[:-1, :, 3:6]
    expected[..., 3:6] += dt * x[:-1, :, 6:9]
    expected[..., 6:9] += dt * u[..., :3]
    expected[..., 9] += dt * u[..., 3]
    residual = np.abs(expected - x[1:]).max(axis=(0, 1), initial=0)
    speeds = np.linalg.norm(x[..., 3:6], axis=-1)
    accelerations = np.linalg.norm(x[..., 6:9], axis=-1)
    tree = cKDTree(origin + (np.argwhere(occupied) + .5) * res)
    halfdiag = res / np.sqrt(2)
    low = origin + 2 * res
    high = origin + np.asarray(occupied.shape) * res - 2 * res
    grid_lower = physical_lower = pair_lower = float('inf')
    ii, jj = np.triu_indices(x.shape[1], 1)
    for before, after in zip(x[:-1, :, :3], x[1:, :, :3]):
        displacement = after - before
        travel = float(np.linalg.norm(displacement, axis=-1).max())
        pieces = max(1, int(np.ceil(travel / .005)))
        trace = before[None] + np.linspace(0, 1, pieces + 1)[:, None, None] * displacement
        flat = trace[..., :2].reshape(-1, 2)
        gap = travel / (2 * pieces)
        grid_lower = min(grid_lower, float(tree.query(flat)[0].min()) - halfdiag - radius - gap)
        outside = np.any((flat < origin) | (flat >= origin + np.asarray(occupied.shape) * res), axis=-1)
        if outside.any():
            grid_lower = min(grid_lower, -radius)
        physical = float(np.minimum(flat - low, high - flat).min())
        if len(boxes):
            delta = np.maximum(boxes[None, :, :2] - flat[:, None], flat[:, None] - boxes[None, :, 2:])
            sdf = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(axis=-1), 0)
            physical = min(physical, float(sdf.min()))
        if len(circles):
            distances = np.linalg.norm(flat[:, None] - circles[None, :, :2], axis=-1) - circles[None, :, 2]
            physical = min(physical, float(distances.min()))
        physical_lower = min(physical_lower, physical - radius - gap)
        # Exact closest separation along the two simultaneously executed line segments.
        relative = before[ii] - before[jj]
        velocity = displacement[ii] - displacement[jj]
        denominator = np.sum(velocity * velocity, axis=-1)
        t = np.clip(np.divide(-np.sum(relative * velocity, axis=-1), denominator,
                              out=np.zeros_like(denominator), where=denominator > 0), 0, 1)
        pair_lower = min(pair_lower, float(np.linalg.norm(relative + t[:, None] * velocity, axis=-1).min()) - 2 * radius)
    altitude_spread = float(np.abs(x[..., 2] - x[0, 0, 2]).max())
    planar = altitude_spread < 1e-10 and float(np.abs(x[..., [5, 8]]).max()) < 1e-10
    return {
        'trace': str(path), 'dynamics': 'uav', 'steps': len(u), 'robots': x.shape[1],
        'duration_s': len(u) * dt,
        'motion_model': 'Paper forward Euler triple integrator; positions joined by straight lines between discrete states, not exact continuous constant-jerk integration',
        'finite_states_controls': bool(np.isfinite(x).all() and np.isfinite(u).all()),
        'euler_state_residual_by_component': residual.tolist(),
        'euler_model_satisfied': bool(residual.max() < 1e-9),
        'plot_projection_residual_max': float(np.abs(projected - x[..., [0, 1, 9]]).max()),
        'fixed_altitude_max_deviation_m': altitude_spread,
        'fixed_height_geometry_applicable': planar,
        'continuous_obstacle_clearance_lower_bound_m': grid_lower,
        'continuous_physical_obstacle_clearance_lower_bound_m': physical_lower,
        'continuous_pair_clearance_lower_bound_m': pair_lower,
        'continuous_safety_certified_by_audit': bool(len(u) and planar and grid_lower > 0 and pair_lower > 0),
        'continuous_physical_safety_certified': bool(len(u) and planar and physical_lower > 0 and pair_lower > 0),
        'max_speed_m_s': float(speeds.max()), 'max_acceleration_m_s2': float(accelerations.max()),
        'v_max_satisfied': bool(speeds.max() <= vmax + 1e-9),
        'a_max_satisfied': bool(accelerations.max() <= amax + 1e-9),
        'speed_violation_samples': int(np.count_nonzero(speeds > vmax + 1e-9)),
        'acceleration_violation_samples': int(np.count_nonzero(accelerations > amax + 1e-9)),
        'limits': {'v_max': vmax, 'a_max': amax, 'enforcement': 'paper soft running cost, no hard clipping'},
        'final_max_speed_m_s': float(speeds[-1].max()),
        'final_max_acceleration_m_s2': float(accelerations[-1].max()),
        'goal_distance_m': float(np.linalg.norm(x[-1, :, :2].mean(axis=0) - (goal + reference.mean(axis=0)))),
        'final_max_robot_goal_error_m': float(np.linalg.norm(x[-1, :, :2] - goal - reference[assignment], axis=-1).max()),
    }
