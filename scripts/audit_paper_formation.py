"""Independently audit saved differential-drive execution, not planner predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import HalfspaceIntersection, ConvexHull


def arc(states: np.ndarray, controls: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Exact zero-order-hold unicycle motion, including zero angular velocity."""
    theta = states[None, :, 2] + times[:, None] * controls[None, :, 1]
    half = 0.5 * times[:, None] * controls[None, :, 1]
    distance = times[:, None] * controls[None, :, 0] * np.sinc(half / np.pi)
    heading = states[None, :, 2] + half
    xy = states[None, :, :2] + distance[:, :, None] * np.stack(
        (np.cos(heading), np.sin(heading)), axis=-1
    )
    return np.concatenate((xy, theta[:, :, None]), axis=-1)


def audit(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        states = data["states"]
        controls = data["controls"]
        accelerations = data["acceleration_controls"] if "acceleration_controls" in data else None
        velocities = data["velocity_history"] if "velocity_history" in data else None
        dt = float(data["dt"])
        occupancy = data["occupancy"].astype(bool)
        origin = data["origin"]
        resolution = float(data["resolution"])
        radius = float(data["robot_radius"])
        reference = data["reference"]
        q = data["q_history"]
        assignment = data["assignment_history"] if "assignment_history" in data else None
        goal = data["goal"] if "goal" in data else None
        limits = {key: float(data[key]) for key in ("v_max", "omega_max") if key in data}

    # Occupied grid squares are enclosed by disks of half-diagonal radius.
    # This gives a conservative lower clearance bound, including between samples.
    occupied_centers = origin + (np.argwhere(occupancy) + 0.5) * resolution
    tree = cKDTree(occupied_centers)
    half_diagonal = resolution / np.sqrt(2.0)
    position_residual = heading_residual = 0.0
    obstacle_lower = pair_lower = float("inf")
    obstacle_sample_upper = pair_sample_min = float("inf")
    n = states.shape[1]
    pair_i, pair_j = np.triu_indices(n, 1)
    for k, u in enumerate(controls):
        speed = float(np.max(np.abs(u[:, 0])))
        pieces = max(1, int(np.ceil(speed * dt / 0.005)))
        times = np.linspace(0.0, dt, pieces + 1)
        trace = arc(states[k], u, times)
        position_residual = max(position_residual, float(np.max(np.linalg.norm(
            trace[-1, :, :2] - states[k + 1, :, :2], axis=-1))))
        heading_delta = trace[-1, :, 2] - states[k + 1, :, 2]
        heading_residual = max(heading_residual, float(np.max(np.abs(
            np.arctan2(np.sin(heading_delta), np.cos(heading_delta))))))
        xy = trace[:, :, :2]
        distances, indices = tree.query(xy.reshape(-1, 2))
        gap = speed * dt / pieces / 2.0
        obstacle_lower = min(obstacle_lower, float(distances.min()) - half_diagonal - radius - gap)
        nearest_square_delta = np.maximum(
            np.abs(xy.reshape(-1, 2) - occupied_centers[indices]) - resolution / 2.0, 0.0)
        obstacle_sample_upper = min(obstacle_sample_upper,
                                    float(np.linalg.norm(nearest_square_delta, axis=1).min()) - radius)
        separation = np.linalg.norm(xy[:, pair_i] - xy[:, pair_j], axis=-1)
        pair_sample_min = min(pair_sample_min, float(separation.min()))
        # Each robot travels at most speed * half sample interval.
        pair_lower = min(pair_lower, float(separation.min()) - 2.0 * gap - 2.0 * radius)

    a = q[:, 2:].reshape(-1, 2, 2)
    singular = np.linalg.svd(a, compute_uv=False)
    scalar = np.trace(a, axis1=1, axis2=2) / 2.0
    departure = np.linalg.norm(a - scalar[:, None, None] * np.eye(2), axis=(1, 2))
    # Four or more robots permit a nontrivial affine-fit residual.
    design = np.column_stack((np.ones(n), reference))
    fit = np.linalg.pinv(design)
    actual_xy = states[:, :, :2]
    if assignment is not None:
        actual_xy = np.take_along_axis(actual_xy, np.argsort(assignment, axis=1)[:, :, None], axis=1)
    fitted = np.einsum("nj,jk,tkd->tnd", design, fit, actual_xy)
    result = {
        "trace": str(path), "steps": len(controls), "robots": n,
        "duration_s": len(controls) * dt,
        "finite_states_controls": bool(np.isfinite(states).all() and np.isfinite(controls).all()),
        "exact_unicycle_position_residual_max_m": position_residual,
        "exact_unicycle_heading_residual_max_rad": heading_residual,
        "continuous_obstacle_clearance_lower_bound_m": obstacle_lower,
        "sample_obstacle_clearance_upper_bound_m": obstacle_sample_upper,
        "continuous_pair_clearance_lower_bound_m": pair_lower,
        "sample_min_pair_center_distance_m": pair_sample_min,
        "max_abs_v_m_s": float(np.abs(velocities if velocities is not None else controls[:, :, 0]).max(initial=0)),
        "max_abs_omega_rad_s": float(np.abs(controls[:, :, 1]).max(initial=0)),
        "reference_det_min": float(np.linalg.det(a).min()),
        "reference_singular_value_min": float(singular.min()),
        "reference_singular_value_max": float(singular.max()),
        "reference_anisotropy_max": float((singular[:, 0] / singular[:, 1]).max()),
        "reference_off_diagonal_max": float(np.abs(a[:, [0, 1], [1, 0]]).max()),
        "reference_departure_from_scalar_identity_max": float(departure.max()),
        "executed_affine_fit_residual_max_m": float(np.linalg.norm(fitted - actual_xy, axis=-1).max()),
        "limits": limits,
        "continuous_safety_certified_by_audit": bool(len(controls) > 0 and obstacle_lower > 0 and pair_lower > 0),
    }
    for name, column in (("v_max", 0), ("omega_max", 1)):
        if name in limits:
            measured = result["max_abs_v_m_s" if column == 0 else "max_abs_omega_rad_s"]
            result[f"{name}_satisfied"] = bool(measured <= limits[name] + 1e-9)
    result["final_max_abs_v_m_s"] = float(np.abs(velocities[-1]).max()) if velocities is not None else (float(np.abs(controls[-1, :, 0]).max()) if len(controls) else 0.)
    result["max_command_delta_v_m_s"] = float(np.abs(np.diff(controls[:, :, 0], axis=0)).max(initial=0))
    result["max_command_delta_omega_rad_s"] = float(np.abs(np.diff(controls[:, :, 1], axis=0)).max(initial=0))
    if accelerations is not None and velocities is not None:
        result["velocity_update_residual_max_m_s"] = float(np.abs(np.diff(velocities, axis=0)-dt*accelerations[:,:,0]).max(initial=0))
        result["midpoint_execution_speed_residual_max_m_s"] = float(np.abs(controls[:,:,0]-(velocities[:-1]+velocities[1:])/2).max(initial=0))
        result["angular_command_residual_max_rad_s"] = float(np.abs(controls[:,:,1]-accelerations[:,:,1]).max(initial=0))
        result["max_abs_linear_acceleration_m_s2"] = float(np.abs(accelerations[:,:,0]).max(initial=0))
        result["motion_model"] = "Discrete midpoint-speed unicycle arc; not exact continuous constant-acceleration motion"
    if goal is not None:
        result["final_max_robot_goal_error_m"] = float(np.linalg.norm(actual_xy[-1] - goal - reference, axis=1).max())
    return result


def audit_regions(path: Path) -> dict:
    """Check exported convex regions against every occupied or unknown grid cell."""
    with np.load(path, allow_pickle=False) as data:
        states, ticks = data["states"], data["planning_ticks"]
        known, occupied = data["known_history"], data["occupancy"]
        origin, res = data["origin"], float(data["resolution"])
        radius, reference = float(data["robot_radius"]), data["reference"]
    with np.load(path.with_name("regions.npz"), allow_pickle=False) as data:
        coefficients, offsets = data["coefficients"], data["offsets"]
        count = int(data["regions_per_plan"])
    planning = json.loads(path.with_name("planning.json").read_text())
    region_lower = float("inf")
    invalid_anchors = uncertified_edges = uncertified_joins = 0
    pair_min = float("inf")
    checked_edges = 0

    def connected(p0, p1, regions):
        return all(any(np.all(h @ a <= b + 1e-7) and np.all(h @ z <= b + 1e-7)
                       for h, b in regions) for a, z in zip(p0, p1))

    def pair_distance(p0, p1):
        lowest = float("inf")
        for i in range(len(reference)):
            for j in range(i + 1, len(reference)):
                d = p0[i] - p0[j]
                v = p1[i] - p1[j] - d
                t = np.clip(-np.dot(d, v) / np.dot(v, v), 0, 1) if np.dot(v, v) > 0 else 0
                lowest = min(lowest, float(np.linalg.norm(d + t * v)))
        return lowest

    for k, tick in enumerate(ticks):
        blocked = origin + (np.argwhere(~known[k] | occupied) + 0.5) * res
        regions = []
        for i in range(count):
            index = k * count + i
            coef = coefficients[offsets[index]:offsets[index + 1]]
            h, b = coef[:, :2], coef[:, 2]
            regions.append((h, b))
            anchor = states[tick, i, :2]
            if np.min(b - h @ anchor) <= 1e-9:
                invalid_anchors += 1
                continue
            vertices = HalfspaceIntersection(np.column_stack((h, -b)), anchor).intersections
            polygon = vertices[ConvexHull(vertices).vertices]
            start, end = polygon, np.roll(polygon, -1, axis=0)
            edge = end - start
            delta = blocked[:, None, :] - start[None, :, :]
            t = np.clip(np.sum(delta * edge, axis=2) / np.sum(edge ** 2, axis=1), 0, 1)
            distance = np.linalg.norm(delta - t[:, :, None] * edge, axis=2).min(axis=1)
            hull = ConvexHull(polygon).equations
            inside = np.all(blocked @ hull[:, :2].T + hull[:, 2] <= 1e-8, axis=1)
            distance[inside] = 0.0
            region_lower = min(region_lower, float(distance.min()) - radius - res / np.sqrt(2))
        route = np.asarray(planning["paths"][k])
        if len(route) < 2:
            continue
        position = route[:, None, :2] + np.einsum("tij,nj->tni", route[:, 2:].reshape(-1, 2, 2), reference)
        for p0, p1 in zip(position[:-1], position[1:]):
            checked_edges += 1
            uncertified_edges += int(not connected(p0, p1, regions))
            pair_min = min(pair_min, pair_distance(p0, p1))
        assignment = planning["assignments"][k]
        target = position[1, assignment]
        uncertified_joins += int(not connected(states[tick, :, :2], target, regions))
        pair_min = min(pair_min, pair_distance(states[tick, :, :2], target))

    return {
        "regions_checked": len(offsets) - 1,
        "invalid_region_anchors": invalid_anchors,
        "region_clearance_to_blocked_cell_disks_lower_bound_m": region_lower,
        "guidance_edges_checked": checked_edges,
        "uncertified_guidance_edges": uncertified_edges,
        "uncertified_actual_state_joins": uncertified_joins,
        "guidance_and_join_min_pair_center_distance_m": pair_min,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--regions", action="store_true", help="also audit saved SFCs and guidance edges")
    args = parser.parse_args()
    results = [audit(path) for path in args.traces]
    if args.regions:
        for path, result in zip(args.traces, results):
            result["regions"] = audit_regions(path)
    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
