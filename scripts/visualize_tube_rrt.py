from __future__ import annotations

import time

_MODULE_START = time.perf_counter()

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon

from formation import (
    FormationLibrary,
    MapBuilder,
    Pose2D,
    RandomCirclesConfig,
    TubeRRTConfig,
    TubeRRTPlanner,
    project_robot_paths,
    transform_slots,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Joint Tube-RRT plan.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--formation", default="square", choices=["square", "column", "horizontal_line", "t_shape"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")

    print(f"timing startup_imports={time.perf_counter() - _MODULE_START:.3f}s", flush=True)
    map_start = time.perf_counter()
    map_config = RandomCirclesConfig(seed=args.seed)
    map_data = MapBuilder().build("random_circles", map_config)
    print(f"timing map_build={time.perf_counter() - map_start:.3f}s", flush=True)
    formation = FormationLibrary.build_default(map_data.robot_radius).get(args.formation)
    planner = TubeRRTPlanner(
        map_data,
        formation,
        Pose2D(*map_data.start_xy, 0.0),
        config=TubeRRTConfig(seed=args.seed, progress_interval=args.progress_interval),
    )
    print("planning...", flush=True)
    planning_start = time.perf_counter()
    result = planner.plan()
    print(f"timing tube_rrt={time.perf_counter() - planning_start:.3f}s", flush=True)
    print(
        f"planning result: success={result.success} path_nodes={len(result.path_poses)} "
        f"tube_bottleneck={result.bottleneck:.3f}",
        flush=True,
    )

    plot_start = time.perf_counter()
    figure, axis = plt.subplots(figsize=(11, 7))
    ox, oy = map_data.origin_xy
    axis.set_xlim(ox, ox + map_data.width_m)
    axis.set_ylim(oy, oy + map_data.height_m)
    axis.set_aspect("equal")
    for primitive in map_data.obstacle_primitives:
        cx, cy = primitive["center_xy"]
        axis.add_patch(Circle((cx, cy), primitive["radius"], color="0.2", alpha=0.8))
    tree_segments = []
    for node in result.tree_nodes:
        if node.parent is None:
            continue
        parent = result.tree_nodes[node.parent]
        tree_segments.append(((parent.pose.x, parent.pose.y), (node.pose.x, node.pose.y)))
    if tree_segments:
        axis.add_collection(LineCollection(tree_segments, colors="tab:blue", alpha=0.12, linewidths=0.5))
    if result.path_poses:
        axis.plot([p.x for p in result.path_poses], [p.y for p in result.path_poses], color="tab:red", linewidth=2.2, label="center path")
        projected = project_robot_paths(result.path_poses, formation.slots)
        for index, robot_path in enumerate(projected):
            axis.plot([p[0] for p in robot_path], [p[1] for p in robot_path], linewidth=1.1, label=f"robot {index}")
        local_min = formation.slots.min(axis=0) - planner.robot_radius
        local_max = formation.slots.max(axis=0) + planner.robot_radius
        corners = [
            (local_min[0], local_min[1]),
            (local_max[0], local_min[1]),
            (local_max[0], local_max[1]),
            (local_min[0], local_max[1]),
        ]
        for pose in result.path_poses[:: max(1, len(result.path_poses) // 10)]:
            for x, y in (robot_path[0] for robot_path in project_robot_paths([pose], formation.slots)):
                axis.add_patch(Circle((x, y), planner.robot_radius, fill=False, color="tab:green", alpha=0.35))
            world_corners = transform_slots(pose, corners)
            axis.add_patch(Polygon(world_corners, closed=True, fill=False, linestyle="--", color="tab:purple", alpha=0.3))
    axis.scatter(*map_data.start_xy, color="tab:green", label="start")
    axis.scatter(*map_data.goal_xy, color="tab:orange", label="goal")
    axis.set_title(f"Joint Tube-RRT: {'success' if result.success else result.failure_reason}; bottleneck={result.bottleneck:.3f} m")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    print(f"timing plot_build={time.perf_counter() - plot_start:.3f}s", flush=True)
    if args.output is None:
        print(f"timing total_before_show={time.perf_counter() - _MODULE_START:.3f}s", flush=True)
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_start = time.perf_counter()
        figure.savefig(args.output, dpi=150)
        save_duration = time.perf_counter() - save_start
        print(f"timing save={save_duration:.3f}s total={time.perf_counter() - _MODULE_START:.3f}s", flush=True)
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
