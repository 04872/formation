"""Quickly visualise any map with start/goal markers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from formation import MapBuilder
from formation.map_config import (
    NarrowEntranceConfig, NarrowingCorridorConfig, ObstacleClusterConfig,
    RightAngleCorridorConfig, SCurveCorridorConfig,
)

CONFIGS = {
    "right_angle_corridor": RightAngleCorridorConfig,
    "narrowing_corridor": NarrowingCorridorConfig,
    "s_curve_corridor": SCurveCorridorConfig,
    "obstacle_cluster": ObstacleClusterConfig,
    "narrow_entrance": NarrowEntranceConfig,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map", choices=list(CONFIGS), default="narrow_entrance", nargs="?")
    p.add_argument("--robot-radius", type=float, default=0.09)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    config = CONFIGS[args.map](robot_radius=args.robot_radius)
    map_data = MapBuilder().build(args.map, config)

    fig, ax = plt.subplots(figsize=(7, 7))
    extent = [
        map_data.origin_xy[0], map_data.origin_xy[0] + map_data.width_m,
        map_data.origin_xy[1], map_data.origin_xy[1] + map_data.height_m,
    ]
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent,
              cmap="gray_r", interpolation="nearest")

    ax.scatter(*map_data.start_xy, c="lime", s=120, marker="o",
               edgecolors="black", linewidths=1.5, zorder=5, label="start")
    ax.scatter(*map_data.goal_xy, c="fuchsia", s=120, marker="*",
               edgecolors="black", linewidths=1.5, zorder=5, label="goal")
    ax.plot([], [], "ko", markersize=10, label="robot")

    ax.set_title(f"{args.map}\nstart {map_data.start_xy}  goal {map_data.goal_xy}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")

    out = args.save or f"results/map_{args.map}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
