"""Replay saved physical robot states and affine references on the sensed map."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formation.paper_formation.plotting import draw_obstacles


def render(trace_path: Path, output: Path, stride: int = 3) -> None:
    with np.load(trace_path, allow_pickle=False) as data:
        states, q = data["states"], data["q_history"]
        occupancy, known = data["occupancy"], data["known_history"]
        ticks, reference = data["planning_ticks"], data["reference"]
        origin, resolution = data["origin"], float(data["resolution"])
        radius, dt = float(data["robot_radius"]), float(data["dt"])
        assignments = data["assignment_history"]
        goal = data["goal"]
        obstacles = data["circles"] if "circles" in data else []

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = (origin[0], origin[0] + occupancy.shape[0] * resolution,
              origin[1], origin[1] + occupancy.shape[1] * resolution)
    draw_obstacles(ax, occupancy, resolution, origin, obstacles)
    unknown = ax.imshow((~known[0]).T, origin="lower", extent=extent,
                        cmap="Blues", vmin=0, vmax=1, alpha=0.25)
    ax.scatter(*(goal + reference).T, marker="*", c="black", s=65, label="goal slots")
    colors = plt.get_cmap("tab10").colors
    circles, headings, tracks, targets = [], [], [], []
    for i in range(states.shape[1]):
        circle = Circle(states[0, i, :2], radius, color=colors[i], alpha=0.85)
        ax.add_patch(circle)
        circles.append(circle)
        headings.append(ax.plot([], [], color="black", lw=1.5)[0])
        tracks.append(ax.plot([], [], color=colors[i], alpha=0.6, lw=1)[0])
        targets.append(ax.plot([], [], "+", color=colors[i], ms=9)[0])
    title = ax.set_title("")
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.text(0.5, 0.02, "Disks: executed robots | +: formation reference | blue tint: unknown at last replan",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    def update(k: int):
        map_index = min(len(known) - 1, max(0, np.searchsorted(ticks, k, side="right") - 1))
        unknown.set_data((~known[map_index]).T)
        matrix = q[k, 2:].reshape(2, 2)
        target = q[k, :2] + reference[assignments[k]] @ matrix.T
        for i, circle in enumerate(circles):
            xy, theta = states[k, i, :2], states[k, i, 2]
            circle.center = xy
            tip = xy + radius * np.array([np.cos(theta), np.sin(theta)])
            headings[i].set_data([xy[0], tip[0]], [xy[1], tip[1]])
            tracks[i].set_data(states[:k + 1, i, 0], states[:k + 1, i, 1])
            targets[i].set_data([target[i, 0]], [target[i, 1]])
        title.set_text(f"{trace_path.parent.name} | t={k * dt:.1f}s\n"
                       f"A = [[{matrix[0, 0]:.2f}, {matrix[0, 1]:.2f}], "
                       f"[{matrix[1, 0]:.2f}, {matrix[1, 1]:.2f}]]")
        return [unknown, title, *circles, *headings, *tracks, *targets]

    frames = list(range(0, len(states), stride))
    if frames[-1] != len(states) - 1:
        frames.append(len(states) - 1)
    animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    animation.save(output, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    render(args.trace, args.output or args.trace.with_name("replay.gif"), args.stride)
