"""Plot e_sim, e_track, e_dist from a scenario timeseries CSV."""
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
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="results/right_angle_corridor_timeseries.csv")
    p.add_argument("--metric", choices=["e_sim", "e_track", "e_dist"], default="e_sim")
    p.add_argument("--output", "-o", type=str, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out = args.output or f"results/{args.metric}_{Path(args.csv).stem}.png"

    fig, ax = plt.subplots(figsize=(6, 2.5))
    color_map = {"e_sim": "#1f77b4", "e_track": "#d62728", "e_dist": "#2ca02c"}
    ax.plot(df["time"], df[args.metric], linewidth=1.0, color=color_map.get(args.metric, "black"))
    ax.set_xlabel("time [s]")
    label_map = {"e_sim": r"$e_{\mathrm{sim},k}$", "e_track": r"$\bar{e}_{\mathrm{track}}$", "e_dist": r"$\bar{e}_{\mathrm{dist}}$"}
    ax.set_ylabel(label_map.get(args.metric, args.metric))
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
