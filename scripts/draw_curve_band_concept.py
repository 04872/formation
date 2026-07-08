"""Curve‑band concept figure — narrow_entrance cycle 5."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_swept_band_rollout_v2 import build_pipeline, map_extent


def main():
    ctx = build_pipeline("narrow_entrance", 3.5, goal_tolerance=None,
                         max_replans=5, replan_interval=10)
    trace = ctx["trace"]
    map_data = ctx["map_data"]
    feasi = getattr(ctx.get("_sim"), "_feasibility", None)
    ci = 4

    C0  = np.array(trace.metadata["per_cycle_preview_points"][ci])
    C_R_all = getattr(feasi, "_band_recenter_history", [])
    C_R = np.array(C_R_all[ci]) if ci < len(C_R_all) else C0
    v2_cache = getattr(feasi, "_v2_cache", {})
    dL = v2_cache.get("delta_L", 0.45)
    dR = v2_cache.get("delta_R", 0.45)

    dC = np.diff(C_R, axis=0)
    tang = np.zeros_like(C_R)
    tang[:-1] = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-12)
    tang[-1] = tang[-2]
    norm = np.column_stack((-tang[:, 1], tang[:, 0]))
    upper = C_R + dL * norm
    lower = C_R - dR * norm

    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    extent = map_extent(map_data)

    # obstacles as grey  (not black)
    from matplotlib.colors import ListedColormap
    grey_cmap = ListedColormap(["white", "0.65"])
    ax.imshow(map_data.occupancy.astype(float), origin="lower", extent=extent,
              cmap=grey_cmap, interpolation="nearest")
    ax.set_aspect("equal")
    ax.axis("off")

    # band
    band_poly = np.vstack([upper, lower[::-1]])
    ax.fill(band_poly[:, 0], band_poly[:, 1],
            fc="#4a90d9", alpha=0.18, ec="none", zorder=2)
    ax.plot(upper[:, 0], upper[:, 1], color="0.55", lw=0.6, zorder=3)
    ax.plot(lower[:, 0], lower[:, 1], color="0.55", lw=0.6, zorder=3)

    # curves
    ax.plot(C0[:, 0], C0[:, 1], "k--", lw=1.0, alpha=0.6, zorder=4,
            label=r"$C^0(s)$")
    ax.plot(C_R[:, 0], C_R[:, 1], "k-", lw=1.8, zorder=4,
            label=r"$C_R(s)$")

    # formation centre  (red filled square at start)
    ax.plot(C0[0, 0], C0[0, 1], "s", color="red", ms=5, zorder=5)
    ax.plot(C0[-1, 0], C0[-1, 1], "ko", ms=4, zorder=5)

    # width annotations at start  — labels beside arrows
    k0 = 0
    ax.annotate("", xy=(upper[k0, 0], upper[k0, 1]),
                xytext=(C_R[k0, 0], C_R[k0, 1]),
                arrowprops=dict(arrowstyle="<->", color="C0", lw=1.0))
    ax.text(upper[k0, 0] + 0.10, upper[k0, 1],
            r"$\delta_L$", fontsize=9, color="C0", va="center")
    ax.annotate("", xy=(lower[k0, 0], lower[k0, 1]),
                xytext=(C_R[k0, 0], C_R[k0, 1]),
                arrowprops=dict(arrowstyle="<->", color="C2", lw=1.0))
    ax.text(lower[k0, 0] + 0.10, lower[k0, 1],
            r"$\delta_R$", fontsize=9, color="C2", va="center")

    # Obstacle labels
    ax.text(0.0, 1.4, "Obstacle", fontsize=8, color="black", ha="center")
    ax.text(0.0, -1.2, "Obstacle", fontsize=8, color="black", ha="center")

    # legend  — C0, CR, centre, local target
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="k", ls="--", lw=1.0, label=r"$C^0(s)$"),
        Line2D([0], [0], color="k", ls="-",  lw=1.8, label=r"$C_R(s)$"),
        Line2D([0], [0], marker="s", color="red",  ls="none", ms=5,
               label="Formation centre"),
        Line2D([0], [0], marker="o", color="black", ls="none", ms=4,
               markerfacecolor="black",
               label="Local target"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower left", ncol=1,
              framealpha=0.9, edgecolor="0.5")

    # Obstacle label  (inside the top grey obstacle area, within crop)
    ax.text(0.0, 1.4, "Obstacle", fontsize=8, color="black", ha="center")

    margin = 1.5
    ax.set_xlim(C0[:, 0].min() - margin, C0[:, 0].max() + margin)
    ax.set_ylim(C0[:, 1].min() - margin, C0[:, 1].max() + margin)

    out = Path("results/curve_band_concept.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
