#!/usr/bin/env python3
"""Fixed-N policy switch (A <-> Cost-Near-5): best-so-far length at equal sample counts and equal online time, and
the share of the final region sequence born in the first N iterations (results of ``ablation_coupling.py``)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A, COST = "single_entry_rrtstar@0.3", "single_entry_cost_near5@0.3"
NS = (100, 250, 500)
ITERS = (250, 500, 1000, 2000, 4000, 6000)
TIMES = (0.25, 0.5, 1.0, 2.0)


def records(run):
    return sorted([r for r in [run["first"]] + run["checkpoints"] if r], key=lambda r: r["iteration"])


def best_by(run, key, limit):
    values = [r["optimised_cost"] for r in records(run) if r["success"] and r[key] <= limit + 1e-12]
    return min(values) if values else math.inf


def regions_at(run, n):
    return next(r["regions"] for r in run["checkpoints"] if r["iteration"] == n)


def label(v):
    return v.replace("single_entry_", "").replace("@0.3", "")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "switch_forest"
    res = [json.loads(line) for line in (out / "runs.jsonl").open()]
    by = {(r["variant"], r["seed"]): r for r in res}
    variants = list(dict.fromkeys(r["variant"] for r in res))
    seeds = sorted({r["seed"] for r in res})
    med = lambda x: float(np.median([v for v in x if math.isfinite(v)])) if x else math.nan
    rows = []
    print("best-so-far length, median over seeds, at equal sample counts / equal online time")
    print(f"{'variant':22s} " + " ".join(f"it{n:>5d}" for n in ITERS) + " | " + " ".join(f"{t:>5.2f}s" for t in TIMES)
          + " | t_total  ms/it  pruned")
    for v in variants:
        runs = [by[(v, s)] for s in seeds]
        row = dict(variant=v, by_iteration={n: med([best_by(r, "iteration", n) for r in runs]) for n in ITERS},
                   by_time={t: med([best_by(r, "online_time_s", t) for r in runs]) for t in TIMES},
                   t_total=med([r["online_time_s"] for r in runs]),
                   ms_per_it=med([1e3 * r["online_time_s"] / r["stats"]["iterations"] for r in runs]),
                   pruned=med([r["stats"].get("cost_pruned", 0) / max(1, r["stats"].get("cost_pruned", 0)
                                                                      + r["stats"].get("cost_projections", 0)) for r in runs]))
        for ref in (A, COST):
            if v == ref:
                continue
            diffs = []
            for s in seeds:
                a, b = by[(ref, s)], by[(v, s)]
                limit = min(a["online_time_s"], b["online_time_s"])
                ca, cb = best_by(a, "online_time_s", limit), best_by(b, "online_time_s", limit)
                if math.isfinite(ca) and math.isfinite(cb):
                    diffs.append(100 * (cb / ca - 1))
            row[f"same_time_vs_{label(ref)}"] = (med(diffs), sum(d < -0.1 for d in diffs), sum(d > 0.1 for d in diffs))
            row[f"same_it_vs_{label(ref)}"] = med([100 * (best_by(by[(v, s)], "iteration", 6000)
                                                         / best_by(by[(ref, s)], "iteration", 6000) - 1) for s in seeds])
        row["early_share"] = {}
        for n in NS:
            shares = []
            for r in runs:
                seq, born = r["checkpoints"][-1]["sequence"], regions_at(r, n)
                shares.append(float(np.mean([i < born for i in seq])))
            row["early_share"][n] = med(shares)
        rows.append(row)
        print(f"{label(v):22s} " + " ".join(f"{row['by_iteration'][n]:7.3f}" for n in ITERS) + " | "
              + " ".join(f"{row['by_time'][t]:6.3f}" for t in TIMES)
              + f" | {row['t_total']:6.2f}  {row['ms_per_it']:.3f}  {row['pruned']:.2f}")
    print("\nequal total time (min of both runs) and equal 6000 samples; share of final sequence born in first N iterations")
    for row in rows:
        parts = []
        for ref in (A, COST):
            k = f"same_time_vs_{label(ref)}"
            if k in row:
                d, b, w = row[k]
                parts.append(f"vs {label(ref)}: time {d:+.2f}% (b{b}/w{w}) samples {row[f'same_it_vs_{label(ref)}']:+.2f}%")
        early = " ".join(f"N{n} {row['early_share'][n]:.2f}" for n in NS)
        print(f"{label(row['variant']):22s} {' | '.join(parts):95s} early: {early}")
    with (out / "switch_summary.json").open("w") as stream:
        json.dump(rows, stream, indent=1)


if __name__ == "__main__":
    main()
