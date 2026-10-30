#!/usr/bin/env python3
"""Summary of cost-guided region selection against A@0.3 and BIT* (results of ``ablation_coupling.py``)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ablation_coupling import cost_at
from external_baselines import best_at

BASE = "single_entry_rrtstar@0.3"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "cost_near_all"
    res = [json.loads(line) for line in (out / "runs.jsonl").open()]
    by = {(r["scenario"], r["variant"], r["seed"]): r for r in res}
    bit = {(r["scenario"], r["seed"]): r for r in map(json.loads, (ROOT / "results" / "external" / "runs.jsonl").open())
           if r["variant"] == "bitstar"}
    scenarios = list(dict.fromkeys(r["scenario"] for r in res))
    variants = list(dict.fromkeys(r["variant"] for r in res))
    seeds = sorted({r["seed"] for r in res})
    med = lambda x: float(np.median(x)) if len(x) else math.nan
    rows = []
    for s in scenarios:
        for v in variants:
            g = [by[(s, v, k)] for k in seeds]
            final = [r["checkpoints"][-1] for r in g]
            row = dict(scenario=s, variant=v, success=sum(f["success"] for f in final),
                       L=med([f["optimised_cost"] for f in final]),
                       t_total=med([r["online_time_s"] for r in g]),
                       t_first=med([r["first"]["online_time_s"] for r in g if r["first"]]),
                       it_first=med([r["first"]["iteration"] for r in g if r["first"]]),
                       ms_per_it_first=med([1e3 * r["first"]["online_time_s"] / r["first"]["iteration"] for r in g if r["first"]]),
                       ms_per_it=med([1e3 * r["online_time_s"] / r["stats"]["iterations"] for r in g]),
                       regions_per_s=med([r["regions"] / r["online_time_s"] for r in g]),
                       select_s=med([r["stats"]["select_time_s"] for r in g]),
                       min_clearance=min(f.get("route_clearance", math.inf) for f in final),
                       vs_bit_10s=med([100 * (f["optimised_cost"] / best_at(bit[(s, k)], math.inf) - 1)
                                       for k, f in zip(seeds, final)]))
            if v != BASE:
                same_t, same_it, first, ratio, it_ratio = [], [], [], [], []
                for k in seeds:
                    a, b = by[(s, BASE, k)], by[(s, v, k)]
                    limit = min(a["online_time_s"], b["online_time_s"])
                    ca, cb = cost_at(a, limit), cost_at(b, limit)
                    if math.isfinite(ca) and math.isfinite(cb):
                        same_t.append(100 * (cb / ca - 1))
                    same_it.append(100 * (b["checkpoints"][-1]["optimised_cost"] / a["checkpoints"][-1]["optimised_cost"] - 1))
                    if a["first"] and b["first"]:
                        first.append(b["first"]["online_time_s"] / a["first"]["online_time_s"])
                        it_ratio.append(b["first"]["iteration"] / a["first"]["iteration"])
                    ratio.append(b["online_time_s"] / a["online_time_s"])
                row.update(same_time=med(same_t), better=sum(x < -0.1 for x in same_t), worse=sum(x > 0.1 for x in same_t),
                           same_it=med(same_it), time_ratio=med(ratio), first_ratio=med(first),
                           first_it_ratio=med(it_ratio))
            rows.append(row)
    with (out / "cost_near_summary.json").open("w") as stream:
        json.dump(rows, stream, indent=1)
    for row in rows:
        extra = (f"  same-time {row['same_time']:+.2f}% (b{row['better']}/w{row['worse']})  same-it {row['same_it']:+.2f}%  "
                 f"time x{row['time_ratio']:.2f}  first time x{row['first_ratio']:.2f} it x{row['first_it_ratio']:.2f}"
                 ) if "same_time" in row else ""
        print(f"{row['scenario']:7s} {row['variant']:40s} succ {row['success']}  L {row['L']:.3f}  t {row['t_total']:.2f}  "
              f"first {row['t_first']:.3f}s @it {row['it_first']:.0f} ({row['ms_per_it_first']:.3f} ms/it; all "
              f"{row['ms_per_it']:.3f} ms/it)  reg/s {row['regions_per_s']:.0f}  select {row['select_s']:.2f}s  "
              f"clr {row['min_clearance']:.3f}  vs BIT* 10 s {row['vs_bit_10s']:+.2f}%{extra}")


if __name__ == "__main__":
    main()
