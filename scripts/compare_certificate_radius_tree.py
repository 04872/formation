#!/usr/bin/env python3
"""6000-attempt paired A / owner-reference / certificate-radius comparison.

Online time excludes every checkpoint (extraction and shared chain optimisation)
and clearance metrics. Runs execute serially and are appended for resumption.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/formation-matplotlib")

import argparse
import csv
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_coupling import DESCRIPTIONS, route_clearance, scenario
from compare_single_tree import initialize_a, step_a
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar
from formation.single_tree_rrtstar import SingleTreeConfig, SingleTreeRRTStar
from formation.certificate_radius_tree import CertificateRadiusConfig, CertificateRadiusTree

VARIANTS = ("A", "single_tree", "certificate_radius")
MILESTONES = (2000, 4000, 6000)


def fixed_checkpoint(p, single, iteration, online, wall_start):
    began = time.perf_counter()
    rec = p._checkpoint(iteration, online)
    rec["raw_success"] = bool(rec["success"] and math.isfinite(rec.get("raw_cost", math.inf)))
    if single and rec["raw_success"]:
        solved_at = time.perf_counter()
        solved = p.geo.chain(p.start, p.goal_xy, [p.cells[i] for i in rec["owners"]],
                             p.config.overlap_margin * (1.0 - 1e-6))
        rec["optimise_time_s"] = time.perf_counter() - solved_at
        rec["tree_route"] = rec["route"]
        rec["optimised_cost"] = solved[0] if solved else math.nan
        rec["route"] = [(q.x, q.y, q.yaw) for q in solved[1]] if solved else []
    if not single and rec["raw_success"]:
        sequence=rec["sequence"]
        last=sequence[-1]
        yaw=p.geo.goal_yaw(p.cells[last],p.goal_xy,p.entry[last].yaw,p.config.overlap_margin)
        rec["tree_route"]=[(p.entry[i].x,p.entry[i].y,p.entry[i].yaw) for i in sequence]+[(*p.goal_xy,yaw)]
    if isinstance(p, CertificateRadiusTree) and rec["raw_success"]:
        rec["edge_corridors"]=[list(p.portal[v]) for v in rec["sequence"][1:]]
        rec["certificate_seeds"]={str(i):[p.cells[i].pose.x,p.cells[i].pose.y,p.cells[i].pose.yaw] for i in rec["owners"]}
        rec["chain_basis"]="actual edge certificate corridor; extra portal delta .02; independent of raw success"
    rec["success"] = bool(rec["success"] and math.isfinite(rec.get("optimised_cost", math.inf)))
    rec.update(checkpoint_time_s=time.perf_counter()-began, wall_time_s=time.perf_counter()-wall_start,
               nodes=len(p.parent), stats=dict(p.stats), solver_failures=p.geo.failures)
    return rec


def run_one(name, variant, seed):
    single = variant != "A"
    common = dict(seed=seed, step=.3, max_iterations=6000, checkpoints=tuple(range(50, 6001, 50)))
    config = (CertificateRadiusConfig(**common, goal_margin=.02) if variant == "certificate_radius" else
              SingleTreeConfig(**common, neighbor_radius=1.5, goal_margin=.02) if single else CouplingConfig(**common))
    map_data, slots, start_pose = scenario(name)
    wall_start = time.perf_counter()
    planner = CertificateRadiusTree if variant == "certificate_radius" else SingleTreeRRTStar if single else SingleEntryRRTStar
    p = planner(map_data, slots, start_pose, config)
    (p.initialize() if single else initialize_a(p))
    init_s = time.perf_counter()-wall_start
    sample_calls, original_sample = 0, p._sample

    def counted_sample():
        nonlocal sample_calls
        sample_calls += 1
        return original_sample()

    p._sample = counted_sample
    records, online, first, error = [], 0., None, None
    iteration = 0
    try:
        for iteration in range(1, 6001):
            began = time.perf_counter()
            (p.step() if single else step_a(p))
            online += time.perf_counter()-began
            p.stats["iterations"] = iteration
            reachable = bool(p.goal_nodes if single else p.goal_cells)
            if first is None and reachable:
                first = fixed_checkpoint(p, single, iteration, online, wall_start)
                first["kind"] = "first"
                records.append(first)
            if iteration % 50 == 0:
                rec = fixed_checkpoint(p, single, iteration, online, wall_start)
                rec["kind"] = "checkpoint"
                records.append(rec)
    except Exception:
        error = traceback.format_exc()
    planning_wall_s = time.perf_counter()-wall_start
    # Match the historical cost-near script: clearance metrics on first/final
    # routes, outside online time; retain routes attaining historical best too.
    metric_at = time.perf_counter()
    keep = {id(first)}
    if records:
        keep.add(id(records[-1]))
    for key, success in (("raw_cost", "raw_success"), ("optimised_cost", "success")):
        good = [r for r in records if r[success]]
        if good:
            keep.add(id(min(good, key=lambda r: r[key])))
    for rec in records:
        if id(rec) in keep:
            if rec.get("route"):
                rec["route_clearance"] = route_clearance(map_data, slots, rec["route"])
                if rec["route_clearance"] < config.safety_distance-1e-6:
                    rec["success"] = False
                    rec["validation_failed"] = True
            if rec.get("tree_route"):
                rec["tree_route_clearance"] = route_clearance(map_data, slots, rec["tree_route"])
                if rec["tree_route_clearance"] < config.safety_distance-1e-6:
                    rec["raw_success"] = rec["success"] = False
                    rec["validation_failed"] = True
        else:
            for key in ("route", "tree_route", "sequence", "owners", "edge_corridors", "certificate_seeds"):
                rec.pop(key, None)
    metric_s = time.perf_counter()-metric_at
    result = dict(scenario=name, variant=variant, seed=seed, config=asdict(config), records=records,
                  first_iteration=first["iteration"] if first else None,
                  first_online_time_s=first["online_time_s"] if first else None,
                  sample_calls=sample_calls, completed_iterations=iteration,
                  online_time_s=online, initialisation_time_s=init_s,
                  checkpoint_time_s=sum(r["checkpoint_time_s"] for r in records),
                  chain_time_s=sum(r.get("optimise_time_s", 0.) for r in records),
                  clearance_metrics_time_s=metric_s, planning_wall_s=planning_wall_s,
                  actual_wall_s=time.perf_counter()-wall_start,
                  nodes=len(p.parent), certificates=len(p.cells), solver_failures=p.geo.failures,
                  stats=dict(p.stats))
    if error:
        result["error"] = error
    return result


def median(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(values)) if values else None


def upto(run, count):
    return [r for r in run.get("records", []) if r["iteration"] <= count]


def csv_write(out, name, rows):
    if not rows:
        return
    with (out/name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(runs, out):
    rows, detailed, paired, pair_runs = [], [], [], []
    scenes = list(dict.fromkeys(r["scenario"] for r in runs))
    for name in scenes:
        for count in MILESTONES:
            for variant in VARIANTS:
                group = [r for r in runs if r["scenario"] == name and r["variant"] == variant]
                entries=[]
                for run in group:
                    recs=upto(run,count)
                    current=next((r for r in reversed(recs) if r["kind"]=="checkpoint"),{})
                    raw=[r for r in recs if r["raw_success"]]
                    chain=[r for r in recs if r["success"]]
                    complete=current.get("iteration")==count and not run.get("error")
                    stats=current.get("stats",{})
                    entry=dict(scenario=name,samples=count,variant=variant,seed=run["seed"],
                               raw_success=bool(raw) and complete,chain_success=bool(chain) and complete,
                               first_iteration=raw[0]["iteration"] if raw else None,
                               first_s=raw[0]["online_time_s"] if raw else None,
                               first_raw=raw[0]["raw_cost"] if raw else None,
                               current_raw=current.get("raw_cost") if current.get("raw_success") else None,
                               best_raw=min((r["raw_cost"] for r in raw),default=None),
                               current_chain=current.get("optimised_cost") if current.get("success") else None,
                               best_chain=min((r["optimised_cost"] for r in chain),default=None),
                               online_s=current.get("online_time_s"),
                               checkpoint_s=sum(r["checkpoint_time_s"] for r in recs),
                               chain_s=sum(r.get("optimise_time_s",0.) for r in recs),
                               actual_final_wall_s=run.get("actual_wall_s") if count==6000 else None,
                               nodes=current.get("nodes"),certificates=current.get("regions"))
                    for key in ("neighbor_candidates","choose_parent_tests","rewire_tests", "local_connect_calls",
                                "certificate_tests","local_certificates","local_certificate_builds","local_unknown",
                                "common_certificate_edges","certificate_cover_edges","edge_cache_hits","local_invalid","local_cap_unknown","local_numerical_unknown","direct_samples",
                                "direct_rejected","radius","select_time_s","coverage_time_s","steer_time_s",
                                "connect_time_s","propagation_time_s"):
                        entry[key]=stats.get(key,0.)
                    entry["raw_improvement_pct"]=100*(1-entry["best_raw"]/entry["first_raw"]) if raw else None
                    entries.append(entry); detailed.append(entry)
                row=dict(scenario=name,samples=count,variant=variant,runs=len(group),
                         raw_success=sum(e["raw_success"] for e in entries),
                         chain_success=sum(e["chain_success"] for e in entries),
                         errors=sum(bool(r.get("error")) for r in group))
                for key in entries[0] if entries else []:
                    if key not in ("scenario","samples","variant","seed","raw_success","chain_success"):
                        row[key]=median([e[key] for e in entries])
                rows.append(row)
    lookup={(r["scenario"],r["variant"],r["seed"],r["samples"]):r for r in detailed}
    for name in scenes:
        for count in MILESTONES:
            for baseline,new in (("A","single_tree"),("A","certificate_radius"),("single_tree","certificate_radius")):
                pairs=[]
                for seed in sorted({r["seed"] for r in runs if r["scenario"]==name}):
                    a,b=(lookup.get((name,v,seed,count),{}) for v in (baseline,new))
                    if not (a.get("raw_success") and b.get("raw_success")):
                        continue
                    pair=dict(scenario=name,samples=count,baseline=baseline,variant=new,seed=seed)
                    for key in ("best_raw","current_raw","online_s","first_s","first_iteration"):
                        pair[key+"_change_pct"]=100*(b[key]/a[key]-1)
                    for key in ("best_chain","current_chain"):
                        pair[key+"_change_pct"]=100*(b[key]/a[key]-1) if a.get(key) and b.get(key) else None
                    pairs.append(pair); pair_runs.append(pair)
                row=dict(scenario=name,samples=count,baseline=baseline,variant=new,common_raw_success=len(pairs))
                for key in ("best_raw","current_raw","online_s","first_s","first_iteration","best_chain","current_chain"):
                    row[key+"_change_pct"]=median([p[key+"_change_pct"] for p in pairs])
                row["common_current_chain"]=sum(p["current_chain_change_pct"] is not None for p in pairs)
                paired.append(row)
    for name,data in (("summary.csv",rows),("milestone_runs.csv",detailed),("paired.csv",paired),("paired_runs.csv",pair_runs)):
        csv_write(out,name,data)
    (out/"summary.json").write_text(json.dumps(dict(summary=rows,paired=paired),indent=2))
    errors=[dict(scenario=r["scenario"],variant=r["variant"],seed=r["seed"],error=r["error"]) for r in runs if r.get("error")]
    failures=dict(errors=errors,solver_failures=sum(r.get("solver_failures",0) for r in runs),
                  incomplete=[dict(scenario=r["scenario"],variant=r["variant"],seed=r["seed"],sample_calls=r.get("sample_calls"))
                              for r in runs if r.get("sample_calls")!=6000],
                  clearance_failures=[dict(scenario=r["scenario"],variant=r["variant"],seed=r["seed"],iteration=q["iteration"])
                                      for r in runs for q in r["records"] if q.get("validation_failed")])
    (out/"failures.json").write_text(json.dumps(failures,indent=2))
    return rows,paired


def plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenes = list(dict.fromkeys(r["scenario"] for r in runs))
    fig, axes = plt.subplots(2, len(scenes), figsize=(5*len(scenes), 7), squeeze=False)
    for col, name in enumerate(scenes):
        for variant, color in (("A", "C3"), ("single_tree", "C0"), ("certificate_radius", "C2")):
            group = [r for r in runs if r["scenario"] == name and r["variant"] == variant]
            for row, (key, success) in enumerate((("raw_cost", "raw_success"), ("optimised_cost", "success"))):
                x, y = [], []
                for count in range(50, 6001, 50):
                    values = [min((r[key] for r in upto(run, count) if r[success]), default=math.inf) for run in group]
                    # Unsolved runs remain +inf in the median; no silent omission.
                    cost = float(np.median(values))
                    if math.isfinite(cost):
                        x.append(count)
                        y.append(cost)
                axes[row, col].plot(x, y, label=variant, color=color)
                axes[row, col].set(title=name, xlabel="Main-loop samples", ylabel="Best raw cost" if row == 0 else "Best cost + chain")
                axes[row, col].grid(alpha=.2)
                axes[row, col].legend()
    fig.tight_layout()
    fig.savefig(out/"anytime_samples.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=["forest", "dense", "slots"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"results"/"certificate_radius_tree")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out/"runs.jsonl"
    runs = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    if not args.analyze_only:
        manifest = dict(command=sys.argv, python=sys.executable, scenarios={s:DESCRIPTIONS[s] for s in args.scenarios},
                        seeds=list(range(args.seeds)), iterations=6000, checkpoint_every=50,
                        timing="serial; online=sum(main-loop steps), excludes initialization/checkpoints/clearance",
                        paired_order="A/single_tree/certificate_radius for even seeds, reverse for odd seeds",
                        baseline="SingleEntryRRTStar, PolyhedralCellModel, step .3, goal margin .02",
                        new="ownerless certificate edges; attempt radius min(1.5,1.5*(log(n+e)^2/n)^.25); goal margin .02",
                        margins="physical .02 unchanged; raw new edges need certificate coverage but no extra portal delta; corridor chain uses delta .02",
                        primary="raw success and raw cost; chain availability separately recorded",
                        sampling="one _sample call per main-loop step; owner RNG is not a main-loop sample")
        if not (out/"experiment.json").exists():
            (out/"experiment.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        source = out/"source"
        source.mkdir(exist_ok=True)
        for filename in ("formation/single_tree_rrtstar.py", "formation/certificate_radius_tree.py", "formation/region_coupling.py", "formation/polyhedral_cell.py", "scripts/compare_single_tree.py", "scripts/compare_certificate_radius_tree.py"):
            if not (source/Path(filename).name).exists():
                shutil.copyfile(ROOT/filename, source/Path(filename).name)
        (out/"environment.json").write_text(json.dumps(dict(python=sys.version, platform=platform.platform(),
            repo_head=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
            packages={p:version(p) for p in ("numpy","scipy","clarabel","matplotlib")}), indent=2))
        done = {(r["scenario"],r["variant"],r["seed"]) for r in runs}
        expected = len(args.scenarios)*args.seeds*3
        session_at, session_runs, session_wall = time.perf_counter(), 0, 0.
        with path.open("a") as stream:
            for name in args.scenarios:
                for seed in range(args.seeds):
                    for variant in (VARIANTS if seed % 2 == 0 else VARIANTS[::-1]):
                        if (name,variant,seed) in done:
                            continue
                        began = time.perf_counter()
                        try:
                            run = run_one(name, variant, seed)
                        except Exception:
                            run = dict(scenario=name, variant=variant, seed=seed, records=[], error=traceback.format_exc())
                        stream.write(json.dumps(run)+"\n")
                        stream.flush()
                        runs.append(run)
                        done.add((name,variant,seed))
                        session_runs += 1
                        session_wall += time.perf_counter()-began
                        eta = session_wall/session_runs*(expected-len(done))/60
                        print(f"{len(done)}/{expected} {name} seed={seed} {variant} samples={run.get('sample_calls')} "
                              f"online={run.get('online_time_s',0):.3f}s wall={run.get('actual_wall_s',0):.3f}s "
                              f"nodes/certs={run.get('nodes')}/{run.get('certificates')} eta={eta:.1f}min "
                              f"status={'ERROR '+run['error'] if run.get('error') else 'ok'}", flush=True)
                scene_runs = [r for r in runs if r["scenario"] == name]
                if len(scene_runs) >= args.seeds*3:
                    _, pairs = analyse(runs, out)
                    print("SCENARIO_COMPLETE",name,json.dumps([p for p in pairs if p["scenario"]==name]),flush=True)
    rows, pairs = analyse(runs, out)
    plot(runs, out)
    print(json.dumps(dict(summary=rows, paired=pairs), indent=2), flush=True)


if __name__ == "__main__":
    main()
