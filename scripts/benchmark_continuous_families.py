#!/usr/bin/env python3
"""Matched family experiments for continuous-cell RRT*, without changing production defaults."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from formation import (LaunchConfig, MapData, Pose2D, RegionTubeConfig, RegionTubeRRTPlanner,
                       TubeRRTConfig)
from formation.tube_cell_first_order import PortalCertificate

VARIANTS = ('seed_launch', 'fixed_portal', 'no_rewire', 'continuous', 'release')


class FixedPortalPlanner(RegionTubeRRTPlanner):
    """Valid fixed geometric witness, without continuous portal minimisation."""
    def _attachment(self, first, second, entry):
        started = time.perf_counter()
        cert = self.cells.overlap(first, second)
        self._stats['attachment_queries'] += 1
        self._stats['attachment_time_s'] += time.perf_counter()-started
        if cert is None:
            return None
        return PortalCertificate(cert.portal, cert.slack, self.metric(entry, cert.portal))


class NoRewirePlanner(RegionTubeRRTPlanner):
    """Continuous launch and ChooseParent; existing nodes keep their parents."""
    def _connect(self, candidate, cell, witness, separated, kind, source_entry=None):
        if kind == 'rewire':
            return None, 'disabled'
        return super()._connect(candidate, cell, witness, separated, kind, source_entry)


def families():
    def circle(x, y, r):
        return dict(type='circle', center_xy=(float(x), float(y)), radius=float(r))
    def vertical(x, gap, halfwidth, radius=.30):
        ys = np.r_[np.arange(.25, gap-halfwidth, .4), np.arange(gap+halfwidth, 9.76, .4)]
        return [circle(x, y, radius) for y in ys]
    square = np.array([[-.22, -.22], [-.22, .22], [.22, -.22], [.22, .22]])
    for level in range(2):
        yield dict(family='open_control', level=level, obstacles=[], slots=square*(1.+level),
                   start=(1.8, 5.), goal=(12.2, 5.))
        halfwidth = (1.05, .75)[level]
        yield dict(family='single_gate', level=level, obstacles=vertical(7., 5., halfwidth),
                   slots=square, start=(1.8, 5.), goal=(12.2, 5.))
        obstacles = vertical(4.7, 3.2, (1.2, .9)[level])+vertical(9.1, 6.8, (1.2, .9)[level])
        yield dict(family='alternating_gates', level=level, obstacles=obstacles,
                   slots=square, start=(1.8, 5.), goal=(12.2, 5.))
        walls = [circle(x, y, .28) for y in (2.8, 7.2) for x in np.arange(1.8, 12.21, .4)]
        posts = [circle(x, y, (.72, .94)[level]) for x, y in zip((4., 6., 8., 10.), (4.1, 5.9, 4.1, 5.9))]
        yield dict(family='slalom', level=level, obstacles=walls+posts,
                   slots=square, start=(1.8, 5.), goal=(12.2, 5.))
        height = (1.4, 2.1)[level]
        obstacles = ([circle(6., y, .28) for y in np.arange(5.-height, 5.+height+.01, .35)]
                     +[circle(x, y, .28) for y in (5.-height, 5.+height) for x in np.arange(6., 10.01, .35)])
        yield dict(family='u_trap', level=level, obstacles=obstacles,
                   slots=square, start=(2., 5.), goal=(8., 5.))
        rng = np.random.default_rng(101+level)
        obstacles = [circle(x+rng.uniform(-.25, .25), y+rng.uniform(-.35, .35), (.38, .55)[level])
                     for x in (3.7, 5.9, 8.1, 10.3) for y in (1.8, 3.9, 6.1, 8.2)]
        yield dict(family='staggered_posts', level=level, obstacles=obstacles,
                   slots=square, start=(1.8, 5.), goal=(12.2, 5.))
        obstacles = [circle(7., y, (.16, .27)[level]) for y in np.arange(.6, 9.51, 1.4)]
        yield dict(family='formation_threading', level=level, obstacles=obstacles,
                   slots=square*2.5, start=(2., 5.), goal=(12., 5.))


def map_data(case):
    grid = np.zeros((100, 140), dtype=bool)
    return MapData(case['family'], .1, 14., 10., (0., 0.), grid, grid, grid.astype(float),
                   case['start'], case['goal'], case['obstacles'], robot_radius=.08, safety_margin=.02)


def make_planner(case, variant, seed, iterations, launch_distance=LaunchConfig.internal_distance):
    cls = FixedPortalPlanner if variant == 'fixed_portal' else NoRewirePlanner if variant == 'no_rewire' else RegionTubeRRTPlanner
    mode = 'point' if variant == 'seed_launch' else 'opening' if variant == 'release' else 'projection'
    return cls(map_data(case), case['slots'], Pose2D(*case['start'], 0.),
               config=TubeRRTConfig(cell_model='polyhedral_tube', seed=seed, max_iterations=iterations,
                                    stop_on_first_goal=False),
               tube_config=RegionTubeConfig(expansion=mode, launch=LaunchConfig(internal_distance=launch_distance)))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args, cases):
    path = args.out_dir/'runs.csv'
    rows = list(csv.DictReader(path.open())) if args.resume and path.exists() else []
    done = {(r['family'], int(r['level']), int(r['seed']), r['variant']) for r in rows}
    for case_index, case in enumerate(cases):
        for seed in range(args.seed_start, args.seed_start+args.seeds):
            # Rotate run order independently of planner RNG to limit order/timing bias.
            shift = (seed+case_index) % len(args.variants)
            for variant in args.variants[shift:]+args.variants[:shift]:
                key = (case['family'], case['level'], seed, variant)
                if key in done:
                    continue
                planner = make_planner(case, variant, seed, args.iterations, args.launch_distance)
                started = time.perf_counter()
                result, error = None, ''
                try:
                    result = planner.plan()
                except RuntimeError as exc:
                    # Solver failures are experiment outcomes, never silently discarded or retried.
                    error = str(exc)
                elapsed = time.perf_counter()-started
                stats = result.overlap_stats if result is not None else planner._stats
                success = result is not None and result.success
                history = list(zip([float(t) for t, _ in result.cost_time_history],
                                   [float(c) for _, c in result.cost_time_history])) if result is not None else []
                row = dict(family=case['family'], level=case['level'], seed=seed, variant=variant,
                           status='solver_error' if error else 'success' if success else 'budget_exhausted',
                           success=success, first_time_s=stats.get('first_goal_time_s', math.nan),
                           first_iteration=result.first_goal_iteration if result is not None else None,
                           final_cost=result.path_cost if success else math.nan,
                           best_found_cost=min((c for _, c in history), default=math.nan),
                           plan_time_s=elapsed, nodes=len(planner._nodes),
                           attachment_time_s=stats.get('attachment_time_s', 0.),
                           attachment_queries=stats.get('attachment_queries', 0),
                           repairs=stats.get('descendant_repairs', 0), rewires=stats.get('rewires', 0),
                           attachment_violation=stats.get('attachment_violation_max', math.nan),
                           cost_error=stats.get('entry_cost_error_max', math.nan),
                           bottleneck=result.bottleneck if success else math.nan,
                           error=error)
                rows.append(row)
                write_csv(path, rows)
                record = dict(**row, history=history,
                              path=[(p.x, p.y, p.yaw) for p in result.path_poses] if success else [])
                folder = args.out_dir/'runs'/f"{case['family']}_{case['level']}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder/f'{seed}_{variant}.json').write_text(json.dumps(record))
                print(json.dumps(dict(run=len(rows), total=len(cases)*args.seeds*len(args.variants),
                                      **{k: row[k] for k in ('family','level','seed','variant','status','first_time_s','final_cost','plan_time_s')})), flush=True)


def median(values):
    values = [v for v in values if math.isfinite(v)]
    return float(np.median(values)) if values else None


def summarize(out):
    rows = list(csv.DictReader((out/'runs.csv').open()))
    groups = {}
    for row in rows:
        groups.setdefault((row['family'], row['variant']), []).append(row)
    summaries = []
    for (family, variant), group in groups.items():
        solved = [r for r in group if r['status']=='success']
        summaries.append(dict(family=family, variant=variant, runs=len(group), successes=len(solved),
                              solver_errors=sum(r['status']=='solver_error' for r in group),
                              first_median_s=median([float(r['first_time_s']) for r in solved]),
                              plan_median_s=median([float(r['plan_time_s']) for r in group]),
                              final_cost_median=median([float(r['final_cost']) for r in solved])))
    write_csv(out/'summary.csv', summaries)
    indexed = {(r['family'], r['level'], r['seed'], r['variant']): r for r in rows}
    pairs = []
    for row in rows:
        if row['variant'] != 'continuous':
            continue
        for alternative in ('seed_launch','fixed_portal','no_rewire','release'):
            other = indexed.get((row['family'],row['level'],row['seed'],alternative))
            if other is None:
                continue
            both = row['status']==other['status']=='success'
            common_time = min(float(row['plan_time_s']), float(other['plan_time_s']))
            def cost_at(r):
                folder = out/'runs'/f"{r['family']}_{r['level']}"
                h = json.loads((folder/f"{r['seed']}_{r['variant']}.json").read_text())['history']
                earlier = [c for t,c in h if t <= common_time]
                # Best observed cost and the actual current-tree cost are separate metrics.
                return (min(earlier), earlier[-1]) if earlier else (math.inf, math.inf)
            (core_cost, core_current), (other_cost, other_current) = cost_at(row), cost_at(other)
            pairs.append(dict(family=row['family'],level=row['level'],seed=row['seed'],alternative=alternative,
                              core_status=row['status'],other_status=other['status'],both_success=both,
                              core_first_faster=both and float(row['first_time_s'])<float(other['first_time_s']),
                              core_cost_change_pct=100*(float(row['final_cost'])/float(other['final_cost'])-1) if both else math.nan,
                              common_time_s=common_time,
                              core_cost_at_common_time=core_cost, other_cost_at_common_time=other_cost,
                              core_wallclock_win=core_cost < other_cost-1e-6,
                              other_wallclock_win=other_cost < core_cost-1e-6,
                              core_current_cost_at_common_time=core_current,
                              other_current_cost_at_common_time=other_current,
                              core_current_wallclock_win=core_current < other_current-1e-6,
                              other_current_wallclock_win=other_current < core_current-1e-6))
    write_csv(out/'paired.csv', pairs)
    return rows, summaries, pairs


def plot(out, cases, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    representative_seed = json.loads((out/'experiment.json').read_text()).get('seed_start', 0)
    family_names = list(dict.fromkeys(c['family'] for c in cases))
    fig, axes = plt.subplots(len(family_names), 2, figsize=(14, 4.3*len(family_names)), constrained_layout=True, squeeze=False)
    for case in cases:
        ax = axes[family_names.index(case['family']),case['level']]
        for obs in case['obstacles']:
            ax.add_patch(Circle(obs['center_xy'], obs['radius'], color='.35'))
        for variant, color in zip(VARIANTS, ('C0','C1','C2','C3','C4')):
            p = out/'runs'/f"{case['family']}_{case['level']}"/f'{representative_seed}_{variant}.json'
            if p.exists():
                record = json.loads(p.read_text()); path = record['path']
                if path:
                    points = np.array(path)
                    ax.plot(points[:,0],points[:,1],label=variant,color=color,linewidth=1.2,alpha=.8)
        for point, color in ((case['start'],'green'),(case['goal'],'red')):
            ax.scatter(*point,color=color,s=25,zorder=5)
        ax.set(xlim=(0,14),ylim=(0,10),title=f"{case['family']} / level {case['level']}",aspect='equal')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7,loc='lower right')
        else:
            ax.text(.5,.03,f'seed {representative_seed}: no returned solution',transform=ax.transAxes,ha='center',fontsize=8)
        ax.grid(alpha=.15)
    fig.savefig(out/f'scenes_and_seed{representative_seed}_paths.svg'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
    x=np.arange(len(family_names)); width=.15
    for j, variant in enumerate(VARIANTS):
        records = {r['family']:r for r in summaries if r['variant']==variant}
        if not records:
            continue
        axes[0].bar(x+(j-2)*width,[records[n]['successes']/records[n]['runs'] for n in family_names],width,label=variant)
        axes[1].bar(x+(j-2)*width,[records[n]['first_median_s'] or np.nan for n in family_names],width,label=variant)
    for ax in axes:
        ax.set_xticks(x, family_names, rotation=30, ha='right'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='success fraction',ylim=(0,1.05)); axes[1].set(ylabel='first solution median [s]',yscale='log')
    fig.savefig(out/'family_comparison.svg'); plt.close(fig)


def diagnose_budget(out):
    """Same-scene, same-RNG A/B: inspect seed, entry and actual admissible launch."""
    case = next(c for c in families() if c['family']=='open_control' and c['level']==0)
    rows, nodes = [], []
    for budget in (1., 1.5):
        planner = make_planner(case, 'continuous', 0, 300, budget)
        result = planner.plan()
        target = Pose2D(*case['goal'], 0.)
        distances = planner._distances_to(target)
        if planner._goal_nodes:
            distances[list(planner._goal_nodes)] = math.inf
        nearest = int(np.argmin(distances))
        node = planner._nodes[nearest]
        launch = planner.launch_search.projection(node.cell, target, node.entry_pose)
        direction = planner.launch_search.offset(node.pose, target)
        motion = planner.launch_search.offset(node.pose, launch)
        signed_launch = float(motion@direction/np.linalg.norm(direction)) if np.linalg.norm(direction)>0. else 0.
        rows.append(dict(launch_distance=budget,success=result.success,nodes=len(result.tree_nodes),
                         nearest_seed=(node.pose.x,node.pose.y,node.pose.yaw),
                         nearest_entry=(node.entry_pose.x,node.entry_pose.y,node.entry_pose.yaw),
                         launch=(launch.x,launch.y,launch.yaw),
                         launch_progress_from_seed=signed_launch,
                         seed_goal_distance=planner.metric(node.pose,target),
                         entry_goal_distance=planner.metric(node.entry_pose,target),
                         launch_goal_distance=planner.metric(launch,target)))
        for i,n in enumerate(result.tree_nodes):
            nodes.append(dict(launch_distance=budget,node=i,parent=n.parent,cost=n.cost,
                              seed_x=n.pose.x,seed_y=n.pose.y,seed_yaw=n.pose.yaw,
                              entry_x=n.entry_pose.x,entry_y=n.entry_pose.y,entry_yaw=n.entry_pose.yaw))
    (out/'budget_diagnostic.json').write_text(json.dumps(rows,indent=2))
    write_csv(out/'budget_diagnostic_nodes.csv', nodes)
    print(json.dumps(rows,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,default=6)
    parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=300)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--diagnose-budget',action='store_true')
    parser.add_argument('--families',nargs='+')
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--launch-distance',type=float,default=LaunchConfig.internal_distance)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results'/'continuous_families')
    args=parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.diagnose_budget:
        diagnose_budget(args.out_dir)
        return
    cases=[c for c in families() if args.families is None or c['family'] in args.families]
    # Verify scenario endpoints before freezing the suite, independent of performance results.
    for case in cases:
        p=make_planner(case,'continuous',0,args.iterations)
        if not p.cells.make_cell(p.start).valid or p.clearances([case['goal'][0]],[case['goal'][1]],[0.])[0] <= 0.:
            raise ValueError(f"Invalid scenario endpoints: {case['family']} {case['level']}")
    manifest=dict(seeds=args.seeds,seed_start=args.seed_start,iterations=args.iterations,variants=args.variants,
                  launch=asdict(LaunchConfig(internal_distance=args.launch_distance)),
                  planner_defaults=asdict(TubeRRTConfig(cell_model='polyhedral_tube', max_iterations=args.iterations, stop_on_first_goal=False)),
                  cell_defaults=asdict(RegionTubeConfig(launch=LaunchConfig(internal_distance=args.launch_distance))),
                  cases=[dict(c,slots=c['slots'].tolist()) for c in cases])
    (args.out_dir/'experiment.json').write_text(json.dumps(manifest,indent=2))
    run(args,cases)
    _,summary,_=summarize(args.out_dir); plot(args.out_dir,cases,summary)


if __name__=='__main__':main()
