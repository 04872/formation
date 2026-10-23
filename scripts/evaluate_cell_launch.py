#!/usr/bin/env python3
"""Measure current-cell release/cache costs, then run matched planner ablations.

../env-rebuilt/bin/python scripts/evaluate_cell_launch.py --planner --seeds 3 --iterations 800
"""
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
from formation import (CellLaunchSearch, LaunchConfig, MapData, PolyhedralCellModel, Pose2D,
                       RegionTubeConfig, TubeRRTConfig, make_tube_rrt_planner)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predictor_scenes():
    def circle(x, y, r=.35):
        return {'type': 'circle', 'center_xy': (float(x), float(y)), 'radius': r}
    xs = np.arange(1.0, 6.01, .35)
    bottom = [circle(x, 3.15) for x in xs]
    top = [circle(x, 4.85) for x in xs]
    corridor = bottom + top
    slots = np.array([[-.18, -.18], [-.18, .18], [.18, -.18], [.18, .18]])
    yield 'corridor_middle', corridor, slots, (2.5, 4., 0.), (.9, 0., 0.)
    yield 'corridor_exit', corridor, slots, (5.7, 4., 0.), (1.1, 0., 0.)
    cap = [circle(6.35, y) for y in np.arange(3.15, 4.86, .3)]
    yield 'dead_end', corridor + cap, slots, (5.25, 4., 0.), (.45, 0., 0.)
    side = bottom + [circle(x, 4.85) for x in xs if x < 3.0 or x > 4.6]
    yield 'side_opening', side, slots, (3.8, 4.15, 0.), (0., 1.1, 0.)
    yield 'obstacle_corner', [circle(5., 4., .65)], slots, (4.2, 5.05, 0.), (1.8, 0., 0.)
    wide = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    yield 'threading_gate', [circle(5., 4., .18)], wide, (4.2, 4., 0.), (.9, 0., 0.)
    yield 'threading_exit', [circle(5., 4., .18)], wide, (5.1, 4., 0.), (1.2, 0., 0.)
    yield 'open_control', [], slots, (4., 4., 0.), (1., 0., 0.)


def evaluate_release(out, launch):
    rows = []
    grid = np.zeros((80, 120), dtype=bool)
    for name, obstacles, slots, start, span in predictor_scenes():
        m = MapData(name, .1, 12., 8., (0., 0.), grid, grid, grid.astype(float),
                    (1., 4.), (10., 4.), obstacles, robot_radius=.08, safety_margin=.02)
        cells = PolyhedralCellModel(slots, m, .10, lazy_geometry=True)
        search = CellLaunchSearch(cells, launch)
        pose = Pose2D(*(np.array(start)+.5*np.array(span)))
        cell = cells.make_cell(pose)
        entry = search.pose(pose, np.array([0., .2*cells.ray_extent(cell, (0., 1.), 0.), 0.]))
        target = Pose2D(*(np.array(start)+2.*np.array(span)))
        before = cells.stats['cells_built']
        search.select(cell, target, 'opening', entry)
        cold = dict(search.last_solve)
        times = []
        for _ in range(20):
            search.select(cell, target, 'opening', entry)
            times.append(search.last_solve['solve_time_s'])
        hot = search.last_solve
        rows.append(dict(scene=name, facets=2*len(cell.offsets),
                         active_facets=int(np.count_nonzero(cell._release_geometry[1])),
                         cold_time_s=cold['solve_time_s'], cached_time_s=float(np.median(times)),
                         cold_support_queries=cold['support_queries'],
                         cached_support_queries=hot['support_queries'],
                         temporary_cells=cells.stats['cells_built']-before,
                         opening_tendency=hot['opening_tendency'],
                         maximum_release_utility=hot['maximum_release_utility'],
                         projection_fallback=hot['projection_fallback'],
                         selected_facet=hot['selected_facet'],
                         constraint_violation=hot['constraint_violation']))
        print(json.dumps(rows[-1]), flush=True)
    write_csv(out / 'release.csv', rows)


def evaluate_planner(out, launch, seeds, iterations, anytime, resume=False):
    rows = list(csv.DictReader((out / 'planner.csv').open())) if resume else []
    histories = list(csv.DictReader((out / 'cost_history.csv').open())) if resume else []
    expansions = {}
    completed = {(r['map'], int(r['seed']), r['mode']) for r in rows}
    grid = np.zeros((80, 120), dtype=bool)
    cases = {
        'corridor_exit': ((2.5, 4.), (9., 4.), lambda q: q.x >= 6.75),
        'side_opening': ((2., 4.), (3.8, 6.5), lambda q: q.y >= 5.35),
        'threading_exit': ((3., 4.), (8., 4.), lambda q: q.x >= 6.),
    }
    for name, obstacles, slots, _, _ in predictor_scenes():
        if name not in cases:
            continue
        start, goal, in_chamber = cases[name]
        m = MapData(name, .1, 12., 8., (0., 0.), grid, grid, grid.astype(float), start, goal,
                    obstacles, robot_radius=.08, safety_margin=.02)
        for seed in range(seeds):
            for mode in ('point', 'projection', 'opening'):
                if (name, seed, mode) in completed:
                    continue
                p = make_tube_rrt_planner(m, slots, Pose2D(*m.start_xy, 0.),
                    config=TubeRRTConfig(cell_model='polyhedral_tube', seed=seed, max_iterations=iterations,
                                         stop_on_first_goal=not anytime),
                    tube_config=RegionTubeConfig(expansion=mode, launch=launch))
                # Observe actual branch waypoints entering a geometrically defined
                # chamber beyond the corridor opening/gate, not a percentage plane.
                store, milestone = p._store, {}
                def timed_store(node):
                    index = store(node)
                    waypoints = [node.entry_pose]
                    if node.certificate is not None and node.certificate.launch is not None:
                        waypoints.append(node.certificate.launch)
                    if not milestone and any(in_chamber(q) for q in waypoints):
                        milestone.update(nodes=index + 1, time_s=time.perf_counter() - started)
                    return index
                p._store = timed_store
                started = time.perf_counter()
                r = p.plan()
                elapsed = time.perf_counter() - started
                stats = r.overlap_stats
                row = dict(map=name, seed=seed, mode=mode, success=r.success,
                           first_iteration=r.first_goal_iteration, first_time_s=stats['first_goal_time_s'],
                           first_cost=stats['first_goal_cost'], final_cost=r.path_cost if r.success else math.nan,
                           plan_time_s=elapsed, nodes=len(r.tree_nodes), pose_queries=stats['pose_queries'],
                           opening_cells=stats['opening_cells'],
                           attachment_queries=stats['attachment_queries'],
                           attachment_time_s=stats['attachment_time_s'],
                           attachment_violation_max=stats['attachment_violation_max'],
                           entry_cost_error_max=stats['entry_cost_error_max'],
                           descendant_repairs=stats['descendant_repairs'],
                           optimization_failed=stats['rejected_optimization_failed'], opening_selected=stats['opening_selected'],
                           bottleneck=r.bottleneck,
                           launch_solve_mean_s=float(np.mean([e.get('solve_time_s', 0.) for e in p.expansion_log]))
                           if p.expansion_log else 0.,
                           launch_violation_max=max((e.get('constraint_violation', 0.) for e in p.expansion_log), default=0.), chamber_nodes=milestone.get('nodes'),
                           chamber_time_s=milestone.get('time_s'))
                rows.append(row)
                for (iteration, cost), (wall_time, _) in zip(r.cost_history, r.cost_time_history):
                    histories.append(dict(map=name, seed=seed, mode=mode, iteration=iteration,
                        time_s=wall_time, cost=cost))
                expansions[f'{name}/{seed}/{mode}'] = p.expansion_log
                print(json.dumps(row), flush=True)
                write_csv(out / 'planner.csv', rows)
                (out / 'expansions.json').write_text(json.dumps(expansions))
                if histories:
                    write_csv(out / 'cost_history.csv', histories)
    write_csv(out / 'planner.csv', rows)
    if histories:
        write_csv(out / 'cost_history.csv', histories)
    (out / 'expansions.json').write_text(json.dumps(expansions))


def summarize_planner(out):
    rows = list(csv.DictReader((out / 'planner.csv').open()))
    histories = list(csv.DictReader((out / 'cost_history.csv').open())) if (out / 'cost_history.csv').exists() else []
    grouped = {(r['map'], r['seed'], r['mode']): r for r in rows}
    comparisons = []
    for r in rows:
        if r['mode'] != 'opening':
            continue
        projected = grouped[(r['map'], r['seed'], 'projection')]
        opening_history = [h for h in histories if (h['map'], h['seed'], h['mode']) == (r['map'], r['seed'], 'opening')]
        projection_history = [h for h in histories if (h['map'], h['seed'], h['mode']) == (r['map'], r['seed'], 'projection')]
        better_cost = False
        for point in opening_history:
            earlier = [float(h['cost']) for h in projection_history if float(h['time_s']) <= float(point['time_s'])]
            if earlier and float(point['time_s']) <= float(projected['plan_time_s']) and float(point['cost']) < min(earlier) - 1e-6:
                better_cost = True
        faster_first = (r['success'] == 'True' and (projected['success'] != 'True' or
                         float(r['first_time_s']) < float(projected['first_time_s'])))
        fewer_chamber_nodes = bool(r['chamber_nodes']) and (not projected['chamber_nodes'] or
                              int(r['chamber_nodes']) < int(projected['chamber_nodes']))
        faster_chamber = bool(r['chamber_time_s']) and (not projected['chamber_time_s'] or
                         float(r['chamber_time_s']) < float(projected['chamber_time_s']))
        comparisons.append(dict(map=r['map'], seed=int(r['seed']), faster_first=faster_first,
                                fewer_chamber_nodes=fewer_chamber_nodes, faster_chamber=faster_chamber,
                                better_cost_at_matched_wall_clock=better_cost))
    summary = {}
    for mode in ('point', 'projection', 'opening'):
        group = [r for r in rows if r['mode'] == mode]
        success = [r for r in group if r['success'] == 'True']
        summary[mode] = dict(runs=len(group), successes=len(success),
                            median_plan_time_s=float(np.median([float(r['plan_time_s']) for r in group])),
                            median_first_time_s=float(np.median([float(r['first_time_s']) for r in success])) if success else None,
                            median_chamber_nodes=float(np.median([int(r['chamber_nodes']) for r in group if r['chamber_nodes']]))
                            if any(r['chamber_nodes'] for r in group) else None)
    summary['paired_comparisons'] = comparisons
    summary['opening_retention_supported_in_this_experiment'] = any(any(r[k] for k in (
        'faster_first', 'fewer_chamber_nodes', 'faster_chamber', 'better_cost_at_matched_wall_clock')) for r in comparisons)
    (out / 'planner_summary.json').write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r['map'] for r in rows))
    fig, axes = plt.subplots(1, len(names), figsize=(15, 4), constrained_layout=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for mode, color in zip(('point', 'projection', 'opening'), ('C0', 'C1', 'C2')):
            labelled = False
            for seed in sorted({r['seed'] for r in rows}):
                h = [x for x in histories if (x['map'], x['seed'], x['mode']) == (name, seed, mode)]
                if h:
                    ax.step([float(x['time_s']) for x in h], [float(x['cost']) for x in h], where='post',
                            color=color, alpha=.6, label=mode if not labelled else None)
                    labelled = True
        ax.set(xlabel='wall-clock [s]', ylabel='best cost', title=name, xscale='log')
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out / 'cost_wall_clock.svg')
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results' / 'cell_launch_continuous')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--planner', action='store_true', help='run matched planner comparison after release diagnostics')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=800)
    parser.add_argument('--anytime', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    launch = LaunchConfig()
    (args.out_dir / 'launch_config.json').write_text(json.dumps(asdict(launch), indent=2))
    evaluate_release(args.out_dir, launch)
    if args.planner:
        evaluate_planner(args.out_dir, launch, args.seeds, args.iterations, args.anytime, args.resume)
        summarize_planner(args.out_dir)


if __name__ == '__main__':
    main()
