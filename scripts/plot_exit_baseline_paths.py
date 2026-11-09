#!/usr/bin/env python3
"""Render recorded raw routes, with no planning, shortcutting, or optimization."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/formation-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_coupling import scenario, ROBOT_RADIUS

SCENES = ('forest', 'dense', 'slots')
METHODS = ('A', 'exit_rewire', 'exit_local', 'rrtstar', 'bitstar')
MATCHED = ('A', 'exit_local', 'rrtstar', 'bitstar')
COLORS = dict(A='#c63b40', exit_rewire='#2b7cba', exit_local='#258b48',
              rrtstar='#7e4bb0', bitstar='#df8619')
LABELS = dict(A='A', exit_rewire='Exit-compensated A', exit_local='Exit + online entry A',
              rrtstar='RRT*', bitstar='BIT*')
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'svg.fonttype': 'none'})


def route_data(route, rho):
    poses = np.asarray(route, dtype=float).reshape(-1, 3)
    if not len(poses):
        return dict(poses=poses, progress=np.array([]), yaw_deg=np.array([]), xy=0., yaw=0., dw=0.)
    delta = np.diff(poses, axis=0)
    dyaw = (delta[:, 2] + math.pi) % (2 * math.pi) - math.pi
    xy = np.hypot(delta[:, 0], delta[:, 1])
    length = np.r_[0., np.cumsum(np.hypot(xy, rho * dyaw))]
    yaw = poses[0, 2] + np.r_[0., np.cumsum(dyaw)]
    return dict(poses=poses, progress=length / length[-1] if length[-1] else length,
                yaw_deg=np.degrees(yaw), xy=float(xy.sum()), yaw=float(np.abs(dyaw).sum()), dw=float(length[-1]))


def success(rec):
    return rec is not None and bool(rec['raw_success']) and bool(rec['tree_route']) and math.isfinite(rec['raw_cost'])


def chosen(run):
    good = [r for r in run['records'] if success(r)]
    return dict(first=next((r for r in good if r['kind'] == 'first'), good[0] if good else None),
                final=run['records'][-1], best=min(good, key=lambda r: r['raw_cost']) if good else None)


def at_budget(run, budget):
    # A later solution is never borrowed, even if the preceding snapshot has no path.
    available = [r for r in run['records'] if r['online_time_s'] <= budget]
    return available[-1] if available else None


def backdrop(ax, map_data):
    for obstacle in map_data.obstacle_primitives:
        ax.add_patch(Circle(obstacle['center_xy'], obstacle['radius'], facecolor='#a5aab0', edgecolor='#737982', lw=.3, zorder=1))
    x, y = map_data.origin_xy
    ax.set(xlim=(x, x + map_data.width_m), ylim=(y, y + map_data.height_m), aspect='equal', xlabel='x [m]', ylabel='y [m]')
    ax.plot(*map_data.start_xy, marker='o', mfc='white', mec='#222', ms=5, zorder=9)
    ax.plot(*map_data.goal_xy, marker='*', color='#222', ms=8, zorder=9)
    ax.grid(alpha=.13, lw=.5)


def draw_route(ax, rec, color, **kwargs):
    if success(rec):
        a = np.asarray(rec['tree_route'])
        ax.plot(a[:, 0], a[:, 1], color=color, **kwargs)


def robots(ax, rec, slots, color, n=7, alpha=.22):
    if not success(rec):
        return
    rho = float(np.linalg.norm(slots, axis=1).max())
    data = route_data(rec['tree_route'], rho)
    for t in np.linspace(.04, .96, n):
        x, y = [float(np.interp(t, data['progress'], data['poses'][:, k])) for k in (0, 1)]
        yaw = math.radians(float(np.interp(t, data['progress'], data['yaw_deg'])))
        c, s = math.cos(yaw), math.sin(yaw)
        for point in np.array((x, y)) + slots @ np.array(((c, s), (-s, c))):
            ax.add_patch(Circle(point, ROBOT_RADIUS, facecolor=color, edgecolor=color, alpha=alpha, lw=.45, zorder=2))
        ax.arrow(x, y, .32*c, .32*s, color=color, alpha=min(1., alpha*2), width=.006,
                 head_width=.10, length_includes_head=True, zorder=4)


def raw_title(rec, rho):
    if not success(rec):
        return 'No path' + (f" at t={rec['online_time_s']:.3f}s" if rec else ' before budget')
    d = route_data(rec['tree_route'], rho)
    return f"Raw J={rec['raw_cost']:.3f}; Lxy={d['xy']:.3f} m\nt={rec['online_time_s']:.3f}s; {rec['kind']}; proposal {rec['iteration']}"


def map_panel(ax, method, rec, map_data, slots, stages=None):
    backdrop(ax, map_data)
    rho = float(np.linalg.norm(slots, axis=1).max())
    color = COLORS[method]
    if stages:
        draw_route(ax, stages['first'], '#687481', ls=':', lw=1.2, alpha=.85, zorder=3)
        draw_route(ax, stages['best'], color, ls='--', lw=1.0, alpha=.7, zorder=4)
    robots(ax, rec, slots, color)
    draw_route(ax, rec, color, lw=1.9, zorder=5)
    if not success(rec):
        ax.text(.5, .5, 'No path', transform=ax.transAxes, ha='center', color=color,
                bbox=dict(fc='white', ec='none', alpha=.85), fontsize=13)
    ax.set_title(f'{LABELS[method]}\n{raw_title(rec, rho)}', color=color, fontsize=9)


def common_budget(lookup, name, seed):
    return min(lookup[name, method, seed]['online_time_s'] for method in MATCHED)


def save(fig, out, stem, svg=False, dpi=155):
    fig.savefig(out / f'{stem}.png', dpi=dpi)
    if svg:
        fig.savefig(out / f'{stem}.svg')
    plt.close(fig)


def overview(lookup, maps, out, matched=False):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    for ax, name in zip(axes, SCENES):
        map_data, slots = maps[name]
        backdrop(ax, map_data)
        budget = common_budget(lookup, name, 0)
        captions = []
        rho = float(np.linalg.norm(slots, axis=1).max())
        for method in MATCHED:
            run = lookup[name, method, 0]
            rec = at_budget(run, budget) if matched else run['records'][-1]
            draw_route(ax, rec, COLORS[method], lw=1.65, zorder=5)
            robots(ax, rec, slots, COLORS[method], n=4, alpha=.10)
            d = route_data(rec['tree_route'], rho) if success(rec) else None
            captions.append(f"{LABELS[method]}: J={rec['raw_cost']:.2f}, Lxy={d['xy']:.2f}" if d else f'{LABELS[method]}: No path')
        ax.set_title(name + (f' | common budget {budget:.3f}s' if matched else ' | final online raw'))
        ax.text(.01, -.17, '\n'.join(captions), transform=ax.transAxes, va='top', fontsize=8.5)
    title = 'Common online budget: latest recorded snapshot at or before budget' if matched else 'Final online raw paths: 6000 proposals + completed BIT* batch tail'
    fig.suptitle(f'Fixed seed 0 | {title}', fontsize=13)
    handles = [Line2D([], [], color=COLORS[m], lw=2, label=LABELS[m]) for m in MATCHED]
    handles += [Patch(facecolor='#a5aab0', label='Physical obstacles'), Line2D([], [], marker='o', mfc='white', mec='#222', ls='', label='Start'),
                Line2D([], [], marker='*', color='#222', ls='', label='Goal')]
    fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False, bbox_to_anchor=(.5, .015))
    fig.text(.5, .083, 'J includes yaw scaled by formation radius; Lxy is centroid XY length. Faint discs/arrows show sparse robot bodies/orientation. No post-processing.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.04, right=.99, top=.87, bottom=.30, wspace=.19)
    save(fig, out, 'seed0_matched_overlay' if matched else 'seed0_final_overlay', svg=True, dpi=190)


def detail(lookup, maps, out, name, seed, matched=False):
    map_data, slots = maps[name]
    rho = float(np.linalg.norm(slots, axis=1).max())
    budget = common_budget(lookup, name, seed)
    fig = plt.figure(figsize=(23, 10.2 if not matched else 8.2))
    grid = fig.add_gridspec(3 if not matched else 2, 5, height_ratios=(3.3, 1.3, 1.4) if not matched else (3.3, 1.3), hspace=.43, wspace=.25)
    yaws, costs = [], []
    for col, method in enumerate(METHODS):
        run = lookup[name, method, seed]
        stages = chosen(run)
        rec = at_budget(run, budget) if matched else stages['final']
        ax = fig.add_subplot(grid[0, col])
        map_panel(ax, method, rec, map_data, slots, None if matched else stages)
        ay = fig.add_subplot(grid[1, col]); yaws.append(ay)
        routes = [(rec, COLORS[method], '-', 1.8)] if matched else [(stages['first'], '#687481', ':', 1.1), (stages['best'], COLORS[method], '--', 1.0), (rec, COLORS[method], '-', 1.8)]
        for route_rec, color, style, width in routes:
            if success(route_rec):
                d = route_data(route_rec['tree_route'], rho)
                ay.plot(d['progress'], d['yaw_deg'], color=color, ls=style, lw=width)
        ay.set(xlim=(0, 1), xlabel='Own normalized d_W length', ylabel='Unwrapped yaw [deg]'); ay.grid(alpha=.2)
        if not matched:
            ac = fig.add_subplot(grid[2, col]); costs.append(ac)
            records = run['records']
            t = [r['online_time_s'] for r in records]
            raw = np.array([r['raw_cost'] if success(r) else np.nan for r in records])
            best = np.minimum.accumulate(np.where(np.isfinite(raw), raw, np.inf))
            best[~np.isfinite(best)] = np.nan
            ac.step(t, raw, where='post', color=COLORS[method], lw=1.3)
            ac.step(t, best, where='post', color=COLORS[method], lw=1.0, ls='--')
            ac.axvline(budget, color='#777', lw=.7, ls=':')
            ac.set(xlabel='Online time [s]', ylabel='Raw J [d_W m]'); ac.grid(alpha=.2)
    for group in (yaws, costs):
        if group:
            lo = min(a.get_ylim()[0] for a in group); hi = max(a.get_ylim()[1] for a in group)
            for a in group: a.set_ylim(lo, hi)
    if costs:
        tmax = max(lookup[name, m, seed]['records'][-1]['online_time_s'] for m in METHODS)
        for a in costs: a.set_xlim(0, tmax)
    selection = f'common budget {budget:.6f}s; latest snapshot <= budget' if matched else 'final raw + first / best observed raw stages'
    fig.suptitle(f'{name} | seed {seed} | {selection}', fontsize=14, y=.985)
    handles = [Line2D([], [], color='#555', lw=1.8, label='Selected raw path'), Patch(facecolor='#a5aab0', label='Physical obstacle'),
               Patch(facecolor='#71a887', alpha=.3, label='Robot bodies at sparse selected poses')]
    if not matched:
        handles += [Line2D([], [], color='#687481', ls=':', label='First successful raw path'), Line2D([], [], color='#555', ls='--', label='Best observed raw path')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, bbox_to_anchor=(.5, .039), fontsize=9)
    foot = 'All routes are stored tree_route poses (x, y, yaw); no chain, shortcut or newly optimized route. J includes yaw; all five maps share physical coordinates.'
    if matched:
        foot += '\nBudget = min final online time of A / Exit + online entry A / RRT* / BIT*. Missing snapshots or unsuccessful snapshots remain No path.'
    else:
        foot += '\nFinal BIT* includes its completed last-batch queue tail after proposal 6000; online cost curves charge that work. Dotted vertical line is common online budget.'
    fig.text(.5, .015, foot, ha='center', fontsize=9)
    fig.subplots_adjust(top=.91, bottom=.14, left=.04, right=.99)
    stem = f'{name}_seed{seed:02d}' + ('_matched' if matched else '')
    save(fig, out, stem, svg=seed == 0)


def metrics_and_metadata(lookup, maps, out, args):
    rows, budgets = [], []
    errors = []
    for name in SCENES:
        rho = float(np.linalg.norm(maps[name][1], axis=1).max())
        for seed in range(10):
            budget = common_budget(lookup, name, seed)
            budgets.append(dict(scenario=name, seed=seed, common_online_budget_s=budget))
            for method in METHODS:
                run = lookup[name, method, seed]
                stages = chosen(run)
                stages['matched'] = at_budget(run, budget)
                # Validate every stored successful route, not just selected routes.
                for rec in run['records']:
                    if success(rec):
                        errors.append(abs(route_data(rec['tree_route'], rho)['dw'] - rec['raw_cost']))
                for selection, rec in stages.items():
                    d = route_data(rec['tree_route'], rho) if success(rec) else None
                    rows.append(dict(scenario=name, seed=seed, variant=method, selection=selection,
                        success=success(rec), raw_cost=rec['raw_cost'] if success(rec) else '',
                        raw_xy_length_m=d['xy'] if d else '', raw_abs_yaw_variation_rad=d['yaw'] if d else '',
                        raw_recomputed_dw=d['dw'] if d else '', rho_m=rho,
                        online_time_s=rec['online_time_s'] if rec else '', iteration=rec['iteration'] if rec else '',
                        record_kind=rec['kind'] if rec else '', record_index=next((i for i,r in enumerate(run['records']) if r is rec), ''),
                        matched_online_budget_s=budget,
                        source=str(args.region_runs if method in METHODS[:3] else args.baseline_runs)))
    assert len(rows) == 600
    max_error = max(errors, default=0.)
    assert max_error < 1e-5, f'Recorded raw J differs from tree_route d_W: {max_error}'
    for filename, data in (('path_metrics.csv', rows), ('matched_budgets.csv', budgets)):
        with (out / filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    metadata = dict(region_source=str(args.region_runs), baseline_source=str(args.baseline_runs), raw_run_count=len(lookup),
        case_count=30, methods=list(METHODS), fixed_overview_seed=0, metric_rows=len(rows),
        selection_rules=dict(final='records[-1], including BIT* last batch tail', first='first successful stored raw record',
                             best='minimum recorded raw_cost among successful raw records',
                             matched='last stored snapshot with online_time_s <= min final online time of A, exit_local, rrtstar, bitstar; no future borrowing'),
        budgets=budgets, successful_routes_validated=len(errors), max_recorded_vs_route_dw_error=max_error,
        metric_definition='sum sqrt(dx^2+dy^2+(rho*wrap_to_pi(dyaw))^2); rho=max norm(slots); Lxy=sum hypot(dx,dy)',
        provenance='Read existing tree_route only; scenario() reconstructs static physical map. No planning, optimization or shortcutting.',
        robot_radius_m=ROBOT_RADIUS, omissions='None: forest/dense/slots x seeds0-9 x all five methods.',
        bit_budget='6000 original proposals; final includes completed last-batch queue tail and charges its time. This is not 6000 queue operations.')
    (out / 'selection.json').write_text(json.dumps(metadata, indent=2))
    return metadata, rows


def gallery(out, metadata, rows):
    table, cards = [], []
    final = {(r['scenario'], r['seed'], r['variant']): r for r in rows if r['selection'] == 'final'}
    for case in metadata['budgets']:
        name, seed = case['scenario'], case['seed']
        stem = f'{name}_seed{seed:02d}'
        vals = [final[name, seed, m] for m in METHODS]
        table.append(f'<tr><td><a href="{stem}.png">{name}, seed {seed}</a> · <a href="{stem}_matched.png">matched</a></td><td>{case["common_online_budget_s"]:.6f}</td>' + ''.join(f'<td>{r["raw_cost"]:.4f}</td>' if r['success'] else '<td>No path</td>' for r in vals) + '</tr>')
        cards.append(f'<details><summary>{name}, seed {seed}: final / first / best</summary><a href="{stem}.png"><img loading="lazy" src="{stem}.png" alt="{name} seed {seed} all five raw methods"></a><p><a href="{stem}_matched.png">Common online budget paths</a></p></details>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stored raw paths: A and sampling baselines</title><style>body{font:15px system-ui,sans-serif;max-width:1700px;margin:25px auto;padding:0 16px;color:#222}img{max-width:100%;height:auto}a{color:#125c97}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{border:1px solid #ddd;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}p{line-height:1.5}summary{cursor:pointer;padding:12px;background:#eee;margin-top:12px;font-size:18px}.note{background:#f2f5f7;padding:13px}</style><h1>Actual stored raw paths: A, exit compensation, RRT* and BIT*</h1><p class="note">All 30 scene/seed cases and all five methods are retained. The overview seed is fixed at 0, not chosen by outcome. Every colored route comes directly from a recorded tree_route (x, y, yaw). No planner or optimizer runs in this plotting script. No A chain or baseline shortcut is drawn. Gray circles are physical obstacles; faint discs are individual robot bodies at sparse poses, with arrows for formation orientation.</p><p>Raw J is d_W arc length, including shortest-angle yaw change scaled by formation radius rho. Lxy measures only centroid XY distance; these metrics differ. Detail figures distinguish first raw (gray dotted), final raw (method-colored solid), and best observed raw (method-colored dashed). Yaw is continuously unwrapped along each route's own normalized d_W arc length. A similar XY shape is not a homotopy classification.</p><p>Final means the last recorded online state. BIT* exhausts 6000 original proposals and then completes the last batch queue tail; that work is included in final online time. This is not a budget of 6000 queue operations. Different methods' final online times differ. Common-time figures instead use each case's minimum final online time over A / Exit + online entry A / RRT* / BIT*, then select the latest existing record at or before that budget. Missing or unsuccessful snapshots show No path; future solutions are never borrowed. Exit-compensated A is also displayed at that budget.</p><p><a href="path_metrics.csv">First/final/best/matched path metrics CSV</a> · <a href="matched_budgets.csv">All 30 common time budgets</a> · <a href="selection.json">Selection and validation metadata</a> · <a href="plot_exit_baseline_paths.py">Reproduction plotting script</a></p><h2>Fixed seed 0: final online raw paths</h2><a href="seed0_final_overlay.png"><img src="seed0_final_overlay.png" alt="Three-scene final online path overview"></a><p><a href="seed0_final_overlay.svg">Vector SVG</a></p><h2>Fixed seed 0: common online budget</h2><a href="seed0_matched_overlay.png"><img src="seed0_matched_overlay.png" alt="Three-scene matched online time overview"></a><p><a href="seed0_matched_overlay.svg">Vector SVG</a></p><h2>All cases: final raw J and exact common online budgets</h2><table><thead><tr><th>Case / path details</th><th>Budget [s]</th>'''
    page += ''.join(f'<th>{html.escape(LABELS[m])}</th>' for m in METHODS) + '</tr></thead><tbody>' + ''.join(table) + '</tbody></table><h2>Five-method map, yaw and online-cost details</h2>' + ''.join(cards) + '</html>'
    (out / 'index.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region-runs', type=Path, default=ROOT / 'results/exit_compensation/runs.jsonl')
    parser.add_argument('--baseline-runs', type=Path, default=ROOT / 'results/exit_baselines/runs.jsonl')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results/exit_baselines/paths')
    parser.add_argument('--validate-only', action='store_true', help='Check source completeness and raw J without rendering')
    args = parser.parse_args()
    runs = [json.loads(line) for source in (args.region_runs, args.baseline_runs) for line in source.read_text().splitlines() if line.strip()]
    lookup = {(r['scenario'], r['variant'], r['seed']): r for r in runs}
    expected = {(name, method, seed) for name in SCENES for method in METHODS for seed in range(10)}
    assert set(lookup) == expected and len(runs) == 150, f'Need exactly 150 raw runs; have {len(runs)}, missing={expected-set(lookup)}'
    for run in runs:
        times = [r['online_time_s'] for r in run['records']]
        assert all(a <= b for a, b in zip(times, times[1:])), (run['scenario'], run['variant'], run['seed'], 'nonmonotonic online time')
    maps = {name: scenario(name)[:2] for name in SCENES}
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    metadata, rows = metrics_and_metadata(lookup, maps, out, args)
    if not args.validate_only:
        overview(lookup, maps, out)
        overview(lookup, maps, out, matched=True)
        for name in SCENES:
            for seed in range(10):
                detail(lookup, maps, out, name, seed)
                detail(lookup, maps, out, name, seed, matched=True)
        gallery(out, metadata, rows)
        shutil.copyfile(__file__, out / Path(__file__).name)
    print(json.dumps(dict(out=str(out), cases=30, runs=len(runs), validated_successful_routes=metadata['successful_routes_validated'], max_dw_error=metadata['max_recorded_vs_route_dw_error'], rendered=not args.validate_only)))


if __name__ == '__main__':
    main()
