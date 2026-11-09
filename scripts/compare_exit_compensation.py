#!/usr/bin/env python3
"""Serial A / ExitCompensated exit-rewire / ExitCompensated local comparison, 6000 raw samples."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/formation-matplotlib')
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_coupling import DESCRIPTIONS, route_clearance, scenario
from compare_single_tree import initialize_a, step_a
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar

VARIANTS = ('A', 'exit_rewire', 'exit_local')
MILESTONES = (2000, 4000, 6000)


def finite(x):
    return x is not None and math.isfinite(x)


def med(values):
    values = [v for v in values if finite(v)]
    return float(np.median(values)) if values else None


def change(a, b):
    return 100 * (b/a - 1) if finite(a) and finite(b) and a else None


def improvement(a, b):
    value = change(a, b)
    return -value if value is not None else None


def run_one(name, variant, seed, samples=6000):
    if variant == 'A':
        cls, config = SingleEntryRRTStar, CouplingConfig(seed=seed, step=.3, max_iterations=samples)
    else:
        from formation.exit_compensated_rrtstar import ExitCompensatedConfig, ExitCompensatedRRTStar
        cls, config = ExitCompensatedRRTStar, ExitCompensatedConfig(seed=seed, step=.3, max_iterations=samples,
                                              enable_local=variant == 'exit_local')
    map_data, slots, start_pose = scenario(name)
    wall_start = time.perf_counter()
    p = cls(map_data, slots, start_pose, config)
    (initialize_a(p) if variant == 'A' else p.initialize())
    init_s = time.perf_counter()-wall_start
    sample_calls, original_sample = 0, p._sample
    def counted_sample():
        nonlocal sample_calls
        sample_calls += 1
        return original_sample()
    p._sample = counted_sample
    records, step_s, maintenance_s, chain_s = [], 0., 0., 0.
    chain_failures, chain_solves, first, error = 0, 0, None, None

    def snapshot(kind):
        nonlocal maintenance_s, chain_s, chain_failures, chain_solves, first
        # A lazily repairs goal ancestors here. Charge that work to online time.
        # ExitCompensated promises read-only extraction: no local optimization from this call.
        began = time.perf_counter()
        sequence, raw, tree_route = [], math.inf, []
        if p.goal_cells:
            sequence, raw = p._region_chain()
            last = sequence[-1]
            endpoint = getattr(p, 'goal_endpoints', {}).get(last)
            yaw = endpoint.yaw if endpoint else p.geo.goal_yaw(p.cells[last], p.goal_xy, p.entry[last].yaw, config.overlap_margin)
            tree_route = [(p.entry[i].x, p.entry[i].y, p.entry[i].yaw) for i in sequence]
            tree_route.append((*p.goal_xy, yaw))
        elapsed = time.perf_counter()-began
        maintenance_s += elapsed
        online = init_s + step_s + maintenance_s
        raw_ok = bool(sequence) and finite(raw)
        rec = dict(kind=kind, iteration=sample_calls, online_time_s=online,
                   historical_step_time_s=step_s, maintenance_time_s=maintenance_s,
                   extraction_time_s=elapsed, raw_success=raw_ok, raw_cost=raw,
                   sequence=sequence, tree_route=tree_route,
                   regions=len(p.cells), nodes=len(p.parent), stats=dict(p.stats),
                   solver_failures=p.geo.failures-chain_failures,
                   online_solver_solves=p.geo.solves-chain_solves)
        chain_at, before_f, before_s = time.perf_counter(), p.geo.failures, p.geo.solves
        solved = p.geo.chain(p.start, p.goal_xy, [p.cells[i] for i in sequence],
                             config.overlap_margin*(1-1e-6)) if raw_ok else None
        elapsed_chain = time.perf_counter()-chain_at
        chain_s += elapsed_chain
        chain_failures += p.geo.failures-before_f
        chain_solves += p.geo.solves-before_s
        rec.update(chain_time_s=elapsed_chain, success=solved is not None,
                   optimised_cost=solved[0] if solved else math.inf,
                   route=[(q.x,q.y,q.yaw) for q in solved[1]] if solved else [],
                   wall_time_s=time.perf_counter()-wall_start)
        records.append(rec)
        if first is None and raw_ok:
            first = rec

    iteration = 0
    try:
        while sample_calls < samples:
            began = time.perf_counter()
            (step_a(p) if variant == 'A' else p.step())
            step_s += time.perf_counter()-began
            iteration += 1
            p.stats['iterations'] = iteration
            if first is None and p.goal_cells:
                snapshot('first')
            if sample_calls % 50 == 0 or sample_calls == samples:
                # At first==milestone reuse the same extraction/chain solve.
                if records and records[-1]['iteration'] == sample_calls:
                    records[-1]['is_checkpoint'] = True
                else:
                    snapshot('checkpoint')
            if sample_calls != iteration:
                raise RuntimeError(f'Expected one raw _sample call per step: {sample_calls=} {iteration=}')
    except Exception:
        error = traceback.format_exc()
    planning_wall = time.perf_counter()-wall_start
    metric_at = time.perf_counter()
    # Validate first/current/best and milestones, preserving every curve and route.
    keep = {id(first)}
    if records:
        keep.add(id(records[-1]))
    for cost, success in (('raw_cost','raw_success'), ('optimised_cost','success')):
        good = [r for r in records if r[success]]
        if good:
            keep.add(id(min(good,key=lambda r:r[cost])))
    keep.update(id(r) for r in records if r['iteration'] in MILESTONES)
    for rec in records:
        if id(rec) not in keep:
            continue
        for route_key, clear_key, valid_key in (('tree_route','raw_clearance','raw_success'),
                                                ('route','chain_clearance','success')):
            if rec.get(route_key):
                clearance = route_clearance(map_data, slots, rec[route_key])
                rec[clear_key] = clearance
                if clearance < config.safety_distance-1e-6:
                    rec[valid_key] = False
                    rec['validation_failed'] = True
    result = dict(scenario=name, variant=variant, seed=seed, config=asdict(config), records=records,
                  sample_calls=sample_calls, completed_iterations=iteration,
                  cell_poses=[(c.pose.x,c.pose.y,c.pose.yaw) for c in p.cells],
                  first_iteration=first['iteration'] if first else None,
                  first_online_time_s=first['online_time_s'] if first else None,
                  online_time_s=init_s+step_s+maintenance_s, online_without_init_s=step_s+maintenance_s, initialisation_time_s=init_s,
                  historical_step_time_s=step_s, maintenance_time_s=maintenance_s,
                  chain_time_s=chain_s, clearance_metrics_time_s=time.perf_counter()-metric_at,
                  planning_wall_s=planning_wall, actual_wall_s=time.perf_counter()-wall_start,
                  nodes=len(p.parent), regions=len(p.cells), stats=dict(p.stats),
                  solver_failures=p.geo.failures-chain_failures, chain_solver_failures=chain_failures,
                  online_solver_solves=p.geo.solves-chain_solves, chain_solver_solves=chain_solves)
    if error:
        result['error'] = error
    return result


def csv_write(out, filename, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (out/filename).open('w') as stream:
            w = csv.DictWriter(stream,fieldnames=fields)
            w.writeheader(); w.writerows(rows)


def selected(run, count=None, budget=None):
    return [r for r in run.get('records',[]) if (count is None or r['iteration']<=count)
            and (budget is None or r['online_time_s']<=budget)]


def measurements(run, recs):
    raw = [r for r in recs if r['raw_success']]
    chain = [r for r in recs if r['success']]
    last = recs[-1] if recs else {}
    best = min((r['raw_cost'] for r in raw),default=None)
    first = raw[0] if raw else {}
    previous, n_improve, n_increase = math.inf, 0, 0
    max_increase_abs, max_increase_pct = 0., 0.
    for r in raw:
        if previous < math.inf:
            n_improve += r['raw_cost'] < previous-1e-8
            n_increase += r['raw_cost'] > previous+1e-8
            max_increase_abs=max(max_increase_abs,r['raw_cost']-previous)
            max_increase_pct=max(max_increase_pct,100*(r['raw_cost']/previous-1))
        previous = r['raw_cost']
    best_so_far, best_updates = math.inf, 0
    for r in raw:
        if r['raw_cost'] < best_so_far-1e-8:
            best_updates += int(best_so_far<math.inf)
            best_so_far = r['raw_cost']
    result = dict(raw_success=bool(raw), current_raw=last.get('raw_cost') if last.get('raw_success') else None,
        best_raw=best, first_raw=first.get('raw_cost'), first_iteration=first.get('iteration'),
        first_s=first.get('online_time_s'), current_chain=last.get('optimised_cost') if last.get('success') else None,
        best_chain=min((r['optimised_cost'] for r in chain),default=None), chain_success=bool(chain),
        online_s=last.get('online_time_s'), historical_step_s=last.get('historical_step_time_s'),
        maintenance_s=last.get('maintenance_time_s'), nodes=last.get('nodes'), regions=last.get('regions'),
        solver_failures=last.get('solver_failures'), raw_improvement_pct=improvement(first.get('raw_cost'),best),
        current_raw_improvement_count=n_improve, current_raw_increase_count=n_increase,
        current_raw_max_increase_abs=max_increase_abs,current_raw_max_increase_pct=max_increase_pct,
        best_raw_improvement_count=best_updates)
    result['raw_chain_gap_pct'] = change(result['current_chain'], result['current_raw'])
    stats=last.get('stats',{})
    result.update({'stat_'+k:v for k,v in stats.items() if isinstance(v,(int,float))})
    return result


def interval_measure(run, lo, hi):
    all_recs=selected(run,count=hi)
    before=selected(run,count=lo)
    before_raw=[r for r in before if r['raw_success']]
    baseline=min((r['raw_cost'] for r in before_raw), default=None)
    best=baseline
    updates,current_improvements=0,0
    previous=before_raw[-1]['raw_cost'] if before_raw else None
    for r in all_recs:
        if not (lo<r['iteration']<=hi and r['raw_success']):
            continue
        value=r['raw_cost']
        if best is None or value<best-1e-8:
            updates+=int(best is not None);best=value
        if previous is not None and value<previous-1e-8:
            current_improvements+=1
        previous=value
    return dict(improvement_pct=improvement(baseline,best),best_improvement_count=updates,
                current_improvement_count=current_improvements,
                baseline_best_raw=baseline,final_best_raw=best)


def analyse(runs,out):
    detailed=[]
    for run in runs:
        for count in MILESTONES:
            recs=selected(run,count=count)
            row=dict(scenario=run['scenario'],variant=run['variant'],seed=run['seed'],samples=count,
                     complete=bool(recs and recs[-1]['iteration']==count and not run.get('error')),
                     error=bool(run.get('error')))
            row.update(measurements(run,recs))
            if not row['complete']:
                row['raw_success']=row['chain_success']=False
            for lo,hi in ((2000,4000),(4000,6000)):
                if count>=hi:
                    row.update({f'{lo}_{hi}_{k}':v for k,v in interval_measure(run,lo,hi).items()})
            detailed.append(row)
    summary=[]
    for name in sorted({r['scenario'] for r in runs}):
        for count in MILESTONES:
            for variant in VARIANTS:
                group=[r for r in detailed if (r['scenario'],r['samples'],r['variant'])==(name,count,variant)]
                if not group:continue
                row=dict(scenario=name,samples=count,variant=variant,runs=len(group),
                         complete=sum(r['complete'] for r in group),raw_success=sum(r['raw_success'] for r in group),
                         chain_success=sum(r['chain_success'] for r in group),errors=sum(r['error'] for r in group))
                for k in set().union(*(r.keys() for r in group)):
                    if k not in row and k not in ('seed','error'):
                        row[k]=med([r.get(k) for r in group])
                row['max_current_raw_increase_abs']=max((r.get('current_raw_max_increase_abs',0.) for r in group),default=0.)
                row['max_current_raw_increase_pct']=max((r.get('current_raw_max_increase_pct',0.) for r in group),default=0.)
                for lo,hi in ((2000,4000),(4000,6000)):
                    if count>=hi:
                        row[f'{lo}_{hi}_improving_runs']=sum((r.get(f'{lo}_{hi}_improvement_pct') or 0)>1e-6 for r in group)
                summary.append(row)
    lookup={(r['scenario'],r['variant'],r['seed'],r['samples']):r for r in detailed}
    paired_runs=[]
    for name in sorted({r['scenario'] for r in runs}):
        seeds=sorted({r['seed'] for r in runs if r['scenario']==name})
        for count in MILESTONES:
            for baseline,new in (('A','exit_rewire'),('A','exit_local'),('exit_rewire','exit_local')):
                for seed in seeds:
                    a,b=(lookup.get((name,v,seed,count),{}) for v in (baseline,new))
                    if a.get('raw_success') and b.get('raw_success'):
                        row=dict(scenario=name,samples=count,baseline=baseline,variant=new,seed=seed)
                        for k in ('best_raw','current_raw','first_raw','best_chain','current_chain','online_s','first_s'):
                            row[k+'_change_pct']=change(a.get(k),b.get(k))
                        paired_runs.append(row)
    paired=[]
    for name in sorted({r['scenario'] for r in runs}):
        for count in MILESTONES:
            for baseline,new in (('A','exit_rewire'),('A','exit_local'),('exit_rewire','exit_local')):
                group=[r for r in paired_runs if (r['scenario'],r['samples'],r['baseline'],r['variant'])==(name,count,baseline,new)]
                row=dict(scenario=name,samples=count,baseline=baseline,variant=new,common_raw_success=len(group))
                for k in ('best_raw','current_raw','first_raw','best_chain','current_chain','online_s','first_s'):
                    row[k+'_change_pct']=med([r[k+'_change_pct'] for r in group])
                row.update(raw_better=sum(r['best_raw_change_pct']<-.1 for r in group),
                           raw_worse=sum(r['best_raw_change_pct']>.1 for r in group))
                paired.append(row)
    # No interpolation: the last checkpoint completed by each shared budget is available.
    by_run={(r['scenario'],r['variant'],r['seed']):r for r in runs}
    matched=[]
    for (name,variant,seed),run in by_run.items():
        if variant=='A':continue
        base=by_run.get((name,'A',seed))
        if not base or run.get('error') or base.get('error'):continue
        budget=min(run['online_time_s'],base['online_time_s'])
        a,b=(measurements(r,selected(r,budget=budget)) for r in (base,run))
        matched.append(dict(scenario=name,variant=variant,seed=seed,budget_s=budget,
                            common_raw_success=a['raw_success'] and b['raw_success'],
                            A_online_s=a['online_s'],variant_online_s=b['online_s'],
                            A_raw=a['best_raw'],variant_raw=b['best_raw'],
                            best_raw_change_pct=change(a['best_raw'],b['best_raw']),
                            best_chain_change_pct=change(a['best_chain'],b['best_chain'])))
    matched_summary=[]
    for name in sorted({r['scenario'] for r in runs}):
        for variant in VARIANTS[1:]:
            group=[r for r in matched if (r['scenario'],r['variant'])==(name,variant)]
            good=[r for r in group if r['common_raw_success']]
            matched_summary.append(dict(scenario=name,variant=variant,pairs=len(group),common_raw_success=len(good),
                budget_s=med([r['budget_s'] for r in group]),
                best_raw_change_pct=med([r['best_raw_change_pct'] for r in good]),
                best_chain_change_pct=med([r['best_chain_change_pct'] for r in good])))
    for filename,rows in (('summary.csv',summary),('milestone_runs.csv',detailed),('paired.csv',paired),
                          ('paired_runs.csv',paired_runs),('matched_online_runs.csv',matched),
                          ('matched_online.csv',matched_summary)):
        csv_write(out,filename,rows)
    curves=[]
    for run in runs:
        best=math.inf
        for rec in run.get('records',[]):
            if rec['raw_success']:best=min(best,rec['raw_cost'])
            curves.append(dict(scenario=run['scenario'],variant=run['variant'],seed=run['seed'],
                iteration=rec['iteration'],online_s=rec['online_time_s'],current_raw=rec['raw_cost'],
                best_raw=best,chain=rec['optimised_cost'],raw_success=rec['raw_success'],chain_success=rec['success']))
    csv_write(out,'curves.csv',curves)
    failures=dict(errors=[dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],error=r['error'])
                          for r in runs if r.get('error')],
                  incomplete=[dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],sample_calls=r.get('sample_calls'))
                              for r in runs if r.get('sample_calls')!=6000],
                  online_solver_failures=sum(r.get('solver_failures',0) for r in runs),
                  chain_solver_failures=sum(r.get('chain_solver_failures',0) for r in runs),
                  validation_failures=[dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],iteration=q['iteration'])
                                      for r in runs for q in r.get('records',[]) if q.get('validation_failed')])
    geometry=[]
    for name,seed in sorted({(r['scenario'],r['seed']) for r in runs}):
        group={r['variant']:r for r in runs if (r['scenario'],r['seed'])==(name,seed)}
        if all(v in group for v in VARIANTS):
            poses=[group[v].get('cell_poses',[]) for v in VARIANTS]
            geometry.append(dict(scenario=name,seed=seed,identical_cell_poses=poses[0]==poses[1]==poses[2],
                                 **{v+'_regions':len(group[v].get('cell_poses',[])) for v in VARIANTS}))
    csv_write(out,'geometry_comparison.csv',geometry)
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    diagnostics=[]
    for name in sorted({r['scenario'] for r in runs}):
        for variant in VARIANTS:
            group=[r for r in runs if (r['scenario'],r['variant'])==(name,variant)]
            if not group:continue
            row=dict(scenario=name,variant=variant,runs=len(group),
                     solver_failures=sum(r.get('solver_failures',0) for r in group),
                     chain_solver_failures=sum(r.get('chain_solver_failures',0) for r in group))
            for key in ('protected_unknown','protected_infeasible','protected_invalid','protection_violations',
                        'goal_cost_increases','rewire_solves','rewire_accepted','local_solves','local_accepted',
                        'compensated_incoming_increases','propagated_nodes'):
                row[key]=sum(r.get('stats',{}).get(key,0) for r in group)
            row['max_exit_delta']=max((r.get('stats',{}).get('exit_max_accepted_delta',-math.inf) for r in group),default=None)
            diagnostics.append(row)
    csv_write(out,'diagnostics.csv',diagnostics)
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,paired=paired,matched_online=matched_summary),indent=2))
    return summary,paired,matched_summary


def plot(runs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scenes=sorted({r['scenario'] for r in runs})
    if not scenes:return
    fig,axes=plt.subplots(2,len(scenes),figsize=(5*len(scenes),7),squeeze=False)
    for col,name in enumerate(scenes):
        for variant in VARIANTS:
            group=[r for r in runs if (r['scenario'],r['variant'])==(name,variant)]
            for row,(key,success) in enumerate((('raw_cost','raw_success'),('optimised_cost','success'))):
                x,y=[],[]
                for count in range(50,6001,50):
                    values=[min((r[key] for r in selected(run,count=count) if r[success]),default=math.inf) for run in group]
                    value=float(np.median(values)) if values else math.inf
                    if finite(value):x.append(count);y.append(value)
                axes[row,col].plot(x,y,label=variant)
                axes[row,col].set(title=name,xlabel='Raw sample calls',ylabel='Best raw tree cost' if row==0 else 'Best post-hoc chain cost')
                axes[row,col].grid(alpha=.2);axes[row,col].legend()
    fig.tight_layout();fig.savefig(out/'anytime_samples.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios',nargs='+',default=['forest','dense','slots'])
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--samples',type=int,default=6000)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results'/'exit_compensation')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    path=out/'runs.jsonl'
    runs=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    if not args.analyze_only:
        manifest=dict(command=sys.argv,python=sys.executable,scenarios={s:DESCRIPTIONS[s] for s in args.scenarios},
            seeds=list(range(args.seeds)),samples=args.samples,checkpoint_every=50,variants=args.variants,
            paired_order='cyclic rotation seed%3, serial; no concurrent CPU experiments',
            timing='online=initialization+step+goal extraction/maintenance; historical step-only also retained; chain and clearance separate',
            primary='raw tree cost; current and best observed separately; chain postprocessing separate',
            matching='last completed checkpoint <= min(pair final online time); no interpolation',
            baseline='unchanged SingleEntryRRTStar(CouplingConfig(step=.3,seed=...))',
            goal='same scenario goal_xy and free-yaw feasible target; A extracts closest feasible yaw, exit variants maintain refreshed endpoint after accepted entry changes; margin .02',
            fixedbudget='ExitCompensated defaults: at most one online local task per raw attempt and one extra constrained rewire solve per raw attempt')
        if not (out/'experiment.json').exists():(out/'experiment.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
        source=out/'source';source.mkdir(exist_ok=True)
        for filename in ('formation/exit_compensated_rrtstar.py','formation/region_coupling.py','formation/polyhedral_cell.py',
                         'scripts/compare_exit_compensation.py','scripts/compare_single_tree.py','scripts/compare_coupling.py'):
            if not (source/Path(filename).name).exists():shutil.copyfile(ROOT/filename,source/Path(filename).name)
        (out/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
            repo_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            packages={p:version(p) for p in ('numpy','scipy','clarabel','matplotlib')}),indent=2))
        done={(r['scenario'],r['variant'],r['seed']) for r in runs}
        expected=len(args.scenarios)*args.seeds*len(args.variants)
        session_runs,session_wall=0,0.
        with path.open('a') as stream:
            for name in args.scenarios:
                for seed in range(args.seeds):
                    shift=seed%len(args.variants)
                    for variant in args.variants[shift:]+args.variants[:shift]:
                        if (name,variant,seed) in done:continue
                        began=time.perf_counter()
                        try:run=run_one(name,variant,seed,args.samples)
                        except Exception:run=dict(scenario=name,variant=variant,seed=seed,records=[],error=traceback.format_exc())
                        stream.write(json.dumps(run)+'\n');stream.flush();runs.append(run)
                        done.add((name,variant,seed));session_runs+=1;session_wall+=time.perf_counter()-began
                        eta=session_wall/session_runs*(expected-len(done))/60
                        last=run.get('records',[{}])[-1] if run.get('records') else {}
                        print(f"{len(done)}/{expected} {name} seed={seed} {variant} samples={run.get('sample_calls')} "
                              f"online={run.get('online_time_s',0):.3f}s wall={run.get('actual_wall_s',0):.3f}s "
                              f"raw={last.get('raw_cost')} nodes={run.get('nodes')} eta={eta:.1f}min "
                              f"status={run.get('error','ok')}",flush=True)
                summary,paired,matched=analyse(runs,out)
                print('SCENARIO_COMPLETE',name,json.dumps([r for r in paired if r['scenario']==name and r['samples']==6000]),flush=True)
    summary,paired,matched=analyse(runs,out);plot(runs,out)
    print(json.dumps(dict(final=[r for r in summary if r['samples']==6000],paired=[r for r in paired if r['samples']==6000],matched=matched),indent=2),flush=True)

if __name__=='__main__':main()
