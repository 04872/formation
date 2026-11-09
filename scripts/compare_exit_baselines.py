#!/usr/bin/env python3
"""6000 original proposals: existing A/exit runs plus local RRT*/BIT* implementations."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/formation-matplotlib')
import argparse
import heapq
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
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from compare_coupling import scenario, route_clearance, DESCRIPTIONS
from compare_exit_compensation import csv_write, finite, med, change, improvement, measurements, selected, interval_measure
from formation.baselines import BaselineConfig, BITStar, RRTStar, distance
from formation.types import wrap_to_pi

VARIANTS = ('A', 'exit_rewire', 'exit_local', 'rrtstar', 'bitstar')
STEPS = dict(forest=.3, dense=1., slots=1.)

class ProposalBudget(Exception):
    pass

class BudgetBITStar(BITStar):
    """Only sampling/batch adapter; search, rewiring and parent updates are inherited."""
    def configure_budget(self, budget, callback):
        self.proposals = 0
        self.valid_samples = 0
        self.informed_rejections = 0
        self.boundary_rejections = 0
        self.invalid_samples = 0
        self.budget = budget
        self.proposal_callback = callback
        self.partial_batch = 0

    def _attempt(self):
        # Complete the previous attempt before recording its boundary; the 6000th
        # proposal can be valid and must enter the final partial batch.
        if self.proposals and self.proposals % 50 == 0:
            self.proposal_callback('checkpoint')
        if self.proposals >= self.budget:
            raise ProposalBudget
        self.proposals += 1

    def _sample(self, at_goal):
        if at_goal:
            self._attempt()
            return self.goal_state()
        if not math.isfinite(self.c_best):
            self._attempt()
            return self.uniform()
        a, b = self.start[:2], self.goal_xy
        center, focal = (a+b)/2., float(np.linalg.norm(b-a))
        major = self.c_best/2.
        minor = math.sqrt(max(major**2-(focal/2.)**2, 0.))
        angle = math.atan2(b[1]-a[1], b[0]-a[0])
        rotation = np.array(((math.cos(angle),-math.sin(angle)),(math.sin(angle),math.cos(angle))))
        span = min(math.pi,self.c_best/self.rho)
        while True:
            self._attempt()
            r, phi = math.sqrt(self.rng.random()), self.rng.uniform(-math.pi,math.pi)
            xy = center+rotation@np.array((major*r*math.cos(phi),minor*r*math.sin(phi)))
            if np.any(xy<self.low) or np.any(xy>self.high):
                self.boundary_rejections += 1
                continue
            q = np.array((*xy,wrap_to_pi(self.start[2]+self.rng.uniform(-span,span))))
            if distance(self.start,q,self.rho)+float(np.hypot(*(xy-self.goal_xy))) < self.c_best:
                return q
            self.informed_rejections += 1

    def _new_batch(self):
        self.batches += 1
        n = self.n
        if math.isfinite(self.c_best):
            idx = np.flatnonzero(self.alive[:n] & ~self.in_tree[:n])
            prune = idx[self.g_hat(idx)+self.h_hat(idx)>=self.c_best]
            self.alive[prune] = False
        self.old[:n] = self.in_tree[:n]
        goal_count = max(1,int(round(self.config.goal_bias*self.config.batch_size)))
        added = 0
        while added < self.config.batch_size:
            try:
                q = self._sample(added<goal_count)
            except ProposalBudget:
                break
            if not self.check.valid(q)[0]:
                self.invalid_samples += 1
                continue
            self.valid_samples += 1
            if self.n == len(self.Q):
                self._grow()
            i = self.n
            self.Q[i],self.alive[i],self.goal[i] = q,True,added<goal_count
            self.children.append(set())
            self.n += 1
            added += 1
        if added < self.config.batch_size:
            self.partial_batch = added
        count = int(self.alive[:self.n].sum())
        self.radius = self.gamma*(math.log(count)/count)**(1./3.)
        self.qe = []
        tree = np.flatnonzero(self.in_tree[:self.n])
        self.qv = [(float(self.g[v]+self.h_hat([v])[0]),int(v)) for v in tree]
        heapq.heapify(self.qv)


def run_one(name, variant, seed, budget=6000):
    data, slots, start = scenario(name)
    cfg = BaselineConfig(seed=seed,clearance=.02,edge_resolution=.02,step=STEPS[name],
                         batch_size=200,goal_bias=.1,rewire_factor=1.1)
    began = time.perf_counter()
    planner = (RRTStar if variant=='rrtstar' else BudgetBITStar)(data,slots,start,cfg)
    if not planner.check.valid(planner.start)[0]:
        raise ValueError('invalid start')
    init_s = time.perf_counter()-began
    records, proposals, steps, maintenance_s, step_s = [],0,0,0.,0.
    first, error, exhausted_s, last_observed_s = None,None,None,0.
    best_seen = math.inf
    valid_steered, invalid_steered = 0,0
    inside_step = False

    def count_now():
        return planner.proposals if variant=='bitstar' else proposals

    def snapshot(kind):
        nonlocal maintenance_s,first,last_observed_s,best_seen
        at = time.perf_counter()
        path = planner.best_path()
        elapsed = time.perf_counter()-at
        maintenance_s += elapsed
        now = time.perf_counter()-began
        count = count_now()
        if records and records[-1]['iteration']==count and kind=='checkpoint' and records[-1]['kind']=='checkpoint':
            return
        cost,route = path if path else (math.inf,[])
        rec = dict(kind=kind,iteration=count,online_time_s=now,raw_success=path is not None,
                   raw_cost=cost,tree_route=[list(map(float,q)) for q in route],
                   success=False,optimised_cost=math.inf,route=[],
                   extraction_time_s=elapsed,maintenance_time_s=maintenance_s,
                   historical_step_time_s=step_s,nodes=planner.size(),regions=planner.size(),
                   completed_steps=steps,stats=dict(states_checked=planner.check.states_checked,
                   edges_checked=planner.edges_checked,vertices=planner.size(),
                   valid_samples=planner.valid_samples if variant=='bitstar' else valid_steered),
                   solver_failures=0)
        records.append(rec)
        last_observed_s=now
        best_seen=min(best_seen,cost)
        if first is None and path:
            first=rec
            rec['first_solution']=True
        return rec

    if variant=='bitstar':
        planner.configure_budget(budget,snapshot)
    else:
        # Count validity of the steered candidate without adding any validity work.
        original_valid=planner.check.valid
        def tracked_valid(q):
            nonlocal valid_steered,invalid_steered
            result=original_valid(q)
            if inside_step and not getattr(tracked_valid,'seen',False):
                tracked_valid.seen=True
                valid_steered+=int(result[0]);invalid_steered+=int(not result[0])
            return result
        planner.check.valid=tracked_valid
    try:
        while True:
            count=count_now()
            if count>=budget and (variant=='rrtstar' or (not planner.qe and not planner.qv)):
                break
            if variant=='rrtstar':
                proposals+=1
                tracked_valid.seen=False
            before_maintenance=maintenance_s
            tick=time.perf_counter();inside_step=True
            planner.step()
            inside_step=False
            step_s+=time.perf_counter()-tick-(maintenance_s-before_maintenance)
            steps+=1
            # O(1) for BIT*, array minimum for RRT*: no path traversal each queue pop.
            detect=time.perf_counter()
            if variant=='bitstar':
                cost=planner.c_best
            elif planner.goal_nodes:
                cost=float(np.min(planner.cost[planner.goal_nodes]+np.asarray(planner.goal_legs)))
            else:
                cost=math.inf
            maintenance_s+=time.perf_counter()-detect
            count=count_now()
            if exhausted_s is None and count>=budget:
                snapshot('budget_exhausted');exhausted_s=time.perf_counter()-began
            elif cost<best_seen-1e-9:
                snapshot('first' if first is None else 'improvement')
            elif variant=='rrtstar' and count%50==0:
                snapshot('checkpoint')
            elif time.perf_counter()-began-last_observed_s>=.05:
                snapshot('time')
        snapshot('tail_drained' if variant=='bitstar' else 'final')
    except Exception:
        error=traceback.format_exc()
    online=time.perf_counter()-began
    metric_at=time.perf_counter()
    # Validate all distinct raw paths; failed dense checks remain explicit failures.
    validation_cache={}
    for rec in records:
        if not rec['tree_route']:continue
        key=tuple(tuple(q) for q in rec['tree_route'])
        if key not in validation_cache:
            validation_cache[key]=route_clearance(data,slots,rec['tree_route'],step=.01)
        rec['raw_clearance']=validation_cache[key]
        if rec['raw_clearance']<cfg.clearance-1e-6:
            rec['raw_success']=False;rec['validation_failed']=True
    metrics_s=time.perf_counter()-metric_at
    shortcut_s=0.
    if records and records[-1]['tree_route']:
        short_at=time.perf_counter()
        path=planner.shortcut([np.array(q) for q in records[-1]['tree_route']])
        shortcut_s=time.perf_counter()-short_at
        records[-1].update(shortcut_route=[q.tolist() for q in path],shortcut_cost=planner.path_length(path),
                              shortcut_clearance=route_clearance(data,slots,[q.tolist() for q in path]))
    count=count_now()
    result=dict(scenario=name,variant=variant,seed=seed,config=asdict(cfg),records=records,
        sample_calls=count,completed_iterations=steps,online_time_s=online,initialisation_time_s=init_s,
        historical_step_time_s=step_s,maintenance_time_s=maintenance_s,
        observation_overhead_time_s=max(0.,online-init_s-step_s-maintenance_s),
        budget_exhausted_online_time_s=exhausted_s,tail_time_s=max(0.,online-exhausted_s) if exhausted_s else None,
        shortcut_time_s=shortcut_s,clearance_metrics_time_s=metrics_s,
        actual_wall_s=time.perf_counter()-began,nodes=planner.size(),
        first_iteration=first['iteration'] if first else None,first_online_time_s=first['online_time_s'] if first else None,
        stats=dict(states_checked=records[-1]['stats']['states_checked'] if records else 0,
                   edges_checked=records[-1]['stats']['edges_checked'] if records else 0,
                   vertices=planner.size(),valid_samples=planner.valid_samples if variant=='bitstar' else valid_steered,
                   invalid_samples=planner.invalid_samples if variant=='bitstar' else invalid_steered,
                   proposals=count,queue_steps=steps if variant=='bitstar' else None,
                   batches=planner.batches if variant=='bitstar' else None,
                   partial_batch=planner.partial_batch if variant=='bitstar' else None,
                   boundary_rejections=planner.boundary_rejections if variant=='bitstar' else None,
                   informed_rejections=planner.informed_rejections if variant=='bitstar' else None))
    if error:result['error']=error
    return result


def analyse(runs,out):
    detailed=[]
    for run in runs:
        for count in (2000,4000,6000):
            recs=selected(run,count=count)
            row=dict(scenario=run['scenario'],variant=run['variant'],seed=run['seed'],samples=count,
                     complete=run.get('sample_calls',0)>=count and not run.get('error'),error=bool(run.get('error')))
            row.update(measurements(run,recs))
            row['final_online_s']=run.get('online_time_s')
            row['tail_s']=run.get('tail_time_s')
            row['completed_steps']=(recs[-1].get('completed_steps',recs[-1].get('stats',{}).get('iterations')) if recs else None)
            row['proposals']=recs[-1]['iteration'] if recs else None
            row.update({f'4000_6000_{k}':v for k,v in interval_measure(run,4000,6000).items()})
            if not row['complete']:row['raw_success']=False
            detailed.append(row)
    summary=[]
    for name in sorted({r['scenario'] for r in runs}):
        for count in (2000,4000,6000):
            for variant in VARIANTS:
                group=[r for r in detailed if (r['scenario'],r['samples'],r['variant'])==(name,count,variant)]
                if not group:continue
                row=dict(scenario=name,samples=count,variant=variant,runs=len(group),
                         complete=sum(r['complete'] for r in group),raw_success=sum(r['raw_success'] for r in group),
                         errors=sum(r['error'] for r in group))
                for key in set().union(*(r.keys() for r in group)):
                    if key not in row and key!='seed':row[key]=med([r.get(key) for r in group])
                summary.append(row)
    lookup={(r['scenario'],r['variant'],r['seed'],r['samples']):r for r in detailed}
    pair_rows=[]
    byrun={(r['scenario'],r['variant'],r['seed']):r for r in runs}
    matched=[]
    for name in sorted({r['scenario'] for r in runs}):
        for baseline in ('A','exit_local'):
            for variant in VARIANTS:
                if variant==baseline:continue
                seeds=sorted({r['seed'] for r in runs if r['scenario']==name and r['variant']==variant})
                for seed in seeds:
                    for count in (2000,4000,6000):
                        a,b=(lookup.get((name,v,seed,count)) for v in (baseline,variant))
                        if a is None or b is None:continue
                        row=dict(scenario=name,baseline=baseline,variant=variant,seed=seed,samples=count,
                                 baseline_success=a['raw_success'],variant_success=b['raw_success'],
                                 common_raw_success=a['raw_success'] and b['raw_success'])
                        for key in ('best_raw','current_raw','first_raw','online_s','first_s'):
                            row[key+'_change_pct']=change(a.get(key),b.get(key)) if row['common_raw_success'] else None
                        pair_rows.append(row)
                    a,b=(byrun.get((name,v,seed)) for v in (baseline,variant))
                    if a is None or b is None or a.get('error') or b.get('error'):continue
                    budget=min(a['online_time_s'],b['online_time_s'])
                    ar,br=(selected(r,budget=budget) for r in (a,b))
                    am,bm=(measurements(r,recs) for r,recs in ((a,ar),(b,br)))
                    matched.append(dict(scenario=name,baseline=baseline,variant=variant,seed=seed,budget_s=budget,
                        baseline_success=am['raw_success'],variant_success=bm['raw_success'],
                        common_raw_success=am['raw_success'] and bm['raw_success'],
                        baseline_online_s=am['online_s'],variant_online_s=bm['online_s'],
                        baseline_iteration=ar[-1]['iteration'] if ar else None,
                        variant_iteration=br[-1]['iteration'] if br else None,
                        baseline_raw=am['best_raw'],variant_raw=bm['best_raw'],
                        best_raw_change_pct=change(am['best_raw'],bm['best_raw'])))
    def pair_summary(rows,keys):
        result=[]
        groups={tuple(r[k] for k in keys) for r in rows}
        for group in sorted(groups):
            allrows=[r for r in rows if tuple(r[k] for k in keys)==group]
            good=[r for r in allrows if r['common_raw_success']]
            row=dict(zip(keys,group));row.update(pairs=len(allrows),common_raw_success=len(good),
                baseline_success=sum(r['baseline_success'] for r in allrows),variant_success=sum(r['variant_success'] for r in allrows),
                raw_better=sum((r.get('best_raw_change_pct') or 0)<-.1 for r in good),
                raw_worse=sum((r.get('best_raw_change_pct') or 0)>.1 for r in good))
            for key in ('best_raw_change_pct','current_raw_change_pct','first_raw_change_pct','online_s_change_pct','first_s_change_pct','budget_s'):
                row[key]=med([r.get(key) for r in good])
            result.append(row)
        return result
    paired=pair_summary(pair_rows,('scenario','samples','baseline','variant'))
    match_summary=pair_summary(matched,('scenario','baseline','variant'))
    curves=[]
    for run in runs:
        best=math.inf
        for rec in run['records']:
            if rec['raw_success']:best=min(best,rec['raw_cost'])
            curves.append(dict(scenario=run['scenario'],variant=run['variant'],seed=run['seed'],
                kind=rec['kind'],iteration=rec['iteration'],online_s=rec['online_time_s'],
                current_raw=rec['raw_cost'],best_raw=best,raw_success=rec['raw_success']))
    for filename,rows in (('summary.csv',summary),('milestone_runs.csv',detailed),('paired.csv',paired),
            ('paired_runs.csv',pair_rows),('matched_online.csv',match_summary),('matched_online_runs.csv',matched),('curves.csv',curves)):
        csv_write(out,filename,rows)
    failures=dict(errors=[dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],error=r['error']) for r in runs if r.get('error')],
        incomplete=[dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],proposals=r.get('sample_calls')) for r in runs if r.get('sample_calls')!=6000],
        validation_failures=[dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],iteration=q['iteration'],kind=q['kind'],clearance=q.get('raw_clearance')) for r in runs for q in r['records'] if q.get('validation_failed')])
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,paired=paired,matched_online=match_summary),indent=2))
    audit_runs([r for r in runs if r['variant'] in ('rrtstar','bitstar')],out)
    return summary


def audit_runs(runs,out):
    """Reproducible proposal/path accounting and timer-comparability diagnostics."""
    audit=[]
    for run in runs:
        recs,stats=run['records'],run.get('stats',{})
        rho=float(np.linalg.norm(scenario(run['scenario'])[1],axis=1).max())
        errors=[]
        for rec in recs:
            if not rec['tree_route']:continue
            delta=np.diff(np.asarray(rec['tree_route']),axis=0)
            delta[:,2]=(delta[:,2]+math.pi)%(2*math.pi)-math.pi
            delta[:,2]*=rho
            errors.append(abs(float(np.linalg.norm(delta,axis=1).sum())-rec['raw_cost']))
        row=dict(scenario=run['scenario'],variant=run['variant'],seed=run['seed'],proposals=run.get('sample_calls'),
            monotone_time=all(a['online_time_s']<=b['online_time_s'] for a,b in zip(recs,recs[1:])),
            all_50_boundaries=not(set(range(50,run.get('sample_calls',0)+1,50))-{r['iteration'] for r in recs}),
            max_path_cost_error=max(errors,default=0.),
            online_equals_components_error=abs(run.get('online_time_s',0)-sum(run.get(k,0) for k in
                ('initialisation_time_s','historical_step_time_s','maintenance_time_s','observation_overhead_time_s'))),
            validation_failures=sum(bool(r.get('validation_failed')) for r in recs),
            final_success=bool(recs and recs[-1]['raw_success']),error=bool(run.get('error')))
        if run['variant']=='bitstar':
            row['proposal_accounting_sum']=sum(stats.get(k,0) for k in
                ('valid_samples','invalid_samples','boundary_rejections','informed_rejections'))
            row['distinct_budget_and_tail']=any(r['kind']=='budget_exhausted' for r in recs) and bool(recs and recs[-1]['kind']=='tail_drained')
        audit.append(row)
    csv_write(out,'audit.csv',audit)
    timing=[]
    for scene in sorted({r['scenario'] for r in runs}):
        for variant in ('rrtstar','bitstar'):
            group=[r for r in runs if (r['scenario'],r['variant'])==(scene,variant) and not r.get('error')]
            if not group:continue
            row=dict(scenario=scene,variant=variant,runs=len(group))
            for key in ('online_time_s','initialisation_time_s','historical_step_time_s','maintenance_time_s',
                        'observation_overhead_time_s','tail_time_s','clearance_metrics_time_s','shortcut_time_s'):
                row[key]=med([r.get(key) for r in group])
            row['observation_overhead_pct']=med([100*r['observation_overhead_time_s']/r['online_time_s'] for r in group])
            timing.append(row)
    csv_write(out,'timing_breakdown.csv',timing)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios',nargs='+',default=['forest','dense','slots'])
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--samples',type=int,default=6000)
    parser.add_argument('--variants',nargs='+',choices=('rrtstar','bitstar'),default=['rrtstar','bitstar'])
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results'/'exit_baselines')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    path=out/'runs.jsonl'
    runs=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    if not args.analyze_only:
        manifest=dict(command=sys.argv,scenarios={n:DESCRIPTIONS[n] for n in args.scenarios},seeds=list(range(args.seeds)),samples=args.samples,
            implementations='formation.baselines local RRTStar and BITStar, not OMPL; no core search or rewiring changes',
            selection='fixed independent historical pilot: results/external/experiment.json; no current-seed tuning',
            parameters=dict(rrt_steps=STEPS,bit_batch=200,goal_bias=.1,rewire_factor=1.1,clearance=.02,edge_resolution=.02),
            budget='raw proposals including invalid states and all informed/boundary rejections; informed XY outside bounds counts before yaw draw; no 6001st proposal; last partial BIT batch drained',
            bit_tail='budget_exhausted at 6000 before search of last sampled batch; tail_drained is final 6000 result; all tail online counted',
            snapshots='actual timestamps and proposal counts: every 50 completed proposals during batch sampling, first/cost improvements and about 0.05 online sec during search; never backfill batch result',
            timing='online includes initialization, sampling, search, cost detection and raw path extraction, plus observation overhead; shortcut and dense validation excluded and separately timed; original A/exit runs reused from previous serial session, not interleaved with new baselines',
            timing_comparability='old A/exit online is init+step+raw extraction and excludes record formatting; baseline online additionally includes driver/record observation overhead, quantified per run; timer definitions are close but not identical',
            comparison='raw paths primary; shortcut final only separate; last observed snapshot <= min same-seed final online budget; no interpolation',
            validity='same physical clearance .02; baselines discrete d_W edge checks .02, A continuous cell certificates with extra portal margin .02; all distinct baseline raw routes densely checked .01 and failures retained',
            valid_samples='BIT accepted collision-free batch samples; RRT valid steered candidates before parent edge check (raw proposals may lie beyond steering radius)',
            order='serial; reverse two baseline order on odd seed; no concurrent CPU experiments')
        (out/'experiment.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
        source=out/'source';source.mkdir(exist_ok=True)
        for filename in ('formation/baselines.py','scripts/compare_exit_baselines.py','scripts/compare_coupling.py','scripts/compare_exit_compensation.py'):
            shutil.copyfile(ROOT/filename,source/Path(filename).name)
        (out/'environment.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            repo_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),packages={p:version(p) for p in ('numpy','scipy','clarabel','matplotlib')}),indent=2))
        done={(r['scenario'],r['variant'],r['seed']) for r in runs}
        total=len(args.scenarios)*args.seeds*len(args.variants)
        session_at=time.perf_counter();completed=0
        with path.open('a') as stream:
            for name in args.scenarios:
                for seed in range(args.seeds):
                    order=args.variants[::(-1 if seed%2 else 1)]
                    for variant in order:
                        if (name,variant,seed) in done:continue
                        try:run=run_one(name,variant,seed,args.samples)
                        except Exception:run=dict(scenario=name,variant=variant,seed=seed,records=[],error=traceback.format_exc())
                        stream.write(json.dumps(run)+'\n');stream.flush();runs.append(run)
                        done.add((name,variant,seed));completed+=1
                        final=run['records'][-1] if run['records'] else {}
                        eta=(time.perf_counter()-session_at)/completed*(total-len(done))/60
                        print(f"{len(done)}/{total} {name} seed={seed} {variant} proposals={run.get('sample_calls')} steps={run.get('completed_iterations')} online={run.get('online_time_s',0):.3f}s tail={run.get('tail_time_s',0) or 0:.3f}s wall={run.get('actual_wall_s',0):.3f}s raw={final.get('raw_cost')} valid={final.get('raw_success')} eta={eta:.1f}min error={run.get('error','none')}",flush=True)
    original=ROOT/'results'/'exit_compensation'/'runs.jsonl'
    old=[json.loads(s) for s in original.read_text().splitlines()]
    combined=old+runs
    summary=analyse(combined,out)
    print(json.dumps([r for r in summary if r['samples']==6000],indent=2),flush=True)
if __name__=='__main__':main()
