#!/usr/bin/env python3
"""Matched serial raw-path comparison for the bounded batch region experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/formation-matplotlib')
import argparse, csv, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_coupling import scenario, route_clearance
from compare_single_tree import initialize_a, step_a
from formation.region_coupling import CouplingConfig, SingleEntryRRTStar
from formation.exit_compensated_rrtstar import ExitCompensatedConfig, ExitCompensatedRRTStar
from formation.batch_region_rrtstar import BatchRegionRRTStar

VARIANTS=('A','exit_rewire','batch_region')
def run_one(name,variant,seed,budget):
    data,slots,start=scenario(name)
    config=(CouplingConfig if variant=='A' else ExitCompensatedConfig)(seed=seed,step=.3,max_iterations=budget)
    if variant!='A': config.enable_local=False
    cls={'A':SingleEntryRRTStar,'exit_rewire':ExitCompensatedRRTStar,'batch_region':BatchRegionRRTStar}[variant]
    began=time.perf_counter(); p=cls(data,slots,start,config)
    (initialize_a(p) if variant=='A' else p.initialize())
    records=[]; first=False; error=None
    sample_calls=0; original_sample=p._sample
    def counted_sample():
        nonlocal sample_calls
        sample_calls += 1
        return original_sample()
    p._sample=counted_sample
    def record(kind,count):
        nonlocal first
        seq, raw, path=[],math.inf,[]
        if p.goal_cells:
            seq,raw=p._region_chain(); last=seq[-1]
            endpoint=getattr(p,'goal_endpoints',{}).get(last)
            yaw=endpoint.yaw if endpoint else p.geo.goal_yaw(p.cells[last],p.goal_xy,p.entry[last].yaw,config.overlap_margin)
            path=[(p.entry[i].x,p.entry[i].y,p.entry[i].yaw) for i in seq]+[(*p.goal_xy,yaw)]
        rec=dict(kind=kind,iteration=count,raw_cost=raw,tree_route=path,raw_success=bool(seq) and math.isfinite(raw),regions=len(p.cells),stats=dict(p.stats),online_solver_solves=p.geo.solves,solver_failures=p.geo.failures)
        records.append(rec)
        rec['online_time_s']=time.perf_counter()-began
        if rec['raw_success'] and not first: first=True; rec['first_solution']=True
    count=0
    try:
        for count in range(1,budget+1):
            (step_a(p) if variant=='A' else p.step())
            if sample_calls != count: raise RuntimeError(f'Unexpected sample count: {sample_calls}, expected {count}')
            p.stats['iterations']=count
            # Include the partial batch and fixed tail in the 6000 record.
            if count==budget and variant=='batch_region': p.flush(tail=True)
            if (not first and p.goal_cells) or count%50==0 or count==budget:
                record('first' if not first and p.goal_cells else 'checkpoint',count)
    except Exception: error=traceback.format_exc()
    online=time.perf_counter()-began
    # Validation is deliberately outside online time; every observed path is checked.
    cache={}; max_error=0.; validation_at=time.perf_counter()
    for rec in records:
        path=rec['tree_route']
        if not path: continue
        key=tuple(tuple(x) for x in path)
        if key not in cache: cache[key]=route_clearance(data,slots,path,step=.01)
        rec['raw_clearance']=cache[key]
        delta=np.diff(np.asarray(path),axis=0); delta[:,2]=(delta[:,2]+math.pi)%(2*math.pi)-math.pi; delta[:,2]*=p.rho
        rec['cost_error']=abs(float(np.linalg.norm(delta,axis=1).sum())-rec['raw_cost']); max_error=max(max_error,rec['cost_error'])
        if cache[key]<config.safety_distance-1e-6 or rec['cost_error']>1e-6:
            rec['raw_success']=False; rec['validation_failed']=True
    first_record=next((r for r in records if r.get('first_solution') and r['raw_success']),None)
    result=dict(scenario=name,variant=variant,seed=seed,sample_calls=sample_calls,online_time_s=online,records=records,
        nodes=len(p.cells),stats=dict(p.stats),solver_failures=p.geo.failures,solves=p.geo.solves,max_cost_error=max_error,
        validation_time_s=time.perf_counter()-validation_at,first_iteration=first_record['iteration'] if first_record else None)
    if error: result['error']=error
    return result

def write_csv(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def analyse(runs,out):
    milestones=[]; pairs=[]; curves=[]
    for r in runs:
        for rec in r['records']:
            curves.append(dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],**{k:rec.get(k) for k in ('kind','iteration','online_time_s','raw_cost','raw_success')}))
        first=next((x for x in r['records'] if x.get('first_solution') and x['raw_success']),None)
        for count in (4000,6000):
            first_at_count=first if first and first['iteration']<=count else None
            available=[x for x in r['records'] if x['iteration']<=count and x['raw_success']]
            current=next((x for x in reversed(r['records']) if x['iteration']==count),None)
            best=min((x['raw_cost'] for x in available),default=math.inf)
            milestones.append(dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],samples=count,complete=r['sample_calls']>=count and not r.get('error'),success=bool(current and current['raw_success'] and not r.get('error')),best=best,current=current['raw_cost'] if current else None,online_s=current['online_time_s'] if current else None,first_J=first_at_count['raw_cost'] if first_at_count else None,first_s=first_at_count['online_time_s'] if first_at_count else None,first_proposal=first_at_count['iteration'] if first_at_count else None,regions=current['regions'] if current else None,solves=current['online_solver_solves'] if current else None,candidates=(current or {}).get('stats',{}).get('batch_candidates'),pending=(current or {}).get('stats',{}).get('batch_pending'),rewire_accepted=(current or {}).get('stats',{}).get('rewire_accepted')))
    lookup={(r['scenario'],r['variant'],r['seed']):r for r in runs}
    for name,variant,seed in sorted(lookup):
        if variant=='A':continue
        a=lookup.get((name,'A',seed)); b=lookup[(name,variant,seed)]
        if not a or a.get('error') or b.get('error'):continue
        limit=min(a['online_time_s'],b['online_time_s'])
        def at(r):
            recs=[x for x in r['records'] if x['online_time_s']<=limit and x['raw_success']]
            return min((x['raw_cost'] for x in recs),default=math.inf)
        aj,bj=at(a),at(b); common=math.isfinite(aj) and math.isfinite(bj)
        pairs.append(dict(scenario=name,variant=variant,seed=seed,budget_s=limit,common_success=common,A=aj,batch=bj,change_pct=100*(bj/aj-1) if common and aj else None))
    write_csv(out/'milestones.csv',milestones);write_csv(out/'matched_time.csv',pairs);write_csv(out/'curves.csv',curves)
    failures=[dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],error=r.get('error'),solver_failures=r['solver_failures'],invalid=sum(bool(x.get('validation_failed')) for x in r['records'])) for r in runs if r.get('error') or r['solver_failures'] or any(x.get('validation_failed') for x in r['records'])]
    (out/'failures.json').write_text(json.dumps(failures,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=6000);ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--scenarios',nargs='+',default=['forest','dense','slots']);ap.add_argument('--variants',nargs='+',default=list(VARIANTS));ap.add_argument('--out-dir',type=Path,default=ROOT/'results/batch_region_trial');args=ap.parse_args()
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True); path=out/'runs.jsonl';runs=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={(x['scenario'],x['variant'],x['seed']) for x in runs}
    for name in args.scenarios:
        for seed in range(args.seeds):
            order=args.variants[seed%len(args.variants):]+args.variants[:seed%len(args.variants)]
            for variant in order:
                if (name,variant,seed) in done:continue
                r=run_one(name,variant,seed,args.samples); runs.append(r)
                with path.open('a') as f:f.write(json.dumps(r)+'\n')
                analyse(runs,out)
                last=r['records'][-1] if r['records'] else {}
                print(name,seed,variant,'time',round(r['online_time_s'],2),'cost',last.get('raw_cost'),'regions',r['nodes'],'error',r.get('error','none'),flush=True)
    analyse(runs,out)
if __name__=='__main__':main()
