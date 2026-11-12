#!/usr/bin/env python3
"""Matched serial raw-path comparison for direct uniform certificate batches."""
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
from formation.direct_batch_region import DirectBatchRegion

VARIANTS=('A','exit_rewire','direct_batch_region')
def run_one(name,variant,seed,budget):
    data,slots,start=scenario(name)
    config=(CouplingConfig if variant=='A' else ExitCompensatedConfig)(seed=seed,step=.3,max_iterations=budget)
    if variant!='A': config.enable_local=False
    cls={'A':SingleEntryRRTStar,'exit_rewire':ExitCompensatedRRTStar,'direct_batch_region':DirectBatchRegion}[variant]
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
            if count==budget and variant=='direct_batch_region':
                if count % p.batch_size: p.flush()
                p.flush(tail=True)
            if (not first and p.goal_cells) or count%50==0 or count==budget:
                record('first' if not first and p.goal_cells else 'checkpoint',count)
            if count%1000==0:
                last=records[-1]
                print(name,variant,seed,'proposal',count,'active',len(p.cells),
                    'valid',p.stats.get('batch_candidates'),'pending',p.stats.get('batch_pending'),
                    'queue',p.stats.get('batch_queue'),'J',last['raw_cost'],
                    'time',round(time.perf_counter()-began,3),flush=True)
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
        validation_time_s=time.perf_counter()-validation_at,first_iteration=first_record['iteration'] if first_record else None,
        configuration_sampling='uniform_xy_yaw' if variant=='direct_batch_region' else 'goal_biased_0.1',
        batch_size=getattr(p,'batch_size',None),work_per_batch=getattr(p,'work_per_batch',None),tail_work=getattr(p,'tail_work',None))
    result['reachable_goal_regions']=len(p.goal_cells)
    if variant=='direct_batch_region':
        goal_regions=[rid for rid,c in enumerate(p.regions)
            if p.geo.goal_yaw(c,p.goal_xy,c.pose.yaw,config.overlap_margin) is not None]
        result['goal_covering_regions']=len(goal_regions)
        result['pending_goal_regions']=sum(rid not in p.active for rid in goal_regions)
        result['active_certificate_ids']=p.node_region
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
        qualified=common and not a['solver_failures'] and not b['solver_failures']
        pairs.append(dict(scenario=name,variant=variant,seed=seed,budget_s=limit,common_success=qualified,
            raw_common_success=common,A=aj,batch=bj,change_pct=100*(bj/aj-1) if qualified and aj else None,
            A_solver_failures=a['solver_failures'],candidate_solver_failures=b['solver_failures']))
    for row in milestones:
        run=lookup[row['scenario'],row['variant'],row['seed']]
        rec=next((x for x in reversed(run['records']) if x['iteration']==row['samples']),{})
        stats=rec.get('stats',{})
        row['raw_success']=row['success']
        row['success']=row['success'] and not run['solver_failures']
        row.update(queue=stats.get('batch_queue'),heap_pops=stats.get('batch_pops'),
                   potential_solves=stats.get('direct_potential_solves'),
                   exact_pops=stats.get('direct_exact_pops'),solver_failures=rec.get('solver_failures'))
        if row['candidates'] is not None:
            row['pending']=row['candidates']-(row['regions']-1)
    write_csv(out/'milestones.csv',milestones);write_csv(out/'matched_time.csv',pairs);write_csv(out/'curves.csv',curves)
    failures=[dict(scenario=r['scenario'],variant=r['variant'],seed=r['seed'],error=r.get('error'),solver_failures=r['solver_failures'],invalid=sum(bool(x.get('validation_failed')) for x in r['records'])) for r in runs if r.get('error') or r['solver_failures'] or any(x.get('validation_failed') for x in r['records'])]
    (out/'failures.json').write_text(json.dumps(failures,indent=2))

def plot_results(runs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors={'A':'tab:blue','exit_rewire':'tab:orange','direct_batch_region':'tab:green'}
    for name,seed in sorted({(r['scenario'],r['seed']) for r in runs}):
        data,slots,start=scenario(name)
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        route_ax,proposal_ax,time_ax=axes
        for item in data.obstacle_primitives:
            if item.get('type')=='circle':
                route_ax.add_patch(Circle(item['center_xy'],item['radius'],color='.65'))
        for r in runs:
            if (r['scenario'],r['seed'])!=(name,seed):continue
            variant=r['variant'];records=r['records'];color=colors[variant]
            valid=[x for x in records if x['raw_success']]
            proposal_ax.plot([x['iteration'] for x in records],[x['raw_cost'] for x in records],color=color,label=variant)
            time_ax.plot([x['online_time_s'] for x in records],[x['raw_cost'] for x in records],color=color,label=variant)
            if valid:
                raw=records[-1]['tree_route']
                route_ax.plot([q[0] for q in raw],[q[1] for q in raw],color=color,label=variant)
                if variant=='direct_batch_region':
                    first_raw=valid[0]['tree_route']
                    route_ax.plot([q[0] for q in first_raw],[q[1] for q in first_raw],
                        color=color,linestyle='--',alpha=.7,label='direct first recorded')
            else:route_ax.plot([],[],color=color,label=variant+' (no solution)')
        route_ax.scatter(*data.start_xy,c='black',marker='o');route_ax.scatter(*data.goal_xy,c='red',marker='*')
        route_ax.set_aspect('equal');route_ax.set_xlim(data.origin_xy[0],data.origin_xy[0]+data.width_m)
        route_ax.set_ylim(data.origin_xy[1],data.origin_xy[1]+data.height_m);route_ax.set_title(f'{name} seed {seed}: raw paths')
        proposal_ax.set_xlabel('raw proposals');time_ax.set_xlabel('online time (s)')
        time_ax.set_xlim(0,max(r['online_time_s'] for r in runs if (r['scenario'],r['seed'])==(name,seed)))
        for ax in (proposal_ax,time_ax):ax.set_ylabel('current raw J');ax.grid(alpha=.2)
        for ax in axes:ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/f'{name}_seed{seed}.png',dpi=160);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=6000);ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--scenarios',nargs='+',default=['forest','dense','slots']);ap.add_argument('--variants',nargs='+',default=list(VARIANTS));ap.add_argument('--out-dir',type=Path,default=ROOT/'results/direct_batch_region');args=ap.parse_args()
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
                plot_results(runs,out)
    analyse(runs,out)
if __name__=='__main__':main()
