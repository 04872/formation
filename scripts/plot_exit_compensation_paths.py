#!/usr/bin/env python3
"""Plot stored raw/chain routes only: no planner execution and no optimization."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/formation-matplotlib')
import argparse
import csv
import html
import json
import math
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_coupling import scenario, ROBOT_RADIUS
VARIANTS=('A','exit_rewire','exit_local')
SCENES=('forest','dense','slots')
COLORS={'A':'#ca4143','exit_rewire':'#287ab8','exit_local':'#24924b'}
LABELS={'A':'A','exit_rewire':'Exit-compensated A','exit_local':'Exit + online entry A'}
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'svg.fonttype':'none'})


def route_data(route,rho):
    a=np.asarray(route,dtype=float)
    if len(a)<2:return dict(poses=a,progress=np.zeros(len(a)),yaw_deg=np.degrees(a[:,2]) if len(a) else [],xy=0.,yaw=0.,dw=0.)
    delta=np.diff(a,axis=0)
    dyaw=(delta[:,2]+math.pi)%(2*math.pi)-math.pi
    xy=np.hypot(delta[:,0],delta[:,1])
    dw=np.hypot(xy,rho*dyaw)
    arclength=np.r_[0.,np.cumsum(dw)]
    yaw=a[0,2]+np.r_[0.,np.cumsum(dyaw)]
    return dict(poses=a,progress=arclength/arclength[-1] if arclength[-1]>0 else arclength,
                yaw_deg=np.degrees(yaw),xy=float(xy.sum()),yaw=float(np.abs(dyaw).sum()),dw=float(dw.sum()))


def sparse_poses(route,rho,n=8):
    d=route_data(route,rho);a=d['poses']
    if len(a)<2:return a
    progress=d['progress']
    return np.column_stack((np.interp(np.linspace(.03,.97,n),progress,a[:,0]),
        np.interp(np.linspace(.03,.97,n),progress,a[:,1]),
        np.interp(np.linspace(.03,.97,n),progress,np.radians(d['yaw_deg']))))


def selections(run):
    good=[q for q in run['records'] if q['raw_success']]
    final=run['records'][-1]
    assert final['iteration']==6000 and final['raw_success']
    first=next((q for q in good if q['kind']=='first'),good[0])
    best=min(good,key=lambda q:q['raw_cost'])
    return dict(first=first,current=final,best=best)


def backdrop(ax,map_data):
    for obstacle in map_data.obstacle_primitives:
        ax.add_patch(Circle(obstacle['center_xy'],obstacle['radius'],facecolor='#9da3aa',edgecolor='#717780',lw=.35,zorder=1))
    x,y=map_data.origin_xy
    ax.set(xlim=(x,x+map_data.width_m),ylim=(y,y+map_data.height_m),aspect='equal')
    ax.plot(*map_data.start_xy,marker='o',mfc='white',mec='#222',ms=5,zorder=8)
    ax.plot(*map_data.goal_xy,marker='*',color='#222',ms=8,zorder=8)
    ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    ax.grid(alpha=.12,lw=.5)


def draw_route(ax,route,**kw):
    a=np.asarray(route)
    if len(a):ax.plot(a[:,0],a[:,1],**kw)


def robots(ax,route,slots,color,rho,n=8):
    for x,y,yaw in sparse_poses(route,rho,n):
        c,s=math.cos(yaw),math.sin(yaw)
        xy=np.array((x,y))+slots@np.array(((c,s),(-s,c)))
        for point in xy:
            ax.add_patch(Circle(point,ROBOT_RADIUS,facecolor=color,edgecolor=color,alpha=.23,lw=.6,zorder=3))
        ax.arrow(x,y,.32*c,.32*s,color=color,alpha=.43,width=.006,head_width=.10,length_includes_head=True,zorder=3)


def legend_handles(include_first=True):
    handles=[Line2D([],[],color='#657080',lw=1.2,ls=':',label='First raw centroid path')] if include_first else []
    handles += [Line2D([],[],color='#555',lw=1.8,label='Final online raw centroid path'),
        Line2D([],[],color='#222',lw=.85,ls='--',label='Stored post-hoc chain path'),
        Patch(facecolor='#bbb',edgecolor='#777',label='Physical obstacle'),
        Patch(facecolor='#81aa92',alpha=.35,label='Robot discs at sparse final raw poses'),
        Line2D([],[],marker='o',mfc='white',mec='#222',ls='',label='Start centroid'),
        Line2D([],[],marker='*',color='#222',ls='',label='Goal centroid')]
    return handles


def map_panel(ax,run,map_data,slots,with_first=True):
    color=COLORS[run['variant']];rho=float(np.linalg.norm(slots,axis=1).max())
    selected=selections(run);rec=selected['current']
    backdrop(ax,map_data)
    if with_first:draw_route(ax,selected['first']['tree_route'],color='#657080',ls=':',lw=1.1,alpha=.8,zorder=4)
    robots(ax,rec['tree_route'],slots,color,rho)
    draw_route(ax,rec['tree_route'],color=color,lw=1.8,zorder=5)
    draw_route(ax,rec['route'],color='#222',ls=(0,(4,3)),lw=.8,alpha=.85,zorder=6)
    raw=route_data(rec['tree_route'],rho)
    ax.set_title(f"{LABELS[run['variant']]}\nRaw J={rec['raw_cost']:.3f}; chain J={rec['optimised_cost']:.3f}; Lxy={raw['xy']:.3f} m",color=color)


def overview(lookup,maps,out):
    fig,axes=plt.subplots(3,3,figsize=(15,13.5))
    for row,name in enumerate(SCENES):
        for col,v in enumerate(VARIANTS):
            map_panel(axes[row,col],lookup[name,v,0],*maps[name],with_first=False)
            if col==0:axes[row,col].set_ylabel(f'{name}\ny [m]')
    fig.suptitle('Fixed seed 0 | Final online raw paths at 6000 raw samples',fontsize=15,y=.99)
    fig.legend(handles=legend_handles(False),loc='lower center',ncol=3,bbox_to_anchor=(.5,.037),frameon=False)
    fig.text(.5,.015,'Lines track the formation centroid; collisions concern the individual robot discs. J=d_W length includes yaw and differs from planar Lxy.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.105,1,.965),h_pad=2.8)
    for ext in ('png','svg'):fig.savefig(out/f'seed0_paths.{ext}',dpi=190)
    plt.close(fig)


def detail(name,seed,group,map_data,slots,out,regression=False):
    fig=plt.figure(figsize=(15,10))
    grid=fig.add_gridspec(3,3,height_ratios=(3.4,1.2,1.25),hspace=.40,wspace=.25)
    rho=float(np.linalg.norm(slots,axis=1).max())
    yaw_axes=[];cost_axes=[]
    for col,v in enumerate(VARIANTS):
        run=group[v];chosen=selections(run);color=COLORS[v]
        ax=fig.add_subplot(grid[0,col]);map_panel(ax,run,map_data,slots)
        ay=fig.add_subplot(grid[1,col]);yaw_axes.append(ay)
        for key,route_key,style,lw in (('first','tree_route',':',1.1),('current','tree_route','-',1.8),('current','route','--',.85)):
            data=route_data(chosen[key][route_key],rho)
            c='#657080' if key=='first' else color if route_key=='tree_route' else '#222'
            ay.plot(data['progress'],data['yaw_deg'],color=c,ls=style,lw=lw)
        ay.set(xlim=(0,1),xlabel='Own normalized d_W arc length',ylabel='Unwrapped yaw [deg]')
        ay.grid(alpha=.2)
        ac=fig.add_subplot(grid[2,col]);cost_axes.append(ac)
        recs=run['records'];raw=np.array([r['raw_cost'] for r in recs])
        raw[~np.array([r['raw_success'] for r in recs])]=np.nan
        x=[r['iteration'] for r in recs]
        ac.step(x,raw,where='post',color=color,lw=1.2,label='Current raw J')
        ac.step(x,np.minimum.accumulate(np.where(np.isfinite(raw),raw,np.inf)),where='post',color=color,ls=':',lw=1.0,label='Best observed raw J')
        ac.set(xlim=(0,6000),xlabel='Raw sample calls',ylabel='Raw J [m in d_W]')
        ac.axvline(4000,color='#999',lw=.7,ls='--');ac.grid(alpha=.2)
        if col==0:ac.legend(loc='upper right',fontsize=8)
    ymin=min(ax.get_ylim()[0] for ax in yaw_axes);ymax=max(ax.get_ylim()[1] for ax in yaw_axes)
    for ax in yaw_axes:ax.set_ylim(ymin,ymax)
    ymin=min(ax.get_ylim()[0] for ax in cost_axes);ymax=max(ax.get_ylim()[1] for ax in cost_axes)
    for ax in cost_axes:ax.set_ylim(ymin,ymax)
    a,b=(selections(group[v])['best']['raw_cost'] for v in ('A','exit_local'))
    suffix=f' | NEGATIVE CASE: combined best raw vs A {100*(b/a-1):+.3f}%' if regression else ''
    fig.suptitle(f'{name} | seed {seed} | Same stored region samples, different online path updates{suffix}',fontsize=13,y=.985)
    fig.legend(handles=legend_handles(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.033),frameon=False,fontsize=8)
    fig.text(.5,.012,'Map paths: first raw (gray dotted), final raw (method color), existing chain (black dashed). Sparse discs show actual robot bodies, not a solid centroid hull.\nJ includes yaw; yaw panels use each path\'s own normalized d_W length. Similar appearance does not establish the same homotopy class.',ha='center',fontsize=8.5)
    fig.subplots_adjust(top=.93,bottom=.14,left=.055,right=.985)
    stem=f'regression_{name}_seed{seed:02d}' if regression else f'{name}_seed{seed:02d}'
    fig.savefig(out/f'{stem}.png',dpi=170)
    if regression:fig.savefig(out/f'{stem}.svg')
    plt.close(fig)
    return stem


def metrics(lookup,maps,out):
    rows=[]
    for name in SCENES:
        slots=maps[name][1];rho=float(np.linalg.norm(slots,axis=1).max())
        for seed in range(10):
            for v in VARIANTS:
                chosen=selections(lookup[name,v,seed])
                for key,rec in chosen.items():
                    baseline=selections(lookup[name,'A',seed])[key]
                    raw=route_data(rec['tree_route'],rho);chain=route_data(rec['route'],rho)
                    rows.append(dict(scenario=name,seed=seed,variant=v,selection=key,iteration=rec['iteration'],
                        raw_cost=rec['raw_cost'],raw_xy_length_m=raw['xy'],raw_abs_yaw_variation_rad=raw['yaw'],
                        raw_abs_yaw_variation_deg=math.degrees(raw['yaw']),raw_recomputed_dw=raw['dw'],
                        chain_cost=rec['optimised_cost'],chain_xy_length_m=chain['xy'],
                        chain_abs_yaw_variation_rad=chain['yaw'],chain_abs_yaw_variation_deg=math.degrees(chain['yaw']),
                        selected_region_sequence_ids=json.dumps(rec['sequence']),
                        same_exact_region_chain_as_A_same_selection=rec['sequence']==baseline['sequence'],
                        same_exact_region_chain_as_own_current=rec['sequence']==chosen['current']['sequence'],
                        raw_change_vs_A_pct=100*(rec['raw_cost']/baseline['raw_cost']-1),
                        route_source=f"runs.jsonl:{name}/{v}/{seed}/iteration={rec['iteration']}/{rec['kind']}",
                        rho_m=rho))
    with (out/'path_metrics.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows


def gallery(lookup,rows,regressions,out):
    table=[]
    negatives={(name,seed) for name,seed in regressions}
    for name in SCENES:
        for seed in range(10):
            chosen={v:selections(lookup[name,v,seed]) for v in VARIANTS}
            vals=[chosen[v]['current']['raw_cost'] for v in VARIANTS]
            paired=[100*(chosen[v]['best']['raw_cost']/chosen['A']['best']['raw_cost']-1) for v in VARIANTS[1:]]
            negative=(name,seed) in negatives
            link=f'{name}_seed{seed:02d}.png'
            mark=f' <strong>negative case</strong> (<a href="regression_{name}_seed{seed:02d}.png">highlight</a>)' if negative else ''
            table.append(f'<tr class="{"negative" if negative else ""}"><td><a href="{link}">{name}, seed {seed}</a>{mark}</td>'+''.join(f'<td>{value:.4f}</td>' for value in vals)+''.join(f'<td>{value:+.3f}%</td>' for value in paired)+'</tr>')
    cards=[]
    for name in SCENES:
        for seed in range(10):
            filename=f'{name}_seed{seed:02d}.png'
            cards.append(f'<details><summary>{name} — seed {seed}</summary><a href="{filename}"><img src="{filename}" loading="lazy" alt="{name} seed {seed} paths, yaw and online cost"></a></details>')
    negative_links='; '.join(f'<a href="regression_{name}_seed{seed:02d}.png">{name}, seed {seed}: explicit regression</a>' for name,seed in regressions)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exit compensation: stored paths</title><style>body{font:15px system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:0 16px;color:#222}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #ddd;padding:6px;text-align:right}th:first-child,td:first-child{text-align:left}.negative{background:#fff0df}summary{cursor:pointer;font-size:18px;padding:10px;background:#eee;margin-top:10px}a{color:#125c97}p{line-height:1.5}.note{background:#f3f5f6;padding:12px}</style><h1>Actual stored paths: A / exit compensation / online entry optimization</h1><p class="note">Selection rule: fixed seed 0 overview, followed by every forest/dense/slots × seeds 0–9 case. All paths come directly from the existing 6000-sample experiment records. No planner or optimizer was run to create these plots. Colored solid lines are final online raw formation-centroid paths; black thin dashed lines are the already-recorded post-hoc chain paths. Robot discs show physical robot bodies at sparse poses. The centroid and a solid formation hull are not collision bodies. Gray discs are physical obstacles.</p><p>J is d_W arc length with yaw scaled by formation radius rho; J differs from planar path length Lxy. Yaw plots show shortest-angle-continuous orientation against each route's own normalized d_W arc length. A similar spatial appearance is not a homotopy classification. The identical-region-chain flags in the CSV compare exact ordered region IDs only.</p><h2>Fixed seed 0</h2><a href="seed0_paths.png"><img src="seed0_paths.png" alt="Seed 0 three-scene by three-method overview"></a><p><a href="seed0_paths.svg">Overview SVG</a> · <a href="path_metrics.csv">All per-path metrics CSV</a> · <a href="selection.json">Selection/provenance metadata</a></p><h2>Negative result retained</h2><p>'''+negative_links+'''</p><h2>All 30 paired cases</h2><p>Raw J columns use the current path at 6000 samples. Percentage columns compare each method's best observed raw J against A's best observed raw J within the same scene and seed; negative is better. These are individual paired cases, not ratios of scene medians.</p><table><thead><tr><th>Case / detail</th><th>A final raw J</th><th>Exit final raw J</th><th>Combined final raw J</th><th>Exit best vs A</th><th>Combined best vs A</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table><h2>Path, yaw and raw-cost details</h2>'''+''.join(cards)+'''</html>'''
    (out/'index.html').write_text(page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=ROOT/'results/exit_compensation/runs.jsonl')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/exit_compensation/paths')
    args=parser.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    runs=[json.loads(line) for line in args.runs.read_text().splitlines()]
    lookup={(r['scenario'],r['variant'],r['seed']):r for r in runs}
    expected={(name,v,seed) for name in SCENES for v in VARIANTS for seed in range(10)}
    assert expected.issubset(lookup) and all(lookup[k]['sample_calls']==6000 for k in expected)
    maps={name:scenario(name)[:2] for name in SCENES}
    overview(lookup,maps,out)
    for name in SCENES:
        for seed in range(10):
            group={v:lookup[name,v,seed] for v in VARIANTS}
            detail(name,seed,group,*maps[name],out)
    regressions=[]
    for seed in range(10):
        a,b=(selections(lookup['slots',v,seed])['best']['raw_cost'] for v in ('A','exit_local'))
        if b>a+1e-8:
            regressions.append(('slots',seed))
            detail('slots',seed,{v:lookup['slots',v,seed] for v in VARIANTS},*maps['slots'],out,regression=True)
    assert len(regressions)==1,regressions
    rows=metrics(lookup,maps,out);gallery(lookup,rows,regressions,out)
    max_cost_error=max(abs(row['raw_cost']-row['raw_recomputed_dw']) for row in rows)
    (out/'selection.json').write_text(json.dumps(dict(source=str(args.runs),overview='fixed seed0; final current raw at6000, existing chain same record',
        all_cases='forest,dense,slots seeds0-9; no omitted cases',raw_run_count=90,detail_case_count=30,metric_rows=len(rows),
        regression_rule='slots combined best observed raw > A best observed raw + 1e-8',regressions=regressions,
        max_recorded_vs_route_dw_error=max_cost_error,robot_radius_m=ROBOT_RADIUS,
        provenance='Only read stored tree_route/route/sequence; scenario() rebuilds static maps. No planning or optimization.',
        metric_definition='XY length; sum absolute shortest-angle yaw changes; dW segment sqrt(dx^2+dy^2+(rho*dyaw)^2). Exact chain compares ordered region ids.'),indent=2))
    shutil.copyfile(__file__,out/Path(__file__).name)
    print(json.dumps(dict(out=str(out),case_count=30,regressions=regressions,metric_rows=len(rows),max_dw_error=max_cost_error)))

if __name__=='__main__':main()
