#!/usr/bin/env python3
"""RRT* radius diagnostic: same steering, optional removal of radius cap."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,time,math,json,inspect,textwrap
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import formation.baselines as base
from formation.baselines import RRTStar,BaselineConfig
from compare_coupling import scenario,route_clearance
from compare_exit_baselines import STEPS

class Diagnostic(RRTStar):
    def __init__(self,*args,uncapped=False):
        super().__init__(*args); self.uncapped=uncapped; self.neighbors=[]; self.radii=[]; self.rewires=0

# Preserve upstream step verbatim except the cap and instrumentation points.
src=textwrap.dedent(inspect.getsource(RRTStar.step))
src=src.replace('radius = min(self.gamma * (math.log(self.n + 1) / (self.n + 1)) ** (1.0 / 3.0), self.config.step)', 'radius = self.gamma * (math.log(self.n + 1) / (self.n + 1)) ** (1.0 / 3.0)\n    if not self.uncapped: radius = min(radius, self.config.step)\n    self.radii.append(radius)')
src=src.replace('order = near[np.argsort', 'self.neighbors.append(len(near))\n    order = near[np.argsort')
src=src.replace('self.children[self.parent[k]].remove(k)', 'self.rewires += 1\n            self.children[self.parent[k]].remove(k)')
assert 'self.rewires += 1' in src and 'self.uncapped' in src and 'self.neighbors.append' in src
scope=vars(base).copy(); exec(src,scope); Diagnostic.step=scope['step']

def run(name,seed,uncapped):
 data,slots,start=scenario(name); cfg=BaselineConfig(seed=seed,clearance=.02,edge_resolution=.02,step=STEPS[name],batch_size=200,goal_bias=.1,rewire_factor=1.1)
 t=time.perf_counter(); p=Diagnostic(data,slots,start,cfg,uncapped=uncapped); assert p.check.valid(p.start)[0]
 first=None; snapshots=[]
 for i in range(1,6001):
  p.step(); best=p.best_path()
  if best and first is None: first=dict(iteration=i,time_s=time.perf_counter()-t,cost=best[0])
  if i%50==0:
   snapshots.append(dict(iteration=i,time_s=time.perf_counter()-t,cost=best[0] if best else None))
 elapsed=time.perf_counter()-t; best=p.best_path(); valid=route_clearance(data,slots,best[1],step=.01) if best else None
 children=np.array([len(x) for x in p.children]); invariant_at=time.perf_counter(); errors=[abs(p.cost[i]-p.cost[p.parent[i]]-base.distance(p.Q[p.parent[i]],p.Q[i],p.rho)) for i in range(1,p.n)]; child_links_ok=all(i in p.children[p.parent[i]] for i in range(1,p.n)) and sum(map(len,p.children))==p.n-1; invariant_s=time.perf_counter()-invariant_at; out=dict(invariant_max_cost_error=max(errors,default=0.),child_links_ok=child_links_ok,invariant_check_s=invariant_s,neighbor_observations=len(p.neighbors),scenario=name,seed=seed,uncapped=uncapped,proposals=6000,elapsed_s=elapsed,first=first,final_cost=best[0] if best else None,route=[q.tolist() for q in best[1]] if best else [],clearance=valid,nodes=p.n,gamma=p.gamma,radius_first=p.radii[0],radius_final=p.radii[-1],neighbors_mean=float(np.mean(p.neighbors)),neighbors_median=float(np.median(p.neighbors)),rewires=p.rewires,children_mean=float(children.mean()),children_max=int(children.max()),edges_checked=p.edges_checked,states_checked=p.check.states_checked,snapshots=snapshots)
 print(name,seed,uncapped,'time',round(elapsed,2),'first',first,'final',out['final_cost'],'neighbors',out['neighbors_mean'],'rewires',p.rewires,flush=True)
 return out
if __name__=='__main__':
 out=[]
 for name,seed in [('forest',0),('dense',2)]:
  for uncapped in [False,True]: out.append(run(name,seed,uncapped))
 Path('results/exit_baselines/diagnostics/rrt_radius.json').write_text(json.dumps(out,indent=2))
