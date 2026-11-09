"""Offline certificate probe on the saved splice; not an online planner."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, json, math, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_coupling import scenario,route_clearance
from formation.exit_compensated_rrtstar import ExitCompensatedConfig,ExitCompensatedRRTStar
from formation.types import Pose2D,wrap_to_pi
import numpy as np
D=json.load(open(ROOT/'results/exit_baselines/diagnostics/dense_seed02.json'))
r=next(x for x in map(json.loads,open(ROOT/'results/exit_compensation/runs.jsonl')) if x['scenario']=='dense' and x['seed']==2 and x['variant']=='exit_local')
p=ExitCompensatedRRTStar(*scenario('dense'),ExitCompensatedConfig(**r['config']))
route=D['bit_suffix_splice']['route']; prefix=D['original']['sequence'][:12]
prefixcells=[p.model.make_cell(Pose2D(*r['cell_poses'][i])) for i in prefix]
margin=p.config.overlap_margin
out={'margin':margin,'chain_solver_margin':margin+1e-7,'source_cost':D['bit_suffix_splice']['cost'],'physical_clearance':D['bit_suffix_splice']['clearance'],'results':[]}
for step in [0.1,0.05,0.02]:
 t0=time.perf_counter(); poses=[]; locations=[]
 for j,(a,b) in enumerate(zip(route[11:],route[12:])):
  a=np.array(a); delta=np.array(b)-a; delta[2]=wrap_to_pi(delta[2]); length=math.sqrt(delta[0]**2+delta[1]**2+(p.rho*delta[2])**2)
  n=math.ceil(length/step)
  for k in range(0 if j==0 else 1,n+1):
   q=a+delta*k/n; q[2]=wrap_to_pi(q[2]); poses.append(Pose2D(*q));locations.append([j,k/n])
 cells=[p.model.make_cell(q) for q in poses]
 # Forward DAG, connection from existing prefix region. Search fewest inserted cells.
 allcells=[prefixcells[-1]]+cells; N=len(allcells); dist=[10**9]*N;prev=[None]*N;dist[0]=0; checks=0
 for i in range(N):
  if dist[i]==10**9: continue
  for j in range(i+1,N):
   if dist[j]<=dist[i]+1:continue
   if p._separated(allcells[i],allcells[j]):continue
   checks+=1; q=p._overlap(allcells[i],allcells[j])
   if q is not None and min(p.geo.slack(allcells[i],q),p.geo.slack(allcells[j],q))>=margin-1e-9:
    dist[j]=dist[i]+1;prev[j]=i
 # last seed goal must support margin at goal
 goalok=p.geo.goal_yaw(cells[-1],p.goal_xy,cells[-1].pose.yaw,margin) is not None
 reach=max(i for i in range(N) if dist[i]<10**9)
 ids=[];x=N-1 if dist[-1]<10**9 and goalok else reach
 while x:ids.append(x-1);x=prev[x]
 ids.reverse()
 chain=prefixcells+[cells[i] for i in ids]
 solved=p.geo.chain(p.start,p.goal_xy,chain,margin+1e-7) if ids and reach==N-1 and goalok else None
 portal_slacks = [min(p.geo.slack(chain[k],q),p.geo.slack(chain[k+1],q)) for k,q in enumerate(solved[1][1:-1])] if solved else []
 goal_slack=p.geo.slack(chain[-1],solved[1][-1]) if solved else None
 lifted=[chain[0].pose.yaw]
 for c in chain[1:]:lifted.append(lifted[-1]+wrap_to_pi(c.pose.yaw-lifted[-1]))
 chart_yaws=([lifted[0]+wrap_to_pi(solved[1][0].yaw-lifted[0])] + [lifted[k]+wrap_to_pi(q.yaw-lifted[k]) for k,q in enumerate(solved[1][1:-1])] + [lifted[-1]+wrap_to_pi(solved[1][-1].yaw-lifted[-1])]) if solved else []
 seam_ok=all(abs((b-a)-wrap_to_pi(b-a))<1e-7 for a,b in zip(chart_yaws,chart_yaws[1:])) if solved else None
 info={'min_portal_slack':min(portal_slacks) if portal_slacks else None,'goal_slack':goal_slack,'start_slack':p.geo.slack(chain[0],p.start) if solved else None,'chart_short_segments':seam_ok,'optimized_route':[[q.x,q.y,q.yaw] for q in solved[1]] if solved else [],'step_dw':step,'samples':len(cells),'valid':sum(c.valid for c in cells),'min_seed_clearance':min(c.clearance for c in cells),'min_seed_slack':min(p.geo.slack(c,c.pose) for c in cells),'reachable_last_index':reach-1,'reachable_last_location':locations[reach-1] if reach else None,'reachable_last_pose':list(vars(poses[reach-1]).values()) if reach else None,'last_reachable':dist[-1]<10**9,'goal_margin_ok':goalok,'selected_count':len(ids),'selected_is_complete':reach==N-1 and goalok,'selected_locations':[locations[i] for i in ids],'selected_seed_poses':[[poses[i].x,poses[i].y,poses[i].yaw] for i in ids],'chain_cost':solved[0] if solved else None,'chain_clearance':route_clearance(p.map_data,p.slots,[[q.x,q.y,q.yaw] for q in solved[1]]) if solved else None,'overlap_checks':checks,'elapsed_s':time.perf_counter()-t0}
 out['results'].append(info); print(json.dumps(info),flush=True)
json.dump(out,open(ROOT/'results/exit_baselines/diagnostics/detour_certificate.json','w'),indent=2)
