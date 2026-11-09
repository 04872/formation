"""Read-only planner replay and offline diagnostics for dense seed 2.

Run from the repository root. Does not change planner implementations or benchmarks.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'results/exit_baselines/diagnostics').mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_coupling import scenario,route_clearance
from formation.exit_compensated_rrtstar import ExitCompensatedConfig,ExitCompensatedRRTStar
from formation.types import Pose2D
import numpy as np
runs=[json.loads(l) for l in open('results/exit_compensation/runs.jsonl')]
r=next(r for r in runs if r['scenario']=='dense' and r['seed']==2 and r['variant']=='exit_local')
rec=next(x for x in r['records'] if x['iteration']==2750)
p=ExitCompensatedRRTStar(*scenario('dense'),ExitCompensatedConfig(**r['config']))
p.initialize()
for i in range(2750):p.step()
seq,cost=p._region_chain()
print('replay',cost,seq==rec['sequence'],len(p.cells),flush=True)
assert seq==rec['sequence'] and abs(cost-rec['raw_cost'])<1e-7
out=dict(original=rec,replay_cost=cost)
# Every present parent tested into the final portion of the chosen chain.
candidates=[]; tried=0
for pos in range(12,len(seq)):
 node=seq[pos]
 for parent in range(len(p.cells)):
  if parent==node or parent==p.parent[node] or p._ancestor(node,parent):continue
  if p._separated(p.cells[parent],p.cells[node]):continue
  z=p._attach(p.entry[parent],p.cells[parent],p.cells[node])
  if z is None or not p._portal_valid(parent,node,z):continue
  tried+=1
  outlets=p._exits(node)
  incoming=p.cost[parent]+p._distance_in(parent,p.entry[parent],z)
  nextq=p.entry[seq[pos+1]] if pos+1<len(seq) else p.goal_endpoints[node]
  oldtail=p.cost[node]+p._distance_in(node,p.entry[node],nextq)
  newcost=cost+incoming+p._distance_in(node,z,nextq)-oldtail
  if newcost>=cost-1e-6:continue
  prefix=[]; cur=parent
  while cur is not None:prefix.append(cur);cur=p.parent[cur]
  newseq=prefix[::-1]+seq[pos:]
  ds=[incoming+p._distance_in(node,z,q)-old for q,old,_ in outlets]
  candidates.append(dict(parent=parent,node=node,pos=pos,raw=newcost,sequence=newseq,max_exit_delta=max(ds) if ds else None,exits=len(outlets),lower_pass=p._can_improve(node,parent,outlets)))
print('tested',tried,'improving',len(candidates),flush=True)
candidates.sort(key=lambda x:x['raw'])
for c in candidates[:30]:
 solved=p.geo.chain(p.start,p.goal_xy,[p.cells[i] for i in c['sequence']],p.config.overlap_margin*(1-1e-6))
 c['chain']=solved[0] if solved else None
 c['route']=[[q.x,q.y,q.yaw] for q in solved[1]] if solved else []
 node,parent=c['node'],c['parent'];exits=p._exits(node)
 z=p._protected_entry(node,parent,exits,'rewire') if exits else None
 c['protected_solution']=z is not None
 c['protected_accepted']=bool(z and p._candidate(node,parent,z,exits))
 c['is_original_chain_parent']=parent in seq
 if c['route']:c['clearance']=route_clearance(p.map_data,p.slots,c['route'])
 print({k:v for k,v in c.items() if k not in ['route','sequence']},flush=True)
out['candidate_count']=len(candidates);out['candidates']=candidates[:30]
out['tail_nodes']=[dict(node=n,parent=p.parent[n],seed=[p.cells[n].pose.x,p.cells[n].pose.y,p.cells[n].pose.yaw],entry=[p.entry[n].x,p.entry[n].y,p.entry[n].yaw],children=sorted(p.children[n])) for n in seq[12:]]
json.dump(out,open('results/exit_baselines/diagnostics/dense_seed02.json','w'),indent=2)
# Offline diagnostic only: can the existing suffix regions support a shorter chain?
# Search seed-distance-weighted overlaps, then optimize the selected chain.
# This is not the planner and does not establish a globally shortest region path.
import heapq
prefix_end=seq.index(24)
allowed=[i for i,c in enumerate(p.cells) if c.pose.x>2.0]
if 24 not in allowed:allowed.append(24)
links={i:[] for i in allowed}; edgecount=0; portals=[]
for ai,i in enumerate(allowed):
 for j in allowed[ai+1:]:
  if p._separated(p.cells[i],p.cells[j]):continue
  portal=p._overlap(p.cells[i],p.cells[j])
  if portal is None:continue
  portals.append((i,j,portal))
  w=p.geo.distance(p.cells[i].pose,p.cells[j].pose)
  links[i].append((j,w));links[j].append((i,w));edgecount+=1
goals=set(p.goal_cells)
dist={24:0.};prev={};queue=[(0.,24)];best=(math.inf,None)
while queue:
 d,i=heapq.heappop(queue)
 if d!=dist[i]:continue
 if i in goals:
  q=p.goal_endpoints[i];v=d+p.geo.distance(p.cells[i].pose,q)
  if v<best[0]:best=(v,i)
 for j,w in links[i]:
  if d+w<dist.get(j,math.inf):
   dist[j]=d+w;prev[j]=i;heapq.heappush(queue,(d+w,j))
suffix=[];i=best[1]
while i is not None:
 suffix.append(i);i=prev.get(i)
suffix.reverse();newseq=seq[:prefix_end]+suffix
solved=p.geo.chain(p.start,p.goal_xy,[p.cells[i] for i in newseq],p.config.overlap_margin*(1-1e-6))
info=dict(nodes=len(allowed),edges=edgecount,prefix_through=24,sequence=newseq,cost=solved[0] if solved else None,route=[[q.x,q.y,q.yaw] for q in solved[1]] if solved else [])
if solved:info['clearance']=route_clearance(p.map_data,p.slots,info['route'])
print('suffix',info,flush=True)
# Audit new suffix links against original tree and birth-time neighbor policy.
info['links']=[]
for a,b in zip(suffix,suffix[1:]):
 late=max(a,b);early=min(a,b)
 ns=p._neighbors(late)
 outlets=p._exits(b)
 lower=p._can_improve(b,a,outlets)
 z=p._protected_entry(b,a,outlets,'rewire') if lower and outlets and not p._ancestor(b,a) else None
 info['links'].append(dict(parent=a,node=b,actual_parent=p.parent[b],in_birth_neighbors=early in ns or p.discovery[late]==early,discovery_parent=p.discovery[late],old_prefix_cost=p.cost[a],desired_chain_position=newseq.index(a),lower_pass=lower,cycle=p._ancestor(b,a),protected_solution=z is not None,protected_accepted=bool(z and p._candidate(b,a,z,outlets))))
out['suffix_diagnostic']=info
json.dump(out,open('results/exit_baselines/diagnostics/dense_seed02.json','w'),indent=2)

# Fixed overlap witnesses furnish a feasible routing diagnostic, not an optimum.
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
points=[p.entry[24]]; memberships={i:[] for i in allowed}; memberships[24].append(0)
for a,b,q in portals:
 k=len(points);points.append(q);memberships[a].append(k);memberships[b].append(k)
goal_ids=[]
for i in allowed:
 if i in goals:
  k=len(points);points.append(p.goal_endpoints[i]);memberships[i].append(k);goal_ids.append(k)
row=[];col=[];values=[];owner={}
for i,ids in memberships.items():
 for ix,u in enumerate(ids):
  for v in ids[ix+1:]:
   w=p._distance_in(i,points[u],points[v])
   row.extend([u,v]);col.extend([v,u]);values.extend([w,w]);owner[u,v]=i;owner[v,u]=i
mat=coo_matrix((values,(row,col)),shape=(len(points),len(points))).tocsr()
d,pr=dijkstra(mat,directed=False,indices=0,return_predecessors=True)
end=min(goal_ids,key=lambda i:d[i]);ids=[];cur=end
while cur>=0:ids.append(cur);cur=int(pr[cur])
ids.reverse()
chain=[24]
for a,b in zip(ids,ids[1:]):
 cell=owner[a,b]
 if cell!=chain[-1]:chain.append(cell)
full=seq[:prefix_end]+chain
solved=p.geo.chain(p.start,p.goal_xy,[p.cells[i] for i in full],p.config.overlap_margin*(1-1e-6))
info=dict(sequence=full,cost=solved[0] if solved else None,route=[[q.x,q.y,q.yaw] for q in solved[1]] if solved else [],witness_suffix=float(d[end]))
if solved:info['clearance']=route_clearance(p.map_data,p.slots,info['route'])
out['portal_diagnostic']=info
print('portal diagnostic',info,flush=True)
json.dump(out,open('results/exit_baselines/diagnostics/dense_seed02.json','w'),indent=2)
# Coverage of the observed BIT* suffix: fixed yaw and any yaw at identical XY.
baseruns=[json.loads(l) for l in open('results/exit_baselines/runs.jsonl')]
br=next(r for r in baseruns if r['scenario']=='dense' and r['seed']==2 and r['variant']=='bitstar')
brec=[x for x in br['records'] if x['online_time_s']<=1.8366601][-1]
print('BIT',brec['iteration'],brec['raw_cost'],flush=True)
coverage=[]
for aa,bb in zip(brec['tree_route'],brec['tree_route'][1:]):
 a=np.array(aa);b=np.array(bb);delta=b-a;delta[2]=(delta[2]+math.pi)%(2*math.pi)-math.pi
 n=max(1,math.ceil(np.linalg.norm(delta[:2])/.05))
 for t in np.linspace(0,1,n+1):
  xyz=a+t*delta
  if not 3.3<=xyz[0]<=8.5:continue
  q=Pose2D(*xyz)
  nearby=[c for c in p.cells if math.hypot(c.pose.x-q.x,c.pose.y-q.y)<=c.guard]
  maxslack=max((p.geo.slack(c,q) for c in nearby),default=-999.)
  anyyaw=any(p.geo.goal_yaw(c,(q.x,q.y),q.yaw,0.) is not None for c in nearby)
  coverage.append(dict(pose=xyz.tolist(),max_slack=maxslack,any_yaw_covered=anyyaw))
out['bit_snapshot']=brec
out['bit_suffix_coverage']=coverage
print('coverage',len(coverage),'fixed yaw uncovered',sum(x['max_slack']<0 for x in coverage),'any yaw uncovered',sum(not x['any_yaw_covered'] for x in coverage),flush=True)
print('any yaw gaps',[x['pose'][:2] for x in coverage if not x['any_yaw_covered']],flush=True)
json.dump(out,open('results/exit_baselines/diagnostics/dense_seed02.json','w'),indent=2)
# Physical feasibility of removing tail vertices without changing retained poses.
raw=rec['tree_route']; dist=[math.inf]*len(raw);dist[12]=0.;prev={};safe_edges=[]
for i in range(12,len(raw)-1):
 for j in range(i+1,len(raw)):
  clear=route_clearance(p.map_data,p.slots,[raw[i],raw[j]])
  if clear<p.config.safety_distance-1e-6:continue
  weight=p.geo.distance(Pose2D(*raw[i]),Pose2D(*raw[j]))
  safe_edges.append(dict(i=i,j=j,clearance=clear))
  if dist[i]+weight<dist[j]:dist[j]=dist[i]+weight;prev[j]=i
ids=[];cur=len(raw)-1
while cur is not None:ids.append(cur);cur=prev.get(cur)
ids.reverse();route=raw[:12]+[raw[i] for i in ids]
cost=sum(p.geo.distance(Pose2D(*a),Pose2D(*b)) for a,b in zip(route,route[1:]))
out['physical_tail_shortcut']=dict(indices=ids,route=route,cost=cost,clearance=route_clearance(p.map_data,p.slots,route),safe_edges=safe_edges)
print('physical shortcut',cost,ids,flush=True)
json.dump(out,open('results/exit_baselines/diagnostics/dense_seed02.json','w'),indent=2)
# Can an unchanged green prefix join the observed BIT suffix by a safe segment?
best=None
brt=brec['tree_route']
for i in range(10,16):
 prefixcost=sum(p.geo.distance(Pose2D(*a),Pose2D(*b)) for a,b in zip(raw[:i+1],raw[1:i+1]))
 for k,(aa,bb) in enumerate(zip(brt,brt[1:])):
  a=np.array(aa);delta=np.array(bb)-a;delta[2]=(delta[2]+math.pi)%(2*math.pi)-math.pi
  for t in np.linspace(0,1,21):
   q=(a+t*delta).tolist()
   if q[0]<raw[i][0] or q[0]>8:continue
   suffix=[raw[i],q]+brt[k+1:]
   value=prefixcost+sum(p.geo.distance(Pose2D(*u),Pose2D(*v)) for u,v in zip(suffix,suffix[1:]))
   if best and value>=best['cost']:continue
   clear=route_clearance(p.map_data,p.slots,[raw[i],q])
   if clear<p.config.safety_distance-1e-6:continue
   rt=raw[:i]+suffix
   best=dict(prefix_index=i,prefix_node=seq[i],cost=value,route=rt,clearance=route_clearance(p.map_data,p.slots,rt))
out['bit_suffix_splice']=best
print('BIT suffix splice', {k:v for k,v in best.items() if k!='route'} if best else None,flush=True)
# Additional cells available only at final 6000 checkpoint, no planner rerun.
finalcells=[p.model.make_cell(Pose2D(*q)) for q in r['cell_poses']]
for x in coverage:
 q=Pose2D(*x['pose'])
 nearby=[c for c in finalcells if math.hypot(c.pose.x-q.x,c.pose.y-q.y)<=c.guard]
 x['final_any_yaw_covered']=any(p.geo.goal_yaw(c,(q.x,q.y),q.yaw,0.) is not None for c in nearby)
print('final any yaw uncovered',sum(not x['final_any_yaw_covered'] for x in coverage),flush=True)
json.dump(out,open('results/exit_baselines/diagnostics/dense_seed02.json','w'),indent=2)

for x in coverage:
 if not x['any_yaw_covered']:
  x['physical_clearance']=route_clearance(p.map_data,p.slots,[x['pose'],x['pose']])
json.dump(out,open('results/exit_baselines/diagnostics/dense_seed02.json','w'),indent=2)
