#!/usr/bin/env python3
"""Replay recorded U-scene trees from generation reset 394, with map/SFC prewarm.
Recorded trees imply an open-loop replay, not an independently simulated rollout.
"""
import json, pickle, sys, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from formation.local_affine.history_persistent import PersistentHistoryDecision
from formation.local_affine.geometry import robot_positions
from formation.local_affine.history_joint_geometry import evaluate_pairs
from formation.paper_formation.world import World
run=ROOT/'results/three_routes/persistent_learning_retained_500s/rrt_affine/seed_0'
out=run/'revisit_replay_diagnostic.json'
p=json.loads((run/'planning.json').read_text()); traj=np.load(run/'trajectory.npz'); reg=np.load(run/'regions.npz'); ref=traj['centered_reference']; known_history=traj['known_history']; coeff=reg['coefficients']; offsets=reg['offsets']; saved=p['frontend_metadata']
def regions(t):
 return [(v[:,:2],v[:,2]) for i in range(t*6,t*6+6) for v in [coeff[offsets[i]:offsets[i+1]]]]
def world(t):
 w=World('history_deadend',obstacle_count=3,map_seed=1,pillar_width=.3);w.known=known_history[t].copy();w.occupied=w.known & w.truth;return w
def winner(t):
 h=saved[t]['statistics']['history_decision'];return next(c for c in h['candidates'] if c['endpoint']==h['selected_endpoint'])
def pos(q):return robot_positions(np.asarray(q),ref).reshape(-1)/np.sqrt(len(ref))
def num(x):return float(x) if np.isfinite(x) else None
cohort={}
poscache=[[pos(r['q']) for r in s['tree']] for s in saved]
for t in range(394,500):
 gen=394 if t<470 else 470
 for s in range(gen,t-4):
  old=winner(s)
  if old['value_source']!='learned':continue
  qo=np.array(saved[s]['tree'][old['endpoint']]['q']); po=poscache[s][old['endpoint']]
  now=saved[t]['statistics']['history_decision']['candidates']; best=min(now,key=lambda c:np.linalg.norm(poscache[t][c['endpoint']]-po))
  qn=np.array(saved[t]['tree'][best['endpoint']]['q']); dist=np.linalg.norm(poscache[t][best['endpoint']]-po)
  if dist>.4:continue
  ro=np.array(saved[s]['tree'][0]['q']);rn=np.array(saved[t]['tree'][0]['q']); u=po-pos(ro);v=pos(qn)-pos(rn)
  cohort.setdefault(t,[]).append(dict(old_round=s,round=t,old_endpoint=old['endpoint'],old_vertex=old['vertex'],new_endpoint=best['endpoint'],new_vertex=best['vertex'],distance=float(dist),roots_distance=float(np.linalg.norm(pos(ro)-pos(rn))),old_root_to_old=float(np.linalg.norm(u)),new_root_to_new=float(np.linalg.norm(v)),direction_cos=float(u@v/(np.linalg.norm(u)*np.linalg.norm(v))),old_q=qo.tolist(),new_q=qn.tolist(),old_root=ro.tolist(),new_root=rn.tolist(),old_value=old['value'],old_score=old['score']))
print('cohort built',sum(map(len,cohort.values())),flush=True)
m=PersistentHistoryDecision(ref)
for t in range(394):
 m._update_map(world(t),regions(t))
 if t%50==0: print('prewarm',t,flush=True)
m._reset_graph();m.round_id=394
# select normally triggers capacity reset at 394: preserve its effect on cert_version
orig_update=m._update_map
state={'t':None}; events=[]; replay=[]; analyses=[]; final_bounds=[]
def update(w,r):
 ret=orig_update(w,r)
 if state['t']==394:m._reset_graph();m._last_world=w
 return ret
m._update_map=update
orig_ingest=m._ingest;orig_disc=m._discover;orig_backup=m.backup_vertex;orig_inv=m._invalidate;orig_cert=m._certify_pending

def snap(i):
 if i>=len(m.qs):return None
 return {'label':m.labels[i], 'h':m.h[i], 'value':m.value(i),'witness':m.witness.get(i),'write_round':m.write_round[i]}
def ingest(nodes,routes,stats,w):
 mp=orig_ingest(nodes,routes,stats,w); state['mapping']=mp;return mp
def disc(ids):
 if state['t'] in cohort:
  st=state['pairs']; posall=robot_positions(np.asarray(m.qs),ref).reshape(len(m.qs),-1)/np.sqrt(len(ref));tree=cKDTree(posall);k=min(len(posall),m.max_neighbors+1)
  for a in st:
   x=state['mapping'][a['new_endpoint']];z=a['old_vertex']; ds,ns=tree.query(posall[x],k=k,distance_upper_bound=m.epsilon);a['knn']=bool(z in np.asarray(ns));a['old_after_ingest']=snap(z)
 ret=orig_disc(ids)
 if state['t'] in cohort:
  for a in state['pairs']:
   x=state['mapping'][a['new_endpoint']];z=a['old_vertex'];a['old_after_discover']=snap(z);a['after_discover_action']='edge' if z in m.edges[x] else 'pending' if z in m.pending[x] else 'absent'
 return ret
def inv(starts,exclude=()):
 before=snap(1719) if state['t']==495 else None
 ret=orig_inv(starts,exclude)
 if state['t']==495:
  after=snap(1719)
  if before and before['label'] is not None and after['label'] is None:events.append({'event':'invalidate_1719','starts':list(starts),'exclude':list(exclude),'before':before,'after':after,'current_backup':state.get('backup')})
 return ret
def backup(x):
 if state['t'] in cohort:
  for a in state['pairs']:
   if state['mapping'][a['new_endpoint']]!=x or 'at_backup' in a:continue
   z=a['old_vertex'];a['old_at_backup']=snap(z);a['at_backup_action']='edge' if z in m.edges[x] else 'pending' if z in m.pending[x] else 'absent';a['attempted']=m._attempted.get((x,z))==m._cert_version;a['too_expensive']=m._too_expensive.get((x,z))==m._cert_version
   edges=[cost+m.value(j) for j,(cost,_) in m.edges[x].items() if np.isfinite(m.value(j))];pending=[d+m.value(j) for j,d in m.pending[x].items() if np.isfinite(m.value(j))];U=min(edges,default=np.inf);L=min([U]+pending)
   a['bounds_before']={'L':num(L),'U':num(U),'pending_count':len(pending)}
   safe,cost=evaluate_pairs(m.qs,[(x,z)],ref,m._last_world,m.regions);cost=float(cost[0]);a['offline']={'certified':bool(safe[0]),'cost':num(cost),'within_cost_limit':bool(safe[0] and cost<=m.epsilon)}
   if safe[0] and cost<=m.epsilon:
    U2=min(edges+[cost+m.value(z)],default=np.inf);pend2=[d+m.value(j) for j,d in m.pending[x].items() if j!=z and np.isfinite(m.value(j))];L2=min([U2]+pend2);resolved=np.isfinite(U2) and L2>=U2-1e-10
    a['forced_one']={'L':num(L2),'U':num(U2),'resolved':bool(resolved),'value_if_resolved':num(U2) if resolved else None,'old_successor_value':m.value(z),'candidate_effective_before':m.value(x),'would_cycle':bool(m._cycle(x,z)) if cost+m.value(z)<=U2+1e-10 else False}
   a['at_backup']=True
 before=snap(1719) if state['t']==495 else None; state['backup']=x
 result=orig_backup(x)
 if state['t']==495:
  after=snap(1719)
  if x==1719 or (before and after and before!=after):events.append({'event':'backup','x':x,'result':result,'before_1719':before,'after_1719':after})
 state['backup']=None
 return result
m._ingest=ingest;m._discover=disc;m._invalidate=inv;m.backup_vertex=backup
for t in range(394,500):
 with open(f'/tmp/retained500_rebuilt/{t:03}.pkl','rb') as f:d=pickle.load(f)
 state={'t':t,'pairs':cohort.get(t,[])}
 for a in state['pairs']:a['old_at_entry']=snap(a['old_vertex'])
 _,info=m.select(d['nodes'],d['stats'],d['goal'],d['world'],d['regions']);old=d['old']
 dif=[]
 for a,b in zip(info['candidates'],old['candidates']):
  for key in ['endpoint','route','vertex','tree_cost','value_source','h','value','score']:
   av,bv=a[key],b[key]
   if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
    if not np.isclose(av,bv,rtol=1e-10,atol=1e-10):dif.append([a['endpoint'],key,av,bv])
   elif av!=bv:dif.append([a['endpoint'],key,av,bv])
 dimensions={k:[info[k],old[k]] for k in ['selected_endpoint','vertices','edges','pending','queue','checks','accepted_links','history_regions'] if info[k]!=old[k]}
 replay.append({'round':t,'candidate_count':[len(info['candidates']),len(old['candidates'])],'differences':dif,'graph_differences':dimensions,'selected_endpoint':info['selected_endpoint'],'selected_route':info['selected_route'],'candidates':info['candidates']})
 for c in info['candidates']:
  x=c['vertex']; edges=[cost+m.value(z) for z,(cost,_) in m.edges[x].items() if np.isfinite(m.value(z))]; pending=[d+m.value(z) for z,d in m.pending[x].items() if np.isfinite(m.value(z))];U=min(edges,default=np.inf);L=min([U]+pending);final_bounds.append({'round':t,'endpoint':c['endpoint'],'vertex':x,'H':c['h'],'value':c['value'],'source':c['value_source'],'L':num(L),'U':num(U),'pending':len(pending),'edges':len(edges)})
 for a in state['pairs']:
  z=a['old_vertex'];x=state['mapping'][a['new_endpoint']];a['old_at_end']=snap(z);a['end_action']='edge' if z in m.edges[x] else 'pending' if z in m.pending[x] else 'absent';a['candidate_final']=next(c for c in info['candidates'] if c['endpoint']==a['new_endpoint']);a['actual_winner']=info['selected_endpoint']
  if 'forced_one' in a and a['forced_one']['resolved'] and not a['forced_one']['would_cycle']:
   candidates=[dict(c) for c in info['candidates']];c=next(c for c in candidates if c['endpoint']==a['new_endpoint']);c['score']=c['tree_cost']+a['forced_one']['U'];best=min(candidates,key=lambda c:c['score']);a['forced_one']['winner_if_candidate_substitution']=best['endpoint'];a['forced_one']['score_substituted']=c['score'];q0=d['nodes'][0].q;qb=d['nodes'][best['endpoint']].q;goal=d['goal'];a['forced_one']['center_retreat']=bool(q0[0]-qb[0]>=.5)
 analyses.extend(state['pairs'])
 print(t,'diff',len(dif),dimensions,'cohort',len(state['pairs']),flush=True)
out.write_text(json.dumps({'definition':'same generation, gap>=5, old recorded winner learned, closest current eligible endpoint member RMS<=0.4; recorded trees replay','replay':replay,'pairs':analyses,'round495_events':events,'final_candidate_bounds':final_bounds},indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')
