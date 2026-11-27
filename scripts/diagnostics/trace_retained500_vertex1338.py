#!/usr/bin/env python3
"""Trace recorded generation 470, isolate first invalidation of vertex 1338."""
import json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from formation.local_affine.history_persistent import PersistentHistoryDecision
from formation.paper_formation.world import World
run=ROOT/'results/three_routes/persistent_learning_retained_500s/rrt_affine/seed_0'
p=json.loads((run/'planning.json').read_text());tr=np.load(run/'trajectory.npz');rg=np.load(run/'regions.npz');ref=tr['centered_reference'];known=tr['known_history'];coef=rg['coefficients'];off=rg['offsets']
def regions(t):return [(v[:,:2],v[:,2]) for i in range(t*6,t*6+6) for v in [coef[off[i]:off[i+1]]]]
def world(t):
 w=World('history_deadend',obstacle_count=3,map_seed=1,pillar_width=.3);w.known=known[t].copy();w.occupied=w.known & w.truth;return w
m=PersistentHistoryDecision(ref)
for t in range(470):
 m._update_map(world(t),regions(t))
 if t%100==0:print('prewarm',t,flush=True)
m._reset_graph();m.round_id=470
state={'t':470,'context':[]};events=[];audit=[]
def snap(x):
 if x>=len(m.qs):return None
 return dict(vertex=x,label=m.labels[x],value=m.value(x),h=m.h[x],witness=m.witness.get(x),write_round=m.write_round[x],provenance=m.provenance[x])
def norm(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,set):return list(x)
 raise TypeError(type(x).__name__)
def push(kind,data,fn,*args,**kw):
 state['context'].append({'kind':kind,**data})
 try:return fn(*args,**kw)
 finally:state['context'].pop()
update=m._update_map
def upd(w,r):
 result=update(w,r)
 if state['t']==470:m._reset_graph();m._last_world=w
 return result
m._update_map=upd
original_inv=m._invalidate;original_backup=m.backup_vertex;original_consider=m._consider_action;original_add_edge=m.add_edge;original_add_pending=m.add_pending

def invalidate(starts,exclude=()):
 starts=list(starts);exclude=list(exclude);before=snap(1338)
 # capture direct witnesses and incoming bound that can propagate invalidation
 details=[]
 if before and before['label'] is not None:
  for x in starts:
   if x<len(m.qs):
    edge=m.edges[1338].get(x);pending=m.pending[1338].get(x)
    details.append({'start':snap(x),'is_target':x==1338,'is_target_witness':m.witness.get(1338)==x,'target_edge':edge,'target_pending_lower':pending,'edge_plus_value':edge[0]+m.value(x) if edge else None,'pending_plus_value':pending+m.value(x) if pending is not None else None})
 result=original_inv(starts,exclude);after=snap(1338)
 if state['t']>=490 and before and before['label'] is not None and after['label'] is None:events.append({'type':'invalidated','round':state['t'],'before':before,'after':after,'starts':starts,'exclude':exclude,'start_details':details,'context':list(state['context'])})
 return result
def consider(a,b,lower_bound,witness_changed=False):
 context={'a':a,'b':b,'lower':float(lower_bound),'a_before':snap(a),'b_before':snap(b),'witness_changed':bool(witness_changed)}
 return push('consider_action',context,original_consider,a,b,lower_bound,witness_changed)
def edge(a,b,cost,source='search'):
 return push('add_edge',{'a':a,'b':b,'cost':float(cost),'source':source},original_add_edge,a,b,cost,source)
def pending(a,b,lower):
 return push('add_pending',{'a':a,'b':b,'lower':float(lower)},original_add_pending,a,b,lower)
def backup(x):
 before=snap(x);before_target=snap(1338)
 result=push('backup',{'x':x,'before':before},original_backup,x)
 after_target=snap(1338)
 if state['t']>=490 and (x==1338 or (before_target and before_target['witness']==x)):
  events.append({'type':'backup_relevant','round':state['t'],'x':x,'before':before,'after':snap(x),'target_before':before_target,'target_after':after_target,'result':result})
 return result
m._invalidate=invalidate;m._consider_action=consider;m.add_edge=edge;m.add_pending=pending;m.backup_vertex=backup
for t in range(470,496):
 state['t']=t
 with open(f'/tmp/retained500_rebuilt/{t:03}.pkl','rb') as f:d=pickle.load(f)
 if t>=490:events.append({'type':'round_entry','round':t,'target':snap(1338)})
 _,info=m.select(d['nodes'],d['stats'],d['goal'],d['world'],d['regions']);old=d['old'];diff=[]
 for a,b in zip(info['candidates'],old['candidates']):
  for key in ('endpoint','route','vertex','tree_cost','h','value','value_source','score'):
   x,y=a[key],b[key]
   if isinstance(x,(int,float)) and isinstance(y,(int,float)):
    if not np.isclose(x,y,atol=1e-10,rtol=1e-10):diff.append([a['endpoint'],key,x,y])
   elif x!=y:diff.append([a['endpoint'],key,x,y])
 dims={k:[info[k],old[k]] for k in ('selected_endpoint','vertices','edges','pending','queue','checks','accepted_links','history_regions') if info[k]!=old[k]}
 audit.append({'round':t,'differences':diff,'graph_differences':dims})
 if t>=490:events.append({'type':'round_end','round':t,'target':snap(1338)})
 print(t,'diff',len(diff),dims,flush=True)
(run/'vertex1338_invalidation_trace.json').write_text(json.dumps({'description':'Recorded trees, prewarmed observed history, capacity generation 470 replay; not a new closed-loop rollout','audit':audit,'events':events},indent=2,default=norm,allow_nan=False)+'\n')
