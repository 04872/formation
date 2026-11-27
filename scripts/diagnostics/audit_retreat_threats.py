#!/usr/bin/env python3
"""Magnitude-only retrospective audit of initial-H competitors on saved trees."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from formation.local_affine.geometry import robot_positions
run=ROOT/'results/three_routes/persistent_learning_retained_500s/rrt_affine/seed_0'
p=json.loads((run/'planning.json').read_text()); z=np.load(run/'trajectory.npz'); ref=z['centered_reference']
rounds=[]; gen=-1
for t,m in enumerate(p['frontend_metadata']):
 d=m['statistics']['history_decision']
 if t==0 or d.get('capacity_reset') or d.get('map_reset') or d.get('goal_reset'): gen+=1
 tree=m['tree']; root=robot_positions(np.array(tree[0]['q']),ref)
 cs=[]
 for c in d['candidates']:
  pos=robot_positions(np.array(tree[c['endpoint']]['q']),ref); vec=(pos-root).ravel(); norm=np.linalg.norm(vec)
  cs.append(dict(c=c,pos=pos,vec=vec,norm=norm,retreat=float(tree[0]['q'][0]-tree[c['endpoint']]['q'][0])))
 rounds.append(dict(t=t,gen=gen,root=root,cs=cs,tree=tree))
def desc(a):
 a=np.asarray(a,dtype=float)
 return dict(count=len(a),min=float(np.min(a)),median=float(np.median(a)),max=float(np.max(a))) if len(a) else dict(count=0)
def run_audit(radius,cosine):
 out=[]
 for row in rounds[400:500]:
  ret=[x for x in row['cs'] if x['retreat']>=.5]
  if not ret: continue
  best=min(ret,key=lambda x:x['c']['score']); bs=best['c']['score']
  threats=[x for x in row['cs'] if x['retreat']<.5 and x['c']['value_source']=='initial' and x['c']['score']<bs]
  old=[(r,x) for r in rounds[:row['t']-4] if r['gen']==row['gen'] for x in r['cs'] if x['c']['value_source']=='learned' and x['c']['write_round'] is not None]
  hist=[]
  if old:
   roots=np.array([r['root'] for r,x in old]); ends=np.array([x['pos'] for r,x in old]); vecs=np.array([x['vec'] for r,x in old]); norms=np.array([x['norm'] for r,x in old]);
   rd=np.sqrt(np.mean(np.sum((roots-row['root'])**2,axis=2),axis=1))
  for x in threats:
   ed=np.sqrt(np.mean(np.sum((ends-x['pos'])**2,axis=2),axis=1)) if old else np.array([])
   co=vecs@x['vec']/(norms*x['norm']) if old and x['norm'] else np.array([])
   inds=np.flatnonzero((rd<=radius)&(ed<=radius)&(co>=cosine)) if len(co) else []
   matches=[(old[i][1]['c']['value']-old[i][1]['c']['h'],i) for i in inds]
   maxpair=max(matches,default=None)
   gap=float(bs-x['c']['score']); rec=dict(endpoint=x['c']['endpoint'],score=x['c']['score'],retreat_m=x['retreat'],required_increase=gap,match_count=len(matches),max_old_correction=float(maxpair[0]) if maxpair else None,covered=bool(maxpair and maxpair[0]>=gap))
   if maxpair:
    v,i=maxpair; rec['max_match']=dict(round=old[i][0]['t'],endpoint=old[i][1]['c']['endpoint'],write_round=old[i][1]['c']['write_round'],root_rms_m=float(rd[i]),endpoint_rms_m=float(ed[i]),direction_cos=float(co[i]))
   hist.append(rec)
  out.append(dict(round=row['t'],generation=row['gen'],best_retreat_endpoint=best['c']['endpoint'],best_retreat_score=bs,best_retreat_m=best['retreat'],threats=hist,uniform_required=max((a['required_increase'] for a in hist),default=0),all_threats_covered=bool(hist) and all(a['covered'] for a in hist)))
 non=[r for r in out if r['threats']]; threats=[x for r in non for x in r['threats']]
 return dict(thresholds=dict(root_and_endpoint_rms_m=radius,direction_cos_min=cosine,minimum_round_lag=5),rounds_with_retreat=len(out),rounds_with_threat=len(non),threat_count=len(threats),individual_gap=desc([x['required_increase'] for x in threats]),uniform_gap=desc([r['uniform_required'] for r in non]),with_similar=sum(x['match_count']>0 for x in threats),without_similar=sum(x['match_count']==0 for x in threats),covered_threats=sum(x['covered'] for x in threats),rounds_all_threats_covered=sum(r['all_threats_covered'] for r in non),rounds=out)
a,b=rounds[490],rounds[495]; ca=min(a['cs'],key=lambda x:x['c']['score']); cb=min(b['cs'],key=lambda x:x['c']['score']); delta=ca['c']['value']-ca['c']['h']; rescored=[(x['c']['score']+(delta if x is cb else 0),x) for x in b['cs']]; score,w=min(rescored,key=lambda x:x[0]);
def choose(cs,scores):
 best=cs[0]
 for x in cs[1:]:
  a,b=x['c'],best['c']; sa,sb=scores[a['endpoint']],scores[b['endpoint']]
  if sa<sb or (sa==sb and len(a['route'])>len(b['route']) and a['route'][:len(b['route'])]==b['route']): best=x
 return best
def info(row,x,score):
 c=x['c']; return dict(round=row['t'],endpoint=c['endpoint'],score=score,route=c['route'],first_edge=c['route'][1],first_edge_q=row['tree'][c['route'][1]]['q'],endpoint_q=row['tree'][c['endpoint']]['q'],retreat_m=x['retreat'],correction=c['value']-c['h'])
result=dict(definition='Retrospective magnitude check, not an algorithm or certificate. Initial-H non-retreat candidates strictly outranking best actual-score >=0.5 m nominal-center-x retreat candidate. Similarity uses corresponding member embeddings and flattened root-to-endpoint direction. Similar historical candidate had learned valid value at observed earlier round; old labels may no longer be valid. Equality reaches tie, strict change needs epsilon. Other candidate scores frozen in single-candidate counterfactual. Coverage of all initial threats does not cover other learned competitors.',main=run_audit(.4,.8),loose=run_audit(.6,.7),strict=run_audit(.3,.9),counterfactual_490_to_495=dict(old_winner=info(a,ca,ca['c']['score']),actual_winner=info(b,cb,cb['c']['score']),transferred_correction=delta,rescored_old_winner=cb['c']['score']+delta,new_winner=info(b,w,score)))
cf=[]
for rec in result['main']['rounds']:
 row=rounds[rec['round']]; scores={x['c']['endpoint']:x['c']['score'] for x in row['cs']}; actual=choose(row['cs'],scores)
 for threat in rec['threats']:
  if threat['max_old_correction'] is not None: scores[threat['endpoint']]+=threat['max_old_correction']
 winner=choose(row['cs'],scores)
 cf.append(dict(round=row['t'],actual_endpoint=actual['c']['endpoint'],actual_first_edge=actual['c']['route'][1],actual_retreat_m=actual['retreat'],endpoint=winner['c']['endpoint'],first_edge=winner['c']['route'][1],retreat_m=winner['retreat'],score=scores[winner['c']['endpoint']]))
result['simultaneous_transfer_counterfactual']=dict(definition='Add max similar old correction to every matched initial threat simultaneously; freeze all other candidate scores, apply production exact-tie prefix rule. Magnitude-only, not a validated transfer or closed-loop result.',rounds=len(cf),first_edge_changed=sum(x['first_edge']!=x['actual_first_edge'] for x in cf),retreat_at_least_0_5=sum(x['retreat_m']>=.5 for x in cf),retreat_at_least_0_25=sum(x['retreat_m']>=.25 for x in cf),records=cf)
out=ROOT/'results/three_routes/persistent_learning_retained_replay/retreat_threat_audit.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
for k in ['main','loose','strict']:
 print(k,{key:value for key,value in result[k].items() if key!='rounds'})
print('counterfactual',json.dumps(result['counterfactual_490_to_495']))
print('simultaneous',{k:v for k,v in result['simultaneous_transfer_counterfactual'].items() if k!='records'})
print(out)
