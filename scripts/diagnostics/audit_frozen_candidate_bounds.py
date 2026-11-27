#!/usr/bin/env python3
"""Score sensitivity using final saved-replay finite-action bounds; no writeback."""
import json, math
from pathlib import Path
R=Path(__file__).resolve().parents[2]
run=R/'results/three_routes/persistent_learning_retained_500s/rrt_affine/seed_0'
p=json.loads((run/'planning.json').read_text()); d=json.loads((run/'revisit_replay_diagnostic.json').read_text())
bounds={(v['round'],v['endpoint']):v for v in d['final_candidate_bounds']}
def choose(cs,scores):
 best=cs[0]
 for c in cs[1:]:
  if scores[c['endpoint']]<scores[best['endpoint']] or (scores[c['endpoint']]==scores[best['endpoint']] and len(c['route'])>len(best['route']) and c['route'][:len(best['route'])]==best['route']): best=c
 return best
rows=[]; errors=[]; score_max=0.; compared=0
for rep in d['replay']:
 t=rep['round']; old=p['frontend_metadata'][t]['statistics']['history_decision']; cs=rep['candidates']; oc=old['candidates']
 if len(cs)!=len(oc): errors.append([t,'count',len(cs),len(oc)]); continue
 for a,b in zip(cs,oc):
  for k in ('endpoint','route','value_source'):
   if a[k]!=b[k]:errors.append([t,k,a[k],b[k]])
  for k in ('tree_cost','h','value','score'):
   err=abs(a[k]-b[k]);score_max=max(score_max,err);compared+=1
   if not math.isclose(a[k],b[k],rel_tol=1e-10,abs_tol=1e-10):errors.append([t,k,a[k],b[k]])
 score={c['endpoint']:c['score'] for c in cs}; actual=choose(cs,score)
 if actual['endpoint']!=old['selected_endpoint'] or actual['endpoint']!=rep['selected_endpoint']:errors.append([t,'winner',actual['endpoint'],old['selected_endpoint'],rep['selected_endpoint']])
 if t<400:continue
 alt=score.copy(); eligible=[]
 for c in cs:
  x=bounds[(t,c['endpoint'])];L,U=x['L'],x['U']
  if c['value_source']=='initial' and L is not None and U is not None and L>c['h']:
   alt[c['endpoint']]=c['tree_cost']+L; eligible.append(c['endpoint'])
 new=choose(cs,alt); tree=p['frontend_metadata'][t]['tree']; root=tree[0]['q'][0]
 def info(c,s):return dict(endpoint=c['endpoint'],first_edge=c['route'][1],route=c['route'],score=s[c['endpoint']],source=c['value_source'],retreat_m=root-tree[c['endpoint']]['q'][0])
 retreats=[c for c in cs if root-tree[c['endpoint']]['q'][0]>=.5];best_retreat=choose(retreats,score) if retreats else None
 alt_retreat=choose(retreats,alt) if retreats else None
 rows.append(dict(round=t,actual=info(actual,score),sensitivity=info(new,alt),eligible_endpoints=eligible,winner_eligible=actual['endpoint'] in eligible,best_retreat=info(best_retreat,score) if best_retreat else None,best_retreat_sensitivity=info(alt_retreat,alt) if alt_retreat else None))
if errors:raise RuntimeError(f'replay mismatch {len(errors)}: {errors[:10]}')
def summary(rs):
 return dict(rounds=len(rs),winner_eligible=sum(r['winner_eligible'] for r in rs),any_eligible=sum(bool(r['eligible_endpoints']) for r in rs),endpoint_changed=sum(r['actual']['endpoint']!=r['sensitivity']['endpoint'] for r in rs),first_edge_changed=sum(r['actual']['first_edge']!=r['sensitivity']['first_edge'] for r in rs),actual_retreat_ge_05=sum(r['actual']['retreat_m']>=.5 for r in rs),sensitivity_retreat_ge_05=sum(r['sensitivity']['retreat_m']>=.5 for r in rs),actual_retreat_ge_025=sum(r['actual']['retreat_m']>=.25 for r in rs),sensitivity_retreat_ge_025=sum(r['sensitivity']['retreat_m']>=.25 for r in rs))
initial=[r for r in rows if r['actual']['source']=='initial']; r495=next(r for r in rows if r['round']==495)
result=dict(definition='Recorded-tree replay, not closed loop. Freeze final successor values and edge/pending actions; learned candidate scores fixed. For initial candidates replace H by L only when L and U are finite and L>H; do not write back. U finite is conservative sampling restriction. L is lower bound for the finite current action model, not an actual remaining-path lower bound. Exact score tie prefers longer route only when earlier route is its prefix.',verification=dict(replay_rounds=len(d['replay']),first_round=d['replay'][0]['round'],last_round=d['replay'][-1]['round'],compared_numeric_fields=compared,max_absolute_numeric_error=score_max,mismatch_count=len(errors)),all_rounds=summary(rows),initial_winners=summary(initial),round_495=dict(**r495,winner_bounds=bounds[(495,r495['actual']['endpoint'])]),records=rows)
out=run/'frozen_candidate_bounds_sensitivity.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for k in ('verification','all_rounds','initial_winners','round_495'):print(k,result[k])
