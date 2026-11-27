#!/usr/bin/env python3
"""Inspect recorded U-scene execution and subsequent selectable tree endpoints.

No replanning. Slots are fixed by the recorded identity assignment. A tree endpoint
is a candidate, not an executed motion or a connection to the next tree.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/three_routes/local_backup_closed_loop_500s'
OUTPUT = BASE/'u_split_support.json'

def pos(q, ref):
    q = np.asarray(q)
    return q[..., None, :2] + np.einsum('...ij,nj->...ni', q[..., 2:].reshape(*q.shape[:-1],2,2), ref)

def runs(flags):
    lens=[]; n=0
    for v in flags:
        if v: n+=1
        elif n: lens.append(n); n=0
    if n: lens.append(n)
    return lens

def calc(depth, settings):
    folder=BASE/f'depth{depth}/rrt_affine/seed_0'
    with (folder/'planning.json').open() as f: plan=json.load(f)
    with np.load(folder/'trajectory.npz') as d:
        states=d['states'][:,:, :2].copy(); ass=d['assignment_history'].copy()
        dt=float(d['dt']); ref=d['centered_reference'].copy()
    assert np.all(ass == np.arange(6)), 'analysis assumes identity assignment'
    ticks=plan['ticks']; ts=np.asarray(ticks)*dt
    xy=states[np.asarray(ticks)]
    frames=plan['frontend_metadata']
    events=[]
    for t in range(2,len(frames)-1):
        # All groups are measured in executed robot positions; north of inflated
        # wall and inside wall's x range respectively, with explicit margin.
        p=xy[t]; outer=np.flatnonzero(p[:,1]>=26.16); inner=np.flatnonzero((p[:,0]>=14.16)&(p[:,1]<=25.34))
        if not len(outer) or not len(inner): continue
        increments=np.diff(xy[t-2:t+1,outer,0],axis=0).mean(axis=1)
        if sum(increments) < settings['outer_dx'] or min(increments)<-0.05: continue
        event={'round':t,'time_s':float(ts[t]),'inner':inner.tolist(),'outer':outer.tolist(),
               'outer_dx_2s':float(sum(increments)),'later':[]}
        for j in range(t+1,min(t+6,len(frames))):
            fm=frames[j]; tree=fm.get('tree',[]); dec=fm.get('statistics',{}).get('history_decision',{})
            candidates=dec.get('candidates',[])
            if not tree:
                event['later'].append({'round':j,'tree_missing':True}); continue
            q=np.asarray([v['q'] for v in tree]); pts=pos(q,ref); root=pts[0]
            d0=np.linalg.norm(root[inner]-[14.,26.],axis=1)
            rec=[]
            for c in candidates:
                k=c['endpoint']; z=pts[k]; dist=np.linalg.norm(z[inner]-[14.,26.],axis=1)
                progress=d0-dist
                stay=bool(np.all(z[outer,1]>=26.16-settings['outside_slack']) and np.all(z[outer,0]-root[outer,0]>=-settings['retreat']))
                near=bool(np.any((progress>=settings['approach_gain']) & (dist<=settings['near_radius'])))
                crossed=z[inner,1]>=26.16
                if stay and (near or crossed.any()):
                    delta=z[outer]-root[outer]
                    rec.append({'endpoint':k,'route':c['route'],'approach':near,'cross_any':bool(crossed.any()),'cross_all':bool(crossed.all()),
                        'inner_best_distance_gain':float(progress.max()),'outer_min_dx':float(delta[:,0].min()),'outer_min_y':float(z[outer,1].min()),
                        'shape_change':float(np.linalg.norm(q[k,2:]-q[0,2:])), 'center_delta':(q[k,:2]-q[0,:2]).tolist(),
                        'score':float(c['score'])})
            counts={k:sum(x[k] for x in rec) for k in ('approach','cross_any','cross_all')}
            # Distinct mechanisms are directly measured, not inferred from endpoint x alone.
            styles={'outer_retreat':sum(x['outer_min_dx'] < -0.1 for x in rec),
                    'deformation':sum(x['shape_change'] >= 0.1 for x in rec),
                    'northward':sum(x['center_delta'][1]>=0.1 for x in rec),
                    'westward':sum(x['center_delta'][0]<=-0.1 for x in rec)}
            examples=[]
            for key in ('approach','cross_any','cross_all'):
                hits=[x for x in rec if x[key]]
                if hits: examples.append({'type':key,**min(hits,key=lambda x:x['score'])})
            event['later'].append({'round':j,'time_s':float(ts[j]),'root_source':fm.get('root_source'),
                'candidate_total':len(candidates),'counts':counts,'styles':styles,'examples':examples})
        events.append(event)
    n=len(events); stats={}
    for kind in ('approach','cross_any','cross_all'):
        flags=[[r.get('counts',{}).get(kind,0)>0 for r in e['later']] for e in events]
        stats[kind]={'events_with_any':sum(any(f) for f in flags),'events_with_all_available':sum(len(f)==5 and all(f) for f in flags),
          'round_1_to_5_hits':[sum(len(f)>i and f[i] for f in flags) for i in range(5)],
          'round_1_to_5_denominators':[sum(len(f)>i for f in flags) for i in range(5)],
          'consecutive_from_first_histogram':dict(Counter(str(next((i for i,x in enumerate(f) if not x),len(f))) for f in flags)),
          'max_consecutive_histogram':dict(Counter(str(max(runs(f),default=0)) for f in flags)),
          'events_with_any_gap':sum(any(not x for x in f) for f in flags)}
    return {'depth':depth,'event_count':n,'settings':settings,'stats':stats,'events':events}

def short_checks(result):
    # Offline retrospective demonstration: candidate endpoint at t+1 joined
    # to a directed original edge of the t+2 tree with t+2 available SFCs.
    sys.path.insert(0, str(ROOT))
    from formation.local_affine.history_joint_geometry import evaluate_pairs
    from formation.paper_formation.world import World
    checks=[]
    for depth in (1,2):
        folder=BASE/f'depth{depth}/rrt_affine/seed_0'
        plan=json.loads((folder/'planning.json').read_text())
        with np.load(folder/'trajectory.npz') as z:
            ref=z['centered_reference'].copy(); known=z['known_history'].copy(); truth=z['occupancy'].copy()
        with np.load(folder/'regions.npz') as z:
            coeff=z['coefficients'].copy(); offsets=z['offsets'].copy()
        events=result['runs']['main'][str(depth)]['events']
        for target in (260,400):
            eligible=[e for e in events if len(e['later'])>=2 and any(x['type']=='approach' for x in e['later'][0].get('examples',[]))]
            if not eligible: continue
            e=min(eligible,key=lambda x:abs(x['round']-target)); first=e['later'][0]; j=first['round']; later=j+1
            cand=next(x for x in first['examples'] if x['type']=='approach'); q0=np.asarray(plan['frontend_metadata'][j]['tree'][cand['endpoint']]['q'])
            tree=plan['frontend_metadata'][later]['tree']; q=np.vstack([q0,np.asarray([v['q'] for v in tree])]); points=pos(q,ref)
            emb=np.linalg.norm((points[1:]-points[0]).reshape(len(tree),-1),axis=1)/np.sqrt(len(ref))
            ids=np.flatnonzero(emb<=.4)
            regions=[]; seen=set()
            for k in range((later+1)*6):
                v=coeff[offsets[k]:offsets[k+1]]; key=v.tobytes()
                if key not in seen: seen.add(key); regions.append((v[:,:2],v[:,2]))
            world=World('history_deadend',obstacle_count=3,map_seed=1,pillar_width=.3)
            assert np.array_equal(world.truth,truth)
            world.known=known[later]; world.occupied=world.known & world.truth
            pairs=np.array([[0,int(i)+1] for i in ids],dtype=int).reshape(-1,2)
            safe,cost=evaluate_pairs(q,pairs,ref,world,regions)
            good=ids[safe & (cost<=.4)]
            inner=np.array(e['inner']); outer=np.array(e['outer']); d0=np.linalg.norm(points[0,inner]-[14.,26.],axis=1)
            witnesses=[]
            for node in good:
                for end in range(1,len(tree)):
                    path=[]; cur=end
                    while cur>=0 and cur!=node:
                        path.append(cur); cur=tree[cur]['parent']
                    if cur!=node: continue
                    endp=points[end+1]; begin=points[int(node)+1]
                    move=np.linalg.norm((endp-begin).ravel())/np.sqrt(len(ref))
                    dist=np.linalg.norm(endp[inner]-[14.,26.],axis=1)
                    join_dist=np.linalg.norm(begin[inner]-[14.,26.],axis=1)
                    join_delta=begin[outer]-points[0,outer]
                    suffix_delta=endp[outer]-begin[outer]
                    if move>=.4 and np.any((join_dist-dist>=.15)&(dist<=3.)) and np.all(begin[outer,1]>=26.) and np.all(endp[outer,1]>=26.) and np.all(join_delta[:,0]>=-.5) and np.all(suffix_delta[:,0]>=-.5) and np.all(endp[outer,0]-points[0,outer,0]>=-.5):
                        index=int(np.flatnonzero(ids==node)[0]); witnesses.append({'join_node':int(node),'end_node':end,'link_cost':float(cost[index]),'suffix_rms_displacement':float(move),'suffix_inner_distance_gain_max':float(np.max(join_dist-dist)),'link_outer_min_dx':float(join_delta[:,0].min()),'suffix_outer_min_dx':float(suffix_delta[:,0].min()),'total_outer_min_dx':float((endp[outer,0]-points[0,outer,0]).min()),'directed_node_ids':[int(node)]+list(reversed(path))}); break
                if witnesses: break
            checks.append({'depth':depth,'event_round':e['round'],'candidate_round':j,'candidate_endpoint':cand['endpoint'],'later_tree_round':later,
              'candidate_from_actual_root_source':first['root_source'],'nearby_nodes':len(ids),'certified_links_cost_le_0_4':len(good),'witness':witnesses[0] if witnesses else None,
              'semantics':'Retrospective connection from an unexecuted candidate to a later tree generated on actual trajectory; uses SFCs available through later round, directed original suffix, RMS suffix displacement >=0.4 m and inner approach gain >=0.15 m on suffix alone; outer x retreat <=0.5 m separately on link, suffix and total and y >=26.0 at endpoints. Future tree was unavailable to original selection; failure to connect is only failure of this finite search and certificate.'})
    return checks

if __name__=='__main__':
    result={'definition':{'time':'one replan per second; records end at 499 s',
      'events':'At t, actual outer robots y>=26.16 and inner robots x>=14.16,y<=25.34; outer mean x increment t-2 to t >= threshold and neither one-second increment less than -0.05 m. Overlapping events retained.',
      'candidate':'Selectable endpoint from recorded history_decision candidates, whose path starts at actual recorded planner root; group IDs fixed at event. Outer endpoint y>=26.16-outside_slack, x retreat no more than retreat m relative to that tree root. Approach means some inner member decreases Euclidean distance to (14,26) by approach_gain and endpoint is within near_radius. Crossing means inner endpoint y>=26.16, any/all separately; no x constraint.',
      'limits':'Endpoint counts are not continuous cross-round paths, not proof of execution, not exhaustive continuous free space. Root may be previous reference instead of fitted state. These trees are from the actual recorded execution; future trees are not online information.'},'runs':{}}
    for name,settings in [('main',dict(outer_dx=.15,outside_slack=.16,retreat=.5,approach_gain=.15,near_radius=3.0)),
       ('loose',dict(outer_dx=.10,outside_slack=.3,retreat=1.,approach_gain=.10,near_radius=4.0)),
       ('strict',dict(outer_dx=.25,outside_slack=.0,retreat=.25,approach_gain=.25,near_radius=2.0))]:
        result['runs'][name]={str(depth):calc(depth,settings) for depth in (1,2)}
    result['short_cross_tree_checks'] = short_checks(result)
    OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False))
    for name,depths in result['runs'].items():
        for depth,r in depths.items(): print(name,depth,'events',r['event_count'], 'stats',r['stats'])
