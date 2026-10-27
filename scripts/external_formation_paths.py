import sys, json, math
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import numpy as np
from compare_coupling import scenario, route_profile, SCENARIOS
from visualize_tube_rrt import straddled_obstacles
from formation.types import Pose2D
OURS='single_entry_rrtstar'
runs=[r for r in map(json.loads,open('results/ablation/runs.jsonl')) if r['variant']==OURS]+[json.loads(l) for l in open('results/external/runs.jsonl')]
best={}
def final_route(r):
    recs=[c for c in [r['first']]+r['checkpoints'] if c and c['success'] and c.get('route')]
    return recs[-1]['route'] if recs else None
rows={}
for name in SCENARIOS:
    m,slots,_=scenario(name)
    for v in (OURS,'rrtstar','bitstar'):
        out=[]
        for r in runs:
            if r['scenario']!=name or r['variant']!=v: continue
            route=final_route(r)
            _,poses,_=route_profile(m,slots,route,step=0.02)
            obs=set(); frac=0
            for p in poses:
                s=straddled_obstacles(Pose2D(*p),slots,m); obs|=set(s); frac+=bool(s)
            th=np.unwrap(poses[:,2]); c,sn=np.cos(th),np.sin(th)
            pts=poses[:,None,:2]+np.stack([c[:,None]*slots[None,:,0]-sn[:,None]*slots[None,:,1], sn[:,None]*slots[None,:,0]+c[:,None]*slots[None,:,1]],-1)
            rob=np.linalg.norm(np.diff(pts,axis=0),axis=-1).sum(0)
            ctr=np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=-1).sum()
            out.append((len(obs),frac/len(poses),ctr,rob.mean(),rob.max(),np.abs(np.diff(th)).sum()))
        a=np.array(out); med=np.median(a,0)
        print(f"{name:7s} {v[:8]:8s} straddled_obs {med[0]:4.0f} (min {a[:,0].min():.0f} max {a[:,0].max():.0f})  straddle_frac {med[1]:.2f}  centre {med[2]:.2f}  robot_mean {med[3]:.2f}  robot_max {med[4]:.2f}  turn {math.degrees(med[5]):6.1f}deg")
