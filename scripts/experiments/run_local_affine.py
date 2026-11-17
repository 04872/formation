#!/usr/bin/env python3
"""Run a local RRT* and centralized Bézier differential-drive experiment."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from formation.local_affine.simulation import simulate
from formation.local_affine.outputs import save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=['open','random_pillars'],default='random_pillars')
    p.add_argument('--mode',choices=['uniform','affine'],default='affine')
    p.add_argument('--formation',choices=['triangle6','square4'],default='triangle6')
    p.add_argument('--spacing',type=float,default=3.5)
    p.add_argument('--obstacle-count',type=int,default=50)
    p.add_argument('--map-seed',type=int,default=1);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--iterations',type=int,default=250)
    p.add_argument('--execution-dt',type=float,default=.02)
    p.add_argument('--perception-period',type=float,default=.2)
    p.add_argument('--replan-period',type=float,default=4.)
    p.add_argument('--max-time',type=float,default=500.)
    p.add_argument('--max-steps',type=int,default=None,help='Legacy override: max time = max steps * execution dt')
    p.add_argument('--replan-steps',type=int,default=None,help='Legacy override: replan period = replan steps * execution dt')
    p.add_argument('--progress',action='store_true')
    p.add_argument('--output',default='results/local_affine')
    a=p.parse_args();dt=a.execution_dt
    result=simulate(a.scene,a.mode,a.formation,a.spacing,a.obstacle_count,a.map_seed,a.seed,
                    a.iterations,a.max_steps,dt,a.replan_steps,a.progress,
                    a.perception_period,a.replan_period,a.max_time)
    out=save(result,a.output,a.scene,a.mode,a.seed,a.map_seed,a.obstacle_count,a.iterations,dt)
    (out/'configuration.json').write_text(json.dumps({'requested':vars(a),'effective_schedule':result.schedule},indent=2)+'\n')


if __name__=='__main__':
    main()
