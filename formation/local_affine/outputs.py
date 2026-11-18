"""Save the local tree, continuous segments, execution and visual summary."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from formation.paper_formation.plotting import draw_obstacles


def save(result, output, scene, mode, seed, map_seed, obstacle_count, iterations, dt, exact_output=False):
    out=Path(output) if exact_output else Path(output)/f'{scene}_{mode}_seed{seed}_mapseed{map_seed}_n{obstacle_count}'
    out.mkdir(parents=True,exist_ok=True)
    w=result.world
    segments=[(i,s) for i,p in enumerate(result.plans) for s in p.get('segments',[])]
    hparts=[np.asarray(s['H']) for _,s in segments]
    bparts=[np.asarray(s['b']) for _,s in segments]
    offsets=np.r_[0,np.cumsum([len(h) for h in hparts])]
    extra=({"uav_states":result.uav_states,"uav_controls":result.uav_controls,"uav_altitude":float(result.uav_states[0,0,2])} if result.dynamics=="uav" else {"omega_max":1.5})
    np.savez_compressed(out/'trajectory.npz',dynamics=result.dynamics,**extra,states=result.states,controls=result.controls,
        configurations=result.configurations,reference=result.reference,occupancy=w.truth,
        known_history=np.asarray(result.known_history),resolution=w.res,origin=np.zeros(2),
        dt=dt,times=np.arange(len(result.states))*dt,
        perception_times=np.asarray(result.perception_ticks)*dt,
        perception_period=result.schedule['perception_period'],replan_period=result.schedule['replan_period'],
        robot_radius=.16,start_center=w.start,goal_center=w.goal,
        circles=w.circles,rectangles=w.rectangles,obstacle_centers=w.obstacle_centers,
        v_max=1.5 if result.dynamics=="uav" else .65,a_max=1.,map_seed=map_seed,
        bezier_controls=np.asarray([s['control_points'] for _,s in segments]).reshape(-1,6,6),
        bezier_durations=np.asarray([s['duration'] for _,s in segments]),
        bezier_plan_indices=np.asarray([i for i,_ in segments],dtype=int),
        segment_H=np.concatenate(hparts) if hparts else np.empty((0,6)),
        segment_b=np.concatenate(bparts) if bparts else np.empty(0),segment_offsets=offsets)
    (out/'plans.json').write_text(json.dumps({'plans':result.plans,'execution':result.execution}))
    goal=w.goal+result.reference
    errors=np.linalg.norm(result.states[-1,:,:2]-goal,axis=1)
    metrics={'termination':result.termination,'steps':len(result.controls),'mode':mode,
             'goal_distance':float(np.linalg.norm(result.configurations[-1,:2]-w.goal)),
             'max_robot_goal_error':float(errors.max()),'elapsed_seconds':result.elapsed,
             'timings_seconds':result.timings,'simulation_duration_seconds':len(result.controls)*dt,
             'timing_note':'sfc includes any lazy distance-field and obstacle-index refresh triggered by SFC queries',
             'plans':len(result.plans),'successful_plans':sum(p['status']=='ok' for p in result.plans),
             'tree_iterations':iterations,'seed':seed,'map_seed':map_seed,'obstacle_count':obstacle_count,
             'note':('Analytic Bézier UAV p/v/a with instantaneous jerk, fixed altitude and independent fixed yaw; true occupancy used only by sensor and output.' if result.dynamics=='uav' else 'Numerical polynomial angular-speed checks; true occupancy used only by sensor and output.'),
             'dynamics':result.dynamics, 'v_max':1.5 if result.dynamics=='uav' else .65, 'a_max':1.,
             'reference_speed_m_s':.55}
    metrics['schedule']=result.schedule
    metrics['phase_seconds']={kind:dt*sum(e['kind']==kind for e in result.execution)
                              for kind in ('idle','turn','curve')}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    fig,ax=plt.subplots(figsize=(12,8));draw_obstacles(ax,w.truth,w.res,np.zeros(2),w.circles,w.rectangles)
    for i in range(len(result.reference)):
        p=result.states[:,i,:2];ax.plot(p[:,0],p[:,1],lw=1.1);ax.scatter(*p[-1],s=16)
    ax.scatter(*w.goal,marker='*',s=100,c='red');ax.set_title(f'{mode}: {result.termination}')
    fig.savefig(out/'trajectory.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)
    return out
