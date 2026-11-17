"""Local-map perception, local RRT*, continuous optimization and execution."""
import time
from dataclasses import dataclass
from itertools import combinations
from types import SimpleNamespace
import numpy as np
from .geometry import robot_positions, certify_edge, distance
from .rrt import plan
from .trajectory import TrajectoryOptimizer, matrix
from .goal_time import goal_duration
from formation.paper_formation.config import FormationGeometry, RADIUS
from .world import LocalWorld as World
from formation.paper_formation.sfc import SFCConstructionError


@dataclass
class Result:
    world: World
    reference: np.ndarray
    states: np.ndarray
    configurations: np.ndarray
    controls: np.ndarray
    plans: list
    known_history: list
    execution: list
    termination: str
    elapsed: float
    timings: dict
    schedule: dict
    perception_ticks: list


def _bezier(control, u):
    work=np.asarray(control,dtype=float).copy()
    while len(work)>1:
        work=(1-u)*work[:-1]+u*work[1:]
    return work[0]


class _SegmentEvaluation:
    """Cached Bézier derivative controls and zero-speed endpoint factors."""
    def __init__(self, segment, E=None):
        self.duration=segment['duration']
        self.control=np.asarray(segment['control_points'])
        self.velocity=5*np.diff(self.control,axis=0)
        self.acceleration=4*np.diff(self.velocity,axis=0)
        self.E=E
        if E is not None:
            reduced=np.einsum('nij,kj->kni',E,self.velocity)
            if np.array_equal(self.control[0],self.control[1]) and np.array_equal(self.control[1],self.control[2]):
                for _ in range(2):
                    n=len(reduced)-1
                    reduced=reduced[1:]*np.array([n/(i+1) for i in range(n)])[:,None,None]
            if np.array_equal(self.control[5],self.control[4]) and np.array_equal(self.control[4],self.control[3]):
                for _ in range(2):
                    n=len(reduced)-1
                    reduced=reduced[:-1]*np.array([n/(n-i) for i in range(n)])[:,None,None]
            self.reduced=reduced
            self.reduced_derivative=(len(reduced)-1)*np.diff(reduced,axis=0)

    def evaluate(self,u):
        return (_bezier(self.control,u),_bezier(self.velocity,u)/self.duration,
                _bezier(self.acceleration,u)/self.duration**2)

    def direction_and_omega(self,u):
        v=_bezier(self.reduced,u)
        a=_bezier(self.reduced_derivative,u) if len(self.reduced_derivative) else np.zeros_like(v)
        norm2=np.sum(v*v,axis=1)
        omega=np.zeros(len(v));moving=norm2>1e-24
        omega[moving]=(v[moving,0]*a[moving,1]-v[moving,1]*a[moving,0])/(self.duration*norm2[moving])
        return v,moving,omega


def evaluate(segment, u):
    """Evaluate a saved segment by numerically stable de Casteljau recursion."""
    return _SegmentEvaluation(segment).evaluate(u)


def endpoint_directions(segment, E, start):
    """Find each robot's first nonzero endpoint control-point difference."""
    c=np.asarray(segment['control_points']);point=np.einsum('nij,kj->kni',E,c)
    directions=np.zeros((len(E),2));found=np.zeros(len(E),bool)
    for k in (range(1,6) if start else range(4,-1,-1)):
        candidate=point[k]-point[0] if start else point[-1]-point[k]
        select=(~found)&(np.linalg.norm(candidate,axis=1)>1e-10)
        directions[select]=candidate[select];found|=select
    return directions,found


def allocate_duration(route, E, velocity, acceleration, dt, min_horizon=0., terminal_goal=False, reserve=False):
    """Select one physical time; initial-point failures remain explicit."""
    distances=[max(np.linalg.norm(e@(b-a)) for e in E) for a,b in zip(route[:-1],route[1:])]
    length=max(distances, default=0.)
    if not terminal_goal and not reserve:
        durations=np.maximum(np.asarray(distances)/.55,dt)
        if durations.sum()<min_horizon:
            durations*=min_horizon/durations.sum()
        return np.ceil(durations/dt-1e-12)*dt
    speed=max(np.linalg.norm(e@velocity) for e in E)
    accel=max(np.linalg.norm(e@acceleration) for e in E)
    estimate=max(length/.65, 2*np.sqrt(length), 2*speed, 2*accel,
                 min_horizon/max(1,len(distances)),dt)
    if terminal_goal and speed<1e-10 and accel<1e-10 and len(distances)==1:
        estimate=max(estimate,5*length/.65,np.sqrt(20*length))
    if reserve:
        estimate=max(estimate,3*speed)
    return np.ceil(estimate/dt-1e-12)*dt


def initial_controls_feasible(optimizer, duration):
    """Check fixed P0..P2, including derivative controls, before solving."""
    q=optimizer.route[0];v=optimizer.initial_velocity;a=optimizer.initial_acceleration
    points=np.array([q,q+v*duration/5,q+2*v*duration/5+a*duration**2/20])
    indices,normals=optimizer.certs[0]
    rotation=optimizer.template.rotations[0].value
    for p in points:
        A=p[2:].reshape(2,2);local=rotation.T@A
        if np.linalg.eigvalsh((local+local.T)/2).min()<.42-1e-8 or np.linalg.norm(A,2)>1.35+1e-8 or abs(A[0,1])+abs(A[1,0])>1.5+1e-8:return False
        if optimizer.mode=='uniform' and max(abs(A[0,1]),abs(A[1,0]),abs(A[0,0]-A[1,1]))>1e-8:return False
        for i,e in enumerate(optimizer.E):
            H,b=optimizer.regions[indices[i]]
            if np.any(H@(e@p)>b+1e-8):return False
        for normal,(i,j) in zip(normals,optimizer.pairs):
            if normal@((optimizer.E[i]-optimizer.E[j])@p)<2*RADIUS+.08-1e-8:return False
    velocities=5*np.diff(points,axis=0)/duration
    accelerations=20*np.diff(points,n=2,axis=0)/duration**2
    return all(np.linalg.norm(e@v)<=optimizer.vmax+1e-8 for e in optimizer.E for v in velocities) and all(np.linalg.norm(e@a)<=optimizer.amax+1e-8 for e in optimizer.E for a in accelerations)


def precheck_route(route, regions, reference, E, velocity, acceleration, mode, duration):
    """Check the initial control points without constructing a cone program."""
    cert=certify_edge(route[0],route[1],regions,reference)
    if cert is None:
        return False
    u,_,vh=np.linalg.svd(((route[0][2:]+route[1][2:])/2).reshape(2,2))
    candidate=SimpleNamespace(route=route,initial_velocity=velocity,initial_acceleration=acceleration,
        certs=[cert],template=SimpleNamespace(rotations=[SimpleNamespace(value=u@vh)]),
        mode=mode,regions=regions,E=E,pairs=list(combinations(range(len(E)),2)),vmax=.65,amax=1.)
    return initial_controls_feasible(candidate,duration)


def simulate(scene='random_pillars', mode='affine', formation='triangle6', spacing=3.5,
             obstacle_count=50, map_seed=1, seed=0, iterations=250, max_steps=None,
             dt=.02, replan_steps=None, progress=False, perception_period=.2,
             replan_period=4., max_time=500.):
    if replan_steps is not None:
        replan_period=replan_steps*dt
    if max_steps is not None:
        max_time=max_steps*dt
    if dt<=0 or perception_period<=0 or replan_period<=0 or max_time<=0:
        raise ValueError('Execution step, periods and maximum time must be positive')
    max_steps=int(np.floor(max_time/dt+1e-10))
    started=time.perf_counter();rng=np.random.default_rng(seed)
    geom=FormationGeometry(formation,spacing);ref=geom.centered_reference;E=matrix(ref);n=len(ref)
    world=World(scene,size=(500,400) if scene=='open' else (120,80),
                obstacle_count=obstacle_count,map_seed=map_seed,pillar_width=.3)
    world.start=np.array([5.,20.]);world.goal=np.array([45.,20.])
    q=np.r_[world.start,np.eye(2).ravel()];goal=np.r_[world.goal,np.eye(2).ravel()]
    qv=np.zeros(6);qa=np.zeros(6);yaw=np.zeros(n)
    states=[np.column_stack((robot_positions(q,ref),yaw))];qs=[q.copy()]
    controls=[];plans=[];known=[];execution=[]; active=[];active_index=0;active_time=0.;active_id=-1
    turn_target=None; stuck=0;stalled_since=None;termination='timeout'; evaluators=[]
    terminal_committed=False
    trajectory_templates={}
    next_perception=0.;next_replan=0.;next_trigger='initial';start_curve_pending=False;perception_ticks=[]
    timings={name:0. for name in ('perception','sfc','frontend','backend','execution')}
    for tick in range(max_steps):
        now=tick*dt;current=robot_positions(q,ref)
        if now+1e-10>=next_perception:
            stamp=time.perf_counter();world.observe(current)
            timings['perception']+=time.perf_counter()-stamp
            perception_ticks.append(tick)
            while next_perception<=now+1e-10:next_perception+=perception_period
        # Keep the previously certified tail while the new candidate is tested.
        if (now+1e-10>=next_replan and turn_target is None and
                not start_curve_pending and not terminal_committed):
            t0=time.perf_counter()
            trigger=next_trigger;next_trigger='periodic'
            next_replan=now+replan_period
            try:
                regions=[world.sfc(p) for p in current]
            except SFCConstructionError as exc:
                plans.append({'tick':tick,'time':now,'trigger':trigger,'status':'sfc_failed','error':str(exc)})
                if not active:
                    next_trigger='retry';stuck+=1
                    if stalled_since is None:stalled_since=now
                regions=[]
            sfc_seconds=time.perf_counter()-t0
            timings['sfc']+=sfc_seconds
            if regions:
                stamp=time.perf_counter()
                nodes, route, stats=plan(q,goal,ref,regions,world,rng,mode,iterations)
                frontend_seconds=time.perf_counter()-stamp
                timings['frontend']+=frontend_seconds
                # Prefer goal directly when it is locally connected, then try
                # distinct first edges ordered by the original tree score.
                route_candidates=list(stats.pop('candidate_routes',[]))
                candidates=[([nodes[i].q.copy() for i in branch],branch)
                            for branch in route_candidates]
                if certify_edge(q,goal,regions,ref) is not None:
                    candidates.insert(0,([q.copy(),goal.copy()],[]))
                found=None;attempts=[];stamp=time.perf_counter();precheck_rejected=0
                route_q=[q.copy()];is_goal=False;selected_route=route
                for candidate,branch in candidates:
                    if len(candidate)<2:continue
                    goal_candidate=np.array_equal(candidate[-1],goal)
                    min_horizon=0. if goal_candidate else replan_period+dt
                    if goal_candidate and len(candidate)==2:
                        duration=goal_duration(q,goal,qv,qa,regions,ref,E,dt)
                    else:
                        duration=allocate_duration(candidate,E,qv,qa,dt,min_horizon,
                                                   terminal_goal=goal_candidate)
                    if duration is None:
                        precheck_rejected+=1
                        continue
                    first_duration=float(np.asarray(duration).reshape(-1)[0])
                    if not precheck_route(candidate,regions,ref,E,qv,qa,mode,first_duration):
                        precheck_rejected+=1
                        continue
                    optimizer=TrajectoryOptimizer(candidate,regions,ref,qv,qa,mode,
                                                  template_cache=trajectory_templates,
                                                  terminal_goal=goal_candidate)
                    if optimizer.problem is None:continue
                    route_q=candidate;is_goal=goal_candidate;selected_route=branch
                    found=optimizer.solve(duration)
                    attempts.append({'role':'tracking','duration':np.asarray(duration).tolist(),'route':branch,**optimizer.last_diagnostic})
                    if found is not None or len(attempts)>=3:break
                route=selected_route
                stats['candidate_count']=len(candidates)
                stats['precheck_rejected']=precheck_rejected
                stats['optimized_candidates']=len(attempts)
                stats['solver_calls']=sum(a.get('solver_calls',0) for a in attempts)
                backend_seconds=time.perf_counter()-stamp
                timings['backend']+=backend_seconds
                record={'tick':tick,'time':now,'trigger':trigger,'status':'ok' if found else 'no_trajectory',
                        'terminal_committed':bool(found is not None and np.array_equal(route_q[-1],goal)),
                        'initial_q_velocity':qv.tolist(),'initial_q_acceleration':qa.tolist(),
                        'minimum_nonterminal_horizon_s':replan_period+dt,
                        'known_index':len(known),
                        'statistics':stats,'tree':[{'q':v.q.tolist(),'parent':v.parent,
                        'cost':v.cost,'edge_cost':v.edge_cost} for v in nodes],
                        'route':route,'optimized_route':[v.tolist() for v in route_q],
                        'reference_route':[v.tolist() for v in route_q],
                        'optimized_knots':([found[0]['control_points'][0]]+
                                           [s['control_points'][-1] for s in found]) if found else [],
                        'regions':[{'H':h.tolist(),'b':b.tolist()} for h,b in regions],
                        'attempts':attempts,'segments':found or [],'elapsed_seconds':time.perf_counter()-t0}
                record['timings']={'sfc':sfc_seconds,'frontend':frontend_seconds,'backend':backend_seconds}
                plans.append(record);known.append(world.known.copy())
                if found:
                    active=found;active_index=0;active_time=0.;active_id=len(plans)-1;stuck=0;stalled_since=None
                    terminal_committed=record['terminal_committed']
                    evaluators=[_SegmentEvaluation(s,E) for s in found]
                    # At rest, rotate in place toward the initial nonzero direction.
                    if np.max(np.abs(qv))<1e-8 and np.max(np.abs(qa))<1e-8:
                        vectors,moving_departure=endpoint_directions(found[0],E,True)
                        turn_target=yaw.copy()
                        turn_target[moving_departure]=np.arctan2(vectors[moving_departure,1],vectors[moving_departure,0])
                elif not active:
                    stuck+=1;next_trigger='retry'
                    if stalled_since is None:stalled_since=now
                if progress:
                    print(f'tick={tick} center={q[:2].tolist()} nodes={len(nodes)} route={len(route_q)} status={record["status"]} seconds={record["elapsed_seconds"]:.2f}',flush=True)
        # A nonterminal trajectory may end while moving. If immediate
        # replanning cannot extend it, stop recording rather than fabricate
        # an idle interval or erase its velocity/acceleration.
        if (not active and turn_target is None and
                (np.max(np.abs(qv))>1e-12 or np.max(np.abs(qa))>1e-12)):
            termination='trajectory_exhausted';break
        stamp=time.perf_counter()
        command=np.zeros((n,2));event={'tick':tick,'kind':'idle'}
        if turn_target is not None:
            delta=np.arctan2(np.sin(turn_target-yaw),np.cos(turn_target-yaw))
            change=np.clip(delta,-1.5*dt,1.5*dt);yaw+=change;command[:,1]=change/dt
            event={'tick':tick,'kind':'turn','omega':command[:,1].tolist()}
            if np.max(np.abs(delta-change))<1e-9:
                turn_target=None;start_curve_pending=True
        elif active:
            seg=active[active_index];start=active_time/seg['duration']
            active_time=min(active_time+dt,seg['duration']);end=active_time/seg['duration']
            evaluator=evaluators[active_index]
            q,qv,qa=evaluator.evaluate(end);vel=np.einsum('nij,j->ni',E,qv)
            speed=np.linalg.norm(vel,axis=1);direction,moving,omega=evaluator.direction_and_omega(end)
            yaw[moving]=np.arctan2(direction[moving,1],direction[moving,0]);command[:,0]=speed
            command[moving,1]=omega[moving]
            if start_curve_pending:
                next_replan=now+replan_period;start_curve_pending=False
            if end>=1.-1e-12:
                direction,has_direction=endpoint_directions(seg,E,False)
                stopped=~moving & has_direction
                yaw[stopped]=np.arctan2(direction[stopped,1],direction[stopped,0])
            event={'tick':tick,'kind':'curve','plan':active_id,'segment':active_index,'u0':start,'u1':end}
            if active_time>=seg['duration']-1e-10:
                if active_index==len(active)-1:
                    q=np.asarray(seg['control_points'])[-1].copy()
                    if terminal_committed:
                        # The exact global-goal trajectory ends at rest.
                        qv=np.zeros(6);qa=np.zeros(6)
                        command[:,0]=0.;command[:,1]=0.
                active_index+=1;active_time=0.
                if active_index>=len(active):
                    active=[];active_index=0;terminal_committed=False
                    next_replan=min(next_replan,now+dt);next_trigger='tail_exhausted'
        controls.append(command);states.append(np.column_stack((robot_positions(q,ref),yaw)));qs.append(q.copy());execution.append(event)
        timings['execution']+=time.perf_counter()-stamp
        if (not active and turn_target is None and
                np.max(np.linalg.norm(robot_positions(q,ref)-robot_positions(goal,ref),axis=1))<.1 and
                np.max(np.linalg.norm(np.einsum('nij,j->ni',E,qv),axis=1))<1e-7):
            termination='goal_reached';break
        if stalled_since is not None and now+dt-stalled_since>=32.:
            termination='stagnation';break
    schedule={'execution_dt':dt,'perception_period':perception_period,'replan_period':replan_period,
              'max_time':max_time,'effective_max_time':max_steps*dt,'max_steps':max_steps,
              'stagnation_timeout':32.,'consecutive_failures_at_end':stuck,
              'perception_updates':len(perception_ticks)}
    return Result(world,ref,np.asarray(states),np.asarray(qs),np.asarray(controls),plans,known,execution,termination,time.perf_counter()-started,timings,schedule,perception_ticks)
