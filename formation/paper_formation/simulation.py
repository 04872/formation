import time
from dataclasses import dataclass
import numpy as np
from .config import FormationGeometry, RADIUS, DT, CONFIG
from .world import World
from .formation import fit_q, positions
from .guidance import FormationGuidance
from .mppi import DistributedMPPI, integrate
from .sfc import SFCConstructionError
from .uav_dynamics import integrate_uav
from .uav_mppi import DistributedUAVMPPI, UAVMPPIConfig

@dataclass
class SimulationResult:
    """Recorded trajectories, local maps, planning decisions, and certificates."""
    world: World
    states: np.ndarray
    controls: np.ndarray
    q_history: np.ndarray
    actual_fit_q: np.ndarray
    paths: list
    planning_ticks: list
    assignments: list
    assignment_history: np.ndarray
    known_history: np.ndarray
    planning_seconds: list
    failures: list
    regions: list
    termination: str
    acceleration_controls: np.ndarray
    velocity_history: np.ndarray
    sfc_error: str
    initialization_q: list
    sequence_diagnostics: list
    mppi_diagnostics: np.ndarray
    guidance_weight: float
    smoothness_weight: float
    geometry: FormationGeometry
    dynamics: str = 'differential_drive'
    uav_states: np.ndarray | None = None
    uav_controls: np.ndarray | None = None
    uav_altitude: float | None = None
    timings: dict | None = None
    stage_times: list | None = None
    elapsed_seconds: float = 0.
    replan_period: float = CONFIG.replan_steps * DT
    stagnation_timeout: float | None = None
    execution_dt: float = DT
    backend_period: float = DT
    backend_ticks: list | None = None
    frontend_kind: str = "paper"
    frontend_metadata: list | None = None
    rrt_iterations: int = 250

def simulate(scene: str, mode: str, seed: int, max_steps: int=350, sequences: int=CONFIG.sequences, steps: int=CONFIG.steps, candidates: int=CONFIG.candidates, samples: int=CONFIG.samples, obstacle_count: int=50, map_seed: int=0, horizon: int=CONFIG.horizon, progress: bool=False, guidance_weight: float=CONFIG.guidance_weight, smoothness_weight: float=CONFIG.smoothness_weight, formation: str="square4", formation_spacing: float=1.3, pillar_width: float=1.0, start_center: list[float] | None=None, goal_center: list[float] | None=None, dynamics: str='differential_drive', uav_altitude: float=1.0, replan_period: float | None=None, max_time: float | None=None, stagnation_timeout: float | None=None, execution_dt: float=DT, backend_period: float | None=None, frontend: str='paper', rrt_iterations: int=250) -> SimulationResult:
    """Run perception, shared SFC generation, guidance, and rolling distributed MPPI."""
    simulation_started=time.perf_counter()
    replan_period=CONFIG.replan_steps*DT if replan_period is None else replan_period
    replan_steps=int(round(replan_period/DT))
    backend_period=DT if backend_period is None else backend_period
    backend_steps=int(round(backend_period/DT))
    substeps=int(round(DT/execution_dt))
    if backend_steps<1 or abs(backend_steps*DT-backend_period)>1e-9:
        raise ValueError('backend period must be a positive multiple of the MPPI prediction step')
    if substeps<1 or abs(substeps*execution_dt-DT)>1e-9:
        raise ValueError('execution dt must divide the MPPI prediction step')
    if replan_steps<1 or abs(replan_steps*DT-replan_period)>1e-9:
        raise ValueError('replan period must be a positive multiple of the MPPI update step')
    if max_time is not None:max_steps=int(np.floor(max_time/DT+1e-10))
    stage_totals={k:0. for k in ('perception','sfc','frontend','backend','execution')}
    stage_times=[];stalled_since=None;backend_ticks=[]
    rng = np.random.default_rng(seed)
    geometry = FormationGeometry(formation, formation_spacing)
    REF = geometry.leader_reference
    CENTERED_REF = geometry.centered_reference
    world = World(scene, obstacle_count=obstacle_count, map_seed=map_seed, pillar_width=pillar_width)
    if start_center is not None:
        world.start = np.asarray(start_center, dtype=float)
    if goal_center is not None:
        world.goal = np.asarray(goal_center, dtype=float)
    # Keep the full untransformed formation inside the two-cell border walls.
    border = 2 * world.res + RADIUS
    lower = border - CENTERED_REF.min(axis=0)
    upper = np.asarray(world.size) * world.res - border - CENTERED_REF.max(axis=0)
    if np.any(lower > upper):
        raise ValueError('Formation does not fit within the map boundary')
    world.start = np.clip(world.start, lower, upper)
    world.goal = np.clip(world.goal, lower, upper)
    states = np.c_[world.start + CENTERED_REF, np.zeros(len(REF))]
    leader_goal = world.goal + CENTERED_REF[geometry.leader_slot]
    previous_a = np.eye(2)
    commands = []
    mppi_diagnostics = []
    velocity_history = [np.zeros(len(REF))]
    assignment = np.arange(len(REF))
    history = [states.copy()]
    controls = []
    qs = [fit_q(states, assignment, geometry)]
    paths = []
    planning_ticks = []
    assignments = []
    known_history = []
    planning_seconds = []
    failures = []
    all_regions = []
    initialization_q = []
    sequence_diagnostics = []
    q_ref_history = [qs[0].copy()]
    assignment_history = [assignment.copy()]
    if frontend=='paper':
        planner = FormationGuidance(mode, rng, sequences, steps, candidates, geometry)
    elif frontend=='rrt':
        from formation.local_affine.mppi_frontend import RRTMPPIFrontend
        planner = RRTMPPIFrontend(mode, rng, geometry, rrt_iterations)
    else:
        raise ValueError('frontend must be paper or rrt')
    frontend_metadata=[]
    if dynamics == 'uav':
        uav_state = np.zeros((len(REF), 10))
        uav_state[:, :2] = states[:, :2]
        uav_state[:, 2] = uav_altitude
        uav_history = [uav_state.copy()]
        uav_commands = []
        uav_config = UAVMPPIConfig(dt=DT, guidance_weight=guidance_weight,
                                   smoothness_weight=smoothness_weight,execution_substeps=substeps)
        controller = DistributedUAVMPPI(rng, uav_state, samples, horizon, uav_config)
    else:
        controller = DistributedMPPI(rng, states, samples, horizon, guidance_weight, smoothness_weight)
    path = np.array([qs[-1]])
    stuck = 0
    termination = 'timeout'
    sfc_error = ''
    for tick in range(max_steps):
        stamp=time.perf_counter()
        world.observe(states[:, :2])
        stage_totals['perception']+=time.perf_counter()-stamp
        if tick % replan_steps == 0:
            started = time.perf_counter()
            # q uses fixed reference-slot coordinates. After reassignment the
            # physical leader need not occupy the zero-offset slot.
            center = states[0, :2] - previous_a @ REF[assignment[0]]
            q = np.r_[center, previous_a.ravel()]
            try:
                stamp=time.perf_counter()
                regions = [world.sfc(x, inflate=True) for x in states[:,:2]]
            except SFCConstructionError as exc:
                termination = 'sfc_failed'
                sfc_error = str(exc)
                break
            sfc_seconds=time.perf_counter()-stamp
            stage_totals['sfc']+=sfc_seconds
            stamp=time.perf_counter()
            if frontend=='paper':
                path,seq_diag = planner.plan(q, world, regions, states[:,:2], assignment, leader_goal)
                if len(path):
                    # Section IV-D uses the prior sequence's first shape.
                    previous_a = path[0, 2:].reshape(2, 2).copy()
                reason = 'no_safe_candidate' if len(path)==0 else 'ok'
                assignment = planner.assign(states, path, regions, assignment)
            else:
                path,metadata=planner.plan(states,assignment,world.goal,world,regions)
                frontend_metadata.append(metadata)
                seq_diag=[]
                reason=metadata['status']
            if len(path):target_path=np.array([positions(qi,geometry)[assignment] for qi in path])
            frontend_seconds=time.perf_counter()-stamp
            stage_totals['frontend']+=frontend_seconds
            stage_times.append({'sfc':sfc_seconds,'frontend':frontend_seconds})
            initialization_q.append(metadata['root_leader'] if frontend=='rrt' else q.tolist())
            sequence_diagnostics.append(seq_diag)
            failures.append(reason)
            all_regions.append(regions)
            join_start = states[:, :2].copy()
            paths.append(path.tolist())
            planning_ticks.append(tick*substeps)
            assignments.append(assignment.tolist())
            known_history.append(world.known.copy())
            planning_seconds.append(time.perf_counter() - started)
            stuck = stuck + 1 if len(path) == 0 else 0
            if len(path):stalled_since=None
            elif stalled_since is None:stalled_since=tick*DT
            if progress:
                print(f'plan tick={tick} center={states[:,:2].mean(axis=0).tolist()} path={len(path)} status={reason} seconds={planning_seconds[-1]:.2f}', flush=True)
        if len(path) > 0:
            offset=tick % replan_steps
            indices=np.minimum(np.arange(horizon)+offset,len(target_path)-1)
            guidance=target_path[indices]
            guidance_q=path[indices]
        else:
            guidance=np.repeat(states[None,:,:2],horizon,axis=0)
            guidance_q=np.repeat(q[None,:],horizon,axis=0)
        if dynamics == 'uav':
            guidance3 = np.concatenate((guidance, np.full((*guidance.shape[:2], 1), uav_altitude)), axis=-1)
            stamp=time.perf_counter()
            optimize_now=tick % backend_steps==0
            command = controller.step(uav_state, guidance3, lambda xyz: world.clearance(xyz[..., :2]),optimize_now=optimize_now)
            if optimize_now:backend_ticks.append(tick*substeps)
            stage_totals['backend']+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            for _ in range(substeps):
                uav_state = integrate_uav(uav_state, command, execution_dt)
                states = uav_state[:, [0, 1, 9]]
                uav_history.append(uav_state.copy());uav_commands.append(command.copy())
                commands.append(uav_state[:, 6:9].copy())
                velocity_history.append(np.linalg.norm(uav_state[:, 3:6], axis=1))
                mppi_diagnostics.append(controller.last_diagnostics.copy())
                history.append(states.copy());controls.append(command.copy())
                qs.append(fit_q(states,assignment,geometry))
                q_ref_history.append(guidance_q[0].copy())
                assignment_history.append(assignment.copy())
        else:
            stamp=time.perf_counter()
            velocity=controller.velocities.copy()
            optimize_now=tick % backend_steps==0
            u,command = controller.step(states,guidance,world,optimize_now=optimize_now)
            if optimize_now:backend_ticks.append(tick*substeps)
            stage_totals['backend']+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            # Execute the selected acceleration/yaw-rate for its .2 s interval
            # at the shared actuator step, without running MPPI again.
            for _ in range(substeps):
                next_velocity=velocity+command[:,0]*execution_dt
                execution=np.column_stack(((velocity+next_velocity)/2,command[:,1]))
                states=integrate(states,execution,dt=execution_dt)
                velocity=next_velocity
                commands.append(command.copy());velocity_history.append(velocity.copy())
                controls.append(execution);history.append(states.copy())
                qs.append(fit_q(states,assignment,geometry))
                q_ref_history.append(guidance_q[0].copy())
                assignment_history.append(assignment.copy())
                mppi_diagnostics.append(controller.last_diagnostics.copy())
            controller.velocities=velocity
        stage_totals['execution']+=time.perf_counter()-stamp
        goal_targets = leader_goal + REF[assignment]
        if np.max(np.linalg.norm(states[:, :2] - goal_targets, axis=1)) < 0.4 and np.max(np.abs(velocity_history[-1])) < 0.1:
            termination = 'goal_reached'
            break
        if ((stagnation_timeout is None and stuck>=8) or
                (stagnation_timeout is not None and stalled_since is not None and
                 (tick+1)*DT-stalled_since>=stagnation_timeout)):
            termination = 'stagnation'
            break
    states = np.array(history)
    controls = np.asarray(controls).reshape(-1, len(REF), 4 if dynamics == 'uav' else 2)
    qs = np.array(qs)
    result = SimulationResult(world, states, controls, np.array(q_ref_history), qs, paths, planning_ticks, assignments, np.array(assignment_history), np.array(known_history), planning_seconds, failures, all_regions, termination, np.asarray(commands).reshape(-1, len(REF), 3 if dynamics == 'uav' else 2), np.array(velocity_history), sfc_error, initialization_q, sequence_diagnostics, np.asarray(mppi_diagnostics).reshape(-1,len(REF),2 if dynamics == 'uav' else 6), guidance_weight, smoothness_weight, geometry)
    result.frontend_kind=frontend;result.frontend_metadata=frontend_metadata;result.rrt_iterations=rrt_iterations
    result.dynamics = dynamics
    result.timings=stage_totals;result.stage_times=stage_times
    result.elapsed_seconds=time.perf_counter()-simulation_started
    result.replan_period=replan_period;result.stagnation_timeout=stagnation_timeout
    result.execution_dt=execution_dt;result.backend_period=backend_period;result.backend_ticks=backend_ticks
    if dynamics == 'uav':
        result.uav_states = np.asarray(uav_history)
        result.uav_controls = np.asarray(uav_commands).reshape(-1, len(REF), 4)
        result.uav_altitude = uav_altitude
    return result
