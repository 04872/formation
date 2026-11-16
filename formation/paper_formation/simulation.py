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

def simulate(scene: str, mode: str, seed: int, max_steps: int=350, sequences: int=CONFIG.sequences, steps: int=CONFIG.steps, candidates: int=CONFIG.candidates, samples: int=CONFIG.samples, obstacle_count: int=50, map_seed: int=0, horizon: int=CONFIG.horizon, progress: bool=False, guidance_weight: float=CONFIG.guidance_weight, smoothness_weight: float=CONFIG.smoothness_weight, formation: str="square4", formation_spacing: float=1.3, pillar_width: float=1.0, start_center: list[float] | None=None, goal_center: list[float] | None=None, dynamics: str='differential_drive', uav_altitude: float=1.0) -> SimulationResult:
    """Run perception, shared SFC generation, guidance, and rolling distributed MPPI."""
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
    frontend = FormationGuidance(mode, rng, sequences, steps, candidates, geometry)
    if dynamics == 'uav':
        uav_state = np.zeros((len(REF), 10))
        uav_state[:, :2] = states[:, :2]
        uav_state[:, 2] = uav_altitude
        uav_history = [uav_state.copy()]
        uav_commands = []
        uav_config = UAVMPPIConfig(dt=DT, guidance_weight=guidance_weight,
                                   smoothness_weight=smoothness_weight)
        controller = DistributedUAVMPPI(rng, uav_state, samples, horizon, uav_config)
    else:
        controller = DistributedMPPI(rng, states, samples, horizon, guidance_weight, smoothness_weight)
    path = np.array([qs[-1]])
    stuck = 0
    termination = 'timeout'
    sfc_error = ''
    for tick in range(max_steps):
        world.observe(states[:, :2])
        if tick % CONFIG.replan_steps == 0:
            started = time.perf_counter()
            # q uses fixed reference-slot coordinates. After reassignment the
            # physical leader need not occupy the zero-offset slot.
            center = states[0, :2] - previous_a @ REF[assignment[0]]
            q = np.r_[center, previous_a.ravel()]
            try:
                regions = [world.sfc(x, inflate=True) for x in states[:,:2]]
            except SFCConstructionError as exc:
                termination = 'sfc_failed'
                sfc_error = str(exc)
                break
            path,seq_diag = frontend.plan(q, world, regions, states[:,:2], assignment, leader_goal)
            if len(path):
                # Section IV-D initializes the next plan with the previous
                # sequence's first formation shape, not its last consumed node.
                previous_a = path[0, 2:].reshape(2, 2).copy()
            reason = 'no_safe_candidate' if len(path)==0 else 'ok'
            assignment = frontend.assign(states, path, regions, assignment)
            initialization_q.append(q.tolist())
            sequence_diagnostics.append(seq_diag)
            failures.append(reason)
            all_regions.append(regions)
            join_start = states[:, :2].copy()
            paths.append(path.tolist())
            planning_ticks.append(tick)
            assignments.append(assignment.tolist())
            known_history.append(world.known.copy())
            planning_seconds.append(time.perf_counter() - started)
            stuck = stuck + 1 if len(path) == 0 else 0
            if progress:
                print(f'plan tick={tick} center={states[:,:2].mean(axis=0).tolist()} path={len(path)} status={reason} seconds={planning_seconds[-1]:.2f}', flush=True)
        if len(path) > 0:
            target_path=np.array([positions(qi,geometry)[assignment] for qi in path])
            offset=tick % CONFIG.replan_steps
            indices=np.minimum(np.arange(horizon)+offset,len(target_path)-1)
            guidance=target_path[indices]
            guidance_q=path[indices]
        else:
            guidance=np.repeat(states[None,:,:2],horizon,axis=0)
            guidance_q=np.repeat(q[None,:],horizon,axis=0)
        if dynamics == 'uav':
            guidance3 = np.concatenate((guidance, np.full((*guidance.shape[:2], 1), uav_altitude)), axis=-1)
            command = controller.step(uav_state, guidance3, lambda xyz: world.clearance(xyz[..., :2]))
            uav_state = integrate_uav(uav_state, command, DT)
            states = uav_state[:, [0, 1, 9]]
            uav_history.append(uav_state.copy())
            uav_commands.append(command.copy())
            u = command
            commands.append(uav_state[:, 6:9].copy())
            velocity_history.append(np.linalg.norm(uav_state[:, 3:6], axis=1))
        else:
            u,command = controller.step(states,guidance,world)
            states=integrate(states,u)
            commands.append(command)
            velocity_history.append(controller.velocities.copy())
        mppi_diagnostics.append(controller.last_diagnostics.copy())
        history.append(states.copy())
        controls.append(u)
        qs.append(fit_q(states, assignment, geometry))
        q_ref_history.append(guidance_q[0].copy())
        assignment_history.append(assignment.copy())
        goal_targets = leader_goal + REF[assignment]
        if np.max(np.linalg.norm(states[:, :2] - goal_targets, axis=1)) < 0.4 and np.max(np.abs(velocity_history[-1])) < 0.1:
            termination = 'goal_reached'
            break
        if stuck >= 8:
            termination = 'stagnation'
            break
    states = np.array(history)
    controls = np.asarray(controls).reshape(-1, len(REF), 4 if dynamics == 'uav' else 2)
    qs = np.array(qs)
    result = SimulationResult(world, states, controls, np.array(q_ref_history), qs, paths, planning_ticks, assignments, np.array(assignment_history), np.array(known_history), planning_seconds, failures, all_regions, termination, np.asarray(commands).reshape(-1, len(REF), 3 if dynamics == 'uav' else 2), np.array(velocity_history), sfc_error, initialization_q, sequence_diagnostics, np.asarray(mppi_diagnostics).reshape(-1,len(REF),2 if dynamics == 'uav' else 6), guidance_weight, smoothness_weight, geometry)
    result.dynamics = dynamics
    if dynamics == 'uav':
        result.uav_states = np.asarray(uav_history)
        result.uav_controls = np.asarray(uav_commands).reshape(-1, len(REF), 4)
        result.uav_altitude = uav_altitude
    return result
