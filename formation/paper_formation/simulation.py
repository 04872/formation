import time
from dataclasses import dataclass
import numpy as np
from .config import REF as CENTERED_REF, LEADER_REF as REF, RADIUS, DT, CONFIG
from .world import World
from .formation import fit_q, positions
from .guidance import FormationGuidance
from .mppi import DistributedMPPI, integrate
from .sfc import SFCConstructionError

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

def simulate(scene: str, mode: str, seed: int, max_steps: int=350, sequences: int=CONFIG.sequences, steps: int=CONFIG.steps, candidates: int=CONFIG.candidates, samples: int=CONFIG.samples, obstacle_count: int=50, map_seed: int=0, horizon: int=CONFIG.horizon, progress: bool=False) -> SimulationResult:
    """Run perception, shared SFC generation, guidance, and rolling distributed MPPI."""
    rng = np.random.default_rng(seed)
    world = World(scene, obstacle_count=obstacle_count, map_seed=map_seed)
    states = np.c_[world.start + CENTERED_REF, np.zeros(len(REF))]
    leader_goal = world.goal + CENTERED_REF[0]
    previous_a = np.eye(2)
    commands = []
    velocity_history = [np.zeros(len(REF))]
    assignment = np.arange(len(REF))
    history = [states.copy()]
    controls = []
    qs = [fit_q(states, assignment)]
    paths = []
    planning_ticks = []
    assignments = []
    known_history = []
    planning_seconds = []
    failures = []
    all_regions = []
    q_ref_history = [qs[0].copy()]
    assignment_history = [assignment.copy()]
    frontend = FormationGuidance(mode, rng, sequences, steps, candidates)
    controller = DistributedMPPI(rng, states, samples, horizon)
    path = np.array([qs[-1]])
    stuck = 0
    termination = 'timeout'
    sfc_error = ''
    for tick in range(max_steps):
        world.observe(states[:, :2])
        if tick % CONFIG.replan_steps == 0:
            started = time.perf_counter()
            q = np.r_[states[0,:2], previous_a.ravel()]
            try:
                regions = [world.sfc(x, inflate=True) for x in states[:,:2]]
            except SFCConstructionError as exc:
                termination = 'sfc_failed'
                sfc_error = str(exc)
                break
            path = frontend.plan(q, world, regions, states[:,:2], assignment, leader_goal)
            reason = 'no_safe_candidate' if len(path)==1 else 'ok'
            new_assignment = frontend.assign(states, path, regions, assignment)
            assignment = new_assignment
            if len(path)>1:
                previous_a = path[1,2:].reshape(2,2).copy()
            failures.append(reason)
            all_regions.append(regions)
            join_start = states[:, :2].copy()
            paths.append(path.tolist())
            planning_ticks.append(tick)
            assignments.append(assignment.tolist())
            known_history.append(world.known.copy())
            planning_seconds.append(time.perf_counter() - started)
            stuck = stuck + 1 if len(path) == 1 else 0
            if progress:
                print(f'plan tick={tick} center={states[:,:2].mean(axis=0).tolist()} path={len(path)} status={reason} seconds={planning_seconds[-1]:.2f}', flush=True)
        if len(path) > 1:
            target_path=np.array([positions(qi)[assignment] for qi in path[1:]])
            offset=tick % CONFIG.replan_steps
            indices=np.minimum(np.arange(horizon)+offset,len(target_path)-1)
            guidance=target_path[indices]
            guidance_q=path[1:][indices]
        else:
            guidance=np.repeat(states[None,:,:2],horizon,axis=0)
            guidance_q=np.repeat(path[:1],horizon,axis=0)
        u,command = controller.step(states,guidance,world)
        states=integrate(states,u)
        commands.append(command)
        velocity_history.append(controller.velocities.copy())
        history.append(states.copy())
        controls.append(u)
        qs.append(fit_q(states, assignment))
        q_ref_history.append(guidance_q[0].copy())
        assignment_history.append(assignment.copy())
        goal_targets = leader_goal + REF[assignment]
        if np.max(np.linalg.norm(states[:, :2] - goal_targets, axis=1)) < 0.4 and np.max(np.abs(controller.velocities)) < 0.1:
            termination = 'goal_reached'
            break
        if stuck >= 8:
            termination = 'stagnation'
            break
    states = np.array(history)
    controls = np.asarray(controls).reshape(-1, len(REF), 2)
    qs = np.array(qs)
    return SimulationResult(world, states, controls, np.array(q_ref_history), qs, paths, planning_ticks, assignments, np.array(assignment_history), np.array(known_history), planning_seconds, failures, all_regions, termination, np.asarray(commands).reshape(-1, len(REF), 2), np.array(velocity_history), sfc_error)
