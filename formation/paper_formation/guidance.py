import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import LEADER_REF as REF, RADIUS, CONFIG
from .formation import positions, edge_safe

def score_candidates(samples, angles, q, goal, current, assignment, obstacle_distance, previous_direction, mode, regions):
    """Vectorized evaluation of exactly the five paper costs before SafeCheck."""
    centers=samples[:,:2]
    matrices=samples[:,2:].reshape(-1,2,2)
    targets=centers[:,None,:]+np.einsum('nij,kj->nki',matrices,REF)
    targets=targets[:,assignment]
    region_contains=np.zeros(targets.shape[:2],dtype=bool)
    for h,b in regions:
        region_contains |= np.all(np.einsum('nij,kj->nik',targets,h)<=b+1e-9,axis=-1)
    safe_cost=100.*np.sum(~region_contains,axis=1)
    distances=np.maximum(obstacle_distance,0.)
    weights=np.where(distances>.25,distances/(distances.sum()+1e-12),0.)
    risk=np.sum(weights*np.linalg.norm(targets-current[None,:,:],axis=-1),axis=1)
    if mode=='uniform':
        shape=np.abs(matrices[:,0,0]-1.)
        continuity=np.abs(matrices[:,0,0]-q[2])
    else:
        shape=np.linalg.norm(matrices-np.eye(2),axis=(1,2))
        continuity=np.linalg.norm(matrices-q[2:].reshape(2,2),axis=(1,2))
    if previous_direction is not None:
        continuity+=np.abs(np.arctan2(np.sin(angles-previous_direction),np.cos(angles-previous_direction)))
    return np.linalg.norm(centers-goal,axis=1)+.6*shape+safe_cost+risk+.25*continuity


def guide(q0, world, rng, mode, regions, current, assignment, goal, sequences=CONFIG.sequences, steps=CONFIG.steps, candidates=CONFIG.candidates):
    """Algorithm 1: select minimum-cost sample, then terminate if unsafe."""
    best, bestcost = [q0], np.inf
    obstacle_distance = np.maximum(world.clearance(current), 0.)
    for _ in range(sequences):
        path, cost, previous_direction = [q0], 0., None
        for step in range(steps):
            q=path[-1]
            direction=np.arctan2(*(goal-q[:2])[::-1])
            choices=[]
            for k in range(candidates):
                angle=direction+rng.uniform(-np.pi/2,np.pi/2)
                center=q[:2]+rng.uniform(.1,.2)*np.array([np.cos(angle),np.sin(angle)])
                if mode=='uniform':
                    a=np.eye(2)*rng.uniform(.42,1.3)
                else:
                    # Same candidate structure with an affine matrix as the shape variable.
                    a=q[2:].reshape(2,2)+rng.normal(0,.12,(2,2))
                sample=np.r_[center,a.ravel()]
                choices.append((sample,angle))
            samples=np.array([item[0] for item in choices])
            angles=np.array([item[1] for item in choices])
            scores=score_candidates(samples,angles,q,goal,current,assignment,obstacle_distance,previous_direction,mode,regions)
            winner=np.argmin(scores)
            score,selected,angle=scores[winner],samples[winner],angles[winner]
            p0=positions(q)[assignment];p1=positions(selected)[assignment]
            connected=all(any(np.all(h@p0[i]<=b+1e-9) and np.all(h@p1[i]<=b+1e-9)
                              for h,b in regions) for i in range(len(REF)))
            # Extra constraints only ensure a nonsingular, well-spaced affine formation.
            if mode=='affine':
                connected = connected and edge_safe(q,selected,regions)
            if not connected:
                break
            path.append(selected);cost+=score;previous_direction=angle
        cost+=(steps-(len(path)-1))*20.
        if cost < bestcost:
            best,bestcost=path,cost
    return np.array(best)

def assign(states, path, regions, current_assignment):
    """Minimize the paper's sum of distances using Hungarian assignment."""
    if len(path)<2:
        return current_assignment
    targets=positions(path[1])
    matrix=np.linalg.norm(states[:,:2,None]-targets.T[None,:,:],axis=1)
    rows,cols=linear_sum_assignment(matrix)
    return cols[np.argsort(rows)]

class FormationGuidance:
    """Sample independent configuration sequences and assign safe target slots."""

    def __init__(self, mode, rng, sequences, steps, candidates):
        (self.mode, self.rng) = (mode, rng)
        (self.sequences, self.steps, self.candidates) = (sequences, steps, candidates)

    def plan(self, q, world, regions, current, assignment, goal):
        return guide(q, world, self.rng, self.mode, regions, current, assignment, goal, self.sequences, self.steps, self.candidates)

    def assign(self, states, path, regions, assignment):
        """Solve the linear distance-minimizing assignment."""
        return assign(states, path, regions, assignment)
