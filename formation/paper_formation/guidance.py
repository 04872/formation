import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import DEFAULT_GEOMETRY, RADIUS, CONFIG
from .formation import positions, edge_safe
from .native_sampler import sample_affine

def score_candidates(samples, angles, q, goal, current, assignment, obstacle_distance, previous_direction, mode, regions, geometry=DEFAULT_GEOMETRY):
    """Vectorized evaluation of exactly the five paper costs before SafeCheck."""
    REF=geometry.leader_reference
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
    return CONFIG.goal_weight*np.linalg.norm(centers-goal,axis=1)+.6*shape+safe_cost+risk+.25*continuity


def guide(q0, world, rng, mode, regions, current, assignment, goal, sequences=CONFIG.sequences, steps=CONFIG.steps, candidates=CONFIG.candidates, geometry=DEFAULT_GEOMETRY):
    """Algorithm 1: initialization is not a sampled reference configuration."""
    REF=geometry.leader_reference
    best,bestcost=[],np.inf
    obstacle_distance=np.maximum(world.clearance(current),0.)
    suspension=CONFIG.goal_weight*np.linalg.norm(np.asarray(world.size)*world.res)+CONFIG.suspension_margin
    diagnostics=[]
    for sequence in range(sequences):
        path,cost,previous_direction=[],0.,None
        reason='completed';attempts=0
        for step in range(steps):
            q=q0 if not path else path[-1]
            direction=np.arctan2(*(goal-q[:2])[::-1])
            samples=np.empty((candidates,6));angles=np.empty(candidates)
            if mode=='uniform':
                # The three columns preserve each candidate's original RNG draw
                # order: angle, radius, scale, before moving to the next one.
                draws=rng.random((candidates,3))
                angles[:]=direction+(-np.pi/2+draws[:,0]*np.pi)
                radii=.1+draws[:,1]*(.2-.1)
                samples[:,0]=q[0]+radii*np.cos(angles)
                samples[:,1]=q[1]+radii*np.sin(angles)
                samples[:,2]=.42+draws[:,2]*(1.3-.42)
                samples[:,3:5]=0.
                samples[:,5]=samples[:,2]
                attempts+=candidates
            else:
                draws,count=sample_affine(rng,direction,q[2:],geometry,candidates)
                attempts+=count
                angles[:]=draws[:,0]
                samples[:,0]=q[0]+draws[:,1]*np.cos(angles)
                samples[:,1]=q[1]+draws[:,1]*np.sin(angles)
                samples[:,2:]=draws[:,2:]
            scores=score_candidates(samples,angles,q,goal,current,assignment,obstacle_distance,previous_direction,mode,regions,geometry)
            winner=np.argmin(scores);score,selected,angle=scores[winner],samples[winner],angles[winner]
            # First selected formation joins the actual robot positions; q0 contains
            # initialization parameters and is not itself a sampled path waypoint.
            p0=current if not path else positions(q,geometry)[assignment]
            p1=positions(selected,geometry)[assignment]
            connected=all(any(np.all(h@p0[i]<=b+1e-9) and np.all(h@p1[i]<=b+1e-9)
                              for h,b in regions) for i in range(len(REF)))
            if not connected:
                reason='first_actual_sfc_join' if not path else 'subsequent_sfc_join'
                break
            if path and mode=='affine' and not edge_safe(q,selected,regions,geometry):
                reason='affine_edge_shape_or_spacing'
                break
            path.append(selected);cost+=float(score);previous_direction=angle
        total=cost+(steps-len(path))*suspension
        diagnostics.append({'sequence':sequence,'length':len(path),'accepted_cost':cost,'suspension':(steps-len(path))*suspension,'total_cost':total,'stop_reason':reason,'shape_sample_attempts':attempts})
        if total<bestcost:
            best,bestcost=path,total
    return np.asarray(best,dtype=float).reshape(-1,6),diagnostics

def assign(states, path, regions, current_assignment, geometry=DEFAULT_GEOMETRY):
    """Minimize the paper's sum of distances using Hungarian assignment."""
    if len(path)<1:
        return current_assignment
    targets=positions(path[0],geometry)
    matrix=np.linalg.norm(states[:,:2,None]-targets.T[None,:,:],axis=1)
    rows,cols=linear_sum_assignment(matrix)
    return cols[np.argsort(rows)]

class FormationGuidance:
    """Sample independent configuration sequences and assign safe target slots."""

    def __init__(self, mode, rng, sequences, steps, candidates, geometry=DEFAULT_GEOMETRY):
        self.geometry = geometry
        (self.mode, self.rng) = (mode, rng)
        (self.sequences, self.steps, self.candidates) = (sequences, steps, candidates)

    def plan(self, q, world, regions, current, assignment, goal):
        return guide(q, world, self.rng, self.mode, regions, current, assignment, goal, self.sequences, self.steps, self.candidates, self.geometry)

    def assign(self, states, path, regions, assignment):
        """Solve the linear distance-minimizing assignment."""
        return assign(states, path, regions, assignment, self.geometry)
