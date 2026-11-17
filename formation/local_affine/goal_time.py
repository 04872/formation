"""Time selection for a single, exactly constrained goal Bézier segment."""
from itertools import combinations
import numpy as np
from .geometry import certify_edge
from .trajectory import differential_failure
from formation.paper_formation.config import RADIUS


def goal_duration(q, goal, velocity, acceleration, regions, reference, E, dt, max_duration=30.):
    """Find first grid time whose fixed six controls pass all hard checks.

    This is a bounded one-dimensional algebraic check, not repeated optimizer
    solves. None means no certified time on the examined grid.
    """
    cert=certify_edge(q,goal,regions,reference)
    if cert is None:return None
    indices,normals=cert
    times=np.arange(1,int(np.floor(max_duration/dt))+1)*dt
    controls=np.empty((len(times),6,6));controls[:,0]=q
    controls[:,1]=q+times[:,None]*velocity/5
    controls[:,2]=q+2*times[:,None]*velocity/5+times[:,None]**2*acceleration/20
    controls[:,3:]=goal
    valid=np.ones(len(times),bool)
    vel=5*np.diff(controls,axis=1)/times[:,None,None]
    acc=20*np.diff(controls,n=2,axis=1)/times[:,None,None]**2
    for i,e in enumerate(E):
        H,b=regions[indices[i]]
        positions=np.einsum('ij,tkj->tki',e,controls)
        valid &= np.all(np.einsum('ij,tkj->tki',H,positions)<=b+1e-8,axis=(1,2))
        valid &= np.all(np.linalg.norm(np.einsum('ij,tkj->tki',e,vel),axis=2)<=.65+1e-8,axis=1)
        valid &= np.all(np.linalg.norm(np.einsum('ij,tkj->tki',e,acc),axis=2)<=1.+1e-8,axis=1)
    for normal,(i,j) in zip(normals,combinations(range(len(E)),2)):
        difference=np.einsum('ij,tkj->tki',E[i]-E[j],controls)
        valid &= np.all(np.einsum('i,tki->tk',normal,difference)>=2*RADIUS+.08-1e-8,axis=1)
    u,_,vh=np.linalg.svd(((q[2:]+goal[2:])/2).reshape(2,2));rotation=u@vh
    A=controls[:,:,2:].reshape(-1,6,2,2);local=np.einsum('ij,tkjl->tkil',rotation.T,A)
    sym=(local+local.transpose(0,1,3,2))/2
    valid &= np.all(np.linalg.eigvalsh(sym)[:,:,0]>=.42-1e-8,axis=1)
    valid &= np.all(np.linalg.svd(A,compute_uv=False)[:,:,0]<=1.35+1e-8,axis=1)
    valid &= np.all(np.abs(A[:,:,0,1])+np.abs(A[:,:,1,0])<=1.5+1e-8,axis=1)
    for i in np.flatnonzero(valid):
        if differential_failure(controls[i],times[i],reference,1.5) is None:
            return float(times[i])
    return None
