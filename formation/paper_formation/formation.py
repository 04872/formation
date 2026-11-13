from itertools import combinations
import numpy as np
from .config import LEADER_REF as REF, RADIUS

REF_PAIR_MIN = min(np.linalg.norm(REF[i]-REF[j]) for i,j in combinations(range(len(REF)),2))

def positions(q):
    return q[:2] + REF @ q[2:].reshape(2, 2).T

def valid_shape(q):
    """Evaluate 2x2 singular values analytically and exact reference pair spacings."""
    a=q[2:].reshape(2,2)
    trace=np.sum(a*a); det=a[0,0]*a[1,1]-a[0,1]*a[1,0]
    disc=np.sqrt(max(0.,trace*trace-4*det*det))
    min_sv2=(trace-disc)/2;max_sv2=(trace+disc)/2
    if not (det>0 and min_sv2>=.42**2 and max_sv2<=1.35**2 and
            abs(a[0,1])+abs(a[1,0])<=1.5):
        return False
    # This lower bound is sufficient for every reference pair.
    if min_sv2*REF_PAIR_MIN**2 >= (2*RADIUS+.08)**2:
        return True
    return all(np.linalg.norm(a@(REF[i]-REF[j]))>=2*RADIUS+.08 for i,j in combinations(range(len(REF)),2))

def _polynomial_nonnegative(poly, tolerance=1e-10):
    """Check a scalar polynomial on [0,1] at endpoints and all stationary points."""
    roots=poly.deriv().roots()
    points=[0.,1.]+[float(root.real) for root in roots if abs(root.imag)<1e-8 and 0<root.real<1]
    return min(float(poly(x)) for x in points)>=-tolerance


def affine_shape_path_safe(q0, q1):
    """Certify singular-value bounds continuously using 2x2 PSD polynomial minors."""
    from numpy.polynomial import Polynomial as P
    a0=q0[2:].reshape(2,2); delta=q1[2:].reshape(2,2)-a0
    sv=np.linalg.svd(a0,compute_uv=False);delta_norm=np.linalg.norm(delta,2)
    shear=max(abs(a0[0,1])+abs(a0[1,0]),abs(a0[0,1]+delta[0,1])+abs(a0[1,0]+delta[1,0]))
    if np.linalg.det(a0)>0 and sv[-1]-delta_norm>=.42 and sv[0]+delta_norm<=1.35 and shear<=1.5:
        return True
    a=[[P([a0[i,j],delta[i,j]]) for j in range(2)] for i in range(2)]
    gram=[[sum((a[k][i]*a[k][j] for k in range(2)),P([0.])) for j in range(2)] for i in range(2)]
    for bound, sign in ((.42**2,1),(1.35**2,-1)):
        m=[[sign*(gram[i][j]-(bound if i==j else 0.)) for j in range(2)] for i in range(2)]
        if not all(_polynomial_nonnegative(v) for v in (m[0][0],m[1][1],m[0][0]*m[1][1]-m[0][1]*m[1][0])):
            return False
    # Invertibility prevents det(A(t)) changing sign; shear magnitude is convex.
    return (np.linalg.det(a0)>0 and np.linalg.det(a0+delta)>0 and
            max(abs(a0[0,1])+abs(a0[1,0]),abs(a0[0,1]+delta[0,1])+abs(a0[1,0]+delta[1,0]))<=1.5)


def edge_safe(q0, q1, regions):
    """Certify environment, spacing and shape over the entire affine edge."""
    p0,p1=positions(q0),positions(q1)
    if not valid_shape(q1):
        return False
    for i,j in combinations(range(len(REF)),2):
        d0=p0[i]-p0[j];v=p1[i]-p1[j]-d0
        t=np.clip(-d0@v/(v@v+1e-12),0,1)
        if np.linalg.norm(d0+t*v)<2*RADIUS+.08:
            return False
    if not affine_shape_path_safe(q0,q1):
        return False
    return all(any(np.all(h@p0[i]<=b+1e-9) and np.all(h@p1[i]<=b+1e-9) for h,b in regions) for i in range(len(REF)))

def fit_q(states, assignment=None):
    """Least squares affine fit in leader-anchored reference coordinates."""
    r=REF if assignment is None else REF[assignment]
    design=np.column_stack((np.ones(len(REF)),r))
    parameters=np.linalg.lstsq(design,states[:,:2],rcond=None)[0]
    return np.r_[parameters[0],parameters[1:].T.ravel()]

