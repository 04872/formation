from itertools import combinations
import numpy as np
from .config import LEADER_REF as REF, RADIUS

def positions(q):
    return q[:2] + REF @ q[2:].reshape(2, 2).T

def valid_shape(q):
    a = q[2:].reshape(2, 2)
    sv = np.linalg.svd(a, compute_uv=False)
    d = np.linalg.norm(positions(q)[:, None] - positions(q)[None, :], axis=2)
    return np.linalg.det(a) > 0 and sv.min() >= 0.42 and (sv.max() <= 1.35) and (abs(a[0, 1]) + abs(a[1, 0]) <= 1.5) and (np.min(d + np.eye(len(REF)) * 100) >= 2 * RADIUS + 0.08)

def edge_safe(q0, q1, regions):
    """Certify a reference edge in shared SFCs and enforce shape and spacing limits."""
    (p0, p1) = (positions(q0), positions(q1))
    if not valid_shape(q1):
        return False
    for (i, j) in combinations(range(len(REF)), 2):
        (d0, v) = (p0[i] - p0[j], p1[i] - p1[j] - (p0[i] - p0[j]))
        t = np.clip(-d0 @ v / (v @ v + 1e-12), 0, 1)
        if np.linalg.norm(d0 + t * v) < 2 * RADIUS + 0.08:
            return False
    A0 = q0[2:].reshape(2, 2)
    A1 = q1[2:].reshape(2, 2)
    margin = np.linalg.norm(A1 - A0, 2) / 24
    for t in np.linspace(0, 1, 13):
        a = (1 - t) * A0 + t * A1
        sv = np.linalg.svd(a, compute_uv=False)
        if np.linalg.det(a) <= 0 or sv.min() < 0.42 + margin or sv.max() > 1.35 - margin:
            return False
        if abs(a[0, 1]) + abs(a[1, 0]) > 1.5:
            return False
    return all((any((np.all(h @ p0[i] <= b + 1e-09) and np.all(h @ p1[i] <= b + 1e-09) for (h, b) in regions)) for i in range(len(REF))))

def fit_q(states, assignment=None):
    """Least squares affine fit in leader-anchored reference coordinates."""
    r=REF if assignment is None else REF[assignment]
    design=np.column_stack((np.ones(len(REF)),r))
    parameters=np.linalg.lstsq(design,states[:,:2],rcond=None)[0]
    return np.r_[parameters[0],parameters[1:].T.ravel()]

