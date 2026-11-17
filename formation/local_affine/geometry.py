"""Formation-space geometry and continuous straight-edge certificates."""
from itertools import combinations
import numpy as np
from formation.paper_formation.config import RADIUS
from formation.paper_formation.formation import valid_shape, affine_shape_path_safe


def positions(q, reference):
    return np.asarray(q)[..., :2, None] + np.einsum('...ij,nj->...in', np.asarray(q)[..., 2:].reshape(*np.asarray(q).shape[:-1], 2, 2), reference)


def robot_positions(q, reference):
    return np.swapaxes(positions(q, reference), -1, -2)


def distance(a, b, reference):
    return float(np.linalg.norm(robot_positions(a, reference) - robot_positions(b, reference)))


def shape_valid(q, mode):
    a = np.asarray(q)[2:].reshape(2, 2)
    if mode == 'uniform' and not np.allclose(a, np.eye(2)*a[0, 0], atol=1e-9):
        return False
    return valid_shape(q)


def region_membership(points, regions):
    """Record each robot's membership in each individual convex region."""
    return np.column_stack([np.all(points @ h.T <= bound+1e-9, axis=1)
                            for h,bound in regions])


def certify_edge_cached(a, b, pa, pb, membership_a, membership_b):
    """Check a directed edge from cached endpoint positions and memberships."""
    common=membership_a & membership_b
    if not np.all(np.any(common,axis=1)):
        return None
    if not affine_shape_path_safe(a, b):
        return None
    selected=np.argmax(common,axis=1)
    normals = []
    for i, j in combinations(range(len(pa)), 2):
        delta = pa[i]-pa[j]
        normal = delta/np.linalg.norm(delta)
        if min(normal@delta, normal@(pb[i]-pb[j])) < 2*RADIUS+.08:
            return None
        normals.append(normal)
    return np.asarray(selected, dtype=int), np.asarray(normals)


def certify_edge(a, b, regions, reference):
    """For each robot select one convex region containing both endpoints."""
    pa,pb=robot_positions(a,reference),robot_positions(b,reference)
    return certify_edge_cached(a,b,pa,pb,region_membership(pa,regions),region_membership(pb,regions))
