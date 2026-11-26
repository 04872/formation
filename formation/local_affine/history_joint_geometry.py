"""Batched certificates and current-map weights for joint-history edges."""
from itertools import combinations

import numpy as np

from formation.paper_formation.config import RADIUS
from formation.paper_formation.formation import affine_shape_path_safe
from .geometry import robot_positions


def evaluate_pairs(qs, pairs, reference, world, regions, certify=True):
    """Return (certified, costs) for pairs of indices into ``qs``.

    A successful certificate applies in both directions along the same
    straight segment. With ``certify=False``, the caller supplies previously
    certified pairs and only the current-map edge weights are recomputed.
    Uncertified pairs have infinite cost.
    """
    qs = np.asarray(qs, dtype=float)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    reference = np.asarray(reference, dtype=float)
    n = len(pairs)
    safe = np.ones(n, dtype=bool)
    costs = np.full(n, np.inf)
    if not n:
        return safe, costs
    positions = robot_positions(qs, reference)
    if certify:
        # Pack memberships so pair intersections remain cheap even when the
        # accumulated history contains hundreds of individual convex SFCs.
        used, inverse = np.unique(pairs, return_inverse=True)
        endpoints = inverse.reshape(-1, 2)
        points = positions[used]
        memberships = np.zeros((len(used), len(reference), (len(regions)+7)//8), dtype=np.uint8)
        for j, (h, b) in enumerate(regions):
            h = np.asarray(h)
            b = np.asarray(b)
            inside = np.ones(points.shape[:2], dtype=bool)
            for normal, bound in zip(h, b):
                inside &= (points[..., 0]*normal[0] + points[..., 1]*normal[1] <= bound+1e-9)
            memberships[..., j//8] |= inside.astype(np.uint8) << (j % 8)
        for first in range(0, n, 4096):
            last = min(first+4096, n)
            a, b = endpoints[first:last].T
            common = np.bitwise_and(memberships[a], memberships[b])
            safe[first:last] = np.all(np.any(common, axis=2), axis=1)
        # Match certify_edge_cached's separating-plane certificate. A plane
        # that separates the entire segment also certifies its reverse.
        for i, j in combinations(range(len(reference)), 2):
            active = np.flatnonzero(safe)
            if not len(active):
                break
            p0 = positions[pairs[active, 0]]
            p1 = positions[pairs[active, 1]]
            delta0 = p0[:, i]-p0[:, j]
            delta1 = p1[:, i]-p1[:, j]
            norm = np.linalg.norm(delta0, axis=1)
            dot = np.einsum('ij,ij->i', delta0, delta1)
            safe[active] &= (norm >= 2*RADIUS+.08) & (dot >= (2*RADIUS+.08)*norm)
        active = np.flatnonzero(safe)
        if len(active):
            a, b = pairs[active].T
            a0 = qs[a, 2:].reshape(-1, 2, 2)
            a1 = qs[b, 2:].reshape(-1, 2, 2)
            singular = np.linalg.svd(a0, compute_uv=False)
            delta_norm = np.linalg.svd(a1-a0, compute_uv=False)[:, 0]
            shear = np.maximum(np.abs(a0[:, 0, 1])+np.abs(a0[:, 1, 0]),
                               np.abs(a1[:, 0, 1])+np.abs(a1[:, 1, 0]))
            easy = ((np.linalg.det(a0) > 0) & (singular[:, -1]-delta_norm >= .42) &
                    (singular[:, 0]+delta_norm <= 1.35) & (shear <= 1.5))
            # G(t) = (1-t)G0 + tG1 - t(1-t) delta.T@delta.
            # Its minimum eigenvalue is bounded below by the linear
            # interpolation of endpoint minimum eigenvalues minus the last
            # term's spectral norm. Minimize that scalar quadratic on [0,1].
            # The maximum singular value and shear are convex along the edge.
            unresolved = np.flatnonzero(~easy)
            if len(unresolved):
                end_sv = np.linalg.svd(a1[unresolved], compute_uv=False)
                lo0 = singular[unresolved, -1]**2
                lo1 = end_sv[:, -1]**2
                d2 = delta_norm[unresolved]**2
                t = np.zeros_like(d2)
                np.divide(lo0 + d2 - lo1, 2*d2, out=t, where=d2 > 0)
                t = np.clip(t, 0., 1.)
                lower = (1-t)*lo0 + t*lo1 - t*(1-t)*d2
                proven = ((np.linalg.det(a0[unresolved]) > 0) &
                          (lower >= .42**2 + 1e-12) &
                          (np.maximum(singular[unresolved, 0], end_sv[:, 0]) <= 1.35 - 1e-12) &
                          (shear[unresolved] <= 1.5))
                easy[unresolved[proven]] = True
            for k in active[~easy]:
                a, b = pairs[k]
                safe[k] = affine_shape_path_safe(qs[a], qs[b])
    selected = np.flatnonzero(safe)
    # Match rrt.edge_cost's three-point quadrature, while amortizing the
    # world's clearance lookup over many pairs and bounding temporary memory.
    for first in range(0, len(selected), 2048):
        indices = selected[first:first+2048]
        a, b = pairs[indices].T
        qa, qb = qs[a], qs[b]
        pa, pb = positions[a], positions[b]
        length = np.linalg.norm((pa-pb).reshape(len(indices), -1), axis=1)/np.sqrt(len(reference))
        nonzero = length != 0
        costs[indices[~nonzero]] = 0.
        if not np.any(nonzero):
            continue
        idx = indices[nonzero]
        qa, qb, pa, pb, length = qa[nonzero], qb[nonzero], pa[nonzero], pb[nonzero], length[nonzero]
        points = np.stack((pa, (pa+pb)/2, pb), axis=1)
        risk = np.maximum(.8-world.clearance(points), 0.).mean(axis=(1, 2))
        shapes = np.stack((qa[:, 2:], (qa[:, 2:]+qb[:, 2:])/2, qb[:, 2:]), axis=1)
        shapes = shapes - np.array([1., 0., 0., 1.])
        shape = np.sum(shapes*shapes, axis=(1, 2))/3
        variation = np.linalg.norm(qb[:, 2:]-qa[:, 2:], axis=1)
        costs[idx] = length*(1+.3*shape+.3*risk)+.08*variation
    return safe, costs
