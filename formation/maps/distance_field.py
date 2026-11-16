from __future__ import annotations

import numpy as np


def _intersection(f: np.ndarray, q: int, p: int) -> float:
    return ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * (q - p))


def _distance_transform_1d(f: np.ndarray) -> np.ndarray:
    n = len(f)
    distances = np.full(n, np.inf, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(f))
    if finite_positions.size == 0:
        return distances

    v = np.empty(n, dtype=int)
    z = np.empty(n + 1, dtype=float)

    k = 0
    first = int(finite_positions[0])
    v[0] = first
    z[0] = -np.inf
    z[1] = np.inf

    for q_raw in finite_positions[1:]:
        q = int(q_raw)
        s = _intersection(f, q, v[k])
        while k > 0 and s <= z[k]:
            k -= 1
            s = _intersection(f, q, v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        diff = q - v[k]
        distances[q] = diff * diff + f[v[k]]
    return distances


def compute_distance_field(occupancy: np.ndarray, resolution: float) -> np.ndarray:
    """Compute Euclidean obstacle distance in meters for each grid cell.

    This uses a separable squared Euclidean distance transform, which is faster
    than graph propagation and yields the exact Euclidean distance to the
    nearest occupied cell center on the grid.
    """
    if occupancy.ndim != 2:
        raise ValueError("Occupancy grid must be 2-D.")

    occupancy = occupancy.astype(bool, copy=False)
    if not np.any(occupancy):
        return np.full(occupancy.shape, np.inf, dtype=float)

    source = np.where(occupancy, 0.0, np.inf)
    row_distance_sq = np.empty_like(source, dtype=float)
    for row in range(source.shape[0]):
        row_distance_sq[row, :] = _distance_transform_1d(source[row, :])

    distance_sq = np.empty_like(source, dtype=float)
    for col in range(source.shape[1]):
        distance_sq[:, col] = _distance_transform_1d(row_distance_sq[:, col])

    return np.sqrt(distance_sq) * resolution


def compute_inflated_occupancy(
    occupancy: np.ndarray,
    inflation_radius_m: float,
    resolution: float,
    distance_field: np.ndarray | None = None,
) -> np.ndarray:
    """Inflate occupied cells by thresholding the distance field."""
    if inflation_radius_m < 0.0:
        raise ValueError("Inflation radius must be non-negative.")

    if distance_field is None:
        distance_field = compute_distance_field(occupancy, resolution)

    inflated = np.array(occupancy, copy=True, dtype=bool)
    if inflation_radius_m == 0.0:
        return inflated

    near_obstacle = distance_field <= (inflation_radius_m + 1e-12)
    inflated |= near_obstacle
    return inflated
