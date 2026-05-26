from __future__ import annotations

import math

import numpy as np

from formation.mpc_controller import query_distance_field
from formation.types import LocalPreviewPath, MapData, Point2D


class SweptBand:
    r"""Swept offset band  B = F(Ω),  F(u,η) = γ(u) + η·n(u).

    The band is rasterised by evaluating, for each grid cell centre x,
    the distance to the nearest normal‑line segment S(u):

        d_u(x)² = a(u)² + ( b(u) − clip( b(u), −r₋(u), r₊(u) ) )²

    where a(u) = t(u)·(x−γ(u)),  b(u) = n(u)·(x−γ(u)).  Taking
    min_u d_u(x) = 0 means x is inside B.  The resulting binary mask is
    then converted to a signed distance field via scipy.
    """

    def __init__(
        self, grid: np.ndarray, origin: tuple[float, float], resolution: float,
    ) -> None:
        self._grid = grid
        self._ox, self._oy = origin
        self._res = resolution
        self._h, self._w = grid.shape

    def margin(self, pt: Point2D) -> float:
        fx = (pt[0] - self._ox) / self._res
        fy = (pt[1] - self._oy) / self._res
        ix0 = int(math.floor(fx)); iy0 = int(math.floor(fy))
        ix1 = ix0 + 1; iy1 = iy0 + 1
        ix0c = max(0, min(ix0, self._w - 1)); ix1c = max(0, min(ix1, self._w - 1))
        iy0c = max(0, min(iy0, self._h - 1)); iy1c = max(0, min(iy1, self._h - 1))
        tx = fx - ix0; ty = fy - iy0
        v00 = self._grid[iy0c, ix0c]; v10 = self._grid[iy0c, ix1c]
        v01 = self._grid[iy1c, ix0c]; v11 = self._grid[iy1c, ix1c]
        return float((1 - ty) * ((1 - tx) * v00 + tx * v10)
                     + ty * ((1 - tx) * v01 + tx * v11))

    def margin_batch(self, pts: np.ndarray) -> np.ndarray:
        fx = (pts[:, 0] - self._ox) / self._res
        fy = (pts[:, 1] - self._oy) / self._res
        ix0 = np.clip(np.floor(fx).astype(int), 0, self._w - 1)
        iy0 = np.clip(np.floor(fy).astype(int), 0, self._h - 1)
        ix1 = np.clip(ix0 + 1, 0, self._w - 1)
        iy1 = np.clip(iy0 + 1, 0, self._h - 1)
        tx = fx - ix0; ty = fy - iy0
        v00 = self._grid[iy0, ix0]; v10 = self._grid[iy0, ix1]
        v01 = self._grid[iy1, ix0]; v11 = self._grid[iy1, ix1]
        return (1 - ty) * ((1 - tx) * v00 + tx * v10) + ty * ((1 - tx) * v01 + tx * v11)


class SweptBandBuilder:
    def __init__(self, lateral_step_m: float = 0.05, max_ray_m: float = 3.0) -> None:
        self.lateral_step = lateral_step_m
        self.max_ray = max_ray_m

    def build(
        self, map_data: MapData, preview_path: LocalPreviewPath,
        clearance_threshold: float,
    ) -> SweptBand:
        pts = np.asarray(preview_path.points_xy, dtype=float)
        norms = np.asarray(preview_path.normals_xy, dtype=float)
        n_pts = len(pts)
        ox, oy = map_data.origin_xy; res = map_data.resolution
        h = int(map_data.height_m / res) + 1
        w = int(map_data.width_m / res) + 1
        if n_pts < 2:
            return SweptBand(np.zeros((max(h, 1), max(w, 1))), map_data.origin_xy, res)

        # ── 1. raycast bounds ─────────────────────────────────────
        step = max(self.lateral_step, res * 0.5)
        ms = int(math.ceil(self.max_ray / step))
        r_minus = np.zeros(n_pts); r_plus = np.zeros(n_pts)
        for i in range(n_pts):
            r_minus[i] = self._raycast(map_data, pts[i], -norms[i], clearance_threshold, step, ms)
            r_plus[i] = self._raycast(map_data, pts[i], norms[i], clearance_threshold, step, ms)

        # ── tangents (unit) ───────────────────────────────────────
        tangents = np.zeros_like(norms)
        for i in range(n_pts):
            if i == 0: d = pts[1] - pts[0]
            elif i == n_pts - 1: d = pts[-1] - pts[-2]
            else: d = pts[i + 1] - pts[i - 1]
            dn = float(np.linalg.norm(d))
            tangents[i] = d / dn if dn > 1e-9 else np.array([1.0, 0.0])

        # ── 2. per‑segment d_B with generous tolerance → mask ──────
        mask = np.zeros((h, w), dtype=np.uint8)
        tol = res  # 1‑cell tolerance for both tangential and normal
        for j in range(n_pts - 1):
            corners = np.asarray([
                pts[j] - r_minus[j] * norms[j],
                pts[j] + r_plus[j] * norms[j],
                pts[j + 1] - r_minus[j + 1] * norms[j + 1],
                pts[j + 1] + r_plus[j + 1] * norms[j + 1],
            ])
            cmin = np.min(corners, axis=0) - res
            cmax = np.max(corners, axis=0) + res
            ix0 = max(0, int(math.floor((cmin[0] - ox) / res)))
            ix1 = min(w - 1, int(math.ceil((cmax[0] - ox) / res)))
            iy0 = max(0, int(math.floor((cmin[1] - oy) / res)))
            iy1 = min(h - 1, int(math.ceil((cmax[1] - oy) / res)))
            for iy in range(iy0, iy1 + 1):
                cy = oy + (iy + 0.5) * res
                for ix in range(ix0, ix1 + 1):
                    cx = ox + (ix + 0.5) * res
                    if _db_segment(
                        np.array([cx, cy]),
                        pts[j], pts[j + 1],
                        tangents[j], tangents[j + 1],
                        norms[j], norms[j + 1],
                        r_minus[j], r_minus[j + 1],
                        r_plus[j], r_plus[j + 1],
                    ) <= tol:
                        mask[iy, ix] = 1

        # ── 3. signed distance transform ──────────────────────────
        grid = _signed_distance_transform(mask, res)
        return SweptBand(grid, map_data.origin_xy, res)

    @staticmethod
    def _raycast(map_data, origin, direction, threshold, step, max_steps):
        prev_d = 0.0
        prev_df = float(query_distance_field(map_data, (float(origin[0]), float(origin[1]))))
        safe = 0.0
        for _ in range(max_steps):
            safe += step
            px = float(origin[0] + safe * direction[0])
            py = float(origin[1] + safe * direction[1])
            df = float(query_distance_field(map_data, (px, py)))
            if df < threshold - 1e-9:
                if prev_df - df > 1e-12:
                    return max(prev_d + (threshold - prev_df) / (prev_df - df) * step, 0.0)
                return max(prev_d, 0.0)
            prev_d = safe; prev_df = df
        return safe


def _db_segment(
    x: np.ndarray,
    p0: np.ndarray, p1: np.ndarray,
    t0: np.ndarray, t1: np.ndarray,
    n0: np.ndarray, n1: np.ndarray,
    l0: float, l1: float,
    r0: float, r1: float,
) -> float:
    r"""d_t(x) for a single segment:  min_{t∈[0,1]}  √( a² + (b−clip(b))² )."""
    seg = p1 - p0
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-12:
        return float("inf")
    t = max(0.0, min(1.0, float(np.dot(x - p0, seg)) / seg_len_sq))
    centre = p0 + t * seg
    tj = t0 + t * (t1 - t0)
    tn = float(np.linalg.norm(tj))
    if tn < 1e-9: return float("inf")
    tj = tj / tn
    nj = np.array([-tj[1], tj[0]])
    a = float(np.dot(x - centre, tj))
    b = float(np.dot(x - centre, nj))
    lt = l0 + t * (l1 - l0)
    rt = r0 + t * (r1 - r0)
    bc = max(-lt, min(rt, b))
    return float(math.sqrt(a * a + (b - bc) * (b - bc)))


# ── signed distance transform ─────────────────────────────────────

def _signed_distance_transform(mask: np.ndarray, res: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    mask_i = mask.astype(np.int32)
    dist_to_outside = distance_transform_edt(mask_i) * res
    dist_to_inside = distance_transform_edt(1 - mask_i) * res
    return np.where(mask == 1, dist_to_outside, -dist_to_inside).astype(float)
