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

        # ── 2. rasterise swept band as a single polygon ───────────────
        # Right boundary: R_j = γ_j + r₊_j · n_j,  j = 0 … N-1
        # Left boundary:  L_j = γ_j − r₋_j · n_j,  j = N-1 … 0  (reversed)
        right_xy = np.column_stack([
            pts[:, 0] + r_plus * norms[:, 0],
            pts[:, 1] + r_plus * norms[:, 1],
        ])
        left_xy = np.column_stack([
            pts[:, 0] - r_minus * norms[:, 0],
            pts[:, 1] - r_minus * norms[:, 1],
        ])
        poly_xy = np.vstack([right_xy, left_xy[::-1]])  # [2N, 2]

        # Convert to pixel coordinates for scanline fill
        poly_px = np.column_stack([
            (poly_xy[:, 0] - ox) / res,
            (poly_xy[:, 1] - oy) / res,
        ])

        ymin = max(0, int(math.floor(np.min(poly_px[:, 1]))))
        ymax = min(h - 1, int(math.ceil(np.max(poly_px[:, 1]))))
        mask = np.zeros((h, w), dtype=np.uint8)

        n_poly = len(poly_px)
        for iy in range(ymin, ymax + 1):
            yy = iy + 0.5  # cell centre in pixel coords
            xs = []
            for k in range(n_poly):
                y0 = poly_px[k, 1]; y1 = poly_px[(k + 1) % n_poly, 1]
                if (y0 <= yy < y1) or (y1 <= yy < y0):
                    x0 = poly_px[k, 0]; x1 = poly_px[(k + 1) % n_poly, 0]
                    xs.append(x0 + (yy - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for p in range(0, len(xs) - 1, 2):
                x0 = max(0, int(math.floor(xs[p])))
                x1 = min(w - 1, int(math.ceil(xs[p + 1])))
                if x0 <= x1:
                    mask[iy, x0:x1 + 1] = 1

        # ── 3. 1‑cell dilation for discretisation tolerance ──────────
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=1).astype(np.uint8)

        # ── 4. signed distance transform ──────────────────────────
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


# ── signed distance transform ─────────────────────────────────────

def _signed_distance_transform(mask: np.ndarray, res: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    mask_i = mask.astype(np.int32)
    dist_to_outside = distance_transform_edt(mask_i) * res
    dist_to_inside = distance_transform_edt(1 - mask_i) * res
    return np.where(mask == 1, dist_to_outside, -dist_to_inside).astype(float)
