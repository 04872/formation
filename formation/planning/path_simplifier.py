from __future__ import annotations

from typing import Iterable

import numpy as np

from formation.types import GridIndex, Point2D


def grid_path_to_world(
    path_rc: Iterable[GridIndex],
    origin_xy: Point2D,
    resolution: float,
) -> list[Point2D]:
    origin_x, origin_y = origin_xy
    world_path: list[Point2D] = []
    for row, col in path_rc:
        x = origin_x + (col + 0.5) * resolution
        y = origin_y + (row + 0.5) * resolution
        world_path.append((x, y))
    return world_path


def remove_duplicate_points(path_rc: Iterable[GridIndex]) -> list[GridIndex]:
    unique: list[GridIndex] = []
    for point in path_rc:
        if not unique or point != unique[-1]:
            unique.append(point)
    return unique


def remove_collinear_points(path_rc: Iterable[GridIndex]) -> list[GridIndex]:
    points = remove_duplicate_points(path_rc)
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    for idx in range(1, len(points) - 1):
        prev_row, prev_col = simplified[-1]
        row, col = points[idx]
        next_row, next_col = points[idx + 1]
        v1 = (row - prev_row, col - prev_col)
        v2 = (next_row - row, next_col - col)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross != 0:
            simplified.append((row, col))
    simplified.append(points[-1])
    return simplified


def bresenham_line(a_rc: GridIndex, b_rc: GridIndex) -> list[GridIndex]:
    row0, col0 = a_rc
    row1, col1 = b_rc

    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1

    points: list[GridIndex] = []
    if d_col > d_row:
        error = d_col // 2
        while col0 != col1:
            points.append((row0, col0))
            error -= d_row
            if error < 0:
                row0 += step_row
                error += d_col
            col0 += step_col
    else:
        error = d_row // 2
        while row0 != row1:
            points.append((row0, col0))
            error -= d_col
            if error < 0:
                col0 += step_col
                error += d_row
            row0 += step_row

    points.append((row1, col1))
    return points


def has_line_of_sight(
    inflated_occupancy: np.ndarray,
    a_rc: GridIndex,
    b_rc: GridIndex,
) -> bool:
    for row, col in bresenham_line(a_rc, b_rc):
        if inflated_occupancy[row, col]:
            return False
    return True


def simplify_grid_path(
    path_rc: Iterable[GridIndex],
    inflated_occupancy: np.ndarray,
) -> list[GridIndex]:
    points = remove_collinear_points(path_rc)
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    anchor_index = 0
    while anchor_index < len(points) - 1:
        next_index = anchor_index + 1
        furthest_visible = next_index
        while next_index < len(points):
            if has_line_of_sight(inflated_occupancy, points[anchor_index], points[next_index]):
                furthest_visible = next_index
                next_index += 1
            else:
                break
        simplified.append(points[furthest_visible])
        anchor_index = furthest_visible
    return simplified
