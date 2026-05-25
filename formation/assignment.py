from __future__ import annotations

import math
from itertools import permutations

import numpy as np

from formation.types import AssignmentResult, Point2D


def compute_best_assignment(
    current_slots_xy: list[Point2D],
    target_slots_xy: list[Point2D],
    *,
    real_positions: list[Point2D] | None = None,
    real_weight: float = 5.0,
) -> AssignmentResult:
    """Assign robots to target slots minimizing combined cost.

    C_{i→j} = real_weight·‖real_i - target_j‖²   [dominant: actual move]
             + 1.0·‖nominal_i - target_j‖²         [tiebreak: slot consistency]

    Falls back to nominal-only cost when real_positions is None.
    """
    if len(current_slots_xy) != len(target_slots_xy):
        raise ValueError("current_slots_xy and target_slots_xy must have the same length.")
    if not current_slots_xy:
        return AssignmentResult(assignment=(), total_cost=0.0, max_cost=0.0, per_robot_costs=[])

    use_real = real_positions is not None and len(real_positions) == len(target_slots_xy)

    best_assignment: tuple[int, ...] | None = None
    best_total_cost = math.inf
    best_max_cost = math.inf
    best_costs: list[float] = []

    for assignment in permutations(range(len(target_slots_xy))):
        per_robot_costs: list[float] = []
        for robot_index, target_index in enumerate(assignment):
            nominal_dist = _distance(current_slots_xy[robot_index], target_slots_xy[target_index])
            if use_real:
                real_dist = _distance(real_positions[robot_index], target_slots_xy[target_index])
                per_robot_costs.append(real_weight * real_dist + nominal_dist)
            else:
                per_robot_costs.append(nominal_dist)
        total_cost = float(sum(per_robot_costs))
        max_cost = float(max(per_robot_costs))
        if total_cost < best_total_cost - 1e-12 or (
            abs(total_cost - best_total_cost) <= 1e-12 and max_cost < best_max_cost
        ):
            best_assignment = assignment
            best_total_cost = total_cost
            best_max_cost = max_cost
            best_costs = per_robot_costs

    assert best_assignment is not None
    return AssignmentResult(
        assignment=best_assignment,
        total_cost=best_total_cost,
        max_cost=best_max_cost,
        per_robot_costs=best_costs,
    )


def permute_slots(slots_xy: list[Point2D], assignment: tuple[int, ...]) -> list[Point2D]:
    arrays = np.asarray(slots_xy, dtype=float)
    return [tuple(arrays[target_index]) for target_index in assignment]


def _distance(a: Point2D, b: Point2D) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
