from __future__ import annotations

import math

import numpy as np

from formation.types import FormationSpec


class FormationLibrary:
    def __init__(self, formations: dict[str, FormationSpec]) -> None:
        self._formations = formations

    @classmethod
    def build_default(
        cls,
        robot_radius: float,
        inter_robot_margin: float = 0.10,
    ) -> "FormationLibrary":
        min_safe_distance = 2.0 * robot_radius + inter_robot_margin
        raw_formations = {
            "square": [
                [-0.30, -0.30],
                [-0.30, 0.30],
                [0.30, -0.30],
                [0.30, 0.30],
            ],
            "column": [
                [-0.90, 0.00],
                [-0.30, 0.00],
                [0.30, 0.00],
                [0.90, 0.00],
            ],
            "horizontal_line": [
                [0.00, -0.90],
                [0.00, -0.30],
                [0.00, 0.30],
                [0.00, 0.90],
            ],
            "t_shape": [
                [0.45, 0.00],
                [-0.15, -0.50],
                [-0.15, 0.00],
                [-0.15, 0.50],
            ],
            "compact": [
                [-0.25, -0.25],
                [-0.25, 0.25],
                [0.25, -0.25],
                [0.25, 0.25],
            ],
        }
        formations: dict[str, FormationSpec] = {}
        for name, slots_list in raw_formations.items():
            slots = np.asarray(slots_list, dtype=float)
            spec = cls._build_spec(name, slots)
            cls.validate(spec, min_safe_distance)
            formations[name] = spec
        return cls(formations)

    @staticmethod
    def _build_spec(name: str, slots: np.ndarray) -> FormationSpec:
        lateral_half_width = float(np.max(np.abs(slots[:, 1])))
        longitudinal_half_length = float(np.max(np.abs(slots[:, 0])))
        bounding_radius = float(np.max(np.linalg.norm(slots, axis=1)))
        min_pairwise_distance = FormationLibrary._compute_min_pairwise_distance(slots)
        # Anchor: rearmost longitudinal pos, lateral midpoint of rearmost row
        min_x = float(np.min(slots[:, 0]))
        rearmost = slots[slots[:, 0] <= min_x + 1e-9]
        anchor_y = float(np.mean(rearmost[:, 1]))
        anchor = np.array([min_x, anchor_y], dtype=float)
        shifted_slots = slots - anchor
        return FormationSpec(
            name=name,
            slots=slots,
            lateral_half_width=lateral_half_width,
            longitudinal_half_length=longitudinal_half_length,
            bounding_radius=bounding_radius,
            min_pairwise_distance=min_pairwise_distance,
            task_utility=lateral_half_width + (1.0 if name == "square" else 0.0),
            anchor_xy=anchor,
            shifted_slots=shifted_slots,
        )

    @staticmethod
    def _compute_min_pairwise_distance(slots: np.ndarray) -> float:
        min_distance = math.inf
        for idx in range(len(slots)):
            for jdx in range(idx + 1, len(slots)):
                distance = float(np.linalg.norm(slots[idx] - slots[jdx]))
                min_distance = min(min_distance, distance)
        return min_distance

    @staticmethod
    def validate(spec: FormationSpec, min_safe_distance: float) -> None:
        if spec.slots.shape != (4, 2):
            raise ValueError(f"Formation {spec.name} must have shape (4, 2).")
        if spec.min_pairwise_distance + 1e-12 < min_safe_distance:
            raise ValueError(
                f"Formation {spec.name} violates minimum slot spacing: "
                f"{spec.min_pairwise_distance:.3f} < {min_safe_distance:.3f}."
            )

    def get(self, name: str) -> FormationSpec:
        return self._formations[name]

    def list(self) -> list[FormationSpec]:
        return [self._formations[name] for name in sorted(self._formations)]

    def names(self) -> list[str]:
        return sorted(self._formations)
