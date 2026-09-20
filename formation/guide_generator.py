from __future__ import annotations

import math

from formation.assignment import permute_slots
from formation.types import AssignmentResult, FormationCandidateEvaluation, FormationGuide, FormationSpec, GuideSample, Point2D


class GuideGenerator:
    def build(
        self,
        selected_eval: FormationCandidateEvaluation,
        selected_formation: FormationSpec,
        current_formation: FormationSpec | None = None,
        current_assignment: "AssignmentResult | None" = None,
    ) -> FormationGuide:
        switched = current_formation is not None and current_formation.name != selected_formation.name
        if not selected_eval.slot_points_by_step_xy:
            return FormationGuide(
                formation_name=selected_formation.name,
                guide_samples=[],
                assignment=selected_eval.assignment,
                transition_alphas=[],
                switched=switched,
                metadata={
                    "formation_name": selected_eval.formation_name,
                    "band_feasible": selected_eval.band_feasible,
                    "is_safe": selected_eval.is_safe,
                    "failure_reason": selected_eval.failure_reason,
                },
            )

        # Use formation's nominal slots (permuted by assignment) with embedding's
        # center + heading. Slot values are local coords; rotated to world frame.
        if switched and selected_eval.assignment is not None and current_formation is not None:
            assignment_tuple = selected_eval.assignment.assignment
        elif current_assignment is not None:
            assignment_tuple = current_assignment.assignment
        else:
            assignment_tuple = tuple(range(len(selected_formation.slots)))

        ordered_slots_local = [selected_formation.slots[j] for j in assignment_tuple]

        guide_samples: list[GuideSample] = []
        for center_xy, heading_rad in zip(
            selected_eval.center_points_xy,
            selected_eval.heading_rads,
        ):
            ch, sh = math.cos(heading_rad), math.sin(heading_rad)
            robot_points_xy: list[Point2D] = []
            for lx, ly in ordered_slots_local:
                wx = center_xy[0] + ch * float(lx) - sh * float(ly)
                wy = center_xy[1] + sh * float(lx) + ch * float(ly)
                robot_points_xy.append((wx, wy))
            guide_samples.append(
                GuideSample(
                    center_xy=center_xy,
                    heading_rad=heading_rad,
                    robot_points_xy=robot_points_xy,
                )
            )

        guide_metadata = {
            "robot_slots_local": ordered_slots_local,
            "target_slots_step0": ordered_slots_local,
            "current_slots_step0": selected_formation.slots,
            "formation_name": selected_eval.formation_name,
            "selected_formation_name": selected_formation.name,
            "selected_evaluation_is_safe": selected_eval.is_safe,
            "band_feasible": selected_eval.band_feasible,
            "is_safe": selected_eval.is_safe,
            "min_slot_clearance_m": selected_eval.min_slot_clearance_m,
            "failure_reason": selected_eval.failure_reason,
            "mean_lateral_offset_m": selected_eval.metadata.get("mean_lateral_offset_m", selected_eval.lateral_offset_m),
            "mean_heading_offset_rad": selected_eval.metadata.get("mean_heading_offset_rad", selected_eval.heading_offset_rad),
            "mean_clearance_m": selected_eval.metadata.get("mean_clearance_m", 0.0),
            "refined_band_centerline": selected_eval.metadata.get("refined_band_centerline", []),
            "safe_strip_cells": selected_eval.metadata.get("safe_strip_cells", []),
            "frontend_status": dict(selected_eval.metadata.get("frontend_status", {})),
            "curve_band_source_mode": selected_eval.metadata.get("curve_band_source_mode"),
            "score_total": selected_eval.score_breakdown.total_score,
            "score_embedding_cost": selected_eval.score_breakdown.embedding_cost,
            "score_offset_cost": selected_eval.score_breakdown.offset_cost,
            "score_heading_cost": selected_eval.score_breakdown.heading_cost,
            "score_switch_cost": selected_eval.score_breakdown.switch_cost,
            "score_task_utility": selected_eval.score_breakdown.task_utility,
            "score_corridor_violation_cost": selected_eval.score_breakdown.corridor_violation_cost,
            "score_min_corridor_margin_m": selected_eval.score_breakdown.min_corridor_margin_m,
            "score_safety_margin_m": selected_eval.score_breakdown.safety_margin_m,
            "score_terminal_alignment_error_rad": float(selected_eval.score_breakdown.metadata.get("score_terminal_alignment_error_rad", 0.0)),
            "score_terminal_alignment_weight": float(selected_eval.score_breakdown.metadata.get("score_terminal_alignment_weight", 0.0)),
            "score_breakdown_metadata": dict(selected_eval.score_breakdown.metadata),
            "selected_evaluation_metadata": dict(selected_eval.metadata),
            "embedding_is_feasible": bool(selected_eval.metadata.get("embedding_is_feasible", selected_eval.is_safe)),
            "embedding_failure_reason": selected_eval.metadata.get("embedding_failure_reason", ""),
            "offset_cost": float(selected_eval.metadata.get("offset_cost", selected_eval.score_breakdown.offset_cost)),
            "heading_cost": float(selected_eval.metadata.get("heading_cost", selected_eval.score_breakdown.heading_cost)),
            "min_corridor_margin_m": float(selected_eval.metadata.get("min_corridor_margin_m", selected_eval.score_breakdown.min_corridor_margin_m)),
            "corridor_violation_cost": float(selected_eval.metadata.get("corridor_violation_cost", selected_eval.score_breakdown.corridor_violation_cost)),
        }
        if selected_eval.assignment is not None:
            guide_metadata["assignment_total_cost"] = selected_eval.assignment.total_cost
            guide_metadata["assignment_max_cost"] = selected_eval.assignment.max_cost
            guide_metadata["assignment_per_robot_costs"] = list(selected_eval.assignment.per_robot_costs)
        guide_metadata.update({
            key: value for key, value in selected_eval.metadata.items()
            if key not in guide_metadata
        })
        from formation.types import AssignmentResult
        used_assignment = AssignmentResult(
            assignment=assignment_tuple,
            total_cost=0.0, max_cost=0.0,
            per_robot_costs=[0.0] * len(assignment_tuple),
        )
        return FormationGuide(
            formation_name=selected_formation.name,
            guide_samples=guide_samples,
            assignment=used_assignment,
            transition_alphas=[1.0 for _ in guide_samples],
            switched=switched,
            metadata=guide_metadata,
        )

    def _transform_slots(
        self,
        formation: FormationSpec,
        center_xy: Point2D,
        heading_rad: float,
    ) -> list[Point2D]:
        cos_heading = math.cos(heading_rad)
        sin_heading = math.sin(heading_rad)
        return [
            (
                center_xy[0] + cos_heading * float(slot[0]) - sin_heading * float(slot[1]),
                center_xy[1] + sin_heading * float(slot[0]) + cos_heading * float(slot[1]),
            )
            for slot in formation.slots
        ]
