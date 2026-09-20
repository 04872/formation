from __future__ import annotations

import math
from dataclasses import dataclass, field

from formation.formation_feasibility import FeasibilityConfig, FormationFeasibility, FormationFeasibilityResult
from formation.guide_generator import GuideGenerator
from formation.types import (
    CurveBand,
    FormationCandidateEvaluation,
    FormationGuide,
    FormationScoreBreakdown,
    FormationSpec,
    LocalPreviewPath,
    MapData,
    RobotState,
    wrap_to_pi,
)


@dataclass
class SelectorWeights:
    corridor_margin: float = 8.0
    embedding_cost: float = 1.0
    corridor_violation: float = 400.0
    switch_cost: float = 1.0
    task_utility: float = 1.0
    switch_hysteresis: float = 0.15


@dataclass
class SelectorConfig:
    lateral_offset_samples: int = 5
    max_heading_offset_rad: float = 0.30
    heading_offset_samples: int = 5
    weights: SelectorWeights = field(default_factory=SelectorWeights)
    formation_preference: tuple[str, ...] = ("horizontal_line", "t_shape", "square", "compact", "column")
    hysteresis_dwell: int = 3


@dataclass
class SelectedFormationResult:
    curve_band: CurveBand | None
    evaluations: list[FormationCandidateEvaluation]
    selected_evaluation: FormationCandidateEvaluation
    selected_formation: FormationSpec
    guide: FormationGuide


class FormationSelector:
    def __init__(self, config: SelectorConfig | None = None) -> None:
        self.config = config or SelectorConfig()
        self.guide_generator = GuideGenerator()
        self.feasibility: FormationFeasibility = FormationFeasibility(
            FeasibilityConfig(
                mode="swept_band_v2",
                max_heading_offset_rad=self.config.max_heading_offset_rad,
                heading_grid_size=max(11, 2 * self.config.heading_offset_samples + 1),
                lateral_grid_size=max(11, 2 * self.config.lateral_offset_samples + 1),
            )
        )
        self._hysteresis_counter: dict[str, int] = {}
        self._preference_rank: dict[str, int] = {name: idx for idx, name in enumerate(self.config.formation_preference)}

    def build_candidate_evaluation(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        feasibility: FormationFeasibilityResult,
    ) -> FormationCandidateEvaluation:
        required = robot_radius + safety_margin
        mean_lat = (
            sum(feasibility.lateral_offsets_m) / len(feasibility.lateral_offsets_m)
            if feasibility.lateral_offsets_m else 0.0
        )
        mean_hdg = (
            sum(feasibility.heading_offsets_rad) / len(feasibility.heading_offsets_rad)
            if feasibility.heading_offsets_rad else 0.0
        )
        preview_alignment_cost = (
            sum(value * value for value in feasibility.lateral_offsets_m) / len(feasibility.lateral_offsets_m)
            if feasibility.lateral_offsets_m else 0.0
        )
        terminal_meta = self._terminal_alignment_metadata(preview_path, feasibility.center_points_xy, feasibility.heading_rads)
        band_feasible = (
            feasibility.min_corridor_margin_m >= -1e-9
            and feasibility.corridor_violation_cost <= 1e-9
        )
        switch_cost = feasibility.assignment.total_cost if feasibility.assignment is not None else 0.0
        score = self._score_candidate(
            formation,
            feasibility,
            switch_cost=switch_cost,
            preview_alignment_cost=preview_alignment_cost,
            terminal_alignment_error_rad=terminal_meta["score_terminal_alignment_error_rad"],
            terminal_alignment_weight=terminal_meta["score_terminal_alignment_weight"],
        )
        frontend_status = {
            "curve_band_built": False,
            "embedding_feasible": feasibility.is_feasible,
            "corridor_feasible": band_feasible,
            "is_safe": feasibility.is_feasible,
            "failure_reason": feasibility.failure_reason,
        }
        metadata = dict(feasibility.metadata)
        metadata.update({
            "mean_lateral_offset_m": mean_lat,
            "mean_heading_offset_rad": mean_hdg,
            "mean_clearance_m": feasibility.mean_clearance_m,
            "required_clearance_m": required,
            "curve_band_source_mode": preview_path.source_mode,
            "refined_band_centerline": list(feasibility.center_points_xy),
            "safe_strip_cells": [],
            "frontend_status": frontend_status,
            "curve_band_metadata": dict(feasibility.metadata),
            "preview_alignment_cost": preview_alignment_cost,
            "embedding_is_feasible": feasibility.is_feasible,
            "embedding_failure_reason": feasibility.failure_reason if not feasibility.is_feasible else "",
            "min_corridor_margin_m": feasibility.min_corridor_margin_m,
            "corridor_violation_cost": feasibility.corridor_violation_cost,
            "offset_cost": feasibility.offset_cost,
            "heading_cost": feasibility.heading_cost,
            "selected_evaluation_is_safe": feasibility.is_feasible,
            "selected_formation_name": formation.name,
            **terminal_meta,
        })

        return FormationCandidateEvaluation(
            formation_name=formation.name,
            band_feasible=band_feasible,
            is_safe=feasibility.is_feasible,
            score_breakdown=score,
            center_points_xy=feasibility.center_points_xy,
            heading_rads=feasibility.heading_rads,
            slot_points_by_step_xy=feasibility.slot_points_by_step_xy,
            assignment=feasibility.assignment,
            embedding_qp_result=feasibility.embedding_qp_result,
            lateral_offset_m=mean_lat,
            heading_offset_rad=mean_hdg,
            min_slot_clearance_m=feasibility.min_slot_clearance_m,
            failure_reason=feasibility.failure_reason,
            metadata=metadata,
        )

    def evaluate_candidate_formation(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand | None,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_states: list[RobotState] | None = None,
    ) -> FormationCandidateEvaluation:
        feasibility = self.feasibility.check(
            map_data, preview_path, curve_band, formation,
            robot_radius, safety_margin,
            current_formation=current_formation,
            current_states=current_states,
        )
        return self.build_candidate_evaluation(
            map_data,
            preview_path,
            formation,
            robot_radius,
            safety_margin,
            feasibility,
        )

    def select_target_formation(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        formations: list[FormationSpec],
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_assignment: "AssignmentResult | None" = None,
        current_states: "list[RobotState] | None" = None,
    ) -> SelectedFormationResult:
        feasibility_results = self.feasibility.check_multi(
            map_data,
            preview_path,
            None,
            formations,
            robot_radius,
            safety_margin,
            current_formation=current_formation,
            current_states=current_states,
            stop_at_first_feasible=False,
        )
        formations_by_name = {formation.name: formation for formation in formations}
        evaluations = [
            self.build_candidate_evaluation(
                map_data,
                preview_path,
                formations_by_name[result.formation_name],
                robot_radius,
                safety_margin,
                result,
            )
            for result in feasibility_results
            if result.formation_name in formations_by_name
        ]
        selected_eval = self._select_best_evaluation(evaluations, current_formation=current_formation)
        selected_formation = formations_by_name[selected_eval.formation_name]
        guide = self.guide_generator.build(
            selected_eval,
            selected_formation,
            current_formation=current_formation,
            current_assignment=current_assignment,
        )
        return SelectedFormationResult(
            curve_band=None,
            evaluations=evaluations,
            selected_evaluation=selected_eval,
            selected_formation=selected_formation,
            guide=guide,
        )

    def _select_best_evaluation(
        self,
        evaluations: list[FormationCandidateEvaluation],
        current_formation: FormationSpec | None = None,
    ) -> FormationCandidateEvaluation:
        feasible = [ev for ev in evaluations if ev.is_safe]
        if not feasible:
            feasible = [ev for ev in evaluations if ev.band_feasible]
        if not feasible:
            return min(evaluations, key=lambda ev: ev.score_breakdown.corridor_violation_cost)

        def preference_key(ev: FormationCandidateEvaluation) -> tuple:
            rank = self._preference_rank.get(ev.formation_name, 99)
            embedding_cost = ev.score_breakdown.embedding_cost
            neg_margin = -ev.score_breakdown.min_corridor_margin_m
            return (rank, embedding_cost, neg_margin)

        best = min(feasible, key=preference_key)

        if current_formation is None:
            return best

        current_name = current_formation.name
        if best.formation_name == current_name:
            self._hysteresis_counter[current_name] = self._hysteresis_counter.get(current_name, 0) + 1
            return best

        best_rank = self._preference_rank.get(best.formation_name, 99)
        current_rank = self._preference_rank.get(current_name, 99)
        if best_rank < current_rank:
            self._hysteresis_counter.clear()
            return best

        current_feasible = any(ev.formation_name == current_name and ev.is_safe for ev in feasible)
        if not current_feasible:
            self._hysteresis_counter.clear()
            return best

        candidate_name = best.formation_name
        self._hysteresis_counter[candidate_name] = self._hysteresis_counter.get(candidate_name, 0) + 1
        if self._hysteresis_counter[candidate_name] >= self.config.hysteresis_dwell:
            return best

        current_eval = next(ev for ev in evaluations if ev.formation_name == current_name)
        return current_eval

    def _score_candidate(
        self,
        formation: FormationSpec,
        feasibility: FormationFeasibilityResult,
        *,
        switch_cost: float,
        preview_alignment_cost: float,
        terminal_alignment_error_rad: float,
        terminal_alignment_weight: float,
    ) -> FormationScoreBreakdown:
        weights = self.config.weights
        embedding_cost = feasibility.offset_cost + feasibility.heading_cost
        score_breakdown = FormationScoreBreakdown(
            min_corridor_margin_m=feasibility.min_corridor_margin_m,
            embedding_cost=embedding_cost,
            corridor_violation_cost=feasibility.corridor_violation_cost,
            switch_cost=switch_cost,
            task_utility=formation.task_utility,
            offset_cost=feasibility.offset_cost,
            heading_cost=feasibility.heading_cost,
            safety_margin_m=feasibility.safety_margin_m,
            mean_clearance_m=feasibility.mean_clearance_m,
            min_slot_clearance_m=feasibility.min_slot_clearance_m,
            preview_alignment_cost=preview_alignment_cost,
            metadata={
                "score_terminal_alignment_error_rad": terminal_alignment_error_rad,
                "score_terminal_alignment_weight": terminal_alignment_weight,
            },
        )
        score_breakdown.total_score = (
            weights.corridor_margin * score_breakdown.min_corridor_margin_m
            - weights.embedding_cost * score_breakdown.embedding_cost
            - weights.corridor_violation * score_breakdown.corridor_violation_cost
            - weights.switch_cost * score_breakdown.switch_cost
            + weights.task_utility * score_breakdown.task_utility
        )
        return score_breakdown

    def _terminal_alignment_metadata(
        self,
        preview_path: LocalPreviewPath,
        center_points_xy: list[tuple[float, float]],
        heading_rads: list[float],
    ) -> dict[str, float]:
        terminal_heading_offset_rad = 0.0
        phi_reference_terminal_rad = 0.0
        terminal_heading_error_rad = 0.0
        score_terminal_alignment_weight = 0.0
        local_subgoal_xy = preview_path.local_subgoal_xy
        if local_subgoal_xy is not None and center_points_xy and heading_rads:
            preview_distance_m = float(
                preview_path.metadata.get(
                    "configured_preview_distance_m",
                    preview_path.observation_distance_m,
                )
            )
            score_terminal_alignment_weight = 1.0 - min(
                max(preview_path.curve_end_distance_m / max(preview_distance_m, 1e-6), 0.0),
                1.0,
            )
            if score_terminal_alignment_weight <= 0.05:
                score_terminal_alignment_weight = 0.0
            center_xy = center_points_xy[-1]
            dx_goal = local_subgoal_xy[0] - center_xy[0]
            dy_goal = local_subgoal_xy[1] - center_xy[1]
            if math.hypot(dx_goal, dy_goal) > 1e-9:
                target_heading = math.atan2(dy_goal, dx_goal)
                terminal_heading_error_rad = abs(wrap_to_pi(target_heading - heading_rads[-1]))
                preview_terminal_heading = heading_rads[-1]
                if preview_path.tangents_xy:
                    tx, ty = preview_path.tangents_xy[-1]
                    if math.hypot(tx, ty) > 1e-9:
                        preview_terminal_heading = math.atan2(ty, tx)
                phi_reference_terminal_rad = max(
                    -self.config.max_heading_offset_rad,
                    min(
                        self.config.max_heading_offset_rad,
                        score_terminal_alignment_weight * wrap_to_pi(target_heading - preview_terminal_heading),
                    ),
                )
        return {
            "terminal_heading_offset_rad": float(terminal_heading_offset_rad),
            "phi_reference_terminal_rad": float(phi_reference_terminal_rad),
            "terminal_heading_error_rad": float(terminal_heading_error_rad),
            "score_terminal_alignment_error_rad": float(terminal_heading_error_rad),
            "score_terminal_alignment_weight": float(score_terminal_alignment_weight),
        }


def select_target_formation(
    map_data: MapData,
    preview_path: LocalPreviewPath,
    formations: list[FormationSpec],
    robot_radius: float,
    safety_margin: float,
    current_formation: FormationSpec | None = None,
    selector: FormationSelector | None = None,
) -> SelectedFormationResult:
    selector_instance = selector or FormationSelector()
    return selector_instance.select_target_formation(
        map_data,
        preview_path,
        formations,
        robot_radius,
        safety_margin,
        current_formation=current_formation,
    )
