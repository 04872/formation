from __future__ import annotations

from dataclasses import dataclass, field

from formation.curve_band import CurveBandBuilder
from formation.formation_feasibility import FormationFeasibility, FeasibilityConfig
from formation.guide_generator import GuideGenerator
from formation.types import (
    AssignmentResult,
    CurveBand,
    EmbeddingQPResult,
    FormationCandidateEvaluation,
    FormationGuide,
    FormationScoreBreakdown,
    FormationSpec,
    LocalPreviewPath,
    MapData,
    RobotState,
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
    curve_band: CurveBand
    evaluations: list[FormationCandidateEvaluation]
    selected_evaluation: FormationCandidateEvaluation
    selected_formation: FormationSpec
    guide: FormationGuide


class FormationSelector:
    def __init__(self, config: SelectorConfig | None = None) -> None:
        self.config = config or SelectorConfig()
        self.band_builder = CurveBandBuilder()
        self.guide_generator = GuideGenerator()
        self.feasibility = FormationFeasibility(
            FeasibilityConfig(
                max_heading_offset_rad=self.config.max_heading_offset_rad,
                heading_grid_size=max(11, 2 * self.config.heading_offset_samples + 1),
                lateral_grid_size=max(11, 2 * self.config.lateral_offset_samples + 1),
            )
        )
        self._hysteresis_counter: dict[str, int] = {}
        self._preference_rank: dict[str, int] = {name: idx for idx, name in enumerate(self.config.formation_preference)}

    def build_curve_band(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        robot_radius: float,
        safety_margin: float,
    ) -> CurveBand:
        return self.band_builder.build(map_data, preview_path, robot_radius, safety_margin)

    def evaluate_candidate_formation(
        self,
        map_data: MapData,
        preview_path: LocalPreviewPath,
        curve_band: CurveBand,
        formation: FormationSpec,
        robot_radius: float,
        safety_margin: float,
        current_formation: FormationSpec | None = None,
        current_states: "list[RobotState] | None" = None,
    ) -> FormationCandidateEvaluation:
        """Evaluate one formation candidate.  Delegates embedding+clearance to
        FormationFeasibility; keeps scoring, assignment, and metadata packaging."""
        feas = self.feasibility.check(
            map_data, preview_path, curve_band, formation,
            robot_radius, safety_margin,
            current_formation=current_formation,
            current_states=current_states,
        )
        required = robot_radius + safety_margin
        score = self._score_candidate(
            formation,
            feas.min_slot_clearance_m,
            feas.safety_margin_m,
            feas.mean_clearance_m,
            feas.assignment.total_cost if feas.assignment else 0.0,
            feas.embedding_qp_result,
        )

        mean_lat = (sum(feas.lateral_offsets_m) / len(feas.lateral_offsets_m)
                    if feas.lateral_offsets_m else 0.0)
        mean_hdg = (sum(feas.heading_offsets_rad) / len(feas.heading_offsets_rad)
                    if feas.heading_offsets_rad else 0.0)
        frontend_status = {
            "curve_band_built": bool(curve_band.samples),
            "embedding_feasible": feas.embedding_qp_result is not None and feas.embedding_qp_result.is_feasible,
            "corridor_feasible": feas.min_corridor_margin_m >= -1e-9,
            "is_safe": feas.is_feasible,
            "failure_reason": feas.failure_reason,
        }
        refined_centerline = list(
            curve_band.metadata.get("refined_band_centerline",
                                    [s.center_xy for s in curve_band.samples]))
        safe_strip = list(
            curve_band.metadata.get("safe_strip_cells",
                                    [c.vertices_xy for c in curve_band.strip_cells]))

        return FormationCandidateEvaluation(
            formation_name=formation.name,
            band_feasible=feas.min_corridor_margin_m >= -1e-9,
            is_safe=feas.is_feasible,
            score_breakdown=score,
            center_points_xy=feas.center_points_xy,
            heading_rads=feas.heading_rads,
            slot_points_by_step_xy=feas.slot_points_by_step_xy,
            assignment=feas.assignment or AssignmentResult(
                assignment=(), total_cost=0.0, max_cost=0.0, per_robot_costs=[]),
            embedding_qp_result=feas.embedding_qp_result,
            lateral_offset_m=mean_lat,
            heading_offset_rad=mean_hdg,
            min_slot_clearance_m=feas.min_slot_clearance_m,
            failure_reason=feas.failure_reason,
            metadata={
                "mean_lateral_offset_m": mean_lat,
                "mean_heading_offset_rad": mean_hdg,
                "mean_clearance_m": feas.mean_clearance_m,
                "required_clearance_m": required,
                "curve_band_source_mode": curve_band.source_mode,
                "refined_band_centerline": refined_centerline,
                "safe_strip_cells": safe_strip,
                "frontend_status": frontend_status,
                "curve_band_metadata": dict(curve_band.metadata),
            },
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
        curve_band = self.build_curve_band(map_data, preview_path, robot_radius, safety_margin)
        formations_by_width = sorted(formations, key=lambda f: f.lateral_half_width, reverse=True)
        evaluations: list[FormationCandidateEvaluation] = []
        for formation in formations_by_width:
            ev = self.evaluate_candidate_formation(
                map_data,
                preview_path,
                curve_band,
                formation,
                robot_radius,
                safety_margin,
                current_formation=current_formation,
                current_states=current_states,
            )
            evaluations.append(ev)
            if ev.is_safe:
                current_evaluated = current_formation is None or any(
                    e.formation_name == current_formation.name for e in evaluations
                )
                if current_evaluated:
                    break
        selected_eval = self._select_best_evaluation(evaluations, current_formation=current_formation)
        selected_formation = next(formation for formation in formations if formation.name == selected_eval.formation_name)
        guide = self.guide_generator.build(selected_eval, selected_formation, current_formation=current_formation, current_assignment=current_assignment)
        return SelectedFormationResult(
            curve_band=curve_band,
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

        # Wider formations have lower preference rank. Always switch to a wider
        # (higher-priority) formation immediately; hysteresis only guards against
        # switching to a narrower (lower-priority) formation to prevent oscillation.
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
        min_slot_clearance_m: float,
        safety_margin_m: float,
        mean_clearance_m: float,
        switch_cost: float,
        embedding_result: "EmbeddingQPResult | None",
    ) -> FormationScoreBreakdown:
        weights = self.config.weights
        if embedding_result is None:
            return FormationScoreBreakdown(
                min_corridor_margin_m=0.0, embedding_cost=0.0,
                corridor_violation_cost=0.0, switch_cost=switch_cost,
                task_utility=formation.task_utility,
                offset_cost=0.0, heading_cost=0.0,
                total_score=-1_000_000.0,
                metadata={"failure_reason": "embedding_infeasible"},
            )
        embedding_cost = embedding_result.offset_cost + embedding_result.heading_cost
        score_breakdown = FormationScoreBreakdown(
            min_corridor_margin_m=embedding_result.min_corridor_margin_m,
            embedding_cost=embedding_cost,
            corridor_violation_cost=embedding_result.corridor_violation_cost,
            switch_cost=switch_cost,
            task_utility=formation.task_utility,
            offset_cost=embedding_result.offset_cost,
            heading_cost=embedding_result.heading_cost,
            safety_margin_m=safety_margin_m,
            mean_clearance_m=mean_clearance_m,
            min_slot_clearance_m=min_slot_clearance_m,
            preview_alignment_cost=float(embedding_result.metadata.get("preview_alignment_cost", 0.0)),
            metadata={
                "inside_slot_count": int(embedding_result.metadata.get("inside_slot_count", 0)),
                "total_slot_count": int(embedding_result.metadata.get("total_slot_count", 0)),
                "inside_slot_ratio": float(embedding_result.metadata.get("inside_slot_ratio", 0.0)),
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
