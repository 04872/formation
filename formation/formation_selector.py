from __future__ import annotations

import math
from dataclasses import dataclass, field

from formation.assignment import compute_best_assignment
from formation.curve_band import CurveBandBuilder
from formation.embedding_qp import EmbeddingQPSolver
from formation.guide_generator import GuideGenerator
from formation.mpc_controller import query_distance_field
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
    Point2D,
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
        self.embedding_solver = EmbeddingQPSolver(
            max_heading_offset_rad=self.config.max_heading_offset_rad,
            heading_grid_size=max(11, 2 * self.config.heading_offset_samples + 1),
            lateral_grid_size=max(11, 2 * self.config.lateral_offset_samples + 1),
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
        if not preview_path.points_xy or not curve_band.samples:
            return self._build_infeasible_evaluation(formation, "empty_preview_or_band")

        embedding_result = self.embedding_solver.solve(preview_path, curve_band, formation)
        if not embedding_result.is_feasible:
            return self._build_infeasible_evaluation(
                formation,
                embedding_result.failure_reason or "embedding_infeasible",
                embedding_result=embedding_result,
            )

        min_slot_clearance_m, mean_clearance_m = self._compute_slot_clearance_stats(
            map_data,
            embedding_result.slot_points_by_step_xy,
        )
        required_clearance_m = robot_radius + safety_margin
        safety_margin_m = min_slot_clearance_m - required_clearance_m
        # Corridor margin already verifies slots against the distance-field-validated
        # strip cells. The polygon-margin check and distance-field check use the same
        # required_clearance (robot_radius + safety_margin). A positive corridor margin
        # means all slots sit inside verified free space.
        is_safe = (
            embedding_result.min_corridor_margin_m >= -1e-9
            and embedding_result.corridor_violation_cost <= 1e-9
        )

        # Only reassign when formation name changes; otherwise identity.
        if current_formation is None or current_formation.name == formation.name:
            n = len(embedding_result.slot_points_by_step_xy[0])
            assignment = AssignmentResult(
                assignment=tuple(range(n)), total_cost=0.0, max_cost=0.0,
                per_robot_costs=[0.0] * n,
            )
        else:
            if current_states:
                current_slots_xy = [(s.x, s.y) for s in current_states]
            else:
                current_slots_xy = self._current_slots_xy(
                    current_formation,
                    embedding_result.center_points_xy[0],
                    embedding_result.heading_rads[0],
                    formation,
                )
            assignment = compute_best_assignment(
                current_slots_xy, embedding_result.slot_points_by_step_xy[0],
            )
        score_breakdown = self._score_candidate(
            formation,
            min_slot_clearance_m,
            safety_margin_m,
            mean_clearance_m,
            assignment.total_cost,
            embedding_result,
        )

        mean_lateral_offset_m = (
            sum(embedding_result.lateral_offsets_m) / len(embedding_result.lateral_offsets_m)
            if embedding_result.lateral_offsets_m
            else 0.0
        )
        mean_heading_offset_rad = (
            sum(embedding_result.heading_offsets_rad) / len(embedding_result.heading_offsets_rad)
            if embedding_result.heading_offsets_rad
            else 0.0
        )
        corridor_feasible = embedding_result.min_corridor_margin_m >= -1e-9
        frontend_status = {
            "curve_band_built": bool(curve_band.samples),
            "embedding_feasible": embedding_result.is_feasible,
            "corridor_feasible": corridor_feasible,
            "is_safe": is_safe,
            "failure_reason": "" if is_safe else self._failure_reason(corridor_feasible, safety_margin_m),
        }
        refined_band_centerline = list(
            curve_band.metadata.get(
                "refined_band_centerline",
                [sample.center_xy for sample in curve_band.samples],
            )
        )
        safe_strip_cells = list(
            curve_band.metadata.get(
                "safe_strip_cells",
                [cell.vertices_xy for cell in curve_band.strip_cells],
            )
        )

        return FormationCandidateEvaluation(
            formation_name=formation.name,
            band_feasible=corridor_feasible,
            is_safe=is_safe,
            score_breakdown=score_breakdown,
            center_points_xy=embedding_result.center_points_xy,
            heading_rads=embedding_result.heading_rads,
            slot_points_by_step_xy=embedding_result.slot_points_by_step_xy,
            assignment=assignment,
            embedding_qp_result=embedding_result,
            lateral_offset_m=mean_lateral_offset_m,
            heading_offset_rad=mean_heading_offset_rad,
            min_slot_clearance_m=min_slot_clearance_m,
            failure_reason=frontend_status["failure_reason"],
            metadata={
                "mean_lateral_offset_m": mean_lateral_offset_m,
                "mean_heading_offset_rad": mean_heading_offset_rad,
                "mean_clearance_m": mean_clearance_m,
                "required_clearance_m": required_clearance_m,
                "curve_band_source_mode": curve_band.source_mode,
                "refined_band_centerline": refined_band_centerline,
                "safe_strip_cells": safe_strip_cells,
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

    def _build_infeasible_evaluation(
        self,
        formation: FormationSpec,
        failure_reason: str,
        embedding_result: EmbeddingQPResult | None = None,
    ) -> FormationCandidateEvaluation:
        preview_alignment_cost = 0.0
        if embedding_result is not None:
            preview_alignment_cost = float(embedding_result.metadata.get("preview_alignment_cost", 0.0))
        return FormationCandidateEvaluation(
            formation_name=formation.name,
            band_feasible=False,
            is_safe=False,
            score_breakdown=FormationScoreBreakdown(
                min_corridor_margin_m=(0.0 if embedding_result is None else embedding_result.min_corridor_margin_m),
                embedding_cost=(0.0 if embedding_result is None else embedding_result.offset_cost + embedding_result.heading_cost),
                corridor_violation_cost=(0.0 if embedding_result is None else embedding_result.corridor_violation_cost),
                switch_cost=0.0,
                task_utility=formation.task_utility,
                total_score=-1_000_000.0,
                offset_cost=0.0 if embedding_result is None else embedding_result.offset_cost,
                heading_cost=0.0 if embedding_result is None else embedding_result.heading_cost,
                preview_alignment_cost=preview_alignment_cost,
                metadata={"failure_reason": failure_reason},
            ),
            embedding_qp_result=embedding_result,
            failure_reason=failure_reason,
            metadata={"frontend_status": {"is_safe": False, "failure_reason": failure_reason}},
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

    def _compute_slot_clearance_stats(
        self,
        map_data: MapData,
        slot_points_by_step_xy: list[list[Point2D]],
    ) -> tuple[float, float]:
        clearances = [
            self._query_clearance(map_data, slot_xy)
            for slot_points_xy in slot_points_by_step_xy
            for slot_xy in slot_points_xy
        ]
        if not clearances:
            return 0.0, 0.0
        return min(clearances), sum(clearances) / len(clearances)

    def _current_slots_xy(
        self,
        current_formation: FormationSpec | None,
        center_xy: Point2D,
        heading_rad: float,
        fallback_formation: FormationSpec,
    ) -> list[Point2D]:
        formation = current_formation or fallback_formation
        return self._transform_slots(formation, center_xy, heading_rad)

    def _transform_slots(self, formation: FormationSpec, center_xy: Point2D, heading_rad: float) -> list[Point2D]:
        cos_heading = math.cos(heading_rad)
        sin_heading = math.sin(heading_rad)
        return [
            (
                center_xy[0] + cos_heading * float(slot[0]) - sin_heading * float(slot[1]),
                center_xy[1] + sin_heading * float(slot[0]) + cos_heading * float(slot[1]),
            )
            for slot in formation.slots
        ]

    def _query_clearance(self, map_data: MapData, point_xy: Point2D) -> float:
        return float(query_distance_field(map_data, point_xy))

    def _score_candidate(
        self,
        formation: FormationSpec,
        min_slot_clearance_m: float,
        safety_margin_m: float,
        mean_clearance_m: float,
        switch_cost: float,
        embedding_result: EmbeddingQPResult,
    ) -> FormationScoreBreakdown:
        weights = self.config.weights
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
                "terminal_heading_error_rad": float(embedding_result.metadata.get("terminal_heading_error_rad", 0.0)),
                "terminal_heading_offset_rad": float(embedding_result.metadata.get("terminal_heading_offset_rad", 0.0)),
                "phi_reference_terminal_rad": float(embedding_result.metadata.get("phi_reference_terminal_rad", 0.0)),
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

    @staticmethod
    def _failure_reason(band_feasible: bool, safety_margin_m: float) -> str:
        if not band_feasible:
            return "slot_outside_safe_corridor"
        if safety_margin_m < -1e-9:
            return "slot_clearance_below_threshold"
        return ""


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
