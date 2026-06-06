from __future__ import annotations

import math
import time
from typing import Any

from formation.metrics import ScenarioMetricsLogger
from formation.mpc_controller import DistributedFormationMPC, query_distance_field
from formation.path_manager import PathManager
from formation.types import FormationControllerReference, MapData, RobotPrediction, RobotState, SimulationTrace


class MultiRobotSimulator:
    def __init__(self, controller: DistributedFormationMPC) -> None:
        self.controller = controller

    def simulate(
        self,
        initial_states: list[RobotState],
        controller_reference: FormationControllerReference,
        map_data: MapData,
    ) -> SimulationTrace:
        if len(initial_states) != controller_reference.robot_count:
            raise ValueError(
                f"Initial state count {len(initial_states)} does not match reference robot count {controller_reference.robot_count}."
            )

        current_states = self._clone_states(initial_states)
        state_history = [self._clone_states(current_states)]
        command_history = []
        prediction_history: list[list[RobotPrediction]] = []
        tracking_errors: list[float] = []
        min_obstacle_clearances: list[float] = []
        min_pairwise_distances: list[float] = []
        previous_predictions: list[RobotPrediction] | None = None

        for step_index in range(max(controller_reference.sample_count - 1, 0)):
            reference_window = controller_reference.window(step_index, self.controller.config.horizon_steps)
            commands, predictions = self.controller.solve_all(
                current_states,
                reference_window,
                map_data,
                previous_predictions=previous_predictions,
            )
            command_history.append(commands)
            prediction_history.append(predictions)
            tracking_errors.append(self._tracking_error(current_states, reference_window))
            min_obstacle_clearances.append(self._min_obstacle_clearance(current_states, map_data))
            min_pairwise_distances.append(self._min_pairwise_distance(current_states))
            current_states = [
                self.controller.propagate_state(state, command)
                for state, command in zip(current_states, commands)
            ]
            state_history.append(self._clone_states(current_states))
            previous_predictions = predictions

        min_obstacle_clearances.append(self._min_obstacle_clearance(current_states, map_data))
        min_pairwise_distances.append(self._min_pairwise_distance(current_states))
        if controller_reference.sample_count > 0:
            tracking_errors.append(
                self._tracking_error(current_states, controller_reference.window(controller_reference.sample_count - 1, 0))
            )

        return SimulationTrace(
            state_history=state_history,
            command_history=command_history,
            prediction_history=prediction_history,
            tracking_errors=tracking_errors,
            min_obstacle_clearances=min_obstacle_clearances,
            min_pairwise_distances=min_pairwise_distances,
            reference_history=[controller_reference],
            metadata={"formation_name": controller_reference.formation_name},
        )

    def simulate_full_path(
        self,
        initial_states: list[RobotState],
        map_data: MapData,
        global_path,
        formations: list,
        preview_planner,
        selector,
        reference_builder,
        robot_radius: float,
        safety_margin: float,
        preview_distance: float,
        current_formation=None,
        goal_tolerance: float | None = None,
        max_replans: int | None = None,
        replan_interval: int = 0,
    ) -> SimulationTrace:
        if not initial_states:
            raise ValueError("initial_states must not be empty.")

        goal_tolerance = goal_tolerance if goal_tolerance is not None else max(self.controller.config.v_max * self.controller.config.dt * 2.0, 0.20)
        path_manager = PathManager(global_path)
        current_states = self._clone_states(initial_states)
        state_history = [self._clone_states(current_states)]
        command_history = []
        prediction_history: list[list[RobotPrediction]] = []
        tracking_errors: list[float] = []
        min_obstacle_clearances: list[float] = []
        min_pairwise_distances: list[float] = []
        reference_history: list[FormationControllerReference] = []
        previous_predictions: list[RobotPrediction] | None = None
        current_assignment = None
        selected_formations: list[str] = []
        preview_ref_history: list[tuple[float, float]] = []
        goal_history: list[float] = []
        cycle_start_indices: list[int] = []
        cycle_end_indices: list[int] = []
        per_cycle_step_counts: list[int] = []
        per_cycle_preview_points: list[list[tuple[float, float]]] = []
        per_cycle_strip_cells: list[list[tuple]] = []
        per_cycle_chord_centers: list[list[tuple[float, float]]] = []
        per_cycle_chord_endpoints: list[list[tuple[tuple[float, float], tuple[float, float]]]] = []
        per_cycle_centerline: list[list[tuple[float, float]]] = []
        total_polyline_length = self._polyline_length(global_path.waypoints_xy)
        total_steps_needed = int(math.ceil(total_polyline_length / (self.controller.config.v_max * self.controller.config.dt * 0.6)))
        default_replans = max(1, int(math.ceil(total_steps_needed / max(replan_interval, 1))))
        max_replans = max_replans if max_replans is not None else default_replans
        plan_wall_time_s = 0.0
        solve_wall_time_s = 0.0
        stop_reason = "max_replans"
        cycle_idx = 0
        robot_count = len(initial_states)
        logger = ScenarioMetricsLogger(
            robot_count, self.controller.config.dt, robot_radius,
            map_data.name, map_data=map_data,
        )
        global_step = 0  # cumulative step counter, never resets across cycles
        print(f"[full-path] max_replans={max_replans} goal_tolerance={goal_tolerance:.2f}m", flush=True)

        for _ in range(max_replans):
            current_goal_distance = self._goal_distance(current_states, map_data.goal_xy)
            if current_goal_distance <= goal_tolerance:
                stop_reason = "goal_reached"
                if logger.T_goal is None:
                    logger.T_goal = global_step * self.controller.config.dt
                    logger.reached_goal = True
                break

            ref_xy = self._centroid(current_states)
            preview_ref_history.append(ref_xy)
            goal_history.append(current_goal_distance)

            plan_start = time.perf_counter()
            window = path_manager.get_local_path_window_from_projection(ref_xy, preview_distance_m=preview_distance)
            preview = preview_planner.plan(map_data, ref_xy, window)
            if not preview.points_xy:
                stop_reason = "empty_preview"
                logger.set_planning_failure()
                break
            # Use swept‑band feasibility when a feasibility module is injected
            feasibility = getattr(self, "_feasibility", None)
            if feasibility is not None:
                from formation.formation_feasibility import FormationFeasibility, FeasibilityConfig
                from formation.types import FormationCandidateEvaluation, FormationScoreBreakdown
                # Build swept band + check formations
                cb = selector.build_curve_band(map_data, preview, robot_radius, safety_margin)
                feas_results = feasibility.check_multi(
                    map_data, preview, cb, formations, robot_radius, safety_margin,
                    current_formation=current_formation, current_states=current_states,
                )
                # Pick best (first feasible, width‑descending)
                feasible = [r for r in feas_results if r.is_feasible]
                if feasible:
                    best_feas = feasible[0]
                    sel_fm = next(f for f in formations if f.name == best_feas.formation_name)
                    # Run full SLSQP centreline optimisation for the selected formation
                    best = feasibility.optimize_centerline(
                        map_data, preview, sel_fm, robot_radius, safety_margin)
                    best.assignment = best_feas.assignment  # use assignment from feasibility check
                else:
                    # All infeasible — fall back to column (narrowest, safest)
                    col_result = next((r for r in feas_results if r.formation_name == "column"), feas_results[-1])
                    best = col_result
                    sel_fm = next(f for f in formations if f.name == best.formation_name)
                # Build a FormationCandidateEvaluation bridge
                eval_ev = FormationCandidateEvaluation(
                    formation_name=best.formation_name,
                    band_feasible=best.is_feasible, is_safe=best.is_feasible,
                    score_breakdown=FormationScoreBreakdown(
                        min_corridor_margin_m=best.min_corridor_margin_m,
                        embedding_cost=0, corridor_violation_cost=0,
                        switch_cost=0, task_utility=sel_fm.task_utility,
                        total_score=0, offset_cost=0, heading_cost=0, metadata={},
                    ),
                    center_points_xy=best.center_points_xy,
                    heading_rads=best.heading_rads,
                    slot_points_by_step_xy=best.slot_points_by_step_xy,
                    assignment=best.assignment,
                    embedding_qp_result=best.embedding_qp_result,
                    lateral_offset_m=0, heading_offset_rad=0,
                    min_slot_clearance_m=best.min_slot_clearance_m,
                    failure_reason=best.failure_reason, metadata=best.metadata,
                )
                guide = selector.guide_generator.build(eval_ev, sel_fm,
                    current_formation=current_formation, current_assignment=current_assignment)
                selection = type('obj', (object,), {
                    'curve_band': cb, 'evaluations': [],
                    'selected_evaluation': eval_ev,
                    'selected_formation': sel_fm, 'guide': guide,
                })()
                # Store centerline for visualization
                per_cycle_centerline.append(list(best.center_points_xy) if best.center_points_xy else [])
            else:
                selection = selector.select_target_formation(
                    map_data, preview, formations, robot_radius, safety_margin,
                    current_formation=current_formation,
                    current_states=current_states,
                    current_assignment=current_assignment,
                )
                per_cycle_centerline.append([])
            per_cycle_preview_points.append(list(preview.points_xy))
            cb = selection.curve_band
            per_cycle_strip_cells.append([cell.vertices_xy for cell in cb.strip_cells])
            per_cycle_chord_centers.append([sample.center_xy for sample in cb.samples])
            per_cycle_chord_endpoints.append([(sample.left_xy, sample.right_xy) for sample in cb.samples])
            controller_reference = reference_builder.build(selection.guide, current_states=current_states)
            if controller_reference.sample_count == 0:
                logger.set_planning_failure()
            plan_wall_time_s += time.perf_counter() - plan_start
            ev_lines = ", ".join(
                f"{ev.formation_name[:6]}={'S' if ev.is_safe else 'I'}:m={ev.score_breakdown.min_corridor_margin_m:+.3f}"
                for ev in (selection.evaluations or [])
            ) if hasattr(selection, 'evaluations') and selection.evaluations else "swept_band"
            print(f"  [{cycle_idx+1}/{max_replans}] plan {plan_wall_time_s:.1f}s | "
                  f"selected {selection.selected_formation.name} "
                  f"({'SAFE' if selection.selected_evaluation.is_safe else 'INFEA'}) | "
                  f"evals=[{ev_lines}] | "
                  f"goal={current_goal_distance:.2f}m", flush=True)

            reference_history.append(controller_reference)
            selected_formations.append(selection.selected_formation.name)
            cycle_start_indices.append(len(command_history))

            steps_this_cycle = max(controller_reference.sample_count - 1, 0) if replan_interval <= 0 else min(replan_interval, max(controller_reference.sample_count - 1, 0))
            mpc_step_report_interval = max(1, steps_this_cycle // 5)
            for step_index in range(steps_this_cycle):
                if step_index % mpc_step_report_interval == 0 and step_index > 0:
                    print(f"    mpc step {step_index}/{steps_this_cycle} "
                          f"(solve {solve_wall_time_s:.1f}s) goal={self._goal_distance(current_states, map_data.goal_xy):.2f}m", flush=True)
                reference_window = controller_reference.window(step_index, self.controller.config.horizon_steps)
                solve_start = time.perf_counter()
                commands, predictions = self.controller.solve_all(
                    current_states,
                    reference_window,
                    map_data,
                    previous_predictions=previous_predictions,
                )
                solve_wall_time_s += time.perf_counter() - solve_start
                command_history.append(commands)
                prediction_history.append(predictions)
                tracking_errors.append(self._tracking_error(current_states, reference_window))
                min_obstacle_clearances.append(self._min_obstacle_clearance(current_states, map_data))
                min_pairwise_distances.append(self._min_pairwise_distance(current_states))
                # ── metrics logger ──────────────────────────────
                for pred in predictions:
                    if pred.metadata.get("solver_status") == "fallback_open_loop":
                        logger.record_mpc_fallback(1)
                sim_time = global_step * self.controller.config.dt
                goal_now = self._goal_distance(current_states, map_data.goal_xy) <= goal_tolerance
                logger.update(global_step, sim_time, current_states, reference_window, goal_now)
                current_states = [
                    self.controller.propagate_state(state, command)
                    for state, command in zip(current_states, commands)
                ]
                state_history.append(self._clone_states(current_states))
                global_step += 1
                previous_predictions = predictions
                if self._goal_distance(current_states, map_data.goal_xy) <= goal_tolerance:
                    stop_reason = "goal_reached"
                    if logger.T_goal is None:
                        logger.T_goal = global_step * self.controller.config.dt
                        logger.reached_goal = True
                    break

            cycle_end_indices.append(len(command_history))
            cycle_steps = cycle_end_indices[-1] - cycle_start_indices[-1]
            per_cycle_step_counts.append(cycle_steps)
            cycle_idx += 1
            print(f"  [{cycle_idx}/{max_replans}] MPC {cycle_steps} steps | "
                  f"solve={solve_wall_time_s:.1f}s | "
                  f"goal={self._goal_distance(current_states, map_data.goal_xy):.2f}m", flush=True)
            current_formation = selection.selected_formation
            current_assignment = selection.guide.assignment
            if stop_reason == "goal_reached":
                break

        final_goal_distance = self._goal_distance(current_states, map_data.goal_xy)
        min_obstacle_clearances.append(self._min_obstacle_clearance(current_states, map_data))
        min_pairwise_distances.append(self._min_pairwise_distance(current_states))
        goal_history.append(final_goal_distance)
        if reference_history:
            tracking_errors.append(self._tracking_error(current_states, reference_history[-1].window(reference_history[-1].sample_count - 1, 0)))

        metrics_result = logger.finalize(timeout=(stop_reason == "max_replans"))
        return SimulationTrace(
            state_history=state_history,
            command_history=command_history,
            prediction_history=prediction_history,
            tracking_errors=tracking_errors,
            min_obstacle_clearances=min_obstacle_clearances,
            min_pairwise_distances=min_pairwise_distances,
            reference_history=reference_history,
            metadata={
                "goal_xy": map_data.goal_xy,
                "start_xy": map_data.start_xy,
                "reached_goal": stop_reason == "goal_reached",
                "goal_distance": final_goal_distance,
                "replanning_cycles": len(reference_history),
                "stop_reason": stop_reason,
                "last_selected_formation": selected_formations[-1] if selected_formations else "",
                "selected_formations": selected_formations,
                "formation_sequence": list(dict.fromkeys(selected_formations)),
                "preview_ref_history": preview_ref_history,
                "goal_history": goal_history,
                "per_cycle_step_counts": per_cycle_step_counts,
                "cycle_start_indices": cycle_start_indices,
                "cycle_end_indices": cycle_end_indices,
                "plan_wall_time_s": plan_wall_time_s,
                "solve_wall_time_s": solve_wall_time_s,
                "per_cycle_preview_points": per_cycle_preview_points,
                "per_cycle_strip_cells": per_cycle_strip_cells,
                "per_cycle_chord_centers": per_cycle_chord_centers,
                "per_cycle_chord_endpoints": per_cycle_chord_endpoints,
                "per_cycle_centerline": per_cycle_centerline,
                "metrics": metrics_result,
                "metrics_timeseries": logger.timeseries_dict(),
            },
        )

    def _clone_states(self, states: list[RobotState]) -> list[RobotState]:
        return [RobotState(**state.__dict__) for state in states]

    def _tracking_error(self, states: list[RobotState], controller_reference: FormationControllerReference) -> float:
        errors = []
        for state, trajectory in zip(states, controller_reference.robot_trajectories):
            if trajectory.samples:
                ref_xy = trajectory.samples[0].position_xy
                errors.append(math.hypot(state.x - ref_xy[0], state.y - ref_xy[1]))
        return max(errors, default=0.0)

    def _min_obstacle_clearance(self, states: list[RobotState], map_data: MapData) -> float:
        return min(query_distance_field(map_data, (state.x, state.y)) for state in states)

    def _min_pairwise_distance(self, states: list[RobotState]) -> float:
        min_distance = math.inf
        for index, state in enumerate(states):
            for other_state in states[index + 1 :]:
                min_distance = min(min_distance, math.hypot(state.x - other_state.x, state.y - other_state.y))
        return min_distance if len(states) >= 2 else math.inf

    def _goal_distance(self, states: list[RobotState], goal_xy: tuple[float, float]) -> float:
        if not states:
            return math.inf
        cx = sum(s.x for s in states) / len(states)
        cy = sum(s.y for s in states) / len(states)
        return math.hypot(cx - goal_xy[0], cy - goal_xy[1])

    def _centroid(self, states: list[RobotState]) -> tuple[float, float]:
        return (
            sum(state.x for state in states) / len(states),
            sum(state.y for state in states) / len(states),
        ) if states else (0.0, 0.0)

    def _polyline_length(self, points_xy: list[tuple[float, float]]) -> float:
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points_xy, points_xy[1:]))

    def initial_states_from_reference(self, controller_reference: FormationControllerReference) -> list[RobotState]:
        return [
            RobotState(x=trajectory.samples[0].position_xy[0], y=trajectory.samples[0].position_xy[1], yaw=trajectory.samples[0].yaw)
            for trajectory in controller_reference.robot_trajectories
        ]


def trace_summary(trace: SimulationTrace) -> dict[str, Any]:
    return {
        "reached_goal": bool(trace.metadata.get("reached_goal", False)),
        "goal_distance": float(trace.metadata.get("goal_distance", math.inf)),
        "replanning_cycles": int(trace.metadata.get("replanning_cycles", 0)),
        "stop_reason": str(trace.metadata.get("stop_reason", "")),
        "last_selected_formation": str(trace.metadata.get("last_selected_formation", "")),
        "selected_formations": list(trace.metadata.get("selected_formations", [])),
        "goal_history": list(trace.metadata.get("goal_history", [])),
        "preview_ref_history": list(trace.metadata.get("preview_ref_history", [])),
        "plan_wall_time_s": float(trace.metadata.get("plan_wall_time_s", 0.0)),
        "solve_wall_time_s": float(trace.metadata.get("solve_wall_time_s", 0.0)),
    }
