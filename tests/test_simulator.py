from __future__ import annotations

import unittest

from formation import (
    ControllerReferenceBuilder,
    DistributedFormationMPC,
    FormationGuide,
    FormationLibrary,
    FormationSelector,
    GlobalPlanner,
    GuideSample,
    MPCConfig,
    MapBuilder,
    MultiRobotSimulator,
    NarrowingCorridorConfig,
    PathManager,
    PreviewCurvePlanner,
    RightAngleCorridorConfig,
    RobotState,
    trace_summary,
)


class MultiRobotSimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.map_data = MapBuilder().build("right_angle_corridor", RightAngleCorridorConfig(robot_radius=0.09))
        self.config = MPCConfig(
            dt=0.2,
            horizon_steps=10,
            v_max=0.6,
            omega_max=1.0,
            robot_radius=0.09,
            safety_margin=0.06,
            inter_robot_margin=0.10,
        )
        self.reference_builder = ControllerReferenceBuilder(self.config)
        self.controller = DistributedFormationMPC(self.config)
        self.simulator = MultiRobotSimulator(self.controller)

    def test_simulator_rolls_out_static_guide(self) -> None:
        guide = FormationGuide(
            formation_name="pair",
            guide_samples=[
                GuideSample(center_xy=(-4.2, -3.5), heading_rad=0.0, robot_points_xy=[(-4.2, -3.8), (-4.2, -3.2)]),
                GuideSample(center_xy=(-4.1, -3.5), heading_rad=0.0, robot_points_xy=[(-4.1, -3.8), (-4.1, -3.2)]),
                GuideSample(center_xy=(-4.0, -3.5), heading_rad=0.0, robot_points_xy=[(-4.0, -3.8), (-4.0, -3.2)]),
                GuideSample(center_xy=(-3.9, -3.5), heading_rad=0.0, robot_points_xy=[(-3.9, -3.8), (-3.9, -3.2)]),
                GuideSample(center_xy=(-3.8, -3.5), heading_rad=0.0, robot_points_xy=[(-3.8, -3.8), (-3.8, -3.2)]),
            ],
            transition_alphas=[1.0] * 5,
        )
        controller_reference = self.reference_builder.build(guide)
        trace = self.simulator.simulate(
            [RobotState(x=-4.2, y=-3.8, yaw=0.0), RobotState(x=-4.2, y=-3.2, yaw=0.0)],
            controller_reference,
            self.map_data,
        )

        self.assertEqual(trace.step_count, controller_reference.sample_count - 1)
        self.assertGreaterEqual(trace.min_obstacle_clearance, self.config.robot_radius + self.config.safety_margin - 1e-3)
        self.assertGreaterEqual(trace.min_pairwise_distance, 2.0 * self.config.robot_radius + self.config.inter_robot_margin - 12e-2)
        self.assertLess(trace.max_tracking_error, 0.35)

    def test_predictions_propagate_between_steps(self) -> None:
        guide = FormationGuide(
            formation_name="pair",
            guide_samples=[
                GuideSample(center_xy=(-4.2, -3.5), heading_rad=0.0, robot_points_xy=[(-4.2, -3.8), (-4.2, -3.2)]),
                GuideSample(center_xy=(-4.15, -3.5), heading_rad=0.05, robot_points_xy=[(-4.15, -3.8), (-4.15, -3.2)]),
                GuideSample(center_xy=(-4.1, -3.5), heading_rad=0.10, robot_points_xy=[(-4.1, -3.8), (-4.1, -3.2)]),
                GuideSample(center_xy=(-4.05, -3.5), heading_rad=0.15, robot_points_xy=[(-4.05, -3.8), (-4.05, -3.2)]),
                GuideSample(center_xy=(-4.0, -3.5), heading_rad=0.20, robot_points_xy=[(-4.0, -3.8), (-4.0, -3.2)]),
            ],
            transition_alphas=[1.0] * 5,
        )
        controller_reference = self.reference_builder.build(guide)
        initial_states = [RobotState(x=-4.2, y=-3.8, yaw=0.0), RobotState(x=-4.2, y=-3.2, yaw=0.0)]

        first_predictions = self.controller.solve_all(
            initial_states,
            controller_reference.window(0, self.config.horizon_steps),
            self.map_data,
        )[1]
        trace = self.simulator.simulate(initial_states, controller_reference, self.map_data)

        self.assertEqual(len(trace.prediction_history), controller_reference.sample_count - 1)
        self.assertTrue(all(len(step_predictions) == 2 for step_predictions in trace.prediction_history))
        self.assertEqual(first_predictions[0].metadata["neighbor_robot_indices"], [1])
        self.assertEqual(first_predictions[1].metadata["neighbor_robot_indices"], [0])
        self.assertTrue(all(value == 0.0 for value in first_predictions[0].obstacle_slacks))
        self.assertTrue(all(value == 0.0 for value in first_predictions[1].obstacle_slacks))
        self.assertTrue(all(len(step) == 1 for step in first_predictions[0].neighbor_slacks))
        self.assertTrue(all(len(step) == 1 for step in first_predictions[1].neighbor_slacks))
        self.assertEqual(trace.metadata["formation_name"], controller_reference.formation_name)
        self.assertEqual(len(trace.reference_history), 1)
        self.assertEqual(trace.reference_history[0].formation_name, controller_reference.formation_name)
        self.assertEqual(trace.final_states[0].v, trace.command_history[-1][0].v)
        self.assertEqual(trace.final_states[1].v, trace.command_history[-1][1].v)
        self.assertEqual(trace.final_states[0].omega, trace.command_history[-1][0].omega)
        self.assertEqual(trace.final_states[1].omega, trace.command_history[-1][1].omega)
        self.assertEqual(trace_summary(trace)["replanning_cycles"], 0)
        self.assertEqual(trace_summary(trace)["selected_formations"], [])
        self.assertEqual(len(trace_summary(trace)["goal_history"]), 0)
        self.assertEqual(len(trace_summary(trace)["preview_ref_history"]), 0)
        self.assertGreaterEqual(first_predictions[0].metadata["objective"], 0.0)
        self.assertGreaterEqual(first_predictions[1].metadata["objective"], 0.0)
        self.assertTrue(first_predictions[0].metadata["solver_status"])
        self.assertTrue(first_predictions[1].metadata["solver_status"])
        self.assertIn("solver_status", first_predictions[0].metadata)
        self.assertIn("solver_status", first_predictions[1].metadata)

    def test_simulate_full_path_tracks_to_goal(self) -> None:
        robot_radius = 0.09
        safety_margin = 0.06
        inter_robot_margin = 0.10
        map_data = MapBuilder().build("right_angle_corridor", RightAngleCorridorConfig(robot_radius=robot_radius))
        global_path = GlobalPlanner().plan(map_data)
        preview_planner = PreviewCurvePlanner()
        selector = FormationSelector()
        library = FormationLibrary.build_default(robot_radius=robot_radius, inter_robot_margin=inter_robot_margin)
        full_path_config = MPCConfig(
            dt=0.2,
            horizon_steps=10,
            v_max=0.8,
            omega_max=1.2,
            robot_radius=robot_radius,
            safety_margin=safety_margin,
            inter_robot_margin=inter_robot_margin,
        )
        reference_builder = ControllerReferenceBuilder(full_path_config)
        full_path_controller = DistributedFormationMPC(full_path_config)
        full_path_simulator = MultiRobotSimulator(full_path_controller)

        ref_xy = map_data.start_xy
        preview = preview_planner.plan(
            map_data,
            ref_xy,
            PathManager(global_path).get_local_path_window(ref_xy, preview_distance_m=3.5),
        )
        selection = selector.select_target_formation(
            map_data,
            preview,
            library.list(),
            robot_radius,
            safety_margin,
        )
        initial_states = full_path_simulator.initial_states_from_reference(reference_builder.build(selection.guide))
        trace = full_path_simulator.simulate_full_path(
            initial_states,
            map_data,
            global_path,
            library.list(),
            preview_planner,
            selector,
            reference_builder,
            robot_radius,
            safety_margin,
            preview_distance=3.5,
            current_formation=selection.selected_formation,
            goal_tolerance=0.8,
            max_replans=12,
        )
        summary = trace_summary(trace)

        self.assertTrue(trace.reference_history)
        self.assertTrue(trace.command_history)
        self.assertLess(summary["goal_distance"], 1.0)
        self.assertGreaterEqual(trace.min_obstacle_clearance, robot_radius + safety_margin - 12e-2)
        self.assertGreaterEqual(trace.min_pairwise_distance, 2.0 * robot_radius + inter_robot_margin - 23e-2)
        self.assertEqual(summary["replanning_cycles"], len(trace.reference_history))
        self.assertTrue(summary["selected_formations"])
        self.assertEqual(len(summary["goal_history"]), summary["replanning_cycles"] + 1)
        self.assertEqual(len(summary["preview_ref_history"]), summary["replanning_cycles"])
        self.assertGreaterEqual(summary["solve_wall_time_s"], 0.0)
        self.assertGreaterEqual(summary["plan_wall_time_s"], 0.0)
        self.assertEqual(len(trace.metadata["cycle_start_indices"]), summary["replanning_cycles"])
        self.assertEqual(len(trace.metadata["cycle_end_indices"]), summary["replanning_cycles"])
        self.assertEqual(len(trace.metadata["per_cycle_step_counts"]), summary["replanning_cycles"])
        self.assertEqual(trace.metadata["selected_formations"], summary["selected_formations"])
        self.assertEqual(trace.metadata["goal_distance"], summary["goal_distance"])
        self.assertEqual(trace.metadata["replanning_cycles"], summary["replanning_cycles"])
        self.assertIn(trace.metadata["stop_reason"], {"goal_reached", "max_replans"})
        self.assertEqual(trace.metadata["last_selected_formation"], summary["last_selected_formation"])
        self.assertEqual(trace.metadata["preview_ref_history"], summary["preview_ref_history"])
        self.assertEqual(trace.metadata["goal_history"], summary["goal_history"])
        self.assertEqual(len(trace.reference_history), summary["replanning_cycles"])
        self.assertGreaterEqual(trace.max_tracking_error, 0.0)
        self.assertTrue(all(reference.metadata.get("terminal_heading_error_rad") is not None for reference in trace.reference_history))
        self.assertTrue(all(reference.metadata.get("score_terminal_alignment_error_rad") is not None for reference in trace.reference_history))
        self.assertTrue(all(reference.metadata.get("score_terminal_alignment_weight") is not None for reference in trace.reference_history))
        self.assertTrue(all(reference.metadata.get("phi_reference_terminal_rad") is not None for reference in trace.reference_history))
        self.assertTrue(all(reference.metadata.get("terminal_heading_offset_rad") is not None for reference in trace.reference_history))
        self.assertTrue(all(reference.metadata.get("embedding_is_feasible") is not None for reference in trace.reference_history))
        self.assertTrue(all(reference.metadata.get("selected_evaluation_is_safe") is not None for reference in trace.reference_history))
        self.assertTrue(all(reference.metadata.get("selected_formation_name") for reference in trace.reference_history))
        self.assertEqual(trace.metadata["goal_xy"], map_data.goal_xy)
        self.assertEqual(trace.metadata["start_xy"], map_data.start_xy)
        self.assertEqual(len(trace.final_states), len(initial_states))
        self.assertTrue(trace.metadata["reached_goal"] or summary["goal_distance"] >= 0.0)


if __name__ == "__main__":
    unittest.main()
