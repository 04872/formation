from __future__ import annotations

import unittest

from formation import (
    ControllerReferenceBuilder,
    DistributedFormationMPC,
    FormationLibrary,
    FormationSelector,
    GlobalPlanner,
    MapBuilder,
    MPCConfig,
    MultiRobotSimulator,
    NarrowingCorridorConfig,
    PathManager,
    PreviewCurvePlanner,
    RightAngleCorridorConfig,
    RobotState,
    trace_summary,
)


class FormationPipelineSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = MapBuilder()
        self.global_planner = GlobalPlanner()
        self.preview_planner = PreviewCurvePlanner()
        self.selector = FormationSelector()
        self.robot_radius = 0.09
        self.safety_margin = 0.06
        self.inter_robot_margin = 0.10
        self.library = FormationLibrary.build_default(
            robot_radius=self.robot_radius,
            inter_robot_margin=self.inter_robot_margin,
        )
        self.mpc_config = MPCConfig(
            dt=0.2,
            horizon_steps=10,
            v_max=0.8,
            omega_max=1.2,
            robot_radius=self.robot_radius,
            safety_margin=self.safety_margin,
            inter_robot_margin=self.inter_robot_margin,
        )
        self.reference_builder = ControllerReferenceBuilder(self.mpc_config)
        self.controller = DistributedFormationMPC(self.mpc_config)
        self.simulator = MultiRobotSimulator(self.controller)

    def test_pipeline_runs_on_right_angle_corridor(self) -> None:
        config = RightAngleCorridorConfig(robot_radius=self.robot_radius)
        map_data = self.builder.build("right_angle_corridor", config)
        global_path = self.global_planner.plan(map_data)
        preview = self.preview_planner.plan(
            map_data,
            map_data.start_xy,
            PathManager(global_path).get_local_path_window(map_data.start_xy, preview_distance_m=3.5),
        )
        result = self.selector.select_target_formation(
            map_data,
            preview,
            self.library.list(),
            self.robot_radius,
            self.safety_margin,
        )
        controller_reference = self.reference_builder.build(result.guide)
        trace = self.simulator.simulate(
            [
                RobotState(x=sample.position_xy[0], y=sample.position_xy[1], yaw=sample.yaw)
                for sample in (trajectory.samples[0] for trajectory in controller_reference.robot_trajectories)
            ],
            controller_reference,
            map_data,
        )

        self.assertTrue(preview.is_safe)
        self.assertTrue(result.selected_evaluation.is_safe)
        self.assertGreater(controller_reference.sample_count, 2)
        self.assertLess(trace.max_tracking_error, 0.75)
        self.assertGreaterEqual(trace.min_obstacle_clearance, self.robot_radius + self.safety_margin - 0.25)
        self.assertGreaterEqual(trace.min_pairwise_distance, 2.0 * self.robot_radius + self.inter_robot_margin - 0.46)

    def test_pipeline_prefers_compact_near_narrowing_throat(self) -> None:
        config = NarrowingCorridorConfig(robot_radius=self.robot_radius)
        map_data = self.builder.build("narrowing_corridor", config)
        global_path = self.global_planner.plan(map_data)
        ref_xy = (2.8, config.centerline_slope * 2.8 + config.centerline_intercept)
        preview = self.preview_planner.plan(
            map_data,
            ref_xy,
            PathManager(global_path).get_local_path_window_from_projection(ref_xy, preview_distance_m=3.5),
        )
        result = self.selector.select_target_formation(
            map_data,
            preview,
            self.library.list(),
            self.robot_radius,
            self.safety_margin,
        )
        controller_reference = self.reference_builder.build(result.guide)
        trace = self.simulator.simulate(
            [
                RobotState(x=sample.position_xy[0], y=sample.position_xy[1], yaw=sample.yaw)
                for sample in (trajectory.samples[0] for trajectory in controller_reference.robot_trajectories)
            ],
            controller_reference,
            map_data,
        )

        self.assertTrue(result.selected_evaluation.is_safe)
        self.assertIn(result.selected_formation.name, {"t_shape", "square", "compact"})
        self.assertLess(trace.max_tracking_error, 0.60)
        self.assertGreaterEqual(trace.min_obstacle_clearance, self.robot_radius + self.safety_margin - 0.25)
        self.assertGreaterEqual(trace.min_pairwise_distance, 2.0 * self.robot_radius + self.inter_robot_margin - 0.46)

    def test_full_pipeline_reaches_goal_with_replanning(self) -> None:
        config = RightAngleCorridorConfig(robot_radius=self.robot_radius)
        map_data = self.builder.build("right_angle_corridor", config)
        global_path = self.global_planner.plan(map_data)
        preview = self.preview_planner.plan(
            map_data,
            map_data.start_xy,
            PathManager(global_path).get_local_path_window(map_data.start_xy, preview_distance_m=3.5),
        )
        result = self.selector.select_target_formation(
            map_data,
            preview,
            self.library.list(),
            self.robot_radius,
            self.safety_margin,
        )
        initial_states = self.simulator.initial_states_from_reference(self.reference_builder.build(result.guide))
        trace = self.simulator.simulate_full_path(
            initial_states,
            map_data,
            global_path,
            self.library.list(),
            self.preview_planner,
            self.selector,
            self.reference_builder,
            self.robot_radius,
            self.safety_margin,
            preview_distance=3.5,
            current_formation=result.selected_formation,
            goal_tolerance=0.9,
            max_replans=12,
        )
        summary = trace_summary(trace)

        self.assertTrue(trace.reference_history)
        self.assertTrue(trace.command_history)
        self.assertLess(summary["goal_distance"], 1.2)
        self.assertGreaterEqual(trace.min_obstacle_clearance, self.robot_radius + self.safety_margin - 20e-2)
        self.assertGreaterEqual(trace.min_pairwise_distance, 2.0 * self.robot_radius + self.inter_robot_margin - 23e-2)
        self.assertGreater(summary["replanning_cycles"], 0)
        self.assertTrue(summary["selected_formations"])
        self.assertEqual(summary["replanning_cycles"], len(trace.reference_history))
        self.assertEqual(len(summary["goal_history"]), summary["replanning_cycles"] + 1)
        self.assertEqual(len(summary["preview_ref_history"]), summary["replanning_cycles"])


if __name__ == "__main__":
    unittest.main()
