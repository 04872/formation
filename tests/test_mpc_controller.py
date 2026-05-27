from __future__ import annotations

import unittest

from formation import (
    ControlCommand,
    DistributedFormationMPC,
    FormationGuide,
    GuideSample,
    MPCConfig,
    MapBuilder,
    RightAngleCorridorConfig,
    RobotState,
)


class DistributedFormationMPCTest(unittest.TestCase):
    def setUp(self) -> None:
        self.map_data = MapBuilder().build("right_angle_corridor", RightAngleCorridorConfig(robot_radius=0.09))
        self.config = MPCConfig(
            dt=0.2,
            horizon_steps=4,
            v_max=0.6,
            omega_max=1.0,
            robot_radius=0.09,
            safety_margin=0.06,
            inter_robot_margin=0.10,
        )
        self.controller = DistributedFormationMPC(self.config)
        from formation.controller_reference import ControllerReferenceBuilder
        self.reference_builder = ControllerReferenceBuilder(self.config)

    def _build_reference(self):
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
        return self.reference_builder.build(guide)

    def test_zero_error_state_tracks_reference(self) -> None:
        controller_reference = self._build_reference()
        own_reference = controller_reference.robot_trajectories[0].window(0, self.config.horizon_steps)
        neighbor_ref = controller_reference.robot_trajectories[1].window(0, self.config.horizon_steps)
        from formation.types import RobotPrediction
        neighbor_prediction = RobotPrediction(
            robot_index=1,
            positions_xy=[sample.position_xy for sample in neighbor_ref.samples],
            yaw_rads=[sample.yaw for sample in neighbor_ref.samples],
            commands=[ControlCommand(v=sample.v_ref, omega=sample.omega_ref) for sample in neighbor_ref.samples[:-1]],
        )

        command, prediction = self.controller.solve_robot(
            RobotState(x=-4.2, y=-3.8, yaw=0.0),
            own_reference,
            [neighbor_prediction],
            self.map_data,
            solver_slot=0,
            neighbor_indices=[1],
        )

        self.assertLess(abs(command.v - own_reference.samples[0].v_ref), 0.50)
        self.assertLess(abs(command.omega), 0.50)
        self.assertEqual(prediction.sample_count, self.config.horizon_steps + 1)
        self.assertTrue(all(value == 0.0 for value in prediction.obstacle_slacks))
        self.assertEqual(prediction.metadata["neighbor_robot_indices"], [1])
        self.assertIn("solver_status", prediction.metadata)
        self.assertGreaterEqual(prediction.metadata["objective"], 0.0)
        self.assertTrue(prediction.metadata["solver_status"])

    def test_heading_error_generates_corrective_omega(self) -> None:
        controller_reference = self._build_reference()
        own_reference = controller_reference.robot_trajectories[0].window(0, self.config.horizon_steps)
        neighbor_ref = controller_reference.robot_trajectories[1].window(0, self.config.horizon_steps)
        from formation.types import RobotPrediction
        neighbor_prediction = RobotPrediction(
            robot_index=1,
            positions_xy=[sample.position_xy for sample in neighbor_ref.samples],
            yaw_rads=[sample.yaw for sample in neighbor_ref.samples],
            commands=[ControlCommand(v=sample.v_ref, omega=sample.omega_ref) for sample in neighbor_ref.samples[:-1]],
        )

        command, _ = self.controller.solve_robot(
            RobotState(x=-4.2, y=-3.8, yaw=0.4),
            own_reference,
            [neighbor_prediction],
            self.map_data,
            solver_slot=0,
            neighbor_indices=[1],
        )

        self.assertGreaterEqual(command.v, 0.0)
        self.assertLess(abs(command.omega), self.config.omega_max + 1e-6)

    def test_close_neighbor_activates_slack(self) -> None:
        controller_reference = self._build_reference()
        own_reference = controller_reference.robot_trajectories[0].window(0, self.config.horizon_steps)
        from formation.types import RobotPrediction
        close_neighbor = RobotPrediction(
            robot_index=1,
            positions_xy=[(-4.18, -3.78)] * (self.config.horizon_steps + 1),
            yaw_rads=[0.0] * (self.config.horizon_steps + 1),
            commands=[ControlCommand(v=0.0, omega=0.0)] * self.config.horizon_steps,
        )

        _, prediction = self.controller.solve_robot(
            RobotState(x=-4.2, y=-3.8, yaw=0.0),
            own_reference,
            [close_neighbor],
            self.map_data,
            solver_slot=0,
            neighbor_indices=[1],
        )

        self.assertEqual(prediction.metadata["neighbor_robot_indices"], [1])
        self.assertTrue(all(value == 0.0 for value in prediction.obstacle_slacks))
        self.assertEqual(len(prediction.neighbor_slacks), self.config.horizon_steps)
        self.assertTrue(all(len(step) == 1 for step in prediction.neighbor_slacks))

    def test_all_controls_respect_bounds(self) -> None:
        from formation import FormationSelector, GlobalPlanner, PathManager, PreviewCurvePlanner, FormationLibrary
        lib = FormationLibrary.build_default(robot_radius=0.09, inter_robot_margin=0.10)
        mp = MapBuilder().build("right_angle_corridor", RightAngleCorridorConfig(robot_radius=0.09))
        gp = GlobalPlanner().plan(mp)
        preview = PreviewCurvePlanner().plan(mp, mp.start_xy, PathManager(gp).get_local_path_window(mp.start_xy, preview_distance_m=3.5))
        selection = FormationSelector().select_target_formation(mp, preview, lib.list(), 0.09, 0.06)
        ref = self.reference_builder.build(selection.guide).window(0, self.config.horizon_steps)

        states = [
            RobotState(x=s.position_xy[0], y=s.position_xy[1], yaw=s.yaw)
            for s in (t.samples[0] for t in ref.robot_trajectories)
        ]
        commands, predictions = self.controller.solve_all(states, ref, mp)

        self.assertEqual(len(commands), ref.robot_count)
        self.assertEqual(len(predictions), ref.robot_count)
        for command in commands:
            self.assertGreaterEqual(command.v, -1e-6)
            self.assertLessEqual(command.v, self.config.v_max + 1e-6)
            self.assertLessEqual(abs(command.omega), self.config.omega_max + 1e-6)

    def test_parallel_and_serial_solve_shapes_match(self) -> None:
        from formation import FormationSelector, GlobalPlanner, PathManager, PreviewCurvePlanner, FormationLibrary
        lib = FormationLibrary.build_default(robot_radius=0.09, inter_robot_margin=0.10)
        mp = MapBuilder().build("right_angle_corridor", RightAngleCorridorConfig(robot_radius=0.09))
        gp = GlobalPlanner().plan(mp)
        preview = PreviewCurvePlanner().plan(mp, mp.start_xy, PathManager(gp).get_local_path_window(mp.start_xy, preview_distance_m=3.5))
        selection = FormationSelector().select_target_formation(mp, preview, lib.list(), 0.09, 0.06)
        ref = self.reference_builder.build(selection.guide).window(0, self.config.horizon_steps)
        states = [
            RobotState(x=s.position_xy[0], y=s.position_xy[1], yaw=s.yaw)
            for s in (t.samples[0] for t in ref.robot_trajectories)
        ]
        serial_controller = DistributedFormationMPC(
            MPCConfig(**{**self.config.__dict__, "parallel_solve": False})
        )
        parallel_controller = DistributedFormationMPC(
            MPCConfig(**{**self.config.__dict__, "parallel_solve": True, "parallel_workers": 2})
        )

        serial_commands, serial_predictions = serial_controller.solve_all(states, ref, mp)
        parallel_commands, parallel_predictions = parallel_controller.solve_all(states, ref, mp)

        self.assertEqual(len(serial_commands), len(parallel_commands))
        self.assertEqual(len(serial_predictions), len(parallel_predictions))
        self.assertEqual(
            [p.robot_index for p in serial_predictions],
            [p.robot_index for p in parallel_predictions],
        )


if __name__ == "__main__":
    unittest.main()
