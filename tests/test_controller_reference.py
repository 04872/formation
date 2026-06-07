from __future__ import annotations

import unittest

from formation import (
    ControllerReferenceBuilder,
    FormationGuide,
    GuideSample,
    MPCConfig,
)


class ControllerReferenceBuilderTest(unittest.TestCase):
    def test_builds_per_robot_reference_trajectories(self) -> None:
        guide = FormationGuide(
            formation_name="square",
            guide_samples=[
                GuideSample(center_xy=(0.0, 0.0), heading_rad=0.0, robot_points_xy=[(-0.3, -0.3), (0.3, 0.3)]),
                GuideSample(center_xy=(0.2, 0.0), heading_rad=0.0, robot_points_xy=[(-0.1, -0.3), (0.5, 0.3)]),
                GuideSample(center_xy=(0.4, 0.0), heading_rad=0.0, robot_points_xy=[(0.1, -0.3), (0.7, 0.3)]),
            ],
            transition_alphas=[1.0, 1.0, 1.0],
        )
        config = MPCConfig(dt=0.2, horizon_steps=2, v_max=2.0, omega_max=2.0)

        controller_reference = ControllerReferenceBuilder(config).build(guide)

        self.assertEqual(controller_reference.robot_count, 2)
        self.assertEqual(controller_reference.sample_count, 3)
        self.assertEqual(controller_reference.transition_alphas, [1.0, 1.0, 1.0])
        self.assertEqual(controller_reference.robot_trajectories[0].sample_count, 3)
        self.assertAlmostEqual(controller_reference.robot_trajectories[0].samples[0].v_ref, 1.0)  # raw=1.0 < clip=1.1, not clipped
        self.assertAlmostEqual(controller_reference.robot_trajectories[0].samples[0].omega_ref, 0.0)

    def test_turning_guide_produces_nonzero_omega_reference(self) -> None:
        guide = FormationGuide(
            formation_name="turn",
            guide_samples=[
                GuideSample(center_xy=(0.0, 0.0), heading_rad=0.0, robot_points_xy=[(0.0, 0.0)]),
                GuideSample(center_xy=(0.1, 0.0), heading_rad=0.2, robot_points_xy=[(0.1, 0.0)]),
                GuideSample(center_xy=(0.2, 0.05), heading_rad=0.4, robot_points_xy=[(0.2, 0.05)]),
            ],
            transition_alphas=[0.0, 0.5, 1.0],
            switched=True,
        )
        config = MPCConfig(dt=0.1, horizon_steps=2, v_max=2.0, omega_max=5.0)

        controller_reference = ControllerReferenceBuilder(config).build(guide)

        self.assertEqual(controller_reference.transition_alphas, [0.0, 0.5, 1.0])
        self.assertGreater(abs(controller_reference.robot_trajectories[0].samples[0].omega_ref), 0.0)

    def test_clips_reference_inputs_to_limits(self) -> None:
        guide = FormationGuide(
            formation_name="fast",
            guide_samples=[
                GuideSample(center_xy=(0.0, 0.0), heading_rad=0.0, robot_points_xy=[(0.0, 0.0)]),
                GuideSample(center_xy=(1.0, 0.0), heading_rad=0.0, robot_points_xy=[(1.0, 0.0)]),
                GuideSample(center_xy=(1.0, 1.0), heading_rad=1.57, robot_points_xy=[(1.0, 1.0)]),
            ],
            transition_alphas=[1.0] * 3,
        )
        config = MPCConfig(dt=1.0, horizon_steps=2, v_max=0.4, omega_max=0.2)

        controller_reference = ControllerReferenceBuilder(config).build(guide)
        sample = controller_reference.robot_trajectories[0].samples[0]

        # v_ref: robot moves 1 m/s, clipped by ρ_v=0.55 → 0.22
        self.assertAlmostEqual(sample.v_ref, 0.22)
        # omega_ref: heading turns 1.57 rad/s, clipped by ρ_ω=0.75 → 0.15
        self.assertAlmostEqual(sample.omega_ref, 0.15)


if __name__ == "__main__":
    unittest.main()
