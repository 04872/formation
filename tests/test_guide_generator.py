from __future__ import annotations

import unittest

import numpy as np

from formation import FormationLibrary
from formation.guide_generator import GuideGenerator
from formation.types import AssignmentResult, FormationCandidateEvaluation, FormationScoreBreakdown


class GuideGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.library = FormationLibrary.build_default(robot_radius=0.09, inter_robot_margin=0.10)
        self.generator = GuideGenerator()

    def test_guide_matches_preview_samples_without_switch(self) -> None:
        square = self.library.get("square")
        evaluation = FormationCandidateEvaluation(
            formation_name=square.name,
            band_feasible=True,
            is_safe=True,
            score_breakdown=FormationScoreBreakdown(total_score=1.0),
            center_points_xy=[(0.0, 0.0), (1.0, 0.0)],
            heading_rads=[0.0, 0.0],
            slot_points_by_step_xy=[
                [tuple(slot) for slot in square.slots],
                [(1.0 + float(slot[0]), float(slot[1])) for slot in square.slots],
            ],
        )

        guide = self.generator.build(evaluation, square, current_formation=square)

        self.assertFalse(guide.switched)
        self.assertEqual(len(guide.guide_samples), 2)
        self.assertEqual(guide.transition_alphas, [1.0, 1.0])
        self.assertEqual(guide.guide_samples[0].robot_points_xy, evaluation.slot_points_by_step_xy[0])
        self.assertEqual(guide.guide_samples[1].robot_points_xy, evaluation.slot_points_by_step_xy[1])

    def test_guide_applies_target_slots_immediately_when_switching(self) -> None:
        square = self.library.get("square")
        compact = self.library.get("compact")
        assignment = AssignmentResult(assignment=(0, 1, 2, 3), total_cost=1.0, max_cost=0.4)
        centers = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
        headings = [0.0, 0.0, 0.0]
        slot_points = [
            [(center[0] + float(slot[0]), center[1] + float(slot[1])) for slot in compact.slots]
            for center in centers
        ]
        evaluation = FormationCandidateEvaluation(
            formation_name=compact.name,
            band_feasible=True,
            is_safe=True,
            score_breakdown=FormationScoreBreakdown(total_score=1.0),
            center_points_xy=centers,
            heading_rads=headings,
            slot_points_by_step_xy=slot_points,
            assignment=assignment,
        )

        guide = self.generator.build(evaluation, compact, current_formation=square)

        self.assertTrue(guide.switched)
        self.assertEqual(guide.transition_alphas, [1.0, 1.0, 1.0])
        # All samples use the target formation's slot positions; the assignment
        # permutes them to match robot order.
        self.assertEqual(guide.guide_samples[0].robot_points_xy, slot_points[0])
        self.assertEqual(guide.guide_samples[1].robot_points_xy, slot_points[1])
        self.assertEqual(guide.guide_samples[2].robot_points_xy, slot_points[2])


if __name__ == "__main__":
    unittest.main()
