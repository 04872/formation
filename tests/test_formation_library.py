from __future__ import annotations

import unittest

import numpy as np

from formation import FormationLibrary


class FormationLibraryTest(unittest.TestCase):
    def test_default_library_contains_expected_formations(self) -> None:
        library = FormationLibrary.build_default(robot_radius=0.18, inter_robot_margin=0.10)
        self.assertEqual(library.names(), ["column", "diamond", "horizontal_line", "square"])

        formations = {spec.name: spec for spec in library.list()}
        self.assertEqual(len(formations), 4)
        for spec in formations.values():
            self.assertEqual(spec.slots.shape, (4, 2))
            self.assertGreaterEqual(spec.lateral_half_width, 0.0)
            self.assertGreaterEqual(spec.longitudinal_half_length, 0.0)
            self.assertGreater(spec.bounding_radius, 0.0)
            self.assertGreater(spec.min_pairwise_distance, 0.0)
            self.assertEqual(spec.task_utility, 0.0)

        self.assertGreater(
            formations["horizontal_line"].lateral_half_width,
            formations["square"].lateral_half_width,
        )
        self.assertGreater(
            formations["column"].longitudinal_half_length,
            formations["diamond"].longitudinal_half_length,
        )

    def test_metadata_matches_expected_geometry(self) -> None:
        library = FormationLibrary.build_default(robot_radius=0.18, inter_robot_margin=0.10)
        square = library.get("square")
        diamond = library.get("diamond")

        self.assertAlmostEqual(square.lateral_half_width, 0.30)
        self.assertAlmostEqual(square.longitudinal_half_length, 0.30)
        self.assertAlmostEqual(square.bounding_radius, np.sqrt(0.18), places=6)
        self.assertAlmostEqual(square.min_pairwise_distance, 0.60)

        self.assertAlmostEqual(diamond.lateral_half_width, 0.45)
        self.assertAlmostEqual(diamond.longitudinal_half_length, 0.45)
        self.assertAlmostEqual(diamond.min_pairwise_distance, np.sqrt(0.405), places=6)


if __name__ == "__main__":
    unittest.main()
