from __future__ import annotations

import unittest

import numpy as np

from formation import FormationLibrary


class FormationLibraryTest(unittest.TestCase):
    def test_default_library_contains_expected_formations(self) -> None:
        library = FormationLibrary.build_default(robot_radius=0.09, inter_robot_margin=0.10)
        self.assertEqual(library.names(), ["column", "horizontal_line", "square", "t_shape"])

        formations = {spec.name: spec for spec in library.list()}
        self.assertEqual(len(formations), 4)
        for spec in formations.values():
            self.assertEqual(spec.slots.shape, (4, 2))
            self.assertGreaterEqual(spec.lateral_half_width, 0.0)
            self.assertGreaterEqual(spec.longitudinal_half_length, 0.0)
            self.assertGreater(spec.bounding_radius, 0.0)
            self.assertGreater(spec.min_pairwise_distance, 0.0)
        self.assertAlmostEqual(formations["square"].task_utility, formations["square"].lateral_half_width + 1.0)

        self.assertGreater(
            formations["horizontal_line"].lateral_half_width,
            formations["square"].lateral_half_width,
        )
        self.assertGreater(
            formations["column"].longitudinal_half_length,
            formations["t_shape"].longitudinal_half_length,
        )
        self.assertGreater(
            formations["t_shape"].lateral_half_width,
            formations["square"].lateral_half_width,
        )

    def test_metadata_matches_expected_geometry(self) -> None:
        library = FormationLibrary.build_default(robot_radius=0.09, inter_robot_margin=0.10)
        square = library.get("square")
        t_shape = library.get("t_shape")

        self.assertAlmostEqual(square.lateral_half_width, 0.25)
        self.assertAlmostEqual(square.longitudinal_half_length, 0.25)
        self.assertAlmostEqual(square.bounding_radius, np.sqrt(0.125), places=6)
        self.assertAlmostEqual(square.min_pairwise_distance, 0.50)

        self.assertAlmostEqual(t_shape.lateral_half_width, 0.50)
        self.assertAlmostEqual(t_shape.longitudinal_half_length, 0.35)
        self.assertAlmostEqual(t_shape.bounding_radius, np.sqrt(0.2725), places=6)
        self.assertAlmostEqual(t_shape.min_pairwise_distance, 0.50)
        np.testing.assert_allclose(
            t_shape.slots,
            np.asarray(
                [
                    [0.35, 0.00],
                    [-0.15, -0.50],
                    [-0.15, 0.00],
                    [-0.15, 0.50],
                ],
                dtype=float,
            ),
        )

        self.assertAlmostEqual(t_shape.min_pairwise_distance, square.min_pairwise_distance)
        self.assertGreater(t_shape.lateral_half_width, square.lateral_half_width)
        self.assertGreater(t_shape.longitudinal_half_length, square.longitudinal_half_length)
        self.assertGreater(t_shape.bounding_radius, square.bounding_radius)
        self.assertLess(t_shape.bounding_radius, library.get("column").bounding_radius)
        self.assertLess(t_shape.longitudinal_half_length, library.get("column").longitudinal_half_length)
        self.assertLess(t_shape.bounding_radius, library.get("horizontal_line").bounding_radius)
        self.assertAlmostEqual(t_shape.task_utility, t_shape.lateral_half_width)
        self.assertEqual(t_shape.name, "t_shape")


if __name__ == "__main__":
    unittest.main()
