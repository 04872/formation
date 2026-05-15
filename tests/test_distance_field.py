from __future__ import annotations

import unittest

import numpy as np

from formation.distance_field import compute_distance_field


class DistanceFieldTest(unittest.TestCase):
    def test_single_obstacle_matches_euclidean_distances(self) -> None:
        occupancy = np.zeros((5, 5), dtype=bool)
        occupancy[2, 2] = True

        distance_field = compute_distance_field(occupancy, resolution=1.0)

        expected = np.array(
            [
                [np.sqrt(8.0), np.sqrt(5.0), 2.0, np.sqrt(5.0), np.sqrt(8.0)],
                [np.sqrt(5.0), np.sqrt(2.0), 1.0, np.sqrt(2.0), np.sqrt(5.0)],
                [2.0, 1.0, 0.0, 1.0, 2.0],
                [np.sqrt(5.0), np.sqrt(2.0), 1.0, np.sqrt(2.0), np.sqrt(5.0)],
                [np.sqrt(8.0), np.sqrt(5.0), 2.0, np.sqrt(5.0), np.sqrt(8.0)],
            ]
        )
        np.testing.assert_allclose(distance_field, expected, rtol=0.0, atol=1e-9)

    def test_empty_occupancy_returns_inf(self) -> None:
        occupancy = np.zeros((3, 4), dtype=bool)
        distance_field = compute_distance_field(occupancy, resolution=0.5)
        self.assertTrue(np.all(np.isinf(distance_field)))


if __name__ == "__main__":
    unittest.main()
