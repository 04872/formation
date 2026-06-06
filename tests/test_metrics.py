"""Unit tests for normalized Laplacian formation similarity error."""

from __future__ import annotations

import math
import unittest

import numpy as np

from formation.metrics import (
    _compute_e_dist,
    _compute_e_sim,
    _compute_e_track,
    _normalized_laplacian,
)


class NormalizedLaplacianTest(unittest.TestCase):
    def test_translation_invariance(self) -> None:
        """e_sim should be 0 when the point set is rigidly translated."""
        P = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.5, 0.289]])
        t = np.array([2.5, -1.7])
        Pref = P + t
        e = _compute_e_sim(P, Pref)
        self.assertLess(e, 1e-10, f"translation invariance violated: e_sim={e}")

    def test_rotation_invariance(self) -> None:
        """e_sim should be near 0 when the point set is rigidly rotated."""
        P = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.5, 0.289]])
        angle = math.radians(57)
        R = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
        Pref = P @ R.T
        e = _compute_e_sim(P, Pref)
        self.assertLess(e, 1e-8, f"rotation invariance violated: e_sim={e}")

    def test_scale_invariance(self) -> None:
        """e_sim should be near 0 for uniform scaling (normalized Laplacian)."""
        P = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.5, 0.289]])
        s = 3.7
        Pref = P * s
        e = _compute_e_sim(P, Pref)
        self.assertLess(e, 1e-8, f"scale invariance violated: e_sim={e}")

    def test_deformation_nonzero(self) -> None:
        """e_sim should be > 0 when the point set is deformed."""
        P = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.5, 0.289]])
        Pref = P.copy()
        Pref[0, 1] += 0.5  # displace one robot
        e = _compute_e_sim(P, Pref)
        self.assertGreater(e, 0.0, f"deformation should give nonzero e_sim")

    def test_identical_points_zero(self) -> None:
        """Identical point sets should yield exactly 0."""
        P = np.array([[0.0, 0.0], [1.0, 0.5], [-0.3, 0.8], [0.6, -0.4]])
        e = _compute_e_sim(P, P)
        self.assertEqual(e, 0.0)

    def test_colinear_no_nan(self) -> None:
        """Colinear robots should not produce NaN (degree > 0)."""
        P = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        Pref = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.5, 0.5]])
        e = _compute_e_sim(P, Pref)
        self.assertFalse(math.isnan(e), "e_sim should not be NaN for colinear points")


class TrackingErrorTest(unittest.TestCase):
    def test_zero_when_matching(self) -> None:
        P = np.array([[0.0, 0.0], [1.0, 0.0]])
        e = _compute_e_track(P, P)
        self.assertEqual(e, 0.0)

    def test_positive_when_mismatch(self) -> None:
        P = np.array([[0.0, 0.0], [1.0, 0.0]])
        Pref = np.array([[0.0, 0.1], [1.0, 0.0]])
        e = _compute_e_track(P, Pref)
        self.assertGreater(e, 0.0)


class DistanceErrorTest(unittest.TestCase):
    def test_zero_when_matching(self) -> None:
        P = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.5, 0.289]])
        e = _compute_e_dist(P, P)
        self.assertEqual(e, 0.0)

    def test_positive_when_stretched(self) -> None:
        P = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.5, 0.289]])
        Pref = P * 1.5
        e = _compute_e_dist(P, Pref)
        self.assertGreater(e, 0.0)


class LaplacianPropertyTest(unittest.TestCase):
    def test_zero_degree_handling(self) -> None:
        """A robot at exactly the same position as another yields W_ij=0 for
        that pair, but degree comes from other robots — so D_ii > 0 still.
        Test that colocated robots do not cause NaN."""
        P = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
        L = _normalized_laplacian(P)
        self.assertFalse(np.any(np.isnan(L)), "NaN in Laplacian with colocated robots")


if __name__ == "__main__":
    unittest.main()
