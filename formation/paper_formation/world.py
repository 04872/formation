from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt
from .config import RADIUS

@dataclass
class World:
    """Maintain the observed grid and reveal local ground truth around actual robots."""
    name: str
    res: float = 0.1
    size: tuple = (120, 80)
    obstacle_count: int = 50
    map_seed: int = 0

    def __post_init__(self):
        if self.name == 'random_circles':
            self.size = (500, 400)
        self.truth = np.zeros(self.size, dtype=bool)
        self.truth[:2, :] = self.truth[-2:, :] = True
        self.truth[:, :2] = self.truth[:, -2:] = True
        if self.name == 'narrow':
            self.truth[55:65, :33] = True
            self.truth[55:65, 47:] = True
        self.circles = np.empty((0, 3))
        if self.name == 'random_circles':
            self.start = np.array([3.0, 20.0])
            self.goal = np.array([47.0, 20.0])
            rng = np.random.default_rng(self.map_seed)
            circles = []
            while len(circles) < self.obstacle_count:
                center = rng.uniform([0.7, 0.7], [49.3, 39.3])
                if min(np.linalg.norm(center - self.start), np.linalg.norm(center - self.goal)) < 2.0:
                    continue
                circles.append([*center, 0.5])
            self.circles = np.asarray(circles, dtype=float).reshape(-1, 3)
            x = (np.arange(self.size[0]) + 0.5) * self.res
            y = (np.arange(self.size[1]) + 0.5) * self.res
            for cx, cy, radius in self.circles:
                self.truth |= (x[:, None] - cx)**2 + (y[None, :] - cy)**2 <= (radius + self.res*np.sqrt(2)/2)**2
        else:
            self.start = np.array([2.0, 4.0])
            self.goal = np.array([10.0, 4.0])
        self.known = np.zeros(self.size, dtype=bool)
        self.occupied = np.zeros(self.size, dtype=bool)
        (self.xx, self.yy) = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')

    def observe(self, positions):
        for p in positions:
            d2 = (self.xx * self.res + self.res / 2 - p[0]) ** 2 + (self.yy * self.res + self.res / 2 - p[1]) ** 2
            visible = d2 <= 3.0 ** 2
            self.known |= visible
            self.occupied[visible] = self.truth[visible]
        blocked = ~self.known | self.occupied
        self.dist = distance_transform_edt(~blocked) * self.res
        self.blocked_points = (np.argwhere(blocked) + 0.5) * self.res

    def clearance(self, points):
        idx = np.floor(np.asarray(points) / self.res).astype(int)
        idx = np.clip(idx, [0, 0], np.array(self.size) - 1)
        return self.dist[idx[..., 0], idx[..., 1]] - self.res * np.sqrt(2) / 2

    def sfc(self, anchor, inflate=True):
        from .sfc import generate_sfc
        return generate_sfc(self.blocked_points, self.res, anchor, inflate)
