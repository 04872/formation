from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.spatial import cKDTree
from .config import RADIUS

PERCEPTION_RADIUS = 3.0

@dataclass
class World:
    """Maintain the observed grid and reveal local ground truth around actual robots."""
    name: str
    res: float = 0.1
    size: tuple = (120, 80)
    obstacle_count: int = 50
    map_seed: int = 0
    pillar_width: float = 1.0

    def __post_init__(self):
        if self.name == 'random_pillars' and self.pillar_width <= 0:
            raise ValueError('Pillar width must be positive')
        if self.name in ('random_circles', 'random_pillars', 'history_deadend'):
            self.size = (500, 400)
        self.truth = np.zeros(self.size, dtype=bool)
        self.truth[:2, :] = self.truth[-2:, :] = True
        self.truth[:, :2] = self.truth[:, -2:] = True
        if self.name == 'narrow':
            self.truth[55:65, :33] = True
            self.truth[55:65, 47:] = True
        self.obstacle_centers = np.empty((0, 2))
        self.circles = np.empty((0, 3))
        self.rectangles = np.empty((0, 4))
        if self.name in ('random_circles', 'random_pillars'):
            self.start = np.array([3.0, 20.0])
            self.goal = np.array([47.0, 20.0])
            rng = np.random.default_rng(self.map_seed)
            centers = []
            while len(centers) < self.obstacle_count:
                center = rng.uniform([0.7, 0.7], [49.3, 39.3])
                if min(np.linalg.norm(center - self.start), np.linalg.norm(center - self.goal)) < 2.0:
                    continue
                centers.append(center)
            centers = np.asarray(centers, dtype=float).reshape(-1, 2)
            self.obstacle_centers = centers
            x = (np.arange(self.size[0]) + 0.5) * self.res
            y = (np.arange(self.size[1]) + 0.5) * self.res
            if self.name == 'random_circles':
                self.circles = np.column_stack((centers, np.full(len(centers),0.5)))
                for cx, cy, radius in self.circles:
                    self.truth |= (x[:, None] - cx)**2 + (y[None, :] - cy)**2 <= (radius + self.res*np.sqrt(2)/2)**2
            else:
                half = self.pillar_width/2
                self.rectangles = np.column_stack((centers[:,0]-half, centers[:,1]-half,
                                                   centers[:,0]+half, centers[:,1]+half))
                for xmin, ymin, xmax, ymax in self.rectangles:
                    # A blocked cell covers every grid square intersecting the box.
                    self.truth |= ((x[:,None] + self.res/2 >= xmin) & (x[:,None] - self.res/2 <= xmax) &
                                   (y[None,:] + self.res/2 >= ymin) & (y[None,:] - self.res/2 <= ymax))
        elif self.name == 'history_deadend':
            # A forward-facing cul-de-sac, with open routes around both sides.
            # Obstacles are physical rectangles as well as conservative grid cells.
            self.start = np.array([5.0, 20.0])
            self.goal = np.array([45.0, 20.0])
            self.rectangles = np.array([[14., 14., 31., 14.5],
                                        [14., 25.5, 31., 26.],
                                        [30.5, 14., 31., 26.]])
            x = (np.arange(self.size[0]) + 0.5) * self.res
            y = (np.arange(self.size[1]) + 0.5) * self.res
            for xmin, ymin, xmax, ymax in self.rectangles:
                self.truth |= ((x[:, None] + self.res/2 >= xmin) &
                               (x[:, None] - self.res/2 <= xmax) &
                               (y[None, :] + self.res/2 >= ymin) &
                               (y[None, :] - self.res/2 <= ymax))
        else:
            self.start = np.array([2.0, 4.0])
            self.goal = np.array([10.0, 4.0])
        self._dirty = True
        self.known = np.zeros(self.size, dtype=bool)
        self.occupied = np.zeros(self.size, dtype=bool)
        (self.xx, self.yy) = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')

    def observe(self, positions):
        radius = PERCEPTION_RADIUS
        for p in positions:
            lower = np.maximum(np.floor((p-radius)/self.res).astype(int)-1, 0)
            upper = np.minimum(np.ceil((p+radius)/self.res).astype(int)+2, self.size)
            if np.any(lower >= upper):
                continue
            x = np.arange(lower[0],upper[0]) * self.res + self.res / 2
            y = np.arange(lower[1],upper[1]) * self.res + self.res / 2
            visible = (x[:,None]-p[0])**2 + (y[None,:]-p[1])**2 <= radius**2
            window = np.s_[lower[0]:upper[0],lower[1]:upper[1]]
            known = self.known[window]
            if np.any(visible & ~known):
                self._dirty = True
            known |= visible
            occupied = self.occupied[window]
            occupied[visible] = self.truth[window][visible]

    def _refresh(self):
        if self._dirty:
            blocked = ~self.known | self.occupied
            self._blocked_mask = blocked
            self._dist = distance_transform_edt(~blocked) * self.res
            self._blocked_points = (np.argwhere(blocked) + 0.5) * self.res
            # For a query in a free cell a nearest blocked center exists on
            # the four-neighbor free/blocked boundary. Queries in blocked
            # cells use their own center, which is nearest on the full lattice.
            boundary = blocked & binary_dilation(~blocked)
            self._blocked_tree = cKDTree((np.argwhere(boundary)+.5)*self.res)
            self._dirty = False

    @property
    def dist(self):
        self._refresh()
        return self._dist

    @property
    def blocked_points(self):
        self._refresh()
        return self._blocked_points

    def clearance(self, points):
        """Conservative continuous distance to disks covering blocked grid cells."""
        self._refresh()
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 2)
        radius = self.res * np.sqrt(2) / 2
        outside = np.any((flat < 0) | (flat >= np.asarray(self.size) * self.res), axis=1)
        cells = np.floor(flat/self.res).astype(int)
        cells = np.clip(cells, 0, np.asarray(self.size)-1)
        free = ~self._blocked_mask[cells[:,0],cells[:,1]] & ~outside
        clearance = np.linalg.norm(flat-(cells+.5)*self.res,axis=1)-radius
        clearance[free] = self._blocked_tree.query(flat[free],workers=1)[0]-radius
        clearance[outside] = -radius
        return clearance.reshape(points.shape[:-1])

    def sfc(self, anchor, inflate=True):
        from .sfc import generate_sfc
        return generate_sfc(self.blocked_points, self.res, anchor, inflate)
