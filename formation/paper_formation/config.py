from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SimulationConfig:
    """Shared physical and rollout parameters."""
    dt: float = 0.2
    robot_radius: float = 0.16
    replan_steps: int = 5
    max_speed: float = 0.65
    max_angular_speed: float = 1.5
    sequences: int = 20
    candidates: int = 256
    uniform_min_radius: float = 0.3
    steps: int = 35
    samples: int = 1120
    horizon: int = 35
    goal_weight: float = 2.0
    suspension_margin: float = 30.0
    guidance_weight: float = 2000.0
    smoothness_weight: float = 1000.0
    mppi_temperature: float = 1000.0
    mppi_noise_std: tuple = (0.22, 0.9)
CONFIG = SimulationConfig()
DT = CONFIG.dt
RADIUS = CONFIG.robot_radius
REF = np.array([[-0.65, -0.65], [-0.65, 0.65], [0.65, -0.65], [0.65, 0.65]])

LEADER_REF = REF - REF[0]

@dataclass(frozen=True)
class FormationGeometry:
    """Immutable reference geometry; spacing is a chosen experiment parameter."""
    name: str = 'square4'
    spacing: float = 1.3
    leader_slot: int = 0

    def __post_init__(self):
        if self.spacing <= 0:
            raise ValueError('Formation spacing must be positive')
        if self.name == 'square4':
            d = self.spacing/2
            points = np.array([[-d,-d],[-d,d],[d,-d],[d,d]])
        elif self.name == 'triangle6':
            d = self.spacing
            points = np.array([[np.sqrt(3)*d,0], [np.sqrt(3)*d/2,-d/2],
                               [np.sqrt(3)*d/2,d/2], [0,-d], [0,0], [0,d]])
            points -= points.mean(axis=0)
        else:
            raise ValueError(f'Unknown formation: {self.name}')
        # Arrays backed by immutable bytes cannot be made writable again by callers.
        points = np.frombuffer(points.tobytes(), dtype=points.dtype).reshape(points.shape)
        anchored_values = points - points[self.leader_slot]
        anchored = np.frombuffer(anchored_values.tobytes(), dtype=points.dtype).reshape(points.shape)
        distances = np.linalg.norm(points[:,None,:]-points[None,:,:],axis=-1)
        pair_min = np.min(distances[np.triu_indices(len(points),1)])
        object.__setattr__(self,'centered_reference',points)
        object.__setattr__(self,'leader_reference',anchored)
        object.__setattr__(self,'pair_min',float(pair_min))


DEFAULT_GEOMETRY = FormationGeometry()
