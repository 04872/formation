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
    steps: int = 35
    samples: int = 1120
    horizon: int = 35
    goal_weight: float = 2.0
    suspension_margin: float = 30.0
    mppi_temperature: float = 1000.0
    mppi_noise_std: tuple = (0.22, 0.9)
CONFIG = SimulationConfig()
DT = CONFIG.dt
RADIUS = CONFIG.robot_radius
REF = np.array([[-0.65, -0.65], [-0.65, 0.65], [0.65, -0.65], [0.65, 0.65]])

LEADER_REF = REF - REF[0]
