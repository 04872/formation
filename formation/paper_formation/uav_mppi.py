"""Distributed MPPI for the paper's ten-state flat-output UAV model."""
from dataclasses import dataclass
import numpy as np
from .paper_costs import guidance_cost, dynamic_cost, smoothness_cost, distance_penalty


@dataclass(frozen=True)
class UAVMPPIConfig:
    """UAV parameters; noise, temperature, acceleration and downwash are assumptions."""
    acceleration_limit: float = 1.0
    noise_std: tuple[float, float, float, float] = (0.22, 0.22, 0.0, 0.9)
    temperature: float = 1000.0
    downwash_factor: float = 0.5
    dt: float = 0.2
    vmax: float = 1.5
    guidance_weight: float = 2000.0
    smoothness_weight: float = 1000.0
    execution_substeps: int = 1


def _held_euler(state, control, dt, substeps):
    """Equivalent repeated Euler substeps with a constant control."""
    h=dt/substeps;n=substeps
    result=np.array(state,copy=True)
    result[..., :3] += dt*state[..., 3:6] + (n*(n-1)/2)*h*h*state[..., 6:9] + (n*(n-1)*(n-2)/6)*h**3*control[..., :3]
    result[..., 3:6] += dt*state[..., 6:9] + (n*(n-1)/2)*h*h*control[..., :3]
    result[..., 6:9] += dt*control[..., :3]
    result[..., 9] += dt*control[..., 3]
    return result


def optimize_uav(states, guidance, nominal, previous_predictions, clearance, rng,
                 samples, config: UAVMPPIConfig):
    """Optimize each UAV against other UAVs' previous predicted positions.

    ``clearance`` maps position arrays (..., 3) to obstacle distances. The
    nominal controls receive weighted *raw* Gaussian noise, as in Algorithm 2.
    No hard control or state clipping is imposed by this model.
    """
    count, horizon, _ = nominal.shape
    optimized = np.empty_like(nominal)
    predicted = np.empty((count, horizon, 3))
    diagnostics = np.empty((count, 2))
    for i in range(count):
        noise = rng.normal(0., config.noise_std, (samples, horizon, 4))
        controls = nominal[i][None, :, :] + noise
        state = np.repeat(states[i][None, :], samples, axis=0)
        costs = np.zeros(samples)
        for t in range(horizon):
            state = _held_euler(state, controls[:, t], config.dt, config.execution_substeps)
            costs += guidance_cost(state[:, :3], guidance[t, i], config.guidance_weight)
            costs += dynamic_cost(np.linalg.norm(state[:, 3:6], axis=-1),
                                  np.linalg.norm(state[:, 6:9], axis=-1),
                                  config.vmax, config.acceleration_limit)
            costs += smoothness_cost(controls[:, t], config.smoothness_weight)
            costs += distance_penalty(clearance(state[:, :3]), .4, .8, 2600., 1.8)
            for j in range(count):
                if j != i:
                    delta = state[:, :3] - previous_predictions[j, t]
                    delta[:, 2] *= config.downwash_factor
                    costs += distance_penalty(np.linalg.norm(delta, axis=-1),
                                              .5, 1.2, 1600., 1.5)
        weights = np.exp(-config.dt * (costs - costs.min()) / config.temperature)
        weights /= weights.sum()
        optimized[i] = nominal[i] + np.einsum('k,ktd->td', weights, noise)
        state = states[i].copy()
        for t in range(horizon):
            state = _held_euler(state, optimized[i, t], config.dt, config.execution_substeps)
            predicted[i, t] = state[:3]
        diagnostics[i] = [1. / np.sum(weights**2), costs.min() * config.dt]
    return optimized, predicted, diagnostics


class DistributedUAVMPPI:
    """Maintain the nominal sequence and previous predicted peer trajectories."""
    def __init__(self, rng, states, samples, horizon, config: UAVMPPIConfig):
        self.rng, self.samples, self.horizon, self.config = rng, samples, horizon, config
        self.nominal = np.zeros((len(states), horizon, 4))
        self.previous_predictions = np.repeat(states[:, None, :3], horizon, axis=1)
        self.u_init = np.zeros((len(states), 4))

    def step(self, states, guidance, clearance, optimize_now=True):
        if optimize_now:
            controls, predictions, self.last_diagnostics = optimize_uav(
                states, guidance, self.nominal, self.previous_predictions, clearance,
                self.rng, self.samples, self.config)
        else:
            controls=self.nominal
            predictions=self.previous_predictions
        self.nominal = np.concatenate((controls[:, 1:], self.u_init[:, None, :]), axis=1)
        self.previous_predictions = np.concatenate((predictions[:, 1:], predictions[:, -1:]), axis=1)
        return controls[:, 0]
