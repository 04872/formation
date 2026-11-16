"""Paper Eq. (4): position, velocity, acceleration and yaw flat outputs."""
import numpy as np


def integrate_uav(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """Advance Eq. (4) with the forward Euler discretization in Eq. (23).

    State order is ``[px, py, pz, vx, vy, vz, ax, ay, az, yaw]``;
    control order is ``[jerk_x, jerk_y, jerk_z, yaw_rate]``. Leading
    dimensions broadcast, allowing both a single state and sampled rollouts.
    """
    x, u = np.broadcast_arrays(np.asarray(state)[..., :, None],
                               np.asarray(control)[..., None, :])
    state_b = x[..., :, 0]
    control_b = u[..., 0, :]
    result = np.array(state_b, dtype=float, copy=True)
    result[..., :3] += dt * state_b[..., 3:6]
    result[..., 3:6] += dt * state_b[..., 6:9]
    result[..., 6:9] += dt * control_b[..., :3]
    result[..., 9] += dt * control_b[..., 3]
    return result

