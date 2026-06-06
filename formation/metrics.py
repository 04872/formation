"""Experiment metrics logger for multi-robot formation navigation.

Per-step metrics are accumulated via :meth:`ScenarioMetricsLogger.update`.
After the trial ends, :func:`finalize` returns a summary dict, and the
stored time-series are available for CSV export.

Does NOT modify planner, formation selection, reference generation, or MPC behaviour.
"""

from __future__ import annotations

import math

import numpy as np


def _normalized_laplacian(positions: np.ndarray) -> np.ndarray:
    """Normalised Laplacian of a complete weighted graph whose edge weights
    are Euclidean distances.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        xy coordinates of N robots.

    Returns
    -------
    L_norm : np.ndarray, shape (N, N)
        I - D^{-1/2} W D^{-1/2}
    """
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    W = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(W, 0.0)
    D = W.sum(axis=1)
    D_inv_sqrt = np.where(D > 1e-9, 1.0 / np.sqrt(D), 0.0)
    L = np.eye(N) - D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :]
    return L


def _compute_e_track(P: np.ndarray, Pref: np.ndarray) -> float:
    """Per-step reference tracking error (metres).

    e_track = (1/N) * sum_i || p_i - p_i_ref ||
    """
    return float(np.mean(np.linalg.norm(P - Pref, axis=1)))


def _compute_e_dist(P: np.ndarray, Pref: np.ndarray) -> float:
    """Per-step formation distance error (metres).

    e_dist = mean_{i<j} | d_ij - d_ij_ref |
    """
    N = P.shape[0]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    d_ref = np.linalg.norm(Pref[:, None, :] - Pref[None, :, :], axis=-1)
    iu = np.triu_indices(N, k=1)
    return float(np.mean(np.abs(d[iu] - d_ref[iu])))


def _compute_e_sim(P: np.ndarray, Pref: np.ndarray) -> float:
    """Per-step formation similarity error (dimensionless, Frobenius norm)."""
    L_P = _normalized_laplacian(P)
    L_Pref = _normalized_laplacian(Pref)
    return float(np.linalg.norm(L_P - L_Pref, "fro"))


class ScenarioMetricsLogger:
    """Accumulates per-step metrics for a single deterministic scenario run."""

    def __init__(
        self,
        robot_count: int,
        control_dt: float,
        robot_radius: float,
        scenario_name: str,
        *,
        map_data=None,
    ) -> None:
        self.robot_count = robot_count
        self.control_dt = control_dt
        self.robot_radius = robot_radius
        self.scenario_name = scenario_name
        self._map_data = map_data

        # ── aggregated flags ───────────────────────────────────────
        self.reached_goal = False
        self.obstacle_collision = False
        self.inter_robot_collision = False
        self.planning_failure = False
        self.control_failure = False
        self.mpc_fallback_count = 0
        self.T_goal: float | None = None

        # ── per-step time-series ───────────────────────────────────
        self.step_indices: list[int] = []
        self.step_times: list[float] = []
        self.e_track_series: list[float] = []
        self.e_dist_series: list[float] = []
        self.e_sim_series: list[float] = []
        self.formation_name_series: list[str] = []
        self.reached_goal_flags: list[bool] = []
        self.obstacle_collision_flags: list[bool] = []
        self.inter_robot_collision_flags: list[bool] = []

    # ── public API ─────────────────────────────────────────────────

    def update(
        self,
        global_step: int,
        sim_time: float,
        states: "list",
        controller_reference,  # FormationControllerReference | None
        reached_goal: bool,
    ) -> None:
        """Record metrics for one control step.

        Parameters
        ----------
        global_step : int
            Cumulative step index (does NOT reset across cycles).
        sim_time : float
            Current simulation time (global_step * dt).
        states : list[RobotState]
            Actual robot states at this step.
        controller_reference : FormationControllerReference or None
            Reference for the current step.  ``samples[0]`` of each
            trajectory is used as the reference position.
        reached_goal : bool
            Whether the formation centroid is within goal tolerance.
        """
        self.step_indices.append(global_step)
        self.step_times.append(sim_time)
        self.reached_goal_flags.append(reached_goal)

        if reached_goal and self.T_goal is None:
            self.T_goal = sim_time
            self.reached_goal = True

        # ── collision detection (internal, no external pre-computation needed) ──
        obs_col = self._check_obstacle_collision(states)
        int_col = self._check_inter_robot_collision(states)
        self.obstacle_collision_flags.append(obs_col)
        self.inter_robot_collision_flags.append(int_col)
        if obs_col:
            self.obstacle_collision = True
        if int_col:
            self.inter_robot_collision = True

        # ── formation errors (skip if reference is unavailable) ─────
        if controller_reference is not None and controller_reference.robot_trajectories:
            try:
                Pref = np.array(
                    [traj.samples[0].position_xy for traj in controller_reference.robot_trajectories],
                    dtype=float,
                )
                P = np.array([[s.x, s.y] for s in states], dtype=float)

                self.e_track_series.append(_compute_e_track(P, Pref))
                self.e_dist_series.append(_compute_e_dist(P, Pref))
                self.e_sim_series.append(_compute_e_sim(P, Pref))
            except (IndexError, AttributeError):
                self._append_nan_errors()
                self.set_planning_failure()
        else:
            self._append_nan_errors()
            self.set_planning_failure()

        # ── formation name ─────────────────────────────────────────
        fm_name = controller_reference.formation_name if controller_reference is not None else ""
        self.formation_name_series.append(fm_name)

    def set_planning_failure(self) -> None:
        self.planning_failure = True

    def set_control_failure(self) -> None:
        self.control_failure = True

    def record_mpc_fallback(self, count: int = 1) -> None:
        self.mpc_fallback_count += count

    def finalize(self, *, timeout: bool = False) -> dict:
        """Return per-scenario summary dict and freeze time-series."""
        success, failure_reason = self._compute_success(timeout)

        e_track_mean = _nanmean(self.e_track_series)
        e_track_max = _nanmax(self.e_track_series)
        e_dist_mean = _nanmean(self.e_dist_series)
        e_dist_max = _nanmax(self.e_dist_series)
        e_sim_mean = _nanmean(self.e_sim_series)
        e_sim_max = _nanmax(self.e_sim_series)

        formation_usage: dict[str, int] = {}
        for name in self.formation_name_series:
            formation_usage[name] = formation_usage.get(name, 0) + 1

        switch_count = sum(
            1 for a, b in zip(self.formation_name_series, self.formation_name_series[1:])
            if a != b
        )

        return {
            "scenario_name": self.scenario_name,
            "success": success,
            "failure_reason": failure_reason,
            "T_goal": self.T_goal,
            "num_steps": len(self.step_indices),

            "e_track_mean": e_track_mean,
            "e_track_max": e_track_max,
            "e_dist_mean": e_dist_mean,
            "e_dist_max": e_dist_max,
            "e_sim_mean": e_sim_mean,
            "e_sim_max": e_sim_max,

            "formation_switch_count": switch_count,
            "formation_usage": formation_usage,

            "obstacle_collision": self.obstacle_collision,
            "inter_robot_collision": self.inter_robot_collision,
            "timeout": timeout,
            "planning_failure": self.planning_failure,
            "control_failure": self.control_failure,
            "mpc_fallback_count": self.mpc_fallback_count,
        }

    def timeseries_dict(self) -> dict[str, list]:
        """Return a column-oriented dict suitable for CSV export."""
        return {
            "scenario_name": [self.scenario_name] * len(self.step_indices),
            "step": self.step_indices,
            "time": self.step_times,
            "formation_name": self.formation_name_series,
            "e_track": self.e_track_series,
            "e_dist": self.e_dist_series,
            "e_sim": self.e_sim_series,
            "reached_goal": self.reached_goal_flags,
            "obstacle_collision": self.obstacle_collision_flags,
            "inter_robot_collision": self.inter_robot_collision_flags,
        }

    # ── internal helpers ───────────────────────────────────────────

    def _check_obstacle_collision(self, states) -> bool:
        if self._map_data is None:
            return False
        # Avoid circular import – import lazily
        from formation.mpc_controller import query_distance_field
        for s in states:
            if query_distance_field(self._map_data, (s.x, s.y)) < self.robot_radius:
                return True
        return False

    def _check_inter_robot_collision(self, states) -> bool:
        min_d = math.inf
        for i in range(len(states)):
            si = states[i]
            for j in range(i + 1, len(states)):
                sj = states[j]
                d = math.hypot(si.x - sj.x, si.y - sj.y)
                if d < min_d:
                    min_d = d
        return min_d < 2.0 * self.robot_radius

    def _append_nan_errors(self) -> None:
        self.e_track_series.append(float("nan"))
        self.e_dist_series.append(float("nan"))
        self.e_sim_series.append(float("nan"))

    def _compute_success(self, timeout: bool) -> tuple[bool, str | None]:
        """Success criteria and failure_reason with priority chain."""
        if self.obstacle_collision:
            return False, "obstacle_collision"
        if self.inter_robot_collision:
            return False, "inter_robot_collision"
        if timeout:
            return False, "timeout"
        if self.planning_failure:
            return False, "planning_failure"
        if self.control_failure:
            return False, "control_failure"
        if not self.reached_goal:
            return False, "not_reached_goal"
        return True, None


def _nanmean(arr: list[float]) -> float | None:
    vals = [v for v in arr if not math.isnan(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def _nanmax(arr: list[float]) -> float | None:
    vals = [v for v in arr if not math.isnan(v)]
    if not vals:
        return None
    return float(np.max(vals))


def summarize_scenarios(results: list[dict]) -> dict:
    """Aggregate per-scenario metric dicts into a multi-scenario summary."""
    successes = [r for r in results if r["success"]]
    n_total = len(results)
    n_success = len(successes)

    failure_counts: dict[str, int] = {}
    for r in results:
        reason = r.get("failure_reason")
        if reason:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    T_goals = [r["T_goal"] for r in successes if r["T_goal"] is not None]
    e_sim_means = [r["e_sim_mean"] for r in successes if r["e_sim_mean"] is not None]
    e_track_means = [r["e_track_mean"] for r in successes if r["e_track_mean"] is not None]
    e_dist_means = [r["e_dist_mean"] for r in successes if r["e_dist_mean"] is not None]

    return {
        "num_scenarios": n_total,
        "num_success": n_success,
        "success_rate": n_success / n_total if n_total > 0 else 0.0,
        "mean_T_goal_success": float(np.mean(T_goals)) if T_goals else None,
        "mean_e_sim_success": float(np.mean(e_sim_means)) if e_sim_means else None,
        "mean_e_track_success": float(np.mean(e_track_means)) if e_track_means else None,
        "mean_e_dist_success": float(np.mean(e_dist_means)) if e_dist_means else None,
        "failure_counts": failure_counts,
    }
