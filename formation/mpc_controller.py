from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from formation.types import (
    ControlCommand,
    FormationControllerReference,
    MapData,
    MPCConfig,
    Point2D,
    RobotPrediction,
    RobotReferenceTrajectory,
    RobotState,
    wrap_to_pi,
)


@dataclass
class _SolverCacheEntry:
    opti: Any
    variables: dict[str, Any]
    parameters: dict[str, Any]


@dataclass(frozen=True)
class _RobotSolveInput:
    robot_index: int
    solver_slot: int
    state: RobotState
    own_reference: RobotReferenceTrajectory
    neighbor_predictions: list[RobotPrediction]
    neighbor_reference_positions: np.ndarray
    neighbor_reference_speeds: np.ndarray
    neighbor_indices: list[int]
    map_data: MapData


class DistributedFormationMPC:
    _stdout_lock = __import__("threading").Lock()

    def __init__(self, config: MPCConfig | None = None) -> None:
        self.config = config or MPCConfig()
        self._solver_cache: dict[tuple[int, int, int, int], _SolverCacheEntry] = {}
        self._cache_lock = __import__("threading").Lock()
        # runtime switch to enable/disable formation-consensus terms
        self._consensus_enabled = True
        # runtime switch to enable/disable CBF constraints and slack penalty
        self._cbf_enabled = True

    def set_consensus_enabled(self, enabled: bool) -> None:
        """Enable or disable the tracking‑error consensus terms (temporary runtime switch)."""
        self._consensus_enabled = bool(enabled)

    def set_cbf_enabled(self, enabled: bool) -> None:
        """Enable or disable the pairwise CBF constraints (temporary runtime switch)."""
        self._cbf_enabled = bool(enabled)

    def solve_all(
        self,
        states: list[RobotState],
        controller_reference: FormationControllerReference,
        map_data: MapData,
        previous_predictions: list[RobotPrediction] | None = None,
    ) -> tuple[list[ControlCommand], list[RobotPrediction]]:
        if len(states) != controller_reference.robot_count:
            raise ValueError(
                f"State count {len(states)} does not match reference robot count {controller_reference.robot_count}."
            )

        horizon = self.config.horizon_steps
        robot_count = len(states)
        solve_inputs = []
        for robot_index, state in enumerate(states):
            neighbor_indices = [j for j in range(robot_count) if j != robot_index]
            neighbor_ref_predictions = [
                self._reference_to_prediction(controller_reference.robot_trajectories[j].window(0, horizon))
                for j in neighbor_indices
            ]
            neighbor_reference_speeds = self._build_neighbor_speed_matrix(neighbor_ref_predictions)
            use_previous = (
                previous_predictions is not None
                and len(previous_predictions) == robot_count
                and self.config.neighbor_prediction_mode == "previous_prediction"
            )
            neighbor_predictions = (
                [previous_predictions[j] for j in neighbor_indices]
                if use_previous
                else neighbor_ref_predictions
            )
            neighbor_reference_positions = self._build_neighbor_position_matrix(neighbor_ref_predictions)
            solve_inputs.append(_RobotSolveInput(
                robot_index=robot_index,
                solver_slot=robot_index,
                state=state,
                own_reference=controller_reference.robot_trajectories[robot_index].window(0, horizon),
                neighbor_predictions=neighbor_predictions,
                neighbor_reference_positions=neighbor_reference_positions,
                neighbor_reference_speeds=neighbor_reference_speeds,
                neighbor_indices=neighbor_indices,
                map_data=map_data,
            ))

        use_parallel = (
            self.config.parallel_solve
            and len(solve_inputs) > 1
            and self._resolve_parallel_workers(len(solve_inputs)) > 1
        )
        if use_parallel:
            with ThreadPoolExecutor(max_workers=self._resolve_parallel_workers(len(solve_inputs))) as executor:
                results = list(executor.map(self._solve_robot_from_input, solve_inputs))
        else:
            results = [self._solve_robot_from_input(solve_input) for solve_input in solve_inputs]

        results.sort(key=lambda item: item[1].robot_index)
        return [command for command, _ in results], [prediction for _, prediction in results]

    def _solve_robot_from_input(self, solve_input: _RobotSolveInput) -> tuple[ControlCommand, RobotPrediction]:
        return self.solve_robot(
            solve_input.state,
            solve_input.own_reference,
            solve_input.neighbor_predictions,
            solve_input.map_data,
            solver_slot=solve_input.solver_slot,
            neighbor_indices=solve_input.neighbor_indices,
            neighbor_reference_positions=solve_input.neighbor_reference_positions,
            neighbor_reference_speeds=solve_input.neighbor_reference_speeds,
        )

    def solve_robot(
        self,
        state: RobotState,
        own_reference: RobotReferenceTrajectory,
        neighbor_predictions: list[RobotPrediction],
        map_data: MapData,
        *,
        solver_slot: int = 0,
        neighbor_indices: list[int] | None = None,
        neighbor_reference_positions: np.ndarray | None = None,
        neighbor_reference_speeds: np.ndarray | None = None,
    ) -> tuple[ControlCommand, RobotPrediction]:
        if own_reference.sample_count <= 0:
            raise ValueError("Robot reference trajectory is empty.")

        solver = self._get_solver(map_data, len(neighbor_predictions), solver_slot)
        padded_ref = own_reference.window(0, self.config.horizon_steps)
        reference_matrix = self._build_reference_matrix(state, padded_ref)
        self._unwrap_reference_heading_in_place(reference_matrix, state.yaw)
        reference_tangents = self._build_tangent_matrix(reference_matrix[:2, :])
        neighbor_positions = self._build_neighbor_position_matrix(neighbor_predictions)
        if neighbor_reference_positions is None:
            neighbor_reference_positions = neighbor_positions
        if neighbor_reference_speeds is None:
            neighbor_reference_speeds = self._build_neighbor_speed_matrix(neighbor_predictions)
        neighbor_reference_tangents = self._build_tangent_matrix(neighbor_reference_positions)
        neighbor_prediction_speeds = self._build_neighbor_speed_matrix(neighbor_predictions)

        H = self.config.horizon_steps
        safe_dist = 2.0 * self.config.robot_radius + self.config.inter_robot_margin
        cbf_rbar_next = self._build_cbf_linearization_vectors(
            reference_matrix,
            neighbor_positions,
            safe_dist,
        )
        self._set_parameter_values(
            solver,
            state,
            reference_matrix,
            reference_tangents,
            neighbor_positions,
            neighbor_reference_positions,
            neighbor_prediction_speeds,
            neighbor_reference_speeds,
            neighbor_reference_tangents,
            cbf_rbar_next,
        )
        self._set_initial_guess(solver, state, reference_matrix)
        with DistributedFormationMPC._stdout_lock:
            old_fd = os.dup(1)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.close(devnull)
            try:
                solution = solver.opti.solve()
            except RuntimeError:
                solution = None
            finally:
                os.dup2(old_fd, 1)
                os.close(old_fd)

        if solution is not None:
            dx_value = np.asarray(solution.value(solver.variables["dx"]), dtype=float)
            du_value = np.asarray(solution.value(solver.variables["du"]), dtype=float)
            eps_value = np.zeros((len(neighbor_predictions), self.config.horizon_steps + 1), dtype=float)
            if solver.variables.get("eps_cbf") is not None:
                eps_value = np.asarray(solution.value(solver.variables["eps_cbf"]), dtype=float)
                if eps_value.ndim == 1:
                    eps_value = eps_value.reshape(len(neighbor_predictions), self.config.horizon_steps)
                if eps_value.size > 0:
                    print(f"    [cbf slack] max={float(np.max(eps_value)):.4f} mean={float(np.mean(eps_value)):.4f}", flush=True)
            absolutes = np.zeros_like(dx_value)
            absolutes[:, 0] = np.array([state.x, state.y, state.yaw], dtype=float)
            for k in range(1, dx_value.shape[1]):
                ref_v = reference_matrix[3, k - 1]
                ref_w = reference_matrix[4, k - 1]
                dt = self.config.dt
                v_k = max(0.0, ref_v + du_value[0, k - 1])
                w_k = ref_w + du_value[1, k - 1]
                absolutes[0, k] = absolutes[0, k - 1] + dt * v_k * math.cos(absolutes[2, k - 1])
                absolutes[1, k] = absolutes[1, k - 1] + dt * v_k * math.sin(absolutes[2, k - 1])
                absolutes[2, k] = wrap_to_pi(absolutes[2, k - 1] + dt * w_k)
            positions_xy = [(float(absolutes[0, k]), float(absolutes[1, k])) for k in range(absolutes.shape[1])]
            yaw_rads = [float(wrap_to_pi(absolutes[2, k])) for k in range(absolutes.shape[1])]
            commands = [
                ControlCommand(
                    v=float(max(0.0, min(self.config.v_max, reference_matrix[3, k] + du_value[0, k]))),
                    omega=float(max(-self.config.omega_max, min(self.config.omega_max, reference_matrix[4, k] + du_value[1, k]))),
                )
                for k in range(du_value.shape[1])
            ]
            neighbor_slacks = [
                [float(eps_value[n, k]) for n in range(eps_value.shape[0])]
                for k in range(eps_value.shape[1])
            ]
            indices = neighbor_indices if neighbor_indices is not None else list(range(len(neighbor_predictions)))
            prediction = RobotPrediction(
                robot_index=solver_slot,
                positions_xy=positions_xy,
                yaw_rads=yaw_rads,
                commands=list(commands),
                obstacle_slacks=[0.0 for _ in range(H + 1)],
                neighbor_slacks=neighbor_slacks,
                metadata={
                    "neighbor_robot_indices": indices,
                    "objective": float(solution.value(solver.opti.f)),
                    "solver_status": solver.opti.stats().get("return_status", "unknown"),
                },
            )
            return commands[0], prediction

        # ── solver-failure fallback: open-loop track the reference ──
        H = self.config.horizon_steps
        dt = self.config.dt
        absolutes = np.zeros((3, H + 1), dtype=float)
        absolutes[:, 0] = np.array([state.x, state.y, state.yaw], dtype=float)
        commands = []
        for k in range(H):
            ref_v = reference_matrix[3, k]
            ref_w = reference_matrix[4, k]
            v_k = float(max(0.0, min(self.config.v_max, ref_v)))
            w_k = float(max(-self.config.omega_max, min(self.config.omega_max, ref_w)))
            commands.append(ControlCommand(v=v_k, omega=w_k))
            absolutes[0, k + 1] = absolutes[0, k] + dt * v_k * math.cos(absolutes[2, k])
            absolutes[1, k + 1] = absolutes[1, k] + dt * v_k * math.sin(absolutes[2, k])
            absolutes[2, k + 1] = wrap_to_pi(absolutes[2, k] + dt * w_k)

        positions_xy = [(float(absolutes[0, k]), float(absolutes[1, k])) for k in range(H + 1)]
        yaw_rads = [float(wrap_to_pi(absolutes[2, k])) for k in range(H + 1)]
        neighbor_slacks = [[0.0 for _ in range(len(neighbor_predictions))] for _ in range(H + 1)]
        indices = neighbor_indices if neighbor_indices is not None else list(range(len(neighbor_predictions)))
        prediction = RobotPrediction(
            robot_index=solver_slot,
            positions_xy=positions_xy,
            yaw_rads=yaw_rads,
            commands=list(commands),
            obstacle_slacks=[0.0 for _ in range(H + 1)],
            neighbor_slacks=neighbor_slacks,
            metadata={
                "neighbor_robot_indices": indices,
                "objective": float("nan"),
                "solver_status": "fallback_open_loop",
            },
        )
        return commands[0], prediction

    def propagate_state(self, state: RobotState, command: ControlCommand) -> RobotState:
        dt = self.config.dt
        next_x = state.x + dt * command.v * math.cos(state.yaw)
        next_y = state.y + dt * command.v * math.sin(state.yaw)
        next_yaw = wrap_to_pi(state.yaw + dt * command.omega)
        return RobotState(x=next_x, y=next_y, yaw=next_yaw, v=command.v, omega=command.omega)

    def _resolve_parallel_workers(self, robot_count: int) -> int:
        if self.config.parallel_workers is not None:
            return max(1, min(self.config.parallel_workers, robot_count))
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, robot_count))

    @staticmethod
    def _unwrap_reference_heading_in_place(reference_matrix: np.ndarray, current_yaw: float) -> None:
        """Put reference yaw on the same continuous branch as the current yaw.

        The MPC optimizes heading error in a linear error coordinate. Keeping the
        reference yaw unwrapped avoids artificial jumps at +/-pi.
        """
        if reference_matrix.shape[1] == 0:
            return
        reference_matrix[2, 0] = current_yaw + wrap_to_pi(reference_matrix[2, 0] - current_yaw)
        for k in range(1, reference_matrix.shape[1]):
            reference_matrix[2, k] = reference_matrix[2, k - 1] + wrap_to_pi(
                reference_matrix[2, k] - reference_matrix[2, k - 1]
            )

    def _build_cbf_linearization_vectors(
        self,
        reference_matrix: np.ndarray,
        neighbor_positions: np.ndarray,
        safe_dist: float,
    ) -> np.ndarray:
        """Build next-step distance-CBF linearization vectors.

        Variables are error coordinates, so the absolute predicted position is
        p_i,k = p_ref_i,k + dx_i,k. For the convexified distance constraint, we
        linearize ||p_i,k+1 - p_j,k+1||^2 at p_ref_i,k+1.
        """
        H = self.config.horizon_steps
        neighbor_count = neighbor_positions.shape[0] // 2
        rbar = np.zeros((2 * neighbor_count, H), dtype=float)
        if neighbor_count == 0:
            return rbar

        # Avoid a zero linearization normal when a reference slot coincides with
        # a neighbor prediction. The exact direction is not important in this
        # degenerate case; it only provides a stable separating normal for the
        # slackened affine constraint.
        min_norm = max(0.05, 0.5 * safe_dist)
        for n in range(neighbor_count):
            for k in range(H):
                ref_next = reference_matrix[:2, k + 1]
                neigh_next = neighbor_positions[2 * n : 2 * n + 2, k + 1]
                vec = ref_next - neigh_next
                norm = float(np.linalg.norm(vec))
                if norm < min_norm:
                    # Prefer the current-step relative direction if available.
                    ref_now = reference_matrix[:2, k]
                    neigh_now = neighbor_positions[2 * n : 2 * n + 2, k]
                    fallback = ref_now - neigh_now
                    fallback_norm = float(np.linalg.norm(fallback))
                    if fallback_norm < 1e-9:
                        fallback = np.array([1.0, 0.0], dtype=float)
                        fallback_norm = 1.0
                    vec = fallback / fallback_norm * min_norm
                rbar[2 * n : 2 * n + 2, k] = vec
        return rbar

    def _build_reference_matrix(self, state: RobotState, own_reference: RobotReferenceTrajectory) -> np.ndarray:
        matrix = np.asarray(
            [
                [sample.position_xy[0] for sample in own_reference.samples],
                [sample.position_xy[1] for sample in own_reference.samples],
                [sample.yaw for sample in own_reference.samples],
                [sample.v_ref for sample in own_reference.samples],
                [sample.omega_ref for sample in own_reference.samples],
            ],
            dtype=float,
        )
        H = matrix.shape[1]
        if H <= 1:
            return matrix
        # Do not blend positions over a long window — only apply a very
        # short yaw blend to avoid abrupt heading jumps. Position blending
        # plus replacing p_ref0 causes double-editing; prefer leaving
        # positions intact so MPC sees the true initial error via dx0.
        orig = matrix[:, :H].copy()
        yaw_blend_steps = min(3, H)
        for k in range(yaw_blend_steps):
            alpha = (k + 1.0) / yaw_blend_steps
            th_target = state.yaw + wrap_to_pi(orig[2, k] - state.yaw)
            matrix[2, k] = state.yaw + alpha * wrap_to_pi(th_target - state.yaw)
        for k in range(H - 1):
            disp = math.hypot(matrix[0, k + 1] - matrix[0, k], matrix[1, k + 1] - matrix[1, k])
            matrix[3, k] = disp / max(self.config.dt, 1e-9)
            dh = wrap_to_pi(matrix[2, k + 1] - matrix[2, k])
            matrix[4, k] = dh / max(self.config.dt, 1e-9)
        matrix[3, :] = np.clip(matrix[3, :], 0.0, self.config.v_max)
        matrix[4, :] = np.clip(matrix[4, :], -self.config.omega_max, self.config.omega_max)
        return matrix

    def _build_tangent_matrix(self, positions_matrix: np.ndarray) -> np.ndarray:
        if positions_matrix.size == 0:
            return np.zeros_like(positions_matrix)
        if positions_matrix.shape[0] % 2 != 0:
            raise ValueError("positions_matrix must have an even number of rows.")
        row_count, sample_count = positions_matrix.shape
        tangents = np.zeros_like(positions_matrix)
        for row_start in range(0, row_count, 2):
            xs = positions_matrix[row_start]
            ys = positions_matrix[row_start + 1]
            for k in range(sample_count):
                if sample_count == 1:
                    dx = 1.0
                    dy = 0.0
                elif k == 0:
                    dx = xs[1] - xs[0]
                    dy = ys[1] - ys[0]
                elif k == sample_count - 1:
                    dx = xs[-1] - xs[-2]
                    dy = ys[-1] - ys[-2]
                else:
                    dx = xs[k + 1] - xs[k - 1]
                    dy = ys[k + 1] - ys[k - 1]
                norm = math.hypot(dx, dy)
                if norm < 1e-9:
                    tangents[row_start, k] = 1.0
                    tangents[row_start + 1, k] = 0.0
                else:
                    tangents[row_start, k] = dx / norm
                    tangents[row_start + 1, k] = dy / norm
        return tangents

    def _build_neighbor_speed_matrix(self, neighbor_predictions: list[RobotPrediction]) -> np.ndarray:
        if not neighbor_predictions:
            return np.zeros((0, self.config.horizon_steps), dtype=float)
        H = self.config.horizon_steps
        speeds = np.zeros((len(neighbor_predictions), H), dtype=float)
        for pred_idx, prediction in enumerate(neighbor_predictions):
            for k in range(H):
                speeds[pred_idx, k] = self._prediction_speed(prediction, k)
        return speeds

    def _get_solver(self, map_data: MapData, neighbor_count: int, solver_slot: int) -> _SolverCacheEntry:
        cache_key = (id(map_data), self.config.horizon_steps, neighbor_count, solver_slot)
        with self._cache_lock:
            if cache_key not in self._solver_cache:
                self._solver_cache[cache_key] = self._build_solver(neighbor_count)
            return self._solver_cache[cache_key]

    def _build_solver(self, neighbor_count: int) -> _SolverCacheEntry:
        ca = _require_casadi()
        opti = ca.Opti()
        H = self.config.horizon_steps
        dt = self.config.dt
        w = self.config.weights

        # Error-state MPC variables:
        #   dx[:, k] = x[:, k] - x_ref[:, k]
        #   du[:, k] = u[:, k] - u_ref[:, k]
        dx = opti.variable(3, H + 1)
        du = opti.variable(2, H)

        dx0 = opti.parameter(3)
        reference = opti.parameter(5, H + 1)
        reference_tangents = opti.parameter(2, H + 1)
        previous_command = opti.parameter(2)

        neighbor_positions = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None
        neighbor_reference_positions = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None
        neighbor_prediction_speeds = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        neighbor_reference_speeds = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        neighbor_reference_tangents = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None
        cbf_rbar_next = opti.parameter(2 * neighbor_count, H) if neighbor_count > 0 else None

        objective = 0
        opti.subject_to(dx[:, 0] == dx0)

        for k in range(H):
            # Linearized unicycle tracking-error dynamics around the reference.
            theta_ref = reference[2, k]
            v_ref = reference[3, k]
            cos_ref = ca.cos(theta_ref)
            sin_ref = ca.sin(theta_ref)

            opti.subject_to(
                dx[0, k + 1]
                == dx[0, k]
                - dt * v_ref * sin_ref * dx[2, k]
                + dt * cos_ref * du[0, k]
            )
            opti.subject_to(
                dx[1, k + 1]
                == dx[1, k]
                + dt * v_ref * cos_ref * dx[2, k]
                + dt * sin_ref * du[0, k]
            )
            opti.subject_to(dx[2, k + 1] == dx[2, k] + dt * du[1, k])

            # Hard input bounds on absolute commands u = u_ref + du.
            opti.subject_to(opti.bounded(0.0, reference[3, k] + du[0, k], self.config.v_max))
            opti.subject_to(opti.bounded(-self.config.omega_max, reference[4, k] + du[1, k], self.config.omega_max))

            objective += (
                w.position * (dx[0, k] ** 2 + dx[1, k] ** 2)
                + w.heading * dx[2, k] ** 2
                + w.input * ca.sumsqr(du[:, k])
            )
            if k > 0:
                objective += w.input_smooth * ca.sumsqr(du[:, k] - du[:, k - 1])

        # Penalize the first absolute command relative to the currently executed command.
        objective += w.initial_input_smooth * (
            (reference[3, 0] + du[0, 0] - previous_command[0]) ** 2
            + (reference[4, 0] + du[1, 0] - previous_command[1]) ** 2
        )

        # Terminal slot-tracking error.
        objective += w.terminal_position * (dx[0, H] ** 2 + dx[1, H] ** 2)

        # ── Clean error-coordinate consensus terms ───────────────────────
        # Absolute position alias: p_i,k = p_ref_i,k + dx_i,k.
        # Neighbor prediction error alias: ehat_j,k = p_hat_j,k - p_ref_j,k.
        # Then relative-position consensus is simply ||e_i,k - ehat_j,k||^2.
        if self._consensus_enabled and neighbor_count > 0:
            for k in range(H):
                if (
                    neighbor_positions is None
                    or neighbor_reference_positions is None
                    or neighbor_prediction_speeds is None
                    or neighbor_reference_speeds is None
                    or neighbor_reference_tangents is None
                ):
                    continue

                slot_progress_i = dx[0, k] * reference_tangents[0, k] + dx[1, k] * reference_tangents[1, k]
                speed_error_i = du[0, k]

                for n in range(neighbor_count):
                    neigh_err_x = neighbor_positions[2 * n, k] - neighbor_reference_positions[2 * n, k]
                    neigh_err_y = neighbor_positions[2 * n + 1, k] - neighbor_reference_positions[2 * n + 1, k]

                    if w.relative_position > 0.0:
                        objective += w.relative_position * (
                            (dx[0, k] - neigh_err_x) ** 2
                            + (dx[1, k] - neigh_err_y) ** 2
                        )

                    if w.progress_sync > 0.0:
                        neigh_progress = (
                            neigh_err_x * neighbor_reference_tangents[2 * n, k]
                            + neigh_err_y * neighbor_reference_tangents[2 * n + 1, k]
                        )
                        objective += w.progress_sync * (slot_progress_i - neigh_progress) ** 2

                    if w.velocity_consensus > 0.0:
                        neigh_speed_error = neighbor_prediction_speeds[n, k] - neighbor_reference_speeds[n, k]
                        objective += w.velocity_consensus * (speed_error_i - neigh_speed_error) ** 2

        # ── Distance-CBF / separation in error coordinates ───────────────
        # h(p) = ||p_i - p_j||^2 - d_safe^2.
        # With p_i = p_ref_i + dx_i, linearize ||p_i - p_j||^2 at
        # p_bar_i = p_ref_i and supplied rbar = p_bar_i - p_j.
        eps_cbf = None
        safe_dist = 2.0 * self.config.robot_radius + self.config.inter_robot_margin
        d_safe_sq = safe_dist ** 2
        if self._cbf_enabled and neighbor_count > 0 and neighbor_positions is not None and cbf_rbar_next is not None:
            eps_cbf = opti.variable(neighbor_count, H)
            for k in range(H):
                for n in range(neighbor_count):
                    opti.subject_to(eps_cbf[n, k] >= 0.0)

                    # Next-step absolute self position alias.
                    p_abs_x = reference[0, k + 1] + dx[0, k + 1]
                    p_abs_y = reference[1, k + 1] + dx[1, k + 1]
                    neigh_x = neighbor_positions[2 * n, k + 1]
                    neigh_y = neighbor_positions[2 * n + 1, k + 1]
                    rbar_x = cbf_rbar_next[2 * n, k]
                    rbar_y = cbf_rbar_next[2 * n + 1, k]

                    # Affine under-estimator of ||p_abs - p_neighbor||^2.
                    h_dist_lin = (
                        2.0 * rbar_x * (p_abs_x - neigh_x)
                        + 2.0 * rbar_y * (p_abs_y - neigh_y)
                        - (rbar_x ** 2 + rbar_y ** 2)
                    )
                    opti.subject_to(h_dist_lin - d_safe_sq + eps_cbf[n, k] >= 0.0)
            objective += w.neighbor_slack * ca.sumsqr(eps_cbf)
        eps_cbf_out = eps_cbf if eps_cbf is not None else None

        opti.minimize(objective)
        opti.solver(
            "qrsqp",
            {
                "print_time": False,
                "print_header": False,
                "qpsol": "osqp",
                "qpsol_options": {"verbose": False},
                "hessian_approximation": "exact",
                "max_iter": 30,
            },
        )
        return _SolverCacheEntry(
            opti=opti,
            variables={"dx": dx, "du": du, "eps_cbf": eps_cbf_out},
            parameters={
                "dx0": dx0,
                "reference": reference,
                "reference_tangents": reference_tangents,
                "previous_command": previous_command,
                "neighbor_positions": neighbor_positions,
                "neighbor_reference_positions": neighbor_reference_positions,
                "neighbor_prediction_speeds": neighbor_prediction_speeds,
                "neighbor_reference_speeds": neighbor_reference_speeds,
                "neighbor_reference_tangents": neighbor_reference_tangents,
                "cbf_rbar_next": cbf_rbar_next,
            },
        )

    def _set_parameter_values(
        self, solver: _SolverCacheEntry, state: RobotState,
        reference_matrix: np.ndarray, reference_tangents: np.ndarray,
        neighbor_positions: np.ndarray, neighbor_reference_positions: np.ndarray,
        neighbor_prediction_speeds: np.ndarray,
        neighbor_reference_speeds: np.ndarray,
        neighbor_reference_tangents: np.ndarray,
        cbf_rbar_next: np.ndarray,
    ) -> None:
        dx0 = np.array([
            state.x - reference_matrix[0, 0],
            state.y - reference_matrix[1, 0],
            wrap_to_pi(state.yaw - reference_matrix[2, 0]),
        ], dtype=float)
        solver.opti.set_value(solver.parameters["dx0"], dx0)
        solver.opti.set_value(solver.parameters["reference"], reference_matrix)
        if solver.parameters.get("previous_command") is not None:
            solver.opti.set_value(solver.parameters["previous_command"], np.array([state.v, state.omega], dtype=float))
        if solver.parameters.get("reference_tangents") is not None:
            solver.opti.set_value(solver.parameters["reference_tangents"], reference_tangents)
        if solver.parameters.get("neighbor_positions") is not None:
            solver.opti.set_value(solver.parameters["neighbor_positions"], neighbor_positions)
        if solver.parameters.get("neighbor_reference_positions") is not None:
            solver.opti.set_value(solver.parameters["neighbor_reference_positions"], neighbor_reference_positions)
        if solver.parameters.get("neighbor_prediction_speeds") is not None:
            solver.opti.set_value(solver.parameters["neighbor_prediction_speeds"], neighbor_prediction_speeds)
        if solver.parameters.get("neighbor_reference_speeds") is not None:
            solver.opti.set_value(solver.parameters["neighbor_reference_speeds"], neighbor_reference_speeds)
        if solver.parameters.get("neighbor_reference_tangents") is not None:
            solver.opti.set_value(solver.parameters["neighbor_reference_tangents"], neighbor_reference_tangents)
        if solver.parameters.get("cbf_rbar_next") is not None:
            solver.opti.set_value(solver.parameters["cbf_rbar_next"], cbf_rbar_next)

    def _set_initial_guess(
        self,
        solver: _SolverCacheEntry,
        state: RobotState,
        reference_matrix: np.ndarray,
    ) -> None:
        H = self.config.horizon_steps
        dt = self.config.dt
        dx_init = np.zeros((3, H + 1), dtype=float)
        dx_init[:, 0] = np.array([
            state.x - reference_matrix[0, 0],
            state.y - reference_matrix[1, 0],
            wrap_to_pi(state.yaw - reference_matrix[2, 0]),
        ], dtype=float)
        du_init = np.zeros((2, H), dtype=float)
        for k in range(H):
            th = reference_matrix[2, k]
            v_ref = reference_matrix[3, k]
            dx_init[0, k + 1] = (
                dx_init[0, k]
                - dt * v_ref * math.sin(th) * dx_init[2, k]
                + dt * math.cos(th) * du_init[0, k]
            )
            dx_init[1, k + 1] = (
                dx_init[1, k]
                + dt * v_ref * math.cos(th) * dx_init[2, k]
                + dt * math.sin(th) * du_init[0, k]
            )
            dx_init[2, k + 1] = dx_init[2, k] + dt * du_init[1, k]
        solver.opti.set_initial(solver.variables["dx"], dx_init)
        solver.opti.set_initial(solver.variables["du"], du_init)
        eps_cbf = solver.variables.get("eps_cbf")
        if eps_cbf is not None:
            solver.opti.set_initial(
                eps_cbf,
                np.zeros((eps_cbf.shape[0], eps_cbf.shape[1]), dtype=float),
            )
        try:
            solver.opti.set_initial(solver.opti.lam_g, 0)
        except Exception:
            pass

    def _prediction_speed(self, prediction: RobotPrediction, k: int) -> float:
        if prediction.commands:
            idx = min(k, len(prediction.commands) - 1)
            return float(prediction.commands[idx].v)
        if len(prediction.positions_xy) >= 2:
            idx = min(k + 1, len(prediction.positions_xy) - 1)
            prev = prediction.positions_xy[max(0, idx - 1)]
            curr = prediction.positions_xy[idx]
            return float(
                math.hypot(curr[0] - prev[0], curr[1] - prev[1]) / max(self.config.dt, 1e-9)
            )
        return 0.0

    def _build_neighbor_position_matrix(self, neighbor_predictions: list[RobotPrediction]) -> np.ndarray:
        if not neighbor_predictions:
            return np.zeros((0, self.config.horizon_steps + 1), dtype=float)
        H = self.config.horizon_steps
        rows = np.zeros((2 * len(neighbor_predictions), H + 1), dtype=float)
        for pred_idx, prediction in enumerate(neighbor_predictions):
            positions = list(prediction.positions_xy)
            while len(positions) < H + 1:
                positions.append(positions[-1])
            for k in range(H + 1):
                rows[2 * pred_idx, k] = positions[k][0]
                rows[2 * pred_idx + 1, k] = positions[k][1]
        return rows

    def _reference_to_prediction(self, trajectory: RobotReferenceTrajectory) -> RobotPrediction:
        positions_xy = [sample.position_xy for sample in trajectory.samples]
        yaw_rads = [sample.yaw for sample in trajectory.samples]
        commands = [
            ControlCommand(v=sample.v_ref, omega=sample.omega_ref)
            for sample in trajectory.samples[: self.config.horizon_steps]
        ]
        return RobotPrediction(
            robot_index=trajectory.robot_index,
            positions_xy=positions_xy,
            yaw_rads=yaw_rads,
            commands=commands,
            obstacle_slacks=[0.0 for _ in positions_xy],
            neighbor_slacks=[[] for _ in positions_xy],
            metadata={"source": "reference"},
        )


def query_distance_field(map_data: MapData, point_xy: Point2D) -> float:
    x, y = point_xy
    origin_x, origin_y = map_data.origin_xy
    grid_x = (x - origin_x) / map_data.resolution - 0.5
    grid_y = (y - origin_y) / map_data.resolution - 0.5
    if grid_x < 0.0 or grid_y < 0.0 or grid_x > map_data.cols - 1 or grid_y > map_data.rows - 1:
        return 0.0

    x0 = int(math.floor(grid_x))
    y0 = int(math.floor(grid_y))
    x1 = min(x0 + 1, map_data.cols - 1)
    y1 = min(y0 + 1, map_data.rows - 1)
    wx = grid_x - x0
    wy = grid_y - y0

    value00 = float(map_data.distance_field[y0, x0])
    value10 = float(map_data.distance_field[y0, x1])
    value01 = float(map_data.distance_field[y1, x0])
    value11 = float(map_data.distance_field[y1, x1])
    return (
        (1.0 - wx) * (1.0 - wy) * value00
        + wx * (1.0 - wy) * value10
        + (1.0 - wx) * wy * value01
        + wx * wy * value11
    )


def _require_casadi():
    try:
        import casadi as ca
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("casadi is required to use DistributedFormationMPC.") from exc
    return ca
