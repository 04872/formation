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
        nc = len(neighbor_predictions)
        n_vx = np.zeros((nc, H), dtype=float)
        n_vy = np.zeros((nc, H), dtype=float)
        for n, pred in enumerate(neighbor_predictions):
            for k in range(H):
                # Only active when reference distance < 1.5*safe_dist
                idx = min(k, len(pred.positions_xy) - 1)
                dx = reference_matrix[0, k] - pred.positions_xy[idx][0]
                dy = reference_matrix[1, k] - pred.positions_xy[idx][1]
                if math.hypot(dx, dy) >= 1.5 * safe_dist:
                    continue
                idx_y = min(k, len(pred.yaw_rads) - 1) if pred.yaw_rads else 0
                yaw = pred.yaw_rads[idx_y] if pred.yaw_rads else 0.0
                spd = self._prediction_speed(pred, k)
                n_vx[n, k] = spd * math.cos(yaw)
                n_vy[n, k] = spd * math.sin(yaw)
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
            n_vx,
            n_vy,
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
        blend_steps = min(H, max(3, H // 2))
        orig = matrix[:, :H].copy()
        for k in range(blend_steps):
            alpha = (k + 1.0) / blend_steps
            matrix[0, k] = state.x + alpha * (orig[0, k] - state.x)
            matrix[1, k] = state.y + alpha * (orig[1, k] - state.y)
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

        dx = opti.variable(3, H + 1)
        du = opti.variable(2, H)

        dx0 = opti.parameter(3)
        reference = opti.parameter(5, H + 1)
        reference_tangents = opti.parameter(2, H + 1)
        neighbor_positions = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None
        neighbor_reference_positions = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None
        neighbor_prediction_speeds = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        neighbor_reference_speeds = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        neighbor_reference_tangents = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None

        objective = 0
        opti.subject_to(dx[0, 0] == dx0[0])
        opti.subject_to(dx[1, 0] == dx0[1])
        opti.subject_to(dx[2, 0] == dx0[2])

        for k in range(H):
            cos_ref = ca.cos(reference[2, k])
            sin_ref = ca.sin(reference[2, k])
            v_ref = reference[3, k]

            dx_next_x = dx[0, k] - dt * v_ref * sin_ref * dx[2, k] + dt * cos_ref * du[0, k]
            dx_next_y = dx[1, k] + dt * v_ref * cos_ref * dx[2, k] + dt * sin_ref * du[0, k]
            dx_next_theta = dx[2, k] + dt * du[1, k]
            opti.subject_to(dx[0, k + 1] == dx_next_x)
            opti.subject_to(dx[1, k + 1] == dx_next_y)
            opti.subject_to(dx[2, k + 1] == dx_next_theta)

            opti.subject_to(opti.bounded(0.0, reference[3, k] + du[0, k], self.config.v_max))
            opti.subject_to(opti.bounded(-self.config.omega_max, reference[4, k] + du[1, k], self.config.omega_max))

            objective += (
                w.position * (dx[0, k]**2 + dx[1, k]**2)
                + w.heading * dx[2, k]**2
                + w.input * ca.sumsqr(du[:, k])
            )
            if k > 0:
                objective += w.input_smooth * ca.sumsqr(du[:, k] - du[:, k - 1])

        # ── tracking-error consensus (keep formation shape under mismatch) ────
        for k in range(H):
            if (
                neighbor_positions is not None
                and neighbor_reference_positions is not None
                and neighbor_prediction_speeds is not None
                and neighbor_reference_speeds is not None
                and neighbor_reference_tangents is not None
                and neighbor_count > 0
                and w.relative_position > 0.0
            ):
                e_ix = dx[0, k]
                e_iy = dx[1, k]
                slot_progress_i = e_ix * reference_tangents[0, k] + e_iy * reference_tangents[1, k]
                speed_error_i = du[0, k]
                for n in range(neighbor_count):
                    nx = neighbor_positions[2 * n, k]
                    ny = neighbor_positions[2 * n + 1, k]
                    ref_dx = reference[0, k] - neighbor_reference_positions[2 * n, k]
                    ref_dy = reference[1, k] - neighbor_reference_positions[2 * n + 1, k]
                    err_x = (reference[0, k] + e_ix - nx) - ref_dx
                    err_y = (reference[1, k] + e_iy - ny) - ref_dy
                    objective += w.relative_position * (err_x**2 + err_y**2)

                    neigh_err_x = nx - neighbor_reference_positions[2 * n, k]
                    neigh_err_y = ny - neighbor_reference_positions[2 * n + 1, k]
                    neigh_progress = (
                        neigh_err_x * neighbor_reference_tangents[2 * n, k]
                        + neigh_err_y * neighbor_reference_tangents[2 * n + 1, k]
                    )
                    objective += w.progress_sync * (slot_progress_i - neigh_progress)**2

                    neigh_speed_error = neighbor_prediction_speeds[n, k] - neighbor_reference_speeds[n, k]
                    objective += w.velocity_consensus * (speed_error_i - neigh_speed_error)**2

        objective += w.terminal_position * (dx[0, H]**2 + dx[1, H]**2)

        # VO CBF: only active when ref distance < 1.5*safe_dist
        safe_dist = 2.0 * self.config.robot_radius + self.config.inter_robot_margin
        cbf_n_vx = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        cbf_n_vy = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        R_sq = safe_dist ** 2

        # Replace VO sqrt-based CBF with discrete linearized disk-center CBF per-step.
        # Linearize at reference point (dx=0, du=0) so constraints are affine in dx,du.
        if neighbor_positions is not None and neighbor_count > 0 and cbf_n_vx is not None:
            eps_cbf = opti.variable(neighbor_count, H)
            # hyperparameters: gamma for class-K term, eps_max upper bound for slack
            gamma = 1.0
            eps_max = 0.5
            for k in range(H):
                for n in range(neighbor_count):
                    opti.subject_to(eps_cbf[n, k] >= 0.0)
                    opti.subject_to(eps_cbf[n, k] <= eps_max)
                    # parameters at linearization point (reference)
                    nx = neighbor_positions[2 * n, k]
                    ny = neighbor_positions[2 * n + 1, k]
                    nvx = cbf_n_vx[n, k]
                    nvy = cbf_n_vy[n, k]
                    ref_x = reference[0, k]
                    ref_y = reference[1, k]
                    ref_th = reference[2, k]
                    ref_v = reference[3, k]

                    # r = self - neighbor (use consistent sign)
                    r0x = ref_x - nx
                    r0y = ref_y - ny

                    # self velocity at linearization point and its partials
                    vix0 = ref_v * ca.cos(ref_th)
                    viy0 = ref_v * ca.sin(ref_th)
                    a_vx_du = ca.cos(ref_th)
                    a_vx_th = -ref_v * ca.sin(ref_th)
                    a_vy_du = ca.sin(ref_th)
                    a_vy_th = ref_v * ca.cos(ref_th)

                    # base scalar value f0 (CBF evaluated at reference)
                    f0 = 2 * (r0x * (vix0 - nvx) + r0y * (viy0 - nvy)) + gamma * (r0x**2 + r0y**2 - R_sq)

                    # linear coefficients for decision variables
                    A_dx0 = 2 * (vix0 - nvx) + 2 * gamma * r0x
                    A_dx1 = 2 * (viy0 - nvy) + 2 * gamma * r0y
                    A_dx2 = 2 * r0x * a_vx_th + 2 * r0y * a_vy_th
                    A_du0 = 2 * r0x * a_vx_du + 2 * r0y * a_vy_du

                    # affine (linearized) CBF constraint: f0 + grad^T * delta + eps >= 0
                    opti.subject_to(
                        f0
                        + A_dx0 * dx[0, k]
                        + A_dx1 * dx[1, k]
                        + A_dx2 * dx[2, k]
                        + A_du0 * du[0, k]
                        + eps_cbf[n, k]
                        >= 0
                    )
            # penalize slack with configured neighbor_slack weight (squared)
            objective += w.neighbor_slack * ca.sumsqr(eps_cbf)
        eps_cbf_out = eps_cbf if (neighbor_count > 0 and cbf_n_vx is not None) else None

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
                "neighbor_positions": neighbor_positions,
                "neighbor_reference_positions": neighbor_reference_positions,
                "neighbor_prediction_speeds": neighbor_prediction_speeds,
                "neighbor_reference_speeds": neighbor_reference_speeds,
                "neighbor_reference_tangents": neighbor_reference_tangents,
                "cbf_n_vx": cbf_n_vx, "cbf_n_vy": cbf_n_vy,
            },
        )

    def _set_parameter_values(
        self, solver: _SolverCacheEntry, state: RobotState,
        reference_matrix: np.ndarray, reference_tangents: np.ndarray,
        neighbor_positions: np.ndarray, neighbor_reference_positions: np.ndarray,
        neighbor_prediction_speeds: np.ndarray,
        neighbor_reference_speeds: np.ndarray,
        neighbor_reference_tangents: np.ndarray,
        n_vx: np.ndarray | None = None, n_vy: np.ndarray | None = None,
    ) -> None:
        dx0 = np.array([
            state.x - reference_matrix[0, 0], state.y - reference_matrix[1, 0],
            wrap_to_pi(state.yaw - reference_matrix[2, 0]),
        ], dtype=float)
        solver.opti.set_value(solver.parameters["dx0"], dx0)
        solver.opti.set_value(solver.parameters["reference"], reference_matrix)
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
        if n_vx is not None and solver.parameters.get("cbf_n_vx") is not None:
            solver.opti.set_value(solver.parameters["cbf_n_vx"], n_vx)
            solver.opti.set_value(solver.parameters["cbf_n_vy"], n_vy)

    def _set_initial_guess(
        self,
        solver: _SolverCacheEntry,
        state: RobotState,
        reference_matrix: np.ndarray,
    ) -> None:
        H = self.config.horizon_steps
        solver.opti.set_initial(solver.variables["dx"], np.zeros((3, H + 1), dtype=float))
        solver.opti.set_initial(solver.variables["du"], np.zeros((2, H), dtype=float))
        if solver.variables.get("eps_n") is not None:
            solver.opti.set_initial(
                solver.variables["eps_n"],
                np.zeros((solver.variables["eps_n"].shape[0], H + 1), dtype=float),
            )

    def _vo_barrier(
        self,
        p_i: tuple[float, float],
        theta_i: float,
        v_i: float,
        p_j: tuple[float, float],
        theta_j: float,
        v_j: float,
        safe_dist: float,
    ) -> float:
        dx = p_j[0] - p_i[0]
        dy = p_j[1] - p_i[1]
        d2 = dx * dx + dy * dy
        d = math.sqrt(max(d2, 1e-12))
        ct_i = math.cos(theta_i)
        st_i = math.sin(theta_i)
        ct_j = math.cos(theta_j)
        st_j = math.sin(theta_j)
        v_i_xy = (v_i * ct_i, v_i * st_i)
        v_j_xy = (v_j * ct_j, v_j * st_j)
        v_rel = (v_j_xy[0] - v_i_xy[0], v_j_xy[1] - v_i_xy[1])
        v_rel_norm = math.sqrt(v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1])
        margin_sq = max(d2 - safe_dist * safe_dist, 0.0)
        cone_term = v_rel_norm * math.sqrt(margin_sq)
        return dx * v_rel[0] + dy * v_rel[1] + cone_term

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

    def _compute_vo_cbf_coeffs(
        self,
        ref_mat: np.ndarray,
        neighbor_predictions: list[RobotPrediction],
        safe_dist: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = ref_mat.shape[1] - 1
        N = len(neighbor_predictions)
        Ax = np.zeros((N, H), dtype=float)
        Ay = np.zeros((N, H), dtype=float)
        Av = np.zeros((N, H), dtype=float)
        Aw = np.zeros((N, H), dtype=float)
        Ath = np.zeros((N, H), dtype=float)
        Bh = np.zeros((N, H), dtype=float)
        if N == 0:
            return Ax, Ay, Av, Aw, Ath, Bh

        eps = 0.01
        max_c = 10.0
        for k in range(H):
            p_i = (float(ref_mat[0, k]), float(ref_mat[1, k]))
            theta_i = float(ref_mat[2, k])
            v_i = float(ref_mat[3, k])
            for n, prediction in enumerate(neighbor_predictions):
                idx = min(k, len(prediction.positions_xy) - 1)
                p_j = prediction.positions_xy[idx]
                theta_j = prediction.yaw_rads[idx] if prediction.yaw_rads else 0.0
                v_j = self._prediction_speed(prediction, k)
                dx = p_j[0] - p_i[0]
                dy = p_j[1] - p_i[1]
                d = math.hypot(dx, dy)
                if d >= safe_dist * 2.0:
                    continue

                base_h = self._vo_barrier(p_i, theta_i, v_i, p_j, theta_j, v_j, safe_dist)
                Bh[n, k] = base_h

                h_xp = self._vo_barrier((p_i[0] + eps, p_i[1]), theta_i, v_i, p_j, theta_j, v_j, safe_dist)
                h_xm = self._vo_barrier((p_i[0] - eps, p_i[1]), theta_i, v_i, p_j, theta_j, v_j, safe_dist)
                h_yp = self._vo_barrier((p_i[0], p_i[1] + eps), theta_i, v_i, p_j, theta_j, v_j, safe_dist)
                h_ym = self._vo_barrier((p_i[0], p_i[1] - eps), theta_i, v_i, p_j, theta_j, v_j, safe_dist)
                h_thp = self._vo_barrier(p_i, theta_i + eps, v_i, p_j, theta_j, v_j, safe_dist)
                h_thm = self._vo_barrier(p_i, theta_i - eps, v_i, p_j, theta_j, v_j, safe_dist)
                h_vp = self._vo_barrier(p_i, theta_i, v_i + eps, p_j, theta_j, v_j, safe_dist)
                h_vm = self._vo_barrier(p_i, theta_i, v_i - eps, p_j, theta_j, v_j, safe_dist)

                Ax[n, k] = max(-max_c, min(max_c, (h_xp - h_xm) / (2.0 * eps)))
                Ay[n, k] = max(-max_c, min(max_c, (h_yp - h_ym) / (2.0 * eps)))
                Ath[n, k] = max(-max_c, min(max_c, (h_thp - h_thm) / (2.0 * eps)))
                Av[n, k] = max(-max_c, min(max_c, (h_vp - h_vm) / (2.0 * eps)))
                Aw[n, k] = 0.0

        return Ax, Ay, Av, Aw, Ath, Bh

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
