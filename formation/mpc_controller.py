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
        H = self.config.horizon_steps; dt = self.config.dt
        # ── heading reference: unwrap aligned to current yaw ─────
        ref_yaws = reference_matrix[2, :].copy()
        uw = np.zeros(H + 1, dtype=float)
        uw[0] = state.yaw + wrap_to_pi(ref_yaws[0] - state.yaw)
        for k in range(1, H + 1):
            uw[k] = uw[k - 1] + wrap_to_pi(ref_yaws[k] - ref_yaws[k - 1])
        # ── nominal rollout from current state ───────────────────
        nominal_state = np.zeros((3, H + 1), dtype=float)
        nominal_state[:, 0] = np.array([state.x, state.y, state.yaw])
        nominal_input = reference_matrix[3:5, :H].copy()
        nominal_input[0] = np.clip(nominal_input[0], 0.0, self.config.v_max)
        nominal_input[1] = np.clip(nominal_input[1], -self.config.omega_max, self.config.omega_max)
        for k in range(H):
            th = nominal_state[2, k]; v = nominal_input[0, k]; om = nominal_input[1, k]
            nominal_state[0, k + 1] = nominal_state[0, k] + dt * v * math.cos(th)
            nominal_state[1, k + 1] = nominal_state[1, k] + dt * v * math.sin(th)
            nominal_state[2, k + 1] = wrap_to_pi(nominal_state[2, k] + dt * om)
        # unwrap nominal heading
        for k in range(1, H + 1):
            d = nominal_state[2, k] - nominal_state[2, k - 1]
            nominal_state[2, k] = nominal_state[2, k - 1] + wrap_to_pi(d)
        # ── Jacobians (column‑major flatten for CasADi) ───────────
        A_mat = np.zeros((9, H), dtype=float)
        B_mat = np.zeros((6, H), dtype=float)
        c_vec = np.zeros((3, H), dtype=float)
        for k in range(H):
            th = nominal_state[2, k]; v = nominal_input[0, k]; om = nominal_input[1, k]
            A_k = np.array([[1.,0.,-dt*v*math.sin(th)], [0.,1.,dt*v*math.cos(th)], [0.,0.,1.]])
            B_k = np.array([[dt*math.cos(th),0.],[dt*math.sin(th),0.],[0.,dt]])
            f_bar = np.array([nominal_state[0,k]+dt*v*math.cos(th),
                              nominal_state[1,k]+dt*v*math.sin(th),
                              nominal_state[2,k]+dt*om])
            c_k = f_bar - A_k @ nominal_state[:, k] - B_k @ nominal_input[:, k]
            A_mat[:, k] = A_k.reshape(-1, order="F")
            B_mat[:, k] = B_k.reshape(-1, order="F")
            c_vec[:, k] = c_k
        ref_tangents = self._build_tangent_matrix(reference_matrix[:2, :])
        # ── neighbors ────────────────────────────────────────────
        neigh_pos = self._build_neighbor_position_matrix(neighbor_predictions)
        if neighbor_reference_positions is None: neighbor_reference_positions = neigh_pos
        if neighbor_reference_speeds is None: neighbor_reference_speeds = self._build_neighbor_speed_matrix(neighbor_predictions)
        neigh_ref_tan = self._build_tangent_matrix(neighbor_reference_positions)
        neigh_pred_spd = self._build_neighbor_speed_matrix(neighbor_predictions)
        # ── CBF: linearise at nominal ────────────────────────────
        safe_dist = 2.0 * self.config.robot_radius + self.config.inter_robot_margin
        ds_sq = safe_dist**2; nc = len(neighbor_predictions)
        cbf_hbar   = np.zeros((nc, H)); cbf_grad   = np.zeros((2*nc, H))
        cbf_hbar_n = np.zeros((nc, H)); cbf_grad_n = np.zeros((2*nc, H))
        for k in range(H):
            nb_nom = nominal_state[:2, k]; nb_nom1 = nominal_state[:2, k + 1]
            for n in range(nc):
                pj_k  = neigh_pos[2*n:2*n+2, k]
                pj_k1 = neigh_pos[2*n:2*n+2, k + 1]
                r_k  = nb_nom - pj_k;  r_k1 = nb_nom1 - pj_k1
                nr_k  = float(np.linalg.norm(r_k)); nr_k1 = float(np.linalg.norm(r_k1))
                cbf_hbar[n,k]   = float(np.sum(r_k**2))
                cbf_hbar_n[n,k] = float(np.sum(r_k1**2))
                # protect against near-zero gradient when robots are close
                fallback = np.array([1.0, 0.0])
                if nr_k < 0.05:
                    cbf_grad[2*n:2*n+2,k] = 2.0 * fallback
                else:
                    cbf_grad[2*n:2*n+2,k] = 2.0 * r_k
                if nr_k1 < 0.05:
                    cbf_grad_n[2*n:2*n+2,k] = 2.0 * fallback
                else:
                    cbf_grad_n[2*n:2*n+2,k] = 2.0 * r_k1
        self._set_parameter_values(
            solver, state, reference_matrix, uw, ref_tangents,
            nominal_state, nominal_input, A_mat, B_mat, c_vec,
            neigh_pos, neighbor_reference_positions,
            neigh_pred_spd, neighbor_reference_speeds, neigh_ref_tan,
            cbf_hbar, cbf_grad, cbf_hbar_n, cbf_grad_n)
        self._set_initial_guess(solver, nominal_state, nominal_input)
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
            x_value = np.asarray(solution.value(solver.variables["X"]), dtype=float)
            u_value = np.asarray(solution.value(solver.variables["U"]), dtype=float)
            eps_value = np.zeros((len(neighbor_predictions), self.config.horizon_steps + 1), dtype=float)
            if solver.variables.get("eps_cbf") is not None:
                eps_value = np.asarray(solution.value(solver.variables["eps_cbf"]), dtype=float)
                if eps_value.ndim == 1:
                    eps_value = eps_value.reshape(len(neighbor_predictions), self.config.horizon_steps)
                if eps_value.size > 0:
                    print(f"    [cbf slack] max={float(np.max(eps_value)):.4f} mean={float(np.mean(eps_value)):.4f}", flush=True)
            # nonlinear rollout from solved U
            dt_v = self.config.dt
            absolutes = np.zeros((3, x_value.shape[1]), dtype=float)
            absolutes[:, 0] = np.array([state.x, state.y, state.yaw], dtype=float)
            for k in range(u_value.shape[1]):
                vk = float(u_value[0, k]); wk = float(u_value[1, k])
                absolutes[0, k + 1] = absolutes[0, k] + dt_v * vk * math.cos(absolutes[2, k])
                absolutes[1, k + 1] = absolutes[1, k] + dt_v * vk * math.sin(absolutes[2, k])
                absolutes[2, k + 1] = wrap_to_pi(absolutes[2, k] + dt_v * wk)
            positions_xy = [(float(absolutes[0, k]), float(absolutes[1, k])) for k in range(absolutes.shape[1])]
            yaw_rads = [float(wrap_to_pi(absolutes[2, k])) for k in range(absolutes.shape[1])]
            commands = [
                ControlCommand(
                    v=float(max(0.0, min(self.config.v_max, u_value[0, k]))),
                    omega=float(max(-self.config.omega_max, min(self.config.omega_max, u_value[1, k]))),
                )
                for k in range(u_value.shape[1])
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
        H = self.config.horizon_steps; dt = self.config.dt; w = self.config.weights

        X = opti.variable(3, H + 1)
        U = opti.variable(2, H)

        state0       = opti.parameter(3)
        reference_pos = opti.parameter(2, H + 1)        # tracking target position
        ref_thetas_uw = opti.parameter(H + 1)            # unwrapped heading ref
        ref_input     = opti.parameter(2, H)             # v_ref, omega_ref
        ref_tangents  = opti.parameter(2, H + 1)         # tangent per step
        nominal_state = opti.parameter(3, H + 1)         # nominal rollout
        nominal_input = opti.parameter(2, H)             # nominal input sequence
        A_mat  = opti.parameter(9, H)                    # linearised A_k (col-major)
        B_mat  = opti.parameter(6, H)                    # linearised B_k (col-major)
        c_vec  = opti.parameter(3, H)                    # defect c_k
        prev_cmd = opti.parameter(2)
        neigh_pos  = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None
        neigh_ref  = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None
        neigh_pred_spd = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        neigh_ref_spd  = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        neigh_ref_tan  = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None
        cbf_hbar   = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        cbf_grad   = opti.parameter(2 * neighbor_count, H) if neighbor_count > 0 else None
        cbf_hbar_n = opti.parameter(neighbor_count, H) if neighbor_count > 0 else None
        cbf_grad_n = opti.parameter(2 * neighbor_count, H) if neighbor_count > 0 else None

        objective = 0
        opti.subject_to(X[:, 0] == state0)

        r_pos, r_theta, r_v, r_w = 0.50, 1.00, self.config.v_max, self.config.omega_max

        for k in range(H):
            Ak = ca.reshape(A_mat[:, k], 3, 3)
            Bk = ca.reshape(B_mat[:, k], 3, 2)
            opti.subject_to(X[:, k + 1] == ca.mtimes(Ak, X[:, k]) + ca.mtimes(Bk, U[:, k]) + c_vec[:, k])

            opti.subject_to(opti.bounded(0.0, U[0, k], self.config.v_max))
            opti.subject_to(opti.bounded(-self.config.omega_max, U[1, k], self.config.omega_max))
            # trust region around nominal (k≥1 for state)
            if k >= 1:
                opti.subject_to(opti.bounded(-r_pos, X[0, k] - nominal_state[0, k], r_pos))
                opti.subject_to(opti.bounded(-r_pos, X[1, k] - nominal_state[1, k], r_pos))
                opti.subject_to(opti.bounded(-r_theta, X[2, k] - nominal_state[2, k], r_theta))
            opti.subject_to(opti.bounded(-r_v, U[0, k] - nominal_input[0, k], r_v))
            opti.subject_to(opti.bounded(-r_w, U[1, k] - nominal_input[1, k], r_w))

            objective += (
                w.position  * ((X[0,k]-reference_pos[0,k])**2 + (X[1,k]-reference_pos[1,k])**2)
                + w.heading * (X[2,k] - ref_thetas_uw[k])**2
                + w.input   * ca.sumsqr(U[:, k] - ref_input[:, k])
            )
            if k > 0:
                objective += w.input_smooth * ca.sumsqr(U[:, k] - U[:, k - 1])

        objective += w.initial_input_smooth * ca.sumsqr(U[:, 0] - prev_cmd)
        # k=H trust region
        opti.subject_to(opti.bounded(-r_pos, X[0, H] - nominal_state[0, H], r_pos))
        opti.subject_to(opti.bounded(-r_pos, X[1, H] - nominal_state[1, H], r_pos))
        opti.subject_to(opti.bounded(-r_theta, X[2, H] - nominal_state[2, H], r_theta))
        objective += w.terminal_position * ((X[0,H]-reference_pos[0,H])**2 + (X[1,H]-reference_pos[1,H])**2)

        # ── consensus ─────────────────────────────────────────────
        if self._consensus_enabled and neigh_pos is not None and neighbor_count > 0:
            for k in range(H):
                e_ix = X[0,k] - reference_pos[0,k]
                e_iy = X[1,k] - reference_pos[1,k]
                slot_prog = e_ix*ref_tangents[0,k] + e_iy*ref_tangents[1,k]
                spd_err_i = U[0,k] - ref_input[0,k]
                for n in range(neighbor_count):
                    ref_dx = reference_pos[0,k] - neigh_ref[2*n,k]
                    ref_dy = reference_pos[1,k] - neigh_ref[2*n+1,k]
                    if w.relative_position > 0.0:
                        err_x = (X[0,k] - neigh_pos[2*n,k]) - ref_dx
                        err_y = (X[1,k] - neigh_pos[2*n+1,k]) - ref_dy
                        objective += w.relative_position * (err_x**2 + err_y**2)
                    if w.progress_sync > 0.0:
                        npx = neigh_pos[2*n,k] - neigh_ref[2*n,k]
                        npy = neigh_pos[2*n+1,k] - neigh_ref[2*n+1,k]
                        nprog = npx*neigh_ref_tan[2*n,k] + npy*neigh_ref_tan[2*n+1,k]
                        objective += w.progress_sync * (slot_prog - nprog)**2
                    if w.velocity_consensus > 0.0:
                        nse = neigh_pred_spd[n,k] - neigh_ref_spd[n,k]
                        objective += w.velocity_consensus * (spd_err_i - nse)**2

        # ── CBF: next-step distance constraint linearised at nominal ──
        eps_cbf = None
        safe_dist = 2.0 * self.config.robot_radius + self.config.inter_robot_margin
        d_safe_sq = safe_dist**2
        if self._cbf_enabled and neigh_pos is not None and neighbor_count > 0:
            eps_cbf = opti.variable(neighbor_count, H)
            for k in range(H):
                for n in range(neighbor_count):
                    opti.subject_to(eps_cbf[n, k] >= 0.0)
                    h_nxt = (cbf_hbar_n[n,k]
                             + ca.dot(cbf_grad_n[2*n:2*n+2,k], X[:2,k+1] - nominal_state[:2,k+1]))
                    opti.subject_to(h_nxt - d_safe_sq + eps_cbf[n,k] >= 0)
            objective += w.neighbor_slack * ca.sumsqr(eps_cbf)
        eps_cbf_out = eps_cbf if eps_cbf is not None else None

        opti.minimize(objective)
        opti.solver("qrsqp", {
            "print_time": False, "print_header": False,
            "qpsol": "osqp", "qpsol_options": {"verbose": False},
            "hessian_approximation": "exact", "max_iter": 30,
        })
        return _SolverCacheEntry(
            opti=opti, variables={"X": X, "U": U, "eps_cbf": eps_cbf_out},
            parameters={
                "state0": state0, "reference_pos": reference_pos,
                "ref_thetas_uw": ref_thetas_uw, "ref_input": ref_input, "ref_tangents": ref_tangents,
                "nominal_state": nominal_state, "nominal_input": nominal_input,
                "A_mat": A_mat, "B_mat": B_mat, "c_vec": c_vec,
                "prev_cmd": prev_cmd, "neigh_pos": neigh_pos, "neigh_ref": neigh_ref,
                "neigh_pred_spd": neigh_pred_spd, "neigh_ref_spd": neigh_ref_spd,
                "neigh_ref_tan": neigh_ref_tan,
                "cbf_hbar": cbf_hbar, "cbf_grad": cbf_grad,
                "cbf_hbar_n": cbf_hbar_n, "cbf_grad_n": cbf_grad_n,
            },
        )

    def _set_parameter_values(
        self, solver: _SolverCacheEntry, state: RobotState,
        reference_matrix: np.ndarray, ref_thetas_uw: np.ndarray, ref_tangents: np.ndarray,
        nominal_state: np.ndarray, nominal_input: np.ndarray,
        A_mat: np.ndarray, B_mat: np.ndarray, c_vec: np.ndarray,
        neigh_pos: np.ndarray, neigh_ref: np.ndarray,
        neigh_pred_spd: np.ndarray, neigh_ref_spd: np.ndarray, neigh_ref_tan: np.ndarray,
        cbf_hbar: np.ndarray, cbf_grad: np.ndarray,
        cbf_hbar_n: np.ndarray, cbf_grad_n: np.ndarray,
    ) -> None:
        p = solver.parameters
        solver.opti.set_value(p["state0"], np.array([state.x, state.y, state.yaw], dtype=float))
        solver.opti.set_value(p["reference_pos"], reference_matrix[:2, :])
        solver.opti.set_value(p["ref_thetas_uw"], ref_thetas_uw)
        H = nominal_input.shape[1]
        solver.opti.set_value(p["ref_input"], reference_matrix[3:5, :H])
        solver.opti.set_value(p["ref_tangents"], ref_tangents)
        solver.opti.set_value(p["nominal_state"], nominal_state)
        solver.opti.set_value(p["nominal_input"], nominal_input)
        solver.opti.set_value(p["A_mat"], A_mat)
        solver.opti.set_value(p["B_mat"], B_mat)
        solver.opti.set_value(p["c_vec"], c_vec)
        if p.get("prev_cmd") is not None:
            solver.opti.set_value(p["prev_cmd"], np.array([state.v, state.omega], dtype=float))
        if p.get("neigh_pos") is not None: solver.opti.set_value(p["neigh_pos"], neigh_pos)
        if p.get("neigh_ref") is not None: solver.opti.set_value(p["neigh_ref"], neigh_ref)
        if p.get("neigh_pred_spd") is not None: solver.opti.set_value(p["neigh_pred_spd"], neigh_pred_spd)
        if p.get("neigh_ref_spd") is not None: solver.opti.set_value(p["neigh_ref_spd"], neigh_ref_spd)
        if p.get("neigh_ref_tan") is not None: solver.opti.set_value(p["neigh_ref_tan"], neigh_ref_tan)
        if p.get("cbf_hbar") is not None: solver.opti.set_value(p["cbf_hbar"], cbf_hbar)
        if p.get("cbf_grad") is not None: solver.opti.set_value(p["cbf_grad"], cbf_grad)
        if p.get("cbf_hbar_n") is not None: solver.opti.set_value(p["cbf_hbar_n"], cbf_hbar_n)
        if p.get("cbf_grad_n") is not None: solver.opti.set_value(p["cbf_grad_n"], cbf_grad_n)

    def _set_initial_guess(
        self,
        solver: _SolverCacheEntry,
        nominal_state: np.ndarray,
        nominal_input: np.ndarray,
    ) -> None:
        solver.opti.set_initial(solver.variables["X"], nominal_state)
        solver.opti.set_initial(solver.variables["U"], nominal_input)
        eps_cbf = solver.variables.get("eps_cbf")
        if eps_cbf is not None:
            solver.opti.set_initial(eps_cbf, np.zeros((eps_cbf.shape[0], eps_cbf.shape[1]), dtype=float))
        try:
            solver.opti.set_initial(solver.opti.lam_g, 0)
        except Exception:
            pass

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
