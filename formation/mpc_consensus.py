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
    x_ir: tuple[float, float, float]
    u_ir: tuple[float, float]
    neighbor_states: list[RobotState]
    neighbor_indices: list[int]
    neighbor_predictions: list[tuple[list[float], list[float]]]
    map_data: MapData
    mode: str = "track"                                      # "switch" | "track"
    ref_world_xy: list[tuple[float, float]] | None = None    # tracking reference
    target_distances: list[float] | None = None              # switch: per-neighbor d_ij


class ConsensusFormationMPC:
    _stdout_lock = __import__("threading").Lock()

    def __init__(self, config: MPCConfig | None = None) -> None:
        self.config = config or MPCConfig()
        self._solver_cache: dict[tuple[int, int, int, int], _SolverCacheEntry] = {}
        self._cache_lock = __import__("threading").Lock()
        self._current_formation: str | None = None
        self._switch_counter: int = 0
        self._switch_dwell: int = 8  # consecutive steps before mode switches

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

        robot_count = len(states)
        goal_idx = min(self.config.horizon_steps, len(controller_reference.center_points_xy) - 1)
        center_xy_all = controller_reference.center_points_xy
        heading_all = controller_reference.center_heading_rads
        x_g = center_xy_all[goal_idx] if center_xy_all else (0.0, 0.0)
        H = self.config.horizon_steps
        dt = self.config.dt
        slots_local = controller_reference.metadata.get("robot_slots_local")
        if slots_local is None or len(slots_local) != robot_count:
            slots_local = [(0.0, 0.0)] * robot_count
        heading_ref = heading_all[0] if heading_all else 0.0
        ch_h, sh_h = math.cos(heading_ref), math.sin(heading_ref)

        # ── Mode detection ──
        fmt_name = controller_reference.formation_name
        switched = (self._current_formation is not None and self._current_formation != fmt_name)
        if switched:
            self._switch_counter = 0
        self._current_formation = fmt_name
        mode = "switch" if self._switch_counter < self._switch_dwell else "track"
        self._switch_counter += 1

        # ── Phase 1: x_ir / u_ir for all robots ──
        all_x_ir: list[tuple[float, float, float]] = []
        all_u_ir: list[tuple[float, float]] = []
        for robot_index, state in enumerate(states):
            local_i = slots_local[robot_index]
            sum_x, sum_y = x_g[0], x_g[1]
            sum_count = 1
            for nj in range(robot_count):
                if nj == robot_index:
                    continue
                local_j = slots_local[nj]
                olx = local_i[0] - local_j[0]
                oly = local_i[1] - local_j[1]
                ox = ch_h * olx - sh_h * oly
                oy = sh_h * olx + ch_h * oly
                ns = states[nj]
                sum_x += ns.x + ox
                sum_y += ns.y + oy
                sum_count += 1
            x_ir_x = sum_x / sum_count
            x_ir_y = sum_y / sum_count
            dx_goal = x_ir_x - state.x
            dy_goal = x_ir_y - state.y
            x_ir_theta = math.atan2(dy_goal, dx_goal) if math.hypot(dx_goal, dy_goal) > 1e-9 else state.yaw
            p_ie_norm = math.hypot(x_ir_x - state.x, x_ir_y - state.y)
            x_th = self.config.v_max * 0.4
            v_ir = (p_ie_norm / max(x_th, 1e-9)) * self.config.v_max if p_ie_norm <= x_th else self.config.v_max
            v_ir = max(0.0, v_ir)
            v_r_x, v_r_y = math.cos(state.yaw), math.sin(state.yaw)
            dot = v_r_x * dx_goal + v_r_y * dy_goal
            cos_angle = min(1.0, max(-1.0, dot / max(p_ie_norm, 1e-9)))
            omega_ir = (self.config.omega_max / math.pi) * math.acos(cos_angle)
            omega_ir = max(-self.config.omega_max, min(self.config.omega_max, omega_ir))
            omega_ir = -abs(omega_ir) if math.atan2(dy_goal, dx_goal) < state.yaw else abs(omega_ir)
            all_x_ir.append((x_ir_x, x_ir_y, x_ir_theta))
            all_u_ir.append((v_ir, omega_ir))

        # ── Phase 2: per-robot inputs ──
        solve_inputs = []
        for robot_index, state in enumerate(states):
            neighbor_states: list[RobotState] = []
            neighbor_indices: list[int] = []
            neighbor_predicted: list[tuple[list[float], list[float]]] = []
            target_distances: list[float] = []
            for j in range(robot_count):
                if j == robot_index:
                    continue
                neighbor_indices.append(j)
                neighbor_states.append(states[j])
                nx_list, ny_list = [states[j].x], [states[j].y]
                px, py, pyaw = states[j].x, states[j].y, states[j].yaw
                v_j, w_j = all_u_ir[j]
                for _ in range(H):
                    px += dt * v_j * math.cos(pyaw)
                    py += dt * v_j * math.sin(pyaw)
                    pyaw += dt * w_j
                    nx_list.append(px)
                    ny_list.append(py)
                neighbor_predicted.append((nx_list, ny_list))
                # Switch-mode: target inter-robot distance from local slots
                li = slots_local[robot_index]
                lj = slots_local[j]
                d_target = math.hypot(li[0] - lj[0], li[1] - lj[1])
                target_distances.append(d_target)

            # Tracking-mode world-frame reference
            ref_world_xy: list[tuple[float, float]] | None = None
            if mode == "track" and len(center_xy_all) > 0:
                li = slots_local[robot_index]
                ref_world_xy = []
                for k in range(min(H + 1, len(center_xy_all))):
                    cx, cy = center_xy_all[k]
                    hk = heading_all[k] if k < len(heading_all) else heading_ref
                    chk, shk = math.cos(hk), math.sin(hk)
                    wx = cx + chk * li[0] - shk * li[1]
                    wy = cy + shk * li[0] + chk * li[1]
                    ref_world_xy.append((wx, wy))

            solve_inputs.append(_RobotSolveInput(
                robot_index=robot_index, solver_slot=robot_index,
                state=state, x_ir=all_x_ir[robot_index], u_ir=all_u_ir[robot_index],
                neighbor_states=neighbor_states, neighbor_indices=neighbor_indices,
                neighbor_predictions=neighbor_predicted, map_data=map_data,
                mode=mode, ref_world_xy=ref_world_xy, target_distances=target_distances,
            ))

        use_parallel = (
            self.config.parallel_solve and len(solve_inputs) > 1
            and self._resolve_parallel_workers(len(solve_inputs)) > 1
        )
        if use_parallel:
            with ThreadPoolExecutor(max_workers=self._resolve_parallel_workers(len(solve_inputs))) as executor:
                results = list(executor.map(self._solve_robot_from_input, solve_inputs))
        else:
            results = [self._solve_robot_from_input(solve_input) for solve_input in solve_inputs]
        results.sort(key=lambda item: item[1].robot_index)
        return [c for c, _ in results], [p for _, p in results]

    def _solve_robot_from_input(self, solve_input: _RobotSolveInput) -> tuple[ControlCommand, RobotPrediction]:
        return self.solve_robot(
            solve_input.state, solve_input.x_ir, solve_input.u_ir,
            solve_input.neighbor_states, solve_input.map_data,
            solver_slot=solve_input.solver_slot,
            neighbor_indices=solve_input.neighbor_indices,
            neighbor_predictions=solve_input.neighbor_predictions,
            mode=solve_input.mode,
            ref_world_xy=solve_input.ref_world_xy,
            target_distances=solve_input.target_distances,
        )

    def solve_robot(
        self,
        state: RobotState,
        x_ir: tuple[float, float, float],
        u_ir: tuple[float, float],
        neighbor_states: list[RobotState],
        map_data: MapData,
        *,
        solver_slot: int = 0,
        neighbor_indices: list[int] | None = None,
        neighbor_predictions: list[tuple[list[float], list[float]]] | None = None,
        mode: str = "track",
        ref_world_xy: list[tuple[float, float]] | None = None,
        target_distances: list[float] | None = None,
    ) -> tuple[ControlCommand, RobotPrediction]:
        solver = self._get_solver(map_data, len(neighbor_states), solver_slot, mode)

        H = self.config.horizon_steps
        neighbor_positions = np.zeros((2 * len(neighbor_states), H + 1), dtype=float)
        if neighbor_predictions:
            for n, (nx_list, ny_list) in enumerate(neighbor_predictions):
                for k in range(min(H + 1, len(nx_list))):
                    neighbor_positions[2 * n, k] = nx_list[k]
                    neighbor_positions[2 * n + 1, k] = ny_list[k]
        else:
            for n, ns in enumerate(neighbor_states):
                for k in range(H + 1):
                    neighbor_positions[2 * n, k] = ns.x
                    neighbor_positions[2 * n + 1, k] = ns.y

        # Build tracking reference array
        ref_track_arr = None
        if mode == "track" and ref_world_xy:
            ref_track_arr = np.zeros((2, H + 1), dtype=float)
            for k in range(min(H + 1, len(ref_world_xy))):
                ref_track_arr[0, k] = ref_world_xy[k][0]
                ref_track_arr[1, k] = ref_world_xy[k][1]

        self._set_parameter_values(solver, state, x_ir, u_ir, neighbor_positions, ref_track_arr, target_distances)
        self._set_initial_guess(solver)
        with ConsensusFormationMPC._stdout_lock:
            old_fd = os.dup(1)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.close(devnull)
            try:
                solution = solver.opti.solve()
                solver_ok = True
            except RuntimeError:
                solution = None
                solver_ok = False
            finally:
                os.dup2(old_fd, 1)
                os.close(old_fd)

        if not solver_ok:
            # Open-loop fallback
            H = self.config.horizon_steps
            dt = self.config.dt
            cmds = [ControlCommand(v=float(u_ir[0]), omega=float(u_ir[1])) for _ in range(H)]
            absolutes = np.zeros((3, H + 1), dtype=float)
            absolutes[:, 0] = np.array([state.x, state.y, state.yaw])
            for k in range(H):
                absolutes[0, k + 1] = absolutes[0, k] + dt * cmds[k].v * math.cos(absolutes[2, k])
                absolutes[1, k + 1] = absolutes[1, k] + dt * cmds[k].v * math.sin(absolutes[2, k])
                absolutes[2, k + 1] = wrap_to_pi(absolutes[2, k] + dt * cmds[k].omega)
            positions_xy = [(float(absolutes[0, k]), float(absolutes[1, k])) for k in range(H + 1)]
            yaw_rads = [float(absolutes[2, k]) for k in range(H + 1)]
            indices = neighbor_indices if neighbor_indices is not None else list(range(len(neighbor_states)))
            return cmds[0], RobotPrediction(
                robot_index=solver_slot, positions_xy=positions_xy, yaw_rads=yaw_rads,
                commands=cmds, obstacle_slacks=[0.0]*(H+1),
                neighbor_slacks=[[0.0]*len(neighbor_states) for _ in range(H+1)],
                metadata={"neighbor_robot_indices": indices, "objective": float("nan"),
                          "solver_status": "fallback_open_loop"},
            )

        dx_value = np.asarray(solution.value(solver.variables["dx"]), dtype=float)
        du_value = np.asarray(solution.value(solver.variables["du"]), dtype=float)
        if neighbor_states and solver.variables["eps_n"] is not None:
            eps_value = np.asarray(solution.value(solver.variables["eps_n"]), dtype=float)
            if eps_value.ndim == 1:
                eps_value = eps_value.reshape(len(neighbor_states), self.config.horizon_steps + 1)
        else:
            eps_value = np.zeros((0, self.config.horizon_steps + 1), dtype=float)

        x_ir_x, x_ir_y, x_ir_theta = x_ir
        dt = self.config.dt
        absolutes = np.zeros((3, dx_value.shape[1]), dtype=float)
        absolutes[:, 0] = np.array([state.x, state.y, state.yaw], dtype=float)
        v_ir_val, w_ir_val = u_ir
        for k in range(1, dx_value.shape[1]):
            v_k = max(0.0, v_ir_val + du_value[0, k - 1])
            w_k = du_value[1, k - 1]
            absolutes[0, k] = absolutes[0, k - 1] + dt * v_k * math.cos(absolutes[2, k - 1])
            absolutes[1, k] = absolutes[1, k - 1] + dt * v_k * math.sin(absolutes[2, k - 1])
            absolutes[2, k] = wrap_to_pi(absolutes[2, k - 1] + dt * w_k)

        positions_xy = [(float(absolutes[0, k]), float(absolutes[1, k])) for k in range(absolutes.shape[1])]
        yaw_rads = [float(wrap_to_pi(absolutes[2, k])) for k in range(absolutes.shape[1])]
        commands = [
            ControlCommand(
                v=float(max(0.0, min(self.config.v_max, v_ir_val + du_value[0, k]))),
                omega=float(max(-self.config.omega_max, min(self.config.omega_max, du_value[1, k]))),
            )
            for k in range(du_value.shape[1])
        ]
        neighbor_slacks = [
            [float(eps_value[n, k]) for n in range(eps_value.shape[0])]
            for k in range(eps_value.shape[1])
        ]

        indices = neighbor_indices if neighbor_indices is not None else list(range(len(neighbor_states)))
        prediction = RobotPrediction(
            robot_index=solver_slot,
            positions_xy=positions_xy,
            yaw_rads=yaw_rads,
            commands=list(commands),
            obstacle_slacks=[0.0 for _ in range(self.config.horizon_steps + 1)],
            neighbor_slacks=neighbor_slacks,
            metadata={
                "neighbor_robot_indices": indices,
                "objective": float(solution.value(solver.opti.f)),
                "solver_status": solver.opti.stats().get("return_status", "unknown"),
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

    def _get_solver(self, map_data: MapData, neighbor_count: int, solver_slot: int, mode: str) -> _SolverCacheEntry:
        cache_key = (id(map_data), self.config.horizon_steps, neighbor_count, solver_slot, mode)
        with self._cache_lock:
            if cache_key not in self._solver_cache:
                self._solver_cache[cache_key] = self._build_solver(map_data, neighbor_count, mode)
            return self._solver_cache[cache_key]

    def _build_solver(self, map_data: MapData, neighbor_count: int, mode: str) -> _SolverCacheEntry:
        ca = _require_casadi()
        opti = ca.Opti()
        H = self.config.horizon_steps
        dt = self.config.dt
        w = self.config.weights

        dx = opti.variable(3, H + 1)
        du = opti.variable(2, H)

        dx0 = opti.parameter(3)
        x_ir_xy = opti.parameter(2)
        x_ir_theta_p = opti.parameter(1)
        v_ir = opti.parameter(1)
        # Tracking mode: world-frame reference positions
        ref_track_xy = opti.parameter(2, H + 1) if mode == "track" else None
        # Switch mode: per-neighbor target distances
        target_dist_p = opti.parameter(neighbor_count) if mode == "switch" and neighbor_count > 0 else None
        neighbor_positions = opti.parameter(2 * neighbor_count, H + 1) if neighbor_count > 0 else None

        safe_neighbor_distance = 2.0 * self.config.robot_radius + self.config.inter_robot_margin
        w_form = 30.0  # formation shape weight (switch mode)

        objective = 0
        opti.subject_to(dx[0, 0] == dx0[0])
        opti.subject_to(dx[1, 0] == dx0[1])
        opti.subject_to(dx[2, 0] == dx0[2])

        for k in range(H):
            self_x_k = x_ir_xy[0] + dx[0, k]
            self_y_k = x_ir_xy[1] + dx[1, k]
            self_theta_k = x_ir_theta_p + dx[2, k]
            v_k = v_ir + du[0, k]
            w_k = du[1, k]
            self_x_next = self_x_k + dt * v_k * ca.cos(self_theta_k)
            self_y_next = self_y_k + dt * v_k * ca.sin(self_theta_k)
            self_theta_next = self_theta_k + dt * w_k
            opti.subject_to(dx[0, k + 1] == self_x_next - x_ir_xy[0])
            opti.subject_to(dx[1, k + 1] == self_y_next - x_ir_xy[1])
            theta_diff = self_theta_next - x_ir_theta_p
            opti.subject_to(dx[2, k + 1] == ca.atan2(ca.sin(theta_diff), ca.cos(theta_diff)))
            opti.subject_to(opti.bounded(0.0, v_k, self.config.v_max))
            opti.subject_to(opti.bounded(-self.config.omega_max, w_k, self.config.omega_max))

            if mode == "track" and ref_track_xy is not None:
                pos_err = (self_x_k - ref_track_xy[0, k])**2 + (self_y_k - ref_track_xy[1, k])**2
                objective += w.position * pos_err
            else:
                # Switch mode: track consensus x_ir + formation shape
                objective += w.position * (dx[0, k]**2 + dx[1, k]**2)

            objective += (
                w.heading * dx[2, k]**2
                + w.input * ca.sumsqr(du[:, k])
            )
            if k > 0:
                objective += w.input_smooth * ca.sumsqr(du[:, k] - du[:, k - 1])

        # Terminal
        if mode == "track" and ref_track_xy is not None:
            last_err_x = (x_ir_xy[0] + dx[0, H]) - ref_track_xy[0, H]
            last_err_y = (x_ir_xy[1] + dx[1, H]) - ref_track_xy[1, H]
            objective += w.terminal_position * (last_err_x**2 + last_err_y**2)
        else:
            objective += w.terminal_position * (dx[0, H]**2 + dx[1, H]**2)

        # Switch mode: formation shape cost — match inter-robot distances
        if mode == "switch" and neighbor_positions is not None and target_dist_p is not None and neighbor_count > 0:
            for k in range(H + 1):
                for n in range(neighbor_count):
                    n_x = neighbor_positions[2 * n, k]
                    n_y = neighbor_positions[2 * n + 1, k]
                    self_x = x_ir_xy[0] + dx[0, k]
                    self_y = x_ir_xy[1] + dx[1, k]
                    d_actual = ca.sqrt((self_x - n_x)**2 + (self_y - n_y)**2 + 1e-9)
                    d_target = target_dist_p[n]
                    objective += w_form * (d_actual - d_target)**2

        # ── Neighbour separation (both modes) ──
        eps_n = None
        if neighbor_positions is not None and neighbor_count > 0:
            eps_n = opti.variable(neighbor_count, H + 1)
            for k in range(H + 1):
                for n in range(neighbor_count):
                    opti.subject_to(eps_n[n, k] >= 0.0)
                    n_x = neighbor_positions[2 * n, k]
                    n_y = neighbor_positions[2 * n + 1, k]
                    self_x = x_ir_xy[0] + dx[0, k]
                    self_y = x_ir_xy[1] + dx[1, k]
                    dist = ca.sqrt((self_x - n_x)**2 + (self_y - n_y)**2 + 1e-9)
                    opti.subject_to(dist + eps_n[n, k] >= safe_neighbor_distance)
                    objective += w.neighbor_slack * eps_n[n, k]**2

        opti.minimize(objective)
        opti.solver(
            "ipopt",
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 100,
                "ipopt.tol": 1e-4,
            },
        )
        params: dict[str, Any] = {
            "dx0": dx0, "x_ir_xy": x_ir_xy,
            "x_ir_theta_p": x_ir_theta_p, "v_ir": v_ir,
            "neighbor_positions": neighbor_positions,
        }
        if ref_track_xy is not None:
            params["ref_track_xy"] = ref_track_xy
        if target_dist_p is not None:
            params["target_dist"] = target_dist_p
        return _SolverCacheEntry(
            opti=opti,
            variables={"dx": dx, "du": du, "eps_n": eps_n},
            parameters=params,
        )

    def _set_parameter_values(
        self, solver: _SolverCacheEntry,
        state: RobotState, x_ir: tuple[float, float, float],
        u_ir: tuple[float, float], neighbor_positions: np.ndarray,
        ref_track_arr: np.ndarray | None = None,
        target_distances: list[float] | None = None,
    ) -> None:
        x_ir_x, x_ir_y, x_ir_theta = x_ir
        solver.opti.set_value(solver.parameters["dx0"], np.array([
            state.x - x_ir_x, state.y - x_ir_y,
            wrap_to_pi(state.yaw - x_ir_theta),
        ], dtype=float))
        solver.opti.set_value(solver.parameters["x_ir_xy"], np.array([x_ir_x, x_ir_y], dtype=float))
        solver.opti.set_value(solver.parameters["x_ir_theta_p"], np.array([x_ir_theta], dtype=float))
        solver.opti.set_value(solver.parameters["v_ir"], np.array([u_ir[0]], dtype=float))
        if solver.parameters.get("neighbor_positions") is not None:
            solver.opti.set_value(solver.parameters["neighbor_positions"], neighbor_positions)
        if ref_track_arr is not None and solver.parameters.get("ref_track_xy") is not None:
            solver.opti.set_value(solver.parameters["ref_track_xy"], ref_track_arr)
        if target_distances is not None and solver.parameters.get("target_dist") is not None:
            solver.opti.set_value(solver.parameters["target_dist"], np.array(target_distances, dtype=float))

    def _set_initial_guess(self, solver: _SolverCacheEntry) -> None:
        H = self.config.horizon_steps
        solver.opti.set_initial(solver.variables["dx"], np.zeros((3, H + 1), dtype=float))
        solver.opti.set_initial(solver.variables["du"], np.zeros((2, H), dtype=float))
        if solver.variables["eps_n"] is not None:
            solver.opti.set_initial(
                solver.variables["eps_n"],
                np.zeros((solver.variables["eps_n"].shape[0], H + 1), dtype=float),
            )

    def _build_neighbor_position_vector(self, neighbor_predictions: list[RobotPrediction]) -> np.ndarray:
        if not neighbor_predictions:
            return np.zeros((0, self.config.horizon_steps + 1), dtype=float)
        horizon = self.config.horizon_steps
        rows = np.zeros((2 * len(neighbor_predictions), horizon + 1), dtype=float)
        for pred_idx, prediction in enumerate(neighbor_predictions):
            normalized = self._normalize_prediction(prediction)
            for step in range(horizon + 1):
                pos = normalized.positions_xy[step] if step < len(normalized.positions_xy) else normalized.positions_xy[-1]
                rows[2 * pred_idx, step] = pos[0]
                rows[2 * pred_idx + 1, step] = pos[1]
        return rows

    def _normalize_prediction(self, prediction: RobotPrediction) -> RobotPrediction:
        sample_count = self.config.horizon_steps + 1
        if not prediction.positions_xy:
            raise ValueError(f"Prediction for robot {prediction.robot_index} is empty.")
        positions_xy = list(prediction.positions_xy[:sample_count])
        yaw_rads = list(prediction.yaw_rads[:sample_count])
        commands = list(prediction.commands[: self.config.horizon_steps])
        obstacle_slacks = list(prediction.obstacle_slacks[:sample_count])
        neighbor_slacks = list(prediction.neighbor_slacks[:sample_count])
        while len(positions_xy) < sample_count:
            positions_xy.append(positions_xy[-1])
        while len(yaw_rads) < sample_count:
            yaw_rads.append(yaw_rads[-1])
        while len(commands) < self.config.horizon_steps:
            commands.append(commands[-1] if commands else ControlCommand(v=0.0, omega=0.0))
        while len(obstacle_slacks) < sample_count:
            obstacle_slacks.append(obstacle_slacks[-1] if obstacle_slacks else 0.0)
        while len(neighbor_slacks) < sample_count:
            neighbor_slacks.append(list(neighbor_slacks[-1]) if neighbor_slacks else [])
        return RobotPrediction(
            robot_index=prediction.robot_index,
            positions_xy=positions_xy,
            yaw_rads=yaw_rads,
            commands=commands,
            obstacle_slacks=obstacle_slacks,
            neighbor_slacks=neighbor_slacks,
            metadata=dict(prediction.metadata),
        )

    def _deform_reference_matrix(self, state: RobotState, own_reference: RobotReferenceTrajectory) -> np.ndarray:
        return np.asarray(
            [
                [sample.position_xy[0] for sample in own_reference.samples],
                [sample.position_xy[1] for sample in own_reference.samples],
                [sample.yaw for sample in own_reference.samples],
                [sample.v_ref for sample in own_reference.samples],
                [sample.omega_ref for sample in own_reference.samples],
            ],
            dtype=float,
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
        raise ModuleNotFoundError("casadi is required to use ConsensusFormationMPC.") from exc
    return ca


