from __future__ import annotations

import math

import numpy as np

from formation.types import (
    FormationControllerReference,
    FormationGuide,
    MPCConfig,
    Point2D,
    RobotReferenceSample,
    RobotReferenceTrajectory,
    RobotState,
    wrap_to_pi,
)


class ControllerReferenceBuilder:
    def __init__(self, config: MPCConfig | None = None) -> None:
        self.config = config or MPCConfig()

    def build(
        self,
        guide: FormationGuide,
        config: MPCConfig | None = None,
        nominal_speed: float | None = None,
        *,
        current_states: "list[RobotState] | None" = None,
    ) -> FormationControllerReference:
        controller_config = config or self.config
        sample_count = len(guide.guide_samples)
        if sample_count <= 0:
            return FormationControllerReference(
                formation_name=guide.formation_name, center_points_xy=[],
                center_heading_rads=[], transition_alphas=[],
                robot_trajectories=[], dt=controller_config.dt,
                horizon_steps=controller_config.horizon_steps,
                metadata={"switched": guide.switched, "nominal_speed": nominal_speed},
            )

        dt = controller_config.dt
        ds_max = controller_config.v_max * dt  # max arc per MPC step

        # ── Subsample keyframes (skip resample if input is already sparse) ──
        if sample_count <= 8:
            kf_xy = [sample.center_xy for sample in guide.guide_samples]
            kf_h = [sample.heading_rad for sample in guide.guide_samples]
        else:
            stride = max(1, sample_count // 8)
            kf_indices = list(range(0, sample_count, stride))
            if kf_indices[-1] != sample_count - 1:
                kf_indices.append(sample_count - 1)
            kf_xy = [guide.guide_samples[i].center_xy for i in kf_indices]
            kf_h = [guide.guide_samples[i].heading_rad for i in kf_indices]

        # Allocate slot local positions (same for every step)
        robot_count = len(guide.guide_samples[0].robot_points_xy)
        slots_local = [
            (float(sl[0]), float(sl[1])) for sl in
            guide.metadata.get("robot_slots_local",
                [(0.0, 0.0)] * robot_count if robot_count else [])
        ]

        # ── Hermite resample (only when subsampled) ──
        if sample_count <= 8:
            dense_xy = np.array([[p[0] for p in kf_xy], [p[1] for p in kf_xy]])
            dense_h = np.array(kf_h)
        else:
            dense_xy, dense_h = self._hermite_resample(kf_xy, kf_h, ds_max)

        transition_alphas = self._normalize_transition_alphas(guide, dense_xy.shape[1])
        robot_trajectories: list[RobotReferenceTrajectory] = []
        unclipped_v_refs: list[list[float]] = []
        unclipped_omega_refs: list[list[float]] = []
        clipped_v_count, clipped_omega_count = 0, 0

        for robot_index in range(robot_count):
            lx, ly = slots_local[robot_index]
            positions_xy: list[Point2D] = []
            for cx, cy, h in zip(dense_xy[0], dense_xy[1], dense_h):
                ch, sh = math.cos(h), math.sin(h)
                wx = cx + ch * lx - sh * ly
                wy = cy + sh * lx + ch * ly
                positions_xy.append((wx, wy))
            # Do not overwrite the reference first sample with the current
            # robot state here. Let the MPC absorb the initial tracking error
            # via its dx0 term (e_0 = p_now - p_ref0). Overwriting p_ref0 and
            # also applying MPC blending can distort the intended reference
            # shape (double-editing); the simulator/selector should prevent
            # choosing formations whose step-0 slots are unreachable.
            heading_rads = self._build_heading_profile(positions_xy, dense_h)
            v_refs, omega_refs = self._build_reference_inputs(
                positions_xy, heading_rads, dt, nominal_speed,
            )
            unclipped_v_refs.append(list(v_refs))
            unclipped_omega_refs.append(list(omega_refs))

            samples: list[RobotReferenceSample] = []
            for step_index, (pos_xy, h, alpha, v_ref, w_ref) in enumerate(
                zip(positions_xy, heading_rads, transition_alphas, v_refs, omega_refs)
            ):
                c_v = min(max(v_ref, 0.0), controller_config.v_max)
                c_w = min(max(w_ref, -controller_config.omega_max), controller_config.omega_max)
                clipped_v_count += int(abs(c_v - v_ref) > 1e-12)
                clipped_omega_count += int(abs(c_w - w_ref) > 1e-12)
                samples.append(RobotReferenceSample(
                    position_xy=pos_xy, yaw=h, v_ref=c_v, omega_ref=c_w,
                    alpha=alpha, t=step_index * dt,
                ))
            robot_trajectories.append(RobotReferenceTrajectory(robot_index=robot_index, samples=samples))

        center_points_xy = [(float(dense_xy[0, i]), float(dense_xy[1, i])) for i in range(dense_xy.shape[1])]
        center_heading_rads = [float(h) for h in dense_h]

        metadata = dict(guide.metadata)
        metadata.update({
            "switched": guide.switched, "nominal_speed": nominal_speed,
            "unclipped_v_refs": unclipped_v_refs,
            "unclipped_omega_refs": unclipped_omega_refs,
            "clipped_v_count": clipped_v_count,
            "clipped_omega_count": clipped_omega_count,
            "terminal_center_heading_rad": center_heading_rads[-1],
            "robot_slots_local": slots_local,
            "terminal_center_xy": center_points_xy[-1],
            "hermite_keyframe_count": len(kf_xy),
        })
        return FormationControllerReference(
            formation_name=guide.formation_name,
            center_points_xy=center_points_xy,
            center_heading_rads=center_heading_rads,
            transition_alphas=transition_alphas,
            robot_trajectories=robot_trajectories,
            dt=controller_config.dt,
            horizon_steps=controller_config.horizon_steps,
            metadata=metadata,
        )

    def _hermite_resample(
        self,
        kf_xy: list[Point2D], kf_h: list[float], ds_max: float,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Hermite cubic interpolation between keyframes, resampled at ds_max."""
        N = len(kf_xy)
        if N < 2:
            return np.array([[kf_xy[0][0]], [kf_xy[0][1]]]), np.array([kf_h[0]])
        # Compute tangent vectors from headings, scaled by chord length
        tangents: list[tuple[float, float]] = []
        for i in range(N):
            ch, sh = math.cos(kf_h[i]), math.sin(kf_h[i])
            if i == 0:
                d = math.hypot(kf_xy[1][0]-kf_xy[0][0], kf_xy[1][1]-kf_xy[0][1])
            elif i == N - 1:
                d = math.hypot(kf_xy[-1][0]-kf_xy[-2][0], kf_xy[-1][1]-kf_xy[-2][1])
            else:
                d = 0.5 * (math.hypot(kf_xy[i][0]-kf_xy[i-1][0], kf_xy[i][1]-kf_xy[i-1][1])
                         + math.hypot(kf_xy[i+1][0]-kf_xy[i][0], kf_xy[i+1][1]-kf_xy[i][1]))
            tangents.append((ch * d, sh * d))

        # Integrate along Hermite spline, drop points at arc step
        all_x, all_y, all_h = [kf_xy[0][0]], [kf_xy[0][1]], [kf_h[0]]
        for seg in range(N - 1):
            p0 = np.array(kf_xy[seg]); p1 = np.array(kf_xy[seg + 1])
            m0 = np.array(tangents[seg]); m1 = np.array(tangents[seg + 1])
            h0, h1 = kf_h[seg], kf_h[seg + 1]
            # Estimate arc length by chord
            seg_len = float(np.linalg.norm(p1 - p0))
            n_samples = max(2, int(seg_len / ds_max))
            for j in range(1, n_samples + 1):
                t = j / n_samples
                t2, t3 = t * t, t * t * t
                pt = ((2*t3 - 3*t2 + 1) * p0 + (t3 - 2*t2 + t) * m0
                      + (-2*t3 + 3*t2) * p1 + (t3 - t2) * m1)
                ht = h0 + t * (h1 - h0)
                all_x.append(float(pt[0])); all_y.append(float(pt[1]))
                all_h.append(float(ht))
        return np.array([all_x, all_y]), np.array(all_h)

    def _normalize_transition_alphas(self, guide: FormationGuide, sample_count: int) -> list[float]:
        if len(guide.transition_alphas) >= sample_count:
            return list(guide.transition_alphas[:sample_count])
        if not guide.transition_alphas:
            fill_value = 0.0 if guide.switched and sample_count > 1 else 1.0
            return [fill_value for _ in range(sample_count)]
        padded = list(guide.transition_alphas)
        while len(padded) < sample_count:
            padded.append(padded[-1])
        return padded

    def _build_reference_inputs(
        self,
        positions_xy: list[Point2D],
        heading_rads: list[float],
        dt: float,
        nominal_speed: float | None,
    ) -> tuple[list[float], list[float]]:
        sample_count = len(positions_xy)
        if sample_count <= 0:
            return [], []

        v_refs: list[float] = []
        omega_refs: list[float] = []
        for step_index in range(sample_count):
            source_index, target_index = self._difference_indices(step_index, sample_count)
            linear_speed = self._distance(positions_xy[source_index], positions_xy[target_index]) / dt
            angular_speed = wrap_to_pi(heading_rads[target_index] - heading_rads[source_index]) / dt
            if nominal_speed is not None and sample_count == 1:
                linear_speed = nominal_speed
            v_refs.append(linear_speed)
            omega_refs.append(angular_speed)
        return v_refs, omega_refs

    def _build_heading_profile(self, positions_xy: list[Point2D], fallback_headings: list[float]) -> list[float]:
        sample_count = len(positions_xy)
        if sample_count <= 0:
            return []
        if sample_count == 1:
            return [float(fallback_headings[0]) if fallback_headings else 0.0]

        headings: list[float] = []
        for step_index in range(sample_count):
            source_index, target_index = self._difference_indices(step_index, sample_count)
            dx = positions_xy[target_index][0] - positions_xy[source_index][0]
            dy = positions_xy[target_index][1] - positions_xy[source_index][1]
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                if step_index > 0:
                    headings.append(headings[-1])
                elif fallback_headings:
                    headings.append(float(fallback_headings[0]))
                else:
                    headings.append(0.0)
            else:
                headings.append(math.atan2(dy, dx))
        return headings

    def _difference_indices(self, step_index: int, sample_count: int) -> tuple[int, int]:
        if sample_count <= 1:
            return 0, 0
        if step_index < sample_count - 1:
            return step_index, step_index + 1
        return sample_count - 2, sample_count - 1

    def _distance(self, point_a_xy: Point2D, point_b_xy: Point2D) -> float:
        return math.hypot(point_b_xy[0] - point_a_xy[0], point_b_xy[1] - point_a_xy[1])
