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

        # ── Strictly use guide centreline (no subsample, no Hermite) ──
        c_xy = np.array([[s.center_xy[0] for s in guide.guide_samples],
                          [s.center_xy[1] for s in guide.guide_samples]])
        N = c_xy.shape[1]
        # centreline heading from finite differences
        psi = np.zeros(N, dtype=float)
        for k in range(N):
            if k == 0:       d = c_xy[:, 1] - c_xy[:, 0]
            elif k == N - 1: d = c_xy[:, -1] - c_xy[:, -2]
            else:            d = c_xy[:, k + 1] - c_xy[:, k - 1]
            dn = float(np.linalg.norm(d))
            psi[k] = math.atan2(d[1], d[0]) if dn > 1e-9 else 0.0

        robot_count = len(guide.guide_samples[0].robot_points_xy)
        slots_local = [
            (float(sl[0]), float(sl[1])) for sl in
            guide.metadata.get("robot_slots_local",
                [(0.0, 0.0)] * robot_count if robot_count else [])
        ]

        transition_alphas = self._normalize_transition_alphas(guide, N)
        robot_trajectories: list[RobotReferenceTrajectory] = []

        for robot_index in range(robot_count):
            lx, ly = slots_local[robot_index]
            positions_xy: list[Point2D] = []
            for k in range(N):
                ch, sh = math.cos(psi[k]), math.sin(psi[k])
                wx = c_xy[0, k] + ch * lx - sh * ly
                wy = c_xy[1, k] + sh * lx + ch * ly
                positions_xy.append((wx, wy))

            # ── per-robot yaw from own slot trajectory ───────────
            robot_yaws: list[float] = []
            for k in range(N):
                if k == 0:
                    d = (positions_xy[1][0] - positions_xy[0][0],
                         positions_xy[1][1] - positions_xy[0][1])
                elif k == N - 1:
                    d = (positions_xy[-1][0] - positions_xy[-2][0],
                         positions_xy[-1][1] - positions_xy[-2][1])
                else:
                    d = (positions_xy[k + 1][0] - positions_xy[k - 1][0],
                         positions_xy[k + 1][1] - positions_xy[k - 1][1])
                dn = math.hypot(d[0], d[1])
                if dn > 1e-9:
                    robot_yaws.append(math.atan2(d[1], d[0]))
                elif robot_yaws:
                    robot_yaws.append(robot_yaws[-1])
                else:
                    robot_yaws.append(float(psi[k]))
            # unwrap
            for k in range(1, len(robot_yaws)):
                robot_yaws[k] = robot_yaws[k - 1] + wrap_to_pi(
                    robot_yaws[k] - robot_yaws[k - 1])

            # ── omega_ref from robot_yaws ────────────────────────
            omega_raw: list[float] = []
            for k in range(N):
                if k < N - 1:
                    dw = wrap_to_pi(robot_yaws[k + 1] - robot_yaws[k])
                    omega_raw.append(dw / max(dt, 1e-9))
                else:
                    omega_raw.append(omega_raw[-1] if omega_raw else 0.0)
            omega_refs = self._smooth_speeds(
                omega_raw, dt, controller_config.omega_max,
                rho=0.75, lam=0.4, lower=-1.0, accel_max=1.3,
            )

            # ── v_ref: projection onto robot's own heading ──────
            v_raw: list[float] = []
            for k in range(N):
                if k < N - 1:
                    dq = np.array(positions_xy[k + 1]) - np.array(positions_xy[k])
                    fwd = np.array([math.cos(robot_yaws[k]), math.sin(robot_yaws[k])])
                    v_tilde = float(np.dot(dq, fwd)) / max(dt, 1e-9)
                    v_raw.append(v_tilde)
                else:
                    v_raw.append(v_raw[-1] if v_raw else 0.0)
            v_refs = self._smooth_speeds(
                v_raw, dt, controller_config.v_max,
                rho=0.55, lam=0.4, lower=0.0, accel_max=0.20,
            )

            samples: list[RobotReferenceSample] = []
            for k, (pos_xy, alpha, v_ref, w_ref) in enumerate(
                zip(positions_xy, transition_alphas, v_refs, omega_refs)
            ):
                samples.append(RobotReferenceSample(
                    position_xy=pos_xy, yaw=float(robot_yaws[k]),
                    v_ref=v_ref, omega_ref=w_ref,
                    alpha=alpha, t=k * dt,
                ))
            robot_trajectories.append(RobotReferenceTrajectory(robot_index=robot_index, samples=samples))

        center_points_xy = [(float(c_xy[0, i]), float(c_xy[1, i])) for i in range(N)]
        center_heading_rads = [float(h) for h in psi]

        metadata = dict(guide.metadata)
        metadata.update({
            "switched": guide.switched, "nominal_speed": nominal_speed,
            "terminal_center_heading_rad": center_heading_rads[-1],
            "robot_slots_local": slots_local,
            "terminal_center_xy": center_points_xy[-1],
            "invalidate_previous_predictions": guide.switched,
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

    @staticmethod
    def _smooth_speeds(
        raw: list[float],
        dt: float,
        max_val: float,
        *,
        rho: float = 0.7,
        lam: float = 0.4,
        lower: float = 0.0,
        accel_max: float | None = None,
    ) -> list[float]:
        """Clip, EMA-smooth, and acceleration-limit a speed/omega profile.

        Steps:
          1. clip to [lower · max_val, rho · max_val]
          2. forward‑backward EMA (zero‑phase)
          3. acceleration limit (optional, absolute max change per dt)
        """
        lo = lower * max_val
        hi = rho * max_val
        N = len(raw)

        # ── 1. clip ──────────────────────────────────────────────
        clipped = [max(lo, min(hi, v)) for v in raw]

        # ── 2. forward‑backward EMA ─────────────────────────────
        fwd = [clipped[0]]
        for k in range(1, N):
            fwd.append(lam * clipped[k] + (1.0 - lam) * fwd[-1])

        bwd = [fwd[-1]]
        for k in range(N - 2, -1, -1):
            bwd.insert(0, lam * fwd[k] + (1.0 - lam) * bwd[0])

        smooth = [(f + b) / 2.0 for f, b in zip(fwd, bwd)]

        # ── 3. acceleration limit ──────────────────────────────
        if accel_max is None:
            return smooth
        out = [smooth[0]]
        for k in range(1, N):
            dv = smooth[k] - out[-1]
            max_step = accel_max * dt
            dv = max(-max_step, min(max_step, dv))
            out.append(out[-1] + dv)
        return out

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
