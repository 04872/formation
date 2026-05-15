from __future__ import annotations

import math

import numpy as np

from formation.distance_field import compute_distance_field, compute_inflated_occupancy
from formation.map_config import (
    BaseMapConfig,
    NarrowEntranceConfig,
    NarrowingCorridorConfig,
    ObstacleClusterConfig,
    RightAngleCorridorConfig,
    SCurveCorridorConfig,
)
from formation.types import MapData


class MapBuilder:
    def build(self, map_type: str, config: BaseMapConfig) -> MapData:
        builders = {
            "right_angle_corridor": self._build_right_angle_corridor,
            "s_curve_corridor": self._build_s_curve_corridor,
            "obstacle_cluster": self._build_obstacle_cluster,
            "narrow_entrance": self._build_narrow_entrance,
            "narrowing_corridor": self._build_narrowing_corridor,
        }
        if map_type not in builders:
            raise ValueError(f"Unsupported map type: {map_type}")
        return builders[map_type](config)

    def _origin_xy(self, config: BaseMapConfig) -> tuple[float, float]:
        return (-config.width_m / 2.0, -config.height_m / 2.0)

    def _grid_shape(self, config: BaseMapConfig) -> tuple[int, int]:
        rows = int(round(config.height_m / config.resolution))
        cols = int(round(config.width_m / config.resolution))
        return rows, cols

    def _make_full_occupancy(self, config: BaseMapConfig) -> np.ndarray:
        return np.ones(self._grid_shape(config), dtype=bool)

    def _make_free_map(self, config: BaseMapConfig) -> np.ndarray:
        grid = np.zeros(self._grid_shape(config), dtype=bool)
        grid[[0, -1], :] = True
        grid[:, [0, -1]] = True
        return grid

    def _cell_centers(self, config: BaseMapConfig) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self._grid_shape(config)
        origin_x, origin_y = self._origin_xy(config)
        xs = origin_x + (np.arange(cols) + 0.5) * config.resolution
        ys = origin_y + (np.arange(rows) + 0.5) * config.resolution
        return np.meshgrid(xs, ys)

    def _finalize_map(
        self,
        name: str,
        config: BaseMapConfig,
        occupancy: np.ndarray,
        obstacle_primitives: list[dict[str, object]],
    ) -> MapData:
        distance_field = compute_distance_field(occupancy, config.resolution)
        inflation_radius = (
            config.robot_radius + config.safety_margin + config.global_planner_margin
        )
        inflated = compute_inflated_occupancy(
            occupancy,
            inflation_radius,
            config.resolution,
            distance_field=distance_field,
        )
        map_data = MapData(
            name=name,
            resolution=config.resolution,
            width_m=config.width_m,
            height_m=config.height_m,
            origin_xy=self._origin_xy(config),
            occupancy=occupancy,
            inflated_occupancy=inflated,
            distance_field=distance_field,
            start_xy=config.start_xy,
            goal_xy=config.goal_xy,
            obstacle_primitives=obstacle_primitives,
            inflation_radius=inflation_radius,
        )
        self._validate_start_goal(map_data)
        return map_data

    def _validate_start_goal(self, map_data: MapData) -> None:
        for point_name, xy in (("start", map_data.start_xy), ("goal", map_data.goal_xy)):
            rc = map_data.world_to_grid(xy)
            if map_data.is_occupied(rc):
                raise ValueError(f"{point_name} {xy} lies inside obstacle on map {map_data.name}.")
            if map_data.is_occupied(rc, inflated=True):
                raise ValueError(
                    f"{point_name} {xy} lies inside inflated obstacle on map {map_data.name}."
                )

    def _rasterize_rectangle(
        self,
        occupancy: np.ndarray,
        config: BaseMapConfig,
        center_xy: tuple[float, float],
        width: float,
        height: float,
    ) -> None:
        x_grid, y_grid = self._cell_centers(config)
        cx, cy = center_xy
        inside = (np.abs(x_grid - cx) <= width / 2.0) & (np.abs(y_grid - cy) <= height / 2.0)
        occupancy[inside] = True

    def _rasterize_circle(
        self,
        occupancy: np.ndarray,
        config: BaseMapConfig,
        center_xy: tuple[float, float],
        radius: float,
    ) -> None:
        x_grid, y_grid = self._cell_centers(config)
        cx, cy = center_xy
        inside = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2
        occupancy[inside] = True

    def _build_right_angle_corridor(self, config: BaseMapConfig) -> MapData:
        if not isinstance(config, RightAngleCorridorConfig):
            config = RightAngleCorridorConfig(**config.__dict__)
        occupancy = self._make_full_occupancy(config)
        x_grid, y_grid = self._cell_centers(config)

        horizontal_leg = (
            (x_grid <= config.corner_x)
            & (np.abs(y_grid - config.horizontal_center_y) <= config.corridor_width / 2.0)
        )
        vertical_leg = (
            (y_grid >= config.corner_y)
            & (np.abs(x_grid - config.vertical_center_x) <= config.corridor_width / 2.0)
        )
        occupancy[horizontal_leg | vertical_leg] = False
        occupancy[[0, -1], :] = True
        occupancy[:, [0, -1]] = True
        obstacle_primitives = [
            {
                "type": "right_angle_corridor",
                "corridor_width": config.corridor_width,
                "corner_xy": (config.corner_x, config.corner_y),
                "horizontal_center_y": config.horizontal_center_y,
                "vertical_center_x": config.vertical_center_x,
            }
        ]
        return self._finalize_map("right_angle_corridor", config, occupancy, obstacle_primitives)

    def _build_s_curve_corridor(self, config: BaseMapConfig) -> MapData:
        if not isinstance(config, SCurveCorridorConfig):
            config = SCurveCorridorConfig(**config.__dict__)
        occupancy = self._make_full_occupancy(config)
        x_grid, y_grid = self._cell_centers(config)
        path_span = config.goal_xy[0] - config.start_xy[0]
        if path_span <= 0.0:
            raise ValueError("S-curve corridor requires goal x to be greater than start x.")
        phase = 2.0 * math.pi * (x_grid - config.start_xy[0]) / path_span
        centerline_y = config.amplitude * np.sin(phase)
        free_mask = np.abs(y_grid - centerline_y) <= config.corridor_width / 2.0
        occupancy[free_mask] = False
        occupancy[[0, -1], :] = True
        occupancy[:, [0, -1]] = True
        obstacle_primitives = [
            {
                "type": "s_curve_corridor",
                "amplitude": config.amplitude,
                "corridor_width": config.corridor_width,
                "period_m": config.period_m,
            }
        ]
        return self._finalize_map("s_curve_corridor", config, occupancy, obstacle_primitives)

    def _build_obstacle_cluster(self, config: BaseMapConfig) -> MapData:
        if not isinstance(config, ObstacleClusterConfig):
            config = ObstacleClusterConfig(**config.__dict__)
        occupancy = self._make_free_map(config)
        obstacle_primitives: list[dict[str, object]] = []
        for center_xy, width, height in config.rectangular_obstacles:
            self._rasterize_rectangle(occupancy, config, center_xy, width, height)
            obstacle_primitives.append(
                {
                    "type": "rectangle",
                    "center_xy": center_xy,
                    "width": width,
                    "height": height,
                }
            )
        for center_xy, radius in config.circular_obstacles:
            self._rasterize_circle(occupancy, config, center_xy, radius)
            obstacle_primitives.append(
                {"type": "circle", "center_xy": center_xy, "radius": radius}
            )
        return self._finalize_map("obstacle_cluster", config, occupancy, obstacle_primitives)

    def _build_narrow_entrance(self, config: BaseMapConfig) -> MapData:
        if not isinstance(config, NarrowEntranceConfig):
            config = NarrowEntranceConfig(**config.__dict__)
        occupancy = self._make_full_occupancy(config)
        x_grid, y_grid = self._cell_centers(config)

        left_room = (
            (x_grid >= -config.width_m / 2.0 + config.resolution)
            & (x_grid <= -config.neck_length / 2.0)
            & (np.abs(y_grid) <= config.room_height / 2.0)
        )
        right_room = (
            (x_grid >= config.neck_length / 2.0)
            & (x_grid <= config.width_m / 2.0 - config.resolution)
            & (np.abs(y_grid) <= config.room_height / 2.0)
        )
        neck = (
            (x_grid >= -config.neck_length / 2.0)
            & (x_grid <= config.neck_length / 2.0)
            & (np.abs(y_grid) <= config.neck_width / 2.0)
        )
        occupancy[left_room | right_room | neck] = False
        occupancy[[0, -1], :] = True
        occupancy[:, [0, -1]] = True
        obstacle_primitives = [
            {
                "type": "narrow_entrance",
                "room_height": config.room_height,
                "neck_width": config.neck_width,
                "neck_length": config.neck_length,
            }
        ]
        return self._finalize_map("narrow_entrance", config, occupancy, obstacle_primitives)

    def _build_narrowing_corridor(self, config: BaseMapConfig) -> MapData:
        if not isinstance(config, NarrowingCorridorConfig):
            config = NarrowingCorridorConfig(**config.__dict__)
        occupancy = self._make_full_occupancy(config)
        x_grid, y_grid = self._cell_centers(config)

        transition_span = config.transition_end_x - config.transition_start_x
        if transition_span <= 0.0:
            raise ValueError("Narrowing corridor requires transition_end_x > transition_start_x.")

        alpha = np.clip(
            (x_grid - config.transition_start_x) / transition_span,
            0.0,
            1.0,
        )
        centerline_y = config.centerline_slope * x_grid + config.centerline_intercept
        transition_width = config.wide_width + alpha * (config.narrow_width - config.wide_width)

        local_width = np.full_like(x_grid, config.wide_width, dtype=float)
        transition_mask = (
            (x_grid >= config.transition_start_x)
            & (x_grid <= config.transition_end_x)
        )
        tail_mask = x_grid > config.tail_start_x
        local_width[transition_mask] = transition_width[transition_mask]
        local_width[tail_mask] = config.tail_width

        inside_corridor = (
            (x_grid >= config.corridor_x_min)
            & (x_grid <= config.tail_end_x)
            & (np.abs(y_grid - centerline_y) <= local_width / 2.0)
        )
        inside_corridor |= (
            (x_grid >= config.corridor_x_min - config.throat_padding_x)
            & (x_grid < config.corridor_x_min)
            & (np.abs(y_grid - centerline_y) <= config.wide_width / 2.0)
        )
        occupancy[inside_corridor] = False
        occupancy[[0, -1], :] = True
        occupancy[:, [0, -1]] = True
        obstacle_primitives = [
            {
                "type": "narrowing_corridor",
                "corridor_x_min": config.corridor_x_min,
                "corridor_x_max": config.corridor_x_max,
                "wide_width": config.wide_width,
                "narrow_width": config.narrow_width,
                "tail_width": config.tail_width,
                "transition_start_x": config.transition_start_x,
                "transition_end_x": config.transition_end_x,
                "tail_start_x": config.tail_start_x,
                "tail_end_x": config.tail_end_x,
                "centerline_slope": config.centerline_slope,
                "centerline_intercept": config.centerline_intercept,
            }
        ]
        return self._finalize_map("narrowing_corridor", config, occupancy, obstacle_primitives)
