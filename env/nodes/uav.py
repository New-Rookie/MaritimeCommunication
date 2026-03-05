"""UAV node – random-waypoint mobility model."""

import numpy as np
from .base_node import BaseNode
from ..config import UAV_CONFIG, MAP_WIDTH, MAP_HEIGHT


class UAVNode(BaseNode):
    def __init__(self, index: int = 0):
        super().__init__("uav", UAV_CONFIG)
        self.position = np.array([
            np.random.uniform(0.1 * MAP_WIDTH, 0.9 * MAP_WIDTH),
            np.random.uniform(0.1 * MAP_HEIGHT, 0.9 * MAP_HEIGHT),
        ])
        self._waypoint = self._random_waypoint()
        self._set_velocity_toward_waypoint()

    def _random_waypoint(self) -> np.ndarray:
        return np.array([
            np.random.uniform(0.05 * MAP_WIDTH, 0.95 * MAP_WIDTH),
            np.random.uniform(0.05 * MAP_HEIGHT, 0.95 * MAP_HEIGHT),
        ])

    def _set_velocity_toward_waypoint(self):
        direction = self._waypoint - self.position
        dist = np.linalg.norm(direction)
        if dist < 1.0:
            self._waypoint = self._random_waypoint()
            direction = self._waypoint - self.position
            dist = np.linalg.norm(direction)
        speed = np.random.uniform(*self.speed_range)
        self.velocity = (direction / dist) * speed

    def update_position(self, dt: float):
        self.position += self.velocity * dt
        # Clamp within map
        self.position = np.clip(self.position, 0, [MAP_WIDTH, MAP_HEIGHT])
        if np.linalg.norm(self.position - self._waypoint) < 500.0:
            self._waypoint = self._random_waypoint()
            self._set_velocity_toward_waypoint()
