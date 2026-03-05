"""Ship node – route-based mobility along shipping lanes."""

import numpy as np
from .base_node import BaseNode
from ..config import SHIP_CONFIG, MAP_WIDTH, MAP_HEIGHT, COASTLINE_X


class ShipNode(BaseNode):
    def __init__(self, index: int = 0):
        super().__init__("ship", SHIP_CONFIG)
        # Start in the ocean area (right of coastline)
        self.position = np.array([
            np.random.uniform(COASTLINE_X + 10_000, MAP_WIDTH - 5_000),
            np.random.uniform(0.05 * MAP_HEIGHT, 0.95 * MAP_HEIGHT),
        ])
        self._route = self._generate_route()
        self._route_idx = 0
        self._set_velocity_toward_next()

    def _generate_route(self) -> list:
        """Generate a simple shipping lane with 4-6 waypoints."""
        n_pts = np.random.randint(4, 7)
        pts = []
        for _ in range(n_pts):
            pts.append(np.array([
                np.random.uniform(COASTLINE_X + 5_000, MAP_WIDTH - 5_000),
                np.random.uniform(0.05 * MAP_HEIGHT, 0.95 * MAP_HEIGHT),
            ]))
        return pts

    def _set_velocity_toward_next(self):
        target = self._route[self._route_idx % len(self._route)]
        direction = target - self.position
        dist = np.linalg.norm(direction)
        if dist < 1.0:
            dist = 1.0
        speed = np.random.uniform(*self.speed_range)
        self.velocity = (direction / dist) * speed

    def update_position(self, dt: float):
        self.position += self.velocity * dt
        self.position = np.clip(
            self.position,
            [COASTLINE_X, 0],
            [MAP_WIDTH, MAP_HEIGHT],
        )
        target = self._route[self._route_idx % len(self._route)]
        if np.linalg.norm(self.position - target) < 1000.0:
            self._route_idx += 1
            if self._route_idx >= len(self._route):
                self._route = self._generate_route()
                self._route_idx = 0
            self._set_velocity_toward_next()
