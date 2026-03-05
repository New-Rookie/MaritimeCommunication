"""Buoy (IoT sensor) node – ocean-current drift mobility."""

import numpy as np
from .base_node import BaseNode
from ..config import BUOY_CONFIG, MAP_WIDTH, MAP_HEIGHT, COASTLINE_X


class BuoyNode(BaseNode):
    def __init__(self, index: int = 0):
        super().__init__("buoy", BUOY_CONFIG)
        self.position = np.array([
            np.random.uniform(COASTLINE_X + 5_000, MAP_WIDTH - 5_000),
            np.random.uniform(0.1 * MAP_HEIGHT, 0.9 * MAP_HEIGHT),
        ])
        self._update_drift()

    def _update_drift(self):
        """Brownian-like drift driven by ocean current."""
        speed = np.random.uniform(*self.speed_range)
        # Predominant eastward current with random perturbation
        angle = np.random.uniform(-np.pi / 3, np.pi / 3)
        self.velocity = speed * np.array([np.cos(angle), np.sin(angle)])

    def update_position(self, dt: float):
        # Small random perturbation each step
        perturbation = np.random.randn(2) * 0.05
        self.position += (self.velocity + perturbation) * dt
        self.position = np.clip(
            self.position,
            [COASTLINE_X, 0],
            [MAP_WIDTH, MAP_HEIGHT],
        )
        # Occasionally change drift direction
        if np.random.random() < 0.01:
            self._update_drift()
