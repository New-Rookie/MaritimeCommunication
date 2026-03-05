"""LEO satellite node – orbital ground-track motion with map wrapping."""

import numpy as np
from .base_node import BaseNode
from ..config import SATELLITE_CONFIG, MAP_WIDTH, MAP_HEIGHT


class SatelliteNode(BaseNode):
    def __init__(self, index: int = 0):
        super().__init__("satellite", SATELLITE_CONFIG)
        # Spread satellites evenly across the map at start
        self.position = np.array([
            (index + 0.5) / max(SATELLITE_CONFIG.count, 1) * MAP_WIDTH,
            np.random.uniform(0.2 * MAP_HEIGHT, 0.8 * MAP_HEIGHT),
        ])
        speed = np.mean(self.speed_range)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = speed * np.array([np.cos(angle), np.sin(angle)])

    def update_position(self, dt: float):
        self.position += self.velocity * dt
        # Periodic wrap-around (simulates orbital re-visit)
        self.position[0] %= MAP_WIDTH
        self.position[1] %= MAP_HEIGHT
