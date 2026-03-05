"""Land base-station node – static, located along the coastline."""

import numpy as np
from .base_node import BaseNode
from ..config import BASE_STATION_CONFIG, COASTLINE_X, MAP_HEIGHT


class BaseStationNode(BaseNode):
    def __init__(self, index: int = 0):
        super().__init__("base_station", BASE_STATION_CONFIG)
        n = max(BASE_STATION_CONFIG.count, 1)
        self.position = np.array([
            COASTLINE_X,
            (index + 0.5) / n * MAP_HEIGHT,
        ])
        self.velocity = np.zeros(2)

    def update_position(self, dt: float):
        pass  # static
