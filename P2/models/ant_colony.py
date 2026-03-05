"""ACO-based link selector."""

from __future__ import annotations
import numpy as np


class ACOLinkSelector:
    def __init__(self, max_neighbours: int = 15,
                 alpha: float = 1.0, beta: float = 2.0,
                 evap: float = 0.1, q: float = 10.0):
        self.alpha = alpha
        self.beta = beta
        self.evap = evap
        self.q = q
        self.pheromone = np.ones(max_neighbours)

    def select_action(self, sinr_values: np.ndarray,
                      valid_mask: np.ndarray) -> int:
        if valid_mask.sum() < 0.5:
            return 0
        heuristic = np.maximum(sinr_values, 1e-10) * valid_mask
        ph = self.pheromone[:len(sinr_values)]
        probs = (ph ** self.alpha) * (heuristic ** self.beta) * valid_mask
        s = probs.sum()
        if s < 1e-15:
            probs = valid_mask.astype(np.float64)
            s = probs.sum()
            if s < 1e-15:
                return 0
        probs = probs / s
        probs = np.nan_to_num(probs, nan=0.0)
        s2 = probs.sum()
        if s2 < 1e-15:
            return 0
        probs = probs / s2
        return int(np.random.choice(len(probs), p=probs))

    def update(self, action: int, reward: float):
        self.pheromone *= (1 - self.evap)
        if 0 <= action < len(self.pheromone):
            self.pheromone[action] += self.q * max(reward, 0)
