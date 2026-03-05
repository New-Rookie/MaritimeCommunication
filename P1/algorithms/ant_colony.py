"""
Ant Colony Optimisation (ACO) for neighbour discovery parameter tuning.

Discrete parameter combinations are treated as graph edges.
Pheromone is deposited proportional to discovery accuracy achieved.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


class AntColonyOptimizer:
    """ACO baseline for neighbour discovery parameter selection."""

    POWER_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
    SCAN_DURATIONS = [0.05, 0.1, 0.2, 0.5]
    VERIFY_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]

    def __init__(
        self,
        n_ants: int = 20,
        alpha_ph: float = 1.0,
        beta_h: float = 2.0,
        evaporation: float = 0.1,
        q: float = 100.0,
    ):
        self.n_ants = n_ants
        self.alpha_ph = alpha_ph
        self.beta_h = beta_h
        self.evaporation = evaporation
        self.q = q

        dims = [len(self.POWER_LEVELS), len(self.SCAN_DURATIONS),
                len(self.VERIFY_THRESHOLDS)]
        self.pheromone = [np.ones(d) for d in dims]

    def select_params(self) -> Tuple[int, int, int]:
        indices = []
        for ph in self.pheromone:
            probs = ph ** self.alpha_ph
            s = probs.sum()
            if s < 1e-15:
                probs = np.ones_like(ph) / len(ph)
            else:
                probs = probs / s
            probs = np.nan_to_num(probs, nan=1.0 / len(ph))
            probs = probs / probs.sum()
            idx = np.random.choice(len(ph), p=probs)
            indices.append(idx)
        return tuple(indices)

    def update_pheromone(self, solutions: List[Tuple[Tuple, float]]):
        for ph in self.pheromone:
            ph *= (1 - self.evaporation)
        for (p, s, t), fitness in solutions:
            self.pheromone[0][p] += self.q * fitness
            self.pheromone[1][s] += self.q * fitness
            self.pheromone[2][t] += self.q * fitness

    def get_action_values(self) -> Tuple[float, float, float]:
        """Return best current parameter combination values."""
        p = int(np.argmax(self.pheromone[0]))
        s = int(np.argmax(self.pheromone[1]))
        t = int(np.argmax(self.pheromone[2]))
        return (self.POWER_LEVELS[p], self.SCAN_DURATIONS[s],
                self.VERIFY_THRESHOLDS[t])
