"""
Greedy strategy for neighbour discovery parameter selection.

Each step picks the parameter combo that maximises estimated
discovery-gain / energy-cost ratio based on the previous step.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class GreedyOptimizer:

    POWER_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
    SCAN_DURATIONS = [0.05, 0.1, 0.2, 0.5]
    VERIFY_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]

    def __init__(self):
        self._scores: dict = {}

    def select_params(self, prev_accuracy: float = 0.0) -> Tuple[float, float, float]:
        best_score = -1e9
        best = (self.POWER_LEVELS[2], self.SCAN_DURATIONS[1],
                self.VERIFY_THRESHOLDS[1])
        for p in self.POWER_LEVELS:
            for s in self.SCAN_DURATIONS:
                for t in self.VERIFY_THRESHOLDS:
                    key = (p, s, t)
                    score = self._scores.get(key, 0.0)
                    if score > best_score:
                        best_score = score
                        best = key
        return best

    def update(self, params: Tuple, accuracy: float, energy: float):
        gain_per_cost = accuracy / max(energy, 1e-10)
        old = self._scores.get(params, 0.0)
        self._scores[params] = 0.7 * old + 0.3 * gain_per_cost
