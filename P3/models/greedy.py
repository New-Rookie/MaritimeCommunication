"""Greedy baseline for resource management.
Allocates resources proportional to demand / capacity ratio."""

from __future__ import annotations
import numpy as np


class GreedyRM:
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Return action in [-1,1]^4: [bw_alloc, comp_alloc, stor_alloc, offload_ratio]."""
        return np.array([0.5, 0.5, 0.5, 0.3])

    def store(self, *args):
        pass

    def update(self):
        return 0.0
