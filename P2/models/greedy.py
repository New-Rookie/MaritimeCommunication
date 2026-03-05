"""Greedy link selector – picks neighbour with best SINR."""

from __future__ import annotations
import numpy as np


class GreedyLinkSelector:

    def select_action(self, sinr_values: np.ndarray,
                      valid_mask: np.ndarray) -> int:
        if valid_mask.sum() < 0.5:
            return 0
        masked = sinr_values * valid_mask - 1e9 * (1 - valid_mask)
        return int(np.argmax(masked))
