"""
Neighbour-Discovery Gymnasium wrapper.

Wraps OceanEnv for the RL-based neighbour discovery optimisation task.
Each agent (node) chooses scanning parameters; the environment returns
discovery accuracy improvement and energy cost.
"""

from __future__ import annotations
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.ocean_env import OceanEnv
from env.config import DEFAULT_NOISE_FACTOR, SINR_THRESHOLD_DB
from P1.mechanisms.indp import INDPMechanism


class NeighborDiscoveryEnv(gym.Env):
    """Multi-agent neighbour discovery environment.

    State  (per agent):
      [num_discovered / total_nodes, energy_remaining_ratio,
       avg_sinr_normalised, noise_level_norm, step_ratio]   → dim 5

    Action (per agent) – 3 discrete dimensions flattened:
      tx_power_level   (5 levels)
      scan_duration    (4 levels: 0.05, 0.1, 0.2, 0.5 s)
      verify_threshold (4 levels: 0.4, 0.5, 0.6, 0.7)
      → total 5×4×4 = 80 discrete actions

    Reward:
      α · Δaccuracy  -  β · energy_consumed  +  γ · new_neighbours
    """

    POWER_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
    SCAN_DURATIONS = [0.05, 0.1, 0.2, 0.5]
    VERIFY_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]

    def __init__(
        self,
        node_counts: Optional[Dict[str, int]] = None,
        noise_factor: float = DEFAULT_NOISE_FACTOR,
        episode_length: int = 50,
        alpha: float = 10.0,
        beta: float = 0.1,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.ocean = OceanEnv(
            node_counts=node_counts,
            noise_factor=noise_factor,
            episode_length=episode_length,
            render_mode="none",
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n_actions = (len(self.POWER_LEVELS) *
                          len(self.SCAN_DURATIONS) *
                          len(self.VERIFY_THRESHOLDS))
        self.obs_dim = 5

        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32,
        )

        self.mechanism = INDPMechanism()
        self._prev_accuracy = 0.0
        self._energy_budget = 100.0
        self._energy_used = 0.0

    @property
    def n_agents(self) -> int:
        return len(self.ocean.nodes)

    def reset(self, *, seed=None, options=None):
        self.ocean.reset(seed=seed)
        self.mechanism.reset()
        self._prev_accuracy = 0.0
        self._energy_used = 0.0
        obs = self._build_obs()
        return obs, {}

    def step(self, actions: np.ndarray):
        """actions: array of shape (n_agents,) with integer action indices."""
        self.ocean.step()

        power_idx, scan_idx, thresh_idx = self._decode_action(
            int(actions[0]) if np.ndim(actions) == 0 else int(actions[0])
        )
        power_ratio = self.POWER_LEVELS[power_idx]
        scan_dur = self.SCAN_DURATIONS[scan_idx]
        verify_th = self.VERIFY_THRESHOLDS[thresh_idx]

        self.mechanism.corr_th = verify_th
        self.mechanism.scan_power_ratio = power_ratio

        discovered, energy = self.mechanism.discover(
            self.ocean.nodes,
            noise_factor=self.ocean.noise_factor,
            scan_duration=scan_dur,
        )

        accuracy = self._compute_accuracy(discovered)
        delta_acc = accuracy - self._prev_accuracy
        new_nb = sum(len(v) for v in discovered.values())
        reward = (self.alpha * delta_acc
                  - self.beta * energy
                  + self.gamma * new_nb / max(len(self.ocean.nodes), 1))

        self._prev_accuracy = accuracy
        self._energy_used += energy

        obs = self._build_obs()
        terminated = self.ocean.step_count >= self.ocean.episode_length
        return obs, float(reward), terminated, False, {
            "accuracy": accuracy,
            "energy": self._energy_used,
            "discovered": discovered,
        }

    def _decode_action(self, a: int):
        n_scan = len(self.SCAN_DURATIONS)
        n_thresh = len(self.VERIFY_THRESHOLDS)
        p = a // (n_scan * n_thresh)
        rem = a % (n_scan * n_thresh)
        s = rem // n_thresh
        t = rem % n_thresh
        p = min(p, len(self.POWER_LEVELS) - 1)
        s = min(s, n_scan - 1)
        t = min(t, n_thresh - 1)
        return p, s, t

    def _compute_accuracy(self, discovered: Dict[int, set]) -> float:
        """F1-based accuracy (harmonic mean of precision and recall)."""
        gt = self.ocean.topology_mgr.adjacency
        total_tp = 0
        total_disc = 0
        total_true = 0
        for nid, true_nb in gt.items():
            true_set = set(true_nb.keys())
            disc_set = discovered.get(nid, set())
            total_tp += len(disc_set & true_set)
            total_disc += len(disc_set)
            total_true += len(true_set)
        precision = total_tp / max(total_disc, 1)
        recall = total_tp / max(total_true, 1)
        if precision + recall < 1e-10:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _build_obs(self) -> np.ndarray:
        n = len(self.ocean.nodes)
        disc_ratio = self._prev_accuracy
        energy_ratio = 1.0 - min(self._energy_used / self._energy_budget, 1.0)
        noise_norm = min(self.ocean.noise_factor / 10.0, 1.0)
        step_ratio = self.ocean.step_count / max(self.ocean.episode_length, 1)
        sinr_vals = []
        for nid, nbs in self.ocean.topology_mgr.adjacency.items():
            for sinr_db in nbs.values():
                sinr_vals.append(sinr_db)
        avg_sinr = np.clip(np.mean(sinr_vals) / 30.0, 0.0, 1.0) if sinr_vals else 0.0
        return np.array([disc_ratio, energy_ratio, avg_sinr,
                         noise_norm, step_ratio], dtype=np.float32)

    def set_noise_factor(self, nf: float):
        self.ocean.noise_factor = nf
        self.ocean.set_noise_factor(nf)
