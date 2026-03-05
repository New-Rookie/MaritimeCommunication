"""
Resource Management Gymnasium environment wrapper.

Each MEC-capable node allocates resources for incoming tasks:
  Action (continuous, 4-dim per agent):
    [bw_alloc_ratio, comp_alloc_ratio, stor_alloc_ratio, offload_ratio]
    Each in [-1, 1], mapped to actual values.

Reward: minimise delay + energy, maximise throughput
  R = -α·T_total - β·E_total + γ·Throughput

Physical models:
  - Shannon capacity for transmission delay
  - M/M/1 queuing for each resource
  - CMOS dynamic power model for computation energy
  - Partial offloading with ratio δ
"""

from __future__ import annotations
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.ocean_env import OceanEnv
from env.channel.interference import calculate_sinr_linear
from env.network.mec import Task
from env.config import (
    DEFAULT_NOISE_FACTOR, MAP_WIDTH, MAP_HEIGHT,
    TASK_DATA_SIZE, TASK_COMPUTE_CYCLES, TASK_RESULT_SIZE,
    CMOS_KAPPA, TASK_ARRIVAL_RATE, NOISE_PSD,
)

HISTORY_LEN = 16
_EPS = 1e-30


class ResourceManagementEnv(gym.Env):
    """Multi-agent resource management environment."""

    def __init__(
        self,
        node_counts: Optional[Dict[str, int]] = None,
        noise_factor: float = DEFAULT_NOISE_FACTOR,
        episode_length: int = 200,
        bw_scale: float = 1.0,
        comp_scale: float = 1.0,
        stor_scale: float = 1.0,
        alpha_delay: float = 1.0,
        beta_energy: float = 0.5,
        gamma_throughput: float = 2.0,
    ):
        super().__init__()
        self.ocean = OceanEnv(
            node_counts=node_counts,
            noise_factor=noise_factor,
            episode_length=episode_length,
            render_mode="none",
        )
        self.bw_scale = bw_scale
        self.comp_scale = comp_scale
        self.stor_scale = stor_scale
        self.alpha = alpha_delay
        self.beta = beta_energy
        self.gamma_tp = gamma_throughput

        self.obs_dim = 12
        self.action_dim = 4

        self._total_delay = 0.0
        self._total_energy = 0.0
        self._total_throughput = 0.0
        self._completed = 0

        # Per-agent state history for IMATD3
        self._obs_history: Dict[int, deque] = {}

    @property
    def n_agents(self):
        return len(self.ocean.get_mec_nodes())

    def reset(self, *, seed=None, options=None):
        self.ocean.reset(seed=seed)
        self._total_delay = 0.0
        self._total_energy = 0.0
        self._total_throughput = 0.0
        self._completed = 0
        self._obs_history.clear()
        if hasattr(self, '_state_hist'):
            del self._state_hist

        for node in self.ocean.nodes:
            node.bandwidth *= self.bw_scale
            node.avail_bw *= self.bw_scale
            node.compute *= self.comp_scale
            node.avail_comp *= self.comp_scale
            node.storage *= self.stor_scale
            node.avail_stor *= self.stor_scale

        return self._get_obs(), {}

    def step(self, actions: Dict[int, np.ndarray]):
        """
        actions: {mec_node_id: [bw_ratio, comp_ratio, stor_ratio, offload_ratio]}
        Each value in [-1, 1], mapped as:
          bw_alloc   = (a[0]+1)/2 * avail_bw
          comp_alloc = (a[1]+1)/2 * avail_comp
          stor_alloc = (a[2]+1)/2 * avail_stor
          offload_ratio = (a[3]+1)/2   i.e. δ ∈ [0, 1]
        """
        self.ocean.step()

        step_delay = 0.0
        step_energy = 0.0
        step_throughput = 0.0

        mec_nodes = self.ocean.get_mec_nodes()
        buoys = self.ocean.get_nodes_by_type("buoy")
        adj = self.ocean.topology_mgr.adjacency

        # Process pending tasks
        tasks = list(self.ocean.mec_mgr.pending_tasks)
        self.ocean.mec_mgr.pending_tasks.clear()

        for task in tasks:
            src_node = None
            for n in self.ocean.nodes:
                if n.id == task.source_id:
                    src_node = n
                    break
            if src_node is None:
                continue

            # Find nearest MEC node
            best_mec = None
            best_sinr = -1e9
            for mn in mec_nodes:
                if mn.id in adj.get(src_node.id, {}):
                    sinr = calculate_sinr_linear(
                        src_node, mn, self.ocean.nodes, self.ocean.noise_factor)
                    if sinr > best_sinr:
                        best_sinr = sinr
                        best_mec = mn

            if best_mec is None or best_sinr < 1e-10:
                continue

            action = actions.get(best_mec.id, np.zeros(4))
            bw_ratio = (action[0] + 1.0) / 2.0
            comp_ratio = (action[1] + 1.0) / 2.0
            stor_ratio = (action[2] + 1.0) / 2.0
            delta = (action[3] + 1.0) / 2.0

            bw_alloc = bw_ratio * best_mec.avail_bw
            comp_alloc = comp_ratio * best_mec.avail_comp
            stor_alloc = stor_ratio * best_mec.avail_stor
            bw_alloc = max(bw_alloc, 1e3)
            comp_alloc = max(comp_alloc, 1e6)
            stor_alloc = max(stor_alloc, 1e3)

            arrival_rate = TASK_ARRIVAL_RATE * len(buoys) / max(len(mec_nodes), 1)
            avg_data = TASK_DATA_SIZE
            avg_cycles = TASK_COMPUTE_CYCLES
            avg_stor = TASK_DATA_SIZE

            if delta < 0.01:
                delay = task.total_delay_no_offload(
                    bw_alloc, best_sinr, comp_alloc,
                    arrival_rate, avg_data, avg_cycles, avg_stor, stor_alloc)
                energy = task.energy_no_offload(
                    comp_alloc, src_node.tx_power, bw_alloc, best_sinr)
            else:
                delay = task.total_delay_with_offload(
                    delta, src_node.compute, comp_alloc,
                    bw_alloc, bw_alloc, best_sinr, best_sinr,
                    arrival_rate, avg_data, avg_cycles, avg_stor, stor_alloc)
                energy = task.energy_with_offload(
                    delta, src_node.compute, src_node.tx_power, 0.05,
                    bw_alloc, bw_alloc, best_sinr, best_sinr)

            delay = min(delay, 100.0)
            energy = min(energy, 100.0)

            step_delay += delay
            step_energy += energy
            step_throughput += task.data_size
            self._completed += 1

            best_mec.consume_resource(
                bw_alloc * 0.1, comp_alloc * 0.1, stor_alloc * 0.1)

        n_tasks = max(len(tasks), 1)
        avg_delay = step_delay / n_tasks
        avg_energy = step_energy / n_tasks
        throughput = step_throughput / self.ocean.dt

        self._total_delay += step_delay
        self._total_energy += step_energy
        self._total_throughput += step_throughput

        reward = (-self.alpha * avg_delay
                  - self.beta * avg_energy
                  + self.gamma_tp * throughput / 1e6)

        terminated = self.ocean.step_count >= self.ocean.episode_length
        info = {
            "avg_delay": avg_delay,
            "avg_energy": avg_energy,
            "throughput": throughput,
            "total_delay": self._total_delay,
            "total_energy": self._total_energy,
            "total_throughput": self._total_throughput / max(self.ocean.step_count, 1),
            "completed_tasks": self._completed,
        }
        return self._get_obs(), float(reward), terminated, False, info

    def _get_obs(self) -> Dict[str, Any]:
        mec_nodes = self.ocean.get_mec_nodes()
        obs = {}
        for mn in mec_nodes:
            o = np.array([
                mn.position[0] / MAP_WIDTH,
                mn.position[1] / MAP_HEIGHT,
                mn.avail_bw / max(mn.bandwidth, 1),
                mn.avail_comp / max(mn.compute, 1),
                mn.avail_stor / max(mn.storage, 1),
                len(self.ocean.topology_mgr.adjacency.get(mn.id, {})) / 20.0,
                self.ocean.noise_factor / 10.0,
                self.ocean.step_count / max(self.ocean.episode_length, 1),
                self._total_delay / max(self.ocean.step_count + 1, 1) / 10.0,
                self._total_energy / max(self.ocean.step_count + 1, 1) / 10.0,
                self._completed / max(self.ocean.step_count + 1, 1) / 5.0,
                float(mn.node_type == "base_station"),
            ], dtype=np.float32)
            obs[mn.id] = o
        return obs

    def get_global_state(self) -> np.ndarray:
        n = len(self.ocean.nodes)
        mec = self.ocean.get_mec_nodes()
        avg_bw = np.mean([m.avail_bw / max(m.bandwidth, 1) for m in mec]) if mec else 0
        avg_comp = np.mean([m.avail_comp / max(m.compute, 1) for m in mec]) if mec else 0
        avg_stor = np.mean([m.avail_stor / max(m.storage, 1) for m in mec]) if mec else 0
        s = np.array([
            n / 100.0,
            len(mec) / 50.0,
            avg_bw, avg_comp, avg_stor,
            self.ocean.noise_factor / 10.0,
            self.ocean.step_count / max(self.ocean.episode_length, 1),
            self._total_delay / max(self.ocean.step_count + 1, 1) / 10.0,
            self._total_energy / max(self.ocean.step_count + 1, 1) / 10.0,
            self._total_throughput / max(self.ocean.step_count + 1, 1) / 1e6,
            self._completed / max(self.ocean.step_count + 1, 1) / 5.0,
        ], dtype=np.float32)
        padded = np.zeros(64, dtype=np.float32)
        padded[:len(s)] = s
        return padded

    def get_obs_history(self, node_id: int, obs: np.ndarray) -> np.ndarray:
        if node_id not in self._obs_history:
            self._obs_history[node_id] = deque(
                [np.zeros(self.obs_dim, dtype=np.float32)] * HISTORY_LEN,
                maxlen=HISTORY_LEN)
        self._obs_history[node_id].append(obs)
        return np.array(list(self._obs_history[node_id]), dtype=np.float32)

    def get_state_history(self) -> np.ndarray:
        gs = self.get_global_state()
        if not hasattr(self, '_state_hist'):
            self._state_hist = deque(
                [np.zeros(64, dtype=np.float32)] * HISTORY_LEN,
                maxlen=HISTORY_LEN)
        self._state_hist.append(gs)
        return np.array(list(self._state_hist), dtype=np.float32)

    def set_noise_factor(self, nf: float):
        self.ocean.noise_factor = nf
        self.ocean.set_noise_factor(nf)
