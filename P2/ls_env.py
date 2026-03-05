"""
Link Selection Gymnasium environment wrapper.

Each agent (data source) selects a next-hop neighbour.
Reward penalises latency, energy, and link-switching.

Optimisation target:
  min J = w1·T_avg + w2·E_avg + w3·S_switch
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
from env.channel.interference import calculate_sinr_linear, SINRBatchCalculator
from env.config import (
    DEFAULT_NOISE_FACTOR, MAP_WIDTH, MAP_HEIGHT, SPEED_OF_LIGHT,
    TASK_DATA_SIZE, TASK_ARRIVAL_RATE, NOISE_PSD,
)


class LinkSelectionEnv(gym.Env):
    """Multi-agent link selection environment."""

    MAX_NEIGHBOURS = 15

    def __init__(
        self,
        node_counts: Optional[Dict[str, int]] = None,
        noise_factor: float = DEFAULT_NOISE_FACTOR,
        episode_length: int = 200,
        w_delay: float = 0.4,
        w_energy: float = 0.3,
        w_switch: float = 0.3,
    ):
        super().__init__()
        self.ocean = OceanEnv(
            node_counts=node_counts,
            noise_factor=noise_factor,
            episode_length=episode_length,
            render_mode="none",
        )
        self.w_delay = w_delay
        self.w_energy = w_energy
        self.w_switch = w_switch

        self._prev_links: Dict[int, int] = {}
        self._switch_count = 0
        self._total_delay = 0.0
        self._total_energy = 0.0
        self._n_transmissions = 0
        self._sinr_calc: Optional[SINRBatchCalculator] = None

    @property
    def n_agents(self):
        return len([n for n in self.ocean.nodes if n.node_type == "buoy"])

    def _rebuild_sinr_calc(self):
        """Rebuild the batch SINR calculator (call after topology update)."""
        self._sinr_calc = SINRBatchCalculator(
            self.ocean.nodes, self.ocean.noise_factor)

    def reset(self, *, seed=None, options=None):
        self.ocean.reset(seed=seed)
        self._prev_links.clear()
        self._switch_count = 0
        self._total_delay = 0.0
        self._total_energy = 0.0
        self._n_transmissions = 0
        self._rebuild_sinr_calc()
        return self._get_obs(), {}

    def step(self, actions: Dict[int, int]):
        self.ocean.step()
        if self.ocean.step_count % self.ocean.topo_update_interval == 0 or self._sinr_calc is None:
            self._rebuild_sinr_calc()

        step_delay = 0.0
        step_energy = 0.0
        step_switches = 0

        sources = [n for n in self.ocean.nodes if n.node_type == "buoy"]
        adj = self.ocean.topology_mgr.adjacency
        id_map = {n.id: n for n in self.ocean.nodes}

        for src in sources:
            neighbours = list(adj.get(src.id, {}).keys())
            if len(neighbours) == 0:
                continue

            action_idx = actions.get(src.id, 0)
            action_idx = min(action_idx, len(neighbours) - 1)
            chosen_id = neighbours[action_idx]

            chosen_node = id_map.get(chosen_id)
            if chosen_node is None:
                continue

            ti = self._sinr_calc.idx_of(src.id)
            ri = self._sinr_calc.idx_of(chosen_id)
            sinr_lin = self._sinr_calc.sinr_linear_pair(ti, ri)
            sinr_lin = max(sinr_lin, 0.01)

            bw = min(src.bandwidth, chosen_node.bandwidth)
            capacity = bw * np.log2(1.0 + sinr_lin)
            data_bits = TASK_DATA_SIZE * 8
            t_tx = data_bits / max(capacity, 1e-10)
            d_prop = src.distance_to(chosen_node) / SPEED_OF_LIGHT

            # M/M/1 queuing delay: arrival rate shared across available MEC nodes
            n_mec = max(len([nd for nd in self.ocean.nodes if nd.is_mec]), 1)
            arrival_rate = TASK_ARRIVAL_RATE * len(sources) / n_mec
            mu_bw = bw / max(TASK_DATA_SIZE * 8, 1e-30)
            w_queue = 1.0 / max(mu_bw - arrival_rate, 1e-6) if mu_bw > arrival_rate else 10.0

            delay = min(t_tx + d_prop + w_queue, 50.0)

            rx_power_w = getattr(chosen_node, 'tx_power', 0.05) * 0.1
            energy = min(src.tx_power * t_tx + rx_power_w * t_tx, 50.0)

            step_delay += delay
            step_energy += energy
            self._n_transmissions += 1

            if src.id in self._prev_links and self._prev_links[src.id] != chosen_id:
                step_switches += 1
            self._prev_links[src.id] = chosen_id

        n_src = max(len(sources), 1)
        avg_delay = step_delay / n_src
        avg_energy = step_energy / n_src
        switch_rate = step_switches / n_src

        self._total_delay += step_delay
        self._total_energy += step_energy
        self._switch_count += step_switches

        reward = -(self.w_delay * avg_delay
                   + self.w_energy * avg_energy
                   + self.w_switch * switch_rate)

        terminated = self.ocean.step_count >= self.ocean.episode_length
        info = {
            "avg_delay": avg_delay,
            "avg_energy": avg_energy,
            "switch_rate": switch_rate,
            "total_delay": self._total_delay,
            "total_energy": self._total_energy,
            "total_switches": self._switch_count,
            "stability": 1.0 - self._switch_count / max(self._n_transmissions, 1),
        }
        return self._get_obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, Any]:
        positions = np.array([n.position / MAP_WIDTH for n in self.ocean.nodes],
                             dtype=np.float32)
        velocities = np.array([n.velocity / 100.0 for n in self.ocean.nodes],
                              dtype=np.float32)
        bw_ratios = np.array([n.avail_bw / max(n.bandwidth, 1) for n in self.ocean.nodes],
                             dtype=np.float32)
        comp_ratios = np.array([n.avail_comp / max(n.compute, 1) for n in self.ocean.nodes],
                               dtype=np.float32)
        load = 1.0 - bw_ratios

        node_features = np.column_stack([
            positions, velocities, bw_ratios.reshape(-1, 1),
            comp_ratios.reshape(-1, 1), load.reshape(-1, 1),
        ]).astype(np.float32)

        edges = self.ocean.topology_mgr.get_edges()
        id_list = [n.id for n in self.ocean.nodes]
        id_to_idx = {nid: i for i, nid in enumerate(id_list)}
        if edges:
            src = [id_to_idx[a] for a, b, _ in edges if a in id_to_idx and b in id_to_idx]
            dst = [id_to_idx[b] for a, b, _ in edges if a in id_to_idx and b in id_to_idx]
            edge_index = np.array([src + dst, dst + src], dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "adjacency": self.ocean.topology_mgr.adjacency,
        }

    def get_local_obs(self, node_id: int) -> np.ndarray:
        id_map = {n.id: n for n in self.ocean.nodes}
        node = id_map.get(node_id)
        if node is None:
            return np.zeros(8, dtype=np.float32)
        nbs = self.ocean.topology_mgr.adjacency.get(node_id, {})
        return np.array([
            node.position[0] / MAP_WIDTH,
            node.position[1] / MAP_HEIGHT,
            node.velocity[0] / 100.0,
            node.velocity[1] / 100.0,
            node.avail_bw / max(node.bandwidth, 1),
            node.avail_comp / max(node.compute, 1),
            len(nbs) / self.MAX_NEIGHBOURS,
            self.ocean.step_count / max(self.ocean.episode_length, 1),
        ], dtype=np.float32)

    def get_valid_mask(self, node_id: int) -> np.ndarray:
        nbs = list(self.ocean.topology_mgr.adjacency.get(node_id, {}).keys())
        mask = np.zeros(self.MAX_NEIGHBOURS, dtype=np.float32)
        mask[:min(len(nbs), self.MAX_NEIGHBOURS)] = 1.0
        return mask

    def get_global_state(self) -> np.ndarray:
        n = len(self.ocean.nodes)
        positions = np.array([nd.position for nd in self.ocean.nodes])
        mean_pos = positions.mean(axis=0) / MAP_WIDTH
        std_pos = positions.std(axis=0) / MAP_WIDTH
        edges = len(self.ocean.topology_mgr.get_edges())
        avg_bw = np.mean([nd.avail_bw / max(nd.bandwidth, 1) for nd in self.ocean.nodes])
        avg_comp = np.mean([nd.avail_comp / max(nd.compute, 1) for nd in self.ocean.nodes])
        step_r = self.ocean.step_count / max(self.ocean.episode_length, 1)
        state = np.array([
            n / 100.0, mean_pos[0], mean_pos[1], std_pos[0], std_pos[1],
            edges / 500.0, avg_bw, avg_comp, step_r,
            self.ocean.noise_factor / 10.0,
            self._switch_count / max(self._n_transmissions + 1, 1),
        ], dtype=np.float32)
        padded = np.zeros(64, dtype=np.float32)
        padded[:len(state)] = state
        return padded

    def get_sinr_values(self, node_id: int) -> np.ndarray:
        """Batch-accelerated SINR for a node's neighbours."""
        nbs = list(self.ocean.topology_mgr.adjacency.get(node_id, {}).keys())
        if self._sinr_calc is not None:
            return self._sinr_calc.sinr_linear_all_neighbours(
                node_id, nbs, self.MAX_NEIGHBOURS)
        vals = np.zeros(self.MAX_NEIGHBOURS, dtype=np.float32)
        return vals

    def set_noise_factor(self, nf: float):
        self.ocean.noise_factor = nf
        self.ocean.set_noise_factor(nf)
