"""
OceanEnv – Gymnasium-compatible simulation of a Space-Air-Ground-Sea
integrated heterogeneous network for Maritime IoT with MEC.

Provides:
  - Node creation & mobility
  - Channel / SINR computation
  - Topology management
  - MEC task generation
  - Pygame visualisation (optional)
"""

from __future__ import annotations
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    MAP_WIDTH, MAP_HEIGHT, DT, DEFAULT_EPISODE_LENGTH,
    DEFAULT_NOISE_FACTOR, NODE_CONFIGS, SINR_THRESHOLD_DB,
    TASK_ARRIVAL_RATE, TASK_DATA_SIZE, TASK_COMPUTE_CYCLES,
)
from .nodes import (
    BaseNode, SatelliteNode, UAVNode, ShipNode, BuoyNode, BaseStationNode,
)
from .mobility.models import update_all_positions
from .channel.interference import calculate_sinr_db, calculate_sinr_linear
from .network.topology import TopologyManager
from .network.mec import MECManager, Task


_NODE_CLS = {
    "satellite": SatelliteNode,
    "uav": UAVNode,
    "ship": ShipNode,
    "buoy": BuoyNode,
    "base_station": BaseStationNode,
}


class OceanEnv(gym.Env):
    """Core simulation environment."""

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(
        self,
        node_counts: Optional[Dict[str, int]] = None,
        noise_factor: float = DEFAULT_NOISE_FACTOR,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        render_mode: str = "none",
        dt: float = DT,
        topo_update_interval: int = 5,
    ):
        super().__init__()
        self.dt = dt
        self.noise_factor = noise_factor
        self.episode_length = episode_length
        self.render_mode = render_mode
        self.topo_update_interval = max(1, topo_update_interval)

        self._node_counts = {}
        for ntype, cfg in NODE_CONFIGS.items():
            self._node_counts[ntype] = (
                node_counts[ntype] if node_counts and ntype in node_counts
                else cfg.count
            )

        self.nodes: List[BaseNode] = []
        self.topology_mgr = TopologyManager(SINR_THRESHOLD_DB)
        self.mec_mgr = MECManager()

        self.step_count = 0
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        BaseNode.reset_id_counter()
        self.nodes.clear()
        self.mec_mgr.reset()
        self.step_count = 0

        for ntype, count in self._node_counts.items():
            cls = _NODE_CLS[ntype]
            for idx in range(count):
                self.nodes.append(cls(index=idx))

        # Initial topology
        self.topology_mgr.build_ground_truth_neighbours(
            self.nodes, self.noise_factor
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: Any = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # 1. Mobility
        update_all_positions(self.nodes, self.dt)

        # 2. Rebuild topology (skip steps for performance: nodes move
        #    slowly relative to comm range so topology is stable over
        #    short intervals)
        if self.step_count % self.topo_update_interval == 0:
            self.topology_mgr.build_ground_truth_neighbours(
                self.nodes, self.noise_factor
            )

        # 3. Generate tasks from buoys (Poisson arrival)
        for node in self.nodes:
            if node.node_type == "buoy":
                if np.random.random() < TASK_ARRIVAL_RATE * self.dt:
                    self.mec_mgr.generate_task(node.id)

        # 4. Resource recovery (partial, proportional to dt)
        for node in self.nodes:
            node.restore_resource(
                node.bandwidth * 0.1 * self.dt,
                node.compute * 0.1 * self.dt,
                node.storage * 0.1 * self.dt,
            )

        obs = self._get_obs()
        reward = 0.0
        terminated = self.step_count >= self.episode_length
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "none":
            return
        if self._renderer is None:
            from .visualization.renderer import PygameRenderer
            self._renderer = PygameRenderer(self)
        self._renderer.draw()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Observation & info helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, Any]:
        """Structured observation dictionary."""
        node_positions = np.array([n.position for n in self.nodes])
        node_types = [n.node_type for n in self.nodes]
        adjacency = self.topology_mgr.adjacency
        return {
            "positions": node_positions,
            "types": node_types,
            "adjacency": adjacency,
            "step": self.step_count,
        }

    def _get_info(self) -> Dict[str, Any]:
        edges = self.topology_mgr.get_edges()
        mec_nodes = [n for n in self.nodes if n.is_mec]
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(edges),
            "num_mec_nodes": len(mec_nodes),
            "pending_tasks": len(self.mec_mgr.pending_tasks),
        }

    # ------------------------------------------------------------------
    # Public helpers for sub-experiments
    # ------------------------------------------------------------------

    def get_nodes_by_type(self, ntype: str) -> List[BaseNode]:
        return [n for n in self.nodes if n.node_type == ntype]

    def get_mec_nodes(self) -> List[BaseNode]:
        return [n for n in self.nodes if n.is_mec]

    def get_sinr_db(self, tx, rx) -> float:
        return calculate_sinr_db(tx, rx, self.nodes, self.noise_factor)

    def get_sinr_linear(self, tx, rx) -> float:
        return calculate_sinr_linear(tx, rx, self.nodes, self.noise_factor)

    def get_ground_truth_neighbours(self, node_id: int) -> Dict[int, float]:
        return dict(self.topology_mgr.adjacency.get(node_id, {}))

    def set_noise_factor(self, nf: float):
        self.noise_factor = nf

    def total_node_count(self) -> int:
        return len(self.nodes)
