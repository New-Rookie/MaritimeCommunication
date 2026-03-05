"""Base class shared by every node type in the SAGIN simulation."""

from __future__ import annotations
import numpy as np
from typing import Dict, Set, Optional
from ..config import NodeConfig


class BaseNode:
    _id_counter: int = 0

    def __init__(self, node_type: str, config: NodeConfig,
                 position: Optional[np.ndarray] = None):
        BaseNode._id_counter += 1
        self.id: int = BaseNode._id_counter
        self.node_type: str = node_type

        # --- static resource caps ---
        self.bandwidth: float = config.bandwidth_hz
        self.compute: float = config.compute_flops
        self.storage: float = config.storage_bytes

        # --- runtime available resources ---
        self.avail_bw: float = config.bandwidth_hz
        self.avail_comp: float = config.compute_flops
        self.avail_stor: float = config.storage_bytes

        # --- communication ---
        self.comm_range: float = config.comm_range_m
        self.tx_power: float = config.tx_power_w
        self.freq: float = config.freq_hz
        self.antenna_height: float = config.antenna_height_m
        self.antenna_gain_dbi: float = config.antenna_gain_dbi
        self.antenna_gain_linear: float = 10 ** (config.antenna_gain_dbi / 10.0)

        # --- mobility ---
        self.is_static: bool = config.is_static
        self.speed_range: tuple = config.speed_range_ms
        self.position: np.ndarray = (
            position.copy() if position is not None else np.zeros(2)
        )
        self.velocity: np.ndarray = np.zeros(2)

        # --- MEC role ---
        self.is_mec: bool = config.is_mec

        # --- neighbour / topology state ---
        self.neighbor_table: Dict[int, float] = {}   # node_id -> SINR (linear)
        self.discovered_topology: Set[int] = set()

        # --- energy tracking ---
        self.energy_consumed: float = 0.0

        # --- unique signature for INDP (immune marker) ---
        # All legitimate nodes share a common "network key" component
        # (analogous to MHC self-markers) plus a unique component.
        if not hasattr(BaseNode, '_network_key'):
            BaseNode._network_key = np.random.randn(32)
            BaseNode._network_key /= np.linalg.norm(BaseNode._network_key)
        unique = np.random.randn(32) * 0.3
        self.signature = BaseNode._network_key + unique
        self.signature /= np.linalg.norm(self.signature)

    # ---- helpers ----------------------------------------------------------

    def distance_to(self, other: BaseNode) -> float:
        return float(np.linalg.norm(self.position - other.position))

    def consume_resource(self, bw: float, comp: float, stor: float):
        self.avail_bw = max(0.0, self.avail_bw - bw)
        self.avail_comp = max(0.0, self.avail_comp - comp)
        self.avail_stor = max(0.0, self.avail_stor - stor)

    def restore_resource(self, bw: float, comp: float, stor: float):
        self.avail_bw = min(self.bandwidth, self.avail_bw + bw)
        self.avail_comp = min(self.compute, self.avail_comp + comp)
        self.avail_stor = min(self.storage, self.avail_stor + stor)

    def reset_resources(self):
        self.avail_bw = self.bandwidth
        self.avail_comp = self.compute
        self.avail_stor = self.storage

    def reset(self):
        self.reset_resources()
        self.neighbor_table.clear()
        self.discovered_topology.clear()
        self.energy_consumed = 0.0

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0

    def __repr__(self):
        return (f"{self.node_type}(id={self.id}, "
                f"pos=[{self.position[0]:.0f},{self.position[1]:.0f}])")
