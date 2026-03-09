"""
Slotted ALOHA — simplest baseline for neighbour discovery.

Each node transmits a beacon in each slot with probability p_aloha.
No SIC, no CFAR, no memory — purely probabilistic access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from Env.config import EnvConfig
from Env.nodes import BaseNode
from Env.phy import received_signal_power, environmental_noise


@dataclass
class ALOHANodeState:
    node_id: int
    lnt: Set[int] = field(default_factory=set)
    tx_slots: int = 0
    rx_slots: int = 0
    collisions: int = 0


class ALOHAProtocol:
    """Pure slotted ALOHA neighbour discovery."""

    def __init__(self, cfg: EnvConfig, p_aloha: float = 0.1):
        self.cfg = cfg
        self.p_aloha = p_aloha
        self.states: Dict[int, ALOHANodeState] = {}

    def init_window(self, nodes: List[BaseNode]):
        self.states = {}
        for node in nodes:
            self.states[node.node_id] = ALOHANodeState(node_id=node.node_id)

    def run_slot(self, nodes: List[BaseNode], cfg: EnvConfig,
                 rng: np.random.Generator,
                 actions: Optional[np.ndarray] = None) -> Dict:
        node_map = {n.node_id: n for n in nodes}
        tx_set: List[int] = []
        rx_set: List[int] = []

        for node in nodes:
            st = self.states[node.node_id]
            if rng.random() < self.p_aloha:
                tx_set.append(node.node_id)
                st.tx_slots += 1
                node.tx_power = cfg.tx_power_w(node.node_type)
            else:
                rx_set.append(node.node_id)
                st.rx_slots += 1
                node.tx_power = 0.0

        detections: Dict[int, List[int]] = {nid: [] for nid in rx_set}

        for rx_id in rx_set:
            rx_node = node_map[rx_id]
            n_env = environmental_noise(cfg, rx_node.node_type, rng)

            powers = []
            for tx_id in tx_set:
                if tx_id == rx_id:
                    continue
                p_sig = received_signal_power(node_map[tx_id], rx_node, cfg, rng)
                powers.append((tx_id, p_sig))

            if not powers:
                continue

            # ALOHA: success only if exactly one transmitter is "strong enough"
            # (capture effect: strongest signal must exceed threshold)
            powers.sort(key=lambda x: x[1], reverse=True)
            strongest_id, strongest_p = powers[0]
            i_rest = sum(p for _, p in powers[1:]) + n_env
            sinr = strongest_p / max(i_rest, 1e-30)

            if sinr >= cfg.gamma_link_linear:
                detections[rx_id].append(strongest_id)
            else:
                self.states[rx_id].collisions += 1

        for rx_id, dets in detections.items():
            for tx_id in dets:
                self.states[rx_id].lnt.add(tx_id)

        return {"detections": detections, "tx_set": tx_set, "rx_set": rx_set}

    def build_discovered_topology(self, n_nodes: int) -> np.ndarray:
        adj = np.zeros((n_nodes, n_nodes), dtype=bool)
        for nid, st in self.states.items():
            for nbr in st.lnt:
                if 0 <= nid < n_nodes and 0 <= nbr < n_nodes:
                    adj[nid, nbr] = True
                    adj[nbr, nid] = True
        return adj

    def compute_f1(self, gt_adj: np.ndarray, n_nodes: int) -> Tuple[float, int, int, int]:
        disc = self.build_discovered_topology(n_nodes)
        tp = fp = fn = 0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                g = gt_adj[i, j]
                d = disc[i, j]
                tp += int(g and d)
                fp += int((not g) and d)
                fn += int(g and (not d))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        return f1, tp, fp, fn

    def compute_energy(self, node_id: int, cfg: EnvConfig) -> float:
        st = self.states.get(node_id)
        if st is None:
            return 0.0
        dt = cfg.delta_t_slot * 1e-3
        e_tx = st.tx_slots * cfg.tx_power_w("ship") * dt
        e_rx = st.rx_slots * cfg.P_listen * dt
        return e_tx + e_rx

    def mean_energy(self, cfg: EnvConfig) -> float:
        if not self.states:
            return 0.0
        return float(np.mean([self.compute_energy(nid, cfg) for nid in self.states]))

    def run_window(self, nodes: List[BaseNode], cfg: EnvConfig,
                   rng: np.random.Generator,
                   actions_per_slot=None) -> Dict:
        self.init_window(nodes)
        for s in range(cfg.N_slot):
            self.run_slot(nodes, cfg, rng)
        return {
            "disc_adj": self.build_discovered_topology(len(nodes)),
            "mean_energy": self.mean_energy(cfg),
        }
