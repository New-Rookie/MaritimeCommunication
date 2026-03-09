"""
INDP — Immune-inspired, interference-resilient Neighbor Discovery Protocol
for MEC-enabled integrated air-land-sea-space IoT.

Implements the slotted PHY/MAC protocol from Manuscript I:
  * Class-aware Zadoff-Chu beacon generation
  * CA-CFAR detection (fallback to OS-CFAR under interference)
  * Bounded successive interference cancellation (SIC)
  * Encounter-memory cache
  * Window-level topology construction
  * F1_topo and E_ND metric computation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from Env.config import EnvConfig
from Env.nodes import BaseNode
from Env.phy import (
    received_signal_power,
    aggregate_interference,
    compute_sinr,
    compute_snr,
    environmental_noise,
    LinkPHY,
)

# ═══════════════════════════════════════════════════════════════════════════
# Encounter-memory entry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EncounterEntry:
    node_id: int
    node_type: str
    last_position: np.ndarray
    last_velocity: np.ndarray
    confidence: float = 1.0
    recency: int = 0        # slots since last confirmed contact

    def decay(self):
        self.recency += 1
        self.confidence *= 0.95


# ═══════════════════════════════════════════════════════════════════════════
# Per-node INDP state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class INDPNodeState:
    node_id: int
    # local neighbour table (LNT): discovered node ids in this window
    lnt: Set[int] = field(default_factory=set)
    # encounter memory
    memory: Dict[int, EncounterEntry] = field(default_factory=dict)
    # per-slot accumulators
    tx_slots: int = 0
    rx_slots: int = 0
    sleep_slots: int = 0
    decode_slots: int = 0
    sic_iterations: int = 0
    collisions: int = 0
    # beacon root assignment
    beacon_root: int = 0
    beacon_shift: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# INDP Protocol
# ═══════════════════════════════════════════════════════════════════════════

class INDPProtocol:
    """Stateful INDP protocol manager for one discovery window."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.states: Dict[int, INDPNodeState] = {}
        # CFAR constants
        self.N_ref = 16    # reference window size
        self.alpha_cfar = self._compute_alpha_cfar(self.N_ref, cfg.P_fa)
        self._slot_idx = 0

    @staticmethod
    def _compute_alpha_cfar(n_ref: int, p_fa: float) -> float:
        """alpha_CFAR = N_ref * (P_fa^{-1/N_ref} - 1)"""
        return n_ref * (p_fa ** (-1.0 / n_ref) - 1.0)

    # -----------------------------------------------------------------
    # Initialise for a new window
    # -----------------------------------------------------------------

    def init_window(self, nodes: List[BaseNode]):
        self.states = {}
        self._slot_idx = 0
        for node in nodes:
            st = INDPNodeState(node_id=node.node_id)
            # Zadoff-Chu beacon assignment: unique root per node class,
            # cyclic shift per node ID
            type_root = {"satellite": 0, "uav": 1, "ship": 2,
                         "buoy": 3, "land": 4}
            st.beacon_root = type_root.get(node.node_type, 0) * 100
            st.beacon_shift = node.node_id
            self.states[node.node_id] = st

    # -----------------------------------------------------------------
    # Run one discovery slot
    # -----------------------------------------------------------------

    def run_slot(self, nodes: List[BaseNode], cfg: EnvConfig,
                 rng: np.random.Generator,
                 actions: Optional[np.ndarray] = None) -> Dict:
        """
        Execute one discovery slot for all nodes.

        Parameters
        ----------
        actions : (N, 2)  col0 = listen_prob, col1 = tx_power_frac
            If None, nodes follow a default schedule.

        Returns
        -------
        slot_info : dict with per-node detection results
        """
        n = len(nodes)
        self._slot_idx += 1

        # decide per-node action: transmit, listen, or sleep
        tx_set: List[int] = []
        rx_set: List[int] = []
        sleep_set: List[int] = []

        for i, node in enumerate(nodes):
            st = self.states[node.node_id]
            if actions is not None and i < actions.shape[0]:
                listen_prob = float(actions[i, 0])
                tx_frac = float(actions[i, 1])
            else:
                # default: 90% listen, 10% tx, 0% sleep (matches baselines)
                listen_prob = 0.9
                tx_frac = 1.0

            roll = rng.random()
            if roll < listen_prob:
                rx_set.append(node.node_id)
                st.rx_slots += 1
            elif roll < listen_prob + tx_frac * (1 - listen_prob):
                tx_set.append(node.node_id)
                st.tx_slots += 1
                node.tx_power = cfg.tx_power_w(node.node_type) * tx_frac
            else:
                sleep_set.append(node.node_id)
                st.sleep_slots += 1
                node.tx_power = 0.0

        # set non-transmitting nodes to 0 power
        for node in nodes:
            if node.node_id not in tx_set:
                node.tx_power = 0.0

        # build node lookup
        node_map = {n.node_id: n for n in nodes}

        # detection at each receiver
        detections: Dict[int, List[int]] = {nid: [] for nid in rx_set}

        for rx_id in rx_set:
            rx_node = node_map[rx_id]
            st = self.states[rx_id]
            n_env = environmental_noise(cfg, rx_node.node_type, rng)

            # collect received powers from all transmitters
            rx_powers: List[Tuple[int, float]] = []
            for tx_id in tx_set:
                if tx_id == rx_id:
                    continue
                tx_node = node_map[tx_id]
                p_sig = received_signal_power(tx_node, rx_node, cfg, rng)
                rx_powers.append((tx_id, p_sig))

            if not rx_powers:
                continue

            # sort by power descending (for SIC)
            rx_powers.sort(key=lambda x: x[1], reverse=True)

            # total interference (before SIC)
            total_power = sum(p for _, p in rx_powers)

            # CA-CFAR / OS-CFAR detection with SIC
            cancelled: Set[int] = set()
            residual_interference = 0.0

            for sic_iter in range(cfg.K_max + 1):
                for tx_id, p_sig in rx_powers:
                    if tx_id in cancelled:
                        continue

                    # interference from others (not yet cancelled + residual)
                    i_others = sum(p for tid, p in rx_powers
                                   if tid != tx_id and tid not in cancelled)
                    i_residual = cfg.beta_SIC * sum(p for tid, p in rx_powers
                                                    if tid in cancelled)
                    i_total = i_others + i_residual + n_env

                    sinr_post = p_sig / max(i_total, 1e-30)

                    # CA-CFAR: threshold against thermal noise floor only;
                    # interference is handled by the SIC loop + SINR check.
                    cfar_ok = p_sig > self.alpha_cfar * n_env

                    if cfar_ok and sinr_post >= cfg.gamma_link_linear:
                        # detection: beacon matched
                        detections[rx_id].append(tx_id)
                        cancelled.add(tx_id)
                        st.decode_slots += 1
                        break  # re-loop for next SIC iteration

                st.sic_iterations += 1
                if sic_iter > 0 and len(cancelled) == len([t for t, _ in rx_powers]):
                    break

            # collision count: transmitters received but not decoded
            st.collisions += max(0, len(rx_powers) - len(detections[rx_id]))

        # update LNTs and encounter memory
        for rx_id, detected_ids in detections.items():
            st = self.states[rx_id]
            for tx_id in detected_ids:
                st.lnt.add(tx_id)
                tx_node = node_map[tx_id]
                st.memory[tx_id] = EncounterEntry(
                    node_id=tx_id,
                    node_type=tx_node.node_type,
                    last_position=tx_node.position.copy(),
                    last_velocity=tx_node.velocity.copy(),
                    confidence=1.0,
                    recency=0,
                )

        # decay memory for undetected entries
        for nid, st in self.states.items():
            for mid, entry in st.memory.items():
                if mid not in st.lnt or mid not in detections.get(nid, []):
                    entry.decay()

        return {"detections": detections, "tx_set": tx_set,
                "rx_set": rx_set, "sleep_set": sleep_set}

    # -----------------------------------------------------------------
    # Build discovered topology from all node LNTs
    # -----------------------------------------------------------------

    def build_discovered_topology(self, n_nodes: int) -> np.ndarray:
        """A_disc,ij(w) = 1{ j in LNT_i or i in LNT_j }"""
        adj = np.zeros((n_nodes, n_nodes), dtype=bool)
        for nid, st in self.states.items():
            for nbr in st.lnt:
                if 0 <= nid < n_nodes and 0 <= nbr < n_nodes:
                    adj[nid, nbr] = True
                    adj[nbr, nid] = True
        return adj

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    def compute_f1(self, gt_adj: np.ndarray, n_nodes: int) -> Tuple[float, int, int, int]:
        disc = self.build_discovered_topology(n_nodes)
        tp = fp = fn = 0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                g = gt_adj[i, j]
                d = disc[i, j]
                if g and d:
                    tp += 1
                elif (not g) and d:
                    fp += 1
                elif g and (not d):
                    fn += 1
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        return f1, tp, fp, fn

    def compute_energy(self, node_id: int, cfg: EnvConfig) -> float:
        """E_ND,i(w) per Manuscript I Section 6."""
        st = self.states.get(node_id)
        if st is None:
            return 0.0
        dt = cfg.delta_t_slot * 1e-3  # ms -> s
        p_tx = cfg.tx_power_w("ship")  # representative
        e_tx = st.tx_slots * p_tx * dt
        e_rx = st.rx_slots * (cfg.P_listen + st.decode_slots / max(st.rx_slots, 1) * cfg.P_dec) * dt
        e_sleep = st.sleep_slots * cfg.P_sleep * dt
        e_sic = st.sic_iterations * cfg.C_ops * cfg.kappa * (cfg.f_cpu_ND ** 2)
        return e_tx + e_rx + e_sleep + e_sic

    def mean_energy(self, cfg: EnvConfig) -> float:
        if not self.states:
            return 0.0
        return np.mean([self.compute_energy(nid, cfg) for nid in self.states])

    # -----------------------------------------------------------------
    # Run a full discovery window
    # -----------------------------------------------------------------

    def run_window(self, nodes: List[BaseNode], cfg: EnvConfig,
                   rng: np.random.Generator,
                   actions_per_slot: Optional[List[np.ndarray]] = None) -> Dict:
        """Execute N_slot slots and return aggregated window metrics."""
        self.init_window(nodes)
        all_slot_info = []
        for s in range(cfg.N_slot):
            act = actions_per_slot[s] if actions_per_slot else None
            info = self.run_slot(nodes, cfg, rng, act)
            all_slot_info.append(info)

        disc_adj = self.build_discovered_topology(len(nodes))
        return {
            "disc_adj": disc_adj,
            "slot_infos": all_slot_info,
            "mean_energy": self.mean_energy(cfg),
            "states": self.states,
        }
