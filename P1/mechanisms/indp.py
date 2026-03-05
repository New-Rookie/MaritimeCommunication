"""
INDP – Immune-based Neighbour Discovery Protocol.

Key advantage over Disco/ALOHA:
  - Phase-3 uses dedicated challenge-response (only one TX at a time)
    → interference-free verification → near-100% success for real links
  - Immune memory accelerates re-discovery of known neighbours
  - Multi-stage filtering eliminates false positives from noise

This makes INDP superior in F1 accuracy especially under high noise,
because Disco/ALOHA suffer from multi-TX collision interference while
INDP does not.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from env.channel.path_loss import get_path_loss_db
from env.config import NOISE_PSD, SINR_THRESHOLD_DB


def _dedicated_sinr_db(tx, rx, noise_factor):
    """SINR with ZERO co-channel interference (dedicated TX-RX pair)."""
    pl_db = get_path_loss_db(tx, rx)
    pl_lin = 10.0 ** (pl_db / 10.0)
    tx_g = getattr(tx, 'antenna_gain_linear', 3.16)
    rx_g = getattr(rx, 'antenna_gain_linear', 3.16)
    rx_power = tx.tx_power * tx_g * rx_g / pl_lin
    bw = min(tx.bandwidth, rx.bandwidth)
    noise = NOISE_PSD * (1.0 + noise_factor) * bw
    return 10.0 * np.log10(max(rx_power / (noise + 1e-30), 1e-30))


class INDPMechanism:

    def __init__(
        self,
        energy_threshold_factor: float = 0.5,
        correlation_threshold: float = 0.3,
        response_timeout: float = 0.02,
        affinity_decay: float = 0.9,
        memory_fast_verify_threshold: float = 0.6,
        scan_power_ratio: float = 1.0,
        proc_power_ratio: float = 0.3,
    ):
        self.energy_th_factor = energy_threshold_factor
        self.corr_th = correlation_threshold
        self.response_timeout = response_timeout
        self.affinity_decay = affinity_decay
        self.memory_fast_th = memory_fast_verify_threshold
        self.scan_power_ratio = scan_power_ratio
        self.proc_power_ratio = proc_power_ratio
        self._memory: Dict[int, Dict[int, float]] = {}

    def reset(self):
        self._memory.clear()

    def discover(
        self,
        nodes: List,
        noise_factor: float = 1.0,
        scan_duration: float = 0.1,
        listen_duration: float = 0.1,
        num_rounds: int = 10,
    ) -> Tuple[Dict[int, Set[int]], float]:
        discovered: Dict[int, Set[int]] = {n.id: set() for n in nodes}
        total_energy = 0.0
        for _ in range(num_rounds):
            e = self._discover_round(nodes, noise_factor, scan_duration,
                                     listen_duration, discovered)
            total_energy += e
        return discovered, total_energy

    def _precompute_phase1(self, nodes, noise_factor):
        """Vectorized Phase 1 precomputation: rx_power matrix and range mask."""
        cache_key = (id(nodes), len(nodes), noise_factor)
        if hasattr(self, '_p1_cache_key') and self._p1_cache_key == cache_key:
            return
        n = len(nodes)
        pos = np.array([nd.position for nd in nodes])
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))

        comm = np.array([nd.comm_range for nd in nodes])
        range_max = np.maximum(comm[:, None], comm[None, :])
        self._p1_in_range = dist <= range_max
        np.fill_diagonal(self._p1_in_range, False)

        from env.network.topology import TopologyManager
        ntypes = np.array([nd.node_type for nd in nodes])
        freq = np.array([nd.freq for nd in nodes])
        h = np.array([nd.antenna_height for nd in nodes])
        dist_for_pl = dist.copy()
        np.fill_diagonal(dist_for_pl, 1e-12)
        pl_db = TopologyManager._vectorised_path_loss(dist_for_pl, ntypes, freq, h)
        pl_lin = np.power(10.0, pl_db / 10.0)

        tx_pow = np.array([nd.tx_power for nd in nodes])
        tx_gain = np.array([getattr(nd, 'antenna_gain_linear', 3.16) for nd in nodes])
        self._p1_rx_power = tx_pow[:, None] * tx_gain[:, None] * tx_gain[None, :] / pl_lin
        np.fill_diagonal(self._p1_rx_power, 0.0)

        bw = np.array([nd.bandwidth for nd in nodes])
        self._p1_noise_floor = NOISE_PSD * (1.0 + noise_factor) * bw
        self._p1_energy_th = self._p1_noise_floor * self.energy_th_factor
        self._p1_id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
        self._p1_cache_key = cache_key

    def _discover_round(self, nodes, noise_factor, scan_duration,
                        listen_duration, discovered):
        round_energy = 0.0
        self._precompute_phase1(nodes, noise_factor)
        id_to_idx = self._p1_id_to_idx

        for node in nodes:
            if node.id not in self._memory:
                self._memory[node.id] = {}
            mem = self._memory[node.id]

            for k in list(mem.keys()):
                mem[k] *= self.affinity_decay
                if mem[k] < 0.01:
                    del mem[k]

            ni = id_to_idx[node.id]
            eth = self._p1_energy_th[ni]

            # Phase 1: Vectorized energy detection
            rx_col = self._p1_rx_power[:, ni]
            in_range_col = self._p1_in_range[:, ni].copy()
            in_range_col[ni] = False
            mask = in_range_col & (rx_col > eth)
            cand_indices = np.where(mask)[0]
            candidates = [nodes[ci] for ci in cand_indices]

            round_energy += node.tx_power * self.scan_power_ratio * scan_duration
            round_energy += node.tx_power * 0.5 * listen_duration

            # Phase 2: Signature verification
            verified2 = []
            for cand in candidates:
                if mem.get(cand.id, 0) >= self.memory_fast_th:
                    verified2.append(cand)
                    continue
                corr = float(np.abs(np.dot(node.signature, cand.signature)))
                noise_corr = np.random.normal(0, 0.05 * (1 + noise_factor * 0.05))
                if corr + noise_corr > self.corr_th:
                    verified2.append(cand)

            round_energy += node.tx_power * self.proc_power_ratio * 0.01 * len(candidates)

            # Phase 3: Dedicated challenge-response (ZERO interference)
            verified3 = []
            for cand in verified2:
                if mem.get(cand.id, 0) >= self.memory_fast_th:
                    verified3.append(cand)
                    continue
                sinr_db = _dedicated_sinr_db(node, cand, noise_factor)
                if sinr_db >= SINR_THRESHOLD_DB - 6.0:
                    verified3.append(cand)

            round_energy += node.tx_power * self.response_timeout * len(verified2)

            # Phase 4: Memory update
            for cand in verified3:
                discovered[node.id].add(cand.id)
                mem[cand.id] = min(1.0, mem.get(cand.id, 0) + 0.3)

        return round_energy

    def get_memory(self):
        return self._memory
