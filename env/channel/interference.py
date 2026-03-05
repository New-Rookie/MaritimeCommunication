"""
SINR calculation with multi-node interference.

SINR_i = P_tx_s · G · PL(d_{s,i})^{-1}
         / (N_env · B  +  Σ_{j≠s} P_tx_j · G · PL(d_{j,i})^{-1})

where  N_env = N_0 · (1 + noise_factor)

Provides both per-pair and batch (vectorised) versions.
"""

from __future__ import annotations
import numpy as np
from typing import List
from .path_loss import get_path_loss_db
from ..config import NOISE_PSD, DEFAULT_NOISE_FACTOR

_EPS = 1e-30


def _pl_linear(pl_db: float) -> float:
    return 10.0 ** (pl_db / 10.0)


def _same_band(f1: float, f2: float, tolerance: float = 0.5) -> bool:
    ratio = max(f1, f2) / max(min(f1, f2), 1.0)
    return ratio < (1.0 + tolerance)


# ------------------------------------------------------------------
# Single-pair calculation (kept for backward compatibility)
# ------------------------------------------------------------------

def calculate_sinr_linear(
    tx_node,
    rx_node,
    all_nodes: List,
    noise_factor: float = DEFAULT_NOISE_FACTOR,
) -> float:
    pl_db = get_path_loss_db(tx_node, rx_node)
    tx_gain = getattr(tx_node, 'antenna_gain_linear', 3.16)
    rx_gain = getattr(rx_node, 'antenna_gain_linear', 3.16)
    rx_power = tx_node.tx_power * tx_gain * rx_gain / _pl_linear(pl_db)

    bw = min(tx_node.bandwidth, rx_node.bandwidth)
    noise_power = NOISE_PSD * (1.0 + noise_factor) * bw

    link_freq = min(tx_node.freq, rx_node.freq)

    interference = 0.0
    for node in all_nodes:
        if node.id == tx_node.id or node.id == rx_node.id:
            continue
        if not _same_band(node.freq, link_freq):
            continue
        pl_j = get_path_loss_db(node, rx_node)
        j_gain = getattr(node, 'antenna_gain_linear', 3.16)
        interference += node.tx_power * j_gain * rx_gain / _pl_linear(pl_j)

    sinr = rx_power / (noise_power + interference + _EPS)
    return sinr


def calculate_sinr_db(
    tx_node,
    rx_node,
    all_nodes: List,
    noise_factor: float = DEFAULT_NOISE_FACTOR,
) -> float:
    sinr_lin = calculate_sinr_linear(tx_node, rx_node, all_nodes, noise_factor)
    return 10.0 * np.log10(max(sinr_lin, _EPS))


# ------------------------------------------------------------------
# Batch (vectorised) SINR for multiple (tx, rx) pairs at once
# ------------------------------------------------------------------

class SINRBatchCalculator:
    """Pre-computes arrays from all_nodes once, then evaluates SINR
    for arbitrary (tx_idx, rx_idx) pairs in vectorised NumPy.

    Usage:
        calc = SINRBatchCalculator(all_nodes, noise_factor)
        sinr_array = calc.sinr_linear(tx_indices, rx_indices)
    """

    def __init__(self, all_nodes: List, noise_factor: float = DEFAULT_NOISE_FACTOR):
        self.n = len(all_nodes)
        self.noise_factor = noise_factor
        self._id_to_idx = {nd.id: i for i, nd in enumerate(all_nodes)}

        pos = np.array([nd.position for nd in all_nodes])
        self._tx_pow = np.array([nd.tx_power for nd in all_nodes])
        self._gain = np.array([getattr(nd, 'antenna_gain_linear', 3.16)
                               for nd in all_nodes])
        self._bw = np.array([nd.bandwidth for nd in all_nodes])
        self._freq = np.array([nd.freq for nd in all_nodes])

        # Pre-compute full path-loss matrix (n,n) using the topology helper
        from ..network.topology import TopologyManager
        ntypes = np.array([nd.node_type for nd in all_nodes])
        h = np.array([nd.antenna_height for nd in all_nodes])
        diff = pos[:, None, :] - pos[None, :, :]
        self._dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(self._dist, _EPS)
        self._pl_db = TopologyManager._vectorised_path_loss(
            self._dist, ntypes, self._freq, h)
        self._pl_lin = np.power(10.0, self._pl_db / 10.0)

        # Pre-compute received power from every node j at every node k: (n,n)
        # rx_power[j,k] = P_j * G_j * G_k / PL(j,k)
        self._rx_pow_matrix = (
            self._tx_pow[:, None] * self._gain[:, None] *
            self._gain[None, :] / self._pl_lin
        )

        # Frequency band mask for co-channel interference
        f_ratio = np.maximum(self._freq[:, None], self._freq[None, :]) / \
                  np.maximum(np.minimum(self._freq[:, None], self._freq[None, :]), 1.0)
        self._cofreq = f_ratio < 1.5

    def idx_of(self, node_id: int) -> int:
        return self._id_to_idx.get(node_id, -1)

    def sinr_linear_pair(self, tx_idx: int, rx_idx: int) -> float:
        """SINR for a single (tx, rx) pair with full interference."""
        if tx_idx < 0 or rx_idx < 0 or tx_idx >= self.n or rx_idx >= self.n:
            return 1e-10
        rx_power = self._rx_pow_matrix[tx_idx, rx_idx]
        bw = min(self._bw[tx_idx], self._bw[rx_idx])
        noise = NOISE_PSD * (1.0 + self.noise_factor) * bw

        # Interference from all co-frequency nodes except tx and rx
        intf_mask = self._cofreq[:, rx_idx].copy()
        intf_mask[tx_idx] = False
        intf_mask[rx_idx] = False
        interference = self._rx_pow_matrix[intf_mask, rx_idx].sum()

        return float(rx_power / (noise + interference + _EPS))

    def sinr_linear_for_rx(self, tx_idx: int, rx_indices: np.ndarray) -> np.ndarray:
        """SINR from one TX to multiple RX nodes."""
        k = len(rx_indices)
        rx_power = self._rx_pow_matrix[tx_idx, rx_indices]  # (k,)
        bw = np.minimum(self._bw[tx_idx], self._bw[rx_indices])
        noise = NOISE_PSD * (1.0 + self.noise_factor) * bw  # (k,)

        link_freq = np.minimum(self._freq[tx_idx], self._freq[rx_indices])
        # For each rx, sum interference from all co-freq nodes excluding tx,rx
        interference = np.zeros(k)
        for ci in range(k):
            ri = rx_indices[ci]
            intf_mask = self._cofreq[:, ri].copy()
            intf_mask[tx_idx] = False
            intf_mask[ri] = False
            interference[ci] = self._rx_pow_matrix[intf_mask, ri].sum()

        return rx_power / (noise + interference + _EPS)

    def sinr_linear_all_neighbours(self, tx_id: int, nb_ids: List[int],
                                    max_k: int = 15) -> np.ndarray:
        """Convenience: SINR from tx_id to up to max_k neighbours.
        Returns array of length max_k (zero-padded).
        """
        vals = np.zeros(max_k, dtype=np.float32)
        if tx_id not in self._id_to_idx:
            return vals
        ti = self._id_to_idx[tx_id]
        nb_idx = []
        for nid in nb_ids[:max_k]:
            if nid in self._id_to_idx:
                nb_idx.append(self._id_to_idx[nid])
        if not nb_idx:
            return vals
        nb_arr = np.array(nb_idx)
        sinr = self.sinr_linear_for_rx(ti, nb_arr)
        vals[:len(sinr)] = sinr
        return vals
