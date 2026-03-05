"""
Network topology management.

- Single-hop neighbour determination via SINR threshold
- Multi-hop flooding for global topology discovery
- Vectorised NumPy implementation for O(n^2) speedup
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Set, Tuple
from ..channel.interference import calculate_sinr_db
from ..config import (
    SINR_THRESHOLD_DB, NOISE_PSD, SPEED_OF_LIGHT,
    ATG_A, ATG_B, ATG_NLOS_EXCESS_DB,
)

_EPS = 1e-12


class TopologyManager:

    def __init__(self, sinr_threshold_db: float = SINR_THRESHOLD_DB):
        self.sinr_threshold_db = sinr_threshold_db
        self.adjacency: Dict[int, Dict[int, float]] = {}
        self.global_topo: Dict[int, Set[int]] = {}

    # ------------------------------------------------------------------
    # Vectorised neighbour detection (replaces Python double-loop)
    # ------------------------------------------------------------------

    def build_ground_truth_neighbours(
        self,
        nodes: List,
        noise_factor: float = 1.0,
    ) -> Dict[int, Dict[int, float]]:
        n = len(nodes)
        self.adjacency.clear()
        for nd in nodes:
            self.adjacency[nd.id] = {}
        if n < 2:
            return self.adjacency

        ids = np.array([nd.id for nd in nodes])
        pos = np.array([nd.position for nd in nodes])          # (n,2)
        tx_pow = np.array([nd.tx_power for nd in nodes])       # (n,)
        tx_gain = np.array([getattr(nd, 'antenna_gain_linear', 3.16)
                            for nd in nodes])                   # (n,)
        bw = np.array([nd.bandwidth for nd in nodes])          # (n,)
        freq = np.array([nd.freq for nd in nodes])             # (n,)
        h = np.array([nd.antenna_height for nd in nodes])      # (n,)
        comm = np.array([nd.comm_range for nd in nodes])       # (n,)
        ntypes = np.array([nd.node_type for nd in nodes])      # (n,) str

        # --- distance matrix (n,n) ---
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, 1e30)

        # --- comm-range mask: max(range_i, range_j) ---
        range_max = np.maximum(comm[:, None], comm[None, :])
        in_range = dist <= range_max

        # --- vectorised path-loss (n,n) ---
        pl_db = self._vectorised_path_loss(dist, ntypes, freq, h)

        # --- pairwise SINR (dedicated channel, no co-channel interference) ---
        # Ground truth represents physical reachability: whether two nodes
        # CAN communicate on a dedicated channel.  Interference is handled
        # separately by the discovery mechanisms (Disco/ALOHA suffer from it,
        # INDP avoids it via dedicated challenge-response).
        pl_lin = np.power(10.0, pl_db / 10.0)
        bw_min = np.minimum(bw[:, None], bw[None, :])
        noise = NOISE_PSD * (1.0 + noise_factor) * bw_min  # (n,n)

        rx_power = (tx_pow[:, None] * tx_gain[:, None] *
                    tx_gain[None, :] / pl_lin)  # (n,n)

        sinr_lin = rx_power / (noise + _EPS)
        sinr_db = 10.0 * np.log10(np.maximum(sinr_lin, _EPS))

        # Cache precomputed data for INDP vectorisation and P2 batch SINR
        self._cached_rx_power = rx_power
        self._cached_pl_db = pl_db
        self._cached_dist = dist
        self._cached_ids = ids

        # --- build adjacency from mask ---
        # Link (i,j) exists if both directions pass threshold and in range
        valid = (in_range &
                 (sinr_db >= self.sinr_threshold_db) &
                 (sinr_db.T >= self.sinr_threshold_db))
        # Only upper triangle to avoid double-counting
        ii, jj = np.where(np.triu(valid, k=1))

        for idx in range(len(ii)):
            i, j = int(ii[idx]), int(jj[idx])
            ni_id, nj_id = int(ids[i]), int(ids[j])
            self.adjacency[ni_id][nj_id] = float(sinr_db[i, j])
            self.adjacency[nj_id][ni_id] = float(sinr_db[j, i])

        return self.adjacency

    @staticmethod
    def _vectorised_path_loss(dist, ntypes, freq, h):
        """Compute path-loss matrix using vectorised NumPy operations."""
        n = len(ntypes)
        pl = np.full((n, n), 200.0)  # default very high loss

        is_sat = (ntypes == "satellite")
        is_uav = (ntypes == "uav")
        is_surface = ((ntypes == "ship") | (ntypes == "buoy") |
                      (ntypes == "base_station"))

        # --- FSPL for satellite links ---
        sat_mask = is_sat[:, None] | is_sat[None, :]
        if sat_mask.any():
            f_sat = np.where(is_sat[:, None].repeat(n, axis=1),
                             freq[:, None].repeat(n, axis=1),
                             freq[None, :].repeat(n, axis=0))
            d_safe = np.maximum(dist, _EPS)
            fspl = 20.0 * np.log10(4.0 * np.pi * d_safe * f_sat / SPEED_OF_LIGHT)
            pl = np.where(sat_mask, fspl, pl)

        # --- Air-to-ground for UAV links (not satellite) ---
        uav_mask = (~sat_mask) & (is_uav[:, None] | is_uav[None, :])
        if uav_mask.any():
            h_uav = np.where(is_uav[:, None].repeat(n, axis=1),
                             h[:, None].repeat(n, axis=1),
                             h[None, :].repeat(n, axis=0))
            f_uav = np.where(is_uav[:, None].repeat(n, axis=1),
                             freq[:, None].repeat(n, axis=1),
                             freq[None, :].repeat(n, axis=0))
            d_h = np.maximum(dist, _EPS)
            theta_deg = np.degrees(np.arctan2(h_uav, d_h))
            p_los = 1.0 / (1.0 + ATG_A * np.exp(-ATG_B * (theta_deg - ATG_A)))
            d_3d = np.sqrt(d_h ** 2 + h_uav ** 2)
            pl_los = 20.0 * np.log10(
                4.0 * np.pi * d_3d * f_uav / SPEED_OF_LIGHT)
            pl_nlos = pl_los + ATG_NLOS_EXCESS_DB
            pl_a2g = p_los * pl_los + (1.0 - p_los) * pl_nlos
            pl = np.where(uav_mask, pl_a2g, pl)

        # --- Two-ray maritime for surface-surface ---
        surf_mask = (~sat_mask) & (~uav_mask) & is_surface[:, None] & is_surface[None, :]
        if surf_mask.any():
            f_max = np.maximum(freq[:, None], freq[None, :])
            wl = SPEED_OF_LIGHT / f_max
            h_prod = h[:, None] * h[None, :]
            d_c = 4.0 * h_prod / wl
            d_safe = np.maximum(dist, _EPS)
            fspl_surf = 20.0 * np.log10(
                4.0 * np.pi * d_safe * f_max / SPEED_OF_LIGHT)
            two_ray = (40.0 * np.log10(d_safe) -
                       20.0 * np.log10(np.maximum(h_prod, _EPS)))
            pl_surf = np.where(d_safe < d_c, fspl_surf, two_ray)
            pl = np.where(surf_mask, pl_surf, pl)

        return pl

    # ------------------------------------------------------------------
    # Multi-hop flooding for global topology discovery
    # ------------------------------------------------------------------

    def flood_topology(self, nodes: List, max_rounds: int = 0) -> Dict[int, Set[int]]:
        known: Dict[int, Set[int]] = {}
        for n in nodes:
            known[n.id] = {n.id} | set(self.adjacency.get(n.id, {}).keys())

        rnd = 0
        while True:
            rnd += 1
            changed = False
            for n in nodes:
                for nb_id in list(self.adjacency.get(n.id, {}).keys()):
                    before = len(known[n.id])
                    known[n.id] |= known.get(nb_id, set())
                    if len(known[n.id]) > before:
                        changed = True
            if not changed:
                break
            if 0 < max_rounds <= rnd:
                break

        self.global_topo = known
        for n in nodes:
            n.discovered_topology = known.get(n.id, set())
        return known

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_edges(self) -> List[Tuple[int, int, float]]:
        seen = set()
        edges = []
        for a, nb_dict in self.adjacency.items():
            for b, sinr in nb_dict.items():
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    edges.append((a, b, sinr))
        return edges

    def get_node_degree(self, node_id: int) -> int:
        return len(self.adjacency.get(node_id, {}))

    def topology_completeness(self, nodes: List) -> float:
        full = {n.id for n in nodes}
        if len(nodes) == 0:
            return 1.0
        complete = sum(
            1 for n in nodes if n.discovered_topology == full
        )
        return complete / len(nodes)
