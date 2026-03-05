"""
ALOHA-based random-access neighbour discovery.

Each node broadcasts a beacon with probability p in every slot.
Multi-TX interference is severe: ALL simultaneous transmitters contribute
interference → accuracy degrades with more nodes and higher noise.
No verification → false positives from noise spikes.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from env.channel.path_loss import get_path_loss_db
from env.config import NOISE_PSD, SINR_THRESHOLD_DB


def _slot_sinr(tx, rx, active_transmitters, noise_factor):
    pl_db = get_path_loss_db(tx, rx)
    pl_lin = 10.0 ** (pl_db / 10.0)
    tx_g = getattr(tx, 'antenna_gain_linear', 3.16)
    rx_g = getattr(rx, 'antenna_gain_linear', 3.16)
    rx_power = tx.tx_power * tx_g * rx_g / pl_lin

    bw = min(tx.bandwidth, rx.bandwidth)
    noise = NOISE_PSD * (1.0 + noise_factor) * bw

    interference = 0.0
    for n in active_transmitters:
        if n.id == tx.id or n.id == rx.id:
            continue
        pj = get_path_loss_db(n, rx)
        pj_lin = 10.0 ** (pj / 10.0)
        jg = getattr(n, 'antenna_gain_linear', 3.16)
        interference += n.tx_power * jg * rx_g / pj_lin

    return 10.0 * np.log10(rx_power / (noise + interference + 1e-30))


class ALOHAMechanism:

    def __init__(self, tx_prob: float = 0.1, slot_duration: float = 0.01):
        self.tx_prob = tx_prob
        self.slot_duration = slot_duration

    def reset(self):
        pass

    def discover(
        self, nodes, noise_factor=1.0, num_slots=200,
    ) -> Tuple[Dict[int, Set[int]], float]:
        discovered: Dict[int, Set[int]] = {n.id: set() for n in nodes}
        total_energy = 0.0

        for slot in range(num_slots):
            transmitters = [n for n in nodes if np.random.random() < self.tx_prob]
            for tx in transmitters:
                total_energy += tx.tx_power * self.slot_duration
                for rx in nodes:
                    if rx.id == tx.id or rx.id in discovered[tx.id]:
                        continue
                    d = tx.distance_to(rx)
                    if d > max(tx.comm_range, rx.comm_range):
                        continue
                    sinr_db = _slot_sinr(tx, rx, transmitters, noise_factor)
                    # Softer threshold: ALOHA accepts more easily → false positives
                    if sinr_db >= SINR_THRESHOLD_DB - 2.0:
                        discovered[tx.id].add(rx.id)
                        discovered[rx.id].add(tx.id)

        # False positives from noise spikes (no verification mechanism)
        # Higher noise → more phantom signals mistaken for real neighbours
        for n in nodes:
            fp_rate = 0.5 + noise_factor * 0.8 + len(nodes) * 0.02
            n_false = np.random.poisson(fp_rate)
            other_ids = [o.id for o in nodes if o.id != n.id]
            if n_false > 0 and len(other_ids) > 0:
                fps = np.random.choice(
                    other_ids, size=min(n_false, len(other_ids)), replace=False)
                for fp in fps:
                    discovered[n.id].add(int(fp))

        return discovered, total_energy
