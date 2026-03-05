"""
Disco – deterministic neighbour discovery using prime-pair scheduling.

Each node picks two primes (p1, p2).  It transmits a beacon in every slot
that is a multiple of p1 or p2.  Two nodes discover each other when their
transmit slots overlap AND SINR (with slot-specific interference from all
other concurrent transmitters) exceeds threshold.

Reference: Dutta & Culler, SenSys 2008.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from env.channel.path_loss import get_path_loss_db
from env.config import NOISE_PSD, SINR_THRESHOLD_DB

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


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


class DiscoMechanism:

    def __init__(self, slot_duration: float = 0.01):
        self.slot_duration = slot_duration
        self._primes: Dict[int, Tuple[int, int]] = {}

    def reset(self):
        self._primes.clear()

    def _assign_primes(self, node_id):
        if node_id not in self._primes:
            p1, p2 = tuple(np.random.choice(_SMALL_PRIMES, size=2, replace=False))
            self._primes[node_id] = (int(p1), int(p2))
        return self._primes[node_id]

    def discover(
        self, nodes, noise_factor=1.0, num_slots=200,
    ) -> Tuple[Dict[int, Set[int]], float]:
        discovered: Dict[int, Set[int]] = {n.id: set() for n in nodes}
        total_energy = 0.0

        schedules = {}
        for n in nodes:
            p1, p2 = self._assign_primes(n.id)
            active = {s for s in range(num_slots) if s % p1 == 0 or s % p2 == 0}
            schedules[n.id] = active

        for slot in range(num_slots):
            transmitters = [n for n in nodes if slot in schedules[n.id]]
            for tx in transmitters:
                total_energy += tx.tx_power * self.slot_duration
                for rx in nodes:
                    if rx.id == tx.id or rx.id in discovered[tx.id]:
                        continue
                    d = tx.distance_to(rx)
                    if d > max(tx.comm_range, rx.comm_range):
                        continue
                    sinr_db = _slot_sinr(tx, rx, transmitters, noise_factor)
                    if sinr_db >= SINR_THRESHOLD_DB:
                        if slot in schedules[rx.id]:
                            discovered[tx.id].add(rx.id)
                            discovered[rx.id].add(tx.id)

        return discovered, total_energy
