"""
Physical-layer computations — unified across all three research chapters.

Implements the thesis-wide locked formulas for:
  P_sig, RSSI, SNR, SINR, aggregate interference,
  communication range, and Shannon service rate.

Includes both scalar (per-link) and vectorized (batch matrix) implementations.
The vectorized compute_all_links_vectorized() produces the same
Dict[Tuple[int,int], LinkPHY] output as the original compute_all_links().
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EnvConfig, _db2lin, _dbm2w
from .channel import (
    compute_path_loss,
    fading_gain,
    environmental_noise,
    doppler_shift,
    C_LIGHT,
    K_BOLTZMANN,
    T0,
    TYPE_ID,
    build_link_class_masks,
    vectorized_path_loss,
    vectorized_fading,
    vectorized_environmental_noise,
    vectorized_doppler,
)
from .nodes import BaseNode


# ═══════════════════════════════════════════════════════════════════════════
# Per-link signal power (scalar — used by protocols)
# ═══════════════════════════════════════════════════════════════════════════

def received_signal_power(tx_node: BaseNode, rx_node: BaseNode,
                          cfg: EnvConfig,
                          rng: np.random.Generator,
                          pl_cache: Optional[Dict] = None) -> float:
    """
    P_sig,ij(t) = P_tx,j * G_tx,j * G_rx,i * 10^(-PL_ij/10) * |g_ij|^2
    Returns power in Watts.
    """
    p_tx = tx_node.tx_power if tx_node.tx_power > 0 else cfg.tx_power_w(tx_node.node_type)
    g_tx, _ = cfg.antenna_gains(tx_node.node_type)
    _, g_rx = cfg.antenna_gains(rx_node.node_type)

    key = (tx_node.node_id, rx_node.node_id)
    if pl_cache is not None and key in pl_cache:
        pl_db = pl_cache[key]
    else:
        pl_db = compute_path_loss(tx_node.node_type, rx_node.node_type,
                                  tx_node.position, rx_node.position, cfg, rng)
        if pl_cache is not None:
            pl_cache[key] = pl_db

    g_fading = fading_gain(tx_node.node_type, rx_node.node_type, cfg, rng)
    p_sig = p_tx * g_tx * g_rx * (10.0 ** (-pl_db / 10.0)) * g_fading
    return max(p_sig, 1e-30)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate interference at a receiver (scalar — used by protocols)
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_interference(rx_node: BaseNode,
                           all_nodes: List[BaseNode],
                           cfg: EnvConfig,
                           rng: np.random.Generator,
                           exclude_ids: Optional[set] = None,
                           pl_cache: Optional[Dict] = None) -> float:
    """
    I_i(t) = sum_{j != i, j active} P_sig,ji(t)
    """
    exclude = exclude_ids or set()
    interference = 0.0
    for node in all_nodes:
        if node.node_id == rx_node.node_id:
            continue
        if node.node_id in exclude:
            continue
        if not node.is_active or node.tx_power <= 0:
            continue
        interference += received_signal_power(node, rx_node, cfg, rng, pl_cache)
    return interference


# ═══════════════════════════════════════════════════════════════════════════
# RSSI / SNR / SINR (scalar)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rssi(p_sig: float, interference: float, n_env: float) -> float:
    return p_sig + interference + n_env


def compute_snr(p_sig: float, n_env: float) -> float:
    return p_sig / max(n_env, 1e-30)


def compute_sinr(p_sig: float, interference: float, n_env: float) -> float:
    return p_sig / max(interference + n_env, 1e-30)


# ═══════════════════════════════════════════════════════════════════════════
# Full per-link PHY snapshot
# ═══════════════════════════════════════════════════════════════════════════

class LinkPHY:
    __slots__ = ("tx_id", "rx_id", "p_sig", "interference", "n_env",
                 "rssi", "snr", "sinr", "pl_db", "doppler", "distance")

    def __init__(self, tx_id: int, rx_id: int, p_sig: float,
                 interference: float, n_env: float,
                 pl_db: float, doppler: float, distance: float):
        self.tx_id = tx_id
        self.rx_id = rx_id
        self.p_sig = p_sig
        self.interference = interference
        self.n_env = n_env
        self.rssi = compute_rssi(p_sig, interference, n_env)
        self.snr = compute_snr(p_sig, n_env)
        self.sinr = compute_sinr(p_sig, interference, n_env)
        self.pl_db = pl_db
        self.doppler = doppler
        self.distance = distance


# ═══════════════════════════════════════════════════════════════════════════
# Original scalar compute_all_links (preserved for reference / fallback)
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_links(nodes: List[BaseNode], cfg: EnvConfig,
                      rng: np.random.Generator) -> Dict[Tuple[int, int], LinkPHY]:
    """Compute PHY quantities for all ordered pairs where the transmitter is active."""
    pl_cache: Dict = {}
    noise_cache: Dict[int, float] = {}
    links: Dict[Tuple[int, int], LinkPHY] = {}

    for rx in nodes:
        if rx.node_id not in noise_cache:
            noise_cache[rx.node_id] = environmental_noise(cfg, rx.node_type, rng)

    for tx in nodes:
        if not tx.is_active:
            continue
        for rx in nodes:
            if tx.node_id == rx.node_id:
                continue
            p_sig = received_signal_power(tx, rx, cfg, rng, pl_cache)
            i_agg = aggregate_interference(rx, nodes, cfg, rng,
                                           exclude_ids={tx.node_id}, pl_cache=pl_cache)
            n_env = noise_cache[rx.node_id]
            dist = float(np.linalg.norm(tx.position - rx.position))
            f_c = cfg.carrier_freq(tx.node_type, rx.node_type)
            fd = doppler_shift(tx.position, rx.position,
                               tx.velocity, rx.velocity, f_c)
            key_pl = (tx.node_id, rx.node_id)
            pl_db = pl_cache.get(key_pl, 0.0)
            links[(tx.node_id, rx.node_id)] = LinkPHY(
                tx.node_id, rx.node_id, p_sig, i_agg, n_env,
                pl_db, fd, dist)
    return links


# ═══════════════════════════════════════════════════════════════════════════
# Antenna gain lookup tables (vectorized helpers)
# ═══════════════════════════════════════════════════════════════════════════

def _build_gain_tables(cfg: EnvConfig):
    """Pre-build linear gain arrays indexed by TYPE_ID for vectorized ops."""
    type_names = ["satellite", "uav", "ship", "buoy", "land"]
    g_tx = np.zeros(5, dtype=np.float64)
    g_rx = np.zeros(5, dtype=np.float64)
    for name in type_names:
        tid = TYPE_ID[name]
        gt, gr = cfg.antenna_gains(name)
        g_tx[tid] = gt
        g_rx[tid] = gr
    return g_tx, g_rx


def _build_txpower_default(cfg: EnvConfig):
    """Default max Tx power per type (used when node.tx_power == 0)."""
    type_names = ["satellite", "uav", "ship", "buoy", "land"]
    p = np.zeros(5, dtype=np.float64)
    for name in type_names:
        p[TYPE_ID[name]] = cfg.tx_power_w(name)
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized compute_all_links — O(N^2) replacement
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_links_vectorized(
    nodes: List[BaseNode],
    cfg: EnvConfig,
    rng: np.random.Generator,
) -> Dict[Tuple[int, int], LinkPHY]:
    """
    Vectorized O(N^2) replacement for compute_all_links.

    Computes the same physical quantities using NumPy matrix operations:
      P_sig[i,j], interference via matrix subtraction, SNR, SINR, Doppler.

    Returns the same Dict[Tuple[int,int], LinkPHY] as the original.
    """
    N = len(nodes)
    if N == 0:
        return {}

    # ── Extract node arrays ───────────────────────────────────────────
    positions = np.stack([n.position for n in nodes])        # (N, 3)
    velocities = np.stack([n.velocity for n in nodes])       # (N, 3)
    type_strs = [n.node_type for n in nodes]
    type_ids = np.array([TYPE_ID[t] for t in type_strs], dtype=np.int32)  # (N,)
    node_ids = np.array([n.node_id for n in nodes], dtype=np.int32)

    p_default = _build_txpower_default(cfg)
    tx_powers = np.array([
        n.tx_power if n.tx_power > 0 else p_default[TYPE_ID[n.node_type]]
        for n in nodes
    ], dtype=np.float64)                                     # (N,)

    active = np.array([n.is_active for n in nodes], dtype=bool)  # (N,)
    has_power = tx_powers > 0
    tx_active = active & has_power                           # (N,)

    # ── Distance matrices ─────────────────────────────────────────────
    diff = positions[:, None, :] - positions[None, :, :]     # (N, N, 3)
    dist_3d = np.linalg.norm(diff, axis=-1)                  # (N, N)
    dist_2d = np.linalg.norm(diff[:, :, :2], axis=-1)        # (N, N)

    # ── Link-class masks ──────────────────────────────────────────────
    mask_sat, mask_uav, mask_sea, mask_terr = build_link_class_masks(type_ids)

    # ── Path loss (N, N) ──────────────────────────────────────────────
    PL = vectorized_path_loss(
        dist_3d, dist_2d, positions, type_ids,
        mask_sat, mask_uav, mask_sea, mask_terr,
        cfg, rng)

    # ── Fading (N, N) ────────────────────────────────────────────────
    mask_rician = mask_sat | mask_uav
    mask_rayleigh = mask_sea | mask_terr
    fading_matrix = vectorized_fading(N, mask_rician, mask_rayleigh, cfg, rng)

    # ── Antenna gains ─────────────────────────────────────────────────
    g_tx_table, g_rx_table = _build_gain_tables(cfg)
    G_tx = g_tx_table[type_ids]   # (N,)
    G_rx = g_rx_table[type_ids]   # (N,)

    # ── P_sig matrix: P_sig[i,j] = power from tx=i arriving at rx=j ──
    PL_linear = np.power(10.0, -PL / 10.0)
    P_sig = (tx_powers[:, None] * G_tx[:, None] * G_rx[None, :] *
             PL_linear * fading_matrix)
    P_sig = np.maximum(P_sig, 1e-30)
    np.fill_diagonal(P_sig, 0.0)
    P_sig *= tx_active[:, None]   # zero out inactive transmitters

    # ── Interference via matrix subtraction (the key O(N^2) trick) ────
    # I[tx, rx] = total power at rx from all active transmitters minus
    #             the desired signal P_sig[tx, rx].
    total_at_rx = P_sig.sum(axis=0)                          # (N,)
    I_matrix = total_at_rx[None, :] - P_sig                  # (N, N)

    # ── Noise per receiver ────────────────────────────────────────────
    N_env = vectorized_environmental_noise(cfg, type_ids, rng)  # (N,)

    # ── SNR / SINR / RSSI ────────────────────────────────────────────
    N_env_row = N_env[None, :]                               # (1, N) for broadcasting
    SNR = P_sig / np.maximum(N_env_row, 1e-30)
    SINR = P_sig / np.maximum(I_matrix + N_env_row, 1e-30)
    RSSI = P_sig + I_matrix + N_env_row

    # ── Carrier frequency matrix for Doppler ──────────────────────────
    is_sat = (type_ids == TYPE_ID["satellite"])
    f_c_vec = np.where(is_sat, cfg.f_c_sat, cfg.f_c_local)  # (N,)
    # link uses f_c_sat if either endpoint is satellite
    f_c_matrix = np.where(
        is_sat[:, None] | is_sat[None, :],
        cfg.f_c_sat, cfg.f_c_local)                          # (N, N)

    # ── Doppler ───────────────────────────────────────────────────────
    doppler_matrix = vectorized_doppler(positions, velocities, f_c_matrix)

    # ── Build output dictionary (same API as compute_all_links) ───────
    links: Dict[Tuple[int, int], LinkPHY] = {}
    tx_indices = np.where(tx_active)[0]

    for ti in tx_indices:
        tx_id = int(node_ids[ti])
        for rj in range(N):
            if ti == rj:
                continue
            rx_id = int(node_ids[rj])
            links[(tx_id, rx_id)] = LinkPHY(
                tx_id, rx_id,
                float(P_sig[ti, rj]),
                float(I_matrix[ti, rj]),
                float(N_env[rj]),
                float(PL[ti, rj]),
                float(doppler_matrix[ti, rj]),
                float(dist_3d[ti, rj]),
            )

    return links


# ═══════════════════════════════════════════════════════════════════════════
# Communication range
# ═══════════════════════════════════════════════════════════════════════════

def communication_range_estimate(type_i: str, type_j: str,
                                 cfg: EnvConfig) -> float:
    """
    Rough analytic estimate of R_comm,ij from the unified threshold:
      R_comm = sup{d : E[SINR(d)] >= gamma_link}
    Uses free-space + average fading to give a ballpark range.
    """
    p_tx = cfg.tx_power_w(type_j)
    g_tx, _ = cfg.antenna_gains(type_j)
    _, g_rx = cfg.antenna_gains(type_i)
    f_c = cfg.carrier_freq(type_i, type_j)
    n_env = K_BOLTZMANN * cfg.B_meas * (T0 * cfg.F_rx + 290.0) * cfg.eta_N
    gamma_lin = cfg.gamma_link_linear

    eirp = p_tx * g_tx * g_rx
    wavelength = C_LIGHT / f_c
    d_max = (wavelength / (4 * math.pi)) * math.sqrt(eirp / (gamma_lin * n_env))
    return max(d_max, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# Shannon service rate
# ═══════════════════════════════════════════════════════════════════════════

def shannon_rate(bandwidth: float, sinr: float) -> float:
    """R_ij(t) = B * log2(1 + SINR)"""
    if sinr <= 0:
        return 0.0
    return bandwidth * math.log2(1.0 + sinr)


K_BOLTZMANN_REF = K_BOLTZMANN
