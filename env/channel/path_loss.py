"""
Path-loss models based on well-established propagation theory.

References:
  - FSPL: Friis transmission equation
  - Two-ray: maritime / over-sea reflection model
  - A2G : ITU-R air-to-ground LoS probability model
"""

from __future__ import annotations
import numpy as np
from ..config import (
    SPEED_OF_LIGHT,
    ATG_A, ATG_B, ATG_NLOS_EXCESS_DB,
)

_EPS = 1e-12  # avoid log(0)


def fspl_db(d_m: float, f_hz: float) -> float:
    """Free-space path loss (Friis).
    PL(dB) = 20·log10(4π·d·f / c)
    """
    d_m = max(d_m, _EPS)
    return 20.0 * np.log10(4.0 * np.pi * d_m * f_hz / SPEED_OF_LIGHT)


def two_ray_maritime_db(d_m: float, h_t: float, h_r: float,
                        f_hz: float) -> float:
    """Two-ray ground-reflection model for over-sea propagation.

    Near field (d < d_c): uses FSPL.
    Far field  (d >= d_c): PL = 40·log10(d) - 20·log10(h_t·h_r)

    Cross-over distance  d_c = 4·h_t·h_r / λ
    """
    wavelength = SPEED_OF_LIGHT / f_hz
    d_c = 4.0 * h_t * h_r / wavelength
    d_m = max(d_m, _EPS)
    if d_m < d_c:
        return fspl_db(d_m, f_hz)
    return 40.0 * np.log10(d_m) - 20.0 * np.log10(max(h_t * h_r, _EPS))


def air_to_ground_db(d_horizontal: float, h_uav: float,
                     f_hz: float) -> float:
    """Probabilistic air-to-ground channel model.

    P_LoS = 1 / (1 + a·exp(-b·(θ - a)))
    PL = P_LoS · PL_LoS + (1 - P_LoS) · PL_NLoS
    """
    d_horizontal = max(d_horizontal, _EPS)
    theta_deg = np.degrees(np.arctan2(h_uav, d_horizontal))
    p_los = 1.0 / (1.0 + ATG_A * np.exp(-ATG_B * (theta_deg - ATG_A)))
    d_3d = np.sqrt(d_horizontal ** 2 + h_uav ** 2)
    pl_los = fspl_db(d_3d, f_hz)
    pl_nlos = pl_los + ATG_NLOS_EXCESS_DB
    return p_los * pl_los + (1.0 - p_los) * pl_nlos


# ---- dispatcher -----------------------------------------------------------

def _is_satellite(t: str) -> bool:
    return t == "satellite"

def _is_aerial(t: str) -> bool:
    return t == "uav"

def _is_surface(t: str) -> bool:
    return t in ("ship", "buoy", "base_station")


def get_path_loss_db(node_a, node_b) -> float:
    """Select the appropriate path-loss model for a pair of nodes."""
    d = node_a.distance_to(node_b)
    ta, tb = node_a.node_type, node_b.node_type

    # satellite link -> FSPL (Ka-band, long distance)
    if _is_satellite(ta) or _is_satellite(tb):
        f = node_a.freq if _is_satellite(ta) else node_b.freq
        return fspl_db(d, f)

    # UAV involved -> air-to-ground
    if _is_aerial(ta) or _is_aerial(tb):
        uav = node_a if _is_aerial(ta) else node_b
        other = node_b if _is_aerial(ta) else node_a
        d_h = max(d, _EPS)
        return air_to_ground_db(d_h, uav.antenna_height, uav.freq)

    # Both surface (ship / buoy / base-station) -> two-ray maritime
    f = max(node_a.freq, node_b.freq)
    return two_ray_maritime_db(d, node_a.antenna_height,
                                node_b.antenna_height, f)
