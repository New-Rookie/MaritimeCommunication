"""
Global configuration for the Space-Air-Ground-Sea integrated ocean IoT MEC simulation.
All physical parameters are based on real-world specifications.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
BOLTZMANN_K = 1.38e-23          # Boltzmann constant (J/K)
TEMPERATURE = 290.0             # Reference temperature (K)
NOISE_PSD = BOLTZMANN_K * TEMPERATURE  # Thermal noise PSD ≈ 4.0e-21 W/Hz
SPEED_OF_LIGHT = 3e8            # m/s
EARTH_RADIUS = 6.371e6          # m

# ---------------------------------------------------------------------------
# Simulation area
# ---------------------------------------------------------------------------
MAP_WIDTH = 100_000.0           # 100 km in metres
MAP_HEIGHT = 100_000.0
COASTLINE_X = 5_000.0           # coastline 5 km from left edge

DT = 1.0                        # simulation time-step (seconds)
DEFAULT_EPISODE_LENGTH = 1000   # steps per episode

# ---------------------------------------------------------------------------
# Frequency bands (Hz)
# ---------------------------------------------------------------------------
FREQ_KA = 26.5e9                # Ka-band for LEO satellite
FREQ_SUB6 = 2.4e9               # Sub-6 GHz for UAV / ship
FREQ_NB_IOT = 900e6             # NB-IoT for buoys
FREQ_CBAND = 3.5e9              # C-band for land base-station

# ---------------------------------------------------------------------------
# SINR threshold for neighbour determination
# ---------------------------------------------------------------------------
SINR_THRESHOLD_DB = 3.0         # dB – minimum SINR for single-hop link

# ---------------------------------------------------------------------------
# Antenna & propagation helpers
# ---------------------------------------------------------------------------
ANTENNA_GAIN_DBI = 5.0
ANTENNA_GAIN_LINEAR = 10 ** (ANTENNA_GAIN_DBI / 10.0)

# Air-to-Ground LoS probability params (open ocean, ITU-R)
ATG_A = 11.95
ATG_B = 0.14
ATG_NLOS_EXCESS_DB = 20.0

# ---------------------------------------------------------------------------
# Default noise factor (tuneable for experiments)
# ---------------------------------------------------------------------------
DEFAULT_NOISE_FACTOR = 1.0

# ---------------------------------------------------------------------------
# Node configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeConfig:
    count: int
    bandwidth_hz: float
    compute_flops: float
    storage_bytes: float
    comm_range_m: float
    tx_power_w: float
    freq_hz: float
    antenna_height_m: float
    speed_range_ms: Tuple[float, float]
    antenna_gain_dbi: float = 5.0
    is_static: bool = False
    is_mec: bool = True


SATELLITE_CONFIG = NodeConfig(
    count=3,
    bandwidth_hz=250e6,
    compute_flops=10e9,
    storage_bytes=64e9,
    comm_range_m=1_000_000.0,
    tx_power_w=10.0,
    freq_hz=FREQ_KA,
    antenna_height_m=780_000.0,       # LEO orbit altitude
    speed_range_ms=(6680.0, 6680.0),  # ground-track ≈ 6.68 km/s
    antenna_gain_dbi=30.0,            # high-gain parabolic antenna
    is_static=False,
    is_mec=True,
)

UAV_CONFIG = NodeConfig(
    count=6,
    bandwidth_hz=20e6,
    compute_flops=5e9,
    storage_bytes=32e9,
    comm_range_m=15_000.0,
    tx_power_w=1.0,
    freq_hz=FREQ_SUB6,
    antenna_height_m=200.0,
    speed_range_ms=(16.7, 33.3),      # 60-120 km/h
    antenna_gain_dbi=8.0,             # directional antenna
    is_static=False,
    is_mec=True,
)

SHIP_CONFIG = NodeConfig(
    count=10,
    bandwidth_hz=50e6,
    compute_flops=20e9,
    storage_bytes=128e9,
    comm_range_m=50_000.0,
    tx_power_w=5.0,
    freq_hz=FREQ_SUB6,
    antenna_height_m=15.0,
    speed_range_ms=(7.7, 12.9),       # 15-25 knots
    antenna_gain_dbi=12.0,            # maritime directional antenna
    is_static=False,
    is_mec=True,
)

BUOY_CONFIG = NodeConfig(
    count=20,
    bandwidth_hz=5e6,
    compute_flops=0.5e9,
    storage_bytes=4e9,
    comm_range_m=15_000.0,
    tx_power_w=0.1,
    freq_hz=FREQ_NB_IOT,
    antenna_height_m=2.0,
    speed_range_ms=(0.3, 0.8),        # 1-3 km/h ocean drift
    antenna_gain_dbi=3.0,             # omnidirectional
    is_static=False,
    is_mec=False,
)

BASE_STATION_CONFIG = NodeConfig(
    count=3,
    bandwidth_hz=100e6,
    compute_flops=50e9,
    storage_bytes=500e9,
    comm_range_m=80_000.0,
    tx_power_w=20.0,
    freq_hz=FREQ_CBAND,
    antenna_height_m=50.0,
    speed_range_ms=(0.0, 0.0),
    antenna_gain_dbi=18.0,            # sectoral antenna
    is_static=True,
    is_mec=True,
)

NODE_CONFIGS = {
    "satellite": SATELLITE_CONFIG,
    "uav": UAV_CONFIG,
    "ship": SHIP_CONFIG,
    "buoy": BUOY_CONFIG,
    "base_station": BASE_STATION_CONFIG,
}

# ---------------------------------------------------------------------------
# MEC / Computation offloading defaults
# ---------------------------------------------------------------------------
TASK_DATA_SIZE = 500e3           # 500 KB per task
TASK_COMPUTE_CYCLES = 1e9       # 1 Giga-cycles per task
TASK_RESULT_SIZE = 50e3          # 50 KB result
CMOS_KAPPA = 1e-28               # CMOS dynamic power coefficient
TASK_ARRIVAL_RATE = 0.5          # tasks/s per buoy

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
MAP_DISPLAY_WIDTH = 1050
MAP_DISPLAY_HEIGHT = 900
SIDEBAR_WIDTH = 350
FPS = 30

# ---------------------------------------------------------------------------
# Colour palette (R, G, B)
# ---------------------------------------------------------------------------
COLOR_OCEAN = (15, 50, 100)
COLOR_COASTLINE = (34, 139, 34)
COLOR_SATELLITE = (255, 60, 60)
COLOR_UAV = (255, 165, 0)
COLOR_SHIP = (220, 220, 220)
COLOR_BUOY = (0, 220, 220)
COLOR_BASE_STATION = (0, 200, 80)
COLOR_LINK = (100, 100, 255)
COLOR_SIDEBAR_BG = (20, 20, 30)
COLOR_TEXT = (200, 200, 200)

NODE_COLORS = {
    "satellite": COLOR_SATELLITE,
    "uav": COLOR_UAV,
    "ship": COLOR_SHIP,
    "buoy": COLOR_BUOY,
    "base_station": COLOR_BASE_STATION,
}
