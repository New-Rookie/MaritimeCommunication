# SAGIN Ocean IoT MEC Simulation Platform

Space-Air-Ground-Sea integrated heterogeneous network simulation for
Maritime IoT with Mobile Edge Computing (MEC).

## Project Structure

```
FullEssay/
├── env/                     # Core simulation environment
│   ├── config.py            # Global parameters (physics, nodes, visualisation)
│   ├── ocean_env.py         # Gymnasium-compatible main environment
│   ├── nodes/               # Five node types: satellite, UAV, ship, buoy, base_station
│   ├── channel/             # Path-loss models (FSPL, two-ray, A2G) + SINR calculation
│   ├── mobility/            # Mobility models per node type
│   ├── network/             # Topology management + MEC queuing model
│   └── visualization/       # Pygame 2D renderer
├── P1/                      # Research 1: Neighbour Discovery (INDP + IPPO)
│   ├── mechanisms/          # INDP, Disco, ALOHA protocols
│   ├── algorithms/          # PPO, IPPO, ACO, Greedy, GA
│   ├── nd_env.py            # Neighbour discovery env wrapper
│   ├── experiments/         # Experiment scripts (7 sub-experiments)
│   └── run_all.py
├── P2/                      # Research 2: Link Selection (MAPPO + GCN)
│   ├── models/              # MAPPO+GCN, MAPPO, MADDPG, Greedy, ACO
│   ├── ls_env.py            # Link selection env wrapper
│   ├── experiments/         # Experiment scripts (6 sub-experiments)
│   └── run_all.py
├── P3/                      # Research 3: Resource Management (IMATD3)
│   ├── models/              # IMATD3, MATD3, MADDPG, Greedy, GA
│   ├── rm_env.py            # Resource management env wrapper
│   ├── experiments/         # Experiment scripts (15 sub-experiments)
│   └── run_all.py
├── run_visualization.py     # Launch Pygame visualisation (local only)
├── run_all_experiments.py   # Run all experiments (headless, for cloud)
├── smoke_test.py            # Quick import & sanity check
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
# For PyTorch with CUDA (V100):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For PyG:
pip install torch-geometric
```

## Quick Start

```bash
# Verify installation
python smoke_test.py

# Visualisation (requires display)
python run_visualization.py

# Run all experiments (headless, suitable for cloud server)
python run_all_experiments.py

# Run individual research experiments
python P1/run_all.py   # Neighbour Discovery
python P2/run_all.py   # Link Selection
python P3/run_all.py   # Resource Management
```

## Hardware Requirements

- GPU: NVIDIA V100 32GB (recommended) or any CUDA-capable GPU
- RAM: 16GB+
- The full experiment suite takes several hours on V100
