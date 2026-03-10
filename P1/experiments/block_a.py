"""
Experiment Block A — Mechanism comparison under noise variation.

Fix N_total = 120, sweep eta_N over {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}.
Compare INDP / Disco / ALOHA.  Primary metric: F1_topo.
Parallelized across (eta_N, seed) with ProcessPoolExecutor.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P1.protocols.indp import INDPProtocol
from P1.protocols.disco import DiscoProtocol
from P1.protocols.aloha import ALOHAProtocol


ETA_N_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
N_SEEDS = 8
N_WINDOWS = 10


def _run_single_config_a(args):
    """Worker: one (eta_N, seed) combo with all 3 protocols."""
    eta_n, seed, n_windows = args
    cfg = EnvConfig(N_total=120, eta_N=eta_n, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="discovery", max_steps=n_windows * cfg.N_slot)
    rng = np.random.default_rng(seed)

    protocols = {
        "INDP": INDPProtocol(cfg),
        "Disco": DiscoProtocol(cfg),
        "ALOHA": ALOHAProtocol(cfg),
    }

    records = []
    for name, proto in protocols.items():
        obs, info = env.reset(seed=seed)
        nodes = env.nodes
        n = len(nodes)
        f1_values = []

        for w in range(n_windows):
            env.recompute_ground_truth()
            result = proto.run_window(nodes, cfg, rng)
            env.set_discovered_topology(result["disc_adj"])
            gt = env.get_ground_truth_topology()
            f1, tp, fp, fn = proto.compute_f1(gt, n)
            f1_values.append(f1)
            actions = np.ones((n, 2), dtype=np.float32)
            obs, _, term, trunc, _ = env.step(actions)
            if term or trunc:
                obs, _ = env.reset(seed=seed)

        records.append({
            "experiment": "A",
            "eta_N": eta_n,
            "seed": seed,
            "mechanism": name,
            "mean_f1_topo": float(np.mean(f1_values)),
        })
    env.close()
    return records


def run_block_a(log_dir: str = "P1/logs", n_seeds: int = N_SEEDS,
                n_windows: int = N_WINDOWS, n_workers: int = None) -> pd.DataFrame:
    """Execute Block A and return aggregated results."""
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    args_list = [(eta_n, seed, n_windows)
                 for eta_n in ETA_N_VALUES
                 for seed in range(n_seeds)]

    all_records = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = pool.map(_run_single_config_a, args_list)
        for result in tqdm(futures, total=len(args_list),
                           desc="Block A", unit="cfg", dynamic_ncols=True):
            all_records.extend(result)

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(log_dir, "block_a_raw.csv"), index=False)

    summary = df.groupby(["eta_N", "mechanism"])["mean_f1_topo"].agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(log_dir, "block_a_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_a()
    print(summary.to_string(index=False))
    print(f"\nBlock A completed in {time.time() - t0:.1f}s")
