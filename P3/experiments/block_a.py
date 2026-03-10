"""
Experiment Block A — Improved MATD3 learning-rate sweep (parallelised).

Fix N_total=120, M_tot=60 Mbit, eta_B=eta_F=eta_S=1.0.
Train Improved MATD3 at lr in {1e-4, 3e-4, 1e-3}.
Log per-episode mean reward, mean T_total, mean E_total, mean Gamma.

Output: block_a_raw.csv, block_a_summary.csv
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P3.algorithms.improved_matd3 import ImprovedMATD3

LR_VALUES = [1e-4, 3e-4, 1e-3]
N_SEEDS = 5
N_EPISODES = 80
N_WINDOWS = 5


def _worker_block_a(
    args: Tuple[float, int, int, int],
) -> List[Dict[str, Any]]:
    lr, seed, n_episodes, n_windows = args
    cfg = EnvConfig(N_total=20, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="resource_mgmt",
                       max_steps=n_windows * 20 + 50)
    rng = np.random.default_rng(seed)

    agent = ImprovedMATD3(min(cfg.N_src, cfg.node_counts["buoy"]), cfg, lr=lr)
    records: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        info = agent.train_episode(env, n_windows=n_windows, rng=rng)
        records.append({
            "experiment": "A",
            "lr": lr,
            "seed": seed,
            "episode": ep,
            "mean_reward": info["mean_reward"],
            "mean_T_total": info["mean_T_total"],
            "mean_E_total": info["mean_E_total"],
            "mean_Gamma": info["mean_Gamma"],
            "policy_loss": info.get("policy_loss", 0),
            "value_loss": info.get("value_loss", 0),
        })
    env.close()
    return records


def run_block_a(
    log_dir: str = "P3/logs",
    n_seeds: int = N_SEEDS,
    n_episodes: int = N_EPISODES,
    n_windows: int = N_WINDOWS,
    n_workers: int | None = None,
) -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    work_units = [
        (lr, seed, n_episodes, n_windows)
        for lr in LR_VALUES
        for seed in range(n_seeds)
    ]

    records: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch in tqdm(
            pool.map(_worker_block_a, work_units),
            total=len(work_units),
            desc="Block A (LR sweep)",
            unit="cfg",
            leave=True,
            dynamic_ncols=True,
        ):
            records.extend(batch)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(log_dir, "block_a_raw.csv"), index=False)

    summary = df.groupby(["lr", "episode"])["mean_reward"].agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(log_dir, "block_a_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    s = run_block_a()
    print(f"\nBlock A completed in {time.time() - t0:.1f}s")
